"""Model engines with a pure-NumPy fallback.

Two engines implement the same interface:

* :class:`SklearnEngine` — delegates to scikit-learn when available.
* :class:`NumPyEngine` — pure NumPy implementations of the same algorithms
  (gradient-descent logistic regression, bootstrap-aggregated decision trees,
  gradient-boosted trees, Gaussian Naive Bayes).  No external dependencies.

The training script calls :func:`select_engine` which returns whichever is
available (or the one forced by configuration).  This guarantees the pipeline
runs and produces honest metrics in *any* environment.
"""

from __future__ import annotations

import abc
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..data.schema import FEATURE_COLUMNS
from ..utils.logging import get_logger

log = get_logger("models.engines")


# ---------------------------------------------------------------------------
# Public protocol / type hints
# ---------------------------------------------------------------------------
@runtime_checkable
class ModelEngine(Protocol):
    """Minimal interface both engines implement."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelEngine":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def get_params(self) -> dict[str, Any]:
        ...

    def set_params(self, **params) -> "ModelEngine":
        ...


# ---------------------------------------------------------------------------
# NumPy building blocks (tree structure, log-loss, softmax, etc.)
# ---------------------------------------------------------------------------
class _DecisionNode:
    __slots__ = ("feature", "threshold", "left", "right", "value", "leaf")

    def __init__(self, feature: int | None = None, threshold: float | None = None,
                 left: "_DecisionNode | None" = None,
                 right: "_DecisionNode | None" = None,
                 value: float | None = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.leaf = value is not None


def _gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def _best_split(X: np.ndarray, y: np.ndarray, n_features: int,
                rng: np.random.Generator) -> tuple[int | None, float | None]:
    """Find the best (feature, threshold) split by Gini impurity."""
    n_samples, n_feats = X.shape
    if n_samples < 2:
        return None, None
    best_gini = 1.0
    best_feat = None
    best_thresh = None
    feat_idx = rng.choice(n_feats, size=min(n_features, n_feats), replace=False)
    for f in feat_idx:
        values = np.unique(X[:, f])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for t in thresholds:
            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            g = (len(left) * _gini(left) + len(right) * _gini(right)) / n_samples
            if g < best_gini:
                best_gini = g
                best_feat = f
                best_thresh = t
    return best_feat, best_thresh


def _build_tree(X: np.ndarray, y: np.ndarray, max_depth: int,
                min_samples_split: int, n_features: int,
                rng: np.random.Generator, depth: int = 0) -> _DecisionNode:
    n_samples = len(y)
    if (depth >= max_depth or n_samples < min_samples_split
            or len(np.unique(y)) == 1):
        # Leaf: return class probability for the positive class
        return _DecisionNode(value=float(np.mean(y)))
    feat, thresh = _best_split(X, y, n_features, rng)
    if feat is None:
        return _DecisionNode(value=float(np.mean(y)))
    left_idx = X[:, feat] <= thresh
    right_idx = ~left_idx
    left = _build_tree(X[left_idx], y[left_idx], max_depth, min_samples_split,
                       n_features, rng, depth + 1)
    right = _build_tree(X[right_idx], y[right_idx], max_depth,
                        min_samples_split, n_features, rng, depth + 1)
    return _DecisionNode(feature=feat, threshold=thresh, left=left, right=right)


def _predict_tree(node: _DecisionNode, x: np.ndarray) -> float:
    while not node.leaf:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# NumPyEngine implementations
# ---------------------------------------------------------------------------
class _LogisticRegressionNumPy:
    """Binary logistic regression via gradient descent with L2 regularisation.

    Includes internal feature standardisation to avoid numerical overflow
    on raw features with different scales.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 lr: float = 0.1, tol: float = 1e-4, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._mean_: np.ndarray | None = None
        self._std_: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self._mean_ is None:
            self._mean_ = X.mean(axis=0)
            self._std_ = X.std(axis=0)
            self._std_ = np.where(self._std_ == 0, 1.0, self._std_)
        return (X - self._mean_) / self._std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticRegressionNumPy":
        X_scaled = self._standardize(X)
        n_samples, n_features = X_scaled.shape
        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(0, 0.01, size=n_features)
        self.intercept_ = 0.0
        prev_loss = np.inf
        for _ in range(self.max_iter):
            logits = X_scaled @ self.coef_ + self.intercept_
            probs = _sigmoid(logits)
            # Binary cross-entropy + L2
            eps = 1e-15
            loss = (-(y * np.log(probs + eps)
                     + (1 - y) * np.log(1 - probs + eps)).mean()
                    + 0.5 * (1.0 / self.C) * np.sum(self.coef_ ** 2))
            grad_w = ((probs - y) @ X_scaled) / n_samples + (1.0 / self.C) * self.coef_
            grad_b = (probs - y).mean()
            # Gradient clipping for stability
            grad_norm = np.sqrt(np.sum(grad_w ** 2) + grad_b ** 2)
            max_norm = 1.0
            if grad_norm > max_norm:
                scale = max_norm / grad_norm
                grad_w *= scale
                grad_b *= scale
            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._standardize(X)
        logits = X_scaled @ self.coef_ + self.intercept_
        p1 = _sigmoid(logits)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self) -> dict:
        return {"C": self.C, "max_iter": self.max_iter, "lr": self.lr,
                "tol": self.tol, "random_state": self.random_state}

    def set_params(self, **params) -> "_LogisticRegressionNumPy":
        for k, v in params.items():
            setattr(self, k, v)
        return self


class _RandomForestNumPy:
    """Bootstrap-aggregated decision trees."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 min_samples_split: int = 2, max_features: str | int = "sqrt",
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[_DecisionNode] = []

    def _n_features(self, n_total: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_total)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_total)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_total)
        return n_total

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomForestNumPy":
        n_samples, n_feats = X.shape
        n_feat = self._n_features(n_feats)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for i in range(self.n_estimators):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = _build_tree(X[idx], y[idx], self.max_depth,
                               self.min_samples_split, n_feat,
                               np.random.default_rng(rng.integers(2**32)))
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.array([[1 - _predict_tree(t, x), _predict_tree(t, x)]
                          for t in self.trees_ for x in X])
        probs = probs.reshape(self.n_estimators, len(X), 2).mean(axis=0)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self) -> dict:
        return {"n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "max_features": self.max_features,
                "random_state": self.random_state}

    def set_params(self, **params) -> "_RandomForestNumPy":
        for k, v in params.items():
            setattr(self, k, v)
        return self


class _GradientBoostingNumPy:
    """Gradient-boosted decision trees for binary classification."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 subsample: float = 1.0, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.trees_: list[_DecisionNode] = []
        self.init_score_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_GradientBoostingNumPy":
        n_samples = len(X)
        rng = np.random.default_rng(self.random_state)
        # Initial prediction = log-odds of the positive class
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.init_score_ = float(np.log(p / (1 - p)))
        preds = np.full(n_samples, self.init_score_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            probs = _sigmoid(preds)
            # Negative gradient of log-loss
            residuals = y - probs
            # Subsample
            if self.subsample < 1.0:
                idx = rng.choice(n_samples,
                                 size=int(n_samples * self.subsample),
                                 replace=False)
                X_sub = X[idx]
                res_sub = residuals[idx]
            else:
                X_sub = X
                res_sub = residuals
            # Fit tree to residuals (regression tree on pseudo-residuals)
            tree = _build_tree_reg(X_sub, res_sub, self.max_depth,
                                   self.min_samples_split, X.shape[1], rng)
            self.trees_.append(tree)
            # Update predictions
            update = np.array([_predict_tree_reg(tree, x) for x in X])
            preds += self.learning_rate * update
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.full(len(X), self.init_score_)
        for tree in self.trees_:
            preds += self.learning_rate * np.array([_predict_tree_reg(tree, x) for x in X])
        p1 = _sigmoid(preds)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self) -> dict:
        return {"n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "subsample": self.subsample,
                "random_state": self.random_state}

    def set_params(self, **params) -> "_GradientBoostingNumPy":
        for k, v in params.items():
            setattr(self, k, v)
        return self


def _build_tree_reg(X: np.ndarray, y: np.ndarray, max_depth: int,
                    min_samples_split: int, n_features: int,
                    rng: np.random.Generator, depth: int = 0) -> _DecisionNode:
    """Regression tree (MSE) for gradient boosting."""
    n_samples = len(y)
    if (depth >= max_depth or n_samples < min_samples_split
            or np.var(y) < 1e-12):
        return _DecisionNode(value=float(np.mean(y)))
    best_mse = np.inf
    best_feat = None
    best_thresh = None
    feat_idx = rng.choice(X.shape[1], size=min(n_features, X.shape[1]), replace=False)
    for f in feat_idx:
        values = np.unique(X[:, f])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for t in thresholds:
            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            mse = (len(left) * np.var(left) + len(right) * np.var(right)) / n_samples
            if mse < best_mse:
                best_mse = mse
                best_feat = f
                best_thresh = t
    if best_feat is None:
        return _DecisionNode(value=float(np.mean(y)))
    left_idx = X[:, best_feat] <= best_thresh
    right_idx = ~left_idx
    left = _build_tree_reg(X[left_idx], y[left_idx], max_depth,
                           min_samples_split, n_features, rng, depth + 1)
    right = _build_tree_reg(X[right_idx], y[right_idx], max_depth,
                            min_samples_split, n_features, rng, depth + 1)
    return _DecisionNode(feature=best_feat, threshold=best_thresh,
                         left=left, right=right)


def _predict_tree_reg(node: _DecisionNode, x: np.ndarray) -> float:
    while not node.leaf:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value


class _NaiveBayesNumPy:
    """Gaussian Naive Bayes for continuous features only."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.class_prior_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NaiveBayesNumPy":
        classes = np.array([0, 1])
        n_classes = len(classes)
        n_features = X.shape[1]
        self.class_prior_ = np.zeros(n_classes)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(classes):
            Xc = X[y == c]
            self.class_prior_[i] = len(Xc) / len(X)
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + self.var_smoothing
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, 2))
        for i in range(2):
            # log prior
            log_prob[:, i] = np.log(self.class_prior_[i])
            # log likelihood (Gaussian)
            diff = X - self.theta_[i]
            log_prob[:, i] += -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[i]) + diff ** 2 / self.var_[i],
                axis=1,
            )
        # Normalise
        max_lp = log_prob.max(axis=1, keepdims=True)
        prob = np.exp(log_prob - max_lp)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def get_params(self) -> dict:
        return {"var_smoothing": self.var_smoothing}

    def set_params(self, **params) -> "_NaiveBayesNumPy":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------
class SklearnEngine:
    """Wrapper around scikit-learn estimators (when available)."""

    def __init__(self, model_type: str, **params):
        self.model_type = model_type
        self.params = params
        self.model_ = None
        self._build()

    def _build(self):
        if self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            self.model_ = LogisticRegression(
                C=self.params.get("C", 1.0),
                max_iter=self.params.get("max_iter", 1000),
                random_state=self.params.get("random_state", 42),
                solver="lbfgs",
            )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model_ = RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 5),
                min_samples_split=self.params.get("min_samples_split", 2),
                max_features=self.params.get("max_features", "sqrt"),
                random_state=self.params.get("random_state", 42),
                n_jobs=self.params.get("n_jobs", 1),
            )
        elif self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            self.model_ = GradientBoostingClassifier(
                n_estimators=self.params.get("n_estimators", 100),
                learning_rate=self.params.get("learning_rate", 0.1),
                max_depth=self.params.get("max_depth", 3),
                min_samples_split=self.params.get("min_samples_split", 2),
                subsample=self.params.get("subsample", 1.0),
                random_state=self.params.get("random_state", 42),
            )
        elif self.model_type == "naive_bayes":
            from sklearn.naive_bayes import GaussianNB
            self.model_ = GaussianNB(
                var_smoothing=self.params.get("var_smoothing", 1e-9),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnEngine":
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def get_params(self) -> dict:
        return {"model_type": self.model_type, **self.params}

    def set_params(self, **params) -> "SklearnEngine":
        self.params.update(params)
        self._build()
        return self


class NumPyEngine:
    """Pure NumPy fallback engine (zero external deps)."""

    def __init__(self, model_type: str, **params):
        self.model_type = model_type
        self.params = params
        self.model_ = self._build()

    def _build(self) -> ModelEngine:
        if self.model_type == "logistic_regression":
            return _LogisticRegressionNumPy(
                C=self.params.get("C", 1.0),
                max_iter=self.params.get("max_iter", 1000),
                lr=self.params.get("lr", 0.1),
                tol=self.params.get("tol", 1e-4),
                random_state=self.params.get("random_state", 42),
            )
        if self.model_type == "random_forest":
            return _RandomForestNumPy(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 5),
                min_samples_split=self.params.get("min_samples_split", 2),
                max_features=self.params.get("max_features", "sqrt"),
                random_state=self.params.get("random_state", 42),
            )
        if self.model_type == "gradient_boosting":
            return _GradientBoostingNumPy(
                n_estimators=self.params.get("n_estimators", 100),
                learning_rate=self.params.get("learning_rate", 0.1),
                max_depth=self.params.get("max_depth", 3),
                min_samples_split=self.params.get("min_samples_split", 2),
                subsample=self.params.get("subsample", 1.0),
                random_state=self.params.get("random_state", 42),
            )
        if self.model_type == "naive_bayes":
            return _NaiveBayesNumPy(
                var_smoothing=self.params.get("var_smoothing", 1e-9),
            )
        raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumPyEngine":
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def get_params(self) -> dict:
        return {"model_type": self.model_type, **self.params}

    def set_params(self, **params) -> "NumPyEngine":
        self.params.update(params)
        self.model_ = self._build()
        return self


# ---------------------------------------------------------------------------
# Engine selection & parameter grids
# ---------------------------------------------------------------------------
def available_engines() -> list[str]:
    """Return list of engines that can be used in this environment."""
    engines = ["numpy"]
    try:
        import sklearn  # noqa: F401
        engines.append("sklearn")
    except ImportError:
        pass
    return engines


def select_engine(model_type: str, engine: str | None = None,
                  **params) -> ModelEngine:
    """Return an instantiated engine.

    Parameters
    ----------
    model_type:
        One of the supported model types.
    engine:
        "auto" | "sklearn" | "numpy".  Defaults to "auto" (prefers sklearn).
    params:
        Forwarded to the engine constructor.
    """
    available = available_engines()
    if engine is None:
        engine = "auto"
    if engine == "auto":
        engine = "sklearn" if "sklearn" in available else "numpy"
    if engine == "sklearn":
        if "sklearn" not in available:
            raise RuntimeError("scikit-learn requested but not installed.")
        return SklearnEngine(model_type, **params)
    if engine == "numpy":
        return NumPyEngine(model_type, **params)
    raise ValueError(f"Unknown engine: {engine}")


# Parameter grids for cross-validation tuning (used by train.py)
PARAM_GRIDS = {
    "logistic_regression": {
        "sklearn": {"C": [0.01, 0.1, 1.0, 10.0]},
        "numpy": {"C": [0.01, 0.1, 1.0, 10.0],
                  "lr": [0.05, 0.1, 0.2],
                  "max_iter": [500, 1000]},
    },
    "random_forest": {
        "sklearn": {"n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7]},
        "numpy": {"n_estimators": [50, 100, 200],
                  "max_depth": [3, 5, 7]},
    },
    "gradient_boosting": {
        "sklearn": {"n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [2, 3]},
        "numpy": {"n_estimators": [50, 100, 200],
                  "learning_rate": [0.05, 0.1, 0.2],
                  "max_depth": [2, 3]},
    },
    "naive_bayes": {
        "sklearn": {"var_smoothing": [1e-12, 1e-9, 1e-6]},
        "numpy": {"var_smoothing": [1e-12, 1e-9, 1e-6]},
    },
}