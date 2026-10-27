"""Model evaluation metrics and comparison utilities.

All metrics are computed on *held-out* data only.  Cross-validation scores
are kept separate from the final test evaluation so there is no leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger("models.evaluation")


@dataclass
class MetricResult:
    """Single model's evaluation on one split."""

    model_name: str
    split: str  # "train" | "val" | "test"
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), int))
    n_samples: int = 0
    positive_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "split": self.split,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
        }


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC using the trapezoidal rule."""
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return 0.5
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = np.cumsum(y_true) / n_pos
    fpr = np.cumsum(1 - y_true) / n_neg
    # trapezoidal integration
    return float(np.trapz(tpr, fpr))


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    model_name: str = "model",
    split: str = "test",
) -> MetricResult:
    """Compute all standard classification metrics."""
    cm = _confusion(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    roc_auc = 0.0
    if y_proba is not None:
        roc_auc = _roc_auc(y_true, y_proba[:, 1])

    return MetricResult(
        model_name=model_name,
        split=split,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        n_samples=len(y_true),
        positive_rate=float(np.mean(y_true)),
    )


def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
    selection_metric: str = "f1",
) -> tuple[float, dict[str, float]]:
    """Stratified K-fold CV returning mean score and per-fold scores."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        res = evaluate_model(y_val, y_pred, y_proba, split="cv")
        if selection_metric == "f1":
            scores.append(res.f1)
        elif selection_metric == "roc_auc":
            scores.append(res.roc_auc)
        elif selection_metric == "accuracy":
            scores.append(res.accuracy)
        else:
            scores.append(res.f1)
    return float(np.mean(scores)), {"folds": scores, "mean": float(np.mean(scores))}


def compare_models(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "test",
) -> pd.DataFrame:
    """Evaluate multiple models and return a comparison DataFrame."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        res = evaluate_model(y, y_pred, y_proba, name, split_name)
        rows.append(res.to_dict())
    df = pd.DataFrame(rows)
    # Sort by f1 descending
    if not df.empty:
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    return df


def calibration_data(y_true: np.ndarray, y_proba: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration curve data (bin means)."""
    prob_pos = y_proba[:, 1]
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(prob_pos, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = binids == b
        if not mask.any():
            continue
        mean_pred = prob_pos[mask].mean()
        frac_pos = y_true[mask].mean()
        count = int(mask.sum())
        rows.append({
            "bin": b,
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "mean_predicted": float(mean_pred),
            "fraction_positive": float(frac_pos),
            "count": count,
        })
    return pd.DataFrame(rows)