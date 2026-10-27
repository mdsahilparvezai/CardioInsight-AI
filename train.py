#!/usr/bin/env python
"""CardioInsight AI - training entrypoint.

Usage:
    python train.py                    # full pipeline with defaults
    python train.py --data-path X.csv  # use a specific CSV
    python train.py --engine numpy     # force NumPy engine
    python train.py --model logistic_regression  # train one model only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset, validate_dataset
from src.features import build_preprocessor, fit_preprocessor, transform_features
from src.models import (
    select_engine, available_engines, PARAM_GRIDS,
    evaluate_model, cross_validate, compare_models,
    save_artifacts, ModelMetadata,
)
from src.utils import get_config, reset_config, get_logger

log = get_logger("train")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CardioInsight AI models")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to dataset CSV (overrides config)")
    parser.add_argument("--engine", type=str, default="auto",
                        choices=["auto", "sklearn", "numpy"],
                        help="Model engine to use")
    parser.add_argument("--model", type=str, default=None,
                        choices=["logistic_regression", "random_forest",
                                 "gradient_boosting", "naive_bayes", "all"],
                        help="Train only this model type")
    parser.add_argument("--cv-folds", type=int, default=None,
                        help="CV folds (overrides config)")
    parser.add_argument("--test-size", type=float, default=None,
                        help="Test split fraction (overrides config)")
    parser.add_argument("--artifacts-dir", type=str, default=None,
                        help="Where to save model artifacts")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer CV folds, smaller param grids")
    return parser.parse_args()


def _merge_args_with_config(args: argparse.Namespace) -> dict:
    """Overlay CLI args on top of YAML config."""
    reset_config()
    cfg = get_config()
    if args.data_path is not None:
        cfg["data"]["path"] = args.data_path
    if args.engine is not None:
        cfg["model"]["engine"] = args.engine
    if args.cv_folds is not None:
        cfg["model"]["cv_folds"] = args.cv_folds
    if args.test_size is not None:
        cfg["data"]["test_size"] = args.test_size
    if args.seed is not None:
        cfg["project"]["random_seed"] = args.seed
    if args.artifacts_dir is not None:
        cfg["project"]["artifacts_dir"] = args.artifacts_dir
    if args.quick:
        cfg["model"]["cv_folds"] = min(cfg["model"]["cv_folds"], 3)
        # Shrink param grids for quick mode
        from src.models.engines import PARAM_GRIDS
        for mtype in PARAM_GRIDS:
            for eng in PARAM_GRIDS[mtype]:
                for key in PARAM_GRIDS[mtype][eng]:
                    PARAM_GRIDS[mtype][eng][key] = [PARAM_GRIDS[mtype][eng][key][0]]
    return cfg


def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    stratify: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split without sklearn."""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = np.arange(n)

    # Stratified shuffle
    for cls in [0, 1]:
        cls_idx = indices[y == cls]
        rng.shuffle(cls_idx)
        n_test = int(len(cls_idx) * test_size)
        n_val = int(len(cls_idx) * val_size)
        if cls == 0:
            test_idx = cls_idx[:n_test]
            val_idx = cls_idx[n_test:n_test + n_val]
            train_idx = cls_idx[n_test + n_val:]
        else:
            test_idx = np.concatenate([test_idx, cls_idx[:n_test]])
            val_idx = np.concatenate([val_idx, cls_idx[n_test:n_test + n_val]])
            train_idx = np.concatenate([train_idx, cls_idx[n_test + n_val:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def main() -> int:
    args = _parse_args()
    cfg = _merge_args_with_config(args)

    seed = cfg["project"]["random_seed"]
    engine_name = cfg["model"]["engine"]
    cv_folds = cfg["model"]["cv_folds"]
    selection_metric = cfg["model"]["selection_metric"]
    artifacts_dir = Path(cfg["project"]["artifacts_dir"])
    available = available_engines()

    log.info("=== CardioInsight AI Training ===")
    log.info("Available engines: %s", available)
    log.info("Selected engine: %s", engine_name)
    log.info("Random seed: %d", seed)

    # 1. Load data
    df = load_dataset(path=cfg["data"]["path"], random_seed=seed)
    val_res = validate_dataset(df)
    if not val_res.is_valid:
        log.error("Dataset validation failed: %s", val_res.errors)
        return 1

    # 2. Prepare features/target
    target_col = cfg["data"]["target_column"]
    feature_cols = list(cfg["preprocessing"]["numeric_features"]
                        + cfg["preprocessing"]["categorical_features"])
    # Ensure feature order matches schema
    from src.data.schema import FEATURE_COLUMNS
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    X_df = df[feature_cols].copy()
    y = df[target_col].values.astype(int)

    # 3. Split
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]
    stratify = cfg["data"]["stratify"]
    X_train, y_train, X_val, y_val, X_test, y_test = _train_test_split(
        X_df.values, y, test_size, val_size, stratify, seed
    )
    log.info("Split sizes - train: %d, val: %d, test: %d",
             len(X_train), len(X_val), len(X_test))

    # 4. Preprocessing (fit on train only)
    preprocessor = build_preprocessor()
    if preprocessor is not None:
        preprocessor = fit_preprocessor(preprocessor, pd.DataFrame(X_train, columns=feature_cols))
        X_train_t = transform_features(preprocessor, pd.DataFrame(X_train, columns=feature_cols))
        X_val_t = transform_features(preprocessor, pd.DataFrame(X_val, columns=feature_cols))
        X_test_t = transform_features(preprocessor, pd.DataFrame(X_test, columns=feature_cols))
    else:
        log.warning("No preprocessor available; using raw features.")
        X_train_t, X_val_t, X_test_t = X_train, X_val, X_test

    # 5. Model selection
    model_types = [args.model] if args.model and args.model != "all" else [
        "logistic_regression", "random_forest", "gradient_boosting", "naive_bayes"
    ]

    trained_models = {}
    val_results = {}

    for mtype in model_types:
        log.info("=== Training %s ===", mtype)

        # Get param grid for this engine
        grid = PARAM_GRIDS.get(mtype, {}).get(engine_name, {})
        if not grid:
            grid = {k: [v] for k, v in PARAM_GRIDS.get(mtype, {}).get("numpy", {}).items()}

        # Grid search on validation set
        best_score = -1.0
        best_model = None
        best_params = {}

        # Expand grid
        import itertools
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            try:
                model = select_engine(mtype, engine=engine_name,
                                      random_state=seed, **params)
                model.fit(X_train_t, y_train)
                y_val_pred = model.predict(X_val_t)
                y_val_proba = model.predict_proba(X_val_t)
                res = evaluate_model(y_val, y_val_pred, y_val_proba,
                                     mtype, "val")
                score = getattr(res, selection_metric)
                log.info("  params=%s -> %s=%.4f", params, selection_metric, score)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
            except Exception as exc:
                log.warning("  params=%s -> ERROR: %s", params, exc)

        if best_model is None:
            log.error("No valid model found for %s", mtype)
            continue

        log.info("Best %s: %s (val %s=%.4f)", mtype, best_params,
                 selection_metric, best_score)

        # Evaluate on train, val, test
        train_res = evaluate_model(
            y_train, best_model.predict(X_train_t),
            best_model.predict_proba(X_train_t), mtype, "train")
        val_res_eval = evaluate_model(
            y_val, best_model.predict(X_val_t),
            best_model.predict_proba(X_val_t), mtype, "val")
        test_res = evaluate_model(
            y_test, best_model.predict(X_test_t),
            best_model.predict_proba(X_test_t), mtype, "test")

        trained_models[mtype] = best_model
        val_results[mtype] = {
            "params": best_params,
            "val_score": best_score,
            "train_metrics": train_res.to_dict(),
            "val_metrics": val_res_eval.to_dict(),
            "test_metrics": test_res.to_dict(),
        }

        log.info("  Test F1=%.4f, AUC=%.4f, Acc=%.4f",
                 test_res.f1, test_res.roc_auc, test_res.accuracy)

    if not trained_models:
        log.error("No models trained successfully.")
        return 1

    # 6. Save the best model (by validation F1)
    best_name = max(val_results, key=lambda k: val_results[k]["val_score"])
    best_model = trained_models[best_name]
    best_info = val_results[best_name]

    metadata = ModelMetadata(
        model_type=best_name,
        engine=engine_name,
        config=cfg,
        train_metrics=best_info["train_metrics"],
        val_metrics=best_info["val_metrics"],
        test_metrics=best_info["test_metrics"],
        feature_names=feature_cols,
        trained_at=datetime.now().isoformat(),
        random_seed=seed,
        n_train_samples=int(len(X_train)),
        n_features=X_train_t.shape[1],
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / f"model_{best_name}_{engine_name}_{seed}.pkl"
    save_artifacts(best_model, preprocessor, metadata, model_path)

    # Also save comparison table
    comp = compare_models(trained_models, X_test_t, y_test, "test")
    comp_path = artifacts_dir / "comparison.csv"
    comp.to_csv(comp_path, index=False)
    log.info("Comparison table saved to %s", comp_path)

    log.info("=== Training complete ===")
    log.info("Best model: %s (engine: %s)", best_name, engine_name)
    log.info("Artifact: %s", model_path)

    # Print final comparison
    print("\n" + "=" * 80)
    print(f"FINAL COMPARISON ON TEST SET (engine={engine_name})")
    print("=" * 80)
    print(comp.to_string(index=False))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())