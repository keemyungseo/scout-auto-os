"""Regression models for Trade Value Estimator."""

from __future__ import annotations

import math
import statistics

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = (
    "expected_roi",
    "expected_peak_roi",
    "expected_hold_time",
    "expected_drawdown",
    "expected_win_prob",
    "expected_sharpe_contrib",
)


def _matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = sorted(rows[0]["x"].keys())
    X = np.array([[float(r["x"].get(n, 0)) for n in names] for r in rows], dtype=float)
    return X, names


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 0.01
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def build_regressors() -> dict:
    models = {
        "linear_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8),
    }
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(n_estimators=150, max_depth=6, random_state=42, verbose=-1)
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(n_estimators=150, max_depth=6, random_state=42, verbosity=0)
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        models["catboost"] = CatBoostRegressor(iterations=150, depth=6, random_state=42, verbose=0)
    except ImportError:
        pass
    return models


def evaluate_regressors(rows: list[dict], target: str) -> list[dict]:
    X, _ = _matrix(rows)
    y = np.array([float(r["y"][target]) for r in rows], dtype=float)
    n_splits = min(5, max(2, len(rows) // 10))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: list[dict] = []

    for name, model in build_regressors().items():
        maes, rmses, r2s, mapes = [], [], [], []
        preds_all = np.zeros(len(y))
        counts = np.zeros(len(y))

        for train_i, test_i in cv.split(X):
            pipe = Pipeline([("scaler", StandardScaler()), ("reg", model)])
            pipe.fit(X[train_i], y[train_i])
            pred = pipe.predict(X[test_i])
            preds_all[test_i] += pred
            counts[test_i] += 1
            maes.append(mean_absolute_error(y[test_i], pred))
            rmses.append(math.sqrt(mean_squared_error(y[test_i], pred)))
            r2s.append(r2_score(y[test_i], pred))
            mapes.append(_mape(y[test_i], pred))

        oof = np.where(counts > 0, preds_all / counts, 0)
        results.append({
            "model": name,
            "target": target,
            "mae": round(statistics.mean(maes), 4),
            "rmse": round(statistics.mean(rmses), 4),
            "r2": round(statistics.mean(r2s), 4),
            "mape_pct": round(statistics.mean(mapes), 2),
            "oof_predictions": oof,
        })
    return results


def best_oof_predictions(rows: list[dict]) -> dict[str, np.ndarray]:
    """Out-of-fold predictions for each target using best R² model."""
    out: dict[str, np.ndarray] = {}
    for target in TARGETS:
        evals = evaluate_regressors(rows, target)
        best = max(evals, key=lambda e: e["r2"])
        out[target] = best["oof_predictions"]
        out[f"_best_model_{target}"] = np.array([best["model"]] * len(rows))
    return out
