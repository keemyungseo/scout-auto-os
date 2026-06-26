"""Classifier comparison for Trade DNA predictor."""

from __future__ import annotations

import statistics

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _matrix(rows: list[dict], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r["x"].get(n, 0)) for n in names] for r in rows], dtype=float)
    y = np.array([int(r["cluster_id"]) for r in rows], dtype=int)
    return X, y


def _false_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    false_runner = sum(1 for t, p in zip(y_true, y_pred) if p == 0 and t == 1)
    false_failed = sum(1 for t, p in zip(y_true, y_pred) if p == 1 and t == 0)
    pred_runner = sum(1 for p in y_pred if p == 0) or 1
    pred_failed = sum(1 for p in y_pred if p == 1) or 1
    return (
        round(false_runner / pred_runner * 100, 2),
        round(false_failed / pred_failed * 100, 2),
    )


def _eval_model(name: str, pipeline: Pipeline, X: np.ndarray, y: np.ndarray, n_splits: int) -> dict:
    n_splits = max(2, min(n_splits, min(np.bincount(y))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, precs, recs, f1s, aucs = [], [], [], [], []
    p0s, p1s, frs, ffs = [], [], [], []

    for train_i, test_i in cv.split(X, y):
        pipeline.fit(X[train_i], y[train_i])
        pred = pipeline.predict(X[test_i])
        proba = pipeline.predict_proba(X[test_i])[:, 1]

        accs.append(accuracy_score(y[test_i], pred))
        precs.append(precision_score(y[test_i], pred, average="weighted", zero_division=0))
        recs.append(recall_score(y[test_i], pred, average="weighted", zero_division=0))
        f1s.append(f1_score(y[test_i], pred, average="weighted", zero_division=0))
        try:
            aucs.append(roc_auc_score(y[test_i], proba))
        except ValueError:
            aucs.append(0.5)

        p0s.append(precision_score(y[test_i], pred, pos_label=0, zero_division=0))
        p1s.append(precision_score(y[test_i], pred, pos_label=1, zero_division=0))
        fr, ff = _false_rates(y[test_i], pred)
        frs.append(fr)
        ffs.append(ff)

    return {
        "model": name,
        "accuracy": round(statistics.mean(accs), 4),
        "precision": round(statistics.mean(precs), 4),
        "recall": round(statistics.mean(recs), 4),
        "f1": round(statistics.mean(f1s), 4),
        "roc_auc": round(statistics.mean(aucs), 4),
        "type0_precision": round(statistics.mean(p0s), 4),
        "type1_precision": round(statistics.mean(p1s), 4),
        "false_runner_rate_pct": round(statistics.mean(frs), 2),
        "false_failed_rate_pct": round(statistics.mean(ffs), 2),
    }


def build_model_map() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "naive_bayes": GaussianNB(),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss", verbosity=0,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
    except ImportError:
        pass
    return models


def cross_cv_proba(pipe: Pipeline, X: np.ndarray, y: np.ndarray, cv) -> np.ndarray:
    proba = np.zeros(len(y), dtype=float)
    for train_i, test_i in cv.split(X, y):
        pipe.fit(X[train_i], y[train_i])
        proba[test_i] = pipe.predict_proba(X[test_i])[:, 1]
    return proba


def compare_classifiers(rows: list[dict]) -> tuple[list[dict], dict, np.ndarray, list[str], Pipeline]:
    names = sorted(rows[0]["x"].keys()) if rows else []
    X, y = _matrix(rows, names)
    n_splits = min(5, min(np.bincount(y)) * 2)
    model_map = build_model_map()

    results: list[dict] = []
    for name, model in model_map.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        results.append(_eval_model(name, pipe, X, y, n_splits))

    best = max(results, key=lambda r: r["f1"])
    best_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model_map[best["model"]]),
    ])
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
    y_pred = cross_val_predict(best_pipe, X, y, cv=cv)

    meta = {
        "best_model": best["model"],
        "feature_names": names,
        "n_samples": len(rows),
        "n_type0": int(sum(y == 0)),
        "n_type1": int(sum(y == 1)),
    }
    return results, meta, y_pred, names, best_pipe
