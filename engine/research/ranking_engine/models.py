"""Ranking models — sklearn + gradient boosting rankers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from scout_auto_os.engine.research.ranking_engine.constants import RANDOM_SEED
from scout_auto_os.engine.research.ranking_engine.dataset import build_group_sizes


@dataclass
class RankingModelBundle:
    name: str
    model: object
    scaler: StandardScaler | None = None
    feature_names: list[str] = field(default_factory=list)
    kind: str = "regressor"


def _sort_by_scan(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["scan_kst"], r["symbol"]))


def train_model(
    name: str,
    train_rows: list[dict],
    feat_names: list[str],
) -> RankingModelBundle:
    train_rows = _sort_by_scan(train_rows)
    X = np.array([[float(r["x"].get(n, 0.0)) for n in feat_names] for r in train_rows], dtype=float)
    y_rel = np.array([int(r["relevance"]) for r in train_rows], dtype=float)
    y_ret = np.array([float(r["return_2h"]) for r in train_rows], dtype=float)
    y_top3 = np.array([int(r["label_top3"]) for r in train_rows], dtype=int)
    groups = build_group_sizes(train_rows)

    if name == "lightgbm_ranker":
        import lightgbm as lgb
        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        model.fit(X, y_rel, group=groups)
        return RankingModelBundle(name, model, feature_names=feat_names, kind="ranker")

    if name == "xgboost_ranker":
        import xgboost as xgb
        model = xgb.XGBRanker(
            objective="rank:ndcg",
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        model.fit(X, y_rel, group=groups)
        return RankingModelBundle(name, model, feature_names=feat_names, kind="ranker")

    if name == "catboost_ranker":
        from catboost import CatBoostRanker, Pool
        group_id = []
        gid = 0
        last = None
        for r in train_rows:
            if r["scan_kst"] != last:
                gid += 1
                last = r["scan_kst"]
            group_id.append(gid)
        pool = Pool(X, y_rel, group_id=group_id)
        model = CatBoostRanker(
            iterations=200,
            learning_rate=0.05,
            random_seed=RANDOM_SEED,
            verbose=False,
        )
        model.fit(pool)
        return RankingModelBundle(name, model, feature_names=feat_names, kind="ranker")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if name == "random_forest":
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X, y_ret)
        return RankingModelBundle(name, model, scaler, feat_names, "regressor")

    if name == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X, y_ret)
        return RankingModelBundle(name, model, scaler, feat_names, "regressor")

    if name == "logistic_top3":
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        model.fit(Xs, y_top3)
        return RankingModelBundle(name, model, scaler, feat_names, "classifier")

    if name == "ridge_return":
        model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        model.fit(Xs, y_ret)
        return RankingModelBundle(name, model, scaler, feat_names, "regressor")

    raise ValueError(f"unknown model {name}")


def predict_scores(bundle: RankingModelBundle, rows: list[dict]) -> np.ndarray:
    feat_names = bundle.feature_names
    X = np.array([[float(r["x"].get(n, 0.0)) for n in feat_names] for r in rows], dtype=float)
    if bundle.scaler is not None:
        X = bundle.scaler.transform(X)
    model = bundle.model
    if bundle.kind == "classifier":
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    return model.predict(X)
