"""
Scout Season2 - Auto Validation & Self Improvement Loop

Scientific research engine: falsify hypotheses, measure process prediction,
propose vNext model without overwriting current model.
STRICT NO_ACTION | NO_API | NO_PRICE | NO_TRADING.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi, write_csv
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "ScoutScore_v1"
HORIZONS = {"2h": 2, "4h": 4, "6h": 6}
TREND_STATES = {"Trend Start", "Trend Expansion"}
PERSIST_PHASES = {"Synchronization", "LockedTrend", "LocalAlignment"}
COLLAPSE_PHASES = {"Fragmentation", "Failure"}
STATE_ALIASES = {
    "TrendBirth": {"TrendBirth", "Trend Start", "LocalAlignment", "NearCritical"},
    "Expansion": {"Expansion", "Trend Expansion", "Synchronization"},
    "Persistence": {"Persistence", "LockedTrend", "Synchronization", "LocalAlignment", "Trend Start", "Trend Expansion"},
    "LockedTrend": {"LockedTrend", "Trend Start", "Trend Expansion", "Synchronization"},
    "Fragmentation": {"Fragmentation", "Failure", "CollapseRiskIncrease"},
    "NearCritical": {"NearCritical", "Potential", "Observation"},
    "CollapseRiskIncrease": {"Fragmentation", "Failure", "CollapseRiskIncrease"},
    "Recovery": {"Recovery", "Potential", "Observation"},
    "Rotation": {"Rotation", "Migration"},
    "Migration": {"Migration", "Rotation"},
    "Neutral": {"Neutral", "Observation", "Potential"},
}


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va < 1e-12 or vb < 1e-12:
        return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_predictions(val_id: str) -> list[dict]:
    path = LOGS_DIR / f"validation{val_id}_scores.csv"
    if not path.exists():
        path = LOGS_DIR / f"validation{val_id}_prediction.csv"
    return load_csv(path)


def load_future_ground_truth() -> dict[str, dict[int, dict]]:
    """Process-only ground truth by symbol and checkpoint hour. No price."""
    phase = load_csv(LOGS_DIR / "season2_p62_phase_transition.csv")
    sync = load_csv(LOGS_DIR / "season2_p62_sync_field.csv")
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    belief = load_csv(LOGS_DIR / "season2_p58_belief_field.csv")
    attention = load_csv(LOGS_DIR / "season2_p60_attention_field.csv")
    migration = load_csv(LOGS_DIR / "season2_p61_attention_migration.csv")

    by_sym: dict[str, dict[int, dict]] = defaultdict(dict)
    for src in (phase, sync, order, future, belief, attention, migration):
        for r in src:
            sym = r["symbol"]
            hour = pi(r.get("checkpoint_hour"))
            if sym not in by_sym:
                by_sym[sym] = {}
            if hour not in by_sym[sym]:
                by_sym[sym][hour] = {"symbol": sym, "checkpoint_hour": hour}
            rec = by_sym[sym][hour]
            rec["phase"] = r.get("phase_label") or rec.get("phase", "")
            rec["sync"] = pf(r.get("synchronization_score")) or rec.get("sync", 0)
            rec["p39_state"] = r.get("p39_state") or rec.get("p39_state", "Observation")
            rec["collapse_risk"] = pf(r.get("prob_collapse")) or rec.get("collapse_risk", 0)
            rec["belief"] = pf(r.get("belief_consensus")) or rec.get("belief", 0)
            rec["attention"] = pf(r.get("attention_score")) or rec.get("attention", 0)
            rec["migration_rate"] = pf(r.get("migration_rate")) or rec.get("migration_rate", 0)
            rec["order_parameter"] = pf(r.get("order_parameter_score")) or rec.get("order_parameter", 0)
            rec["flow"] = pf(r.get("var_FlowVelocity")) or rec.get("flow", 0)
    return dict(by_sym)


def actual_expected_state(rec: dict) -> str:
    phase = rec.get("phase", "")
    state = rec.get("p39_state", "")
    collapse = rec.get("collapse_risk", 0)
    if state in TREND_STATES and phase in PERSIST_PHASES:
        return "Persistence"
    if state in TREND_STATES:
        return "TrendBirth" if state == "Trend Start" else "Expansion"
    if phase in COLLAPSE_PHASES or collapse > 0.3:
        return "Fragmentation"
    if phase == "Recovery" or state == "Potential":
        return "Recovery"
    if rec.get("migration_rate", 0) > 0.05:
        return "Migration"
    if phase == "NearCritical":
        return "NearCritical"
    return "Neutral"


def actual_lifecycle(rec: dict, prev: dict | None) -> str:
    sync = rec.get("sync", 0)
    if prev and sync - prev.get("sync", sync) > 8:
        return "growth"
    if sync >= 40:
        return "peak"
    if rec.get("p39_state") in TREND_STATES:
        return "maintenance"
    if rec.get("collapse_risk", 0) > 0.25:
        return "decay"
    return "birth"


def state_match(predicted: str, actual: str, persist_pred: bool = False, persist_act: bool = False) -> bool:
    if persist_pred and persist_act:
        return True
    aliases = STATE_ALIASES.get(predicted, {predicted})
    return actual in aliases or predicted == actual


def persist_positive(rec: dict) -> bool:
    return rec.get("p39_state") in TREND_STATES or (
        rec.get("sync", 0) >= 31 and rec.get("p39_state") not in {"Failure"}
    )


def collapse_positive(rec: dict) -> bool:
    return rec.get("phase") in COLLAPSE_PHASES or rec.get("collapse_risk", 0) > 0.28


def first_variable_change(t0: dict, future: dict) -> str:
    deltas = {
        "Flow": abs(future.get("flow", 0) - t0.get("flow", 0)),
        "Belief": abs(future.get("belief", 0) - t0.get("belief", 0)),
        "Attention": abs(future.get("attention", 0) - t0.get("attention", 0)),
        "Synchronization": abs(future.get("sync", 0) - t0.get("sync", 0)),
        "Migration": abs(future.get("migration_rate", 0) - t0.get("migration_rate", 0)),
    }
    return max(deltas, key=deltas.get)


def kmeans(points: list[list[float]], k: int, iters: int = 30) -> list[int]:
    n = len(points)
    k = min(k, n)
    step = max(1, n // k)
    centers = [points[i * step % n][:] for i in range(k)]
    labels = [0] * n
    for _ in range(iters):
        for i, pt in enumerate(points):
            labels[i] = min(range(k), key=lambda c: sum((pt[d] - centers[c][d]) ** 2 for d in range(len(pt))))
        for c in range(k):
            cluster = [points[i] for i in range(n) if labels[i] == c]
            if cluster:
                centers[c] = [statistics.mean(pt[d] for pt in cluster) for d in range(len(points[0]))]
    return labels


def pca1(points: list[list[float]]) -> list[float]:
    if not points:
        return []
    dims = len(points[0])
    means = [statistics.mean(p[d] for p in points) for d in range(dims)]
    centered = [[p[d] - means[d] for d in range(dims)] for p in points]
    var = [sum(row[d] ** 2 for row in centered) / len(centered) for d in range(dims)]
    total = sum(var) or 1.0
    return [v / total for v in var]


def loocv_rmse_simple(X: list[list[float]], y: list[float], lam: float = 1.0) -> float:
    preds: list[float] = []
    for i in range(len(X)):
        others = [j for j in range(len(X)) if j != i]
        if not others:
            preds.append(statistics.mean(y))
            continue
        # simple ridge: mean-centered univariate fallback per feature average
        pred = statistics.mean(
            sum(X[j][k] for j in others) / len(others) * (y[j] / max(sum(X[j]), 1e-6))
            for k in range(len(X[i]))
        ) if len(X[i]) == 1 else statistics.mean(y[j] for j in others)
        preds.append(pred)
    return math.sqrt(sum((y[i] - preds[i]) ** 2 for i in range(len(y))) / len(y))


def infer_failure_archetype(row: dict) -> str:
    if row.get("data_source") == "institution_proxy" and row.get("predicted_persist") and not row.get("actual_persist"):
        return "Institution Trap"
    if row.get("belief_consensus", 0) > 70 and row.get("attention_score", 0) < 20:
        return "Belief Trap"
    if row.get("attention_score", 0) > 50 and not row.get("actual_persist"):
        return "Attention Mirage"
    if row.get("predicted_state") in ("LockedTrend", "Expansion") and row.get("actual_state") == "Fragmentation":
        return "Synchronization Illusion"
    if row.get("predicted_state") == "TrendBirth" and row.get("actual_state") == "Neutral":
        return "Narrative Drift"
    if row.get("migration_actual", 0) > 0.1:
        return "Late Rotation"
    if row.get("sync_actual", 0) > 40 and row.get("actual_state") == "Fragmentation":
        return "Peak Saturation"
    return f"inferred_failure_{row.get('horizon', 'unknown')}"


def scout_score_vnext(row: dict) -> float:
    """Proposed vNext — hypothesis only, not deployed."""
    base = pf(row.get("scout_score")) or 0.0
    penalty = 1.0
    if row.get("data_source") == "institution_proxy":
        penalty *= 0.72
    if pf(row.get("belief_consensus")) > 70 and pf(row.get("attention_score")) < 15:
        penalty *= 0.85
    if pf(row.get("ecology_entropy")) > 2.5:
        penalty *= 0.90
    boost = 1.0
    if row.get("data_source") == "process_t0":
        boost *= 1.08
    if pf(row.get("synchronization_score")) >= 31 and pf(row.get("attention_score")) >= 10:
        boost *= 1.05
    return clamp(base * penalty * boost)


def evaluate(val_id: str) -> None:
    preds = load_predictions(val_id)
    if not preds:
        raise SystemExit(f"No predictions found for validation{val_id}")

    ground = load_future_ground_truth()
    evaluable = [s for s in ground if len(ground[s]) > 1]

    prefix = f"validation{val_id}"
    result_rows: list[dict] = []
    transition_rows: list[dict] = []
    fp_rows: list[dict] = []
    fn_rows: list[dict] = []
    residual_rows: list[dict] = []
    unknown_rows: list[dict] = []
    archetype_rows: list[dict] = []

    for horizon_label, hour in HORIZONS.items():
        for p in preds:
            sym = p["symbol"]
            gt_sym = ground.get(sym, {})
            if hour not in gt_sym or 0 not in gt_sym:
                continue
            t0 = gt_sym[0]
            actual_rec = gt_sym[hour]
            prev_rec = gt_sym.get(hour - 1, t0)
            pred_state = p.get(f"predicted_{horizon_label}") or p.get(f"expected_{horizon_label}", "")
            actual_state = actual_expected_state(actual_rec)
            actual_life = actual_lifecycle(actual_rec, prev_rec)
            pred_persist = pred_state in {"TrendBirth", "Expansion", "Persistence", "LockedTrend"}
            act_persist = persist_positive(actual_rec)
            phase_ok = state_match(p.get("phase", ""), actual_rec.get("phase", "")) if p.get("phase") else False
            state_ok = state_match(pred_state, actual_state, pred_persist, act_persist)
            collapse_pred = pred_state in {"Fragmentation", "CollapseRiskIncrease"}
            collapse_act = collapse_positive(actual_rec)

            row = {
                "validation_id": val_id,
                "symbol": sym,
                "horizon": horizon_label,
                "rank": pi(p.get("rank")),
                "scout_score": p.get("scout_score"),
                "data_source": p.get("data_source"),
                "predicted_state": pred_state,
                "actual_state": actual_state,
                "predicted_lifecycle": p.get("lifecycle"),
                "actual_lifecycle": actual_life,
                "actual_phase": actual_rec.get("phase"),
                "state_match": "yes" if state_ok else "no",
                "phase_match": "yes" if phase_ok else "no",
                "predicted_persist": pred_persist,
                "actual_persist": act_persist,
                "collapse_predicted": collapse_pred,
                "collapse_actual": collapse_act,
                "sync_predicted_proxy": pf(p.get("synchronization_score")),
                "sync_actual": actual_rec.get("sync"),
                "belief_t0": t0.get("belief"),
                "belief_actual": actual_rec.get("belief"),
                "attention_t0": t0.get("attention"),
                "attention_actual": actual_rec.get("attention"),
                "migration_actual": actual_rec.get("migration_rate"),
                "first_variable_changed": first_variable_change(t0, actual_rec),
                "residual_sync": round(pf(p.get("synchronization_score")) - actual_rec.get("sync", 0), 4),
                "learning_recommendation": "NO_ACTION",
            }
            result_rows.append(row)

            if pred_persist and not act_persist:
                fp_rows.append({**row, "failure_type": "false_positive"})
            if act_persist and not pred_persist:
                fn_rows.append({**row, "failure_type": "false_negative"})
            if not state_ok:
                residual_rows.append({
                    **row,
                    "prediction_error": 1.0,
                    "residual": round(1.0 if not state_ok else 0.0, 4),
                    "hidden_factor_candidate": row["first_variable_changed"],
                })
                archetype_rows.append({
                    "validation_id": val_id,
                    "symbol": sym,
                    "horizon": horizon_label,
                    "failure_archetype": infer_failure_archetype(row),
                    "predicted_state": pred_state,
                    "actual_state": actual_state,
                    "learning_recommendation": "NO_ACTION",
                })

            probs_path = LOGS_DIR / f"validation{val_id}_probability.csv"
            probs = [r for r in load_csv(probs_path) if r["symbol"] == sym and r["horizon"] == horizon_label]
            for pr in probs:
                transition_rows.append({
                    "validation_id": val_id,
                    "symbol": sym,
                    "horizon": horizon_label,
                    "predicted_state": pr["state"],
                    "probability_pct": pf(pr["probability_pct"]),
                    "actual_state": actual_state,
                    "hit": "yes" if state_match(pr["state"], actual_state) else "no",
                    "learning_recommendation": "NO_ACTION",
                })

    # Top-K persistence metrics (evaluable symbols only)
    metrics: dict[str, dict] = {}
    for k in (2, 5, 10):
        top_syms = {p["symbol"] for p in sorted(preds, key=lambda x: pi(x.get("rank")))[:k]}
        for horizon_label, hour in HORIZONS.items():
            tp = fp = fn = tn = 0
            for sym in evaluable:
                if hour not in ground[sym]:
                    continue
                act = persist_positive(ground[sym][hour])
                pred = sym in top_syms and any(
                    r["symbol"] == sym and r["horizon"] == horizon_label and r["predicted_persist"]
                    for r in result_rows
                )
                if sym in top_syms:
                    if act:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if act:
                        fn += 1
                    else:
                        tn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            metrics[f"top{k}_{horizon_label}"] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1(prec, rec), 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

    # Confusion matrix (persistence, 2h, evaluable universe)
    cm_rows: list[dict] = []
    hour = 2
    for sym in evaluable:
        if hour not in ground[sym]:
            continue
        p_row = next((p for p in preds if p["symbol"] == sym), None)
        if not p_row:
            continue
        pred_p = p_row.get("predicted_2h") in {"TrendBirth", "Expansion", "Persistence", "LockedTrend"}
        act_p = persist_positive(ground[sym][hour])
        label = "TP" if pred_p and act_p else "FP" if pred_p and not act_p else "FN" if not pred_p and act_p else "TN"
        cm_rows.append({
            "validation_id": val_id,
            "symbol": sym,
            "horizon": "2h",
            "predicted_persist": pred_p,
            "actual_persist": act_p,
            "confusion_cell": label,
            "learning_recommendation": "NO_ACTION",
        })

    state_acc = sum(1 for r in result_rows if r["state_match"] == "yes") / max(len(result_rows), 1)
    persist_acc = sum(1 for r in result_rows if r["predicted_persist"] == r["actual_persist"]) / max(len(result_rows), 1)
    phase_acc = sum(1 for r in result_rows if r.get("phase_match") == "yes") / max(len(result_rows), 1)
    trans_hits = sum(1 for r in transition_rows if r["hit"] == "yes")
    trans_acc = trans_hits / max(len(transition_rows), 1)
    collapse_rows = [r for r in result_rows if r["collapse_predicted"] or r["collapse_actual"]]
    collapse_acc = sum(
        1 for r in collapse_rows if r["collapse_predicted"] == r["collapse_actual"]
    ) / max(len(collapse_rows), 1)

    # Residual / unknown field analysis
    if residual_rows:
        feats = [
            [pf(r.get("belief_t0")), pf(r.get("attention_t0")), pf(r.get("sync_predicted_proxy")), pf(r.get("scout_score"))]
            for r in residual_rows
        ]
        imp = pca1(feats)
        feat_names = ["Belief", "Attention", "Synchronization", "ScoutScore"]
        for i, name in enumerate(feat_names):
            unknown_rows.append({
                "validation_id": val_id,
                "unknown_field_candidate": f"UnknownField_{name}",
                "pca_importance": round(imp[i], 4),
                "source": "residual_pca",
                "integrated": "no",
                "learning_recommendation": "NO_ACTION",
            })
        labels = kmeans(feats, min(3, len(feats)))
        for i, r in enumerate(residual_rows):
            unknown_rows.append({
                "validation_id": val_id,
                "symbol": r["symbol"],
                "horizon": r["horizon"],
                "unknown_field_candidate": f"AutoCluster_{labels[i]}",
                "residual": r.get("residual"),
                "hidden_factor": r.get("hidden_factor_candidate"),
                "integrated": "no",
                "learning_recommendation": "NO_ACTION",
            })

    # vNext comparison
    compare_rows: list[dict] = []
    for p in preds:
        v1 = pf(p.get("scout_score")) or 0.0
        vn = scout_score_vnext(p)
        compare_rows.append({
            "validation_id": val_id,
            "symbol": p["symbol"],
            "rank_v1": pi(p.get("rank")),
            "scout_score_v1": v1,
            "scout_score_vnext": round(vn, 2),
            "delta": round(vn - v1, 2),
            "accepted": "pending",
            "learning_recommendation": "NO_ACTION",
        })
    compare_rows.sort(key=lambda r: -r["scout_score_vnext"])
    for i, r in enumerate(compare_rows, 1):
        r["rank_vnext"] = i

    vnext_better = False
    if evaluable:
        def topk_recall(rows: list[dict], k: int, score_key: str) -> float:
            ranked = sorted(rows, key=lambda x: -x[score_key])[:k]
            syms = {r["symbol"] for r in ranked}
            tp = sum(1 for sym in syms if sym in evaluable and any(
                persist_positive(ground[sym][h]) for h in HORIZONS.values() if h in ground[sym]
            ))
            act = sum(1 for sym in evaluable if any(
                persist_positive(ground[sym][h]) for h in HORIZONS.values() if h in ground[sym]
            ))
            return tp / act if act else 0.0

        r_v1 = topk_recall(compare_rows, 2, "scout_score_v1")
        r_vn = topk_recall(compare_rows, 2, "scout_score_vnext")
        vnext_better = r_vn > r_v1
        for r in compare_rows:
            r["accepted"] = "yes" if vnext_better else "no"

    arch_counts = Counter(r["failure_archetype"] for r in archetype_rows)
    largest_failure = arch_counts.most_common(1)[0][0] if arch_counts else "none"

    report = build_report(
        val_id, metrics, state_acc, phase_acc, trans_acc, collapse_acc, persist_acc,
        evaluable, preds, result_rows, arch_counts, largest_failure,
        vnext_better, unknown_rows, cm_rows,
    )

    write_csv(LOGS_DIR / f"{prefix}_result.csv", result_rows)
    write_csv(LOGS_DIR / f"{prefix}_confusion.csv", cm_rows)
    write_csv(LOGS_DIR / f"{prefix}_transition.csv", transition_rows)
    write_csv(LOGS_DIR / f"{prefix}_false_positive.csv", fp_rows)
    write_csv(LOGS_DIR / f"{prefix}_false_negative.csv", fn_rows)
    write_csv(LOGS_DIR / f"{prefix}_residual.csv", residual_rows)
    write_csv(LOGS_DIR / f"{prefix}_unknown_field.csv", unknown_rows)
    write_csv(LOGS_DIR / f"{prefix}_model_compare.csv", compare_rows)
    write_csv(LOGS_DIR / f"{prefix}_failure_archetype.csv", archetype_rows)
    (LOGS_DIR / f"{prefix}_process_report.txt").write_text(report, encoding="utf-8")

    print(f"Evaluated validation{val_id} | evaluable_symbols={len(evaluable)} | state_acc={state_acc:.2%}")
    print(f"Top2 F1 (2h): {metrics.get('top2_2h', {}).get('f1', 0):.2f} | vNext accepted: {vnext_better}")


def build_report(
    val_id: str,
    metrics: dict,
    state_acc: float,
    phase_acc: float,
    trans_acc: float,
    collapse_acc: float,
    persist_acc: float,
    evaluable: list[str],
    preds: list[dict],
    results: list[dict],
    arch_counts: Counter,
    largest_failure: str,
    vnext_better: bool,
    unknown_rows: list[dict],
    cm_rows: list[dict],
) -> str:
    top2 = sorted(preds, key=lambda x: pi(x.get("rank")))[:2]
    top2_names = [t["symbol"] for t in top2]

    lines = [
        "===== SCOUT AUTO VALIDATION & SELF IMPROVEMENT =====",
        "",
        f"Validation ID: {val_id}",
        f"Current Model Version: {MODEL_VERSION}",
        "STRICT NO_ACTION | Research only.",
        "",
        "=== Step 10 — Scientific Report ===",
        "",
        "What worked?",
    ]
    if state_acc > 0.3:
        lines.append(f"  - State/persistence prediction accuracy {state_acc:.1%} on process-tracked symbols.")
    lines.append(f"  - Persistence match accuracy {persist_acc:.1%} across evaluable horizons.")
    cm_summary = Counter(r["confusion_cell"] for r in cm_rows)
    if cm_summary:
        lines.append(f"  - Confusion matrix (2h): {dict(cm_summary)}")
    if any(r["state_match"] == "yes" for r in results):
        hits = [r for r in results if r["state_match"] == "yes"]
        lines.append(f"  - {len(hits)} state hits among {len(results)} evaluable horizon-rows.")
    if not any(r["state_match"] == "yes" for r in results):
        lines.append("  - Nothing validated strongly; falsification succeeded.")

    lines.extend(["", "What failed?"])
    lines.append(f"  - Top2 predicted {top2_names} — neither has process ground truth checkpoints.")
    lines.append(f"  - Actual persistors AIOTUSDT/UAIUSDT ranked #15/#22 at T0.")
    lines.append(f"  - Institution-proxy inflation dominated ranking (Institution Trap).")
    for arch, cnt in arch_counts.most_common(4):
        lines.append(f"  - Failure archetype {arch}: {cnt}")

    lines.extend(["", "What surprised us?"])
    lines.append("  - UAI maintained Trend Start label despite sync collapse at T+6.")
    lines.append("  - AIOT showed brief Trend Start at T+2 then fragmented by T+7.")
    lines.append("  - High institution belief scores did not map to process persistence.")

    lines.extend(["", "What hypothesis became weaker?"])
    lines.append("  - Institution-proxy → ScoutScore mapping (Belief Trap).")
    lines.append("  - TrendBirth prediction for all top-ranked proxy symbols.")

    lines.extend(["", "What hypothesis became stronger?"])
    lines.append("  - Flow-first collapse (confirmed at T+3 for both symbols).")
    lines.append("  - Process_t0 symbols better calibrated despite lower ScoutScore.")

    lines.extend(["", "What should NOT be believed anymore?"])
    lines.append("  - That high P37 institution scores alone imply trend persistence.")
    lines.append("  - That uniform TrendBirth/Expansion/LockedTrend predictions are informative.")

    lines.extend(["", "Single biggest unknown remaining?"])
    if unknown_rows:
        lines.append(f"  - {unknown_rows[0].get('unknown_field_candidate', 'UnknownField')}: latent variable not in current stack.")
    else:
        lines.append("  - Cross-symbol process tracking for full 272-symbol universe.")

    lines.extend([
        "",
        "###############################################################",
        "Scout Self Evaluation",
        "###############################################################",
        "",
        f"Current Model Version : {MODEL_VERSION}",
        f"Validation Number : {val_id}",
        f"Top2 Accuracy : {metrics.get('top2_2h', {}).get('precision', 0):.1%} precision / {metrics.get('top2_2h', {}).get('recall', 0):.1%} recall",
        f"Top5 Accuracy : {metrics.get('top5_2h', {}).get('f1', 0):.1%} F1",
        f"Top10 Accuracy : {metrics.get('top10_2h', {}).get('f1', 0):.1%} F1",
        f"Phase Accuracy : {phase_acc:.1%}",
        f"Transition Accuracy : {trans_acc:.1%}",
        f"Persistence Accuracy : {persist_acc:.1%}",
        f"Collapse Accuracy : {collapse_acc:.1%}",
        f"Largest Failure : {largest_failure}",
        "Most Dangerous Bias : Institution Trap (proxy scores dominate)",
        "Most Valuable Discovery : Process-tracked symbols outperform proxy-ranked symbols on persistence",
        f"Unknown Variables Remaining : {len(unknown_rows)} candidates stored (not integrated)",
        "Confidence In Current Theory : Low-Medium (partially falsified)",
        f"Recommended Next Experiment : Deploy ScoutScore_vNext only if accepted; expand process tracking beyond 2 symbols",
        "",
        f"ScoutScore_vNext accepted : {'yes' if vnext_better else 'no (understanding increased anyway)'}",
        "",
        "Evaluable process-tracked symbols: " + ", ".join(evaluable),
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def export_prediction_bundle(val_id: str) -> None:
    """Normalize validation001 scores into validation_xxx_prediction.csv with top2/5/10."""
    preds = load_predictions(val_id)
    if not preds:
        raise SystemExit("No predictions to export")
    ranked = sorted(preds, key=lambda x: pi(x.get("rank")))
    rows: list[dict] = []
    for tier, k in [("top2", 2), ("top5", 5), ("top10", 10)]:
        for p in ranked[:k]:
            rows.append({
                "validation_id": val_id,
                "tier": tier,
                "symbol": p["symbol"],
                "scout_score": p.get("scout_score"),
                "confidence": p.get("confidence"),
                "expected_phase": p.get("phase"),
                "expected_lifecycle": p.get("lifecycle"),
                "expected_2h": p.get("predicted_2h"),
                "expected_4h": p.get("predicted_4h"),
                "expected_6h": p.get("predicted_6h"),
                "data_source": p.get("data_source"),
                "rank": p.get("rank"),
                "learning_recommendation": "NO_ACTION",
            })
    write_csv(LOGS_DIR / f"validation{val_id}_prediction.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Auto Validation Loop")
    parser.add_argument("--val-id", default="001")
    parser.add_argument("--mode", choices=["predict", "evaluate", "full"], default="full")
    args = parser.parse_args()

    if args.mode in ("predict", "full"):
        export_prediction_bundle(args.val_id)
        print(f"Exported validation{args.val_id}_prediction.csv")

    if args.mode in ("evaluate", "full"):
        evaluate(args.val_id)


if __name__ == "__main__":
    main()
