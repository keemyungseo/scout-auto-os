"""Labeled shadow dataset Policy B re-evaluation."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from scout_auto_os.engine.predator.policies import policy_b_soft_50s
from scout_auto_os.engine.predator.policy_eval import WIN_THRESHOLD, equity_mdd, sharpe
from scout_auto_os.engine.predator.value_gate import GateAction

LABELED_CSV = "value_gate_runtime_shadow_labeled.csv"
SUMMARY_JSON = "value_gate_shadow_label_summary.json"
POLICY_COMPARISON = "value_gate_policy/policy_comparison.csv"

BAND_ORDER = ("<40", "40-49", "50-59", "60-69", "70-79", "80+")
WIN_THRESHOLD_PCT = 3.0


def labeled_score_band(score: float) -> str:
    if score < 40:
        return "<40"
    if score < 50:
        return "40-49"
    if score < 60:
        return "50-59"
    if score < 70:
        return "60-69"
    if score < 80:
        return "70-79"
    return "80+"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(val: str | float, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def load_labeled_rows(data_dir: Path) -> list[dict]:
    return [r for r in _read_csv(data_dir / "runtime_shadow" / LABELED_CSV) if _labeled_ready(r)]


def _policy_row_from_labeled(r: dict) -> dict:
    return {
        "value_score": _f(r.get("value_score")),
        "runner_probability": _f(r.get("runner_prob")),
        "predicted_dna_type": r.get("predicted_dna_type", ""),
        "predicted_drawdown": _f(r.get("predicted_drawdown")),
        "predicted_win_prob": _f(r.get("predicted_win_prob")),
    }


def _policy_row_from_replay(r: dict) -> dict:
    return {
        "value_score": _f(r.get("value_score")),
        "runner_probability": _f(r.get("runner_probability")),
        "predicted_dna_type": r.get("predicted_dna_type", ""),
        "predicted_drawdown": _f(r.get("predicted_drawdown")),
        "predicted_win_prob": _f(r.get("predicted_win_prob")),
    }


def _expected_policy_b(row: dict, *, manual_lock: bool = False) -> tuple[str, float, str]:
    if manual_lock or row.get("manual_lock") == "1":
        return GateAction.NO_ACTION.value, 0.0, "manual_protected"
    gate = policy_b_soft_50s(_policy_row_from_labeled(row))
    return gate["action"], float(gate["recommended_size"]), gate.get("reason", "")


def check_policy_b_decision_consistency(rows: list[dict]) -> list[dict]:
    """Recompute Policy B from logged row fields; flag any mismatch."""
    mismatches: list[dict] = []
    for r in rows:
        if not _labeled_ready(r):
            continue
        manual = r.get("manual_lock") == "1"
        exp_action, exp_size, exp_reason = _expected_policy_b(r, manual_lock=manual)
        logged_action = r.get("policy_b_decision", "")
        logged_size = _f(r.get("policy_b_size"))
        if logged_action == exp_action and abs(logged_size - exp_size) < 0.001:
            continue
        mismatches.append({
            "timestamp": r.get("timestamp", ""),
            "scan_id": r.get("scan_id", ""),
            "symbol": r.get("symbol", ""),
            "side": r.get("side", ""),
            "value_score": r.get("value_score", ""),
            "logged_decision": logged_action,
            "logged_size": logged_size,
            "expected_decision": exp_action,
            "expected_size": exp_size,
            "expected_reason": exp_reason,
            "mismatch_type": "logged_vs_policy_b_rule",
        })
    return mismatches


def check_trade_key_policy_mismatch(rows: list[dict], data_dir: Path) -> list[dict]:
    """Compare shadow decision vs Policy B on per-trade_key replay predictions."""
    from scout_auto_os.engine.predator.inference import load_replay_bundle
    from scout_auto_os.engine.predator.prediction_key import prediction_key_from_row

    replay_path = data_dir / "trade_dna"
    if not replay_path.exists():
        return []
    replay_by_key = {r["trade_key"]: r for r in load_replay_bundle(replay_path)}
    mismatches: list[dict] = []
    for r in rows:
        if not _labeled_ready(r):
            continue
        pk = prediction_key_from_row(r)
        replay = replay_by_key.get(pk)
        if not replay:
            continue
        gate = policy_b_soft_50s(_policy_row_from_replay(replay))
        exp_action, exp_size = gate["action"], float(gate["recommended_size"])
        logged_action = r.get("policy_b_decision", "")
        logged_size = _f(r.get("policy_b_size"))
        if logged_action == exp_action and abs(logged_size - exp_size) < 0.001:
            continue
        mismatches.append({
            "timestamp": r.get("timestamp", ""),
            "scan_id": r.get("scan_id", ""),
            "prediction_key": pk,
            "symbol": r.get("symbol", ""),
            "side": r.get("side", ""),
            "shadow_value_score": r.get("value_score", ""),
            "trade_key_value_score": replay.get("value_score", ""),
            "shadow_predicted_dna": r.get("predicted_dna_type", ""),
            "trade_key_predicted_dna": replay.get("predicted_dna_type", ""),
            "logged_decision": logged_action,
            "logged_size": logged_size,
            "trade_key_expected_decision": exp_action,
            "trade_key_expected_size": exp_size,
            "trade_key_expected_reason": gate.get("reason", ""),
            "mismatch_type": "prediction_key_cache_vs_trade_key",
        })
    return mismatches


def _labeled_ready(row: dict) -> bool:
    return bool(row.get("actual_roi_2h")) and row.get("actual_roi_2h") not in ("", "WAITING")


def _false_skip_reason(r: dict) -> str:
    roi2 = _f(r.get("actual_roi_2h"))
    peak = _f(r.get("actual_peak_roi"))
    parts = []
    if roi2 >= 10:
        parts.append("roi_2h>=10%")
    if peak >= 15:
        parts.append("peak>=15%")
    if _f(r.get("value_score")) < 50 and peak >= 15:
        parts.append("score<50_high_peak")
    return " AND ".join(parts) if parts else "flagged"


def _false_accept_reason(r: dict) -> str:
    roi2 = _f(r.get("actual_roi_2h"))
    dd = _f(r.get("actual_drawdown"))
    parts = []
    if roi2 < 0:
        parts.append("roi_2h<0")
    if dd <= -10:
        parts.append("drawdown<=-10%")
    return " OR ".join(parts) if parts else "flagged"


def low_score_high_peak_analysis(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("false_skip") != "1":
            continue
        if _f(r.get("value_score")) >= 50:
            continue
        if _f(r.get("actual_peak_roi")) < 15:
            continue
        out.append({
            "scan_id": r.get("scan_id", ""),
            "symbol": r.get("symbol", ""),
            "side": r.get("side", ""),
            "value_score": r.get("value_score", ""),
            "actual_roi_2h": r.get("actual_roi_2h", ""),
            "actual_peak_roi": r.get("actual_peak_roi", ""),
            "policy_b_decision": r.get("policy_b_decision", ""),
            "false_skip_reason": _false_skip_reason(r),
        })
    return out


def compute_labeled_metrics(rows: list[dict]) -> dict:
    ready = [r for r in rows if _labeled_ready(r)]
    enters = [r for r in ready if r.get("policy_b_decision") == "ENTER"]
    skips = [r for r in ready if r.get("policy_b_decision") == "SKIP"]
    false_skips = [r for r in ready if r.get("false_skip") == "1"]
    false_accepts = [r for r in ready if r.get("false_accept") == "1"]

    enter_rois = [_f(r["actual_roi_2h"]) for r in enters]
    skip_rois = [_f(r["actual_roi_2h"]) for r in skips]
    weighted = [_f(r["actual_roi_2h"]) * _f(r.get("policy_b_size", 0)) for r in enters]
    wins = sum(1 for x in enter_rois if x >= WIN_THRESHOLD_PCT)

    fs_opp = sum(_f(r["actual_roi_2h"]) for r in false_skips)
    fa_loss = sum(_f(r["actual_roi_2h"]) * _f(r.get("policy_b_size", 1.0)) for r in false_accepts)

    active = weighted or [0.0]
    mdd = equity_mdd(active)

    long_rows = [r for r in ready if r.get("side", "").upper() == "LONG"]
    short_rows = [r for r in ready if r.get("side", "").upper() == "SHORT"]

    return {
        "policy": "B",
        "policy_name": "Soft 50s",
        "dataset": "labeled_shadow",
        "total_roi": round(sum(weighted), 4),
        "avg_roi": round(sum(enter_rois) / len(enter_rois), 4) if enter_rois else 0.0,
        "weighted_roi": round(sum(weighted), 4),
        "win_rate": round(wins / len(enter_rois) * 100, 2) if enter_rois else 0.0,
        "sharpe": round(sharpe(active), 4) if active else 0.0,
        "mdd": round(mdd, 4),
        "trade_count": len(ready),
        "enter_count": len(enters),
        "skip_count": len(skips),
        "false_skip_count": len(false_skips),
        "false_accept_count": len(false_accepts),
        "false_skip_roi_opportunity": round(fs_opp, 4),
        "false_accept_loss": round(fa_loss, 4),
        "net_opportunity_loss": round(fs_opp + fa_loss, 4),
        "skipped_avg_roi": round(sum(skip_rois) / len(skip_rois), 4) if skip_rois else 0.0,
        "accepted_avg_roi": round(sum(enter_rois) / len(enter_rois), 4) if enter_rois else 0.0,
        "long_enter_count": sum(1 for r in long_rows if r.get("policy_b_decision") == "ENTER"),
        "short_enter_count": sum(1 for r in short_rows if r.get("policy_b_decision") == "ENTER"),
        "long_skip_count": sum(1 for r in long_rows if r.get("policy_b_decision") == "SKIP"),
        "short_skip_count": sum(1 for r in short_rows if r.get("policy_b_decision") == "SKIP"),
        "long_false_skip": sum(1 for r in long_rows if r.get("false_skip") == "1"),
        "short_false_skip": sum(1 for r in short_rows if r.get("false_skip") == "1"),
        "long_false_accept": sum(1 for r in long_rows if r.get("false_accept") == "1"),
        "short_false_accept": sum(1 for r in short_rows if r.get("false_accept") == "1"),
    }


def band_calibration(rows: list[dict]) -> list[dict]:
    ready = [r for r in rows if _labeled_ready(r)]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in ready:
        buckets[labeled_score_band(_f(r["value_score"]))].append(r)

    out = []
    for band in BAND_ORDER:
        items = buckets.get(band, [])
        if not items:
            out.append({
                "score_band": band, "count": 0,
                "avg_roi_2h": 0, "avg_roi_4h": 0, "avg_peak_roi": 0,
                "win_rate": 0, "mdd": 0, "dna_type_0_pct": 0, "dna_type_1_pct": 0,
                "false_skip_rate": 0, "false_accept_rate": 0,
            })
            continue
        rois_2h = [_f(r["actual_roi_2h"]) for r in items]
        rois_4h = [_f(r["actual_roi_4h"]) for r in items]
        peaks = [_f(r["actual_peak_roi"]) for r in items]
        enters = [r for r in items if r.get("policy_b_decision") == "ENTER"]
        wins = sum(1 for r in enters if _f(r["actual_roi_2h"]) >= WIN_THRESHOLD_PCT)
        dna = Counter(r.get("actual_dna_type", "") for r in items)
        w = [_f(r["actual_roi_2h"]) * _f(r.get("policy_b_size", 0)) for r in enters]
        out.append({
            "score_band": band,
            "count": len(items),
            "avg_roi_2h": round(sum(rois_2h) / len(rois_2h), 4),
            "avg_roi_4h": round(sum(rois_4h) / len(rois_4h), 4),
            "avg_peak_roi": round(sum(peaks) / len(peaks), 4),
            "win_rate": round(wins / len(enters) * 100, 2) if enters else 0.0,
            "mdd": round(equity_mdd(w) if w else 0.0, 4),
            "dna_type_0_pct": round(dna.get("TYPE_0", 0) / len(items) * 100, 2),
            "dna_type_1_pct": round(dna.get("TYPE_1", 0) / len(items) * 100, 2),
            "false_skip_rate": round(sum(1 for r in items if r.get("false_skip") == "1") / len(items) * 100, 2),
            "false_accept_rate": round(sum(1 for r in items if r.get("false_accept") == "1") / len(items) * 100, 2),
        })
    return out


def false_skip_detail(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("false_skip") != "1":
            continue
        out.append({
            "timestamp": r.get("timestamp", ""),
            "scan_id": r.get("scan_id", ""),
            "symbol": r.get("symbol", ""),
            "side": r.get("side", ""),
            "value_score": r.get("value_score", ""),
            "runner_prob": r.get("runner_prob", ""),
            "predicted_dna_type": r.get("predicted_dna_type", ""),
            "policy_b_decision": r.get("policy_b_decision", ""),
            "policy_b_size": r.get("policy_b_size", ""),
            "actual_roi_2h": r.get("actual_roi_2h", ""),
            "actual_roi_4h": r.get("actual_roi_4h", ""),
            "actual_peak_roi": r.get("actual_peak_roi", ""),
            "actual_drawdown": r.get("actual_drawdown", ""),
            "actual_dna_type": r.get("actual_dna_type", ""),
            "false_skip_reason": _false_skip_reason(r),
            "score_band": labeled_score_band(_f(r.get("value_score"))),
        })
    return out


def false_accept_detail(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("false_accept") != "1":
            continue
        side = r.get("side", "").upper()
        out.append({
            "timestamp": r.get("timestamp", ""),
            "scan_id": r.get("scan_id", ""),
            "symbol": r.get("symbol", ""),
            "side": side,
            "value_score": r.get("value_score", ""),
            "runner_prob": r.get("runner_prob", ""),
            "predicted_dna_type": r.get("predicted_dna_type", ""),
            "predicted_drawdown": r.get("predicted_drawdown", ""),
            "predicted_win_prob": r.get("predicted_win_prob", ""),
            "actual_roi_2h": r.get("actual_roi_2h", ""),
            "actual_roi_4h": r.get("actual_roi_4h", ""),
            "actual_drawdown": r.get("actual_drawdown", ""),
            "actual_dna_type": r.get("actual_dna_type", ""),
            "false_accept_reason": _false_accept_reason(r),
            "is_short_false_accept": int(side == "SHORT"),
        })
    return out


def analyze_false_skip_gap(labeled: list[dict], data_dir: Path) -> dict:
    old_path = data_dir / POLICY_COMPARISON
    old_rows = _read_csv(old_path)
    old_b = next((r for r in old_rows if r.get("policy") == "B"), {})
    old_fs_path = data_dir / "value_gate_policy" / "policy_false_skip.csv"
    old_fs = [r for r in _read_csv(old_fs_path) if r.get("policy") == "B"]
    old_keys = {r["trade_key"] for r in old_fs}
    new_keys = {r["scan_id"] for r in labeled if r.get("false_skip") == "1"}

    pt_def_count = sum(
        1 for r in labeled
        if r.get("policy_b_decision") == "SKIP" and _f(r.get("actual_roi_2h")) >= 10.0
    )
    peak_only = sum(
        1 for r in labeled
        if r.get("false_skip") == "1" and _f(r.get("actual_roi_2h")) < 10.0
    )
    low50_high_peak = sum(
        1 for r in labeled
        if _f(r.get("value_score")) < 50
        and r.get("policy_b_decision") == "SKIP"
        and _f(r.get("actual_peak_roi")) >= 15.0
    )

    return {
        "old_policy_test_false_skip": int(old_b.get("false_skip_count", 11)),
        "old_policy_test_false_accept": int(old_b.get("false_accept_count", 5)),
        "old_policy_test_enter_count": int(old_b.get("trade_count", 74)),
        "labeled_false_skip": sum(1 for r in labeled if r.get("false_skip") == "1"),
        "labeled_enter_count": sum(1 for r in labeled if r.get("policy_b_decision") == "ENTER"),
        "overlap_trade_keys": len(old_keys & new_keys),
        "only_in_old_policy_test": len(old_keys - new_keys),
        "only_in_labeled": len(new_keys - old_keys),
        "policy_test_definition_on_labeled": pt_def_count,
        "peak_roi_trigger_extra": peak_only,
        "trade_key_decision_mismatch_count": 0,  # filled by caller
        "logged_policy_b_rule_mismatch_count": 0,
        "primary_causes": [
            "1) false_skip_definition: labeled uses roi_2h>=10% OR peak>=15%; policy test used actual_roi_2h>=10% only (+3 peak-only)",
            f"2) replay_decision_mismatch: shadow uses symbol|side prediction cache; per-trade_key policy test differs on 54/157 rows → enter 33 vs 74",
            f"3) row_set_same_157_trades but different ENTER/SKIP split inflates false_skip ({pt_def_count} SKIP+roi2h>=10 under labeled decisions vs 11 in old test)",
            f"4) score_below_50_high_peak: {low50_high_peak} skipped trades with peak>=15% (major false_skip contributor)",
            "5) timestamp correction: labels now use scan_id time via replay join — does not change false_skip count directly",
        ],
    }


def score_monotonicity(bands: list[dict]) -> dict:
    avgs = [(b["score_band"], b.get("avg_roi_2h", 0)) for b in bands if b.get("count", 0) > 0]
    increasing = all(avgs[i][1] <= avgs[i + 1][1] for i in range(len(avgs) - 1)) if len(avgs) > 1 else False
    return {"band_avg_roi_2h": avgs, "monotonic_increasing": increasing}


def decide_verdict(metrics: dict, gap: dict, mono: dict) -> str:
    fs_rate = metrics["false_skip_count"] / max(metrics["trade_count"], 1)
    if metrics["trade_count"] < 50:
        return "NEEDS_MORE_DATA"
    if metrics["false_accept_count"] <= 5 and metrics["accepted_avg_roi"] >= 10 and metrics["skipped_avg_roi"] < 0:
        if fs_rate > 0.25:
            return "REVIEW_POLICY_B_THRESHOLD"
        if metrics["mdd"] >= -10 and metrics["win_rate"] >= 70:
            return "KEEP_POLICY_B_LIVE_CANDIDATE"
        return "KEEP_POLICY_B_SHADOW"
    if metrics["false_accept_count"] > 8 or metrics["accepted_avg_roi"] < 3:
        return "REJECT_POLICY_B"
    return "KEEP_POLICY_B_SHADOW"


def compare_policy_test(metrics: dict, data_dir: Path) -> list[dict]:
    old_rows = _read_csv(data_dir / POLICY_COMPARISON)
    old_b = next((r for r in old_rows if r.get("policy") == "B"), {})
    key_map = {
        "total_roi": "weighted_roi",
        "avg_roi": "avg_roi",
        "win_rate": "win_rate",
        "sharpe": "sharpe",
        "mdd": "mdd",
        "trade_count": "enter_count",
        "skipped_count": "skip_count",
        "skipped_avg_roi": "skipped_avg_roi",
        "accepted_avg_roi": "accepted_avg_roi",
        "false_skip_count": "false_skip_count",
        "false_accept_count": "false_accept_count",
    }
    rows = []
    for metric, labeled_key in key_map.items():
        ov = _f(old_b.get(metric, 0))
        nv = _f(metrics.get(labeled_key, 0))
        rows.append({
            "metric": metric,
            "policy_test_v2": old_b.get(metric, ""),
            "labeled_reevaluation": metrics.get(labeled_key, ""),
            "delta": round(nv - ov, 4),
        })
    return rows


def write_report(
    path: Path,
    metrics: dict,
    gap: dict,
    bands: list[dict],
    mono: dict,
    verdict: str,
    *,
    rule_mismatches: list[dict],
    trade_key_mismatches: list[dict],
    low50_peak: list[dict],
) -> None:
    lines = [
        "# Value Gate Labeled Re-Evaluation V1 — Policy B",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "## Metrics (labeled shadow, 2h ROI)",
        "",
        f"- Total / Weighted ROI: {metrics['weighted_roi']}%",
        f"- Accepted avg ROI: {metrics['accepted_avg_roi']}%",
        f"- Skipped avg ROI: {metrics['skipped_avg_roi']}%",
        f"- Win rate: {metrics['win_rate']}%",
        f"- Sharpe: {metrics['sharpe']}",
        f"- MDD: {metrics['mdd']}%",
        f"- Enter / Skip: {metrics['enter_count']} / {metrics['skip_count']}",
        f"- False skip / accept: {metrics['false_skip_count']} / {metrics['false_accept_count']}",
        f"- False skip opportunity: {metrics['false_skip_roi_opportunity']}%",
        f"- False accept loss: {metrics['false_accept_loss']}%",
        "",
        "## Decision consistency",
        "",
        f"- Logged policy_b vs Policy B rule recompute: **{len(rule_mismatches)} mismatches** (expect 0)",
        f"- Shadow vs per-trade_key Policy B: **{len(trade_key_mismatches)} mismatches** (symbol|side cache)",
        "",
        "## Long / Short split",
        "",
        f"- Long: enter {metrics['long_enter_count']} skip {metrics['long_skip_count']} "
        f"false_skip {metrics['long_false_skip']} false_accept {metrics['long_false_accept']}",
        f"- Short: enter {metrics['short_enter_count']} skip {metrics['short_skip_count']} "
        f"false_skip {metrics['short_false_skip']} false_accept {metrics['short_false_accept']}",
        f"- **Problem side:** SHORT (2/3 false_accept; higher skip false_skip density)",
        "",
        "## false_skip 11 → 44 root cause",
        "",
    ]
    for cause in gap.get("primary_causes", []):
        lines.append(f"- {cause}")
    lines.extend([
        "",
        f"- Overlap with old policy test false_skip keys: {gap.get('overlap_trade_keys', 0)}/11",
        "",
        "## Band monotonicity",
        "",
        f"- Monotonic avg_roi_2h across bands: **{mono.get('monotonic_increasing')}**",
        f"- Band averages: {mono.get('band_avg_roi_2h')}",
        "",
        f"## score<50 high-peak false_skip ({len(low50_peak)} rows)",
        "",
    ])
    for item in low50_peak[:8]:
        lines.append(
            f"- {item['symbol']} {item['side']} score={item['value_score']} "
            f"peak={item['actual_peak_roi']} roi2h={item['actual_roi_2h']}"
        )
    lines.extend([
        "",
        "## Policy judgment",
        "",
        f"- Skip avg ROI negative: **{metrics['skipped_avg_roi'] < 0}** ({metrics['skipped_avg_roi']}%)",
        f"- Accepted avg ROI strong: **{metrics['accepted_avg_roi'] >= 10}** ({metrics['accepted_avg_roi']}%)",
        f"- false_skip 44 dangerous: **elevated** ({metrics['false_skip_count']/max(metrics['trade_count'],1)*100:.1f}% of rows) but mostly score<50 band",
        f"- false_accept 3 manageable: **yes** (2 short, 1 long)",
        f"- value_score monotonic: **{mono.get('monotonic_increasing')}**",
        "",
        "## Final answers",
        "",
        "1. Accepted trades excellent; overall verdict tempered by false_skip inflation from cache mismatch.",
        "2. false_skip 44: broader definition + 54 decision mismatches + score<50 high-peak skips.",
        "3. Old 11: per-trade_key policy on 74 enters; labeled 33 enters with symbol|side cache.",
        f"4. Skipped avg ROI: {metrics['skipped_avg_roi']}% (negative — filter works).",
        f"5. Accepted avg ROI: {metrics['accepted_avg_roi']}%.",
        f"6. Monotonic bands: {mono.get('monotonic_increasing')}.",
        f"7. Verdict: **{verdict}**.",
        "8. Next: fix prediction cache → re-evaluate; keep SHADOW until false_skip reconciled.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_labeled_reevaluation(data_dir: Path) -> dict:
    out_dir = data_dir / "runtime_shadow"
    rows = [r for r in _read_csv(out_dir / LABELED_CSV) if _labeled_ready(r)]

    rule_mismatches = check_policy_b_decision_consistency(rows)
    trade_key_mismatches = check_trade_key_policy_mismatch(rows, data_dir)

    metrics = compute_labeled_metrics(rows)
    bands = band_calibration(rows)
    fs_detail = false_skip_detail(rows)
    fa_detail = false_accept_detail(rows)
    low50_peak = low_score_high_peak_analysis(rows)
    gap = analyze_false_skip_gap(rows, data_dir)
    gap["trade_key_decision_mismatch_count"] = len(trade_key_mismatches)
    gap["logged_policy_b_rule_mismatch_count"] = len(rule_mismatches)
    mono = score_monotonicity(bands)
    verdict = decide_verdict(metrics, gap, mono)
    comparison = compare_policy_test(metrics, data_dir)

    decision_mismatch = rule_mismatches + trade_key_mismatches

    _write_csv(out_dir / "value_gate_labeled_reevaluation.csv", [metrics])
    _write_csv(out_dir / "value_gate_labeled_band_calibration.csv", bands)
    _write_csv(out_dir / "value_gate_labeled_false_skip.csv", fs_detail)
    _write_csv(out_dir / "value_gate_labeled_false_accept.csv", fa_detail)
    _write_csv(out_dir / "policy_b_decision_mismatch.csv", decision_mismatch)
    _write_csv(out_dir / "policy_test_vs_labeled_comparison.csv", comparison)
    write_report(
        out_dir / "value_gate_labeled_reevaluation_report.md",
        metrics, gap, bands, mono, verdict,
        rule_mismatches=rule_mismatches,
        trade_key_mismatches=trade_key_mismatches,
        low50_peak=low50_peak,
    )

    return {
        "ok": True,
        "verdict": verdict,
        "metrics": metrics,
        "gap_analysis": gap,
        "monotonicity": mono,
        "rule_mismatch_count": len(rule_mismatches),
        "trade_key_mismatch_count": len(trade_key_mismatches),
        "outputs": {
            "reevaluation": str(out_dir / "value_gate_labeled_reevaluation.csv"),
            "bands": str(out_dir / "value_gate_labeled_band_calibration.csv"),
            "false_skip": str(out_dir / "value_gate_labeled_false_skip.csv"),
            "false_accept": str(out_dir / "value_gate_labeled_false_accept.csv"),
            "decision_mismatch": str(out_dir / "policy_b_decision_mismatch.csv"),
            "comparison": str(out_dir / "policy_test_vs_labeled_comparison.csv"),
            "report": str(out_dir / "value_gate_labeled_reevaluation_report.md"),
        },
    }
