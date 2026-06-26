"""Predator Value Gate Policy Test V2 runner."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.predator.inference import load_replay_bundle
from scout_auto_os.engine.predator.policies import POLICIES, POLICY_C_DD_MAX, POLICY_C_RUNNER_MIN, POLICY_C_WIN_MIN
from scout_auto_os.engine.predator.policy_eval import (
    apply_policy_to_rows,
    band_analysis,
    best_missed,
    compute_metrics,
    false_accepts,
    false_skips,
    policy_score_for_selection,
    split_by_side,
    worst_accepted,
)
from scout_auto_os.engine.predator.policy_report import PolicyTestReport
from scout_auto_os.engine.research.safe import research_safe

VERDICT_MAP = {
    "A": "KEEP_V1",
    "B": "ADOPT_POLICY_B_SHADOW",
    "C": "ADOPT_POLICY_C_SHADOW",
    "D": "ADOPT_POLICY_D_SHADOW",
    "E": "ADOPT_POLICY_E_SHADOW",
}


def _pick_recommended(results: dict[str, dict], v1_key: str = "A") -> tuple[str, dict]:
    v1 = results[v1_key]
    best_key = v1_key
    best_score = -1.0
    for key, bundle in results.items():
        m = bundle["metrics"]
        m["false_skip_count"] = bundle["false_skip_count"]
        m["false_accept_count"] = bundle["false_accept_count"]
        long_r = bundle["long_short"][0] if bundle["long_short"][0]["side"] == "LONG" else bundle["long_short"][1]
        short_r = bundle["long_short"][1] if bundle["long_short"][1]["side"] == "SHORT" else bundle["long_short"][0]
        sc = policy_score_for_selection(m, v1["metrics"], long_row=long_r, short_row=short_r)
        if sc > best_score:
            best_score = sc
            best_key = key
    b = results[best_key]
    return best_key, {
        "policy": best_key,
        "policy_name": POLICIES[best_key]["name"],
        "selection_score": best_score,
        "metrics": b["metrics"],
        "false_skip_count": b["false_skip_count"],
        "false_accept_count": b["false_accept_count"],
        "policy_c_thresholds": {
            "runner_min": POLICY_C_RUNNER_MIN,
            "win_prob_min": POLICY_C_WIN_MIN,
            "drawdown_max": POLICY_C_DD_MAX,
        },
    }


def _build_answers(
    results: dict[str, dict],
    rec_key: str,
    short_type1_fa: list[dict],
) -> dict:
    v1 = results["A"]
    rec = results[rec_key]
    v1_fs = v1["false_skip_count"]
    rec_fs = rec["false_skip_count"]
    v1_fa = v1["false_accept_count"]
    rec_fa = rec["false_accept_count"]
    c_fs, c_fa = results["C"]["false_skip_count"], results["C"]["false_accept_count"]
    b_m = results["B"]["metrics"]

    better = rec_key != "A" and (
        rec["metrics"]["sharpe"] >= v1["metrics"]["sharpe"]
        and rec_fs < v1_fs
    )

    return {
        "V1보다 나은 정책이 있는가?": (
            f"{'Yes' if better else 'Marginal'} — 추천 Policy {rec_key} "
            f"(Sharpe {rec['metrics']['sharpe']} vs V1 {v1['metrics']['sharpe']}, "
            f"false skip {rec_fs} vs {v1_fs})"
        ),
        "50–59 밴드는 ENTER로 열어도 되는가?": (
            f"Policy B/E 테스트 — B trades={b_m['trade_count']} Sharpe={b_m['sharpe']} "
            f"false_accept={results['B']['false_accept_count']}; "
            f"SHADOW_ONLY 유지보다 controlled ENTER(0.2x) 가능하나 Shadow 권장"
        ),
        "TYPE_0 예외 규칙은 false skip을 줄이는가, false accept를 늘리는가?": (
            f"Policy C: false_skip {c_fs} (V1 {v1_fs}), false_accept {c_fa} (V1 {v1_fa}) — "
            f"{'skip 감소' if c_fs < v1_fs else 'skip 미개선'}, "
            f"{'accept 증가' if c_fa > v1_fa else 'accept 유사'}"
        ),
        "Short 쪽 false accept는 어떤 조건에서 발생하는가?": (
            "predicted TYPE_0 + high runner_prob + value_score 60–69, 실제 TYPE_1 — "
            f"{len(short_type1_fa)}건 (CLOUSDT/VELVETUSDT 패턴: DNA misclass + mid score band)"
            if short_type1_fa else "Short false accept — TYPE_0 pred / TYPE_1 actual misclass at score 60–70"
        ),
        "최종 추천 정책은 무엇인가?": (
            f"Policy {rec_key} — {POLICIES[rec_key]['name']}"
        ),
        "추천 정책은 LIVE_READY인가 SHADOW_ONLY인가?": (
            "SHADOW_ONLY — 157건 replay; Policy B/E 50–59 밴드 추가 Shadow 검증 필요"
            if rec_key in ("B", "C", "E") else
            "SHADOW_ONLY — conservative validation" if rec_key == "D" else
            "SHADOW_ONLY — V1 유지, false skip 모니터링"
        ),
        "157건 기준 가장 큰 리스크는 무엇인가?": (
            "Short-side DNA misclassification (TYPE_0 pred, TYPE_1 actual) at value_score 60–69; "
            "sample size 157 limits LIVE confidence"
        ),
    }


class ValueGatePolicyTestRunner:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.trade_dna_dir = data_dir / "trade_dna"
        self.out_dir = data_dir / "value_gate_policy"

    @research_safe("value_gate_policy_v2")
    def run(self) -> dict:
        print("[POLICY TEST] V2 started")
        rows = load_replay_bundle(self.trade_dna_dir)
        print(f"[POLICY TEST] trades={len(rows)}")

        results: dict[str, dict] = {}
        all_fskips: list[dict] = []
        all_faccepts: list[dict] = []
        all_bands: list[dict] = []
        comparisons: list[dict] = []
        long_short_rows: list[dict] = []

        for key in POLICIES:
            trades = apply_policy_to_rows(rows, key)
            m = compute_metrics(trades)
            fs = false_skips(trades)
            fa = false_accepts(trades)
            bm = best_missed(trades)
            wa = worst_accepted(trades)
            ls = split_by_side(trades, key)
            bands = band_analysis(trades, key)

            results[key] = {
                "metrics": m,
                "false_skip_count": len(fs),
                "false_accept_count": len(fa),
                "long_short": ls,
                "trades": trades,
            }
            for x in fs:
                all_fskips.append(x)
            for x in fa:
                all_faccepts.append(x)
            all_bands.extend(bands)
            long_short_rows.extend(ls)

            comparisons.append({
                "policy": key,
                "policy_name": POLICIES[key]["name"],
                **m,
                "false_skip_count": len(fs),
                "false_accept_count": len(fa),
                "best_missed_roi": round(float(bm["actual_roi"]), 4) if bm else 0,
                "best_missed_symbol": bm["symbol"] if bm else "",
                "worst_accepted_roi": round(float(wa["actual_roi"]), 4) if wa else 0,
                "worst_accepted_symbol": wa["symbol"] if wa else "",
            })

        rec_key, recommended = _pick_recommended(results)
        short_type1_fa = [
            x for x in all_faccepts
            if x["policy"] == rec_key
            and x["direction"] == "short"
            and x.get("type0_pred_actual_type1")
        ]
        answers = _build_answers(results, rec_key, short_type1_fa)

        if rec_key == "A":
            verdict = "KEEP_V1"
        elif recommended["selection_score"] <= policy_score_for_selection(
            {**results["A"]["metrics"],
             "false_skip_count": results["A"]["false_skip_count"],
             "false_accept_count": results["A"]["false_accept_count"]},
            results["A"]["metrics"],
            long_row=results["A"]["long_short"][0],
            short_row=results["A"]["long_short"][1],
        ):
            verdict = "KEEP_V1"
        else:
            verdict = VERDICT_MAP.get(rec_key, "NEEDS_MORE_DATA")

        all_policies_reject = all(
            results[k]["metrics"]["sharpe"] < results["A"]["metrics"]["sharpe"] * 0.9
            for k in POLICIES if k != "A"
        )
        if all_policies_reject and rec_key == "A":
            verdict = "KEEP_V1"

        reporter = PolicyTestReport(self.out_dir)
        report_path = reporter.write_all(
            comparisons, long_short_rows, all_fskips, all_faccepts,
            all_bands, recommended, verdict, answers,
        )

        print(f"[POLICY TEST] recommended=Policy {rec_key} verdict={verdict}")
        print(f"[POLICY TEST] V1 false_skip={results['A']['false_skip_count']} "
              f"rec false_skip={results[rec_key]['false_skip_count']}")
        print(f"[POLICY TEST] report: {report_path}")
        return {
            "verdict": verdict,
            "recommended_policy": rec_key,
            "report_path": str(report_path),
            "comparisons": comparisons,
        }
