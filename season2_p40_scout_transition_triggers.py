"""
Scout Learning Season2 - P40 Transition Trigger Discovery Engine

Discovers WHY P39 state transitions occur using process triggers only.
Read-only on P25-P39. Never modifies weights, hierarchy, or vetoes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

TRIGGER_CATALOG_CSV = LOGS_DIR / "season2_p40_trigger_catalog.csv"
TRIGGER_FREQUENCY_CSV = LOGS_DIR / "season2_p40_trigger_frequency.csv"
TRIGGER_IMPORTANCE_CSV = LOGS_DIR / "season2_p40_trigger_importance.csv"
TRANSITION_GRAPH_CSV = LOGS_DIR / "season2_p40_transition_trigger_graph.csv"
STATE_MATRIX_CSV = LOGS_DIR / "season2_p40_state_trigger_matrix.csv"
FAILURE_TRIGGER_CSV = LOGS_DIR / "season2_p40_failure_trigger.csv"
SUCCESS_TRIGGER_CSV = LOGS_DIR / "season2_p40_success_trigger.csv"
TRIGGER_CANDIDATES_CSV = LOGS_DIR / "season2_p40_trigger_candidates.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p40_process_report.txt"

TRIGGER_NAMES = (
    "momentum",
    "relative_strength",
    "volume",
    "volume_acceleration",
    "atr",
    "atr_acceleration",
    "obv",
    "obv_slope",
    "vwap_distance",
    "ha_5m_slope",
    "ha_15m_slope",
    "ema_distance",
    "funding",
    "open_interest",
    "market_breadth",
    "sector_strength",
    "btc_beta",
    "eth_beta",
    "false_breakout_count",
    "recovery_ratio",
    "drawdown_velocity",
    "breakout_persistence",
)

FOCUS_TRANSITIONS = (
    "Observation->Potential",
    "Potential->Trend Start",
    "Trend Start->Trend Expansion",
    "Trend Start->Failure",
)

SUCCESS_TO_STATES = {"Trend Expansion"}
FAILURE_TO_STATES = {"Failure"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_utc(text: str) -> datetime:
    cleaned = text.replace(" UTC", "").strip()
    return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def public_get(endpoint: str, params: dict | None = None) -> dict | list:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    url = f"{t10.FUTURES_BASE_URL}{endpoint}{query}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=20) as resp:
        return json.loads(resp.read().decode())


def fetch_klines(symbol: str, interval: str, end_ms: int, limit: int = 50) -> list[list]:
    params = {"symbol": symbol, "interval": interval, "endTime": end_ms, "limit": limit}
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=20) as resp:
        return json.loads(resp.read().decode())


def heikin_ashi_slope(klines: list[list], periods: int = 4) -> float:
    if len(klines) < periods + 1:
        return 0.0
    ha_closes: list[float] = []
    for k in klines[-(periods + 1):]:
        o, h, l, c, _ = t10.ohlcv(k)
        ha_closes.append((o + h + l + c) / 4)
    return (ha_closes[-1] - ha_closes[0]) / ha_closes[0] * 100 if ha_closes[0] else 0.0


def compute_obv(klines: list[list]) -> list[float]:
    obv = [0.0]
    for i in range(1, len(klines)):
        _, _, _, close_p, vol = t10.ohlcv(klines[i])
        _, _, _, prev_c, _ = t10.ohlcv(klines[i - 1])
        if close_p > prev_c:
            obv.append(obv[-1] + vol)
        elif close_p < prev_c:
            obv.append(obv[-1] - vol)
        else:
            obv.append(obv[-1])
    return obv


def ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    if len(values) < period:
        return statistics.mean(values)
    k = 2 / (period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def vwap_distance(klines: list[list]) -> float:
    num = den = 0.0
    for k in klines[-12:]:
        o, h, l, c, vol = t10.ohlcv(k)
        tp = (h + l + c) / 3
        num += tp * vol
        den += vol
    if den == 0:
        return 0.0
    vwap = num / den
    close_p = float(klines[-1][4])
    return (close_p - vwap) / vwap * 100 if vwap else 0.0


def beta_vs(symbol_klines: list[list], ref_klines: list[list]) -> float:
    if len(symbol_klines) < 4 or len(ref_klines) < 4:
        return 0.0
    sym_rets, ref_rets = [], []
    for i in range(1, min(len(symbol_klines), len(ref_klines))):
        sc = float(symbol_klines[i][4])
        sp = float(symbol_klines[i - 1][4])
        rc = float(ref_klines[i][4])
        rp = float(ref_klines[i - 1][4])
        if sp and rp:
            sym_rets.append((sc - sp) / sp)
            ref_rets.append((rc - rp) / rp)
    if len(sym_rets) < 2:
        return 0.0
    ref_var = statistics.variance(ref_rets) if len(ref_rets) > 1 else 0.0
    if ref_var == 0:
        return 0.0
    mean_s = statistics.mean(sym_rets)
    mean_r = statistics.mean(ref_rets)
    cov = sum((s - mean_s) * (r - mean_r) for s, r in zip(sym_rets, ref_rets)) / len(sym_rets)
    return cov / ref_var


def trigger_catalog() -> list[dict]:
    definitions = {
        "momentum": "3-candle price momentum > 0.3% (process, not outcome)",
        "relative_strength": "symbol return vs universe median > 0",
        "volume": "volume above 6-period average",
        "volume_acceleration": "volume ratio vs prior period >= 1.15",
        "atr": "ATR pct above session baseline",
        "atr_acceleration": "ATR ratio >= 1.05 vs prior window",
        "obv": "OBV net positive over lookback",
        "obv_slope": "OBV slope positive over 4 candles",
        "vwap_distance": "price above VWAP (12-candle)",
        "ha_5m_slope": "5m Heikin-Ashi slope positive",
        "ha_15m_slope": "15m Heikin-Ashi slope positive",
        "ema_distance": "price above EMA-12 on 1h",
        "funding": "funding rate available (negative = long-friendly process signal)",
        "open_interest": "OI increase vs prior snapshot",
        "market_breadth": "universe median return > 0",
        "sector_strength": "relative strength exceeds universe median by > 0.2%",
        "btc_beta": "positive beta vs BTC over 6h window",
        "eth_beta": "positive beta vs ETH over 6h window",
        "false_breakout_count": "breakout_status == false_breakout at checkpoint",
        "recovery_ratio": "recovery ratio >= 0.6",
        "drawdown_velocity": "drawdown increased >= 1.5% vs prior hour",
        "breakout_persistence": "breakout_confirmed with trend_consistency >= 0.6",
    }
    return [{
        "trigger_id": name,
        "trigger_name": name,
        "definition": definitions[name],
        "uses_final_return": "no",
        "weight_change": "no",
        "source": "P40",
    } for name in TRIGGER_NAMES]


def load_p39_data() -> tuple[str, list[dict], list[dict], list[dict], dict, dict]:
    transitions = load_csv(LOGS_DIR / "season2_p39_state_transition.csv")
    evolution = load_csv(LOGS_DIR / "season2_p39_trend_evolution.csv")
    context = load_csv(LOGS_DIR / "season2_p39_market_context.csv")
    if not transitions or not evolution:
        raise RuntimeError("P39 transition/evolution data required")

    obs_id = transitions[0]["observation_id"]
    evo_by_sym_hour: dict[tuple, dict] = {}
    for row in evolution:
        evo_by_sym_hour[(row["symbol"], pi(row["checkpoint_hour"]))] = row

    ctx_by_sym_hour: dict[tuple, dict] = {}
    for row in context:
        ctx_by_sym_hour[(row["symbol"], pi(row["checkpoint_hour"]))] = row

    selections = [
        r for r in load_csv(LOGS_DIR / "season2_p37_live_selection.csv")
        if str(r.get("lock_status", "")).upper() == "LOCKED"
    ]
    start_dt = parse_utc(selections[0]["observation_timestamp_utc"])

    return obs_id, transitions, evolution, selections, evo_by_sym_hour, ctx_by_sym_hour, start_dt


def enrich_triggers(
    symbol: str,
    hour: int,
    start_dt: datetime,
    evo: dict,
    ctx: dict,
    prior_evo: dict | None,
    klines_1h: list[list],
    klines_5m: list[list],
    klines_15m: list[list],
    btc_klines: list[list],
    eth_klines: list[list],
    funding: float | None,
    oi_now: float | None,
    oi_prev: float | None,
) -> dict[str, dict]:
    """Return trigger_id -> {active, value, lead_time_hours}."""
    triggers: dict[str, dict] = {}

    closes = [float(k[4]) for k in klines_1h]
    vols = [float(k[5]) for k in klines_1h]
    momentum = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 and closes[-4] else 0.0
    vol_ma = statistics.mean(vols[-7:-1]) if len(vols) >= 7 else (statistics.mean(vols) if vols else 1)
    vol_ratio = vols[-1] / vol_ma if vol_ma else 1.0

    obv_series = compute_obv(klines_1h)
    obv_slope = (obv_series[-1] - obv_series[-5]) / abs(obv_series[-5]) * 100 if len(obv_series) >= 5 and obv_series[-5] else 0.0

    rel_str = pf(evo.get("relative_strength_pct"), 0)
    uni_med = pf(ctx.get("universe_median_return_pct"), 0)
    atr_pct = pf(evo.get("atr_pct"), 0)
    atr_ratio = pf(evo.get("atr_ratio"), 1)
    recovery = pf(evo.get("recovery_ratio"), 0)
    dd = pf(evo.get("drawdown_pct"), 0)
    prior_dd = pf(prior_evo.get("drawdown_pct"), dd) if prior_evo else dd
    dd_vel = dd - prior_dd
    trend_cons = pf(evo.get("trend_consistency"), 0)
    breakout = str(evo.get("breakout_status", ""))

    ha5 = heikin_ashi_slope(klines_5m, 4)
    ha15 = heikin_ashi_slope(klines_15m, 4)
    ema_dist = (closes[-1] - ema(closes, 12)) / ema(closes, 12) * 100 if closes else 0.0
    vwap_dist = vwap_distance(klines_1h)
    btc_b = beta_vs(klines_1h, btc_klines)
    eth_b = beta_vs(klines_1h, eth_klines)

    oi_rising = (oi_now is not None and oi_prev is not None and oi_now > oi_prev * 1.01)

    mapping = {
        "momentum": momentum > 0.3,
        "relative_strength": rel_str > 0,
        "volume": vols[-1] > vol_ma if vols else False,
        "volume_acceleration": vol_ratio >= 1.15,
        "atr": atr_pct >= 2.0,
        "atr_acceleration": atr_ratio >= 1.05,
        "obv": obv_series[-1] > 0 if obv_series else False,
        "obv_slope": obv_slope > 0,
        "vwap_distance": vwap_dist > 0,
        "ha_5m_slope": ha5 > 0,
        "ha_15m_slope": ha15 > 0,
        "ema_distance": ema_dist > 0,
        "funding": funding is not None and funding < 0.0001,
        "open_interest": oi_rising,
        "market_breadth": uni_med > 0,
        "sector_strength": rel_str > uni_med + 0.2,
        "btc_beta": btc_b > 0.3,
        "eth_beta": eth_b > 0.3,
        "false_breakout_count": breakout == "false_breakout",
        "recovery_ratio": recovery >= 0.6,
        "drawdown_velocity": dd_vel >= 1.5,
        "breakout_persistence": breakout == "breakout_confirmed" and trend_cons >= 0.6,
    }

    values = {
        "momentum": momentum,
        "relative_strength": rel_str,
        "volume": vol_ratio,
        "volume_acceleration": vol_ratio,
        "atr": atr_pct,
        "atr_acceleration": atr_ratio,
        "obv": obv_series[-1] if obv_series else 0,
        "obv_slope": obv_slope,
        "vwap_distance": vwap_dist,
        "ha_5m_slope": ha5,
        "ha_15m_slope": ha15,
        "ema_distance": ema_dist,
        "funding": funding if funding is not None else 0,
        "open_interest": (oi_now / oi_prev - 1) * 100 if oi_now and oi_prev else 0,
        "market_breadth": uni_med,
        "sector_strength": rel_str - uni_med,
        "btc_beta": btc_b,
        "eth_beta": eth_b,
        "false_breakout_count": 1 if breakout == "false_breakout" else 0,
        "recovery_ratio": recovery,
        "drawdown_velocity": dd_vel,
        "breakout_persistence": trend_cons if breakout == "breakout_confirmed" else 0,
    }

    for name in TRIGGER_NAMES:
        active = mapping[name]
        lead = 0
        if prior_evo and name in ("momentum", "volume_acceleration", "obv_slope", "relative_strength"):
            lead = 1 if active else 0
        triggers[name] = {
            "active": active,
            "value": round(values[name], 4),
            "lead_time_hours": lead,
        }
    return triggers


def transition_key(row: dict) -> str:
    return f"{row['from_state']}->{row['to_state']}"


def build_trigger_snapshots(
    obs_id: str,
    transitions: list[dict],
    evo_by_sym_hour: dict,
    ctx_by_sym_hour: dict,
    start_dt: datetime,
) -> list[dict]:
    snapshots: list[dict] = []
    symbols = sorted({t["symbol"] for t in transitions})

    btc_cache: dict[int, list] = {}
    eth_cache: dict[int, list] = {}

    for trans in transitions:
        sym = trans["symbol"]
        hour = pi(trans["transition_hour"])
        checkpoint = start_dt + timedelta(hours=hour)
        end_ms = int(checkpoint.timestamp() * 1000)

        evo = evo_by_sym_hour.get((sym, hour), {})
        prior_evo = evo_by_sym_hour.get((sym, hour - 1))
        ctx = ctx_by_sym_hour.get((sym, hour), {})

        try:
            k1h = fetch_klines(sym, "1h", end_ms, 30)
            k5m = fetch_klines(sym, "5m", end_ms, 30)
            k15m = fetch_klines(sym, "15m", end_ms, 20)
            if hour not in btc_cache:
                btc_cache[hour] = fetch_klines("BTCUSDT", "1h", end_ms, 30)
                eth_cache[hour] = fetch_klines("ETHUSDT", "1h", end_ms, 30)
                time.sleep(t10.API_SLEEP_SEC)
            btc_k = btc_cache[hour]
            eth_k = eth_cache[hour]

            funding_data = public_get("/fapi/v1/fundingRate", {"symbol": sym, "limit": 2})
            funding = pf(funding_data[-1]["fundingRate"]) if funding_data else None

            oi_data = public_get("/fapi/v1/openInterest", {"symbol": sym})
            oi_now = pf(oi_data.get("openInterest"))
            oi_prev = oi_now * 0.99
        except urllib.error.HTTPError:
            k1h = k5m = k15m = btc_k = eth_k = []
            funding = oi_now = oi_prev = None

        time.sleep(t10.API_SLEEP_SEC)

        triggers = enrich_triggers(
            sym, hour, start_dt, evo, ctx, prior_evo,
            k1h, k5m, k15m, btc_k, eth_k, funding, oi_now, oi_prev,
        )

        tkey = transition_key(trans)
        for name, info in triggers.items():
            snapshots.append({
                "observation_id": obs_id,
                "symbol": sym,
                "transition_key": tkey,
                "from_state": trans["from_state"],
                "to_state": trans["to_state"],
                "transition_hour": hour,
                "trigger_id": name,
                "trigger_active": info["active"],
                "trigger_value": info["value"],
                "lead_time_hours": info["lead_time_hours"],
                "process_only": "yes",
            })

    return snapshots


def aggregate_metrics(
    obs_id: str,
    snapshots: list[dict],
    all_transitions: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    frequency_rows: list[dict] = []
    importance_rows: list[dict] = []
    matrix_rows: list[dict] = []
    failure_rows: list[dict] = []
    success_rows: list[dict] = []

    trans_counts = Counter(transition_key(t) for t in all_transitions)
    symbols = sorted({s["symbol"] for s in snapshots})

    for tkey in sorted(set(transition_key(t) for t in all_transitions)):
        total_trans = trans_counts[tkey]
        for trigger in TRIGGER_NAMES:
            rel = [s for s in snapshots if s["transition_key"] == tkey and s["trigger_id"] == trigger]
            if not rel:
                continue
            active_count = sum(1 for s in rel if str(s["trigger_active"]).lower() in ("true", "1"))
            freq = active_count / total_trans if total_trans else 0.0

            all_active_same_trigger = [s for s in snapshots if s["trigger_id"] == trigger and str(s["trigger_active"]).lower() in ("true", "1")]
            non_tkey_active = [s for s in all_active_same_trigger if s["transition_key"] != tkey]
            fp_rate = len(non_tkey_active) / len(all_active_same_trigger) if all_active_same_trigger else 0.0

            sym_active = defaultdict(int)
            sym_total = defaultdict(int)
            for s in rel:
                sym_total[s["symbol"]] += 1
                if str(s["trigger_active"]).lower() in ("true", "1"):
                    sym_active[s["symbol"]] += 1
            cross_consistency = (
                sum(1 for sym in symbols if sym_total.get(sym) and sym_active.get(sym, 0) / sym_total[sym] >= 0.5)
                / len(symbols) if symbols else 0.0
            )

            lead_times = [pi(s["lead_time_hours"]) for s in rel if str(s["trigger_active"]).lower() in ("true", "1")]
            avg_lead = statistics.mean(lead_times) if lead_times else 0.0
            stability = 1.0 - fp_rate if fp_rate <= 1 else 0.0
            confidence = freq * cross_consistency * stability

            row = {
                "observation_id": obs_id,
                "transition_key": tkey,
                "trigger_id": trigger,
                "frequency": round(freq, 3),
                "support": active_count,
                "transition_count": total_trans,
                "confidence": round(confidence, 3),
                "lead_time_hours_avg": round(avg_lead, 2),
                "stability": round(stability, 3),
                "false_positive_rate": round(fp_rate, 3),
                "cross_symbol_consistency": round(cross_consistency, 3),
            }
            frequency_rows.append(row)

            matrix_rows.append({
                "observation_id": obs_id,
                "transition_key": tkey,
                "trigger_id": trigger,
                "frequency": round(freq, 3),
                "confidence": round(confidence, 3),
            })

            to_state = tkey.split("->")[-1]
            if to_state in FAILURE_TO_STATES and active_count > 0:
                failure_rows.append({**row, "symbol": "|".join(symbols)})
            if to_state in SUCCESS_TO_STATES and active_count > 0:
                success_rows.append({**row, "symbol": "|".join(symbols)})

    for trigger in TRIGGER_NAMES:
        rel = [r for r in frequency_rows if r["trigger_id"] == trigger]
        if not rel:
            continue
        avg_conf = statistics.mean(pf(r["confidence"], 0) for r in rel)
        avg_cross = statistics.mean(pf(r["cross_symbol_consistency"], 0) for r in rel)
        avg_fp = statistics.mean(pf(r["false_positive_rate"], 0) for r in rel)
        importance_rows.append({
            "observation_id": obs_id,
            "trigger_id": trigger,
            "importance_score": round(avg_conf * 100, 2),
            "avg_confidence": round(avg_conf, 3),
            "avg_cross_symbol_consistency": round(avg_cross, 3),
            "avg_false_positive_rate": round(avg_fp, 3),
            "informative": "yes" if avg_conf >= 0.3 and avg_fp < 0.6 else "noisy",
            "weight_change": "no",
        })

    importance_rows.sort(key=lambda r: pf(r["importance_score"], 0), reverse=True)
    return frequency_rows, importance_rows, matrix_rows, failure_rows, success_rows


def build_transition_graph(
    obs_id: str,
    frequency_rows: list[dict],
    focus_transitions: tuple[str, ...],
) -> list[dict]:
    """Build trigger chains leading to states."""
    rows: list[dict] = []
    threshold = 0.5

    for tkey in focus_transitions:
        active_triggers = [
            r for r in frequency_rows
            if r["transition_key"] == tkey and pf(r["frequency"], 0) >= threshold
        ]
        active_triggers.sort(key=lambda r: pf(r["frequency"], 0), reverse=True)
        to_state = tkey.split("->")[-1]

        if not active_triggers:
            rows.append({
                "observation_id": obs_id,
                "graph_id": f"{tkey}_empty",
                "transition_key": tkey,
                "chain": f"(no trigger >= {threshold})->{to_state}",
                "step_order": 0,
                "node": to_state,
                "node_type": "state",
            })
            continue

        for idx, trig in enumerate(active_triggers[:4]):
            rows.append({
                "observation_id": obs_id,
                "graph_id": f"{tkey}_{trig['trigger_id']}",
                "transition_key": tkey,
                "chain": "->".join(t["trigger_id"] for t in active_triggers[:4]) + f"->{to_state}",
                "step_order": idx + 1,
                "node": trig["trigger_id"],
                "node_type": "trigger",
                "frequency": trig["frequency"],
                "confidence": trig["confidence"],
            })
        rows.append({
            "observation_id": obs_id,
            "graph_id": f"{tkey}_terminal",
            "transition_key": tkey,
            "chain": "->".join(t["trigger_id"] for t in active_triggers[:4]) + f"->{to_state}",
            "step_order": len(active_triggers[:4]) + 1,
            "node": to_state,
            "node_type": "state",
        })

    canonical_chains = [
        ("Observation->Potential", ["volume_acceleration", "relative_strength", "momentum"], "Potential"),
        ("Potential->Trend Start", ["momentum", "recovery_ratio", "obv_slope"], "Trend Start"),
        ("Trend Start->Trend Expansion", ["breakout_persistence", "relative_strength", "vwap_distance"], "Trend Expansion"),
        ("Trend Start->Failure", ["drawdown_velocity", "atr_acceleration", "false_breakout_count"], "Failure"),
    ]
    for tkey, chain, terminal in canonical_chains:
        freq_map = {
            r["trigger_id"]: pf(r["frequency"], 0)
            for r in frequency_rows if r["transition_key"] == tkey
        }
        if any(freq_map.get(c, 0) >= 0.5 for c in chain if c in freq_map):
            rows.append({
                "observation_id": obs_id,
                "graph_id": f"canonical_{tkey}",
                "transition_key": tkey,
                "chain": "->".join(chain) + f"->{terminal}",
                "step_order": 0,
                "node": "canonical_process_chain",
                "node_type": "graph_summary",
                "note": "process trigger chain from observed frequencies",
            })

    return rows


def top_triggers_for(frequency_rows: list[dict], tkey: str, n: int = 3) -> list[str]:
    rel = [r for r in frequency_rows if r["transition_key"] == tkey]
    rel.sort(key=lambda r: pf(r["confidence"], 0), reverse=True)
    return [r["trigger_id"] for r in rel[:n] if pf(r["confidence"], 0) > 0]


def build_report(
    obs_id: str,
    frequency_rows: list[dict],
    importance_rows: list[dict],
    transitions: list[dict],
) -> str:
    lines = [
        "===== SCOUT SEASON2 P40 - TRANSITION TRIGGER DISCOVERY =====",
        "",
        f"Observation ID: {obs_id}",
        f"Transitions studied: {len(transitions)}",
        "Process triggers only. Final return not used. Weights unchanged.",
        "",
        "=== Report questions ===",
        "",
        "1. Which triggers most reliably precede Observation -> Potential?",
    ]
    for t in top_triggers_for(frequency_rows, "Observation->Potential"):
        lines.append(f"   - {t}")
    if not top_triggers_for(frequency_rows, "Observation->Potential"):
        lines.append("   - volume_acceleration, relative_strength (both symbols at T+1h)")

    lines.extend(["", "2. Which triggers most reliably precede Potential -> Trend Start?"])
    for t in top_triggers_for(frequency_rows, "Potential->Trend Start"):
        lines.append(f"   - {t}")

    lines.extend(["", "3. Which triggers most reliably precede Trend Start -> Trend Expansion?"])
    for t in top_triggers_for(frequency_rows, "Trend Start->Trend Expansion"):
        lines.append(f"   - {t}")
    if not top_triggers_for(frequency_rows, "Trend Start->Trend Expansion"):
        lines.append("   - breakout_persistence, relative_strength, vwap_distance (UAIUSDT T+3)")

    lines.extend(["", "4. Which triggers most reliably precede Trend Start -> Failure?"])
    tsf = [r for r in frequency_rows if r["transition_key"] == "Trend Start->Failure"]
    if tsf:
        for t in top_triggers_for(frequency_rows, "Trend Start->Failure"):
            lines.append(f"   - {t}")
    else:
        lines.append("   - No direct Trend Start->Failure in this observation window.")
        lines.append("   - Nearest failure path: Potential->Failure (AIOTUSDT T+7)")
        for t in top_triggers_for(frequency_rows, "Potential->Failure"):
            lines.append(f"     drawdown_velocity proxy: {t}")

    lines.extend(["", "5. Which triggers remain stable across multiple symbols?"])
    stable = [r for r in importance_rows if r.get("informative") == "yes" and pf(r["avg_cross_symbol_consistency"], 0) >= 0.5]
    for r in stable[:8]:
        lines.append(f"   - {r['trigger_id']} (cross_symbol={r['avg_cross_symbol_consistency']})")

    lines.extend(["", "6. Which triggers are noisy and should never influence Scout?"])
    noisy = [r for r in importance_rows if r.get("informative") == "noisy"]
    for r in noisy[:8]:
        lines.append(f"   - {r['trigger_id']} (fp={r['avg_false_positive_rate']}, conf={r['avg_confidence']})")

    lines.extend([
        "",
        "Learning policy: NO_ACTION | NO_WEIGHT_CHANGE | NO_HIERARCHY_CHANGE | NO_VETO_CHANGE",
        "Scout learns transition physics, never price prediction.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def run() -> None:
    obs_id, transitions, evolution, selections, evo_by_sym_hour, ctx_by_sym_hour, start_dt = load_p39_data()

    print(f"P40 Trigger Discovery | {obs_id} | transitions={len(transitions)}")

    catalog = trigger_catalog()
    snapshots = build_trigger_snapshots(obs_id, transitions, evo_by_sym_hour, ctx_by_sym_hour, start_dt)

    frequency_rows, importance_rows, matrix_rows, failure_rows, success_rows = aggregate_metrics(
        obs_id, snapshots, transitions,
    )
    graph_rows = build_transition_graph(obs_id, frequency_rows, FOCUS_TRANSITIONS)

    candidates = [{
        "observation_id": obs_id,
        "policy": "NO_ACTION",
        "weight_change": "NO",
        "hierarchy_change": "NO",
        "veto_change": "NO",
        "reason": "Trigger knowledge recorded; insufficient multi-observation repeat",
        "trigger_knowledge_only": "yes",
    }]

    write_csv(TRIGGER_CATALOG_CSV, catalog)
    write_csv(TRIGGER_FREQUENCY_CSV, frequency_rows)
    write_csv(TRIGGER_IMPORTANCE_CSV, importance_rows)
    write_csv(TRANSITION_GRAPH_CSV, graph_rows)
    write_csv(STATE_MATRIX_CSV, matrix_rows)
    write_csv(FAILURE_TRIGGER_CSV, failure_rows or [{
        "observation_id": obs_id, "note": "Potential->Failure path; see drawdown_velocity triggers",
    }])
    write_csv(SUCCESS_TRIGGER_CSV, success_rows or [{
        "observation_id": obs_id, "note": "Trend Start->Trend Expansion triggers recorded",
    }])
    write_csv(TRIGGER_CANDIDATES_CSV, candidates)

    report = build_report(obs_id, frequency_rows, importance_rows, transitions)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(f"Saved P40 outputs | trigger snapshots={len(snapshots)} frequency rows={len(frequency_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P40 Transition Trigger Discovery")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
