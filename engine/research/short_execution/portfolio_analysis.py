"""Research 5 — Long3 + Short3 independent portfolio analysis."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.ranking_engine.metrics import equity_mdd, sharpe


def _sector(symbol: str) -> str:
    s = symbol.upper()
    for suffix in ("USDT", "USDC", "BUSD"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def portfolio_mix_analysis(
    long_picks: list[dict],
    short_picks: list[dict],
    hold_field_long: str = "return_2h",
    hold_field_short: str = "short_return_2h",
) -> tuple[list[dict], dict]:
    """Per-scan Long3 + Short3 returns — engines independent, no mutual block."""
    long_by: dict[str, list[dict]] = defaultdict(list)
    short_by: dict[str, list[dict]] = defaultdict(list)
    for p in long_picks:
        long_by[p["scan_kst"]].append(p)
    for p in short_picks:
        short_by[p["scan_kst"]].append(p)

    scans = sorted(set(long_by) | set(short_by))
    scan_rows: list[dict] = []
    long_rets: list[float] = []
    short_rets: list[float] = []
    combined_rets: list[float] = []
    overlap_symbols = 0
    same_sector = 0
    both_loss = 0
    both_win = 0
    scan_count = 0

    for scan in scans:
        lp = long_by.get(scan, [])
        sp = short_by.get(scan, [])
        if not lp and not sp:
            continue
        scan_count += 1
        lr = [float(p.get(hold_field_long, 0)) for p in lp]
        sr = [float(p.get(hold_field_short, 0)) for p in sp]
        long_rets.extend(lr)
        short_rets.extend(sr)
        combined = lr + sr
        combined_rets.extend(combined)

        lsyms = {p["symbol"] for p in lp}
        ssyms = {p["symbol"] for p in sp}
        overlap = lsyms & ssyms
        overlap_symbols += len(overlap)

        lsec = {_sector(s) for s in lsyms}
        ssec = {_sector(s) for s in ssyms}
        same_sector += len(lsec & ssec)

        l_avg = statistics.mean(lr) if lr else 0
        s_avg = statistics.mean(sr) if sr else 0
        if l_avg < 0 and s_avg < 0:
            both_loss += 1
        if l_avg > 0 and s_avg > 0:
            both_win += 1

        scan_rows.append({
            "scan_kst": scan,
            "long_count": len(lp),
            "short_count": len(sp),
            "long_avg_return_2h": round(l_avg, 4),
            "short_avg_return_2h": round(s_avg, 4),
            "combined_avg_return_2h": round(statistics.mean(combined), 4) if combined else 0,
            "symbol_overlap": len(overlap),
            "sector_overlap": len(lsec & ssec),
            "long_symbols": ",".join(sorted(lsyms)[:5]),
            "short_symbols": ",".join(sorted(ssyms)[:5]),
        })

    def _corr(a: list[float], b: list[float]) -> float:
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pairs) < 3:
            return 0.0
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in pairs)
        den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
        den_y = sum((y - my) ** 2 for y in ys) ** 0.5
        if den_x < 1e-9 or den_y < 1e-9:
            return 0.0
        return round(num / (den_x * den_y), 4)

    scan_long_avg = [float(r["long_avg_return_2h"]) for r in scan_rows if r["long_count"]]
    scan_short_avg = [float(r["short_avg_return_2h"]) for r in scan_rows if r["short_count"]]
    paired = [
        (float(r["long_avg_return_2h"]), float(r["short_avg_return_2h"]))
        for r in scan_rows if r["long_count"] and r["short_count"]
    ]
    corr = _corr([p[0] for p in paired], [p[1] for p in paired]) if paired else 0.0

    summary = {
        "scan_count": scan_count,
        "long_trades": len(long_rets),
        "short_trades": len(short_rets),
        "total_exposure_per_scan_avg": round(
            (len(long_rets) + len(short_rets)) / max(scan_count, 1), 2,
        ),
        "long_avg_return_2h": round(statistics.mean(long_rets), 4) if long_rets else 0,
        "short_avg_return_2h": round(statistics.mean(short_rets), 4) if short_rets else 0,
        "combined_avg_return_2h": round(statistics.mean(combined_rets), 4) if combined_rets else 0,
        "long_sharpe": sharpe(long_rets),
        "short_sharpe": sharpe(short_rets),
        "combined_sharpe": sharpe(combined_rets),
        "combined_mdd": equity_mdd(combined_rets),
        "scan_level_long_short_corr": corr,
        "total_symbol_overlap_events": overlap_symbols,
        "total_sector_overlap_events": same_sector,
        "simultaneous_loss_scans": both_loss,
        "simultaneous_win_scans": both_win,
        "simultaneous_loss_pct": round(both_loss / max(scan_count, 1) * 100, 2),
        "simultaneous_win_pct": round(both_win / max(scan_count, 1) * 100, 2),
        "capital_split_recommendation": "50/50 long-short notional (independent engines)",
    }
    return scan_rows, summary
