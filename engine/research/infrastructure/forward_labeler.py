"""Forward label generation — 2h+ outcomes from klines only."""

from __future__ import annotations

from scout_auto_os.engine.research.target_discovery.label_builder import compute_label_metrics


def build_forward_labels(klines: list, entry: float | None = None) -> dict | None:
    if not klines or len(klines) < 2:
        return None
    m = compute_label_metrics(klines, entry)
    if not m:
        return None

    ret2 = float(m.get("return_2h", 0))
    vol = float(m.get("intrabar_volatility", 0)) or 1e-9
    sharpe_contrib = round(ret2 / vol, 4)

    return {
        "return_2h": ret2,
        "return_minus_dd": float(m.get("return_minus_dd", 0)),
        "max_drawdown_2h": float(m.get("max_drawdown_2h", 0)),
        "max_up_2h": float(m.get("max_up_2h", 0)),
        "min_return_2h": float(m.get("min_return_2h", 0)),
        "mfe_2h": float(m.get("mfe_2h", 0)),
        "mae_2h": float(m.get("mae_2h", 0)),
        "intrabar_sharpe": float(m.get("intrabar_sharpe", 0)),
        "sharpe_contribution": sharpe_contrib,
        "return_30m": float(m.get("return_30m", 0)),
        "return_1h": float(m.get("return_1h", 0)),
        "return_4h": float(m.get("return_4h", 0)),
    }


def label_scan_from_forward(
    scan_kst: str,
    symbols: list[str],
    fwd: dict[tuple[str, str], list],
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    labeled = 0
    for sym in symbols:
        klines = fwd.get((scan_kst, sym), [])
        labels = build_forward_labels(klines)
        if labels:
            rows.append({"scan_kst": scan_kst, "symbol": sym, **labels})
            labeled += 1
    return rows, labeled
