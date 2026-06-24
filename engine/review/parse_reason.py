"""Parse A6 brief reason strings for review CSV."""

from __future__ import annotations

import re


def parse_reason_fields(reason: str) -> dict:
    out = {"reason_1h": "", "reason_2h": "", "range_pct": 0.0}
    if not reason:
        return out
    m1 = re.search(r"1h=([^\s|]+)", reason)
    m2 = re.search(r"2h=([^\s|]+)", reason)
    mr = re.search(r"(?:1h_)?rng=([\d.]+)%", reason)
    if m1:
        out["reason_1h"] = m1.group(1)
    if m2:
        out["reason_2h"] = m2.group(1)
    if mr:
        out["range_pct"] = float(mr.group(1))
    return out
