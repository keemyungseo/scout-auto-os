"""Manual position protection — highest priority guard."""

from __future__ import annotations


def is_protected(position: dict) -> bool:
    if int(position.get("manual_lock") or 0):
        return True
    if not int(position.get("auto_manage", 1)):
        return True
    if str(position.get("source", "")).upper() == "MANUAL":
        return True
    return False


def can_enter(symbol: str, occupied: set[str], locked: set[str], open_positions: list[dict]) -> bool:
    if symbol in locked:
        return False
    if symbol in occupied:
        return False
    for p in open_positions:
        if p.get("symbol") == symbol and is_protected(p):
            return False
    return True


def can_exit(position: dict) -> bool:
    return not is_protected(position)


def can_auto_manage(position: dict) -> bool:
    return not is_protected(position)


def guard_row(position: dict, event: str, detail: str) -> dict:
    return {
        "symbol": position.get("symbol", ""),
        "position_id": position.get("position_id", ""),
        "source": position.get("source", ""),
        "auto_manage": position.get("auto_manage", 1),
        "manual_lock": position.get("manual_lock", 0),
        "event": event,
        "detail": detail,
        "blocked": is_protected(position),
    }
