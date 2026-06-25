"""Research safety — failures must never affect LIVE trading."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def research_safe(label: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                print(f"[RESEARCH ERROR] {label}: {exc}")
                return None
        return wrapper  # type: ignore
    return decorator
