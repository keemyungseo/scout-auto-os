"""5m candle cache with partial (in-progress) candle support."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from scout_research_r006_pilot_execution_engine import Bar


@dataclass
class Candle:
    t_ms: int
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0
    closed: bool = True

    def to_bar(self) -> Bar:
        return Bar(self.t_ms, self.o, self.h, self.l, self.c)


@dataclass
class SymbolCandles:
    closed: deque[Candle] = field(default_factory=lambda: deque(maxlen=500))
    partial: Candle | None = None
    last_update_ms: int = 0

    def closed_bars(self) -> list[Bar]:
        return [c.to_bar() for c in self.closed]

    def all_bars(self, include_partial: bool = True) -> list[Bar]:
        bars = self.closed_bars()
        if include_partial and self.partial is not None:
            bars = bars + [self.partial.to_bar()]
        return bars

    def bars_since(self, start_ms: int, include_partial: bool = True) -> list[Bar]:
        out = [c.to_bar() for c in self.closed if c.t_ms >= start_ms]
        if include_partial and self.partial is not None and self.partial.t_ms >= start_ms:
            out.append(self.partial.to_bar())
        return out


class CandleBuilder:
    """Build and maintain 5m candles from kline events."""

    def __init__(self, max_closed: int = 500) -> None:
        self._max_closed = max_closed
        self._symbols: dict[str, SymbolCandles] = {}
        self._lock = Lock()

    def ensure(self, symbol: str) -> SymbolCandles:
        sym = symbol.upper()
        with self._lock:
            if sym not in self._symbols:
                self._symbols[sym] = SymbolCandles(deque(maxlen=self._max_closed))
            return self._symbols[sym]

    def bootstrap_rest(self, symbol: str, klines: list[list]) -> None:
        """Load historical closed klines from REST."""
        sc = self.ensure(symbol)
        with self._lock:
            sc.closed.clear()
            for k in klines:
                if len(k) < 7:
                    continue
                sc.closed.append(Candle(
                    t_ms=int(k[0]),
                    o=float(k[1]),
                    h=float(k[2]),
                    l=float(k[3]),
                    c=float(k[4]),
                    v=float(k[5]),
                    closed=True,
                ))
            if sc.closed:
                sc.last_update_ms = sc.closed[-1].t_ms

    def update_kline(
        self,
        symbol: str,
        t_ms: int,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        is_closed: bool,
    ) -> Candle | None:
        """Apply websocket kline tick. Returns closed candle if one just finished."""
        sc = self.ensure(symbol)
        closed_candle: Candle | None = None
        with self._lock:
            candle = Candle(t_ms, o, h, l, c, v, closed=is_closed)
            sc.partial = candle
            sc.last_update_ms = t_ms
            if is_closed:
                if not sc.closed or sc.closed[-1].t_ms != t_ms:
                    sc.closed.append(candle)
                else:
                    sc.closed[-1] = candle
                sc.partial = None
                closed_candle = candle
        return closed_candle

    def get_price(self, symbol: str) -> float:
        sc = self.ensure(symbol)
        with self._lock:
            if sc.partial is not None:
                return sc.partial.c
            if sc.closed:
                return sc.closed[-1].c
        return 0.0

    def get_bars(self, symbol: str, include_partial: bool = True) -> list[Bar]:
        sc = self.ensure(symbol)
        with self._lock:
            return sc.all_bars(include_partial=include_partial)

    def get_bars_since(self, symbol: str, start_ms: int, include_partial: bool = True) -> list[Bar]:
        sc = self.ensure(symbol)
        with self._lock:
            return sc.bars_since(start_ms, include_partial=include_partial)

    def get_partial(self, symbol: str) -> Candle | None:
        sc = self.ensure(symbol)
        with self._lock:
            return sc.partial

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._symbols.keys())
