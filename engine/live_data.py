"""Binance Futures public WebSocket + REST live data engine (no API key)."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from datetime import timezone, timedelta
from pathlib import Path

from scout_auto_os.engine.candle_builder import CandleBuilder
from scout_auto_os.storage.db import now_kst
from scout_research_r005_execution_statistics import parse_kst
from scout_research_r006_pilot_execution_engine import Bar

KST = timezone(timedelta(hours=9))


def _candle_start_ms(anchor_kst: str) -> int:
    dt = parse_kst(anchor_kst)
    minute = (dt.minute // 5) * 5
    candle_start = dt.replace(minute=minute, second=0, microsecond=0)
    return int(candle_start.timestamp() * 1000)


class LiveDataEngine:
    """Background websocket consumer with REST bootstrap."""

    def __init__(self, config: dict, root: Path) -> None:
        live_cfg = config.get("live_data", {})
        self.enabled = bool(live_cfg.get("enabled", True))
        self.ws_base = live_cfg.get("ws_base", "wss://fstream.binance.com")
        self.rest_base = live_cfg.get("rest_base", "https://fapi.binance.com")
        self.interval = live_cfg.get("interval", "5m")
        self.bootstrap_limit = int(live_cfg.get("bootstrap_limit", 100))
        self.reconnect_sec = float(live_cfg.get("reconnect_sec", 5))
        self.heartbeat_timeout = float(live_cfg.get("heartbeat_timeout_sec", 60))
        log_dir = Path(live_cfg.get("log_dir", "logs/live"))
        if not log_dir.is_absolute():
            log_dir = root / log_dir
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.builder = CandleBuilder()
        self._subscribed: set[str] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ws = None
        self._connected = False
        self._last_msg_time = 0.0
        self._last_price: dict[str, float] = {}
        self._status_path = root / "scout_auto_os" / "data" / "live_status.json"
        self._setup_loggers()

    def _setup_loggers(self) -> None:
        self.ws_log = self._file_logger("websocket", self.log_dir / "websocket.log")
        self.price_log = self._file_logger("live_price", self.log_dir / "live_price.log")
        self.candle_log = self._file_logger("candle", self.log_dir / "candle.log")

    @staticmethod
    def _file_logger(name: str, path: Path) -> logging.Logger:
        logger = logging.getLogger(f"scout_auto_os.{name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
        return logger

    @property
    def connected(self) -> bool:
        if not self.enabled or not self._connected:
            return False
        if time.time() - self._last_msg_time > self.heartbeat_timeout:
            return False
        return True

    def start(self) -> None:
        if not self.enabled:
            self.ws_log.info("live_data disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="LiveDataEngine")
        self._thread.start()
        self.ws_log.info("live_data thread started")

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._connected = False
        self._write_status()

    def subscribe(self, symbols: list[str]) -> None:
        if not self.enabled:
            return
        reconnect = False
        new_syms: list[str] = []
        with self._lock:
            for s in symbols:
                sym = s.upper()
                if sym not in self._subscribed:
                    self._subscribed.add(sym)
                    new_syms.append(sym)
                    reconnect = True
        for sym in new_syms:
            self._bootstrap_symbol(sym)
        if reconnect and self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _bootstrap_symbol(self, symbol: str) -> None:
        try:
            klines = self._rest_klines(symbol, self.bootstrap_limit)
            if klines:
                self.builder.bootstrap_rest(symbol, klines)
                px = float(klines[-1][4])
                self._last_price[symbol] = px
                self.candle_log.info(f"bootstrap {symbol} bars={len(klines)} last={px}")
        except Exception as exc:
            self.ws_log.info(f"bootstrap failed {symbol}: {exc}")

    def _rest_klines(self, symbol: str, limit: int) -> list[list]:
        params = urllib.parse.urlencode({
            "symbol": symbol.upper(),
            "interval": self.interval,
            "limit": limit,
        })
        url = f"{self.rest_base}/fapi/v1/klines?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "ScoutAutoOS/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def get_price(self, symbol: str) -> float:
        sym = symbol.upper()
        px = self.builder.get_price(sym)
        if px > 0:
            return px
        return self._last_price.get(sym, 0.0)

    def get_bars(self, symbol: str, scan_kst: str | None = None) -> list[Bar]:
        if scan_kst:
            return self.get_forward_bars(symbol, scan_kst)
        return self.builder.get_bars(symbol.upper(), include_partial=True)

    def get_forward_bars(self, symbol: str, anchor_kst: str) -> list[Bar]:
        start_ms = _candle_start_ms(anchor_kst)
        return self.builder.get_bars_since(symbol.upper(), start_ms, include_partial=True)

    def health(self) -> dict:
        return {
            "connected": self.connected,
            "subscribed_count": len(self._subscribed),
            "symbols": sorted(self._subscribed)[:20],
            "last_message_age_sec": round(time.time() - self._last_msg_time, 1) if self._last_msg_time else None,
            "last_update": now_kst(),
        }

    def _write_status(self) -> None:
        payload = self.health()
        payload["enabled"] = self.enabled
        self._status_path.parent.mkdir(parents=True, exist_ok=True)
        self._status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._connect_and_listen()
            except Exception as exc:
                self._connected = False
                self.ws_log.info(f"websocket error: {exc}")
                self._write_status()
            if not self._stop.is_set():
                time.sleep(self.reconnect_sec)

    def _connect_and_listen(self) -> None:
        try:
            import websocket
        except ImportError as exc:
            self.ws_log.info(f"websocket-client not installed: {exc}")
            self._connected = False
            self._write_status()
            time.sleep(30)
            return

        streams = self._stream_names()
        if not streams:
            time.sleep(2)
            return

        url = f"{self.ws_base}/stream?streams={'/'.join(streams)}"
        self.ws_log.info(f"connecting streams={len(streams)}")

        def on_open(ws):
            self._connected = True
            self._last_msg_time = time.time()
            self.ws_log.info("websocket connected")
            self._write_status()

        def on_message(ws, message: str):
            self._last_msg_time = time.time()
            self._handle_message(message)

        def on_error(ws, error):
            self.ws_log.info(f"websocket on_error: {error}")
            self._connected = False

        def on_close(ws, code, msg):
            self.ws_log.info(f"websocket closed code={code} msg={msg}")
            self._connected = False
            self._write_status()

        self._ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws.run_forever(ping_interval=20, ping_timeout=10)

    def _stream_names(self) -> list[str]:
        with self._lock:
            syms = sorted(self._subscribed)
        return [f"{s.lower()}@kline_{self.interval}" for s in syms]

    def _handle_message(self, raw: str) -> None:
        msg = json.loads(raw)
        data = msg.get("data", msg)
        if data.get("e") != "kline":
            return
        k = data["k"]
        symbol = data["s"].upper()
        t_ms = int(k["t"])
        o, h, l, c, v = float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"])
        is_closed = bool(k["x"])
        closed = self.builder.update_kline(symbol, t_ms, o, h, l, c, v, is_closed)
        self._last_price[symbol] = c
        self.price_log.info(f"{symbol} price={c} closed={is_closed}")
        if closed:
            self.candle_log.info(f"{symbol} CLOSE t={t_ms} o={o} h={h} l={l} c={c} v={v}")
        else:
            self.candle_log.info(f"{symbol} PARTIAL c={c} h={h} l={l}")
