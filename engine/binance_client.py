"""Binance Futures signed REST client (R015)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class OrderResult:
    ok: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    price: float = 0.0
    qty: float = 0.0
    status: str = ""
    error: str = ""
    raw: dict | None = None


@dataclass
class ExchangePosition:
    symbol: str
    side: str
    qty: float
    entry_price: float
    unrealized_pnl: float
    leverage: int


class BinanceClient:
    def __init__(self, rest_base: str = "https://fapi.binance.com", timeout: float = 15.0) -> None:
        self.rest_base = rest_base.rstrip("/")
        self.timeout = timeout
        self.api_key = os.environ.get("BINANCE_API_KEY", "").strip()
        self.api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
        self._step_cache: dict[str, tuple[float, float, float]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _sign(self, params: dict) -> str:
        query = urllib.parse.urlencode(params)
        sig = hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"{query}&signature={sig}"

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        signed: bool = False,
    ) -> dict | list:
        params = dict(params or {})
        url = f"{self.rest_base}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        body = None

        if signed:
            if not self.configured:
                raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET not set")
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            body = self._sign(params).encode()
            url = f"{url}?{body.decode()}"
            body = None
        elif params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode()
            try:
                payload = json.loads(err_body)
                msg = payload.get("msg", err_body)
            except json.JSONDecodeError:
                msg = err_body
            raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc

    def _lot_filters(self, symbol: str) -> tuple[float, float, float]:
        sym = symbol.upper()
        if sym in self._step_cache:
            return self._step_cache[sym]
        info = self._request("GET", "/fapi/v1/exchangeInfo")
        for s in info.get("symbols", []):
            if s["symbol"] == sym:
                step = min_qty = min_notional = 0.0
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                        min_qty = float(f["minQty"])
                    elif f["filterType"] == "MIN_NOTIONAL":
                        min_notional = float(f.get("notional", f.get("minNotional", 0)))
                self._step_cache[sym] = (step, min_qty, min_notional)
                return step, min_qty, min_notional
        raise RuntimeError(f"symbol not found: {sym}")

    @staticmethod
    def _round_step(qty: float, step: float) -> float:
        if step <= 0:
            return qty
        precision = max(0, len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0)
        floored = int(qty / step) * step
        return round(floored, precision)

    def qty_from_usdt(self, symbol: str, usdt: float, price: float) -> float:
        if price <= 0:
            raise RuntimeError("price must be > 0")
        step, min_qty, min_notional = self._lot_filters(symbol)
        raw = usdt / price
        qty = self._round_step(raw, step)
        if qty < min_qty:
            qty = min_qty
        if min_notional > 0 and qty * price < min_notional:
            qty = self._round_step(min_notional / price * 1.01, step)
        return qty

    def market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> OrderResult:
        sym = symbol.upper()
        side = side.upper()
        params: dict = {
            "symbol": sym,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        try:
            data = self._request("POST", "/fapi/v1/order", params, signed=True)
            fill_px = float(data.get("avgPrice") or 0)
            if fill_px <= 0:
                fills = data.get("fills") or []
                if fills:
                    fill_px = float(fills[0].get("price", 0))
            exec_qty = float(data.get("executedQty") or qty)
            return OrderResult(
                ok=True,
                order_id=str(data.get("orderId", "")),
                symbol=sym,
                side=side,
                price=fill_px,
                qty=exec_qty,
                status=data.get("status", "FILLED"),
                raw=data,
            )
        except Exception as exc:
            return OrderResult(ok=False, symbol=sym, side=side, qty=qty, error=str(exc))

    def get_positions(self) -> list[ExchangePosition]:
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        out: list[ExchangePosition] = []
        for p in data:
            amt = float(p.get("positionAmt", 0))
            if abs(amt) < 1e-12:
                continue
            out.append(ExchangePosition(
                symbol=p["symbol"],
                side="LONG" if amt > 0 else "SHORT",
                qty=abs(amt),
                entry_price=float(p.get("entryPrice", 0)),
                unrealized_pnl=float(p.get("unRealizedProfit", 0)),
                leverage=int(float(p.get("leverage", 1))),
            ))
        return out

    def get_position(self, symbol: str) -> ExchangePosition | None:
        for p in self.get_positions():
            if p.symbol == symbol.upper():
                return p
        return None

    def set_leverage(self, symbol: str, leverage: int = 1) -> None:
        self._request(
            "POST", "/fapi/v1/leverage",
            {"symbol": symbol.upper(), "leverage": leverage},
            signed=True,
        )

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> None:
        try:
            self._request(
                "POST", "/fapi/v1/marginType",
                {"symbol": symbol.upper(), "marginType": margin_type},
                signed=True,
            )
        except RuntimeError as exc:
            if "No need to change margin type" not in str(exc):
                raise
