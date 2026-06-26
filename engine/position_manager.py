"""Open position lifecycle — State Engine V1.4 + Position Evaluation V1."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from scout_auto_os.engine.emergency_risk_guard import EmergencyRiskGuard
from scout_auto_os.engine.expected_ev_engine import compute_live_ev
from scout_auto_os.engine.position_state_manager import PositionStateManager
from scout_auto_os.storage.db import Database, now_kst


class PositionManager:
    def __init__(
        self,
        config: dict,
        db: Database,
        csv_dir: Path,
        adapter,
        risk_guard: EmergencyRiskGuard | None = None,
        state_manager: PositionStateManager | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.csv_dir = csv_dir
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter
        self.max_long = int(config["position"]["max_long_slots"])
        self.risk_guard = risk_guard or EmergencyRiskGuard(config)
        self.state_manager = state_manager

    def open_positions(self) -> list[dict]:
        return self.db.fetchall(
            "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_time"
        )

    def occupied_symbols(self) -> set[str]:
        rows = self.open_positions()
        return {r["symbol"] for r in rows}

    def long_slots_used(self) -> int:
        return len([p for p in self.open_positions() if p["side"] == "LONG" and p["engine"] == "A6_LONG"])

    def portfolio_long_slots_used(self) -> int:
        return len([p for p in self.open_positions() if p["side"] == "LONG" and p["engine"] == "PORTFOLIO_LONG"])

    def short_slots_used(self) -> int:
        return len([p for p in self.open_positions() if p["side"] == "SHORT" and p["engine"] == "PORTFOLIO_SHORT"])

    def has_slot(self) -> bool:
        return self.long_slots_used() < self.max_long

    def create_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        source: str,
        engine: str,
        a6_score: float = 0.0,
        expected_ev: float = 0.0,
        manual_lock: bool = False,
        auto_manage: bool = True,
        rank: int = 1,
        primary_reason: str = "",
    ) -> str:
        pid = self.db.new_id("pos")
        ts = now_kst()
        thesis_id = ""
        if manual_lock or source.upper() == "MANUAL" or not auto_manage:
            source = "MANUAL"
            auto_manage = False
            manual_lock = True

        pos_row = {
            "position_id": pid,
            "symbol": symbol,
            "side": side,
            "source": source,
            "engine": engine,
            "entry_time": ts,
            "entry_price": entry_price,
            "manual_lock": int(manual_lock),
            "auto_manage": int(auto_manage),
            "a6_score": a6_score,
        }
        if self.state_manager:
            thesis_id = self.state_manager.create_thesis_for_position({
                **pos_row,
                "expected_ev": expected_ev,
            })

        self.db.execute(
            """INSERT INTO positions
            (position_id, symbol, side, source, engine, entry_time, entry_price, current_price,
             unrealized_pnl_pct, status, manual_lock, auto_manage, last_update_time, a6_score,
             expected_ev, exit_plan, thesis_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                pid, symbol, side, source, engine, ts, entry_price, entry_price,
                0.0, "OPEN", int(manual_lock), int(auto_manage), ts, a6_score,
                expected_ev, "position_evaluation_v1", thesis_id,
            ),
        )
        if self.state_manager:
            bars = self.adapter.get_bars(symbol, ts)
            self.state_manager.register_entry(pid, symbol, ts, bars or [], side=side)
        self._sync_csv()
        return pid

    def update_prices(self) -> list[dict]:
        updated: list[dict] = []
        for pos in self.open_positions():
            bars = self.adapter.get_bars(pos["symbol"], pos["entry_time"])
            if bars:
                px = bars[-1].c
            else:
                px = self.adapter.get_price(pos["symbol"])
            if px <= 0:
                continue
            entry = pos["entry_price"]
            upnl = (px - entry) / entry * 100 if pos["side"] == "LONG" else (entry - px) / entry * 100
            ev = compute_live_ev(
                pos["symbol"], bars or [], pos.get("a6_score") or 0, pos["entry_time"],
            )
            alive_line = ""
            thesis_line = ""
            if self.state_manager and bars:
                alive = self.state_manager.update_current(
                    pos["position_id"], pos["symbol"], pos["entry_time"], bars, side=pos["side"],
                )
                if alive:
                    alive_line = f"alive={alive.alive_score} rec={alive.hold_recommendation}"
                thesis = self.state_manager.evaluation.store.get_by_position(pos["position_id"])
                if thesis:
                    thesis_line = f"thesis={thesis.thesis_id[:8]}"
            exit_plan = f"{thesis_line} {alive_line} rem_ev={ev['remaining_ev']:.2f}%".strip()
            self.db.execute(
                """UPDATE positions SET current_price=?, unrealized_pnl_pct=?,
                   last_update_time=?, exit_plan=?, expected_ev=? WHERE position_id=?""",
                (px, upnl, now_kst(), exit_plan, ev["expected_ev"], pos["position_id"]),
            )
            updated.append({
                **pos, "current_price": px, "unrealized_pnl_pct": upnl, **ev,
            })
        self._sync_csv()
        return updated

    def check_exits(
        self,
        execution,
        alert_mgr,
        *,
        allow_auto_exit: bool = True,
        symbol_exit_allowed=None,
    ) -> list[dict]:
        closed: list[dict] = []
        for pos in self.open_positions():
            if pos["manual_lock"] or not pos["auto_manage"] or pos["source"] == "MANUAL":
                continue
            if not allow_auto_exit:
                continue
            sym = pos["symbol"]
            if symbol_exit_allowed is not None and not symbol_exit_allowed(sym):
                continue
            bars = self.adapter.get_bars(pos["symbol"], pos["entry_time"])
            if not bars:
                continue
            exit_px = bars[-1].c
            entry = pos["entry_price"]
            pnl_pct = (exit_px - entry) / entry * 100 if pos["side"] == "LONG" else (entry - exit_px) / entry * 100

            rg = self.risk_guard.evaluate(pos, pnl_pct)
            if rg.should_exit:
                reason = f"risk_guard_{rg.reason}"
                print(
                    f"[RISK GUARD EXIT] symbol={pos['symbol']} roi={rg.roi_pct:.2f} "
                    f"pnl={rg.pnl_usdt:.4f} reason={rg.reason}"
                )
                self._close_position(pos, exit_px, pnl_pct, reason, execution, alert_mgr, "AUTO")
                alert_mgr.risk_guard_exit_alert(pos["symbol"], rg.roi_pct, rg.pnl_usdt, rg.reason)
                closed.append(pos)
                continue

            if self.state_manager:
                decision = self.state_manager.maybe_review(pos, bars, pnl_pct)
                if decision and decision.should_exit:
                    reason = decision.reason
                    print(
                        f"[STATE EXIT] symbol={pos['symbol']} roi={pnl_pct:.2f}% "
                        f"reason={reason} hold={self.state_manager.hold_minutes(pos['entry_time'])}m"
                    )
                    self._close_position(pos, exit_px, pnl_pct, reason, execution, alert_mgr, "AUTO")
                    closed.append(pos)
                continue

        return closed

    def close_by_user(self, position_id: str, execution, alert_mgr, exit_price: float | None = None) -> None:
        pos = self.db.fetchone("SELECT * FROM positions WHERE position_id=?", (position_id,))
        if not pos or pos["status"] != "OPEN":
            return
        px = exit_price or pos["current_price"] or pos["entry_price"]
        entry = pos["entry_price"]
        ret = (px - entry) / entry * 100 if pos["side"] == "LONG" else (entry - px) / entry * 100
        self._close_position(pos, px, ret, "CLOSED_BY_USER", execution, alert_mgr, "USER")

    def _close_position(self, pos, exit_px, pnl_pct, reason, execution, alert_mgr, closed_by: str) -> None:
        execution.paper_exit(
            pos["position_id"], pos["symbol"], pos["side"], exit_px, pnl_pct, reason, pos["engine"],
        )
        status = "CLOSED_BY_USER" if closed_by == "USER" else "CLOSED"
        self.db.execute(
            """UPDATE positions SET status=?, realized_pnl_pct=?, exit_reason=?,
               current_price=?, last_update_time=? WHERE position_id=?""",
            (status, pnl_pct, reason, exit_px, now_kst(), pos["position_id"]),
        )
        if self.state_manager:
            self.state_manager.on_close(pos["position_id"])
        hold_min = 0
        try:
            t0 = datetime.strptime(pos["entry_time"], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(now_kst(), "%Y-%m-%d %H:%M:%S")
            hold_min = int((t1 - t0).total_seconds() / 60)
        except ValueError:
            pass
        alert_mgr.exit_alert(pos["symbol"], exit_px, pnl_pct, hold_min, reason)
        self._sync_csv()

    def _sync_csv(self) -> None:
        rows = self.db.fetchall("SELECT * FROM positions ORDER BY entry_time DESC")
        path = self.csv_dir / "positions.csv"
        if not rows:
            if path.exists():
                path.unlink()
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
