"""Streamlit dashboard for Scout Auto OS — pre-deploy operator panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.bot_control import BotControl
from scout_auto_os.engine.manual_override import ManualOverride
from scout_auto_os.engine.report_manager import ReportManager
from scout_auto_os.storage.db import Database, now_kst

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_resource
def load_db() -> tuple[Database, dict, Path]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    sqlite = Path(config["storage"]["sqlite_path"])
    if not sqlite.is_absolute():
        sqlite = ROOT / sqlite
    csv_dir = Path(config["storage"]["csv_dir"])
    if not csv_dir.is_absolute():
        csv_dir = ROOT / csv_dir
    return Database(sqlite), config, csv_dir


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _render_slot(label: str, slot: dict, config: dict, db: Database, csv_dir: Path) -> None:
    st.markdown(f"### {label}")
    if slot.get("state") == "EMPTY":
        st.info("EMPTY — waiting for next A6 candidate")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Symbol", slot.get("symbol", ""))
    c2.metric("PnL %", f"{slot.get('unrealized_pnl_pct', 0) or 0:.2f}")
    c3.metric("expected_ev", f"{slot.get('expected_ev', 0) or 0:.2f}%")
    st.write({
        "entry_price": slot.get("entry_price"),
        "current_price": slot.get("current_price"),
        "exit_plan": slot.get("exit_plan"),
        "exit_reason": slot.get("exit_reason") or "(open)",
        "source": slot.get("source"),
        "manual_lock": slot.get("manual_lock"),
    })
    if slot.get("source") == "AUTO" and not slot.get("manual_lock"):
        if st.button(f"Force Close {label}", key=f"close_{label}"):
            manual = ManualOverride(config, db, csv_dir, DATA_DIR)
            manual.queue_force_close(slot.get("symbol", ""), slot.get("position_id", ""))
            st.warning(f"Queued force_close for {slot.get('symbol')}")


@st.fragment(run_every=5)
def operator_panel(db: Database, config: dict, csv_dir: Path) -> None:
    api = _load_json(DATA_DIR / "execution_api.json")
    status = _load_json(DATA_DIR / "engine_status.json")
    bot = api.get("bot", {})
    bot_state = status.get("bot_state") or status.get("state", "unknown")

    st.subheader("Bot Control")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bot Status", bot_state.upper())
    c2.metric("Mode", "PAPER" if bot.get("paper_mode", True) else "REAL")
    c3.metric("Kill Switch", "ON" if bot.get("kill_switch") else "OFF")
    c4.metric("New Entries", "ALLOWED" if bot.get("new_entries_allowed", True) else "BLOCKED")

    bc = BotControl(DATA_DIR / "bot_control.json")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Bot Stop (block new entries)", type="primary"):
            bc.request_bot_stop("dashboard")
            st.error("New entries blocked — position monitoring continues")
    with b2:
        if st.button("Enable Kill Switch"):
            bc.set_kill_switch(True, "dashboard")
            st.warning("Kill switch ON")
    with b3:
        if st.button("Disable Kill Switch"):
            bc.set_kill_switch(False, "dashboard")
            st.success("Kill switch OFF")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        _render_slot("Slot 1", api.get("slot1", {"slot": 1, "state": "EMPTY"}), config, db, csv_dir)
    with col_b:
        _render_slot("Slot 2", api.get("slot2", {"slot": 2, "state": "EMPTY"}), config, db, csv_dir)

    st.subheader("All Open Positions")
    if api.get("positions"):
        st.dataframe(api["positions"], use_container_width=True)
    else:
        st.info("No open positions")

    if api.get("manual_positions"):
        st.subheader("Manual / Locked (not auto-managed)")
        st.dataframe(api["manual_positions"], use_container_width=True)


@st.fragment(run_every=5)
def live_panel() -> None:
    status = _load_json(DATA_DIR / "engine_status.json")
    metrics = _load_json(DATA_DIR / "live_metrics.json")
    live = status.get("live", metrics.get("live", {}))
    if live.get("connected"):
        st.success("Live Data: Connected")
    else:
        st.error("Live Data: Disconnected")
    st.caption(f"Last metrics: {metrics.get('timestamp', 'n/a')}")


def main() -> None:
    st.set_page_config(page_title="Scout Auto OS", layout="wide")
    st.title("Scout Auto OS — Operator Dashboard")
    db, config, csv_dir = load_db()
    reports = ReportManager(config, db, csv_dir)

    operator_panel(db, config, csv_dir)
    live_panel()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Engine Timeline")
        last_scan = db.fetchone("SELECT MAX(timestamp) AS t FROM candidates")
        st.write({
            "last_scan": _load_json(DATA_DIR / "engine_status.json").get("last_scan") or (last_scan or {}).get("t"),
            "last_update": _load_json(DATA_DIR / "engine_status.json").get("last_update", now_kst()),
        })
    with col2:
        st.subheader("Account Summary")
        st.json(reports.account_summary())

    st.subheader("Recent Exits (exit_reason)")
    trades = db.fetchall(
        "SELECT timestamp, symbol, action, price, pnl_pct, reason FROM trades "
        "WHERE action='EXIT' ORDER BY timestamp DESC LIMIT 20"
    )
    st.dataframe(trades, use_container_width=True)

    st.subheader("TOP5 Latest Scan")
    latest = db.fetchone("SELECT MAX(timestamp) AS t FROM candidates")
    if latest and latest.get("t"):
        top5 = db.fetchall(
            "SELECT rank, symbol, a6_score, expected_ev, entry_price FROM candidates "
            "WHERE timestamp=? ORDER BY rank",
            (latest["t"],),
        )
        st.dataframe(top5, use_container_width=True)


if __name__ == "__main__":
    main()
