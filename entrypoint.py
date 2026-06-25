"""Scout Live V1 — Docker entrypoint (env → runtime config → main loop)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

APP_ROOT = Path("/app")
PKG = APP_ROOT / "scout_auto_os"
DATA = Path(os.environ.get("SCOUT_DATA_DIR", "/app/data"))
LOGS = Path(os.environ.get("SCOUT_LOG_DIR", "/app/logs"))

for sub in (
    DATA,
    LOGS,
    DATA / "research",
    LOGS / "auto_os",
    LOGS / "live",
    LOGS / "phase19_winner_dna" / "kline_cache",
    LOGS / "universe_research" / "snapshots",
):
    sub.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "research_bundle"))

seed_candidates = APP_ROOT / "research_bundle" / "seed" / "candidates.jsonl"
runtime_candidates = LOGS / "phase19_winner_dna" / "candidates.jsonl"
if seed_candidates.exists() and not runtime_candidates.exists():
    shutil.copy2(seed_candidates, runtime_candidates)

import scout_phase16_human_blind_test as p16
import scout_phase19_winner_ranking_dna as p19

p19.OUT_DIR = LOGS / "phase19_winner_dna"
p19.CACHE_DIR = p19.OUT_DIR / "kline_cache"
p19.CANDIDATES_PATH = p19.OUT_DIR / "candidates.jsonl"
p16.CACHE_DIR = p19.CACHE_DIR

with (PKG / "config.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mode = os.getenv("MODE", "LIVE").strip().upper()
cfg["paper_mode"] = mode != "LIVE"
cfg["mode"] = "live" if mode == "LIVE" else "paper"

if os.getenv("TRADE_SIZE"):
    cfg.setdefault("execution", {})["order_size_usdt"] = float(os.environ["TRADE_SIZE"])
if os.getenv("LEVERAGE"):
    cfg.setdefault("execution", {})["leverage"] = int(os.environ["LEVERAGE"])

report_time = os.getenv("REPORT_TIME", "08:00").strip()
cfg.setdefault("loop", {})["daily_report_hour_kst"] = int(report_time.split(":")[0])

cfg.setdefault("storage", {})["sqlite_path"] = str(DATA / "scout_auto_os.db")
cfg.setdefault("storage", {})["csv_dir"] = str(LOGS / "auto_os")
cfg.setdefault("data", {})["kline_cache_dir"] = str(LOGS / "phase19_winner_dna" / "kline_cache")
cfg.setdefault("live_data", {})["log_dir"] = str(LOGS / "live")

if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
    cfg.setdefault("alerts", {})["telegram"] = True

research_enabled = os.getenv("RESEARCH_ENABLED", "false").strip().lower() in ("1", "true", "yes")
cfg.setdefault("research", {})["enabled"] = research_enabled
if os.getenv("RESEARCH_SCAN_INTERVAL_MIN"):
    cfg.setdefault("research", {})["scan_interval_min"] = int(os.environ["RESEARCH_SCAN_INTERVAL_MIN"])
if os.getenv("RESEARCH_TOP_N"):
    cfg.setdefault("research", {})["top_n"] = int(os.environ["RESEARCH_TOP_N"])

runtime_cfg = DATA / "config.runtime.yaml"
runtime_cfg.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

os.environ.setdefault("SCOUT_UNIVERSE_CACHE_ONLY", "false")

from scout_auto_os.main import main

if __name__ == "__main__":
    sys.argv = ["scout_auto_os.main", "--config", str(runtime_cfg)]
    main()
