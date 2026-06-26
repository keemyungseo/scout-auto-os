#!/usr/bin/env python3
"""Run SCOUT Command Center Safety Control API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.control.control_api import ControlService, create_control_app


def _data_root() -> Path:
    """Align with main.py / Docker SCOUT_DATA_DIR volume (/app/data)."""
    env = os.environ.get("SCOUT_DATA_DIR", "").strip()
    if env:
        return Path(env)
    return PKG / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="SCOUT Command Center Control API")
    parser.add_argument(
        "--host",
        default=os.environ.get("CONTROL_API_HOST", "0.0.0.0"),
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("CONTROL_API_PORT", "8787")))
    parser.add_argument(
        "--data-root",
        default=None,
        help="Data root (guardian, runtime_shadow). Default: SCOUT_DATA_DIR or scout_auto_os/data",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Install: pip install fastapi uvicorn")
        sys.exit(1)

    data_root = Path(args.data_root) if args.data_root else _data_root()
    control_dir = data_root / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    bot_control = data_root / "bot_control.json"
    svc = ControlService(control_dir, bot_control_path=bot_control, data_dir=data_root)
    app = create_control_app(svc)
    print(f"[CONTROL API] http://{args.host}:{args.port}/command-center")
    print(f"[CONTROL API] status: http://{args.host}:{args.port}/control/status")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
