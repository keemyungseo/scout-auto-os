"""A6 Reverse Short — shadow placeholder only."""

from __future__ import annotations

from scout_auto_os.storage.db import Database


class ScoutReverseShadow:
    """No short entries in R013. Logs shadow observations only."""

    def __init__(self, config: dict, db: Database) -> None:
        self.config = config
        self.db = db
        self.enabled = bool(config["short_engine"].get("enabled", False))
        self.shadow_mode = bool(config["short_engine"].get("shadow_mode", True))

    def observe(self, top5: list[dict]) -> None:
        if not self.shadow_mode:
            return
        if self.enabled:
            self.db.log_event("scout_reverse_shadow", "blocked_enabled_without_live", {})
            return
        for row in top5[:3]:
            self.db.log_event(
                "scout_reverse_shadow",
                "shadow_candidate",
                {"symbol": row["symbol"], "a6_score": row["a6_score"], "note": "short_disabled"},
            )

    def try_entry(self, *args, **kwargs) -> None:
        if self.enabled:
            raise RuntimeError("Short engine live entry forbidden in R013")
        return None
