"""Research Engine settings — env + runtime config resolution."""

from __future__ import annotations

import os


def _env_truthy(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on", "enabled")


def research_enabled(config: dict | None = None) -> bool:
    """
    Resolve research.enabled.

    Priority: RESEARCH_ENABLED env (container .env) > config.research.enabled > False.
    """
    env_val = _env_truthy("RESEARCH_ENABLED")
    if env_val is not None:
        return env_val
    if config:
        return bool(config.get("research", {}).get("enabled", False))
    return False


def research_int_env(name: str, config_key: str, config: dict, default: int) -> int:
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip():
        try:
            return int(raw)
        except ValueError:
            pass
    rcfg = config.get("research", {})
    if config_key in rcfg and rcfg[config_key] is not None:
        return int(rcfg[config_key])
    return default
