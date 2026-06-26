"""SCOUT Command Center — Security Layer V1."""

from scout_auto_os.engine.control.security.auth import AuthManager
from scout_auto_os.engine.control.security.install import register_security

__all__ = ["AuthManager", "register_security"]
