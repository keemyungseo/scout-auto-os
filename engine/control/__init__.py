"""SCOUT Command Center — safety control API."""

from scout_auto_os.engine.control.control_api import ControlService, create_control_app
from scout_auto_os.engine.control.dashboard import register_dashboard_routes
from scout_auto_os.engine.control.safety_guard import SafetyGuard

__all__ = ["ControlService", "SafetyGuard", "create_control_app", "register_dashboard_routes"]
