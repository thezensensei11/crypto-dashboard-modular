"""Dashboard tabs"""

from .universe import UniverseTab
from .dashboard import DashboardTab
from .controls import ControlsTab
from .settings import SettingsTab
from .cache_inspector import CacheInspectorTab
from .backtester import BacktesterTab
from .shock import ShockTab

__all__ = [
    'UniverseTab',
    'DashboardTab', 
    'ControlsTab',
    'SettingsTab',
    'CacheInspectorTab',
    'BacktesterTab',
    'ShockTab'
]