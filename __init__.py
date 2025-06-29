"""
Crypto Dashboard Modular - A professional crypto metrics dashboard
"""

__version__ = "2.0.0"
__author__ = "Your Name"

# Make the package namespace available
from . import config
from . import data
from . import metrics
from . import ui
from . import utils

__all__ = [
    "config",
    "data", 
    "metrics",
    "ui",
    "utils",
    "__version__",
]