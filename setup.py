"""
Legacy setup.py for backward compatibility
Modern Python projects should use pyproject.toml instead
"""

import warnings
from setuptools import setup

warnings.warn(
    "setup.py is deprecated. Please use 'pip install -e .' with pyproject.toml instead.",
    DeprecationWarning,
    stacklevel=2
)

# Delegate to pyproject.toml
setup()