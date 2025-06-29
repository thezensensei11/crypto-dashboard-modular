"""
Celery-based task scheduling
"""

from .celery_app import app
from .tasks import (
    detect_and_fill_gaps,
    aggregate_candles,
    cleanup_old_data,
    optimize_database,
    collect_historical_data
)

__all__ = [
    'app',
    'detect_and_fill_gaps',
    'aggregate_candles', 
    'cleanup_old_data',
    'optimize_database',
    'collect_historical_data'
]
