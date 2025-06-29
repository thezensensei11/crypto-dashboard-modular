"""
Core modules for crypto dashboard
"""

from .config import get_settings, Settings
from .models import (
    OHLCVData, Event, EventType, Interval, 
    DataSource, BatchData, ValidationResult,
    AggregationRequest, ProcessingStats
)
from .constants import (
    MAX_PRICE_SPIKE_PCT, MIN_VOLUME_THRESHOLD,
    DEFAULT_BATCH_SIZE, BATCH_TIMEOUT_SECONDS
)
from .exceptions import (
    CryptoDataException, CollectorException,
    ValidationException, ProcessingException,
    DatabaseException, MessageBusException
)

__all__ = [
    # Config
    'get_settings', 'Settings',
    # Models
    'OHLCVData', 'Event', 'EventType', 'Interval',
    'DataSource', 'BatchData', 'ValidationResult',
    'AggregationRequest', 'ProcessingStats',
    # Constants
    'MAX_PRICE_SPIKE_PCT', 'MIN_VOLUME_THRESHOLD',
    'DEFAULT_BATCH_SIZE', 'BATCH_TIMEOUT_SECONDS',
    # Exceptions
    'CryptoDataException', 'CollectorException',
    'ValidationException', 'ProcessingException',
    'DatabaseException', 'MessageBusException'
]
