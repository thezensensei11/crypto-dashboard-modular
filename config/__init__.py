"""Configuration module for crypto dashboard"""

from .settings import Settings, get_settings
from .constants import (
    # API Configuration
    BINANCE_BASE_URL,
    HYPERLIQUID_API_URL,
    MAX_REQUESTS_PER_MINUTE,
    MIN_REQUEST_INTERVAL,
    
    # Cache Configuration
    CACHE_DIR,
    MIN_CACHE_DAYS,
    API_DELAY_BUFFER_MINUTES,
    
    # UI Configuration
    PAGE_TITLE,
    PAGE_ICON,
    
    # Available options
    INTERVALS,
    DEFAULT_INTERVAL,
    METRIC_TYPES,
    POPULAR_SYMBOLS,
    BASE_COLUMNS,
    
    # Display thresholds
    PRICE_THRESHOLDS,
    
    # Theme colors
    COLORS
)

__all__ = [
    # Settings
    'Settings',
    'get_settings',
    
    # API Constants
    'BINANCE_BASE_URL',
    'HYPERLIQUID_API_URL',
    'MAX_REQUESTS_PER_MINUTE',
    'MIN_REQUEST_INTERVAL',
    
    # Cache Constants
    'CACHE_DIR',
    'MIN_CACHE_DAYS',
    'API_DELAY_BUFFER_MINUTES',
    
    # UI Constants
    'PAGE_TITLE',
    'PAGE_ICON',
    'INTERVALS',
    'DEFAULT_INTERVAL',
    'METRIC_TYPES',
    'POPULAR_SYMBOLS',
    'BASE_COLUMNS',
    'PRICE_THRESHOLDS',
    'COLORS'
]