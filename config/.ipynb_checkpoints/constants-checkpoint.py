"""
Application constants and configuration values
"""

# API Configuration
BINANCE_BASE_URL = "https://fapi.binance.com"
HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz/info"

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 2000
MIN_REQUEST_INTERVAL = 0.03  # 30ms between requests

# Cache Configuration  
CACHE_DIR = "smart_cache"
MIN_CACHE_DAYS = 30
API_DELAY_BUFFER_MINUTES = 5

# UI Configuration
PAGE_TITLE = "Altcoin Dashboard"
PAGE_ICON = "📊"

# Available intervals
INTERVALS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
DEFAULT_INTERVAL = '1h'

# Available metrics
METRIC_TYPES = [
    'beta',
    'upside_beta', 
    'downside_beta',
    'correlation',
    'volatility',
    'sharpe_ratio',
    'sortino_ratio'
]

# Popular symbols for quick add
POPULAR_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

# Column display configuration
BASE_COLUMNS = ['Symbol', 'Price', '24h Return %', '24h Volume Change %']

# Price display thresholds
PRICE_THRESHOLDS = {
    'high_precision': 0.01,    # Below this, use 8 decimals
    'medium_precision': 1.0,   # Below this, use 5 decimals  
    'low_precision': 1000.0    # Above this, use 2 decimals
}

# Theme colors
COLORS = {
    'primary': '#2de19a',       # Hyperliquid green
    'primary_dark': '#26c987',
    'positive': '#26c987',      # Green for positive values
    'negative': '#ff4b4b',      # Red for negative values
    'background': '#0a0a0a',
    'background_light': '#1a1a1a',
    'border': '#2a2a2a',
    'text': '#e0e0e0',
    'text_muted': '#888'
}
