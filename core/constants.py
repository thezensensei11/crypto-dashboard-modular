

"""
System constants
"""

# Time constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7

# Data quality thresholds
MAX_PRICE_SPIKE_PCT = 50.0  # Maximum acceptable price change %
MIN_VOLUME_THRESHOLD = 0.0  # Minimum volume to consider valid
MAX_TIME_GAP_MINUTES = 5    # Maximum gap before considering data missing

# Batch processing
DEFAULT_BATCH_SIZE = 1000
MAX_BATCH_SIZE = 10000
BATCH_TIMEOUT_SECONDS = 5.0

# Redis keys
REDIS_STREAM_PATTERN = "{prefix}:stream:{event_type}"
REDIS_CONSUMER_GROUP_PATTERN = "{prefix}:group:{component}"
REDIS_STATS_KEY_PATTERN = "{prefix}:stats:{component}:{stat_type}"

# Database
DB_SCHEMA_VERSION = 1
OHLCV_TABLE_PREFIX = "ohlcv_"

