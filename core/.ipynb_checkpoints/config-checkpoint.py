# TODO: Copy from 'Core Configuration Module' artifact
# File: core/config.py

# Paste the code below this line:

"""
Core configuration management for the crypto data platform
Place in: crypto-dashboard/core/config.py
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from enum import Enum


class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class RedisConfig(BaseSettings):
    """Redis configuration"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Stream settings
    stream_prefix: str = "crypto"
    max_stream_length: int = 100000  # Max messages per stream
    consumer_group_prefix: str = "processors"
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"


class DuckDBConfig(BaseSettings):
    """DuckDB configuration"""
    path: Path = Field(default="crypto_data.duckdb", env="DUCKDB_PATH")
    read_only: bool = Field(default=False, env="DUCKDB_READ_ONLY")
    
    # Performance settings
    memory_limit: str = "4GB"
    threads: int = 4
    
    # Retention settings (days)
    retention_1m: int = 30      # 1 month for 1-minute data
    retention_5m: int = 30      # 1 month for 5-minute data
    retention_15m: int = 365    # 1 year for 15-minute data
    # No retention for larger intervals (keep forever)
    
    @validator('path', pre=True)
    def ensure_path(cls, v):
        return Path(v)
    
    class Config:
        env_prefix = "DUCKDB_"


class BinanceConfig(BaseSettings):
    """Binance API configuration"""
    futures_base_url: str = "https://fapi.binance.com"
    spot_base_url: str = "https://api.binance.com"
    ws_futures_url: str = "wss://fstream.binance.com"
    ws_spot_url: str = "wss://stream.binance.com:9443"
    
    # API limits
    weight_limit: int = 2400  # Per minute
    order_limit: int = 1200   # Per minute
    
    # Request settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    class Config:
        env_prefix = "BINANCE_"


class CollectorConfig(BaseSettings):
    """Data collector configuration"""
    # WebSocket settings
    ws_reconnect_interval: int = 5
    ws_heartbeat_interval: int = 30
    ws_max_subscriptions: int = 200  # Max streams per connection
    
    # REST settings
    rest_batch_size: int = 1000      # Candles per request
    rest_max_lookback_days: int = 90 # Max historical data per request
    rest_parallel_requests: int = 10  # Concurrent requests
    
    # Intervals to collect (standard Binance intervals)
    intervals: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    
    # Symbols to track (can be overridden)
    symbols: List[str] = Field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"
    ])
    
    class Config:
        env_prefix = "COLLECTOR_"


class ProcessorConfig(BaseSettings):
    """Data processor configuration"""
    # Batch processing
    batch_size: int = 1000
    batch_timeout: float = 5.0  # seconds
    
    # Validation
    max_price_change_pct: float = 50.0  # Max % change to accept
    min_volume: float = 0.0             # Min volume to accept
    
    # Aggregation
    aggregation_intervals: Dict[str, List[str]] = {
        "5m": ["1m"],           # 5m from 1m candles
        "15m": ["5m"],          # 15m from 5m candles
        "1h": ["15m"],          # 1h from 15m candles
        "4h": ["1h"],           # 4h from 1h candles
        "1d": ["4h", "1h"],     # 1d from 4h or 1h candles
        "1w": ["1d"]            # 1w from 1d candles
    }
    
    class Config:
        env_prefix = "PROCESSOR_"


class CeleryConfig(BaseSettings):
    """Celery configuration"""
    broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Task settings
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = ["json"]
    timezone: str = "UTC"
    enable_utc: bool = True
    
    # Performance
    worker_prefetch_multiplier: int = 1
    worker_max_tasks_per_child: int = 1000
    
    # Beat schedule update interval
    beat_max_loop_interval: int = 5
    
    class Config:
        env_prefix = "CELERY_"


class Settings(BaseSettings):
    """Main application settings"""
    
    # Pydantic v2 configuration - MUST BE FIRST!
    model_config = {
        'extra': 'allow',  # This allows extra fields
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False
    }
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Dashboard specific (add these to stop validation errors)
    use_new_infrastructure: bool = Field(default=True)
    streamlit_server_port: int = Field(default=8501)
    streamlit_server_headless: bool = Field(default=True)
    
    # Component configs
    redis: RedisConfig = Field(default_factory=RedisConfig)
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    
    # Feature flags
    enable_websocket: bool = Field(default=True, env="ENABLE_WEBSOCKET")
    enable_rest_collector: bool = Field(default=True, env="ENABLE_REST_COLLECTOR")
    enable_data_processor: bool = Field(default=True, env="ENABLE_DATA_PROCESSOR")


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)"""
    # Pydantic v2 configuration
    model_config = {
        'extra': 'allow',  # Allow extra fields from .env
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False
    }

    # Dashboard-specific settings (optional)
    use_new_infrastructure: bool = Field(default=True)
    streamlit_server_port: int = Field(default=8501)
    streamlit_server_headless: bool = Field(default=True)

    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset settings (useful for testing)"""
    global _settings
    _settings = None
