"""
Fixed DuckDB Collector that handles column mismatches
Place this file in: crypto-dashboard-modular/data/duckdb_collector.py
"""

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import time
import sys
import os

# Fix import paths
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from infrastructure.database.duckdb_manager import get_db_manager

logger = logging.getLogger(__name__)

# Columns expected by DuckDB (13 total)
DUCKDB_SCHEMA_COLUMNS = [
    'symbol', 'interval', 'timestamp', 'open', 'high', 'low', 'close',
    'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
    'created_at'
]

# Columns returned by Binance API (12 total, excluding symbol/interval)
BINANCE_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
]


class BinanceDataCollector:
    """
    DuckDB-compatible Binance data collector
    Handles column mismatches between Binance API and DuckDB schema
    """
    
    BASE_URL = "https://fapi.binance.com/fapi/v1"
    MIN_DATA_DAYS = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    def __init__(self):
        self.logger = logger
        self.db_manager = get_db_manager(read_only=False)
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Rate limiting
        
        # For compatibility with existing code
        self._sync_collector = self
        self._async_collector = None
    
    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_price_data(
        self, 
        symbol: str, 
        interval: str = '1h', 
        lookback_days: int = 30, 
        force_cache: bool = False
    ) -> pd.DataFrame:
        """Get price data with proper error handling and caching"""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=max(lookback_days, self.MIN_DATA_DAYS))
            
            # Check cache first if not forcing refresh
            if not force_cache and self.db_manager:
                try:
                    cached_data = self.db_manager.get_ohlcv(symbol, interval, start_date, end_date)
                    if not cached_data.empty:
                        latest = cached_data['timestamp'].max()
                        # Handle timezone comparison properly
                        if pd.Timestamp(latest).tz_localize(None) > pd.Timestamp(end_date).tz_localize(None) - timedelta(hours=2):
                            self.cache_hit_count += 1
                            self.logger.info(f"Cache hit for {symbol} {interval}")
                            return self._ensure_min_data(cached_data, symbol, interval, lookback_days)
                except Exception as e:
                    self.logger.warning(f"Cache check failed: {e}")
            
            # Fetch from API
            self.logger.info(f"Fetching {symbol} {interval} from API")
            data = self._fetch_from_binance(symbol, interval, start_date, end_date)
            
            if data.empty:
                self.logger.warning(f"No data fetched for {symbol}")
                # Return cached data if available
                try:
                    return self.db_manager.get_ohlcv(symbol, interval, start_date, end_date)
                except:
                    return pd.DataFrame()
            
            # Save to database
            self._save_to_db(data, symbol, interval)
            
            return self._ensure_min_data(data, symbol, interval, lookback_days)
            
        except Exception as e:
            self.logger.error(f"Error getting price data for {symbol}: {e}")
            # Try to return cached data on error
            try:
                cached = self.db_manager.get_ohlcv(symbol, interval, start_date, end_date)
                if not cached.empty:
                    return cached
            except:
                pass
            return pd.DataFrame()
    
    def _fetch_from_binance(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data from Binance API with retry logic"""
        endpoint = f"{self.BASE_URL}/klines"
        
        # Convert to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        all_data = []
        current_start = start_ms
        
        while current_start < end_ms:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000  # Max allowed by Binance
            }
            
            # Retry logic
            for attempt in range(self.MAX_RETRIES):
                try:
                    self._rate_limit()
                    response = requests.get(endpoint, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    self.api_call_count += 1
                    break
                except Exception as e:
                    if attempt == self.MAX_RETRIES - 1:
                        self.logger.error(f"Failed to fetch {symbol} after {self.MAX_RETRIES} attempts: {e}")
                        return pd.DataFrame()
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
            
            if not data:
                break
            
            df = pd.DataFrame(data, columns=BINANCE_COLUMNS)
            all_data.append(df)
            
            # If we got less than limit, we've reached the end
            if len(data) < 1000:
                break
                
            # Move to next batch
            current_start = int(data[-1][0]) + 1
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        
        # Convert timestamp to datetime
        result['timestamp'] = pd.to_datetime(result['timestamp'], unit='ms', utc=True)
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        
        # Handle trades column
        if 'trades' in result.columns:
            result['trades'] = pd.to_numeric(result['trades'], errors='coerce').fillna(0).astype('int64')
        
        # Handle taker columns - these might be very large numbers
        for col in ['taker_buy_base', 'taker_buy_quote']:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        
        # Drop unnecessary columns
        columns_to_keep = [col for col in result.columns if col not in ['close_time', 'ignore']]
        result = result[columns_to_keep]
        
        return result
    
    def _save_to_db(self, data: pd.DataFrame, symbol: str, interval: str):
        """Save data to DuckDB with proper column handling"""
        try:
            if data.empty:
                return
            
            # The DuckDBManager will handle adding symbol, interval, and created_at
            # Just ensure we have the right columns from Binance
            save_data = data.copy()
            
            # Log what we're saving
            self.logger.info(f"Saving {len(save_data)} rows for {symbol} {interval}")
            self.logger.debug(f"Columns before save: {list(save_data.columns)}")
            
            # Let DuckDBManager handle the column mapping
            saved_count = self.db_manager.insert_ohlcv_batch(save_data, symbol, interval)
            
            self.logger.info(f"Successfully saved {saved_count} rows to DuckDB")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to DuckDB: {e}")
            self.logger.error(f"Data shape: {data.shape}, columns: {list(data.columns)}")
            raise
    
    def _ensure_min_data(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        interval: str, 
        lookback_days: int
    ) -> pd.DataFrame:
        """Ensure we return at least MIN_DATA_DAYS of data"""
        if data.empty:
            return data
        
        min_days = max(lookback_days, self.MIN_DATA_DAYS)
        min_date = datetime.now(timezone.utc) - timedelta(days=min_days)
        
        # Handle timezone for comparison
        data_min_timestamp = data['timestamp'].min()
        if hasattr(data_min_timestamp, 'tz') and data_min_timestamp.tz is not None:
            compare_timestamp = data_min_timestamp
        else:
            compare_timestamp = pd.Timestamp(data_min_timestamp).tz_localize('UTC')
        
        # If we have enough data, return it
        if compare_timestamp <= pd.Timestamp(min_date):
            return data
        
        # Otherwise, try to get more from the database
        try:
            full_data = self.db_manager.get_ohlcv(
                symbol, 
                interval, 
                min_date, 
                datetime.now(timezone.utc)
            )
            if not full_data.empty:
                return full_data
        except:
            pass
        
        return data
    
    def get_price_data_batch(
        self, 
        symbols: List[str], 
        interval: str = '1h',
        lookback_days: int = 30, 
        force_cache: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols"""
        result = {}
        for symbol in symbols:
            try:
                data = self.get_price_data(symbol, interval, lookback_days, force_cache)
                if not data.empty:
                    result[symbol] = data
            except Exception as e:
                self.logger.error(f"Failed to get data for {symbol}: {e}")
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "total_symbols": 0,
            "total_rows": 0,
            "total_size_mb": 0,
            "api_calls": self.api_call_count,
            "cache_hits": self.cache_hit_count
        }
        
        try:
            if self.db_manager:
                with self.db_manager.get_connection() as conn:
                    # Get row count and symbol count
                    result = conn.execute("""
                        SELECT COUNT(*) as row_count, 
                               COUNT(DISTINCT symbol) as symbol_count 
                        FROM ohlcv
                    """).fetchone()
                    
                    if result:
                        stats['total_rows'] = result[0]
                        stats['total_symbols'] = result[1]
                    
                    # Get database file size
                    db_path = Path("crypto_data.duckdb")
                    if db_path.exists():
                        stats['total_size_mb'] = db_path.stat().st_size / (1024 * 1024)
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
        
        return stats
    
    def reset_counters(self):
        """Reset API call and cache hit counters"""
        self.api_call_count = 0
        self.cache_hit_count = 0
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache - not implemented for DuckDB"""
        self.logger.warning("Cache clearing not implemented for DuckDB backend")
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from database or return defaults"""
        try:
            symbols = self.db_manager.get_symbols()
            if symbols:
                return symbols
        except:
            pass
        
        # Return default symbols if database is empty
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
    
    # Alias methods for compatibility
    def get_klines(self, **kwargs):
        """Alias for get_price_data for compatibility"""
        return self.get_price_data(**kwargs)