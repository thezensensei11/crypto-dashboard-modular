"""
Enhanced DuckDB Collector that fetches from Binance when data is missing or stale
This replaces the basic compatibility layer with actual data collection logic
Place in: crypto-dashboard-modular/infrastructure/database/enhanced_duckdb_collector.py
"""

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
import logging
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from infrastructure.database.duckdb_manager import get_db_manager
from crypto_dashboard_modular.config.constants import (
    BINANCE_FUTURES_BASE_URL,
    DEFAULT_SYMBOLS,
    INTERVALS
)

logger = logging.getLogger(__name__)


class EnhancedDuckDBCollector:
    """
    Smart collector that:
    1. Checks DuckDB first
    2. Fetches from Binance if data is missing or stale
    3. Stores fetched data in DuckDB
    4. Returns the combined data
    """
    
    def __init__(self):
        self.use_async = use_async  # Ignored, but kept for compatibility
        self.db_manager = get_db_manager(read_only=False)
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.logger = logger
        self.base_url = BINANCE_FUTURES_BASE_URL
        self.session = requests.Session()
        
        # For compatibility with existing code
        
        # Staleness thresholds (in minutes)
        self.staleness_threshold = {
            '1m': 2, '5m': 10, '15m': 30, '30m': 60,
            '1h': 120, '4h': 480, '1d': 1440
        }
    
    def _fetch_from_binance(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int
    ) -> pd.DataFrame:
        """Fetch data from Binance API"""
        url = f"{self.base_url}/klines"
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1000
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                self.api_call_count += 1
                
                data = response.json()
                
                if not data:
                    break
                
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
                
                # Convert price and volume columns to float
                for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                           'taker_buy_base', 'taker_buy_quote']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['trades'] = pd.to_numeric(df['trades'], errors='coerce').astype('Int64')
                
                # Drop the ignore column
                df = df.drop('ignore', axis=1)
                
                all_data.append(df)
                
                # Move to next batch
                if len(df) > 0:
                    last_timestamp = df['timestamp'].max()
                    current_start = int((last_timestamp + timedelta(minutes=1)).timestamp() * 1000)
                else:
                    break
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching from Binance: {e}")
                break
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
            combined = combined.sort_values('timestamp')
            return combined
        
        return pd.DataFrame()
    
    def _is_data_stale(self, latest_timestamp: datetime, interval: str) -> bool:
        """Check if data is stale based on interval"""
        if latest_timestamp is None:
            return True
            
        current_time = datetime.now(timezone.utc)
        age_minutes = (current_time - latest_timestamp).total_seconds() / 60
        
        threshold = self.staleness_threshold.get(interval, 60)
        return age_minutes > threshold
    
    def get_price_data(
        self,
        symbol: str,
        interval: str = '1h',
        lookback_days: int = 30,
        force_cache: bool = False
    ) -> pd.DataFrame:
        """
        Get price data - checks DuckDB first, fetches from Binance if needed
        
        Args:
            symbol: Trading symbol
            interval: Candle interval
            lookback_days: Days of historical data
            force_cache: If True, only use cached data (no API calls)
        """
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=lookback_days)
            
            # First, get what we have in DuckDB
            cached_data = self.db_manager.get_ohlcv(symbol, interval, start_date, end_date)
            
            if force_cache:
                # Only return cached data
                if not cached_data.empty:
                    self.cache_hit_count += 1
                return cached_data
            
            # Check if we need to fetch new data
            need_fetch = False
            fetch_start = None
            fetch_end = None
            
            if cached_data.empty:
                # No data at all - fetch entire range
                need_fetch = True
                fetch_start = start_date
                fetch_end = end_date
                logger.info(f"No cached data for {symbol} {interval}, fetching {lookback_days} days")
            else:
                # Check if data is stale
                latest_timestamp = cached_data['timestamp'].max()
                
                if self._is_data_stale(latest_timestamp, interval):
                    # Fetch from latest timestamp to now
                    need_fetch = True
                    fetch_start = latest_timestamp + timedelta(minutes=1)
                    fetch_end = end_date
                    logger.info(f"Stale data for {symbol} {interval}, updating from {fetch_start}")
                else:
                    # Data is fresh enough
                    self.cache_hit_count += 1
                    logger.debug(f"Using cached data for {symbol} {interval}")
            
            # Fetch new data if needed
            if need_fetch and fetch_start and fetch_end:
                # Convert to milliseconds for Binance API
                start_ms = int(fetch_start.timestamp() * 1000)
                end_ms = int(fetch_end.timestamp() * 1000)
                
                # Fetch from Binance
                new_data = self._fetch_from_binance(symbol, interval, start_ms, end_ms)
                
                if not new_data.empty:
                    # Store in DuckDB
                    rows_inserted = self.db_manager.insert_ohlcv_batch(new_data, symbol, interval)
                    logger.info(f"Inserted {rows_inserted} new rows for {symbol} {interval}")
                    
                    # Combine with cached data
                    if not cached_data.empty:
                        # Merge cached and new data
                        all_data = pd.concat([cached_data, new_data], ignore_index=True)
                        all_data = all_data.drop_duplicates(subset=['timestamp'], keep='last')
                        all_data = all_data.sort_values('timestamp')
                        
                        # Filter to requested date range
                        all_data = all_data[
                            (all_data['timestamp'] >= start_date) & 
                            (all_data['timestamp'] <= end_date)
                        ]
                        
                        return all_data
                    else:
                        return new_data
            
            return cached_data
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_price_data_batch(
        self,
        symbols: List[str],
        interval: str = '1h',
        lookback_days: int = 30,
        force_cache: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Get price data for multiple symbols"""
        result = {}
        
        for symbol in symbols:
            result[symbol] = self.get_price_data(symbol, interval, lookback_days, force_cache)
            
        return result
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols from database"""
        try:
            with self.db_manager.get_connection() as conn:
                result = conn.execute(
                    "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
                ).fetchall()
                symbols = [row[0] for row in result]
                
                # If no symbols in DB, return default symbols
                if not symbols:
                    return DEFAULT_SYMBOLS
                    
                return symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return DEFAULT_SYMBOLS
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        try:
            with self.db_manager.get_connection() as conn:
                # Get symbol count
                symbol_count = conn.execute(
                    "SELECT COUNT(DISTINCT symbol) FROM ohlcv"
                ).fetchone()[0]
                
                # Get total rows
                total_rows = conn.execute(
                    "SELECT COUNT(*) FROM ohlcv"
                ).fetchone()[0]
                
                # Get database size
                db_path = self.db_manager.db_path
                size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
                
                return {
                    'total_symbols': symbol_count,
                    'total_files': symbol_count,  # For compatibility
                    'total_rows': total_rows,
                    'total_size_mb': size_mb,
                    'api_calls': self.api_call_count,
                    'cache_hits': self.cache_hit_count
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'total_symbols': 0,
                'total_files': 0,
                'total_rows': 0,
                'total_size_mb': 0,
                'api_calls': self.api_call_count,
                'cache_hits': self.cache_hit_count
            }
    
    def reset_counters(self):
        """Reset API call and cache hit counters"""
        self.api_call_count = 0
        self.cache_hit_count = 0
    
    # Additional methods for compatibility
    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get funding rate from Binance API"""
        try:
            url = f"{self.base_url}/premiumIndex"
            response = self.session.get(url, params={'symbol': symbol})
            response.raise_for_status()
            self.api_call_count += 1
            
            data = response.json()
            return float(data.get('lastFundingRate', 0))
            
        except Exception as e:
            logger.error(f"Error getting funding rate for {symbol}: {e}")
            return None
    
    def get_ticker_24hr(self, symbol: str) -> Optional[Dict]:
        """Get 24hr ticker data from Binance API"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            response = self.session.get(url, params={'symbol': symbol})
            response.raise_for_status()
            self.api_call_count += 1
            
            data = response.json()
            return {
                'symbol': data['symbol'],
                'lastPrice': float(data['lastPrice']),
                'priceChange': float(data['priceChange']),
                'priceChangePercent': float(data['priceChangePercent']),
                'volume': float(data['volume']),
                'highPrice': float(data['highPrice']),
                'lowPrice': float(data['lowPrice']),
            }
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None


# For easy import in compatibility layer
DuckDBCompatibleCollector = EnhancedDuckDBCollector