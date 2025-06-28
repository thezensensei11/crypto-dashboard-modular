import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Union
import requests
import time
import logging
from smart_data_manager import SmartDataManager

class BinanceFuturesCollector:
    """Enhanced Binance Futures data collector with smart caching and UTC handling"""
    
    # Minimum data to always maintain in cache AND return
    MIN_CACHE_DAYS = 30
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        
        # Create session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3,
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Keep connection alive
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Accept': 'application/json',
        })
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize smart data manager
        self.data_manager = SmartDataManager()
        
        # Rate limiting - Binance allows 2400 requests/min, we'll use 2000 to be safe
        self.request_count = 0
        self.request_reset_time = time.time() + 60
        self.max_requests_per_minute = 2000  # Increased from 1200
        self.min_request_interval = 0.03  # 30ms between requests (allows ~33 requests/sec)
        self.last_request_time = 0
        
        # Track API calls for diagnostics
        self.api_call_count = 0
        self.cache_hit_count = 0
    
    def _get_utc_now(self) -> datetime:
        """Get current time in UTC"""
        return datetime.now(timezone.utc)
    
    def _ensure_utc(self, dt: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime is in UTC"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            # Assume naive datetimes are in local time - convert to UTC
            self.logger.warning(f"Converting naive datetime {dt} to UTC (assuming local time)")
            # Get local timezone offset
            local_offset = datetime.now() - datetime.utcnow()
            return dt - local_offset
        elif dt.tzinfo != timezone.utc:
            return dt.astimezone(timezone.utc)
        return dt
    
    def _rate_limit(self):
        """Smart rate limiting to maximize throughput while staying safe"""
        current_time = time.time()
        
        # Reset counter if minute has passed
        if current_time > self.request_reset_time:
            self.request_count = 0
            self.request_reset_time = current_time + 60
            self.logger.debug(f"Reset rate limit counter, new window until {datetime.fromtimestamp(self.request_reset_time)}")
        
        # If approaching limit, wait for reset
        if self.request_count >= self.max_requests_per_minute - 10:
            wait_time = self.request_reset_time - current_time
            if wait_time > 0:
                self.logger.info(f"Rate limit approaching ({self.request_count}/{self.max_requests_per_minute}), waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.request_count = 0
                self.request_reset_time = time.time() + 60
        
        # Ensure minimum time between requests (prevents bursting)
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.request_count += 1
        self.last_request_time = time.time()
    
    def get_exchange_info(self) -> pd.DataFrame:
        """Get exchange information and trading symbols"""
        self._rate_limit()
        
        endpoint = f"{self.base_url}/fapi/v1/exchangeInfo"
        response = self.session.get(endpoint)
        response.raise_for_status()
        
        data = response.json()
        symbols_data = []
        
        for symbol in data['symbols']:
            if symbol['status'] == 'TRADING' and symbol['contractType'] == 'PERPETUAL':
                symbols_data.append({
                    'symbol': symbol['symbol'],
                    'baseAsset': symbol['baseAsset'],
                    'quoteAsset': symbol['quoteAsset']
                })
        
        return pd.DataFrame(symbols_data)
    
    def get_klines_smart(self, symbol: str, interval: str, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        lookback_days: Optional[int] = None,
                        force_use_cache: bool = False) -> pd.DataFrame:
        """
        Smart kline fetching that always returns at least MIN_CACHE_DAYS of data
        
        Args:
            force_use_cache: If True, only use cached data without fetching
        """
        # Determine date range and ensure UTC
        if lookback_days:
            end_date = self._ensure_utc(end_date) or self._get_utc_now()
            start_date = end_date - timedelta(days=lookback_days)
        else:
            end_date = self._ensure_utc(end_date) or self._get_utc_now()
            start_date = self._ensure_utc(start_date) or (end_date - timedelta(days=30))
        
        # IMPORTANT: Always ensure we fetch/return at least MIN_CACHE_DAYS
        # This ensures price_metrics.py always has 30 days to work with
        extended_start_date = end_date - timedelta(days=self.MIN_CACHE_DAYS)
        if start_date > extended_start_date:
            self.logger.info(f"Extending start date from {start_date} to {extended_start_date} to ensure {self.MIN_CACHE_DAYS} days")
            start_date = extended_start_date
        
        self.logger.info(f"get_klines_smart for {symbol} {interval}: {start_date} to {end_date}, force_cache={force_use_cache}")
        
        # If force_use_cache, return whatever we have (but still try to get 30 days)
        if force_use_cache:
            self.cache_hit_count += 1
            cached_data = self.data_manager.load_data(symbol, interval, start_date, end_date)
            if cached_data is None:
                self.logger.warning(f"No cached data for {symbol} {interval} when force_use_cache=True")
                return pd.DataFrame()
            return cached_data
        
        # Identify what data we need
        gaps = self.data_manager.identify_data_gaps(symbol, interval, start_date, end_date)
        
        if not gaps:
            # All data is cached!
            self.cache_hit_count += 1
            self.logger.info(f"Cache hit for {symbol} {interval} - no API calls needed!")
            return self.data_manager.load_data(symbol, interval, start_date, end_date)
        
        # Fetch only the gaps
        all_new_data = []
        for gap_start, gap_end in gaps:
            self.logger.info(f"Fetching gap: {gap_start} to {gap_end}")
            new_data = self._fetch_klines_from_api(symbol, interval, gap_start, gap_end)
            if not new_data.empty:
                all_new_data.append(new_data)
        
        # Save new data (will merge with existing)
        if all_new_data:
            combined_new = pd.concat(all_new_data, ignore_index=True)
            self.data_manager.save_data(symbol, interval, combined_new)
        
        # Load complete dataset
        return self.data_manager.load_data(symbol, interval, start_date, end_date)
    
    def _fetch_klines_from_api(self, symbol: str, interval: str,
                              start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch kline data directly from Binance API
        
        IMPORTANT:
        - Timestamps represent candle OPEN times
        - The 'close' price is the price at the END of the candle period
        - Ongoing candles will have fluctuating close prices
        """
        endpoint = f"{self.base_url}/fapi/v1/klines"
        
        all_data = []
        current_start = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # Debug logging for timestamp issues
        self.logger.debug(f"start_date: {start_date}, timestamp: {start_date.timestamp()}, ms: {current_start}")
        self.logger.debug(f"end_date: {end_date}, timestamp: {end_date.timestamp()}, ms: {end_timestamp}")
        
        # Verify timestamps are reasonable
        current_time = self._get_utc_now()
        if start_date > current_time:
            self.logger.warning(f"Start date {start_date} is in the future! Using current time.")
            return pd.DataFrame()
        
        self.logger.info(f"Fetching {symbol} from API: {start_date} to {end_date}")
        fetch_start_time = time.time()
        
        while current_start < end_timestamp:
            self._rate_limit()
            self.api_call_count += 1
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_timestamp,
                'limit': 1500
            }
            
            try:
                api_start = time.time()
                self.logger.debug(f"API call #{self.api_call_count}: fetching up to 1500 candles from {datetime.fromtimestamp(current_start/1000)}")
                
                response = self.session.get(endpoint, params=params)
                response.raise_for_status()
                
                api_time = time.time() - api_start
                self.logger.info(f"API response time: {api_time*1000:.0f}ms")
                
                data = response.json()
                
                if not data:
                    self.logger.info(f"No data returned for {symbol} in this time range")
                    break
                
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # IMPORTANT: 'timestamp' is the OPEN time of the candle
                # 'close_time' is when the candle closes (timestamp + interval - 1ms)
                # The 'close' price is the price at close_time, not at timestamp!
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
                
                # Check if we got data from the requested time range
                first_timestamp = df['timestamp'].iloc[0]
                last_timestamp = df['timestamp'].iloc[-1]
                
                self.logger.info(f"Received {len(df)} candles from {first_timestamp} to {last_timestamp}")
                
                # IMPORTANT: Check if Binance returned data from before our requested start time
                # This happens when requesting future data - Binance returns the most recent available
                if last_timestamp < start_date:
                    self.logger.warning(f"Binance returned old data! Asked for >= {start_date}, got up to {last_timestamp}")
                    self.logger.warning("This likely means no newer data is available yet (API delay or future timestamp)")
                    
                    # Let's check what the actual latest available data is
                    test_params = {
                        'symbol': symbol,
                        'interval': interval,
                        'limit': 1
                    }
                    test_response = self.session.get(endpoint, params=test_params)
                    if test_response.status_code == 200:
                        latest_data = test_response.json()
                        if latest_data:
                            latest_time = datetime.fromtimestamp(latest_data[0][0] / 1000, tz=timezone.utc)
                            self.logger.info(f"Latest available candle for {symbol} starts at: {latest_time}")
                    
                    break
                
                # Convert numeric columns
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Filter to only include data from our requested range
                df_filtered = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                if len(df_filtered) == 0:
                    self.logger.warning(f"No data in requested range after filtering")
                    break
                
                # Only keep complete candles (not ongoing ones)
                complete_mask = df_filtered['close_time'] <= current_time
                if not complete_mask.all():
                    self.logger.info(f"Filtering out {(~complete_mask).sum()} ongoing candles")
                    df_filtered = df_filtered[complete_mask]
                
                if len(df_filtered) == 0:
                    self.logger.warning("No complete candles in this batch")
                    break
                
                # Store the close_time before dropping it
                last_close_time_ms = int(df_filtered['close_time'].iloc[-1].timestamp() * 1000)
                
                # Keep essential columns plus close_time for validation
                df_filtered = df_filtered[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
                
                all_data.append(df_filtered)
                
                # Check if we got all data
                if len(data) < 1500:
                    self.logger.info(f"Got less than 1500 candles ({len(data)}), assuming end of available data")
                    break
                else:
                    # Move to next batch - start from the close time of the last candle + 1ms
                    current_start = last_close_time_ms + 1
                    self.logger.debug(f"Moving to next batch starting from {datetime.fromtimestamp(current_start/1000, tz=timezone.utc)}")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    self.logger.error(f"Bad request to Binance API. Check symbol {symbol} and parameters.")
                    self.logger.error(f"Request params: {params}")
                else:
                    self.logger.error(f"HTTP error {e.response.status_code}: {e}")
                break
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                break
        
        total_fetch_time = time.time() - fetch_start_time
        self.logger.info(f"Total fetch time for {symbol}: {total_fetch_time:.2f}s for {len(all_data)} batches")
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Remove close_time column before returning (it was just for validation)
            if 'close_time' in result.columns:
                result = result.drop('close_time', axis=1)
            # Ensure no duplicates and sorted
            original_len = len(result)
            result = result.drop_duplicates(subset=['timestamp'])
            if len(result) < original_len:
                self.logger.info(f"Removed {original_len - len(result)} duplicate candles")
            result = result.sort_values('timestamp').reset_index(drop=True)
            self.logger.info(f"Total fetched: {len(result)} candles from {result['timestamp'].min()} to {result['timestamp'].max()}")
            return result
        else:
            self.logger.warning(f"No new data fetched for {symbol}")
            return pd.DataFrame()
    
    def get_klines(self, symbol: str, interval: str, **kwargs) -> pd.DataFrame:
        """Compatibility wrapper - always use smart caching"""
        return self.get_klines_smart(symbol, interval, **kwargs)
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data"""
        if symbol:
            # Clear specific symbol
            symbol_dir = self.data_manager.cache_dir / symbol
            if symbol_dir.exists():
                import shutil
                shutil.rmtree(symbol_dir)
            # Update metadata
            if symbol in self.data_manager.metadata["symbols"]:
                del self.data_manager.metadata["symbols"][symbol]
                self.data_manager._save_metadata()
        else:
            # Clear all
            self.data_manager.clear_all_cache()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        inventory = self.data_manager.get_data_inventory()
        total_size_mb = 0
        
        # Calculate total size
        for symbol_data in self.data_manager.metadata["symbols"].values():
            for info in symbol_data.values():
                total_size_mb += info.get("file_size_mb", 0)
        
        return {
            "total_symbols": inventory["total_symbols"],
            "total_files": inventory["total_files"],
            "total_rows": inventory["total_rows"],
            "total_size_mb": total_size_mb,
            "api_calls": self.api_call_count,
            "cache_hits": self.cache_hit_count
        }
    
    def reset_counters(self):
        """Reset API call counters"""
        self.api_call_count = 0
        self.cache_hit_count = 0