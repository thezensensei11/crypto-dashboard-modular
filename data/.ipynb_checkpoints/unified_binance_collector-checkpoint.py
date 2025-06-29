"""
Unified Binance Data Collector - Async-only implementation
Consolidates all data collection logic into a single, optimized module
"""

import aiohttp
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time

from crypto_dashboard_modular.data.smart_cache import SmartDataManager


logger = logging.getLogger(__name__)


class TokenBucket:
    """Async token bucket for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        """Acquire tokens, waiting if necessary"""
        async with self.lock:
            # Refill tokens based on time passed
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # Wait if not enough tokens
            while self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.refill_rate
                await asyncio.sleep(wait_time)
                
                # Refill again after waiting
                now = time.time()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now
            
            self.tokens -= tokens


class BinanceDataCollector:
    """
    Unified async Binance data collector with smart caching
    
    Features:
    - Async-only implementation for maximum performance
    - Smart gap detection and minimal API calls
    - Connection pooling and rate limiting
    - Automatic retries and error handling
    """
    
    # Constants
    BASE_URL = "https://fapi.binance.com"
    MIN_CACHE_DAYS = 30  # Always maintain at least 30 days of data
    MAX_BATCH_SIZE = 50  # Maximum symbols per batch request
    
    def __init__(self):
        self.logger = logger
        self.data_manager = SmartDataManager()
        
        # Rate limiter: 2000 requests/min = ~33.3 requests/sec
        self.rate_limiter = TokenBucket(
            capacity=100,     # Burst capacity
            refill_rate=33.3  # Tokens per second
        )
        
        # Connection management
        self.connector = None
        self.session = None
        
        # Statistics tracking
        self.api_call_count = 0
        self.cache_hit_count = 0
        
        # Initialize session on first use
        self._session_lock = asyncio.Lock()
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized"""
        if self.session is None:
            async with self._session_lock:
                if self.session is None:  # Double-check pattern
                    timeout = aiohttp.ClientTimeout(total=30, connect=5)
                    self.connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=30,
                        ttl_dns_cache=300,
                        enable_cleanup_closed=True
                    )
                    
                    self.session = aiohttp.ClientSession(
                        connector=self.connector,
                        timeout=timeout,
                        headers={
                            'Accept': 'application/json',
                            'User-Agent': 'CryptoDashboard/2.0'
                        }
                    )
    
    async def close(self):
        """Close aiohttp session and connector"""
        if self.session:
            await self.session.close()
            self.session = None
        if self.connector:
            await self.connector.close()
            self.connector = None
    
    def _get_utc_now(self) -> datetime:
        """Get current UTC time"""
        return datetime.now(timezone.utc)
    
    def _ensure_utc(self, dt: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime is UTC"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            return dt.astimezone(timezone.utc)
        return dt
    
    async def get_price_data(
        self,
        symbol: str,
        interval: str = '1h',
        lookback_days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_cache: bool = False
    ) -> pd.DataFrame:
        """
        Get price data for a single symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
            lookback_days: Number of days to look back (overrides start_date)
            start_date: Start date for data
            end_date: End date for data (defaults to now)
            force_cache: If True, only use cached data
        
        Returns:
            DataFrame with OHLCV data
        """
        await self._ensure_session()
        
        # Determine date range
        end_date = self._ensure_utc(end_date) or self._get_utc_now()
        
        if lookback_days:
            start_date = end_date - timedelta(days=lookback_days)
        else:
            start_date = self._ensure_utc(start_date) or (end_date - timedelta(days=30))
        
        # Ensure minimum data requirement
        min_start = end_date - timedelta(days=self.MIN_CACHE_DAYS)
        if start_date > min_start:
            self.logger.info(f"Extending start date to ensure {self.MIN_CACHE_DAYS} days")
            start_date = min_start
        
        # Check cache first
        if force_cache:
            self.cache_hit_count += 1
            cached_data = self.data_manager.load_data(symbol, interval, start_date, end_date)
            return cached_data if cached_data is not None else pd.DataFrame()
        
        # Identify gaps
        gaps = self.data_manager.identify_data_gaps(symbol, interval, start_date, end_date)
        
        if not gaps:
            self.cache_hit_count += 1
            self.logger.info(f"Cache hit for {symbol} {interval}")
            return self.data_manager.load_data(symbol, interval, start_date, end_date)
        
        # Fetch missing data
        all_new_data = []
        for gap_start, gap_end in gaps:
            self.logger.info(f"Fetching gap: {gap_start} to {gap_end}")
            new_data = await self._fetch_klines(symbol, interval, gap_start, gap_end)
            if not new_data.empty:
                all_new_data.append(new_data)
        
        # Save and return combined data
        if all_new_data:
            combined_new = pd.concat(all_new_data, ignore_index=True)
            self.data_manager.save_data(symbol, interval, combined_new)
        
        return self.data_manager.load_data(symbol, interval, start_date, end_date)
    
    async def get_price_data_batch(
        self,
        symbols: List[str],
        interval: str = '1h',
        lookback_days: int = 30,
        force_cache: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get price data for multiple symbols concurrently
        
        Args:
            symbols: List of trading symbols
            interval: Candle interval
            lookback_days: Number of days to look back
            force_cache: If True, only use cached data
        
        Returns:
            Dict mapping symbol to DataFrame
        """
        await self._ensure_session()
        
        # Process in batches to avoid overwhelming the API
        results = {}
        
        for i in range(0, len(symbols), self.MAX_BATCH_SIZE):
            batch = symbols[i:i + self.MAX_BATCH_SIZE]
            
            # Create tasks for concurrent fetching
            tasks = []
            for symbol in batch:
                task = self.get_price_data(
                    symbol=symbol,
                    interval=interval,
                    lookback_days=lookback_days,
                    force_cache=force_cache
                )
                tasks.append((symbol, task))
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            # Process results
            for (symbol, _), result in zip(tasks, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching {symbol}: {result}")
                    results[symbol] = pd.DataFrame()
                else:
                    results[symbol] = result
        
        return results
    
    async def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch kline data from Binance API"""
        all_data = []
        
        # Convert to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        current_ms = int(self._get_utc_now().timestamp() * 1000)
        
        # Extend end time to current if needed for latest data
        if end_ms < current_ms:
            self.logger.info(f"Extending end time to fetch latest data")
            end_ms = current_ms
        
        # Fetch in chunks (max 1500 per request)
        current_start = start_ms
        
        while current_start < end_ms:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Prepare request
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1500
            }
            
            try:
                # Make request
                async with self.session.get(
                    f"{self.BASE_URL}/fapi/v1/klines",
                    params=params
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    self.api_call_count += 1
                    
                    if not data:
                        break
                    
                    # Convert to DataFrame
                    df = self._parse_klines(data)
                    all_data.append(df)
                    
                    # Update start time for next batch
                    last_timestamp = data[-1][0]
                    current_start = last_timestamp + 1
                    
                    # Avoid fetching same data
                    if len(data) < 1500:
                        break
                        
            except aiohttp.ClientError as e:
                self.logger.error(f"API error fetching {symbol}: {e}")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error fetching {symbol}: {e}")
                break
        
        # Combine all data
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            result = result.drop_duplicates(subset=['timestamp'], keep='last')
            result = result.sort_values('timestamp').reset_index(drop=True)
            
            # Filter out future candles
            current_time = self._get_utc_now()
            result = result[result['timestamp'] <= current_time]
            
            # Filter to requested range
            result = result[
                (result['timestamp'] >= start_date) & 
                (result['timestamp'] <= end_date)
            ]
            
            self.logger.info(f"Fetched {len(result)} candles for {symbol}")
            return result
        
        return pd.DataFrame()
    
    def _parse_klines(self, klines: List[List]) -> pd.DataFrame:
        """Parse kline data from Binance API response"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Select relevant columns
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    async def get_exchange_info(self) -> pd.DataFrame:
        """Get exchange information and trading symbols"""
        await self._ensure_session()
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get(f"{self.BASE_URL}/fapi/v1/exchangeInfo") as response:
                response.raise_for_status()
                data = await response.json()
                self.api_call_count += 1
                
                symbols_data = []
                for symbol in data['symbols']:
                    if symbol['status'] == 'TRADING' and symbol['contractType'] == 'PERPETUAL':
                        symbols_data.append({
                            'symbol': symbol['symbol'],
                            'baseAsset': symbol['baseAsset'],
                            'quoteAsset': symbol['quoteAsset']
                        })
                
                return pd.DataFrame(symbols_data)
                
        except Exception as e:
            self.logger.error(f"Error fetching exchange info: {e}")
            return pd.DataFrame()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        inventory = self.data_manager.get_data_inventory()
        
        total_size_mb = 0
        for symbol_data in self.data_manager.metadata.get("symbols", {}).values():
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
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data"""
        if symbol:
            # Clear specific symbol
            symbol_dir = self.data_manager.cache_dir / symbol
            if symbol_dir.exists():
                import shutil
                shutil.rmtree(symbol_dir)
            
            # Update metadata
            if symbol in self.data_manager.metadata.get("symbols", {}):
                del self.data_manager.metadata["symbols"][symbol]
                self.data_manager._save_metadata()
        else:
            # Clear all
            self.data_manager.clear_all_cache()
    
    def reset_counters(self):
        """Reset API call counters"""
        self.api_call_count = 0
        self.cache_hit_count = 0
    
    # Compatibility methods for sync usage
    def get_klines(self, **kwargs) -> pd.DataFrame:
        """Sync wrapper for backward compatibility"""
        return asyncio.run(self.get_price_data(**kwargs))
    
    def get_klines_smart(self, **kwargs) -> pd.DataFrame:
        """Sync wrapper for backward compatibility"""
        return asyncio.run(self.get_price_data(**kwargs))


# Module-level collector instance
_collector_instance = None


async def get_collector() -> BinanceDataCollector:
    """Get or create the collector instance"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = BinanceDataCollector()
        await _collector_instance._ensure_session()
    return _collector_instance


def cleanup_collector():
    """Cleanup the collector instance"""
    global _collector_instance
    if _collector_instance is not None:
        asyncio.run(_collector_instance.close())
        _collector_instance = None