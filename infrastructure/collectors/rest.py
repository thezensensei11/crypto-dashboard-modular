"""
REST API data collector for historical data
Place in: crypto-dashboard/infrastructure/collectors/rest.py
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import logging

from core.config import get_settings
from core.models import OHLCVData, Event, EventType, DataSource, BatchData, Interval
from core.exceptions import CollectorException
from infrastructure.message_bus.bus import get_message_bus, publish_event
from infrastructure.database.manager import get_db_manager
from .base import BaseCollector

logger = logging.getLogger(__name__)


class RESTCollector(BaseCollector):
    """
    REST API collector for historical market data from Binance
    
    Features:
    - Smart gap detection and backfilling
    - Rate limiting and request management
    - Parallel request processing
    - Batch data publishing
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.config = self.settings.collector
        self.binance_config = self.settings.binance
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._request_weight = 0
        self._weight_reset_time = datetime.now(timezone.utc)
        self._request_semaphore = asyncio.Semaphore(self.config.rest_parallel_requests)
        
        # Database manager for gap detection
        self.db_manager = get_db_manager()
    
    async def start(self):
        """Start the REST collector"""
        logger.info("Starting REST collector...")
        
        # Initialize HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.binance_config.timeout)
        )
        
        # Initialize message bus
        self.message_bus = await get_message_bus()
        
        # Publish startup event
        await publish_event(
            EventType.COLLECTOR_STARTED,
            source="RESTCollector",
            data={"collector_type": "rest"}
        )
    
    async def stop(self):
        """Stop the REST collector"""
        logger.info("Stopping REST collector...")
        
        # Close HTTP session
        if self._session:
            await self._session.close()
        
        # Publish shutdown event
        await publish_event(
            EventType.COLLECTOR_STOPPED,
            source="RESTCollector",
            data={"collector_type": "rest"}
        )
    
    async def collect_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> int:
        """
        Collect historical data for a symbol
        
        Returns:
            Number of candles collected
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        logger.info(f"Collecting {symbol} {interval} data from {start_time} to {end_time}")
        
        total_collected = 0
        current_start = start_time
        
        while current_start < end_time:
            # Calculate batch end time
            batch_end = min(
                current_start + timedelta(days=self.config.rest_max_lookback_days),
                end_time
            )
            
            # Fetch batch
            try:
                data = await self._fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=batch_end
                )
                
                if data:
                    # Create batch and publish
                    batch = BatchData(
                        symbol=symbol,
                        interval=interval,
                        data=data,
                        source=DataSource.REST_API
                    )
                    
                    await self._publish_batch(batch)
                    total_collected += len(data)
                    
                    # Update current_start to last timestamp
                    current_start = data[-1].timestamp + timedelta(seconds=Interval(interval).seconds)
                else:
                    # No data returned, move to next period
                    current_start = batch_end
                    
            except Exception as e:
                logger.error(f"Error collecting data: {e}")
                # Move to next period to avoid infinite loop
                current_start = batch_end
        
        logger.info(f"Collected {total_collected} candles for {symbol} {interval}")
        return total_collected
    
    async def backfill_gaps(
        self,
        symbol: str,
        interval: str,
        lookback_days: Optional[int] = None
    ) -> int:
        """
        Detect and backfill data gaps
        
        Returns:
            Number of candles collected
        """
        if lookback_days is None:
            # Use retention policy to determine lookback
            if interval == "1m" or interval == "5m":
                lookback_days = self.settings.duckdb.retention_1m
            elif interval == "15m":
                lookback_days = self.settings.duckdb.retention_15m
            else:
                lookback_days = 365  # Default to 1 year
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        
        # Get gaps from database
        gaps = self.db_manager.get_data_gaps(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if not gaps:
            logger.info(f"No gaps found for {symbol} {interval}")
            return 0
        
        logger.info(f"Found {len(gaps)} gaps for {symbol} {interval}")
        
        total_collected = 0
        
        # Fill each gap
        for gap_start, gap_end in gaps:
            collected = await self.collect_historical_data(
                symbol=symbol,
                interval=interval,
                start_time=gap_start,
                end_time=gap_end
            )
            total_collected += collected
        
        return total_collected
    
    async def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[OHLCVData]:
        """Fetch klines from Binance API"""
        async with self._request_semaphore:
            # Check rate limits
            await self._check_rate_limits(weight=10)
            
            # Build request parameters
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000),
                "limit": min(limit, self.config.rest_batch_size)
            }
            
            url = f"{self.binance_config.futures_base_url}/fapi/v1/klines"
            
            try:
                async with self._session.get(url, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise CollectorException(f"API error {response.status}: {text}")
                    
                    data = await response.json()
                    
                    # Parse klines
                    ohlcv_list = []
                    for kline in data:
                        ohlcv = OHLCVData(
                            symbol=symbol,
                            interval=interval,
                            timestamp=datetime.fromtimestamp(kline[0] / 1000),
                            open=Decimal(str(kline[1])),
                            high=Decimal(str(kline[2])),
                            low=Decimal(str(kline[3])),
                            close=Decimal(str(kline[4])),
                            volume=Decimal(str(kline[5])),
                            quote_volume=Decimal(str(kline[7])),
                            trades=int(kline[8]),
                            taker_buy_base=Decimal(str(kline[9])),
                            taker_buy_quote=Decimal(str(kline[10])),
                            source=DataSource.REST_API
                        )
                        ohlcv_list.append(ohlcv)
                    
                    return ohlcv_list
                    
            except asyncio.TimeoutError:
                raise CollectorException("Request timeout")
            except Exception as e:
                raise CollectorException(f"Failed to fetch klines: {e}")
    
    async def _check_rate_limits(self, weight: int):
        """Check and update rate limits"""
        now = datetime.now(timezone.utc)
        
        # Reset weight if minute has passed
        if (now - self._weight_reset_time).seconds >= 60:
            self._request_weight = 0
            self._weight_reset_time = now
        
        # Check if we would exceed limit
        if self._request_weight + weight > self.binance_config.weight_limit:
            # Wait until next minute
            wait_time = 60 - (now - self._weight_reset_time).seconds
            logger.warning(f"Rate limit reached, waiting {wait_time} seconds")
            await asyncio.sleep(wait_time)
            
            # Reset counters
            self._request_weight = 0
            self._weight_reset_time = datetime.now(timezone.utc)
        
        # Update weight
        self._request_weight += weight
    
    async def _publish_batch(self, batch: BatchData):
        """Publish batch data to message bus"""
        # Split large batches
        chunk_size = 100
        
        for i in range(0, batch.count, chunk_size):
            chunk_data = batch.data[i:i + chunk_size]
            
            # Publish historical data event
            await publish_event(
                EventType.HISTORICAL_DATA,
                source="RESTCollector",
                data={
                    "symbol": batch.symbol,
                    "interval": batch.interval,
                    "count": len(chunk_data),
                    "time_range": {
                        "start": chunk_data[0].timestamp.isoformat(),
                        "end": chunk_data[-1].timestamp.isoformat()
                    },
                    "data": [d.dict() for d in chunk_data]
                }
            )
    
    async def collect_multiple_symbols(
        self,
        symbols: List[str],
        intervals: List[str],
        lookback_days: int = 7
    ) -> Dict[str, int]:
        """
        Collect data for multiple symbols and intervals
        
        Returns:
            Dictionary of symbol -> candles collected
        """
        results = {}
        tasks = []
        
        for symbol in symbols:
            for interval in intervals:
                task = self.backfill_gaps(
                    symbol=symbol,
                    interval=interval,
                    lookback_days=lookback_days
                )
                tasks.append((symbol, interval, task))
        
        # Process in parallel
        for symbol, interval, task in tasks:
            try:
                count = await task
                key = f"{symbol}_{interval}"
                results[key] = count
            except Exception as e:
                logger.error(f"Failed to collect {symbol} {interval}: {e}")
                results[f"{symbol}_{interval}"] = 0
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get collector status"""
        return {
            "session_active": self._session is not None and not self._session.closed,
            "request_weight": self._request_weight,
            "weight_reset_time": self._weight_reset_time.isoformat()
        }
