"""
Data processor for validation, aggregation, and database writing
Place in: crypto-dashboard/infrastructure/processors/data_processor.py
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from decimal import Decimal
import pandas as pd

from core.config import get_settings
from core.models import (
    OHLCVData, Event, EventType, ValidationResult, 
    AggregationRequest, ProcessingStats, Interval
)
from core.exceptions import ProcessingException, ValidationException
from core.constants import MAX_PRICE_SPIKE_PCT, MIN_VOLUME_THRESHOLD
from infrastructure.message_bus.bus import get_message_bus, publish_event
from infrastructure.database.manager import get_db_manager

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates OHLCV data for quality and consistency"""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.processor
    
    def validate_ohlcv(self, data: OHLCVData) -> ValidationResult:
        """Validate single OHLCV data point"""
        result = ValidationResult(is_valid=True, data=data)
        
        # Check basic constraints
        if data.open <= 0 or data.high <= 0 or data.low <= 0 or data.close <= 0:
            result.add_error("Price values must be positive")
        
        if data.high < data.low:
            result.add_error("High price cannot be less than low price")
        
        if data.high < max(data.open, data.close):
            result.add_error("High price must be >= max(open, close)")
        
        if data.low > min(data.open, data.close):
            result.add_error("Low price must be <= min(open, close)")
        
        if data.volume < self.config.min_volume:
            result.add_error(f"Volume {data.volume} below minimum {self.config.min_volume}")
        
        # Check for price spikes
        price_change_pct = abs((float(data.close) - float(data.open)) / float(data.open) * 100)
        if price_change_pct > self.config.max_price_change_pct:
            result.add_warning(f"Large price change: {price_change_pct:.2f}%")
        
        # Check timestamp
        if data.timestamp > datetime.now(timezone.utc):
            result.add_error("Timestamp cannot be in the future")
        
        return result
    
    def validate_batch(self, data_list: List[OHLCVData]) -> List[ValidationResult]:
        """Validate batch of OHLCV data"""
        results = []
        
        for i, data in enumerate(data_list):
            result = self.validate_ohlcv(data)
            
            # Additional batch validations
            if i > 0:
                prev_data = data_list[i-1]
                
                # Check sequence
                if data.timestamp <= prev_data.timestamp:
                    result.add_error("Timestamps must be in ascending order")
                
                # Check for gaps
                expected_gap = timedelta(seconds=Interval(data.interval).seconds)
                actual_gap = data.timestamp - prev_data.timestamp
                
                if actual_gap > expected_gap * 1.5:
                    result.add_warning(f"Gap detected: {actual_gap}")
            
            results.append(result)
        
        return results


class DataAggregator:
    """Aggregates OHLCV data to higher timeframes"""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.processor
        self.db_manager = get_db_manager()
    
    def can_aggregate(self, source_interval: str, target_interval: str) -> bool:
        """Check if aggregation is possible"""
        aggregation_map = self.config.aggregation_intervals
        return (target_interval in aggregation_map and 
                source_interval in aggregation_map[target_interval])
    
    def aggregate_candles(
        self,
        candles: List[OHLCVData],
        target_interval: str
    ) -> List[OHLCVData]:
        """Aggregate candles to target interval"""
        if not candles:
            return []
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([c.dict() for c in candles])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Determine aggregation period
        interval_map = {
            '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
        }
        
        if target_interval not in interval_map:
            raise ProcessingException(f"Unsupported target interval: {target_interval}")
        
        # Group by time period
        grouped = df.groupby([
            pd.Grouper(key='timestamp', freq=interval_map[target_interval]),
            'symbol'
        ])
        
        aggregated = []
        
        for (timestamp, symbol), group in grouped:
            if group.empty:
                continue
            
            agg_candle = OHLCVData(
                symbol=symbol,
                interval=target_interval,
                timestamp=timestamp,
                open=Decimal(str(group.iloc[0]['open'])),
                high=Decimal(str(group['high'].max())),
                low=Decimal(str(group['low'].min())),
                close=Decimal(str(group.iloc[-1]['close'])),
                volume=Decimal(str(group['volume'].sum())),
                quote_volume=Decimal(str(group['quote_volume'].sum())) if 'quote_volume' in group else None,
                trades=int(group['trades'].sum()) if 'trades' in group else None,
                taker_buy_base=Decimal(str(group['taker_buy_base'].sum())) if 'taker_buy_base' in group else None,
                taker_buy_quote=Decimal(str(group['taker_buy_quote'].sum())) if 'taker_buy_quote' in group else None
            )
            
            aggregated.append(agg_candle)
        
        return aggregated
    
    async def process_aggregation_request(self, request: AggregationRequest) -> int:
        """Process aggregation request"""
        # Fetch source data
        df = self.db_manager.read_ohlcv_data(
            symbol=request.symbol,
            interval=request.source_interval,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        if df.empty:
            logger.warning(f"No data found for aggregation: {request.symbol} {request.source_interval}")
            return 0
        
        # Convert to OHLCVData objects
        candles = []
        for _, row in df.iterrows():
            candles.append(OHLCVData(**row.to_dict()))
        
        # Aggregate
        aggregated = self.aggregate_candles(candles, request.target_interval)
        
        if aggregated:
            # Write to database
            written = self.db_manager.write_ohlcv_data(
                aggregated,
                request.target_interval,
                upsert=True
            )
            
            logger.info(f"Aggregated {len(candles)} {request.source_interval} candles "
                       f"to {written} {request.target_interval} candles")
            
            return written
        
        return 0


class DataProcessor:
    """
    Main data processor that orchestrates validation, aggregation, and storage
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.processor
        
        # Components
        self.validator = DataValidator()
        self.aggregator = DataAggregator()
        self.db_manager = get_db_manager()
        self.message_bus: Optional[MessageBus] = None
        
        # Processing state
        self._running = False
        self._stats = ProcessingStats()
        self._buffer: Dict[str, List[OHLCVData]] = {}
        self._buffer_lock = asyncio.Lock()
        self._last_flush_time = datetime.now(timezone.utc)
    
    async def start(self):
        """Start the data processor"""
        logger.info("Starting data processor...")
        self._running = True
        
        # Initialize message bus
        self.message_bus = await get_message_bus()
        
        # Subscribe to events
        await self.message_bus.subscribe(
            event_types=[
                EventType.PRICE_UPDATE,
                EventType.CANDLE_CLOSED,
                EventType.HISTORICAL_DATA
            ],
            handler=self.process_event,
            component="data_processor"
        )
        
        # Start flush task
        asyncio.create_task(self._flush_loop())
        
        # Publish startup event
        await publish_event(
            EventType.PROCESSOR_STARTED,
            source="DataProcessor",
            data={"processor_type": "main"}
        )
    
    async def stop(self):
        """Stop the data processor"""
        logger.info("Stopping data processor...")
        self._running = False
        
        # Flush remaining buffer
        await self._flush_all_buffers()
        
        # Publish shutdown event
        await publish_event(
            EventType.PROCESSOR_STOPPED,
            source="DataProcessor",
            data={"processor_type": "main", "stats": self._stats.dict()}
        )
    
    async def process_event(self, event: Event):
        """Process incoming events"""
        try:
            self._stats.increment('processed_count')
            
            if event.type == EventType.HISTORICAL_DATA:
                await self._process_historical_data(event)
            elif event.type in [EventType.PRICE_UPDATE, EventType.CANDLE_CLOSED]:
                await self._process_live_data(event)
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            self._stats.increment('error_count')
            
            # Publish error event
            await publish_event(
                EventType.ERROR_OCCURRED,
                source="DataProcessor",
                data={
                    "error": str(e),
                    "event_type": event.type.value,
                    "event_id": event.id
                }
            )
    
    async def _process_historical_data(self, event: Event):
        """Process historical data batch"""
        data_list = []
        
        # Parse OHLCV data from event
        for data_dict in event.data.get('data', []):
            try:
                ohlcv = OHLCVData(**data_dict)
                data_list.append(ohlcv)
            except Exception as e:
                logger.error(f"Failed to parse OHLCV data: {e}")
                self._stats.increment('error_count')
        
        if not data_list:
            return
        
        # Validate batch
        validation_results = self.validator.validate_batch(data_list)
        
        valid_data = []
        for result in validation_results:
            if result.is_valid:
                valid_data.append(result.data)
                self._stats.increment('validated_count')
            else:
                logger.warning(f"Invalid data: {result.errors}")
                self._stats.increment('invalid_count')
        
        # Add to buffer
        if valid_data:
            await self._add_to_buffer(valid_data)
    
    async def _process_live_data(self, event: Event):
        """Process live WebSocket data"""
        if 'ohlcv' in event.data:
            try:
                ohlcv = OHLCVData(**event.data['ohlcv'])
                
                # Validate
                result = self.validator.validate_ohlcv(ohlcv)
                
                if result.is_valid:
                    self._stats.increment('validated_count')
                    
                    # For closed candles, add to buffer
                    if event.type == EventType.CANDLE_CLOSED or event.data.get('is_closed'):
                        await self._add_to_buffer([ohlcv])
                        
                        # Trigger aggregation if needed
                        await self._check_aggregation_needed(ohlcv)
                else:
                    logger.warning(f"Invalid live data: {result.errors}")
                    self._stats.increment('invalid_count')
                    
            except Exception as e:
                logger.error(f"Failed to process live data: {e}")
                self._stats.increment('error_count')
    
    async def _add_to_buffer(self, data_list: List[OHLCVData]):
        """Add data to buffer for batch writing"""
        async with self._buffer_lock:
            for data in data_list:
                key = f"{data.symbol}_{data.interval}"
                if key not in self._buffer:
                    self._buffer[key] = []
                self._buffer[key].append(data)
    
    async def _flush_loop(self):
        """Periodically flush buffer to database"""
        while self._running:
            await asyncio.sleep(self.config.batch_timeout)
            await self._flush_buffers()
    
    async def _flush_buffers(self):
        """Flush buffers that meet criteria"""
        async with self._buffer_lock:
            current_time = datetime.now(timezone.utc)
            time_since_flush = (current_time - self._last_flush_time).seconds
            
            for key, data_list in list(self._buffer.items()):
                # Flush if buffer is large enough or timeout reached
                if (len(data_list) >= self.config.batch_size or 
                    time_since_flush >= self.config.batch_timeout):
                    
                    if data_list:
                        symbol, interval = key.split('_', 1)
                        await self._write_to_database(data_list, interval)
                        self._buffer[key] = []
            
            self._last_flush_time = current_time
    
    async def _flush_all_buffers(self):
        """Flush all buffers immediately"""
        async with self._buffer_lock:
            for key, data_list in self._buffer.items():
                if data_list:
                    symbol, interval = key.split('_', 1)
                    await self._write_to_database(data_list, interval)
            self._buffer.clear()
    
    async def _write_to_database(self, data_list: List[OHLCVData], interval: str):
        """Write data to database"""
        try:
            # Publish write started event
            await publish_event(
                EventType.DB_WRITE_STARTED,
                source="DataProcessor",
                data={
                    "count": len(data_list),
                    "interval": interval,
                    "symbols": list(set(d.symbol for d in data_list))
                }
            )
            
            # Write to database
            written = self.db_manager.write_ohlcv_data(data_list, interval, upsert=True)
            self._stats.increment('written_count', written)
            
            # Publish write completed event
            await publish_event(
                EventType.DB_WRITE_COMPLETED,
                source="DataProcessor",
                data={
                    "count": written,
                    "interval": interval
                }
            )
            
        except Exception as e:
            logger.error(f"Database write failed: {e}")
            self._stats.increment('error_count')
            
            # Publish write failed event
            await publish_event(
                EventType.DB_WRITE_FAILED,
                source="DataProcessor",
                data={
                    "error": str(e),
                    "count": len(data_list),
                    "interval": interval
                }
            )
    
    async def _check_aggregation_needed(self, candle: OHLCVData):
        """Check if aggregation is needed for closed candle"""
        # Only aggregate from base intervals
        if candle.interval not in ['1m', '5m', '15m', '1h', '4h']:
            return
        
        # Check which aggregations are possible
        for target_interval, source_intervals in self.config.aggregation_intervals.items():
            if candle.interval in source_intervals:
                # Publish aggregation needed event
                await publish_event(
                    EventType.AGGREGATION_NEEDED,
                    source="DataProcessor",
                    data={
                        "symbol": candle.symbol,
                        "source_interval": candle.interval,
                        "target_interval": target_interval,
                        "timestamp": candle.timestamp.isoformat()
                    }
                )

