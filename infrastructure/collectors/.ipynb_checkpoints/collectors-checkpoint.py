"""
Standalone Data Collector Services
Independent services for REST API and WebSocket data collection
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Set, Any
import logging
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import websockets
from contextlib import asynccontextmanager

from message_bus_events import MessageBus, Event, EventType, create_event
from duckdb_schema_dal import get_db_manager, DataInterval

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for data collectors"""
    binance_base_url: str = "https://fapi.binance.com"
    binance_ws_url: str = "wss://fstream.binance.com"
    max_requests_per_minute: int = 2000
    websocket_reconnect_delay: int = 5
    batch_size: int = 100
    
    # Intervals to collect
    rest_intervals: List[str] = None
    ws_streams: List[str] = None
    
    def __post_init__(self):
        if self.rest_intervals is None:
            self.rest_intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
        if self.ws_streams is None:
            self.ws_streams = ['trade', 'bookTicker', 'miniTicker']


class BaseCollector(ABC):
    """Base class for all collectors"""
    
    def __init__(self, config: CollectorConfig, message_bus: MessageBus):
        self.config = config
        self.message_bus = message_bus
        self.db_manager = get_db_manager()
        self._running = False
        
    @abstractmethod
    async def start(self):
        """Start the collector"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the collector"""
        pass
    
    def publish_event(self, event_type: EventType, data: Dict[str, Any]):
        """Publish event to message bus"""
        event = create_event(
            event_type=event_type,
            source=self.__class__.__name__,
            data=data
        )
        self.message_bus.publish_event(event)


class RESTDataCollector(BaseCollector):
    """
    REST API Data Collector Service
    Handles historical data fetching with smart gap detection
    """
    
    def __init__(self, config: CollectorConfig, message_bus: MessageBus):
        super().__init__(config, message_bus)
        self.session = None
        self.rate_limiter = RateLimiter(config.max_requests_per_minute)
        self.symbols_to_collect: Set[str] = set()
        
    async def start(self):
        """Start the REST collector"""
        self._running = True
        self.session = aiohttp.ClientSession()
        
        # Publish collector started event
        self.publish_event(EventType.COLLECTOR_STARTED, {
            'collector': 'RESTDataCollector',
            'intervals': self.config.rest_intervals
        })
        
        try:
            # Start collection tasks
            await self._run_collection_loop()
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the collector"""
        self._running = False
        if self.session:
            await self.session.close()
        
        self.publish_event(EventType.COLLECTOR_STOPPED, {
            'collector': 'RESTDataCollector'
        })
    
    async def _run_collection_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                # Get symbols to collect from configuration or database
                symbols = await self._get_symbols_to_collect()
                
                # Collect data for each symbol and interval
                tasks = []
                for symbol in symbols:
                    for interval in self.config.rest_intervals:
                        task = self._collect_symbol_data(symbol, interval)
                        tasks.append(task)
                
                # Process in batches to avoid overwhelming the system
                for i in range(0, len(tasks), self.config.batch_size):
                    batch = tasks[i:i + self.config.batch_size]
                    await asyncio.gather(*batch, return_exceptions=True)
                
                # Wait before next cycle
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(10)
    
    async def _get_symbols_to_collect(self) -> List[str]:
        """Get list of symbols to collect"""
        # This could be from config, database, or message bus
        # For now, return from database
        return self.db_manager.get_symbols() or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    async def _collect_symbol_data(self, symbol: str, interval: str):
        """Collect data for a specific symbol/interval"""
        try:
            # Check what data we need
            gaps = await self._identify_data_gaps(symbol, interval)
            
            if not gaps:
                logger.debug(f"No gaps for {symbol} {interval}")
                return
            
            # Fetch data for each gap
            for start_date, end_date in gaps:
                await self._fetch_and_store_data(symbol, interval, start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error collecting {symbol} {interval}: {e}")
            self.publish_event(EventType.DATA_FETCH_FAILED, {
                'symbol': symbol,
                'interval': interval,
                'error': str(e)
            })
    
    async def _identify_data_gaps(self, symbol: str, interval: str) -> List[tuple]:
        """Identify gaps in data"""
        # Get current time
        now = datetime.now(timezone.utc)
        
        # Get latest timestamp from database
        latest = self.db_manager.get_latest_timestamp(symbol, interval)
        
        if not latest:
            # No data, fetch last 30 days
            start = now - timedelta(days=30)
            return [(start, now)]
        
        # Check if we need new data
        interval_minutes = self._interval_to_minutes(interval)
        expected_next = latest + timedelta(minutes=interval_minutes)
        
        if expected_next < now:
            return [(expected_next, now)]
        
        return []
    
    async def _fetch_and_store_data(
        self, 
        symbol: str, 
        interval: str,
        start_date: datetime,
        end_date: datetime
    ):
        """Fetch data from Binance API and store in database"""
        endpoint = f"{self.config.binance_base_url}/fapi/v1/klines"
        
        all_data = []
        current_start = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        while current_start < end_timestamp:
            # Rate limit
            await self.rate_limiter.acquire()
            
            # Prepare request
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_timestamp,
                'limit': 1500
            }
            
            # Make request
            async with self.session.get(endpoint, params=params) as response:
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")
                
                data = await response.json()
                
                if not data:
                    break
                
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Process data
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                all_data.append(df)
                
                # Update start time for next batch
                if len(data) < 1500:
                    break
                else:
                    current_start = int(df['timestamp'].max().timestamp() * 1000) + 1
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Publish to message bus instead of direct DB write
            self.publish_event(EventType.DATA_FETCH_COMPLETED, {
                'symbol': symbol,
                'interval': interval,
                'rows': len(combined),
                'dataframe': combined.to_dict('records'),
                'start': str(combined['timestamp'].min()),
                'end': str(combined['timestamp'].max())
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval to minutes"""
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return mapping.get(interval, 60)


class WebSocketCollector(BaseCollector):
    """
    WebSocket Data Collector Service
    Handles real-time data streaming
    """
    
    def __init__(self, config: CollectorConfig, message_bus: MessageBus):
        super().__init__(config, message_bus)
        self.websockets: Dict[str, Any] = {}
        self.subscriptions: Set[str] = set()
        
    async def start(self):
        """Start WebSocket collector"""
        self._running = True
        
        self.publish_event(EventType.COLLECTOR_STARTED, {
            'collector': 'WebSocketCollector',
            'streams': self.config.ws_streams
        })
        
        try:
            await self._run_websocket_loop()
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop WebSocket collector"""
        self._running = False
        
        # Close all websockets
        for ws in self.websockets.values():
            await ws.close()
        
        self.publish_event(EventType.COLLECTOR_STOPPED, {
            'collector': 'WebSocketCollector'
        })
    
    async def subscribe(self, symbols: List[str], streams: List[str]):
        """Subscribe to specific symbols and streams"""
        for symbol in symbols:
            for stream in streams:
                sub = f"{symbol.lower()}@{stream}"
                self.subscriptions.add(sub)
    
    async def unsubscribe(self, symbols: List[str], streams: List[str]):
        """Unsubscribe from symbols and streams"""
        for symbol in symbols:
            for stream in streams:
                sub = f"{symbol.lower()}@{stream}"
                self.subscriptions.discard(sub)
    
    async def _run_websocket_loop(self):
        """Main WebSocket loop"""
        while self._running:
            try:
                # Connect to WebSocket
                async with websockets.connect(self._build_ws_url()) as websocket:
                    logger.info("Connected to Binance WebSocket")
                    
                    # Handle messages
                    async for message in websocket:
                        if not self._running:
                            break
                        
                        await self._process_ws_message(message)
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    await asyncio.sleep(self.config.websocket_reconnect_delay)
    
    def _build_ws_url(self) -> str:
        """Build WebSocket URL with subscriptions"""
        if not self.subscriptions:
            # Default subscriptions
            self.subscriptions = {
                'btcusdt@miniTicker',
                'ethusdt@miniTicker',
                'solusdt@miniTicker'
            }
        
        streams = '/'.join(self.subscriptions)
        return f"{self.config.binance_ws_url}/stream?streams={streams}"
    
    async def _process_ws_message(self, message: str):
        """Process WebSocket message"""
        try:
            data = json.loads(message)
            
            if 'stream' not in data:
                return
            
            stream = data['stream']
            payload = data['data']
            
            # Process based on stream type
            if stream.endswith('@trade'):
                await self._process_trade(payload)
            elif stream.endswith('@miniTicker'):
                await self._process_mini_ticker(payload)
            elif stream.endswith('@bookTicker'):
                await self._process_book_ticker(payload)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def _process_trade(self, data: Dict):
        """Process trade data"""
        self.publish_event(EventType.TRADE_UPDATE, {
            'symbol': data['s'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'timestamp': datetime.fromtimestamp(data['T'] / 1000, tz=timezone.utc).isoformat(),
            'is_buyer_maker': data['m']
        })
    
    async def _process_mini_ticker(self, data: Dict):
        """Process mini ticker data"""
        self.publish_event(EventType.PRICE_UPDATE, {
            'symbol': data['s'],
            'price': float(data['c']),
            'open': float(data['o']),
            'high': float(data['h']),
            'low': float(data['l']),
            'volume': float(data['v']),
            'quote_volume': float(data['q'])
        })
        
        # Update live price in database
        self.db_manager.update_live_price(
            data['s'],
            float(data['c']),
            datetime.now(timezone.utc)
        )
    
    async def _process_book_ticker(self, data: Dict):
        """Process order book ticker data"""
        self.publish_event(EventType.ORDERBOOK_UPDATE, {
            'symbol': data['s'],
            'bid': float(data['b']),
            'bid_qty': float(data['B']),
            'ask': float(data['a']),
            'ask_qty': float(data['A'])
        })


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int):
        self.capacity = requests_per_minute // 10  # Burst capacity
        self.refill_rate = requests_per_minute / 60  # Tokens per second
        self.tokens = float(self.capacity)
        self.last_refill = asyncio.get_event_loop().time()
    
    async def acquire(self, tokens: int = 1):
        """Acquire tokens, wait if necessary"""
        while True:
            now = asyncio.get_event_loop().time()
            
            # Refill tokens
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            
            # Wait for tokens
            wait_time = (tokens - self.tokens) / self.refill_rate
            await asyncio.sleep(wait_time)


# Service manager for running collectors
class CollectorManager:
    """Manages all data collectors"""
    
    def __init__(self, config: CollectorConfig, message_bus: MessageBus):
        self.config = config
        self.message_bus = message_bus
        self.collectors = {}
        
    async def start_all(self):
        """Start all collectors"""
        # Start REST collector
        rest_collector = RESTDataCollector(self.config, self.message_bus)
        self.collectors['rest'] = rest_collector
        
        # Start WebSocket collector
        ws_collector = WebSocketCollector(self.config, self.message_bus)
        self.collectors['websocket'] = ws_collector
        
        # Start all collectors
        tasks = []
        for collector in self.collectors.values():
            task = asyncio.create_task(collector.start())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def stop_all(self):
        """Stop all collectors"""
        for collector in self.collectors.values():
            await collector.stop()
    
    def get_collector(self, name: str) -> Optional[BaseCollector]:
        """Get collector by name"""
        return self.collectors.get(name)


# Main entry point for running as standalone service
async def main():
    """Run data collectors as standalone service"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize components
    config = CollectorConfig()
    message_bus = MessageBus()
    
    # Create and start manager
    manager = CollectorManager(config, message_bus)
    
    try:
        logger.info("Starting data collectors...")
        await manager.start_all()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())