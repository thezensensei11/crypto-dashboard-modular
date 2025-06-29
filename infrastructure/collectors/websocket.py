"""
WebSocket data collector for live streaming data
Place in: crypto-dashboard/infrastructure/collectors/websocket.py
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any
from decimal import Decimal

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.config import get_settings
from core.models import OHLCVData, Event, EventType, DataSource
from core.exceptions import CollectorException
from infrastructure.message_bus.bus import get_message_bus, publish_event
from .base import BaseCollector

logger = logging.getLogger(__name__)


class WebSocketCollector(BaseCollector):
    """
    WebSocket collector for live market data from Binance
    
    Features:
    - Auto-reconnection with exponential backoff
    - Multiple symbol subscription management
    - Real-time event publishing to message bus
    - Connection health monitoring
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.config = self.settings.collector
        self.binance_config = self.settings.binance
        
        # WebSocket state
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._subscriptions: Set[str] = set()
        self._running = False
        self._reconnect_count = 0
        self._last_ping_time = datetime.now(timezone.utc)
        
        # Subscription management
        self._symbol_streams: Dict[str, Set[str]] = {}
        
    async def start(self):
        """Start the WebSocket collector"""
        logger.info("Starting WebSocket collector...")
        self._running = True
        
        # Initialize message bus
        self.message_bus = await get_message_bus()
        
        # Publish startup event
        await publish_event(
            EventType.COLLECTOR_STARTED,
            source="WebSocketCollector",
            data={"collector_type": "websocket"}
        )
        
        # Start collection loop
        await self._collection_loop()
    
    async def stop(self):
        """Stop the WebSocket collector"""
        logger.info("Stopping WebSocket collector...")
        self._running = False
        
        # Close WebSocket connection
        if self._websocket:
            await self._websocket.close()
        
        # Publish shutdown event
        await publish_event(
            EventType.COLLECTOR_STOPPED,
            source="WebSocketCollector",
            data={"collector_type": "websocket"}
        )
    
    async def subscribe_symbols(self, symbols: List[str], streams: Optional[List[str]] = None):
        """Subscribe to symbols and streams"""
        if streams is None:
            streams = ["kline_1m", "miniTicker"]  # Default streams
        
        for symbol in symbols:
            symbol_lower = symbol.lower()
            if symbol_lower not in self._symbol_streams:
                self._symbol_streams[symbol_lower] = set()
            
            for stream in streams:
                self._symbol_streams[symbol_lower].add(stream)
                self._subscriptions.add(f"{symbol_lower}@{stream}")
        
        # Reconnect to apply new subscriptions
        if self._websocket:
            await self._websocket.close()
    
    async def unsubscribe_symbols(self, symbols: List[str], streams: Optional[List[str]] = None):
        """Unsubscribe from symbols and streams"""
        for symbol in symbols:
            symbol_lower = symbol.lower()
            
            if streams:
                for stream in streams:
                    if symbol_lower in self._symbol_streams:
                        self._symbol_streams[symbol_lower].discard(stream)
                    self._subscriptions.discard(f"{symbol_lower}@{stream}")
            else:
                # Remove all streams for symbol
                if symbol_lower in self._symbol_streams:
                    del self._symbol_streams[symbol_lower]
                self._subscriptions = {
                    sub for sub in self._subscriptions 
                    if not sub.startswith(f"{symbol_lower}@")
                }
        
        # Reconnect to apply subscription changes
        if self._websocket:
            await self._websocket.close()
    
    async def _collection_loop(self):
        """Main collection loop with auto-reconnection"""
        while self._running:
            try:
                # Connect to WebSocket
                await self._connect()
                
                # Reset reconnect count on successful connection
                self._reconnect_count = 0
                
                # Listen for messages
                await self._listen()
                
            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            
            if self._running:
                # Calculate reconnect delay with exponential backoff
                delay = min(
                    self.config.ws_reconnect_interval * (2 ** self._reconnect_count),
                    60  # Max 60 seconds
                )
                logger.info(f"Reconnecting in {delay} seconds...")
                await asyncio.sleep(delay)
                self._reconnect_count += 1
    
    async def _connect(self):
        """Connect to Binance WebSocket"""
        # Build subscription list
        if not self._subscriptions:
            # Default subscriptions if none specified
            for symbol in self.config.symbols[:5]:  # Top 5 symbols
                self._subscriptions.add(f"{symbol.lower()}@kline_1m")
                self._subscriptions.add(f"{symbol.lower()}@miniTicker")
        
        # Build WebSocket URL with streams
        streams = "/".join(sorted(self._subscriptions))
        ws_url = f"{self.binance_config.ws_futures_url}/stream?streams={streams}"
        
        logger.info(f"Connecting to WebSocket with {len(self._subscriptions)} subscriptions")
        
        # Connect with configuration
        self._websocket = await websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        )
        
        logger.info("WebSocket connected successfully")
    
    async def _listen(self):
        """Listen for WebSocket messages"""
        async for message in self._websocket:
            if not self._running:
                break
            
            try:
                data = json.loads(message)
                await self._process_message(data)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process WebSocket message and publish events"""
        # Handle stream data
        if 'stream' in data and 'data' in data:
            stream_name = data['stream']
            stream_data = data['data']
            
            # Parse stream type
            if '@kline' in stream_name:
                await self._process_kline(stream_data)
            elif '@miniTicker' in stream_name:
                await self._process_mini_ticker(stream_data)
            elif '@trade' in stream_name:
                await self._process_trade(stream_data)
    
    async def _process_kline(self, data: Dict[str, Any]):
        """Process kline (candlestick) data"""
        try:
            kline = data['k']
            
            # Create OHLCV data
            ohlcv = OHLCVData(
                symbol=data['s'],
                interval=kline['i'],
                timestamp=datetime.fromtimestamp(kline['t'] / 1000),
                open=Decimal(kline['o']),
                high=Decimal(kline['h']),
                low=Decimal(kline['l']),
                close=Decimal(kline['c']),
                volume=Decimal(kline['v']),
                quote_volume=Decimal(kline['q']),
                trades=int(kline['n']),
                taker_buy_base=Decimal(kline['V']),
                taker_buy_quote=Decimal(kline['Q']),
                source=DataSource.WEBSOCKET,
                received_at=datetime.now(timezone.utc)
            )
            
            # Publish event
            event_type = EventType.CANDLE_CLOSED if kline['x'] else EventType.PRICE_UPDATE
            
            await publish_event(
                event_type,
                source="WebSocketCollector",
                data={
                    "ohlcv": ohlcv.dict(),
                    "is_closed": kline['x']
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    async def _process_mini_ticker(self, data: Dict[str, Any]):
        """Process mini ticker data"""
        try:
            # Extract relevant fields with proper field names
            ticker_data = {
                "symbol": data.get('s', ''),
                "close": Decimal(str(data.get('c', '0'))),
                "open": Decimal(str(data.get('o', '0'))),
                "high": Decimal(str(data.get('h', '0'))),
                "low": Decimal(str(data.get('l', '0'))),
                "volume": Decimal(str(data.get('v', '0'))),
                "quote_volume": Decimal(str(data.get('q', '0'))),
                "timestamp": datetime.fromtimestamp(data.get('E', 0) / 1000),
                "price_change": Decimal(str(data.get('P', '0'))),  # Uppercase P
                "price_change_percent": Decimal(str(data.get('P', '0')))  # Same field
            }
            
            # Publish price update event
            await publish_event(
                EventType.PRICE_UPDATE,
                source="WebSocketCollector",
                data={"ticker": ticker_data}
            )
            
        except Exception as e:
            logger.error(f"Error processing mini ticker data: {e}")
            logger.debug(f"Data received: {data}")  # Log the actual data for debugging

    async def _process_trade(self, data: Dict[str, Any]):
        """Process trade data"""
        try:
            trade_data = {
                "symbol": data['s'],
                "price": Decimal(data['p']),
                "quantity": Decimal(data['q']),
                "timestamp": datetime.fromtimestamp(data['T'] / 1000),
                "is_buyer_maker": data['m']
            }
            
            # Publish trade event
            await publish_event(
                EventType.TRADE_UPDATE,
                source="WebSocketCollector",
                data={"trade": trade_data}
            )
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get collector status"""
        return {
            "running": self._running,
            "connected": self._websocket is not None and not self._websocket.closed,
            "subscriptions": len(self._subscriptions),
            "symbols": len(self._symbol_streams),
            "reconnect_count": self._reconnect_count,
            "last_ping": self._last_ping_time.isoformat()
        }
