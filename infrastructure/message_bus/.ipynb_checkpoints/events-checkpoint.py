"""
Redis-based Message Bus and Event System for Crypto Data Platform
Handles pub/sub, event streaming, and distributed task coordination
"""

import redis
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod
import pickle
import traceback

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in the system"""
    # Data collection events
    DATA_FETCH_REQUESTED = "data.fetch.requested"
    DATA_FETCH_COMPLETED = "data.fetch.completed"
    DATA_FETCH_FAILED = "data.fetch.failed"
    
    # WebSocket events
    PRICE_UPDATE = "price.update"
    ORDERBOOK_UPDATE = "orderbook.update"
    TRADE_UPDATE = "trade.update"
    
    # System events
    COLLECTOR_STARTED = "collector.started"
    COLLECTOR_STOPPED = "collector.stopped"
    DATABASE_UPDATED = "database.updated"
    
    # Alert events
    PRICE_ALERT = "alert.price"
    VOLUME_ALERT = "alert.volume"
    ERROR_ALERT = "alert.error"


@dataclass
class Event:
    """Base event structure"""
    id: str
    type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            type=EventType(data['type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            data=data['data'],
            metadata=data.get('metadata')
        )


class MessageBus:
    """
    Redis Streams-based message bus for event-driven architecture
    Supports both pub/sub and persistent event streams
    """
    
    def __init__(
        self, 
        redis_url: str = "redis://localhost:6379",
        stream_prefix: str = "crypto",
        max_stream_length: int = 10000
    ):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.stream_prefix = stream_prefix
        self.max_stream_length = max_stream_length
        self.subscribers = {}
        self._running = False
        
    def publish_event(self, event: Event) -> str:
        """
        Publish event to Redis Stream
        Returns message ID
        """
        stream_name = f"{self.stream_prefix}:{event.type.value}"
        
        # Add to stream with automatic trimming
        message_id = self.redis_client.xadd(
            stream_name,
            event.to_dict(),
            maxlen=self.max_stream_length,
            approximate=True
        )
        
        # Also publish to pub/sub for real-time subscribers
        self.redis_client.publish(
            f"{self.stream_prefix}:pubsub:{event.type.value}",
            json.dumps(event.to_dict())
        )
        
        logger.debug(f"Published event {event.id} to stream {stream_name}, ID: {message_id}")
        return message_id
    
    def subscribe(
        self, 
        event_types: List[EventType], 
        handler: Callable[[Event], None],
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None
    ):
        """
        Subscribe to event types
        If consumer_group is provided, uses consumer groups for load balancing
        """
        for event_type in event_types:
            stream_name = f"{self.stream_prefix}:{event_type.value}"
            
            if consumer_group:
                # Create consumer group if doesn't exist
                try:
                    self.redis_client.xgroup_create(
                        stream_name, 
                        consumer_group, 
                        id='0',
                        mkstream=True
                    )
                except redis.ResponseError:
                    # Group already exists
                    pass
            
            key = (event_type, consumer_group, consumer_name)
            if key not in self.subscribers:
                self.subscribers[key] = []
            self.subscribers[key].append(handler)
    
    async def start_consuming(self):
        """Start consuming events from streams"""
        self._running = True
        tasks = []
        
        # Group subscribers by consumer group
        grouped = {}
        for (event_type, group, consumer), handlers in self.subscribers.items():
            if group not in grouped:
                grouped[group] = []
            grouped[group].append((event_type, consumer, handlers))
        
        # Start consumer for each group
        for group, subscriptions in grouped.items():
            if group:
                task = asyncio.create_task(
                    self._consume_with_group(group, subscriptions)
                )
            else:
                task = asyncio.create_task(
                    self._consume_without_group(subscriptions)
                )
            tasks.append(task)
        
        # Also start pub/sub listener for real-time events
        tasks.append(asyncio.create_task(self._listen_pubsub()))
        
        await asyncio.gather(*tasks)
    
    async def _consume_with_group(self, group: str, subscriptions):
        """Consume from streams using consumer groups"""
        streams = {}
        handlers_map = {}
        
        for event_type, consumer, handlers in subscriptions:
            stream_name = f"{self.stream_prefix}:{event_type.value}"
            streams[stream_name] = '>'  # Read new messages
            handlers_map[stream_name] = handlers
        
        consumer_name = subscriptions[0][1] or "default"
        
        while self._running:
            try:
                # Read from multiple streams
                messages = self.redis_client.xreadgroup(
                    group,
                    consumer_name,
                    streams,
                    count=10,
                    block=1000  # Block for 1 second
                )
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        try:
                            event = Event.from_dict(data)
                            
                            # Call all handlers for this stream
                            for handler in handlers_map.get(stream_name, []):
                                await self._call_handler(handler, event)
                            
                            # Acknowledge message
                            self.redis_client.xack(stream_name, group, message_id)
                            
                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            # Could implement retry logic here
                            
            except Exception as e:
                logger.error(f"Error in consumer group {group}: {e}")
                await asyncio.sleep(1)
    
    async def _consume_without_group(self, subscriptions):
        """Consume from streams without consumer groups"""
        last_ids = {}
        handlers_map = {}
        
        for event_type, _, handlers in subscriptions:
            stream_name = f"{self.stream_prefix}:{event_type.value}"
            last_ids[stream_name] = '$'  # Start from latest
            handlers_map[stream_name] = handlers
        
        while self._running:
            try:
                # Read from multiple streams
                messages = self.redis_client.xread(
                    last_ids,
                    count=10,
                    block=1000
                )
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        try:
                            event = Event.from_dict(data)
                            
                            # Call all handlers
                            for handler in handlers_map.get(stream_name, []):
                                await self._call_handler(handler, event)
                            
                            # Update last ID
                            last_ids[stream_name] = message_id
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            
            except Exception as e:
                logger.error(f"Error in consumer: {e}")
                await asyncio.sleep(1)
    
    async def _listen_pubsub(self):
        """Listen to pub/sub for real-time events"""
        pubsub = self.redis_client.pubsub()
        
        # Subscribe to all event types we have handlers for
        channels = set()
        for (event_type, _, _), _ in self.subscribers.items():
            channel = f"{self.stream_prefix}:pubsub:{event_type.value}"
            channels.add(channel)
        
        if channels:
            pubsub.subscribe(*channels)
        
        while self._running:
            try:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    data = json.loads(message['data'])
                    event = Event.from_dict(data)
                    
                    # Find handlers for this event type
                    for (event_type, _, _), handlers in self.subscribers.items():
                        if event_type == event.type:
                            for handler in handlers:
                                await self._call_handler(handler, event)
                                
            except Exception as e:
                logger.error(f"Error in pub/sub listener: {e}")
                await asyncio.sleep(1)
    
    async def _call_handler(self, handler: Callable, event: Event):
        """Call handler with proper error handling"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                await asyncio.get_event_loop().run_in_executor(None, handler, event)
        except Exception as e:
            logger.error(f"Handler error for event {event.id}: {e}\n{traceback.format_exc()}")
    
    def stop(self):
        """Stop consuming events"""
        self._running = False
    
    def get_stream_info(self, event_type: EventType) -> Dict:
        """Get information about a stream"""
        stream_name = f"{self.stream_prefix}:{event_type.value}"
        info = self.redis_client.xinfo_stream(stream_name)
        return {
            'length': info['length'],
            'first_entry': info['first-entry'],
            'last_entry': info['last-entry'],
            'consumer_groups': self.redis_client.xinfo_groups(stream_name)
        }
    
    def replay_events(
        self, 
        event_type: EventType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        handler: Callable[[Event], None]
    ):
        """Replay historical events from stream"""
        stream_name = f"{self.stream_prefix}:{event_type.value}"
        
        start_id = '-' if not start_time else f"{int(start_time.timestamp() * 1000)}-0"
        end_id = '+' if not end_time else f"{int(end_time.timestamp() * 1000)}-0"
        
        # Read all events in range
        events = self.redis_client.xrange(stream_name, start_id, end_id)
        
        for message_id, data in events:
            try:
                event = Event.from_dict(data)
                handler(event)
            except Exception as e:
                logger.error(f"Error replaying event {message_id}: {e}")


class EventHandler(ABC):
    """Abstract base class for event handlers"""
    
    @abstractmethod
    async def handle(self, event: Event):
        """Handle the event"""
        pass
    
    @property
    @abstractmethod
    def event_types(self) -> List[EventType]:
        """Event types this handler processes"""
        pass


class DatabaseWriterHandler(EventHandler):
    """Handler that writes data to DuckDB"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    @property
    def event_types(self) -> List[EventType]:
        return [
            EventType.DATA_FETCH_COMPLETED,
            EventType.PRICE_UPDATE
        ]
    
    async def handle(self, event: Event):
        """Write data to database"""
        if event.type == EventType.DATA_FETCH_COMPLETED:
            # Write OHLCV data
            data = pd.DataFrame(event.data['candles'])
            rows = self.db_manager.insert_ohlcv_batch(
                data,
                event.data['symbol'],
                event.data['interval']
            )
            logger.info(f"Wrote {rows} rows for {event.data['symbol']}")
            
        elif event.type == EventType.PRICE_UPDATE:
            # Update live price
            self.db_manager.update_live_price(
                event.data['symbol'],
                event.data['price'],
                event.timestamp,
                event.data.get('bid'),
                event.data.get('ask')
            )


class NotificationHandler(EventHandler):
    """Handler for sending notifications"""
    
    @property
    def event_types(self) -> List[EventType]:
        return [
            EventType.PRICE_ALERT,
            EventType.VOLUME_ALERT,
            EventType.ERROR_ALERT
        ]
    
    async def handle(self, event: Event):
        """Send notification based on event"""
        # Implement notification logic (email, webhook, etc.)
        logger.info(f"Notification: {event.type.value} - {event.data}")


# Helper functions
def create_event(
    event_type: EventType,
    source: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """Create a new event with generated ID"""
    import uuid
    return Event(
        id=str(uuid.uuid4()),
        type=event_type,
        timestamp=datetime.utcnow(),
        source=source,
        data=data,
        metadata=metadata
    )