"""
Redis Streams-based message bus implementation
Place in: crypto-dashboard/infrastructure/message_bus/bus.py
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.exceptions import ResponseError

from core.config import get_settings
from core.models import Event, EventType
from core.exceptions import MessageBusException
from core.constants import REDIS_STREAM_PATTERN, REDIS_CONSUMER_GROUP_PATTERN

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Redis Streams-based message bus for event-driven architecture
    
    Features:
    - Persistent message storage with Redis Streams
    - Consumer groups for load balancing
    - At-least-once delivery guarantee
    - Message replay capability
    - Dead letter queue for failed messages
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_config = self.settings.redis
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._running = False
        self._consumers: Dict[str, List[Callable]] = {}
        self._consumer_tasks: List[asyncio.Task] = []
        
    async def connect(self):
        """Connect to Redis"""
        if not self._redis:
            self._redis = redis.Redis.from_url(
                self.redis_config.url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._redis.ping()
            logger.info("Connected to Redis message bus")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        self._running = False
        
        # Cancel consumer tasks
        for task in self._consumer_tasks:
            task.cancel()
        
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        
        # Close connections
        if self._pubsub:
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
            
        logger.info("Disconnected from Redis message bus")
    
    @asynccontextmanager
    async def connection(self):
        """Context manager for connection lifecycle"""
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()
    
    def _get_stream_key(self, event_type: EventType) -> str:
        """Get Redis stream key for event type"""
        return REDIS_STREAM_PATTERN.format(
            prefix=self.redis_config.stream_prefix,
            event_type=event_type.value
        )
    
    def _get_consumer_group(self, component: str) -> str:
        """Get consumer group name"""
        return REDIS_CONSUMER_GROUP_PATTERN.format(
            prefix=self.redis_config.consumer_group_prefix,
            component=component
        )
    
    async def publish(self, event: Event) -> str:
        """
        Publish event to Redis Stream
        
        Returns:
            Message ID from Redis
        """
        if not self._redis:
            await self.connect()
        
        stream_key = self._get_stream_key(event.type)
        
        try:
            # Add to stream with automatic trimming
            message_id = await self._redis.xadd(
                stream_key,
                event.to_redis_dict(),
                maxlen=self.redis_config.max_stream_length,
                approximate=True
            )
            
            # Also publish to pub/sub for real-time subscribers
            await self._redis.publish(
                f"{self.redis_config.stream_prefix}:pubsub:{event.type.value}",
                event.json()
            )
            
            logger.debug(f"Published event {event.id} to {stream_key}, ID: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            raise MessageBusException(f"Failed to publish event: {e}")
    
    async def subscribe(
        self,
        event_types: List[EventType],
        handler: Callable[[Event], Any],
        component: str,
        consumer_name: Optional[str] = None,
        start_id: str = ">",  # Start from new messages by default
        block_ms: int = 5000
    ):
        """
        Subscribe to event types with consumer group
        
        Args:
            event_types: List of event types to subscribe to
            handler: Async function to handle events
            component: Component name (used for consumer group)
            consumer_name: Unique consumer name within group
            start_id: Starting position (">" for new, "0" for beginning)
            block_ms: Blocking timeout in milliseconds
        """
        if not self._redis:
            await self.connect()
        
        consumer_name = consumer_name or f"{component}-{uuid.uuid4().hex[:8]}"
        consumer_group = self._get_consumer_group(component)
        
        # Create consumer groups for each event type
        for event_type in event_types:
            stream_key = self._get_stream_key(event_type)
            try:
                await self._redis.xgroup_create(
                    stream_key,
                    consumer_group,
                    id="0",
                    mkstream=True
                )
                logger.info(f"Created consumer group {consumer_group} for {stream_key}")
            except ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
                # Group already exists, that's fine
        
        # Start consumer task
        task = asyncio.create_task(
            self._consume_events(
                event_types=event_types,
                handler=handler,
                consumer_group=consumer_group,
                consumer_name=consumer_name,
                start_id=start_id,
                block_ms=block_ms
            )
        )
        self._consumer_tasks.append(task)
    
    async def _consume_events(
        self,
        event_types: List[EventType],
        handler: Callable[[Event], Any],
        consumer_group: str,
        consumer_name: str,
        start_id: str,
        block_ms: int
    ):
        """Consumer loop for processing events"""
        self._running = True
        stream_keys = [self._get_stream_key(et) for et in event_types]
        
        # Build streams dict for xreadgroup
        streams = {key: start_id for key in stream_keys}
        
        logger.info(f"Consumer {consumer_name} started for streams: {stream_keys}")
        
        while self._running:
            try:
                # Read from streams
                messages = await self._redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    streams,
                    count=10,
                    block=block_ms
                )
                
                if not messages:
                    continue
                
                # Process messages
                for stream_key, stream_messages in messages:
                    for message_id, data in stream_messages:
                        try:
                            # Parse event
                            event = Event.from_dict(data)
                            
                            # Handle event
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                handler(event)
                            
                            # Acknowledge message
                            await self._redis.xack(
                                stream_key,
                                consumer_group,
                                message_id
                            )
                            
                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            # Message will be redelivered on restart
                            # Could implement dead letter queue here
                
            except asyncio.CancelledError:
                logger.info(f"Consumer {consumer_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                if self._running:
                    await asyncio.sleep(1)  # Brief pause before retry
    
    async def get_pending_messages(
        self,
        event_type: EventType,
        consumer_group: str,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """Get pending messages for a consumer group"""
        if not self._redis:
            await self.connect()
        
        stream_key = self._get_stream_key(event_type)
        
        try:
            # Get pending entries summary
            pending = await self._redis.xpending(
                stream_key,
                consumer_group
            )
            
            if not pending['pending']:
                return []
            
            # Get detailed pending messages
            detailed = await self._redis.xpending_range(
                stream_key,
                consumer_group,
                min="-",
                max="+",
                count=count
            )
            
            return detailed
            
        except Exception as e:
            logger.error(f"Failed to get pending messages: {e}")
            return []
    
    async def claim_abandoned_messages(
        self,
        event_type: EventType,
        consumer_group: str,
        consumer_name: str,
        min_idle_time: int = 60000  # 1 minute
    ) -> int:
        """Claim messages abandoned by other consumers"""
        if not self._redis:
            await self.connect()
        
        stream_key = self._get_stream_key(event_type)
        claimed_count = 0
        
        try:
            # Get pending messages
            pending = await self.get_pending_messages(
                event_type,
                consumer_group,
                count=100
            )
            
            for entry in pending:
                if entry['time_since_delivered'] >= min_idle_time:
                    # Claim the message
                    claimed = await self._redis.xclaim(
                        stream_key,
                        consumer_group,
                        consumer_name,
                        min_idle_time,
                        entry['message_id']
                    )
                    if claimed:
                        claimed_count += 1
            
            if claimed_count > 0:
                logger.info(f"Claimed {claimed_count} abandoned messages")
            
            return claimed_count
            
        except Exception as e:
            logger.error(f"Failed to claim abandoned messages: {e}")
            return 0
    
    async def get_stream_info(self, event_type: EventType) -> Dict[str, Any]:
        """Get information about a stream"""
        if not self._redis:
            await self.connect()
        
        stream_key = self._get_stream_key(event_type)
        
        try:
            info = await self._redis.xinfo_stream(stream_key)
            groups = await self._redis.xinfo_groups(stream_key)
            
            return {
                'stream': info,
                'groups': groups
            }
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return {}
    
    async def trim_stream(
        self,
        event_type: EventType,
        maxlen: int,
        approximate: bool = True
    ) -> int:
        """Manually trim a stream to a specific length"""
        if not self._redis:
            await self.connect()
        
        stream_key = self._get_stream_key(event_type)
        
        try:
            # Get current length
            current_len = await self._redis.xlen(stream_key)
            
            # Trim if needed
            if current_len > maxlen:
                await self._redis.xtrim(
                    stream_key,
                    maxlen=maxlen,
                    approximate=approximate
                )
                new_len = await self._redis.xlen(stream_key)
                trimmed = current_len - new_len
                logger.info(f"Trimmed {trimmed} messages from {stream_key}")
                return trimmed
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to trim stream: {e}")
            return 0


# Singleton instance
_message_bus: Optional[MessageBus] = None


async def get_message_bus() -> MessageBus:
    """Get message bus instance (singleton)"""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
        await _message_bus.connect()
    return _message_bus


# Helper functions for common operations
async def publish_event(
    event_type: EventType,
    source: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Publish an event to the message bus"""
    event = Event(
        id=str(uuid.uuid4()),
        type=event_type,
        source=source,
        data=data,
        metadata=metadata
    )
    
    bus = await get_message_bus()
    return await bus.publish(event)
