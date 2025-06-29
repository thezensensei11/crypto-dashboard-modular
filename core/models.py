

# Paste the code below this line:

"""
Core data models and constants
Place in: crypto-dashboard/core/models.py
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, validator


class Interval(str, Enum):
    """Standard trading intervals"""
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    
    @property
    def seconds(self) -> int:
        """Get interval in seconds"""
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800
        }
        return mapping[self.value]
    
    @property
    def minutes(self) -> int:
        """Get interval in minutes"""
        return self.seconds // 60
    
    @classmethod
    def is_standard(cls, interval: str) -> bool:
        """Check if interval is standard"""
        return interval in [i.value for i in cls]


class EventType(str, Enum):
    """System event types"""
    # Data events
    PRICE_UPDATE = "price.update"
    CANDLE_CLOSED = "candle.closed"
    HISTORICAL_DATA = "historical.data"
    
    # Processing events
    DATA_VALIDATED = "data.validated"
    DATA_INVALID = "data.invalid"
    AGGREGATION_NEEDED = "aggregation.needed"
    AGGREGATION_COMPLETED = "aggregation.completed"
    
    # Database events
    DB_WRITE_STARTED = "db.write.started"
    DB_WRITE_COMPLETED = "db.write.completed"
    DB_WRITE_FAILED = "db.write.failed"
    
    # System events
    COLLECTOR_STARTED = "collector.started"
    COLLECTOR_STOPPED = "collector.stopped"
    PROCESSOR_STARTED = "processor.started"
    PROCESSOR_STOPPED = "processor.stopped"
    
    # Error events
    ERROR_OCCURRED = "error.occurred"
    CONNECTION_LOST = "connection.lost"
    CONNECTION_RESTORED = "connection.restored"


class DataSource(str, Enum):
    """Data source types"""
    WEBSOCKET = "websocket"
    REST_API = "rest_api"
    DATABASE = "database"
    AGGREGATOR = "aggregator"


class OHLCVData(BaseModel):
    """Standard OHLCV data model"""
    symbol: str
    interval: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Optional[Decimal] = None
    trades: Optional[int] = None
    taker_buy_base: Optional[Decimal] = None
    taker_buy_quote: Optional[Decimal] = None
    
    # Metadata
    source: Optional[DataSource] = None
    received_at: Optional[datetime] = None
    
    @validator('open', 'high', 'low', 'close', 'volume', pre=True)
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal"""
        if v is not None:
            return Decimal(str(v))
        return v
    
    @validator('interval')
    def validate_interval(cls, v):
        """Validate interval format"""
        if not Interval.is_standard(v):
            # Allow custom intervals but log warning
            pass
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = self.dict(exclude={'source', 'received_at'})
        # Convert Decimal to float for storage
        for key in ['open', 'high', 'low', 'close', 'volume', 
                    'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
            if data.get(key) is not None:
                data[key] = float(data[key])
        return data
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat()
        }


class Event(BaseModel):
    """System event model"""
    id: str = Field(..., description="Unique event ID")
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(..., description="Event source (component name)")
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_redis_dict(self) -> Dict[str, str]:
        """Convert to Redis-compatible dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.json(exclude={'metadata'}),
            'metadata': self.json(include={'metadata'}) if self.metadata else '{}'
        }


class BatchData(BaseModel):
    """Batch of OHLCV data"""
    symbol: str
    interval: str
    data: List[OHLCVData]
    source: DataSource
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def count(self) -> int:
        """Number of records in batch"""
        return len(self.data)
    
    @property
    def time_range(self) -> Optional[tuple]:
        """Get time range of batch"""
        if not self.data:
            return None
        timestamps = [d.timestamp for d in self.data]
        return min(timestamps), max(timestamps)


class ValidationResult(BaseModel):
    """Data validation result"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data: Optional[OHLCVData] = None
    
    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)


class AggregationRequest(BaseModel):
    """Request to aggregate data"""
    symbol: str
    source_interval: str
    target_interval: str
    start_time: datetime
    end_time: datetime
    
    @validator('target_interval')
    def validate_aggregation(cls, v, values):
        """Validate aggregation is possible"""
        source = values.get('source_interval')
        if source and v:
            source_seconds = Interval(source).seconds
            target_seconds = Interval(v).seconds
            if target_seconds <= source_seconds:
                raise ValueError(f"Cannot aggregate {source} to {v}")
        return v


class ProcessingStats(BaseModel):
    """Processing statistics"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processed_count: int = 0
    validated_count: int = 0
    invalid_count: int = 0
    written_count: int = 0
    error_count: int = 0
    processing_time_ms: Optional[float] = None
    
    def increment(self, field: str, value: int = 1):
        """Increment a counter"""
        current = getattr(self, field, 0)
        setattr(self, field, current + value)


