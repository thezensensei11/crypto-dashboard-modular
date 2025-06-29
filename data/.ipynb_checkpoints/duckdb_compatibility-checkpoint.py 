"""
Data models and types for the crypto dashboard
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

class MetricType(Enum):
    """Types of metrics available"""
    BETA = "beta"
    UPSIDE_BETA = "upside_beta"
    DOWNSIDE_BETA = "downside_beta"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"

class ColumnType(Enum):
    """Types of columns in the dashboard"""
    METRIC = "metric"
    CALCULATED = "calculated"
    PRICE = "price"

@dataclass
class MetricConfig:
    """Configuration for a metric calculation"""
    name: str
    metric: str
    interval: str
    lookback_days: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    params: Dict[str, Any] = None
    type: str = "metric"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'metric': self.metric,
            'interval': self.interval,
            'lookback_days': self.lookback_days,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'params': self.params or {},
            'type': self.type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MetricConfig':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            metric=data['metric'],
            interval=data['interval'],
            lookback_days=data.get('lookback_days'),
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            params=data.get('params', {}),
            type=data.get('type', 'metric')
        )

@dataclass
class CalculatedColumn:
    """Configuration for a calculated column"""
    name: str
    formula: str
    dependencies: List[str]
    type: str = "calculated"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'type': self.type,
            'formula': self.formula,
            'dependencies': self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CalculatedColumn':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            formula=data['formula'],
            dependencies=data.get('dependencies', []),
            type=data.get('type', 'calculated')
        )

@dataclass
class PriceData:
    """Live price data structure"""
    symbol: str
    price: float
    return_24h: Optional[float] = None
    volume_change_24h: Optional[float] = None
    volume_ratio_30d: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class CacheStats:
    """Cache statistics"""
    total_symbols: int
    total_files: int
    total_rows: int
    total_size_mb: float
    api_calls: int = 0
    cache_hits: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.api_calls + self.cache_hits
        return (self.cache_hits / total * 100) if total > 0 else 0

@dataclass
class FetchDiagnostics:
    """Performance diagnostics for data fetching"""
    start_time: float
    end_time: float
    total_api_calls: int
    total_cache_hits: int
    force_cache_mode: bool
    
    @property
    def duration(self) -> float:
        """Total duration in seconds"""
        return self.end_time - self.start_time
    
    @property
    def cache_efficiency(self) -> float:
        """Cache hit rate percentage"""
        total = self.total_api_calls + self.total_cache_hits
        return (self.total_cache_hits / total * 100) if total > 0 else 0
