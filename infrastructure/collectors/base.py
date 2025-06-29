"""
Base collector class for all data collectors
Place in: crypto-dashboard/infrastructure/collectors/base.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from infrastructure.message_bus.bus import MessageBus

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors
    
    Provides common functionality and interface that all collectors must implement
    """
    
    def __init__(self):
        self.message_bus: Optional[MessageBus] = None
        self._stats = {
            "messages_sent": 0,
            "errors": 0,
            "started_at": None,
            "last_message_at": None
        }
    
    @abstractmethod
    async def start(self):
        """Start the collector"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the collector"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get collector status"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics"""
        return self._stats.copy()
    
    def increment_stat(self, stat: str, value: int = 1):
        """Increment a statistic counter"""
        if stat in self._stats and isinstance(self._stats[stat], (int, float)):
            self._stats[stat] += value
        else:
            self._stats[stat] = value


