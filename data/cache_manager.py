"""
Smart data cache manager wrapper
This wraps the existing SmartDataManager with a cleaner interface
"""

import pandas as pd
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the existing manager
from smart_data_manager import SmartDataManager as _SmartDataManager
from config.constants import CACHE_DIR

logger = logging.getLogger(__name__)

class SmartDataManager:
    """
    Clean interface wrapper for smart data management
    Delegates to the existing SmartDataManager
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self._manager = _SmartDataManager(cache_dir)
        self.logger = logger
    
    def save_data(self, symbol: str, interval: str, data: pd.DataFrame):
        """Save data to cache"""
        self._manager.save_data(symbol, interval, data)
    
    def load_data(
        self, 
        symbol: str, 
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Load cached data for a symbol/interval"""
        return self._manager.load_data(symbol, interval, start_date, end_date)
    
    def identify_gaps(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Identify what data needs to be fetched"""
        return self._manager.identify_data_gaps(symbol, interval, start_date, end_date)
    
    def get_inventory(self) -> Dict:
        """Get a summary of cached data"""
        return self._manager.get_data_inventory()
    
    def clear_all(self):
        """Clear all cached data"""
        self._manager.clear_all_cache()
    
    def verify_integrity(self, symbol: str, interval: str) -> Dict:
        """Verify data integrity for a symbol/interval"""
        return self._manager.verify_data_integrity(symbol, interval)
    
    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get the latest timestamp for cached data"""
        return self._manager.get_latest_timestamp(symbol, interval)
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory path"""
        return self._manager.cache_dir
