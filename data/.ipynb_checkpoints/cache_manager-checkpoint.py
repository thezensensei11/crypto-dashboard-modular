"""
Modified Smart data cache manager wrapper  
This version uses a local compatibility file to avoid import issues
Replace crypto-dashboard-modular/data/cache_manager.py with this
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import logging
import os

# Check if we should use DuckDB
USE_DUCKDB = os.environ.get('USE_DUCKDB', 'false').lower() == 'true'

if USE_DUCKDB:
    try:
        # Try to use local compatibility file first
        from .duckdb_compatibility import DuckDBCompatibleDataManager as _SmartDataManager
    except ImportError:
        # Fallback to infrastructure import with full path
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        
        from infrastructure.database.compatibility_layer import DuckDBCompatibleDataManager as _SmartDataManager
else:
    # Use original implementation
    from crypto_dashboard_modular.data.smart_cache import SmartDataManager as _SmartDataManager

from crypto_dashboard_modular.config.constants import CACHE_DIR

logger = logging.getLogger(__name__)

class SmartDataManager:
    """
    Clean interface wrapper for smart data management
    Now supports both original and DuckDB backends
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self._manager = _SmartDataManager(cache_dir)
        self.logger = logger
        
        if USE_DUCKDB:
            self.logger.info("Using DuckDB backend for data management")
    
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
        if hasattr(self._manager, 'identify_data_gaps'):
            return self._manager.identify_data_gaps(symbol, interval, start_date, end_date)
        else:
            # DuckDB doesn't have gap detection yet
            return []
    
    def get_inventory(self) -> Dict:
        """Get a summary of cached data"""
        if hasattr(self._manager, 'get_data_inventory'):
            return self._manager.get_data_inventory()
        else:
            # Return basic info for DuckDB
            return {
                'type': 'DuckDB',
                'location': 'crypto_data.duckdb'
            }
    
    def clear_all(self):
        """Clear all cached data"""
        self._manager.clear_all()
    
    def get_cache_info(self) -> Dict:
        """Get cache information"""
        if hasattr(self._manager, 'get_cache_info'):
            return self._manager.get_cache_info()
        else:
            return self.get_inventory()