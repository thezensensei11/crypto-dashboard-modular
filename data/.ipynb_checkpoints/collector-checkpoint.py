"""
Binance data collector wrapper
This wraps the existing BinanceFuturesCollector with a cleaner interface
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

# Import the existing collector
from binance_data_collector import BinanceFuturesCollector
from config.constants import CACHE_DIR

logger = logging.getLogger(__name__)

class BinanceDataCollector:
    """
    Clean interface wrapper for Binance data collection
    Delegates to the existing BinanceFuturesCollector
    """
    
    def __init__(self):
        self._collector = BinanceFuturesCollector()
        self.logger = logger
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        try:
            symbols_df = self._collector.get_exchange_info()
            if not symbols_df.empty:
                return sorted(symbols_df['symbol'].tolist())
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {e}")
        return []
    
    def get_price_data(
        self,
        symbol: str,
        interval: str = '1h',
        lookback_days: int = 30,
        force_cache: bool = False
    ) -> pd.DataFrame:
        """
        Get price data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Candle interval
            lookback_days: Number of days to look back
            force_cache: If True, only use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        return self._collector.get_klines_smart(
            symbol=symbol,
            interval=interval,
            lookback_days=lookback_days,
            force_use_cache=force_cache
        )
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self._collector.get_cache_stats()
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data"""
        self._collector.clear_cache(symbol)
    
    def reset_counters(self):
        """Reset API call counters"""
        self._collector.reset_counters()
    
    @property
    def api_call_count(self) -> int:
        """Get number of API calls made"""
        return self._collector.api_call_count
    
    @property
    def cache_hit_count(self) -> int:
        """Get number of cache hits"""
        return self._collector.cache_hit_count
