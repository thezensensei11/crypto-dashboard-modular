
# Place in: crypto-dashboard/data/infrastructure_adapter.py
"""
Adapter to make new infrastructure compatible with existing dashboard
"""

import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from infrastructure.database.manager import get_db_manager
from infrastructure.collectors.rest import RESTCollector
from core.config import get_settings

logger = logging.getLogger(__name__)


class InfrastructureAdapter:
    """
    Adapter class that mimics the old BinanceDataCollector interface
    but uses the new infrastructure underneath
    """
    
    def __init__(self):
        self.logger = logger
        self.db_manager = get_db_manager()
        self.settings = get_settings()
        
        # Compatibility attributes
        self.api_call_count = 0
        self.cache_hit_count = 0
        
        # For async operations
        self._loop = None
        self._rest_collector = None
    
    def _get_event_loop(self):
        """Get or create event loop"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def get_price_data(
        self, 
        symbol: str, 
        interval: str = '1h',
        lookback_days: int = 30,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get price data - compatible with old interface
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        
        # First try to get from database
        df = self.db_manager.read_ohlcv_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if not df.empty:
            self.cache_hit_count += 1
            return self._format_dataframe(df)
        
        # If no data or force refresh, fetch from API
        if force_refresh or df.empty:
            logger.info(f"Fetching fresh data for {symbol} {interval}")
            
            # Run async collector in sync context
            loop = self._get_event_loop()
            
            async def fetch():
                collector = RESTCollector()
                await collector.start()
                
                try:
                    count = await collector.collect_historical_data(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time
                    )
                    logger.info(f"Collected {count} candles")
                finally:
                    await collector.stop()
            
            if loop.is_running():
                # If called from async context
                task = asyncio.create_task(fetch())
                loop.run_until_complete(task)
            else:
                # If called from sync context
                loop.run_until_complete(fetch())
            
            self.api_call_count += 1
            
            # Wait a moment for data to be processed
            import time
            time.sleep(2)
            
            # Fetch again from database
            df = self.db_manager.read_ohlcv_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
        
        return self._format_dataframe(df)
    
    def get_price_data_batch(
        self,
        symbols: List[str],
        interval: str = '1h',
        lookback_days: int = 30,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get price data for multiple symbols - compatible with old interface
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.get_price_data(
                    symbol=symbol,
                    interval=interval,
                    lookback_days=lookback_days,
                    force_refresh=force_refresh
                )
                results[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    async def get_price_data_batch_async(
        self,
        symbols: List[str],
        interval: str = '1h',
        lookback_days: int = 30,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Async version for compatibility"""
        # Just wrap the sync version
        return self.get_price_data_batch(
            symbols=symbols,
            interval=interval,
            lookback_days=lookback_days,
            force_refresh=force_refresh
        )
    
    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format dataframe to match expected structure"""
        if df.empty:
            return df
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add any missing columns that the old interface had
        if 'close_time' not in df.columns:
            df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=1)
        
        return df
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics - compatible with old interface"""
        db_stats = self.db_manager.get_data_stats()
        
        return {
            'api_calls': self.api_call_count,
            'cache_hits': self.cache_hit_count,
            'db_size_mb': db_stats.get('db_size_mb', 0),
            'total_records': sum(
                db_stats.get(f"{interval}_count", 0) 
                for interval in ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
            ),
            'symbols': db_stats.get('symbol_count', 0)
        }
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols"""
        return self.db_manager.get_symbols()
    
    def clear_cache(self):
        """Clear cache - no-op for compatibility"""
        logger.info("Clear cache called - no-op in new infrastructure")
        pass
