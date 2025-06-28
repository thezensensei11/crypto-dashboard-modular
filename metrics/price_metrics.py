"""
Price and volume metrics module
Clean interface for price-related calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing price metrics
from price_metrics import PriceMetrics as _PriceMetrics

logger = logging.getLogger(__name__)

class PriceMetrics:
    """Calculate price and volume related metrics"""
    
    @staticmethod
    def calculate_all(price_data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Calculate all price-related metrics
        
        Args:
            price_data: DataFrame with OHLCV data
            symbol: Symbol name for logging
            
        Returns:
            Dictionary with all price metrics
        """
        return _PriceMetrics.calculate_price_metrics(price_data, symbol)
    
    @staticmethod
    def calculate_return_24h(price_data: pd.DataFrame) -> float:
        """Calculate 24-hour return percentage"""
        if price_data is None or len(price_data) < 24:
            return np.nan
        
        current_price = price_data['close'].iloc[-1]
        price_24h_ago = price_data['close'].iloc[-24]
        
        return ((current_price - price_24h_ago) / price_24h_ago) * 100
    
    @staticmethod
    def calculate_volume_change_24h(price_data: pd.DataFrame) -> float:
        """Calculate 24-hour volume change percentage"""
        if price_data is None or len(price_data) < 48:
            return np.nan
        
        volume_24h = price_data['volume'].iloc[-24:].sum()
        previous_volume = price_data['volume'].iloc[-48:-24].sum()
        
        if previous_volume == 0:
            return np.nan
        
        return ((volume_24h - previous_volume) / previous_volume) * 100
    
    @staticmethod
    def calculate_volume_ratio_30d(price_data: pd.DataFrame) -> float:
        """Calculate 24h volume / 30-day average daily volume"""
        if price_data is None or len(price_data) < 24:
            return np.nan
        
        # 24h volume
        volume_24h = price_data['volume'].iloc[-24:].sum()
        
        # Calculate average daily volume
        if len(price_data) >= 720:  # 30 days * 24 hours
            # Full 30 days available
            daily_volumes = []
            for i in range(30):
                start_idx = len(price_data) - (30 - i) * 24
                end_idx = start_idx + 24
                if end_idx <= len(price_data):
                    daily_volume = price_data['volume'].iloc[start_idx:end_idx].sum()
                    daily_volumes.append(daily_volume)
        else:
            # Use available days
            available_days = len(price_data) // 24
            if available_days < 2:
                return np.nan
            
            daily_volumes = []
            for i in range(available_days):
                start_idx = i * 24
                end_idx = (i + 1) * 24
                daily_volume = price_data['volume'].iloc[start_idx:end_idx].sum()
                daily_volumes.append(daily_volume)
        
        if not daily_volumes:
            return np.nan
        
        avg_daily_volume = np.mean(daily_volumes)
        
        return volume_24h / avg_daily_volume if avg_daily_volume > 0 else np.nan
    
    @staticmethod
    def get_current_price(price_data: pd.DataFrame) -> float:
        """Get the current (most recent) price"""
        if price_data is None or price_data.empty:
            return np.nan
        
        return price_data['close'].iloc[-1]
