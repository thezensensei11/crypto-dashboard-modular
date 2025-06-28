"""
Price and Volume Metrics Module
Extends the dashboard with additional price-related metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PriceMetrics:
    """Calculate price and volume related metrics from cached data"""
    
    @staticmethod
    def calculate_price_metrics(price_data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Calculate all price-related metrics from OHLCV data
        
        Args:
            price_data: DataFrame with OHLCV data (must have timestamp, close, volume columns)
            symbol: Symbol name for logging
            
        Returns:
            Dictionary with metrics: Price, 24h Return %, 24h Volume Change %, 24h Vol/30d Avg
        """
        metrics = {
            'Price': np.nan,
            '24h Return %': np.nan,
            '24h Volume Change %': np.nan,
            '24h Vol/30d Avg': np.nan  # New metric
        }
        
        if price_data is None or price_data.empty:
            logger.warning(f"No price data for {symbol}")
            return metrics
        
        try:
            # Current price (most recent close)
            if len(price_data) > 0:
                metrics['Price'] = price_data['close'].iloc[-1]
            
            # 24h return and volume change (assuming hourly data)
            if len(price_data) >= 24:
                # Price 24h ago
                price_24h_ago = price_data['close'].iloc[-24]
                price_return_24h = ((metrics['Price'] - price_24h_ago) / price_24h_ago) * 100
                metrics['24h Return %'] = price_return_24h
                
                # 24 hour volume (sum of last 24 hours)
                volume_24h = price_data['volume'].iloc[-24:].sum()
                
                # Volume change calculation
                # Compare last 24 hours volume to previous 24 hours
                if len(price_data) >= 48:
                    # Previous 24 hours total volume
                    previous_volume = price_data['volume'].iloc[-48:-24].sum()
                    
                    if previous_volume > 0:
                        volume_change = ((volume_24h - previous_volume) / previous_volume) * 100
                        metrics['24h Volume Change %'] = volume_change
                    else:
                        logger.warning(f"{symbol}: Previous 24h volume is 0, cannot calculate change")
                else:
                    logger.info(f"{symbol}: Not enough data for 24h volume comparison (need 48 hours)")
                
                # New metric: 24h volume / 30-day average daily volume
                # Since binance_data_collector now ensures 30 days minimum, we should have the data
                # For 30 days, we need 30 * 24 = 720 hours of data
                if len(price_data) >= 720:
                    # Get last 720 hours (30 days) of data
                    last_30_days = price_data.iloc[-720:]
                    
                    # Calculate daily volumes for the last 30 days
                    # Group by day and sum volumes
                    daily_volumes = []
                    for i in range(30):
                        start_idx = i * 24
                        end_idx = (i + 1) * 24
                        if end_idx <= len(last_30_days):
                            daily_volume = last_30_days['volume'].iloc[start_idx:end_idx].sum()
                            daily_volumes.append(daily_volume)
                    
                    if daily_volumes:
                        # Calculate average daily volume
                        avg_daily_volume_30d = np.mean(daily_volumes)
                        
                        if avg_daily_volume_30d > 0:
                            # Calculate the ratio
                            volume_ratio = volume_24h / avg_daily_volume_30d
                            metrics['24h Vol/30d Avg'] = volume_ratio
                            logger.debug(f"{symbol}: 24h volume={volume_24h:.2f}, 30d avg daily={avg_daily_volume_30d:.2f}, ratio={volume_ratio:.2f}")
                        else:
                            logger.warning(f"{symbol}: 30-day average daily volume is 0")
                    else:
                        logger.warning(f"{symbol}: Could not calculate daily volumes")
                else:
                    # If we have less than 30 days, use what we have
                    available_days = len(price_data) // 24
                    if available_days >= 2:  # Need at least 2 days for meaningful average
                        logger.info(f"{symbol}: Using {available_days} days for volume average (less than 30 available)")
                        
                        # Calculate daily volumes for available days
                        daily_volumes = []
                        for i in range(available_days):
                            start_idx = i * 24
                            end_idx = (i + 1) * 24
                            if end_idx <= len(price_data):
                                daily_volume = price_data['volume'].iloc[start_idx:end_idx].sum()
                                daily_volumes.append(daily_volume)
                        
                        if daily_volumes:
                            avg_daily_volume = np.mean(daily_volumes)
                            if avg_daily_volume > 0:
                                volume_ratio = volume_24h / avg_daily_volume
                                metrics['24h Vol/30d Avg'] = volume_ratio
                                logger.info(f"{symbol}: Calculated ratio with {available_days} days: {volume_ratio:.2f}")
                    else:
                        logger.info(f"{symbol}: Not enough data for volume ratio (need at least 48 hours, have {len(price_data)})")
            else:
                logger.info(f"{symbol}: Not enough data for 24h metrics (only {len(price_data)} hours)")
                
        except Exception as e:
            logger.error(f"Error calculating price metrics for {symbol}: {e}")
            import traceback
            traceback.print_exc()
        
        return metrics
    
    @staticmethod
    def calculate_volume_metrics(price_data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Calculate detailed volume metrics
        
        Args:
            price_data: DataFrame with OHLCV data
            symbol: Symbol name
            
        Returns:
            Dictionary with volume metrics
        """
        metrics = {
            'Current Volume': np.nan,
            '24h Avg Volume': np.nan,
            '24h Volume': np.nan,
            'Volume/Avg Ratio': np.nan
        }
        
        if price_data is None or price_data.empty or 'volume' not in price_data.columns:
            return metrics
        
        try:
            # Current volume (last candle)
            if len(price_data) > 0:
                metrics['Current Volume'] = price_data['volume'].iloc[-1]
            
            # 24h metrics
            if len(price_data) >= 24:
                last_24h_volume = price_data['volume'].iloc[-24:]
                metrics['24h Volume'] = last_24h_volume.sum()
                metrics['24h Avg Volume'] = last_24h_volume.mean()
                
                # Volume ratio (current vs average)
                if metrics['24h Avg Volume'] > 0:
                    metrics['Volume/Avg Ratio'] = metrics['Current Volume'] / metrics['24h Avg Volume']
                    
        except Exception as e:
            logger.error(f"Error calculating volume metrics for {symbol}: {e}")
        
        return metrics
    
    @staticmethod
    def format_price(price: float, symbol: str = None) -> str:
        """
        Format price with appropriate precision
        """
        if pd.isna(price):
            return "N/A"
        
        # Determine appropriate decimal places based on price magnitude
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 10:
            return f"${price:.3f}"
        elif price >= 1:
            return f"${price:.4f}"
        elif price >= 0.01:
            return f"${price:.5f}"
        else:
            # For very small prices, use scientific notation or more decimals
            return f"${price:.8f}".rstrip('0').rstrip('.')
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Format percentage with appropriate decimals and color"""
        if pd.isna(value):
            return "N/A"
        
        # Format with sign
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.2f}%"
    
    @staticmethod
    def format_volume(volume: float) -> str:
        """Format volume with appropriate units"""
        if pd.isna(volume):
            return "N/A"
        
        if volume >= 1e9:
            return f"{volume/1e9:.2f}B"
        elif volume >= 1e6:
            return f"{volume/1e6:.2f}M"
        elif volume >= 1e3:
            return f"{volume/1e3:.2f}K"
        else:
            return f"{volume:.0f}"
    
    @staticmethod
    def format_volume_ratio(ratio: float) -> str:
        """Format volume ratio"""
        if pd.isna(ratio):
            return "N/A"
        
        return f"{ratio:.2f}x"


def calculate_enhanced_metrics(collector, symbol: str, interval: str = '1h', 
                             lookback_days: int = 30, force_cache: bool = True) -> Dict[str, float]:
    """
    Calculate enhanced price and volume metrics for a symbol
    
    This is a convenience function that can be called from dashboard.py
    
    Args:
        collector: BinanceFuturesCollector instance
        symbol: Trading symbol
        interval: Candle interval (default: 1h)
        lookback_days: Days of data to fetch (default: 30 for volume comparisons)
        force_cache: Whether to force using cached data only
        
    Returns:
        Dictionary with all calculated metrics
    """
    try:
        # Get price data from cache - with new collector, this ensures 30 days minimum
        price_data = collector.get_klines(
            symbol=symbol,
            interval=interval,
            lookback_days=lookback_days,
            force_use_cache=force_cache
        )
        
        # Calculate metrics
        price_metrics = PriceMetrics()
        metrics = price_metrics.calculate_price_metrics(price_data, symbol)
        
        # Add volume metrics if needed
        volume_metrics = price_metrics.calculate_volume_metrics(price_data, symbol)
        
        # Combine all metrics
        all_metrics = {**metrics, **volume_metrics}
        
        return all_metrics
        
    except Exception as e:
        logger.error(f"Error calculating enhanced metrics for {symbol}: {e}")
        return {
            'Price': np.nan,
            '24h Return %': np.nan,
            '24h Volume Change %': np.nan,
            '24h Vol/30d Avg': np.nan
        }


# Integration helper for dashboard.py
def extend_row_with_price_metrics(row: Dict, symbol: str, collector, force_cache: bool = True) -> Dict:
    """
    Extend a metrics row with price and volume data
    
    This function is designed to be called from dashboard.py's calculate_metrics function
    
    Args:
        row: Existing row dictionary
        symbol: Trading symbol
        collector: BinanceFuturesCollector instance
        force_cache: Whether to use cached data only
        
    Returns:
        Updated row dictionary with price metrics
    """
    # Get enhanced metrics
    metrics = calculate_enhanced_metrics(
        collector=collector,
        symbol=symbol,
        interval='1h',
        lookback_days=30,  # Now we can use 30 days since collector ensures minimum cache
        force_cache=force_cache
    )
    
    # Update row with metrics
    row['Price'] = metrics.get('Price', np.nan)
    row['24h Return %'] = metrics.get('24h Return %', np.nan)
    row['24h Volume Change %'] = metrics.get('24h Volume Change %', np.nan)
    row['24h Vol/30d Avg'] = metrics.get('24h Vol/30d Avg', np.nan)
    
    return row