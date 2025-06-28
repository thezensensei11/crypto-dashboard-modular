import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import logging

@dataclass
class MetricConfig:
    """Configuration for a metric calculation"""
    name: str
    metric_type: str
    interval: str
    lookback_days: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    params: Dict = None

class MetricsEngine:
    """Engine for calculating various trading metrics using smart caching"""
    
    def __init__(self, data_collector):
        self.collector = data_collector
        self.logger = logging.getLogger(__name__)
        self.benchmark_symbol = 'BTCUSDT'
        
    def calculate_single_metric(self, symbol: str, metric_name: str, interval: str,
                              lookback_days: Optional[int] = None,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              force_use_cache: bool = False,
                              **kwargs) -> float:
        """
        Calculate a single metric for a symbol
        
        Args:
            force_use_cache: If True, only uses cached data without fetching new data
        """
        
        try:
            self.logger.info(f"Calculating {metric_name} for {symbol} with force_use_cache={force_use_cache}")
            
            # Get data for the symbol using smart caching
            symbol_data = self.collector.get_klines(
                symbol=symbol,
                interval=interval,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date,
                force_use_cache=force_use_cache
            )
            
            if symbol_data is None or symbol_data.empty:
                self.logger.warning(f"No data available for {symbol} {interval}")
                return np.nan
            
            # For metrics that need benchmark
            if metric_name in ['beta', 'upside_beta', 'downside_beta', 'correlation',
                              'conditional_upside_beta', 'conditional_downside_beta', 'squared_residual']:
                benchmark_data = self.collector.get_klines(
                    symbol=self.benchmark_symbol,
                    interval=interval,
                    lookback_days=lookback_days,
                    start_date=start_date,
                    end_date=end_date,
                    force_use_cache=force_use_cache
                )
                
                if benchmark_data is None or benchmark_data.empty:
                    self.logger.warning(f"No benchmark data available for {interval}")
                    return np.nan
                
                # Align data
                symbol_returns, benchmark_returns = self._align_returns(symbol_data, benchmark_data)
                
                if len(symbol_returns) < 2:
                    return np.nan
                
                # Calculate specific metric
                if metric_name == 'beta':
                    return self._calculate_beta(symbol_returns, benchmark_returns)
                elif metric_name == 'upside_beta':
                    return self._calculate_upside_beta(symbol_returns, benchmark_returns)
                elif metric_name == 'downside_beta':
                    return self._calculate_downside_beta(symbol_returns, benchmark_returns)
                elif metric_name == 'correlation':
                    return self._calculate_correlation(symbol_returns, benchmark_returns)
                elif metric_name == 'conditional_upside_beta':
                    percentile = kwargs.get('percentile', 10)
                    return self._calculate_conditional_upside_beta(symbol_returns, benchmark_returns, percentile)
                elif metric_name == 'conditional_downside_beta':
                    percentile = kwargs.get('percentile', 10)
                    return self._calculate_conditional_downside_beta(symbol_returns, benchmark_returns, percentile)
                elif metric_name == 'squared_residual':
                    return self._calculate_squared_residual(symbol_returns, benchmark_returns)
            
            # For standalone metrics
            elif metric_name == 'volatility':
                returns = self._calculate_returns(symbol_data)
                return self._calculate_volatility(returns)
            elif metric_name == 'sharpe_ratio':
                returns = self._calculate_returns(symbol_data)
                risk_free_rate = kwargs.get('risk_free_rate', 0)
                return self._calculate_sharpe_ratio(returns, risk_free_rate)
            elif metric_name == 'sortino_ratio':
                returns = self._calculate_returns(symbol_data)
                risk_free_rate = kwargs.get('risk_free_rate', 0)
                return self._calculate_sortino_ratio(returns, risk_free_rate)
            elif metric_name == 'volume_ratio_30d':
                # New metric: 24h volume / 30-day average daily volume
                return self._calculate_volume_ratio_30d(symbol_data)
            
            return np.nan
            
        except Exception as e:
            self.logger.error(f"Error calculating {metric_name} for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return np.nan
    
    def _calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data
        
        Note: Each row's 'close' price is the price at the END of that candle period.
        For a candle with timestamp 22:00 (1h), the close price is at 22:59:59.
        """
        if 'close' not in data.columns:
            self.logger.error("No 'close' column in data")
            return pd.Series()
        return data['close'].pct_change().dropna()
    
    def _align_returns(self, symbol_data: pd.DataFrame, benchmark_data: pd.DataFrame):
        """Align symbol and benchmark returns by timestamp"""
        # Calculate returns
        symbol_data = symbol_data.copy()
        benchmark_data = benchmark_data.copy()
        
        # Ensure timestamp is not the index
        if symbol_data.index.name == 'timestamp':
            symbol_data = symbol_data.reset_index()
        if benchmark_data.index.name == 'timestamp':
            benchmark_data = benchmark_data.reset_index()
        
        # Calculate returns
        symbol_data['returns'] = symbol_data['close'].pct_change()
        benchmark_data['returns'] = benchmark_data['close'].pct_change()
        
        # Merge on timestamp
        merged = pd.merge(
            symbol_data[['timestamp', 'returns']],
            benchmark_data[['timestamp', 'returns']],
            on='timestamp',
            suffixes=('_symbol', '_benchmark'),
            how='inner'
        )
        
        # Drop NaN values
        merged = merged.dropna()
        
        self.logger.info(f"Aligned data: {len(merged)} rows from {len(symbol_data)} symbol and {len(benchmark_data)} benchmark rows")
        
        return merged['returns_symbol'], merged['returns_benchmark']
    
    def _calculate_beta(self, symbol_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        if len(symbol_returns) < 2:
            return np.nan
            
        covariance = np.cov(symbol_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return np.nan
        
        return covariance / benchmark_variance
    
    def _calculate_upside_beta(self, symbol_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate upside beta (when benchmark returns are positive)"""
        mask = benchmark_returns > 0
        
        if mask.sum() < 2:
            return np.nan
        
        upside_symbol_returns = symbol_returns[mask]
        upside_benchmark_returns = benchmark_returns[mask]
        
        return self._calculate_beta(upside_symbol_returns, upside_benchmark_returns)
    
    def _calculate_downside_beta(self, symbol_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate downside beta (when benchmark returns are negative)"""
        mask = benchmark_returns < 0
        
        if mask.sum() < 2:
            return np.nan
        
        downside_symbol_returns = symbol_returns[mask]
        downside_benchmark_returns = benchmark_returns[mask]
        
        return self._calculate_beta(downside_symbol_returns, downside_benchmark_returns)
    
    def _calculate_conditional_upside_beta(self, symbol_returns: pd.Series, 
                                         benchmark_returns: pd.Series, 
                                         percentile: int = 10) -> float:
        """Calculate beta during top percentile benchmark returns"""
        threshold = np.percentile(benchmark_returns, 100 - percentile)
        mask = benchmark_returns >= threshold
        
        if mask.sum() < 2:
            return np.nan
        
        return self._calculate_beta(symbol_returns[mask], benchmark_returns[mask])
    
    def _calculate_conditional_downside_beta(self, symbol_returns: pd.Series,
                                           benchmark_returns: pd.Series,
                                           percentile: int = 10) -> float:
        """Calculate beta during bottom percentile benchmark returns"""
        threshold = np.percentile(benchmark_returns, percentile)
        mask = benchmark_returns <= threshold
        
        if mask.sum() < 2:
            return np.nan
        
        return self._calculate_beta(symbol_returns[mask], benchmark_returns[mask])
    
    def _calculate_correlation(self, symbol_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate correlation coefficient"""
        if len(symbol_returns) < 2:
            return np.nan
        return symbol_returns.corr(benchmark_returns)
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return np.nan
        # Adjust annualization factor based on data frequency
        # This is a simplified approach - you might want to make it more sophisticated
        periods_per_year = 365 * 24  # Assuming hourly data as default
        
        return returns.std() * np.sqrt(periods_per_year)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return np.nan
            
        excess_returns = returns - risk_free_rate
        
        if returns.std() == 0:
            return np.nan
        
        return excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return np.nan
            
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.nan
        
        return excess_returns.mean() / downside_returns.std()
    
    def _calculate_squared_residual(self, symbol_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate squared residual (upside_beta - downside_beta)^2"""
        upside_beta = self._calculate_upside_beta(symbol_returns, benchmark_returns)
        downside_beta = self._calculate_downside_beta(symbol_returns, benchmark_returns)
        
        if np.isnan(upside_beta) or np.isnan(downside_beta):
            return np.nan
        
        return (upside_beta - downside_beta) ** 2
    
    def _calculate_volume_ratio_30d(self, data: pd.DataFrame) -> float:
        """Calculate 24h volume / 30-day average daily volume"""
        if 'volume' not in data.columns:
            self.logger.error("No volume column in data")
            return np.nan
        
        # Need at least 24 hours for 24h volume
        if len(data) < 24:
            self.logger.info("Not enough data for 24h volume")
            return np.nan
        
        # Calculate 24h volume (sum of last 24 hours)
        volume_24h = data['volume'].iloc[-24:].sum()
        
        # For 30-day average, we need 720 hours (30 * 24)
        # With the new collector ensuring 30 days minimum, we should have this
        if len(data) >= 720:
            # Get last 720 hours
            last_30_days = data.iloc[-720:]
            
            # Calculate daily volumes
            daily_volumes = []
            for i in range(30):
                start_idx = i * 24
                end_idx = (i + 1) * 24
                if end_idx <= len(last_30_days):
                    daily_volume = last_30_days['volume'].iloc[start_idx:end_idx].sum()
                    daily_volumes.append(daily_volume)
            
            if daily_volumes:
                avg_daily_volume_30d = np.mean(daily_volumes)
                
                if avg_daily_volume_30d > 0:
                    return volume_24h / avg_daily_volume_30d
                else:
                    self.logger.warning("30-day average daily volume is 0")
                    return np.nan
        else:
            # If we don't have 30 days, use whatever we have
            # Calculate how many complete days we have
            complete_days = len(data) // 24
            if complete_days >= 2:  # At least 2 days
                daily_volumes = []
                for i in range(complete_days):
                    start_idx = i * 24
                    end_idx = (i + 1) * 24
                    daily_volume = data['volume'].iloc[start_idx:end_idx].sum()
                    daily_volumes.append(daily_volume)
                
                avg_daily_volume = np.mean(daily_volumes)
                
                if avg_daily_volume > 0:
                    self.logger.info(f"Using {complete_days} days for average instead of 30")
                    return volume_24h / avg_daily_volume
            
        return np.nan
    
    def calculate_metrics_batch(self, symbols: List[str], configs: List[MetricConfig]) -> pd.DataFrame:
        """Calculate multiple metrics for multiple symbols efficiently"""
        results = []
        
        for symbol in symbols:
            if symbol == self.benchmark_symbol:
                continue
            
            row = {'Symbol': symbol}
            
            for config in configs:
                try:
                    value = self.calculate_single_metric(
                        symbol=symbol,
                        metric_name=config.metric_type,
                        interval=config.interval,
                        lookback_days=config.lookback_days,
                        start_date=config.start_date,
                        end_date=config.end_date,
                        **(config.params or {})
                    )
                    row[config.name] = value
                except Exception as e:
                    self.logger.error(f"Error calculating {config.name} for {symbol}: {e}")
                    row[config.name] = np.nan
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def clear_cache(self):
        """Clear all cached data through the collector"""
        if hasattr(self.collector, 'clear_cache'):
            self.collector.clear_cache()
            self.logger.info("Cleared all cached data")