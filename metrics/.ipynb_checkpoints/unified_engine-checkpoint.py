"""
Unified Metrics Engine - Single implementation for all metric calculations
Replaces both MetricsEngine and AsyncMetricsEngine
"""

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Union
import logging
import time

from crypto_dashboard_modular.data.unified_binance_collector import BinanceDataCollector
from crypto_dashboard_modular.data.models import MetricConfig, CalculatedColumn
from crypto_dashboard_modular.metrics.price_metrics import PriceMetrics
from crypto_dashboard_modular.metrics.calculated_metrics import CalculatedMetrics

logger = logging.getLogger(__name__)


class MetricsEngine:
    """
    Unified metrics engine with async data fetching
    Calculates various financial metrics for crypto assets
    """
    
    def __init__(self, collector: BinanceDataCollector):
        """Initialize with a data collector"""
        self.collector = collector
        self.logger = logger
        self.benchmark_symbol = 'BTCUSDT'
        
        # Initialize metric calculators
        self.price_metrics = PriceMetrics()
        self.calculated_metrics = CalculatedMetrics()
        
        # Registry for custom metrics
        self._metric_calculators = {}
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register default metric calculators"""
        # Price-based metrics
        price_metrics = [
            'beta', 'upside_beta', 'downside_beta', 'correlation',
            'volatility', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'calmar_ratio', 'omega_ratio',
            'treynor_ratio', 'information_ratio'
        ]
        
        for metric in price_metrics:
            if hasattr(self.price_metrics, f'calculate_{metric}'):
                self._metric_calculators[metric] = getattr(self.price_metrics, f'calculate_{metric}')
    
    async def calculate_metrics_batch(
        self,
        symbols: List[str],
        metric_configs: List[MetricConfig],
        force_cache: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Calculate multiple metrics for multiple symbols concurrently
        
        Args:
            symbols: List of trading symbols
            metric_configs: List of metric configurations
            force_cache: Whether to use cached data only
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with calculated metrics
        """
        start_time = time.time()
        
        # Analyze data requirements
        data_requirements = self._analyze_data_requirements(symbols, metric_configs)
        
        # Fetch all required data concurrently
        fetch_start = time.time()
        all_data = await self._fetch_all_data_async(data_requirements, force_cache)
        fetch_time = time.time() - fetch_start
        
        # Calculate metrics
        calc_start = time.time()
        results = []
        total_items = len(symbols)
        
        for idx, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(idx + 1, total_items, f"Calculating metrics for {symbol}")
            
            row = {'Symbol': symbol}
            
            # Skip benchmark symbol for relative metrics
            if symbol == self.benchmark_symbol:
                for config in metric_configs:
                    row[config.name] = np.nan
            else:
                for config in metric_configs:
                    try:
                        value = await self._calculate_metric_async(
                            symbol, config, all_data, force_cache
                        )
                        row[config.name] = value
                    except Exception as e:
                        self.logger.error(f"Error calculating {config.name} for {symbol}: {e}")
                        row[config.name] = np.nan
            
            results.append(row)
        
        calc_time = time.time() - calc_start
        total_time = time.time() - start_time
        
        self.logger.info(
            f"Calculated {len(metric_configs)} metrics for {len(symbols)} symbols "
            f"in {total_time:.2f}s (fetch: {fetch_time:.2f}s, calc: {calc_time:.2f}s)"
        )
        
        return pd.DataFrame(results)
    
    def _analyze_data_requirements(
        self,
        symbols: List[str],
        metric_configs: List[MetricConfig]
    ) -> Dict[tuple, set]:
        """Analyze what data needs to be fetched"""
        requirements = {}
        
        # Always need benchmark for relative metrics
        needs_benchmark = any(
            config.metric in ['beta', 'upside_beta', 'downside_beta', 'correlation', 
                            'treynor_ratio', 'information_ratio']
            for config in metric_configs
        )
        
        for config in metric_configs:
            key = (
                config.interval,
                config.lookback_days,
                config.start_date,
                config.end_date
            )
            
            if key not in requirements:
                requirements[key] = set()
            
            # Add all symbols
            requirements[key].update(symbols)
            
            # Add benchmark if needed
            if needs_benchmark and config.metric in ['beta', 'upside_beta', 'downside_beta', 
                                                   'correlation', 'treynor_ratio', 'information_ratio']:
                requirements[key].add(self.benchmark_symbol)
        
        return requirements
    
    async def _fetch_all_data_async(
        self,
        data_requirements: Dict[tuple, set],
        force_cache: bool
    ) -> Dict[str, pd.DataFrame]:
        """Fetch all required data concurrently"""
        all_data = {}
        
        for (interval, lookback_days, start_date, end_date), symbols in data_requirements.items():
            # Convert set to list for batch fetch
            symbol_list = list(symbols)
            
            # Determine parameters
            if lookback_days:
                batch_data = await self.collector.get_price_data_batch(
                    symbols=symbol_list,
                    interval=interval,
                    lookback_days=lookback_days,
                    force_cache=force_cache
                )
            else:
                # Use date range
                tasks = []
                for symbol in symbol_list:
                    task = self.collector.get_price_data(
                        symbol=symbol,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date,
                        force_cache=force_cache
                    )
                    tasks.append((symbol, task))
                
                results = await asyncio.gather(*[task for _, task in tasks])
                batch_data = {symbol: data for (symbol, _), data in zip(tasks, results)}
            
            # Store data with composite keys
            for symbol, data in batch_data.items():
                key = f"{symbol}_{interval}_{lookback_days or f'{start_date}_{end_date}'}"
                all_data[key] = data
        
        return all_data
    
    async def _calculate_metric_async(
        self,
        symbol: str,
        config: MetricConfig,
        all_data: Dict[str, pd.DataFrame],
        force_cache: bool
    ) -> float:
        """Calculate a single metric using pre-fetched data"""
        # Get data key
        data_key = f"{symbol}_{config.interval}_{config.lookback_days or f'{config.start_date}_{config.end_date}'}"
        
        # Get symbol data
        price_data = all_data.get(data_key, pd.DataFrame())
        if price_data.empty:
            return np.nan
        
        # Get benchmark data if needed
        benchmark_data = None
        if config.metric in ['beta', 'upside_beta', 'downside_beta', 'correlation', 
                           'treynor_ratio', 'information_ratio']:
            benchmark_key = f"{self.benchmark_symbol}_{config.interval}_{config.lookback_days or f'{config.start_date}_{config.end_date}'}"
            benchmark_data = all_data.get(benchmark_key, pd.DataFrame())
        
        # Calculate metric
        calculator = self._metric_calculators.get(config.metric)
        if calculator:
            params = config.params or {}
            if benchmark_data is not None and not benchmark_data.empty:
                return calculator(price_data, benchmark_data, **params)
            else:
                return calculator(price_data, **params)
        else:
            self.logger.warning(f"Unknown metric: {config.metric}")
            return np.nan
    
    # Sync wrapper methods for backward compatibility
    def calculate_metric(
        self,
        symbol: str,
        metric_config: MetricConfig,
        force_cache: bool = False
    ) -> float:
        """Calculate a single metric (sync wrapper)"""
        return asyncio.run(self.calculate_metrics_batch(
            [symbol], [metric_config], force_cache
        ))[symbol].iloc[0]
    
    def calculate_batch(
        self,
        symbols: List[str],
        metric_configs: List[MetricConfig],
        force_cache: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Calculate multiple metrics (sync wrapper)"""
        return asyncio.run(self.calculate_metrics_batch(
            symbols, metric_configs, force_cache, progress_callback
        ))
    
    def calculate_calculated_columns(
        self,
        df: pd.DataFrame,
        calculated_columns: List[CalculatedColumn]
    ) -> pd.DataFrame:
        """Apply calculated columns to dataframe"""
        return self.calculated_metrics.apply_calculated_columns(df, calculated_columns)
    
    def register_metric(self, name: str, calculator_func: Callable):
        """Register a custom metric calculator"""
        self._metric_calculators[name] = calculator_func
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics"""
        return list(self._metric_calculators.keys())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'mode': 'async',
            'api_calls': self.collector.api_call_count,
            'cache_hits': self.collector.cache_hit_count,
            'cache_hit_rate': (
                self.collector.cache_hit_count / 
                max(1, self.collector.api_call_count + self.collector.cache_hit_count) * 100
            )
        }