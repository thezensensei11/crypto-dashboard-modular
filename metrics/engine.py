"""
Metrics calculation engine
Wraps the existing MetricsEngine with additional functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing engine
from metrics_engine import MetricsEngine as _MetricsEngine, MetricConfig
from data.collector import BinanceDataCollector
from data.models import MetricConfig as NewMetricConfig, CalculatedColumn

logger = logging.getLogger(__name__)

class MetricsEngine:
    """
    Enhanced metrics engine with modular metric calculation
    """
    
    def __init__(self, collector: BinanceDataCollector):
        self._engine = _MetricsEngine(collector._collector)
        self.collector = collector
        self.logger = logger
        self.benchmark_symbol = 'BTCUSDT'
        
        # Registry for custom metrics
        self._metric_calculators = {}
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register default metric calculators"""
        # These use the existing engine methods
        default_metrics = [
            'beta', 'upside_beta', 'downside_beta', 'correlation',
            'volatility', 'sharpe_ratio', 'sortino_ratio'
        ]
        
        for metric in default_metrics:
            self._metric_calculators[metric] = self._calculate_via_engine
    
    def calculate_metric(
        self,
        symbol: str,
        metric_config: NewMetricConfig,
        force_cache: bool = False
    ) -> float:
        """
        Calculate a single metric for a symbol
        
        Args:
            symbol: Trading symbol
            metric_config: Metric configuration
            force_cache: Whether to use cached data only
            
        Returns:
            Calculated metric value
        """
        # Convert to old MetricConfig format for compatibility
        old_config = MetricConfig(
            name=metric_config.name,
            metric_type=metric_config.metric,
            interval=metric_config.interval,
            lookback_days=metric_config.lookback_days,
            start_date=metric_config.start_date,
            end_date=metric_config.end_date,
            params=metric_config.params
        )
        
        # Use existing engine
        return self._engine.calculate_single_metric(
            symbol=symbol,
            metric_name=metric_config.metric,
            interval=metric_config.interval,
            lookback_days=metric_config.lookback_days,
            start_date=metric_config.start_date,
            end_date=metric_config.end_date,
            force_use_cache=force_cache,
            **(metric_config.params or {})
        )
    
    def _calculate_via_engine(
        self,
        symbol: str,
        metric_name: str,
        data: pd.DataFrame,
        **kwargs
    ) -> float:
        """Delegate to existing engine"""
        # This is a placeholder - the actual calculation happens in calculate_metric
        return np.nan
    
    def calculate_batch(
        self,
        symbols: List[str],
        metric_configs: List[NewMetricConfig],
        force_cache: bool = False,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Calculate multiple metrics for multiple symbols
        
        Args:
            symbols: List of symbols
            metric_configs: List of metric configurations
            force_cache: Whether to use cached data only
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with calculated metrics
        """
        results = []
        total_items = len(symbols)
        
        for idx, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(idx + 1, total_items, f"Processing {symbol}")
            
            row = {'Symbol': symbol}
            
            # Skip metrics for benchmark
            if symbol == self.benchmark_symbol:
                for config in metric_configs:
                    row[config.name] = np.nan
            else:
                for config in metric_configs:
                    try:
                        value = self.calculate_metric(symbol, config, force_cache)
                        row[config.name] = value
                    except Exception as e:
                        self.logger.error(f"Error calculating {config.name} for {symbol}: {e}")
                        row[config.name] = np.nan
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def register_metric(self, name: str, calculator_func):
        """Register a custom metric calculator"""
        self._metric_calculators[name] = calculator_func
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics"""
        return list(self._metric_calculators.keys())
