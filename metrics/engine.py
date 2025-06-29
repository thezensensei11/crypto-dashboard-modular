"""
Temporary redirect for metrics engine
"""
try:
    from .unified_engine import MetricsEngine
except ImportError:
    try:
        from metrics.unified_engine import MetricsEngine
    except ImportError:
        # Fallback stub
        import pandas as pd
        
        class MetricsEngine:
            def __init__(self, collector):
                self.collector = collector
                
            def calculate_batch(self, symbols, metric_configs, force_cache=False, progress_callback=None):
                return pd.DataFrame({'Symbol': symbols})
                
            def get_performance_stats(self):
                return {'mode': 'stub', 'api_calls': 0, 'cache_hits': 0}
