
"""
Celery tasks for scheduled operations
"""

from celery import current_app as app
from datetime import datetime, timedelta, timezone
import logging

from infrastructure.scheduler.celery_app import AsyncTask
from infrastructure.collectors.rest import RESTCollector
from infrastructure.processors.data_processor import DataAggregator
from infrastructure.database.manager import get_db_manager
from infrastructure.message_bus.bus import publish_event
from core.models import EventType, AggregationRequest
from core.config import get_settings

logger = logging.getLogger(__name__)


@app.task(bind=True, base=AsyncTask)
def detect_and_fill_gaps(self):
    """Detect and fill data gaps for all symbols"""
    async def _run():
        settings = get_settings()
        db_manager = get_db_manager()
        collector = RESTCollector()
        
        try:
            await collector.start()
            
            # Get all active symbols
            symbols = settings.collector.symbols
            
            # Check gaps for each symbol and key intervals
            gaps_found = 0
            for symbol in symbols:
                for interval in ['1m', '5m', '15m', '1h']:
                    # Determine lookback based on interval
                    if interval in ['1m', '5m']:
                        lookback_hours = 4  # Last 4 hours for short intervals
                    else:
                        lookback_hours = 24  # Last 24 hours for longer intervals
                    
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(hours=lookback_hours)
                    
                    # Check for gaps
                    gaps = db_manager.get_data_gaps(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if gaps:
                        gaps_found += len(gaps)
                        logger.info(f"Found {len(gaps)} gaps for {symbol} {interval}")
                        
                        # Fill gaps
                        for gap_start, gap_end in gaps:
                            await collector.collect_historical_data(
                                symbol=symbol,
                                interval=interval,
                                start_time=gap_start,
                                end_time=gap_end
                            )
            
            await collector.stop()
            
            # Publish event
            await publish_event(
                EventType.DATA_VALIDATED,
                source="GapDetector",
                data={
                    "gaps_found": gaps_found,
                    "symbols_checked": len(symbols)
                }
            )
            
            return f"Checked {len(symbols)} symbols, found {gaps_found} gaps"
            
        except Exception as e:
            logger.error(f"Gap detection failed: {e}")
            raise
    
    return self.run_async(_run())


@app.task(bind=True, base=AsyncTask)
def aggregate_candles(self, source_interval: str, target_interval: str):
    """Aggregate candles to higher timeframe"""
    async def _run():
        settings = get_settings()
        db_manager = get_db_manager()
        aggregator = DataAggregator()
        
        try:
            # Get all symbols
            symbols = db_manager.get_symbols(interval=source_interval)
            
            if not symbols:
                logger.warning(f"No symbols found for {source_interval}")
                return "No symbols to aggregate"
            
            total_aggregated = 0
            
            for symbol in symbols:
                # Get last aggregation timestamp
                latest = db_manager.get_latest_timestamp(symbol, target_interval)
                
                # Determine time range
                end_time = datetime.now(timezone.utc)
                if latest:
                    start_time = latest + timedelta(seconds=1)
                else:
                    # Start from reasonable default
                    start_time = end_time - timedelta(days=7)
                
                if start_time >= end_time:
                    continue
                
                # Create aggregation request
                request = AggregationRequest(
                    symbol=symbol,
                    source_interval=source_interval,
                    target_interval=target_interval,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Process aggregation
                count = await aggregator.process_aggregation_request(request)
                total_aggregated += count
            
            # Publish completion event
            await publish_event(
                EventType.AGGREGATION_COMPLETED,
                source="Aggregator",
                data={
                    "source_interval": source_interval,
                    "target_interval": target_interval,
                    "symbols_processed": len(symbols),
                    "candles_created": total_aggregated
                }
            )
            
            return f"Aggregated {total_aggregated} candles for {len(symbols)} symbols"
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise
    
    return self.run_async(_run())


@app.task(bind=True)
def cleanup_old_data(self):
    """Clean up old data based on retention policy"""
    try:
        db_manager = get_db_manager()
        db_manager.cleanup_old_data()
        
        return "Data cleanup completed"
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


@app.task(bind=True)
def optimize_database(self):
    """Optimize database performance"""
    try:
        db_manager = get_db_manager()
        
        # Get stats before optimization
        stats_before = db_manager.get_data_stats()
        
        # Run VACUUM ANALYZE
        with db_manager.connection() as conn:
            conn.execute("VACUUM ANALYZE")
        
        # Get stats after
        stats_after = db_manager.get_data_stats()
        
        logger.info(f"Database optimized. Size: {stats_after['db_size_mb']}MB")
        
        return {
            "size_before_mb": stats_before['db_size_mb'],
            "size_after_mb": stats_after['db_size_mb']
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


@app.task(bind=True, base=AsyncTask)
def collect_historical_data(self, symbols: List[str], intervals: List[str], lookback_days: int):
    """Manually triggered historical data collection"""
    async def _run():
        collector = RESTCollector()
        
        try:
            await collector.start()
            
            results = await collector.collect_multiple_symbols(
                symbols=symbols,
                intervals=intervals,
                lookback_days=lookback_days
            )
            
            await collector.stop()
            
            return results
            
        except Exception as e:
            logger.error(f"Historical collection failed: {e}")
            raise
    
    return self.run_async(_run())
