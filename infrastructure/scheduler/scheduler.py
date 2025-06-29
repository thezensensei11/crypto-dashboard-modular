"""
Scheduler Service using Celery for automated data collection tasks
Handles periodic data fetching, cleanup, and maintenance tasks
"""

from celery import Celery, Task
from celery.schedules import crontab
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
import json

from message_bus_events import MessageBus, EventType, create_event
from duckdb_schema_dal import get_db_manager
from standalone_collectors import CollectorConfig

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_CONFIG = {
    'broker_url': 'redis://localhost:6379/0',
    'result_backend': 'redis://localhost:6379/0',
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_time_limit': 300,  # 5 minutes
    'task_soft_time_limit': 240,  # 4 minutes
}

# Create Celery app
app = Celery('crypto_scheduler')
app.conf.update(CELERY_CONFIG)

# Initialize components
message_bus = MessageBus()
db_manager = get_db_manager()


class DataCollectionTask(Task):
    """Base task class with error handling and notifications"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        event = create_event(
            EventType.ERROR_ALERT,
            source='Scheduler',
            data={
                'task': self.name,
                'task_id': task_id,
                'error': str(exc),
                'args': args,
                'kwargs': kwargs
            }
        )
        message_bus.publish_event(event)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info(f"Task {self.name} completed successfully: {task_id}")


@app.task(base=DataCollectionTask, bind=True, name='collect_historical_data')
def collect_historical_data(self, symbols: List[str], intervals: List[str], lookback_days: int = 1):
    """
    Task to collect historical data for specified symbols and intervals
    """
    logger.info(f"Starting historical data collection for {len(symbols)} symbols")
    
    results = {
        'success': [],
        'failed': []
    }
    
    for symbol in symbols:
        for interval in intervals:
            try:
                # Publish data fetch request
                event = create_event(
                    EventType.DATA_FETCH_REQUESTED,
                    source='Scheduler',
                    data={
                        'symbol': symbol,
                        'interval': interval,
                        'lookback_days': lookback_days
                    }
                )
                message_bus.publish_event(event)
                
                results['success'].append(f"{symbol}:{interval}")
                
            except Exception as e:
                logger.error(f"Error requesting data for {symbol} {interval}: {e}")
                results['failed'].append(f"{symbol}:{interval}")
    
    return results


@app.task(base=DataCollectionTask, bind=True, name='update_universe')
def update_universe(self):
    """
    Update the universe of symbols to collect
    Fetches active trading symbols from Binance
    """
    import requests
    
    try:
        # Fetch exchange info
        response = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
        data = response.json()
        
        # Filter active perpetual contracts
        active_symbols = []
        for symbol in data['symbols']:
            if (symbol['status'] == 'TRADING' and 
                symbol['contractType'] == 'PERPETUAL' and
                symbol['quoteAsset'] == 'USDT'):
                active_symbols.append(symbol['symbol'])
        
        # Store in database or config
        logger.info(f"Found {len(active_symbols)} active symbols")
        
        # You could store this in Redis or database
        # For now, just return
        return active_symbols
        
    except Exception as e:
        logger.error(f"Error updating universe: {e}")
        raise


@app.task(base=DataCollectionTask, bind=True, name='cleanup_old_data')
def cleanup_old_data(self, retention_days: Dict[str, int]):
    """
    Clean up old data based on retention policies
    
    Args:
        retention_days: Dict mapping interval to days to retain
                       e.g., {'1m': 7, '1h': 30, '1d': 365}
    """
    logger.info("Starting data cleanup task")
    
    deleted_count = 0
    
    with db_manager.get_connection() as conn:
        for interval, days in retention_days.items():
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old data
            result = conn.execute("""
                DELETE FROM ohlcv
                WHERE interval = ? AND timestamp < ?
            """, [interval, cutoff_date])
            
            count = result.rowcount
            deleted_count += count
            
            logger.info(f"Deleted {count} rows for interval {interval}")
    
    # Optimize database after cleanup
    db_manager.optimize_database()
    
    return {'deleted_rows': deleted_count}


@app.task(base=DataCollectionTask, bind=True, name='check_data_gaps')
def check_data_gaps(self, symbols: List[str], intervals: List[str]):
    """
    Check for gaps in data and trigger collection if needed
    """
    gaps_found = []
    
    for symbol in symbols:
        for interval in intervals:
            # Get data gaps for last 7 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)
            
            gaps = db_manager.get_data_gaps(symbol, interval, start_date, end_date)
            
            if gaps:
                gaps_found.append({
                    'symbol': symbol,
                    'interval': interval,
                    'gaps': len(gaps),
                    'total_missing_hours': sum(
                        (gap[1] - gap[0]).total_seconds() / 3600 
                        for gap in gaps
                    )
                })
                
                # Trigger collection for gaps
                collect_historical_data.delay([symbol], [interval], 7)
    
    if gaps_found:
        logger.warning(f"Found data gaps: {gaps_found}")
    
    return gaps_found


@app.task(base=DataCollectionTask, bind=True, name='calculate_metrics')
def calculate_metrics(self, symbols: List[str]):
    """
    Calculate derived metrics for symbols
    This could include technical indicators, volatility, etc.
    """
    results = []
    
    for symbol in symbols:
        try:
            # Get recent data
            data = db_manager.get_ohlcv(
                symbol, '1h',
                start_date=datetime.utcnow() - timedelta(days=30)
            )
            
            if len(data) < 24:
                continue
            
            # Calculate simple metrics
            metrics = {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'volatility_24h': data['close'].pct_change().tail(24).std() * 100,
                'volume_24h': data['volume'].tail(24).sum(),
                'high_24h': data['high'].tail(24).max(),
                'low_24h': data['low'].tail(24).min(),
                'price_change_24h': (data['close'].iloc[-1] / data['close'].iloc[-24] - 1) * 100
            }
            
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
    
    return results


@app.task(base=DataCollectionTask, bind=True, name='generate_alerts')
def generate_alerts(self):
    """
    Check for alert conditions and generate notifications
    """
    alerts = []
    
    # Get latest prices
    with db_manager.get_connection() as conn:
        # Check for large price movements
        price_changes = conn.execute("""
            WITH price_data AS (
                SELECT 
                    symbol,
                    close as current_price,
                    LAG(close, 24) OVER (PARTITION BY symbol ORDER BY timestamp) as price_24h_ago
                FROM ohlcv
                WHERE interval = '1h'
                AND timestamp > CURRENT_TIMESTAMP - INTERVAL '2 days'
            )
            SELECT 
                symbol,
                current_price,
                price_24h_ago,
                (current_price / price_24h_ago - 1) * 100 as change_pct
            FROM price_data
            WHERE price_24h_ago IS NOT NULL
            AND ABS((current_price / price_24h_ago - 1) * 100) > 10
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) = 1
        """).fetchall()
        
        for row in price_changes:
            alert_event = create_event(
                EventType.PRICE_ALERT,
                source='Scheduler',
                data={
                    'symbol': row[0],
                    'current_price': row[1],
                    'change_24h': row[3],
                    'alert_type': 'large_movement'
                }
            )
            message_bus.publish_event(alert_event)
            alerts.append(alert_event.to_dict())
    
    return alerts


# Define periodic tasks schedule
app.conf.beat_schedule = {
    # Collect 1-minute data every minute for top symbols
    'collect-1m-data': {
        'task': 'collect_historical_data',
        'schedule': crontab(minute='*'),
        'args': (['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], ['1m'], 1)
    },
    
    # Collect hourly data every hour
    'collect-1h-data': {
        'task': 'collect_historical_data',
        'schedule': crontab(minute=0),
        'args': ([], ['1h'], 1)  # Empty list means all symbols
    },
    
    # Collect daily data at midnight
    'collect-1d-data': {
        'task': 'collect_historical_data',
        'schedule': crontab(hour=0, minute=0),
        'args': ([], ['1d'], 1)
    },
    
    # Check for data gaps every 15 minutes
    'check-gaps': {
        'task': 'check_data_gaps',
        'schedule': crontab(minute='*/15'),
        'args': (['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], ['1m', '1h'])
    },
    
    # Update universe daily at 1 AM
    'update-universe': {
        'task': 'update_universe',
        'schedule': crontab(hour=1, minute=0),
    },
    
    # Clean up old data daily at 3 AM
    'cleanup-data': {
        'task': 'cleanup_old_data',
        'schedule': crontab(hour=3, minute=0),
        'kwargs': {
            'retention_days': {
                '1m': 7,      # Keep 1-minute data for 7 days
                '5m': 30,     # Keep 5-minute data for 30 days
                '15m': 90,    # Keep 15-minute data for 90 days
                '1h': 365,    # Keep hourly data for 1 year
                '1d': 3650    # Keep daily data for 10 years
            }
        }
    },
    
    # Calculate metrics every 5 minutes
    'calculate-metrics': {
        'task': 'calculate_metrics',
        'schedule': crontab(minute='*/5'),
        'args': (['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],)
    },
    
    # Check for alerts every minute
    'generate-alerts': {
        'task': 'generate_alerts',
        'schedule': crontab(minute='*'),
    }
}


# Manual task triggers
@app.task(name='manual_collect')
def manual_collect(symbols: List[str], intervals: List[str], lookback_days: int):
    """
    Manually trigger data collection
    Can be called from dashboard or API
    """
    return collect_historical_data.delay(symbols, intervals, lookback_days)


@app.task(name='backfill_data')
def backfill_data(symbol: str, interval: str, start_date: str, end_date: str):
    """
    Backfill historical data for a specific date range
    """
    logger.info(f"Backfilling {symbol} {interval} from {start_date} to {end_date}")
    
    # Convert dates
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    days = (end - start).days
    
    # Trigger collection
    return collect_historical_data.delay([symbol], [interval], days)


# Commands to run the scheduler:
# 1. Start Celery worker: celery -A scheduler_service worker --loglevel=info
# 2. Start Celery beat: celery -A scheduler_service beat --loglevel=info
# 3. Start Flower for monitoring: celery -A scheduler_service flower

if __name__ == '__main__':
    # For testing
    app.start()