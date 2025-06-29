"""
Celery scheduler configuration and tasks
Place in: crypto-dashboard/infrastructure/scheduler/celery_app.py
"""

from celery import Celery, Task
from celery.schedules import crontab
from datetime import datetime, timedelta
import logging
import asyncio
from typing import List, Dict, Any

from core.config import get_settings
from core.models import EventType, AggregationRequest
from infrastructure.message_bus.bus import publish_event
from infrastructure.database.manager import get_db_manager
from infrastructure.collectors.rest import RESTCollector

logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create Celery app
app = Celery('crypto_scheduler')

# Configure Celery
app.conf.update(
    broker_url=settings.celery.broker_url,
    result_backend=settings.celery.result_backend,
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    worker_prefetch_multiplier=settings.celery.worker_prefetch_multiplier,
    worker_max_tasks_per_child=settings.celery.worker_max_tasks_per_child,
    beat_max_loop_interval=settings.celery.beat_max_loop_interval,
    
    # Task routes
    task_routes={
        'scheduler.tasks.collect_historical_data': {'queue': 'collectors'},
        'scheduler.tasks.aggregate_data': {'queue': 'processors'},
        'scheduler.tasks.cleanup_old_data': {'queue': 'maintenance'},
    },
    
    # Task time limits
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
)


class AsyncTask(Task):
    """Base task class for async operations"""
    
    def run_async(self, coro):
        """Run async coroutine in sync context"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# Beat schedule for periodic tasks
app.conf.beat_schedule = {
    # Gap detection and backfill - every 5 minutes
    'detect-and-fill-gaps': {
        'task': 'scheduler.tasks.detect_and_fill_gaps',
        'schedule': crontab(minute='*/5'),
        'options': {'queue': 'collectors'}
    },
    
    # Aggregation tasks
    'aggregate-5m-candles': {
        'task': 'scheduler.tasks.aggregate_candles',
        'schedule': crontab(minute='*/5'),
        'args': ('1m', '5m'),
        'options': {'queue': 'processors'}
    },
    
    'aggregate-15m-candles': {
        'task': 'scheduler.tasks.aggregate_candles',
        'schedule': crontab(minute='*/15'),
        'args': ('5m', '15m'),
        'options': {'queue': 'processors'}
    },
    
    'aggregate-1h-candles': {
        'task': 'scheduler.tasks.aggregate_candles',
        'schedule': crontab(minute='0'),  # Every hour
        'args': ('15m', '1h'),
        'options': {'queue': 'processors'}
    },
    
    'aggregate-4h-candles': {
        'task': 'scheduler.tasks.aggregate_candles',
        'schedule': crontab(hour='*/4', minute='0'),
        'args': ('1h', '4h'),
        'options': {'queue': 'processors'}
    },
    
    'aggregate-1d-candles': {
        'task': 'scheduler.tasks.aggregate_candles',
        'schedule': crontab(hour='0', minute='0'),  # Daily at midnight
        'args': ('1h', '1d'),
        'options': {'queue': 'processors'}
    },
    
    'aggregate-1w-candles': {
        'task': 'scheduler.tasks.aggregate_candles',
        'schedule': crontab(day_of_week='1', hour='0', minute='0'),  # Weekly on Monday
        'args': ('1d', '1w'),
        'options': {'queue': 'processors'}
    },
    
    # Data cleanup - daily at 2 AM
    'cleanup-old-data': {
        'task': 'scheduler.tasks.cleanup_old_data',
        'schedule': crontab(hour='2', minute='0'),
        'options': {'queue': 'maintenance'}
    },
    
    # Database optimization - weekly
    'optimize-database': {
        'task': 'scheduler.tasks.optimize_database',
        'schedule': crontab(day_of_week='0', hour='3', minute='0'),  # Sunday 3 AM
        'options': {'queue': 'maintenance'}
    },
}


