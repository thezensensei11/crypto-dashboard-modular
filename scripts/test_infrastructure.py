
"""
Test script to verify infrastructure is working
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from core.config import get_settings
from core.models import OHLCVData, EventType
from infrastructure.message_bus.bus import get_message_bus, publish_event
from infrastructure.database.manager import get_db_manager

logging.basicConfig(level=logging.INFO)


async def test_infrastructure():
    """Test all infrastructure components"""
    print("Testing Crypto Data Infrastructure...")
    print("=" * 50)
    
    # 1. Test configuration
    print("\n1. Testing configuration...")
    settings = get_settings()
    print(f"   Environment: {settings.environment}")
    print(f"   Redis URL: {settings.redis.url}")
    print(f"   DuckDB Path: {settings.duckdb.path}")
    
    # 2. Test database
    print("\n2. Testing DuckDB...")
    db_manager = get_db_manager()
    stats = db_manager.get_data_stats()
    print(f"   Database stats: {stats}")
    
    # 3. Test message bus
    print("\n3. Testing Redis message bus...")
    bus = await get_message_bus()
    
    # Publish test event
    event_id = await publish_event(
        EventType.PRICE_UPDATE,
        source="TestScript",
        data={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
    )
    print(f"   Published test event: {event_id}")
    
    # 4. Test writing to database
    print("\n4. Testing database write...")
    test_data = OHLCVData(
        symbol="BTCUSDT",
        interval="1m",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=1),
        open="50000.0",
        high="50100.0",
        low="49900.0",
        close="50050.0",
        volume="100.5",
        quote_volume="5025000.0",
        trades=1500
    )
    
    written = db_manager.write_ohlcv_data([test_data], "1m", upsert=True)
    print(f"   Written {written} test records")
    
    # 5. Test reading from database
    print("\n5. Testing database read...")
    df = db_manager.read_ohlcv_data(
        symbol="BTCUSDT",
        interval="1m",
        limit=10
    )
    print(f"   Read {len(df)} records")
    
    await bus.disconnect()
    
    print("\n✅ All tests passed! Infrastructure is ready.")
    print("\nNext steps:")
    print("1. Start services: docker-compose up -d")
    print("2. Monitor logs: docker-compose logs -f")
    print("3. Access dashboard: http://localhost:8501")
    print("4. Monitor Celery: http://localhost:5555")


if __name__ == "__main__":
    asyncio.run(test_infrastructure())
