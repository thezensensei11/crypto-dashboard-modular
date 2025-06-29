
"""
Script to run data collectors
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from infrastructure.collectors import WebSocketCollector, RESTCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def run_websocket_collector():
    """Run WebSocket collector"""
    collector = WebSocketCollector()
    
    try:
        # Subscribe to default symbols
        await collector.subscribe_symbols(
            ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            ["kline_1m", "miniTicker"]
        )
        
        await collector.start()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            status = collector.get_status()
            logging.info(f"WebSocket collector status: {status}")
            
    except KeyboardInterrupt:
        logging.info("Shutting down WebSocket collector...")
        await collector.stop()


async def run_rest_collector():
    """Run REST collector"""
    collector = RESTCollector()
    
    try:
        await collector.start()
        
        # Initial backfill
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        for symbol in symbols:
            for interval in ["1m", "5m", "15m", "1h"]:
                await collector.backfill_gaps(symbol, interval, lookback_days=1)
        
        # Keep running and check for gaps periodically
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            for symbol in symbols:
                await collector.backfill_gaps(symbol, "1m", lookback_days=1)
                
    except KeyboardInterrupt:
        logging.info("Shutting down REST collector...")
        await collector.stop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_collector.py [websocket|rest]")
        sys.exit(1)
    
    collector_type = sys.argv[1]
    
    if collector_type == "websocket":
        asyncio.run(run_websocket_collector())
    elif collector_type == "rest":
        asyncio.run(run_rest_collector())
    else:
        print(f"Unknown collector type: {collector_type}")
        sys.exit(1)

