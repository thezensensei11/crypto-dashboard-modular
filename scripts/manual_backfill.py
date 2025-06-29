
"""
Manual backfill script for specific symbols and dates
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from infrastructure.collectors.rest import RESTCollector
from core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def manual_backfill(
    symbols: list,
    intervals: list,
    days: int,
    end_date: str = None
):
    """
    Manually backfill historical data
    
    Args:
        symbols: List of symbols to backfill
        intervals: List of intervals to backfill
        days: Number of days to look back
        end_date: End date (YYYY-MM-DD format), defaults to today
    """
    # Parse end date
    if end_date:
        end_time = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_time = datetime.now(timezone.utc)
    
    start_time = end_time - timedelta(days=days)
    
    logger.info(f"Backfilling data from {start_time} to {end_time}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Intervals: {intervals}")
    
    # Initialize collector
    collector = RESTCollector()
    await collector.start()
    
    try:
        total_collected = 0
        
        for symbol in symbols:
            for interval in intervals:
                logger.info(f"\nBackfilling {symbol} {interval}...")
                
                collected = await collector.collect_historical_data(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time
                )
                
                total_collected += collected
                logger.info(f"Collected {collected} candles for {symbol} {interval}")
        
        logger.info(f"\n✅ Backfill complete! Total candles collected: {total_collected}")
        
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        raise
    finally:
        await collector.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Manually backfill historical crypto data"
    )
    
    parser.add_argument(
        "--symbols",
        "-s",
        nargs="+",
        default=["BTCUSDT"],
        help="Symbols to backfill (e.g., BTCUSDT ETHUSDT)"
    )
    
    parser.add_argument(
        "--intervals",
        "-i",
        nargs="+",
        default=["1h"],
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        help="Intervals to backfill"
    )
    
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=7,
        help="Number of days to backfill"
    )
    
    parser.add_argument(
        "--end-date",
        "-e",
        type=str,
        help="End date in YYYY-MM-DD format (default: today)"
    )
    
    args = parser.parse_args()
    
    # Run backfill
    asyncio.run(manual_backfill(
        symbols=args.symbols,
        intervals=args.intervals,
        days=args.days,
        end_date=args.end_date
    ))


if __name__ == "__main__":
    main()
