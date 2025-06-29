"""
Binance to DuckDB Sync Service
This service fetches data from Binance and stores it in DuckDB
Place this file in: crypto-dashboard-modular/infrastructure/collectors/binance_to_duckdb_sync.py
"""

import pandas as pd
import requests
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from infrastructure.database.duckdb_manager import get_db_manager
from crypto_dashboard_modular.config.constants import (
    BINANCE_FUTURES_BASE_URL,
    DEFAULT_SYMBOLS,
    INTERVALS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceToDuckDBSync:
    """Service to sync Binance data to DuckDB"""
    
    def __init__(self):
        self.db_manager = get_db_manager(read_only=False)
        self.base_url = BINANCE_FUTURES_BASE_URL
        self.session = requests.Session()
        self.api_call_count = 0
        
    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch klines from Binance API"""
        url = f"{self.base_url}/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            self.api_call_count += 1
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
            
            # Convert price and volume columns to float
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                       'taker_buy_base', 'taker_buy_quote']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce').astype('Int64')
            
            # Drop the ignore column
            df = df.drop('ignore', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol} {interval}: {e}")
            return pd.DataFrame()
    
    def sync_symbol(
        self,
        symbol: str,
        interval: str,
        lookback_days: int = 30,
        update_existing: bool = True
    ) -> int:
        """Sync data for a single symbol/interval combination"""
        try:
            # Check what data we already have
            latest_timestamp = self.db_manager.get_latest_timestamp(symbol, interval)
            
            if latest_timestamp and update_existing:
                # Start from last timestamp
                start_time = int((latest_timestamp + timedelta(minutes=1)).timestamp() * 1000)
                logger.info(f"Updating {symbol} {interval} from {latest_timestamp}")
            else:
                # Start from lookback_days ago
                start_time = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
                logger.info(f"Fetching {symbol} {interval} for last {lookback_days} days")
            
            end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # Fetch in batches
            all_data = []
            current_start = start_time
            
            while current_start < end_time:
                # Rate limiting
                time.sleep(0.1)  # 100ms delay between requests
                
                batch_data = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    limit=1000
                )
                
                if batch_data.empty:
                    break
                
                all_data.append(batch_data)
                
                # Move to next batch
                last_timestamp = batch_data['timestamp'].max()
                current_start = int((last_timestamp + timedelta(minutes=1)).timestamp() * 1000)
                
                # Check if we've reached the end
                if len(batch_data) < 1000:
                    break
            
            if not all_data:
                logger.warning(f"No data fetched for {symbol} {interval}")
                return 0
            
            # Combine all batches
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates
            combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Sort by timestamp
            combined_data = combined_data.sort_values('timestamp')
            
            # Insert into database
            rows_inserted = self.db_manager.insert_ohlcv_batch(combined_data, symbol, interval)
            
            logger.info(f"Inserted {rows_inserted} rows for {symbol} {interval}")
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Error syncing {symbol} {interval}: {e}")
            return 0
    
    def sync_multiple(
        self,
        symbols: List[str],
        intervals: List[str],
        lookback_days: int = 30
    ) -> Dict[str, int]:
        """Sync multiple symbol/interval combinations"""
        results = {}
        total_combinations = len(symbols) * len(intervals)
        current = 0
        
        logger.info(f"Starting sync for {len(symbols)} symbols and {len(intervals)} intervals")
        
        for symbol in symbols:
            for interval in intervals:
                current += 1
                logger.info(f"Progress: {current}/{total_combinations} - Syncing {symbol} {interval}")
                
                rows = self.sync_symbol(symbol, interval, lookback_days)
                results[f"{symbol}_{interval}"] = rows
                
                # Rate limiting between symbols
                time.sleep(0.5)
        
        logger.info(f"Sync completed. API calls made: {self.api_call_count}")
        return results
    
    def run_continuous_sync(
        self,
        symbols: List[str],
        intervals: List[str],
        update_interval_minutes: int = 5
    ):
        """Run continuous sync - updates data every N minutes"""
        logger.info(f"Starting continuous sync - updating every {update_interval_minutes} minutes")
        
        while True:
            try:
                logger.info(f"Starting sync cycle at {datetime.now()}")
                
                # Sync all combinations
                results = self.sync_multiple(symbols, intervals, lookback_days=1)
                
                # Log results
                total_rows = sum(results.values())
                logger.info(f"Sync cycle completed: {total_rows} total rows inserted/updated")
                
                # Wait for next cycle
                logger.info(f"Waiting {update_interval_minutes} minutes until next sync...")
                time.sleep(update_interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Continuous sync stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous sync: {e}")
                logger.info("Waiting 1 minute before retry...")
                time.sleep(60)


def main():
    """Main entry point for sync service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync Binance data to DuckDB')
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS,
                       help='Symbols to sync (default: DEFAULT_SYMBOLS from config)')
    parser.add_argument('--intervals', nargs='+', default=['1h', '4h', '1d'],
                       help='Intervals to sync (default: 1h, 4h, 1d)')
    parser.add_argument('--lookback', type=int, default=30,
                       help='Days to look back for initial sync (default: 30)')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous sync')
    parser.add_argument('--update-interval', type=int, default=5,
                       help='Minutes between updates in continuous mode (default: 5)')
    
    args = parser.parse_args()
    
    # Create sync service
    sync_service = BinanceToDuckDBSync()
    
    if args.continuous:
        # Run continuous sync
        sync_service.run_continuous_sync(
            symbols=args.symbols,
            intervals=args.intervals,
            update_interval_minutes=args.update_interval
        )
    else:
        # Run one-time sync
        results = sync_service.sync_multiple(
            symbols=args.symbols,
            intervals=args.intervals,
            lookback_days=args.lookback
        )
        
        # Print summary
        print("\nSync Summary:")
        print("-" * 40)
        for key, rows in results.items():
            print(f"{key}: {rows} rows")
        print("-" * 40)
        print(f"Total API calls: {sync_service.api_call_count}")


if __name__ == "__main__":
    main()