"""
Migration script to move existing data to new infrastructure
Place in: crypto-dashboard/scripts/migrate_data.py
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from infrastructure.database.manager import get_db_manager
from core.models import OHLCVData, Interval
from core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataMigrator:
    """Migrate data from old structure to new infrastructure"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.settings = get_settings()
        self.stats = {
            "files_processed": 0,
            "records_migrated": 0,
            "errors": 0
        }
    
    def migrate_parquet_files(self, data_dir: Path = Path("data/cache")):
        """Migrate existing parquet files to DuckDB"""
        logger.info(f"Scanning {data_dir} for parquet files...")
        
        if not data_dir.exists():
            logger.warning(f"Directory {data_dir} does not exist")
            return
        
        parquet_files = list(data_dir.glob("*.parquet"))
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        for file_path in parquet_files:
            try:
                # Parse filename to extract symbol and interval
                # Expected format: BTCUSDT_1h_data.parquet
                parts = file_path.stem.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    interval = parts[1]
                    
                    # Skip if not a valid interval
                    if not Interval.is_standard(interval):
                        logger.warning(f"Skipping {file_path}: invalid interval {interval}")
                        continue
                    
                    logger.info(f"Migrating {file_path} -> {symbol} {interval}")
                    
                    # Read parquet file
                    df = pd.read_parquet(file_path)
                    
                    # Convert to OHLCVData objects
                    records = []
                    for _, row in df.iterrows():
                        try:
                            # Map columns (adjust based on your old schema)
                            ohlcv = OHLCVData(
                                symbol=symbol,
                                interval=interval,
                                timestamp=pd.to_datetime(row.get('timestamp', row.get('time'))),
                                open=float(row['open']),
                                high=float(row['high']),
                                low=float(row['low']),
                                close=float(row['close']),
                                volume=float(row['volume']),
                                quote_volume=float(row.get('quote_volume', 0)),
                                trades=int(row.get('trades', 0)),
                                taker_buy_base=float(row.get('taker_buy_base', 0)),
                                taker_buy_quote=float(row.get('taker_buy_quote', 0))
                            )
                            records.append(ohlcv)
                        except Exception as e:
                            logger.error(f"Error parsing row: {e}")
                            self.stats["errors"] += 1
                    
                    if records:
                        # Write to DuckDB
                        written = self.db_manager.write_ohlcv_data(
                            records, 
                            interval, 
                            upsert=True
                        )
                        self.stats["records_migrated"] += written
                        logger.info(f"Migrated {written} records from {file_path}")
                    
                    self.stats["files_processed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to migrate {file_path}: {e}")
                self.stats["errors"] += 1
    
    def migrate_json_configs(self):
        """Migrate existing JSON configuration files"""
        config_files = {
            "universe_config.json": "symbols",
            "columns_config.json": "columns",
            "dashboard_state.json": "state"
        }
        
        for filename, config_type in config_files.items():
            file_path = Path(filename)
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    logger.info(f"Migrated {filename}: {len(data) if isinstance(data, list) else 'config'} items")
                    
                    # The new system will use these files as-is
                    # Just log for confirmation
                    
                except Exception as e:
                    logger.error(f"Failed to read {filename}: {e}")
    
    def verify_migration(self):
        """Verify migrated data"""
        logger.info("\nVerifying migration...")
        
        # Get database stats
        db_stats = self.db_manager.get_data_stats()
        
        # Check each interval
        for interval in Interval:
            count = db_stats.get(f"{interval.value}_count", 0)
            if count > 0:
                logger.info(f"  {interval.value}: {count:,} records")
                
                # Sample check - get latest timestamp
                symbols = self.db_manager.get_symbols(interval.value)
                if symbols:
                    sample_symbol = symbols[0]
                    latest = self.db_manager.get_latest_timestamp(
                        sample_symbol, 
                        interval.value
                    )
                    if latest:
                        logger.info(f"    Latest {sample_symbol}: {latest}")
        
        logger.info(f"\nDatabase size: {db_stats['db_size_mb']} MB")
        logger.info(f"Total symbols: {db_stats['symbol_count']}")
    
    def run_migration(self):
        """Run complete migration"""
        logger.info("Starting data migration...")
        logger.info("=" * 50)
        
        # 1. Migrate parquet files
        self.migrate_parquet_files()
        
        # 2. Migrate JSON configs
        self.migrate_json_configs()
        
        # 3. Verify
        self.verify_migration()
        
        # Print summary
        logger.info("\nMigration Summary:")
        logger.info("=" * 50)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Records migrated: {self.stats['records_migrated']:,}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        if self.stats['errors'] == 0:
            logger.info("\n✅ Migration completed successfully!")
        else:
            logger.warning(f"\n⚠️ Migration completed with {self.stats['errors']} errors")


def main():
    """Main migration entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate data to new infrastructure")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual migration"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No data will be migrated")
        # Just scan and report
        data_dir = args.data_dir
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            logger.info(f"Found {len(parquet_files)} parquet files to migrate")
            for f in parquet_files[:5]:  # Show first 5
                logger.info(f"  - {f.name}")
            if len(parquet_files) > 5:
                logger.info(f"  ... and {len(parquet_files) - 5} more")
    else:
        migrator = DataMigrator()
        migrator.run_migration()


if __name__ == "__main__":
    main()
