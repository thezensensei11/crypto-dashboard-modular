"""
Migration script to move existing parquet data to DuckDB
Run this BEFORE making any breaking changes
Place this file in: crypto-dashboard-modular/infrastructure/database/migrate_data.py
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from infrastructure.database.duckdb_manager import get_db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_parquet_to_duckdb():
    """Migrate existing parquet files to DuckDB"""
    cache_dir = Path("smart_cache")
    
    if not cache_dir.exists():
        logger.info("No cache directory found at smart_cache/")
        return False
    
    # Initialize database
    db_manager = get_db_manager()
    logger.info("Initialized DuckDB database")
    
    total_migrated = 0
    failed_migrations = []
    
    # Iterate through all symbol directories
    for symbol_dir in cache_dir.iterdir():
        if not symbol_dir.is_dir():
            continue
            
        symbol = symbol_dir.name
        logger.info(f"Processing symbol: {symbol}")
        
        # Process each interval file
        for parquet_file in symbol_dir.glob("*.parquet"):
            interval = parquet_file.stem
            
            try:
                # Read parquet file
                df = pd.read_parquet(parquet_file)
                
                if df.empty:
                    logger.warning(f"Empty file: {symbol} {interval}")
                    continue
                
                # Ensure timestamp column exists and is datetime
                if 'timestamp' not in df.columns:
                    logger.error(f"No timestamp column in {symbol} {interval}")
                    failed_migrations.append((symbol, interval))
                    continue
                
                # Convert timestamp to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Insert into database
                rows_inserted = db_manager.insert_ohlcv_batch(df, symbol, interval)
                total_migrated += rows_inserted
                
                logger.info(f"Migrated {rows_inserted} rows for {symbol} {interval}")
                
            except Exception as e:
                logger.error(f"Error migrating {symbol} {interval}: {e}")
                failed_migrations.append((symbol, interval))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Migration Summary:")
    logger.info(f"Total rows migrated: {total_migrated:,}")
    logger.info(f"Failed migrations: {len(failed_migrations)}")
    
    if failed_migrations:
        logger.error("Failed to migrate:")
        for symbol, interval in failed_migrations:
            logger.error(f"  - {symbol} {interval}")
    
    return len(failed_migrations) == 0


def verify_migration():
    """Verify that migration was successful"""
    logger.info("\nVerifying migration...")
    
    db_manager = get_db_manager()
    
    # Get summary from database
    with db_manager.get_connection() as conn:
        summary = conn.execute("""
            SELECT 
                COUNT(DISTINCT symbol) as symbols,
                COUNT(DISTINCT interval) as intervals,
                COUNT(*) as total_rows,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest
            FROM ohlcv
        """).fetchone()
        
        logger.info(f"Database contains:")
        logger.info(f"  Symbols: {summary[0]}")
        logger.info(f"  Intervals: {summary[1]}")
        logger.info(f"  Total rows: {summary[2]:,}")
        logger.info(f"  Date range: {summary[3]} to {summary[4]}")
        
        # Show sample data
        sample = conn.execute("""
            SELECT symbol, interval, COUNT(*) as rows
            FROM ohlcv
            GROUP BY symbol, interval
            ORDER BY symbol, interval
            LIMIT 10
        """).fetchall()
        
        logger.info("\nSample data:")
        for row in sample:
            logger.info(f"  {row[0]} {row[1]}: {row[2]:,} rows")


if __name__ == "__main__":
    print("Starting parquet to DuckDB migration...")
    print("This will NOT delete your existing parquet files")
    print("="*50)
    
    success = migrate_parquet_to_duckdb()
    
    if success:
        verify_migration()
        print("\n✅ Migration completed successfully!")
        print("Your data is now in crypto_data.duckdb")
        print("Original parquet files are still in smart_cache/")
    else:
        print("\n❌ Migration had some failures. Check the logs above.")