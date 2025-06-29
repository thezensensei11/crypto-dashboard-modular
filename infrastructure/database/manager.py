"""
Enhanced DuckDB Manager with time-series optimizations
Place in: crypto-dashboard/infrastructure/database/manager.py
"""

import duckdb
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from pathlib import Path
import logging
import threading
from functools import lru_cache

from core.config import get_settings
from core.models import OHLCVData, Interval
from core.exceptions import DatabaseException
from core.constants import OHLCV_TABLE_PREFIX, DB_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class DuckDBManager:
    """
    Enhanced DuckDB Manager with time-series optimizations
    
    Features:
    - Partitioned tables by interval
    - Optimized indexes for time-series queries
    - Connection pooling
    - Automatic data retention
    - Query optimization
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.duckdb
        self.db_path = self.config.path
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get thread-local connection"""
        if not hasattr(self._thread_local, 'conn'):
            self._thread_local.conn = duckdb.connect(
                str(self.db_path),
                read_only=self.config.read_only
            )
            # Configure connection
            self._configure_connection(self._thread_local.conn)
        return self._thread_local.conn
    
    def _configure_connection(self, conn: duckdb.DuckDBPyConnection):
        """Configure connection settings"""
        conn.execute(f"SET memory_limit='{self.config.memory_limit}'")
        conn.execute(f"SET threads={self.config.threads}")
        # Enable parallel query execution
        # # conn.execute("SET enable_parallel_query=true")  # Not supported in this DuckDB version  # Not supported in this DuckDB version
        # # conn.execute("SET enable_parallel_append=true")  # Not supported in this DuckDB version  # Not supported in this DuckDB version
    
        # Add any supported configurations here
        try:
            conn.execute("SET enable_progress_bar=false")  # Disable progress bar for scripts
        except:
            pass  # Ignore if not supported
    @contextmanager
    def connection(self):
        """Context manager for database connection"""
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise DatabaseException(f"Database operation failed: {e}")
    
    def _initialize_database(self):
        """Initialize database schema"""
        if self.config.read_only:
            return
        
        with self.connection() as conn:
            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create main OHLCV tables for each interval
            for interval in Interval:
                self._create_ohlcv_table(conn, interval.value)
            
            # Create aggregation tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregation_status (
                    symbol VARCHAR NOT NULL,
                    source_interval VARCHAR NOT NULL,
                    target_interval VARCHAR NOT NULL,
                    last_aggregated_timestamp TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, source_interval, target_interval)
                )
            """)
            
            # Create data quality metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    symbol VARCHAR NOT NULL,
                    interval VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    missing_candles INTEGER DEFAULT 0,
                    invalid_candles INTEGER DEFAULT 0,
                    total_candles INTEGER DEFAULT 0,
                    completeness_pct DOUBLE,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, interval, date)
                )
            """)
            
            # Set schema version
            current_version = conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0]
            
            if current_version != DB_SCHEMA_VERSION:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    [DB_SCHEMA_VERSION]
                )
    
    def _create_ohlcv_table(self, conn: duckdb.DuckDBPyConnection, interval: str):
        """Create OHLCV table for specific interval"""
        table_name = f"{OHLCV_TABLE_PREFIX}{interval}"
        
        # Create table with optimal column order for compression
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                quote_volume DOUBLE,
                trades INTEGER,
                taker_buy_base DOUBLE,
                taker_buy_quote DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Create optimized indexes
        # Index for symbol + time range queries
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_timestamp 
            ON {table_name}(symbol, timestamp)
        """)
        
        # Index for time-based queries across all symbols
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp 
            ON {table_name}(timestamp)
        """)
        
        # Index for latest data queries
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at 
            ON {table_name}(created_at DESC)
        """)
    
    def write_ohlcv_data(
        self,
        data: List[OHLCVData],
        interval: str,
        upsert: bool = True
    ) -> int:
        """
        Write OHLCV data to database
        
        Args:
            data: List of OHLCV data points
            interval: Data interval
            upsert: If True, update existing records
        
        Returns:
            Number of records written
        """
        if not data:
            return 0
        
        table_name = f"{OHLCV_TABLE_PREFIX}{interval}"
        
        # Convert to DataFrame for bulk insert
        df = pd.DataFrame([d.to_dict() for d in data])
        # Add created_at column if not present
        if 'created_at' not in df.columns:
            df['created_at'] = datetime.now(timezone.utc)
        df['interval'] = interval  # Add interval column
        
        # Remove interval from DataFrame as it's in table name
        df = df.drop('interval', axis=1)
        
        with self.connection() as conn:
            if upsert:
                # Use INSERT OR REPLACE for upsert behavior
                # First, create temp table
                conn.execute(f"CREATE TEMP TABLE temp_{table_name} AS SELECT * FROM {table_name} LIMIT 0")
                
                # Insert data into temp table
                conn.execute(f"INSERT INTO temp_{table_name} SELECT * FROM df")
                
                # Merge into main table
                conn.execute(f"""
                    INSERT OR REPLACE INTO {table_name}
                    SELECT * FROM temp_{table_name}
                """)
                
                # Drop temp table
                conn.execute(f"DROP TABLE temp_{table_name}")
            else:
                # Simple insert
                conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            
            # Get affected rows
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            
            logger.info(f"Wrote {len(data)} records to {table_name}")
            return len(data)
    
    def read_ohlcv_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Read OHLCV data from database"""
        table_name = f"{OHLCV_TABLE_PREFIX}{interval}"
        
        # Build query
        query_parts = [f"SELECT * FROM {table_name} WHERE symbol = ?"]
        params = [symbol]
        
        if start_time:
            query_parts.append("AND timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            query_parts.append("AND timestamp <= ?")
            params.append(end_time)
        
        query_parts.append("ORDER BY timestamp ASC")
        
        if limit:
            query_parts.append(f"LIMIT {limit}")
        
        query = " ".join(query_parts)
        
        with self.connection() as conn:
            df = conn.execute(query, params).fetchdf()
            df['interval'] = interval  # Add interval column back
            return df
    
    def get_latest_timestamp(
        self,
        symbol: str,
        interval: str
    ) -> Optional[datetime]:
        """Get latest timestamp for symbol/interval"""
        table_name = f"{OHLCV_TABLE_PREFIX}{interval}"
        
        with self.connection() as conn:
            result = conn.execute(
                f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?",
                [symbol]
            ).fetchone()
            
            return result[0] if result and result[0] else None
    
    def get_data_gaps(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Find gaps in data"""
        table_name = f"{OHLCV_TABLE_PREFIX}{interval}"
        interval_seconds = Interval(interval).seconds
        
        with self.connection() as conn:
            # Get all timestamps in range
            df = conn.execute(f"""
                SELECT timestamp 
                FROM {table_name} 
                WHERE symbol = ? 
                AND timestamp >= ? 
                AND timestamp <= ?
                ORDER BY timestamp
            """, [symbol, start_time, end_time]).fetchdf()
            
            if df.empty:
                return [(start_time, end_time)]
            
            # Find gaps
            gaps = []
            timestamps = pd.to_datetime(df['timestamp'])
            
            for i in range(len(timestamps) - 1):
                expected_next = timestamps[i] + timedelta(seconds=interval_seconds)
                actual_next = timestamps[i + 1]
                
                if actual_next > expected_next:
                    gaps.append((expected_next, actual_next - timedelta(seconds=interval_seconds)))
            
            # Check start and end gaps
            if timestamps[0] > start_time:
                gaps.insert(0, (start_time, timestamps[0] - timedelta(seconds=interval_seconds)))
            
            if timestamps[-1] < end_time:
                gaps.append((timestamps[-1] + timedelta(seconds=interval_seconds), end_time))
            
            return gaps
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        if self.config.read_only:
            return
        
        with self.connection() as conn:
            # Clean 1m data
            if self.config.retention_1m > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_1m)
                deleted = conn.execute(
                    f"DELETE FROM {OHLCV_TABLE_PREFIX}1m WHERE timestamp < ?",
                    [cutoff]
                ).rowcount
                if deleted:
                    logger.info(f"Deleted {deleted} old 1m records")
            
            # Clean 5m data
            if self.config.retention_5m > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_5m)
                deleted = conn.execute(
                    f"DELETE FROM {OHLCV_TABLE_PREFIX}5m WHERE timestamp < ?",
                    [cutoff]
                ).rowcount
                if deleted:
                    logger.info(f"Deleted {deleted} old 5m records")
            
            # Clean 15m data
            if self.config.retention_15m > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_15m)
                deleted = conn.execute(
                    f"DELETE FROM {OHLCV_TABLE_PREFIX}15m WHERE timestamp < ?",
                    [cutoff]
                ).rowcount
                if deleted:
                    logger.info(f"Deleted {deleted} old 15m records")
            
            # Run VACUUM to reclaim space
            conn.execute("VACUUM")
    
    def get_symbols(self, interval: str = "1h") -> List[str]:
        """Get all available symbols"""
        table_name = f"{OHLCV_TABLE_PREFIX}{interval}"
        
        with self.connection() as conn:
            result = conn.execute(
                f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol"
            ).fetchall()
            return [row[0] for row in result]
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        with self.connection() as conn:
            # Get table sizes
            for interval in Interval:
                table_name = f"{OHLCV_TABLE_PREFIX}{interval.value}"
                try:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {table_name}"
                    ).fetchone()[0]
                    stats[f"{interval.value}_count"] = count
                except:
                    stats[f"{interval.value}_count"] = 0
            
            # Get total database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            stats['db_size_mb'] = round(db_size / 1024 / 1024, 2)
            
            # Get symbol count
            try:
                symbols = conn.execute(
                    f"SELECT COUNT(DISTINCT symbol) FROM {OHLCV_TABLE_PREFIX}1h"
                ).fetchone()[0]
                stats['symbol_count'] = symbols
            except:
                stats['symbol_count'] = 0
        
        return stats
    
    def export_to_parquet(
        self,
        symbol: str,
        interval: str,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """Export data to Parquet file"""
        df = self.read_ohlcv_data(symbol, interval, start_time, end_time)
        
        if not df.empty:
            df.to_parquet(output_path, engine='pyarrow', compression='snappy')
            logger.info(f"Exported {len(df)} records to {output_path}")
    
    def import_from_parquet(
        self,
        input_path: Path,
        interval: str,
        upsert: bool = True
    ) -> int:
        """Import data from Parquet file"""
        df = pd.read_parquet(input_path, engine='pyarrow')
        
        # Convert DataFrame to OHLCVData objects
        data = []
        for _, row in df.iterrows():
            data.append(OHLCVData(**row.to_dict()))
        
        return self.write_ohlcv_data(data, interval, upsert)
    
    def close(self):
        """Close database connection"""
        if hasattr(self._thread_local, 'conn'):
            self._thread_local.conn.close()
            delattr(self._thread_local, 'conn')


# Singleton instance
_db_manager: Optional[DuckDBManager] = None


def get_db_manager() -> DuckDBManager:
    """Get database manager instance (singleton)"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DuckDBManager()
    return _db_manager
