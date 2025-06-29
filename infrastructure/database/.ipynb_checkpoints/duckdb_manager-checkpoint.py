"""
DuckDB Database Manager for Crypto Data Platform
Fixed version with proper singleton connection management
Place this file in: crypto-dashboard-modular/infrastructure/database/duckdb_manager.py
"""

import duckdb
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import logging
from pathlib import Path
import threading
import atexit

logger = logging.getLogger(__name__)


class DuckDBConnectionPool:
    """Singleton connection pool to prevent configuration conflicts"""
    _instance = None
    _lock = threading.Lock()
    _connections = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._connections = {}
            atexit.register(self.close_all)
    
    def get_connection(self, db_path: str, read_only: bool = False):
        """Get or create a connection with consistent configuration"""
        key = (str(db_path), read_only)
        
        if key not in self._connections:
            with self._lock:
                if key not in self._connections:
                    # Always use the same configuration
                    config = {
                        'threads': 4,
                        'memory_limit': '4GB',
                        'max_memory': '4GB'
                    }
                    
                    try:
                        conn = duckdb.connect(
                            str(db_path),
                            read_only=read_only,
                            config=config
                        )
                        self._connections[key] = conn
                        logger.info(f"Created new DuckDB connection: {key}")
                    except Exception as e:
                        logger.error(f"Failed to create connection: {e}")
                        raise
        
        return self._connections[key]
    
    def close_all(self):
        """Close all connections"""
        for key, conn in self._connections.items():
            try:
                conn.close()
                logger.info(f"Closed connection: {key}")
            except:
                pass
        self._connections.clear()


class DuckDBManager:
    """
    DuckDB Manager for time-series crypto data
    Fixed to handle column mismatches and connection conflicts
    """
    
    # Expected columns from DuckDB schema
    SCHEMA_COLUMNS = [
        'symbol', 'interval', 'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'created_at'
    ]
    
    def __init__(self, db_path: str = "crypto_data.duckdb", read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._pool = DuckDBConnectionPool()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema if not exists"""
        if self.read_only and not self.db_path.exists():
            raise FileNotFoundError(f"Database {self.db_path} not found")
            
        with self.get_connection() as conn:
            # Create main OHLCV table with all expected columns
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol VARCHAR NOT NULL,
                    interval VARCHAR NOT NULL,
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
                    PRIMARY KEY (symbol, interval, timestamp)
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp 
                ON ohlcv(symbol, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_interval_timestamp 
                ON ohlcv(interval, timestamp)
            """)
            
            # Create metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_metadata (
                    symbol VARCHAR NOT NULL,
                    interval VARCHAR NOT NULL,
                    first_timestamp TIMESTAMP NOT NULL,
                    last_timestamp TIMESTAMP NOT NULL,
                    row_count INTEGER NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, interval)
                )
            """)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = self._pool.get_connection(str(self.db_path), self.read_only)
        # Return a cursor-like object that doesn't close the shared connection
        class ConnectionWrapper:
            def __init__(self, conn):
                self.conn = conn
                
            def execute(self, *args, **kwargs):
                return self.conn.execute(*args, **kwargs)
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Don't close the shared connection
                pass
        
        yield ConnectionWrapper(conn)
    
    def insert_ohlcv_batch(self, data: pd.DataFrame, symbol: str, interval: str) -> int:
        """Insert OHLCV data in batch with column handling"""
        if self.read_only:
            raise RuntimeError("Cannot insert data in read-only mode")
            
        if data.empty:
            return 0
        
        with self.get_connection() as conn:
            # Prepare data with all required columns
            insert_data = data.copy()
            
            # Add symbol and interval
            insert_data['symbol'] = symbol
            insert_data['interval'] = interval
            
            # Ensure timestamp is timezone-aware
            if 'timestamp' in insert_data.columns:
                if insert_data['timestamp'].dt.tz is None:
                    insert_data['timestamp'] = insert_data['timestamp'].dt.tz_localize('UTC')
            
            # Add created_at if missing
            if 'created_at' not in insert_data.columns:
                insert_data['created_at'] = datetime.now(timezone.utc)
            
            # Ensure we have all required columns (add defaults for missing ones)
            for col in self.SCHEMA_COLUMNS:
                if col not in insert_data.columns:
                    if col in ['quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']:
                        insert_data[col] = 0
                    elif col == 'created_at':
                        insert_data[col] = datetime.now(timezone.utc)
            
            # Select only the columns that exist in schema
            existing_columns = [col for col in self.SCHEMA_COLUMNS if col in insert_data.columns]
            insert_data = insert_data[existing_columns]
            
            # Register the DataFrame as a temporary view
            conn.register('temp_insert_data', insert_data)
            
            try:
                # Use INSERT OR REPLACE with the registered view
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv 
                    SELECT * FROM temp_insert_data
                """)
                
                # Update metadata
                self._update_metadata(conn, symbol, interval)
                
                logger.info(f"Inserted {len(insert_data)} rows for {symbol} {interval}")
                return len(insert_data)
                
            finally:
                # Unregister the temporary view
                conn.unregister('temp_insert_data')
    
    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get OHLCV data for a symbol/interval"""
        with self.get_connection() as conn:
            query = """
                SELECT timestamp, open, high, low, close, volume,
                       quote_volume, trades, taker_buy_base, taker_buy_quote
                FROM ohlcv
                WHERE symbol = ? AND interval = ?
            """
            
            params = [symbol, interval]
            
            if start_date:
                query += " AND timestamp >= ?"
                # Ensure timezone is handled properly
                if hasattr(start_date, 'tz') and start_date.tz is not None:
                    params.append(start_date)
                else:
                    params.append(pd.Timestamp(start_date).tz_localize('UTC'))
                
            if end_date:
                query += " AND timestamp <= ?"
                # Ensure timezone is handled properly
                if hasattr(end_date, 'tz') and end_date.tz is not None:
                    params.append(end_date)
                else:
                    params.append(pd.Timestamp(end_date).tz_localize('UTC'))
            
            query += " ORDER BY timestamp"
            
            if limit:
                query += f" LIMIT {limit}"
            
            result_df = conn.execute(query, params).df()
            
            # Ensure timestamp column has proper timezone
            if not result_df.empty and 'timestamp' in result_df.columns:
                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], utc=True)
            
            return result_df
    
    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get latest timestamp for symbol/interval"""
        with self.get_connection() as conn:
            result = conn.execute("""
                SELECT MAX(timestamp) as latest
                FROM ohlcv
                WHERE symbol = ? AND interval = ?
            """, [symbol, interval]).fetchone()
            
            if result and result[0]:
                return pd.Timestamp(result[0]).to_pydatetime().replace(tzinfo=timezone.utc)
            return None
    
    def _update_metadata(self, conn, symbol: str, interval: str):
        """Update metadata for symbol/interval"""
        conn.execute("""
            INSERT OR REPLACE INTO data_metadata (symbol, interval, first_timestamp, last_timestamp, row_count, last_updated)
            SELECT 
                ?, 
                ?,
                MIN(timestamp),
                MAX(timestamp),
                COUNT(*),
                CURRENT_TIMESTAMP
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
        """, [symbol, interval, symbol, interval])
    
    def get_metadata(self) -> pd.DataFrame:
        """Get metadata for all symbol/interval combinations"""
        with self.get_connection() as conn:
            return conn.execute("""
                SELECT * FROM data_metadata
                ORDER BY symbol, interval
            """).df()
    
    def get_symbols(self) -> List[str]:
        """Get list of unique symbols in database"""
        with self.get_connection() as conn:
            result = conn.execute("""
                SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol
            """).fetchall()
            return [row[0] for row in result]
    
    def get_intervals(self, symbol: Optional[str] = None) -> List[str]:
        """Get list of intervals, optionally for a specific symbol"""
        with self.get_connection() as conn:
            if symbol:
                query = "SELECT DISTINCT interval FROM ohlcv WHERE symbol = ? ORDER BY interval"
                params = [symbol]
            else:
                query = "SELECT DISTINCT interval FROM ohlcv ORDER BY interval"
                params = []
            
            result = conn.execute(query, params).fetchall()
            return [row[0] for row in result]
    
    def delete_old_data(self, days: int = 30) -> int:
        """Delete data older than specified days"""
        if self.read_only:
            raise RuntimeError("Cannot delete data in read-only mode")
            
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        with self.get_connection() as conn:
            result = conn.execute("""
                DELETE FROM ohlcv WHERE timestamp < ?
            """, [cutoff_date])
            
            deleted_count = result.rowcount
            
            # Update metadata
            conn.execute("""
                DELETE FROM data_metadata 
                WHERE NOT EXISTS (
                    SELECT 1 FROM ohlcv 
                    WHERE ohlcv.symbol = data_metadata.symbol 
                    AND ohlcv.interval = data_metadata.interval
                )
            """)
            
            return deleted_count


# Singleton instance getter
_manager_instance = None
_manager_lock = threading.Lock()


def get_db_manager(db_path: str = "crypto_data.duckdb", read_only: bool = False) -> DuckDBManager:
    """Get singleton DuckDBManager instance"""
    global _manager_instance
    
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = DuckDBManager(db_path, read_only)
    
    # For read-only access, create a new instance
    if read_only and not _manager_instance.read_only:
        return DuckDBManager(db_path, read_only)
    
    return _manager_instance