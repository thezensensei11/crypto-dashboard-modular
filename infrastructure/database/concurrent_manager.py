#!/usr/bin/env python3
"""
Concurrent DuckDB Manager - Allows read access while processor runs
Add this to your infrastructure/database/ directory
Save as: infrastructure/database/concurrent_manager.py
"""

import duckdb
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path
import logging
import threading
import time

logger = logging.getLogger(__name__)


class ConcurrentDuckDBManager:
    """
    DuckDB Manager that releases connections between operations
    to allow concurrent access from other processes
    """
    
    def __init__(self, db_path: str = "crypto_data.duckdb", connection_timeout: float = 0.1):
        self.db_path = Path(db_path)
        self.connection_timeout = connection_timeout
        self._write_lock = threading.Lock()
        
        # Initialize schema if needed
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Your existing schema initialization
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
    
    @contextmanager
    def _get_connection(self, read_only: bool = False):
        """
        Get a connection that is automatically closed after use
        This ensures we don't hold locks longer than necessary
        """
        conn = None
        start_time = time.time()
        
        try:
            # Try to connect with timeout
            while True:
                try:
                    conn = duckdb.connect(
                        str(self.db_path),
                        read_only=read_only,
                        config={
                            'threads': 4,
                            'memory_limit': '4GB'
                        }
                    )
                    break
                except Exception as e:
                    if time.time() - start_time > self.connection_timeout:
                        raise
                    time.sleep(0.01)
            
            yield conn
            
        finally:
            if conn:
                conn.close()
    
    def write_ohlcv_data(self, data: List[Any], interval: str, upsert: bool = True) -> int:
        """
        Write OHLCV data with minimal lock time
        """
        with self._write_lock:  # Ensure only one write at a time
            with self._get_connection() as conn:
                # Convert data to DataFrame if needed
                if not isinstance(data, pd.DataFrame):
                    df = pd.DataFrame([d.__dict__ if hasattr(d, '__dict__') else d for d in data])
                else:
                    df = data
                
                # Add interval if not present
                if 'interval' not in df.columns:
                    df['interval'] = interval
                
                # Write data
                if upsert:
                    # Create temp table
                    conn.execute("CREATE TEMP TABLE temp_ohlcv AS SELECT * FROM ohlcv WHERE 1=0")
                    conn.execute("INSERT INTO temp_ohlcv SELECT * FROM df")
                    
                    # Upsert
                    conn.execute("""
                        INSERT OR REPLACE INTO ohlcv 
                        SELECT * FROM temp_ohlcv
                    """)
                    
                    result = conn.execute("SELECT COUNT(*) FROM temp_ohlcv").fetchone()[0]
                else:
                    conn.execute("INSERT INTO ohlcv SELECT * FROM df")
                    result = len(df)
                
                return result
    
    def read_ohlcv_data(
        self, 
        symbol: str, 
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read OHLCV data using read-only connection
        """
        with self._get_connection(read_only=True) as conn:
            query = """
                SELECT * FROM ohlcv 
                WHERE symbol = ? AND interval = ?
            """
            params = [symbol, interval]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            return conn.execute(query, params).df()
    
    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get latest timestamp for symbol/interval"""
        with self._get_connection(read_only=True) as conn:
            result = conn.execute("""
                SELECT MAX(timestamp) FROM ohlcv 
                WHERE symbol = ? AND interval = ?
            """, [symbol, interval]).fetchone()
            
            return result[0] if result and result[0] else None
    
    def close(self):
        """No persistent connections to close"""
        logger.info("ConcurrentDuckDBManager closed (no persistent connections)")


# Factory function
def get_concurrent_db_manager() -> ConcurrentDuckDBManager:
    """Get concurrent database manager instance"""
    return ConcurrentDuckDBManager()
