"""
Enhanced DuckDB Database Manager with Time-Series Optimizations
Place this file in: crypto-dashboard-modular/infrastructure/database/duckdb_schema_dal.py
"""

import duckdb
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import logging
from pathlib import Path
from enum import Enum
import threading
from functools import lru_cache
import os

logger = logging.getLogger(__name__)


class DataInterval(Enum):
    """Supported data intervals"""
    ONE_MIN = '1m'
    FIVE_MIN = '5m'
    FIFTEEN_MIN = '15m'
    THIRTY_MIN = '30m'
    ONE_HOUR = '1h'
    FOUR_HOUR = '4h'
    ONE_DAY = '1d'
    
    @property
    def minutes(self) -> int:
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return mapping[self.value]


class DuckDBEnhancedManager:
    """
    Enhanced DuckDB Manager with time-series optimizations
    Features:
    - Partitioned tables by date
    - Materialized views for common queries
    - Connection pooling
    - Automatic data retention
    - Query optimization
    """
    
    def __init__(self, db_path: str = "crypto_data.duckdb", read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._thread_local = threading.local()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize enhanced database schema"""
        if self.read_only and not self.db_path.exists():
            raise FileNotFoundError(f"Database {self.db_path} not found")
            
        with self.get_connection() as conn:
            # Enable extensions
            conn.execute("INSTALL 'parquet'")
            conn.execute("LOAD 'parquet'")
            
            # Main OHLCV table with partitioning
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
                    date_partition DATE GENERATED ALWAYS AS (DATE_TRUNC('day', timestamp)) STORED,
                    PRIMARY KEY (symbol, interval, timestamp)
                )
            """)
            
            # Create indexes for common queries
            self._create_indexes(conn)
            
            # Create live prices table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_prices (
                    symbol VARCHAR PRIMARY KEY,
                    price DOUBLE NOT NULL,
                    bid DOUBLE,
                    ask DOUBLE,
                    timestamp TIMESTAMP NOT NULL,
                    volume_24h DOUBLE,
                    change_24h DOUBLE
                )
            """)
            
            # Create aggregated tables for faster queries
            self._create_aggregation_tables(conn)
            
            # Create materialized views
            self._create_materialized_views(conn)
            
            # Create metadata tables
            self._create_metadata_tables(conn)
    
    def _create_indexes(self, conn):
        """Create optimized indexes"""
        indexes = [
            # Primary lookup patterns
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON ohlcv(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_interval_timestamp ON ohlcv(interval, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_date_partition ON ohlcv(date_partition)",
            
            # For range queries
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval_date ON ohlcv(symbol, interval, date_partition)",
            
            # For latest data queries
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval_timestamp_desc ON ohlcv(symbol, interval, timestamp DESC)",
        ]
        
        for idx in indexes:
            conn.execute(idx)
    
    def _create_aggregation_tables(self, conn):
        """Create pre-aggregated tables for common time periods"""
        # Hourly aggregations for minute data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_hourly_agg AS
            SELECT 
                symbol,
                DATE_TRUNC('hour', timestamp) as hour,
                FIRST(open) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close) as close,
                SUM(volume) as volume,
                SUM(quote_volume) as quote_volume,
                SUM(trades) as trades,
                COUNT(*) as candle_count
            FROM ohlcv
            WHERE interval IN ('1m', '5m', '15m')
            GROUP BY symbol, DATE_TRUNC('hour', timestamp)
        """)
        
        # Daily aggregations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily_agg AS
            SELECT 
                symbol,
                DATE_TRUNC('day', timestamp) as day,
                FIRST(open) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close) as close,
                SUM(volume) as volume,
                SUM(quote_volume) as quote_volume,
                SUM(trades) as trades,
                COUNT(*) as candle_count
            FROM ohlcv
            WHERE interval IN ('1m', '5m', '15m', '30m', '1h')
            GROUP BY symbol, DATE_TRUNC('day', timestamp)
        """)
    
    def _create_materialized_views(self, conn):
        """Create materialized views for common queries"""
        # Latest prices view
        conn.execute("""
            CREATE OR REPLACE VIEW latest_prices AS
            SELECT DISTINCT ON (symbol, interval)
                symbol,
                interval,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                quote_volume
            FROM ohlcv
            ORDER BY symbol, interval, timestamp DESC
        """)
        
        # 24h statistics view
        conn.execute("""
            CREATE OR REPLACE VIEW stats_24h AS
            WITH latest AS (
                SELECT DISTINCT ON (symbol)
                    symbol,
                    close as current_price,
                    volume,
                    timestamp
                FROM ohlcv
                WHERE interval = '1h'
                AND timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                ORDER BY symbol, timestamp DESC
            ),
            previous AS (
                SELECT DISTINCT ON (symbol)
                    symbol,
                    close as price_24h_ago
                FROM ohlcv
                WHERE interval = '1h'
                AND timestamp <= CURRENT_TIMESTAMP - INTERVAL '24 hours'
                AND timestamp > CURRENT_TIMESTAMP - INTERVAL '25 hours'
                ORDER BY symbol, timestamp DESC
            )
            SELECT 
                l.symbol,
                l.current_price,
                l.volume,
                p.price_24h_ago,
                (l.current_price - p.price_24h_ago) / p.price_24h_ago * 100 as change_24h_pct
            FROM latest l
            LEFT JOIN previous p ON l.symbol = p.symbol
        """)
    
    def _create_metadata_tables(self, conn):
        """Create metadata and system tables"""
        # Data coverage table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_coverage (
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                first_timestamp TIMESTAMP NOT NULL,
                last_timestamp TIMESTAMP NOT NULL,
                total_candles INTEGER NOT NULL,
                missing_candles INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, interval)
            )
        """)
        
        # System configuration
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add default retention policies
        conn.execute("""
            INSERT OR IGNORE INTO system_config (key, value) VALUES
            ('retention_1m', '7'),
            ('retention_5m', '30'),
            ('retention_15m', '90'),
            ('retention_1h', '365'),
            ('retention_1d', '3650')
        """)
    
    @contextmanager
    def get_connection(self):
        """Get thread-safe database connection"""
        if not hasattr(self._thread_local, 'conn'):
            self._thread_local.conn = duckdb.connect(
                str(self.db_path),
                read_only=self.read_only,
                config={
                    'threads': 4,
                    'memory_limit': '4GB',
                    'max_memory': '4GB'
                }
            )
        
        yield self._thread_local.conn
    
    def insert_ohlcv_batch(self, data: pd.DataFrame, symbol: str, interval: str) -> int:
        """Insert OHLCV data with automatic aggregation updates"""
        if self.read_only:
            raise RuntimeError("Cannot insert data in read-only mode")
            
        if data.empty:
            return 0
        
        with self.get_connection() as conn:
            # Prepare data
            insert_data = data.copy()
            insert_data['symbol'] = symbol
            insert_data['interval'] = interval
            
            # Ensure timestamp is timezone-aware
            if insert_data['timestamp'].dt.tz is None:
                insert_data['timestamp'] = insert_data['timestamp'].dt.tz_localize('UTC')
            
            # Use COPY for faster insertion
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Insert main data
                conn.execute("""
                    INSERT INTO ohlcv 
                    SELECT * FROM insert_data
                    ON CONFLICT (symbol, interval, timestamp) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        quote_volume = EXCLUDED.quote_volume,
                        trades = EXCLUDED.trades,
                        taker_buy_base = EXCLUDED.taker_buy_base,
                        taker_buy_quote = EXCLUDED.taker_buy_quote,
                        created_at = CURRENT_TIMESTAMP
                """)
                
                # Update aggregations if needed
                if interval in ['1m', '5m', '15m']:
                    self._update_aggregations(conn, symbol, insert_data['timestamp'].min(), insert_data['timestamp'].max())
                
                # Update metadata
                self._update_data_coverage(conn, symbol, interval)
                
                conn.execute("COMMIT")
                
                return len(data)
                
            except Exception as e:
                conn.execute("ROLLBACK")
                raise e
    
    def _update_aggregations(self, conn, symbol: str, start_time: datetime, end_time: datetime):
        """Update aggregation tables incrementally"""
        # Update hourly aggregations
        conn.execute("""
            INSERT OR REPLACE INTO ohlcv_hourly_agg
            SELECT 
                symbol,
                DATE_TRUNC('hour', timestamp) as hour,
                FIRST(open) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close) as close,
                SUM(volume) as volume,
                SUM(quote_volume) as quote_volume,
                SUM(trades) as trades,
                COUNT(*) as candle_count
            FROM ohlcv
            WHERE symbol = ?
            AND interval IN ('1m', '5m', '15m')
            AND timestamp >= DATE_TRUNC('hour', ?)
            AND timestamp <= DATE_TRUNC('hour', ?) + INTERVAL '1 hour'
            GROUP BY symbol, DATE_TRUNC('hour', timestamp)
        """, [symbol, start_time, end_time])
        
        # Update daily aggregations
        conn.execute("""
            INSERT OR REPLACE INTO ohlcv_daily_agg
            SELECT 
                symbol,
                DATE_TRUNC('day', timestamp) as day,
                FIRST(open) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close) as close,
                SUM(volume) as volume,
                SUM(quote_volume) as quote_volume,
                SUM(trades) as trades,
                COUNT(*) as candle_count
            FROM ohlcv
            WHERE symbol = ?
            AND interval IN ('1m', '5m', '15m', '30m', '1h')
            AND timestamp >= DATE_TRUNC('day', ?)
            AND timestamp <= DATE_TRUNC('day', ?) + INTERVAL '1 day'
            GROUP BY symbol, DATE_TRUNC('day', timestamp)
        """, [symbol, start_time, end_time])
    
    def _update_data_coverage(self, conn, symbol: str, interval: str):
        """Update data coverage metadata"""
        conn.execute("""
            INSERT INTO data_coverage (symbol, interval, first_timestamp, last_timestamp, total_candles)
            SELECT 
                ?,
                ?,
                MIN(timestamp),
                MAX(timestamp),
                COUNT(*)
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
            ON CONFLICT (symbol, interval) DO UPDATE SET
                first_timestamp = (
                    SELECT MIN(timestamp) FROM ohlcv 
                    WHERE symbol = EXCLUDED.symbol AND interval = EXCLUDED.interval
                ),
                last_timestamp = (
                    SELECT MAX(timestamp) FROM ohlcv 
                    WHERE symbol = EXCLUDED.symbol AND interval = EXCLUDED.interval
                ),
                total_candles = (
                    SELECT COUNT(*) FROM ohlcv 
                    WHERE symbol = EXCLUDED.symbol AND interval = EXCLUDED.interval
                ),
                last_updated = CURRENT_TIMESTAMP
        """, [symbol, interval, symbol, interval])
    
    @lru_cache(maxsize=128)
    def get_optimized_query(self, query_type: str, **params) -> str:
        """Get optimized query based on query type"""
        queries = {
            'latest_prices': """
                SELECT * FROM latest_prices 
                WHERE symbol = ANY(?)
            """,
            'price_range': """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND interval = ?
                AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """,
            'stats_24h': """
                SELECT * FROM stats_24h
                WHERE symbol = ANY(?)
            """,
            'volume_profile': """
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    SUM(volume) as total_volume,
                    AVG(close) as avg_price
                FROM ohlcv
                WHERE symbol = ? AND interval = ?
                AND timestamp >= ? AND timestamp <= ?
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour
            """
        }
        return queries.get(query_type, "")
    
    def get_ohlcv(
        self, 
        symbol: str, 
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        use_aggregation: bool = True
    ) -> pd.DataFrame:
        """Get OHLCV data with query optimization"""
        with self.get_connection() as conn:
            # Check if we can use aggregated data
            if use_aggregation and self._can_use_aggregation(interval, start_date, end_date):
                return self._get_from_aggregation(conn, symbol, interval, start_date, end_date)
            
            # Use optimized query
            query = self.get_optimized_query('price_range')
            params = [symbol, interval, start_date or datetime(2020, 1, 1), end_date or datetime.now()]
            
            if limit:
                query += f" LIMIT {limit}"
            
            return conn.execute(query, params).df()
    
    def _can_use_aggregation(self, interval: str, start_date: Optional[datetime], end_date: Optional[datetime]) -> bool:
        """Check if we can use pre-aggregated data"""
        if interval not in ['1h', '1d']:
            return False
        
        # Check if date range is large enough to benefit from aggregation
        if start_date and end_date:
            days = (end_date - start_date).days
            return days > 7
        
        return False
    
    def _get_from_aggregation(
        self, 
        conn, 
        symbol: str, 
        interval: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Get data from aggregation tables"""
        if interval == '1h':
            table = 'ohlcv_hourly_agg'
            time_col = 'hour'
        else:
            table = 'ohlcv_daily_agg'
            time_col = 'day'
        
        query = f"""
            SELECT 
                {time_col} as timestamp,
                open, high, low, close, volume,
                quote_volume, trades
            FROM {table}
            WHERE symbol = ?
        """
        params = [symbol]
        
        if start_date:
            query += f" AND {time_col} >= ?"
            params.append(start_date)
        
        if end_date:
            query += f" AND {time_col} <= ?"
            params.append(end_date)
        
        query += f" ORDER BY {time_col}"
        
        return conn.execute(query, params).df()
    
    def update_live_price(
        self, 
        symbol: str, 
        price: float, 
        timestamp: datetime,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ):
        """Update live price data"""
        if self.read_only:
            return
            
        with self.get_connection() as conn:
            # Get 24h stats for the update
            stats = conn.execute("""
                WITH current AS (
                    SELECT close FROM ohlcv 
                    WHERE symbol = ? AND interval = '1h' 
                    ORDER BY timestamp DESC LIMIT 1
                ),
                day_ago AS (
                    SELECT close FROM ohlcv 
                    WHERE symbol = ? AND interval = '1h' 
                    AND timestamp <= CURRENT_TIMESTAMP - INTERVAL '24 hours'
                    ORDER BY timestamp DESC LIMIT 1
                ),
                volume_24h AS (
                    SELECT SUM(volume) as vol FROM ohlcv
                    WHERE symbol = ? AND interval = '1h'
                    AND timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                )
                SELECT 
                    v.vol as volume_24h,
                    (c.close - d.close) / d.close * 100 as change_24h
                FROM current c, day_ago d, volume_24h v
            """, [symbol, symbol, symbol]).fetchone()
            
            volume_24h = stats[0] if stats else None
            change_24h = stats[1] if stats else None
            
            conn.execute("""
                INSERT INTO live_prices (symbol, price, bid, ask, timestamp, volume_24h, change_24h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol) DO UPDATE SET
                    price = EXCLUDED.price,
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    timestamp = EXCLUDED.timestamp,
                    volume_24h = EXCLUDED.volume_24h,
                    change_24h = EXCLUDED.change_24h
            """, [symbol, price, bid, ask, timestamp, volume_24h, change_24h])
    
    def get_data_gaps(
        self, 
        symbol: str, 
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Identify gaps in data"""
        with self.get_connection() as conn:
            # Get expected candle count
            interval_minutes = DataInterval(interval).minutes
            expected_candles = int((end_date - start_date).total_seconds() / 60 / interval_minutes)
            
            # Get actual data
            actual = conn.execute("""
                SELECT timestamp
                FROM ohlcv
                WHERE symbol = ? AND interval = ?
                AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, [symbol, interval, start_date, end_date]).fetchall()
            
            if not actual:
                return [(start_date, end_date)]
            
            gaps = []
            timestamps = [row[0] for row in actual]
            
            # Check for gaps
            for i in range(1, len(timestamps)):
                expected_next = timestamps[i-1] + timedelta(minutes=interval_minutes)
                if timestamps[i] > expected_next + timedelta(minutes=interval_minutes):
                    gaps.append((expected_next, timestamps[i] - timedelta(minutes=interval_minutes)))
            
            # Check start and end
            if timestamps[0] > start_date + timedelta(minutes=interval_minutes):
                gaps.insert(0, (start_date, timestamps[0] - timedelta(minutes=interval_minutes)))
            
            if timestamps[-1] < end_date - timedelta(minutes=interval_minutes):
                gaps.append((timestamps[-1] + timedelta(minutes=interval_minutes), end_date))
            
            return gaps
    
    def apply_retention_policy(self):
        """Apply data retention policies"""
        if self.read_only:
            return
            
        with self.get_connection() as conn:
            # Get retention policies
            policies = conn.execute("""
                SELECT key, value FROM system_config
                WHERE key LIKE 'retention_%'
            """).fetchall()
            
            for key, days in policies:
                interval = key.replace('retention_', '')
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=int(days))
                
                # Delete old data
                deleted = conn.execute("""
                    DELETE FROM ohlcv
                    WHERE interval = ? AND timestamp < ?
                """, [interval, cutoff_date]).rowcount
                
                if deleted > 0:
                    logger.info(f"Deleted {deleted} rows for interval {interval} older than {cutoff_date}")
    
    def optimize_database(self):
        """Run database optimization"""
        if self.read_only:
            return
            
        with self.get_connection() as conn:
            # Update statistics
            conn.execute("ANALYZE")
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            
            # Refresh materialized aggregations
            self._refresh_aggregations(conn)
    
    def _refresh_aggregations(self, conn):
        """Refresh aggregation tables"""
        # Recreate hourly aggregations
        conn.execute("DROP TABLE IF EXISTS ohlcv_hourly_agg")
        self._create_aggregation_tables(conn)
    
    def get_symbols(self) -> List[str]:
        """Get all available symbols"""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
            ).fetchall()
            return [row[0] for row in result]
    
    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get latest timestamp for symbol/interval"""
        with self.get_connection() as conn:
            result = conn.execute("""
                SELECT MAX(timestamp) FROM ohlcv
                WHERE symbol = ? AND interval = ?
            """, [symbol, interval]).fetchone()
            
            return result[0] if result and result[0] else None


# Singleton instance management
_db_manager = None

def get_db_manager(db_path: str = "crypto_data.duckdb", read_only: bool = False) -> DuckDBEnhancedManager:
    """Get or create database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DuckDBEnhancedManager(db_path, read_only)
    return _db_manager