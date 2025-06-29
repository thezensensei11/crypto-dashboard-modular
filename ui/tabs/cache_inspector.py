"""
Data Inspector Tab - For viewing and verifying DuckDB data
Updated from Cache Inspector to show DuckDB data
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import os

from crypto_dashboard_modular.config import Settings, INTERVALS
from crypto_dashboard_modular.utils.formatting import format_price, format_volume

logger = logging.getLogger(__name__)

class CacheInspectorTab:
    """Tab for inspecting data in DuckDB"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.use_duckdb = os.environ.get('USE_DUCKDB', 'false').lower() == 'true'
    
    def render(self):
        """Render the data inspector tab"""
        # Update title based on data source
        if self.use_duckdb:
            st.header("DuckDB Data Inspector")
            source_info = "📊 Data Source: **DuckDB Database**"
        else:
            st.header("Cache Data Inspector")
            source_info = "📁 Data Source: **Parquet Cache Files**"
        
        st.markdown(source_info)
        
        st.markdown("""
        This tool allows you to inspect the raw data for any symbol and interval combination.
        Use it to verify data freshness and quality.
        """)
        
        # Show data source status
        if self.use_duckdb:
            self._show_duckdb_status()
        
        # Check if we have symbols
        if not st.session_state.universe:
            st.warning("No symbols in universe. Add symbols in the Universe Manager first.")
            return
        
        st.divider()
        
        # Selection controls
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            selected_symbol = st.selectbox(
                "Select Symbol",
                options=sorted(st.session_state.universe),
                help="Choose a symbol to inspect its data"
            )
        
        with col2:
            selected_interval = st.selectbox(
                "Select Interval",
                options=INTERVALS,
                index=INTERVALS.index('1h'),  # Default to 1h
                help="Choose the candle interval"
            )
        
        with col3:
            lookback_days = st.number_input(
                "Days",
                min_value=1,
                max_value=365,
                value=7,
                help="Number of days to display"
            )
        
        with col4:
            st.write("")  # Spacing
            show_data = st.button("**Show Data**", type="primary", use_container_width=True)
        
        # Display data if button clicked
        if show_data and selected_symbol and selected_interval:
            self._display_data(selected_symbol, selected_interval, lookback_days)
    
    def _show_duckdb_status(self):
        """Show DuckDB connection status and stats"""
        try:
            from infrastructure.database.duckdb_manager import get_db_manager
            
            db_manager = get_db_manager(read_only=True)
            
            with db_manager.get_connection() as conn:
                # Get database size
                db_size = db_manager.db_path.stat().st_size / (1024 * 1024) if db_manager.db_path.exists() else 0
                
                # Get row count
                row_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
                
                # Get symbol count
                symbol_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv").fetchone()[0]
                
                # Get date range
                date_range = conn.execute("""
                    SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
                    FROM ohlcv
                """).fetchone()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Database Size", f"{db_size:.1f} MB")
                
                with col2:
                    st.metric("Total Rows", f"{row_count:,}")
                
                with col3:
                    st.metric("Symbols", symbol_count)
                
                with col4:
                    if date_range[0] and date_range[1]:
                        date_span = (date_range[1] - date_range[0]).days
                        st.metric("Data Span", f"{date_span} days")
                    else:
                        st.metric("Data Span", "No data")
                
        except Exception as e:
            st.error(f"Error connecting to DuckDB: {str(e)}")
    
    def _display_data(self, symbol: str, interval: str, lookback_days: int):
        """Display data for the selected symbol and interval"""
        st.divider()
        
        with st.spinner(f"Loading data for {symbol} {interval}..."):
            try:
                if self.use_duckdb:
                    # Get data from DuckDB
                    data = self._get_duckdb_data(symbol, interval, lookback_days)
                else:
                    # Get data from cache
                    data = st.session_state.data_manager.load_data(
                        symbol=symbol,
                        interval=interval
                    )
                    if data is not None and not data.empty:
                        # Filter by lookback days
                        cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
                        data = data[data['timestamp'] >= cutoff_date]
                
                if data is None or data.empty:
                    st.error(f"No data found for {symbol} {interval}")
                    return
                
                # Display summary statistics
                self._display_data_summary(data, symbol, interval)
                
                # Display the data table
                self._display_data_table(data, symbol, interval)
                
                # Display data quality checks
                self._display_data_quality(data, symbol, interval)
                
                # Display data freshness
                self._display_data_freshness(data, symbol, interval)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.error(f"Error in data inspector: {e}", exc_info=True)
    
    def _get_duckdb_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """Get data from DuckDB"""
        from infrastructure.database.duckdb_manager import get_db_manager
        
        db_manager = get_db_manager(read_only=True)
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        
        return db_manager.get_ohlcv(symbol, interval, start_date, end_date)
    
    def _display_data_summary(self, data: pd.DataFrame, symbol: str, interval: str):
        """Display summary statistics for the data"""
        st.subheader(f"Data Summary: {symbol} {interval}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(data))
            st.metric("Date Range", f"{data['timestamp'].min().strftime('%Y-%m-%d')} to {data['timestamp'].max().strftime('%Y-%m-%d')}")
        
        with col2:
            st.metric("Latest Close", format_price(data.iloc[-1]['close']))
            st.metric("Avg Volume", format_volume(data['volume'].mean()))
        
        with col3:
            st.metric("High (Period)", format_price(data['high'].max()))
            st.metric("Low (Period)", format_price(data['low'].min()))
        
        with col4:
            price_change = (data.iloc[-1]['close'] - data.iloc[0]['close']) / data.iloc[0]['close'] * 100
            st.metric("Price Change", f"{price_change:+.2f}%")
            missing_count = self._count_missing_candles(data, interval)
            st.metric("Missing Candles", missing_count)
    
    def _display_data_freshness(self, data: pd.DataFrame, symbol: str, interval: str):
        """Display data freshness information"""
        st.subheader("Data Freshness")
        
        latest_timestamp = data['timestamp'].max()
        current_time = datetime.now(timezone.utc)
        
        # Calculate age
        data_age = current_time - latest_timestamp
        
        # Determine freshness status
        interval_minutes = self._get_interval_minutes(interval)
        
        if data_age.total_seconds() < interval_minutes * 60:
            status = "🟢 **Fresh** - Data is up to date"
            color = "green"
        elif data_age.total_seconds() < interval_minutes * 60 * 2:
            status = "🟡 **Recent** - Data is slightly behind"
            color = "yellow"
        else:
            status = "🔴 **Stale** - Data needs updating"
            color = "red"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"Status: {status}")
        
        with col2:
            st.metric("Latest Data", latest_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))
        
        with col3:
            hours_old = data_age.total_seconds() / 3600
            if hours_old < 1:
                age_str = f"{int(data_age.total_seconds() / 60)} minutes"
            elif hours_old < 24:
                age_str = f"{hours_old:.1f} hours"
            else:
                age_str = f"{hours_old / 24:.1f} days"
            
            st.metric("Data Age", age_str)
        
        # Show sync recommendation if using DuckDB
        if self.use_duckdb and data_age.total_seconds() > interval_minutes * 60 * 2:
            st.info("""
            **To update data:**
            ```bash
            python infrastructure/collectors/binance_to_duckdb_sync.py --symbols {} --intervals {} --lookback 1
            ```
            """.format(symbol, interval))
    
    def _display_data_table(self, data: pd.DataFrame, symbol: str, interval: str):
        """Display the data table"""
        st.subheader("Raw Data (Most Recent First)")
        
        # Prepare display data
        display_data = data.copy()
        display_data = display_data.sort_values('timestamp', ascending=False)
        
        # Format columns for display
        display_data['open'] = display_data['open'].apply(format_price)
        display_data['high'] = display_data['high'].apply(format_price)
        display_data['low'] = display_data['low'].apply(format_price)
        display_data['close'] = display_data['close'].apply(format_price)
        display_data['volume'] = display_data['volume'].apply(format_volume)
        
        if 'quote_volume' in display_data.columns:
            display_data['quote_volume'] = display_data['quote_volume'].apply(format_volume)
        
        # Show first 100 rows
        st.dataframe(
            display_data.head(100),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Full Data as CSV",
            data=csv,
            file_name=f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def _display_data_quality(self, data: pd.DataFrame, symbol: str, interval: str):
        """Display data quality checks"""
        st.subheader("Data Quality Checks")
        
        quality_checks = []
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        if missing_values == 0:
            quality_checks.append("✅ No missing values")
        else:
            quality_checks.append(f"❌ {missing_values} missing values found")
        
        # Check for duplicate timestamps
        duplicate_timestamps = data['timestamp'].duplicated().sum()
        if duplicate_timestamps == 0:
            quality_checks.append("✅ No duplicate timestamps")
        else:
            quality_checks.append(f"❌ {duplicate_timestamps} duplicate timestamps found")
        
        # Check for negative prices
        negative_prices = (data[['open', 'high', 'low', 'close']] < 0).sum().sum()
        if negative_prices == 0:
            quality_checks.append("✅ No negative prices")
        else:
            quality_checks.append(f"❌ {negative_prices} negative prices found")
        
        # Check for zero volume
        zero_volume = (data['volume'] == 0).sum()
        if zero_volume == 0:
            quality_checks.append("✅ No zero volume candles")
        else:
            quality_checks.append(f"⚠️ {zero_volume} candles with zero volume")
        
        # Check OHLC consistency
        ohlc_issues = ((data['high'] < data['low']) | 
                      (data['high'] < data['open']) | 
                      (data['high'] < data['close']) |
                      (data['low'] > data['open']) |
                      (data['low'] > data['close'])).sum()
        
        if ohlc_issues == 0:
            quality_checks.append("✅ OHLC values are consistent")
        else:
            quality_checks.append(f"❌ {ohlc_issues} candles with OHLC inconsistencies")
        
        # Check for gaps
        missing_candles = self._count_missing_candles(data, interval)
        if missing_candles == 0:
            quality_checks.append("✅ No missing candles in sequence")
        else:
            quality_checks.append(f"⚠️ {missing_candles} missing candles detected")
        
        # Display checks
        for check in quality_checks:
            st.write(check)
    
    def _count_missing_candles(self, data: pd.DataFrame, interval: str) -> int:
        """Count missing candles in the data"""
        if len(data) < 2:
            return 0
        
        interval_minutes = self._get_interval_minutes(interval)
        expected_timedelta = timedelta(minutes=interval_minutes)
        
        # Sort by timestamp
        sorted_data = data.sort_values('timestamp')
        
        # Calculate time differences
        time_diffs = sorted_data['timestamp'].diff()[1:]  # Skip first NaN
        
        # Count gaps (where difference is more than expected)
        missing = 0
        for diff in time_diffs:
            if diff > expected_timedelta:
                # Calculate how many candles are missing
                missing += int(diff.total_seconds() / (interval_minutes * 60)) - 1
        
        return missing
    
    def _get_interval_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        interval_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
            '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return interval_map.get(interval, 60)