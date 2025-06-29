"""
Settings Tab - DuckDB-only version without cache management
"""

import streamlit as st
import pandas as pd
import time
import logging

from crypto_dashboard_modular.config import Settings

logger = logging.getLogger(__name__)


class SettingsTab:
    """Application settings - DuckDB version"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def render(self):
        """Render the settings tab"""
        st.header("Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_database_info()
        
        with col2:
            self._render_configuration_management()
    
    def _render_database_info(self):
        """Render database information"""
        st.subheader("Database Information")
        
        # Get database statistics
        if 'collector' in st.session_state:
            db_stats = st.session_state.collector.get_cache_stats()
            
            st.info(f"""
            **DuckDB Statistics:**
            - Total Symbols: {db_stats.get('total_symbols', 0)}
            - Total Rows: {db_stats.get('total_rows', 0):,}
            - Database Size: {db_stats.get('total_size_mb', 0):.1f} MB
            - API Calls Made: {db_stats.get('api_calls', 0)}
            - Database Hits: {db_stats.get('cache_hits', 0)}
            """)
            
            # Database location
            st.info("""
            **Database Location:**
            `crypto_data.duckdb`
            
            The database stores all historical price data and is automatically
            updated when you fetch new data.
            """)
        else:
            st.warning("Database not initialized")
        
        st.divider()
        
        # Data retention info
        st.subheader("Data Retention")
        st.info("""
        DuckDB automatically manages data storage with configurable retention policies:
        - 1m data: 7 days
        - 5m data: 30 days  
        - 15m data: 90 days
        - 1h data: 365 days
        - 1d data: 10 years
        """)
    
    def _render_configuration_management(self):
        """Render configuration management section"""
        st.subheader("Configuration Management")
        
        # Show current configuration
        with st.expander("View Current Configuration", expanded=False):
            st.json({
                'data': self.settings.app_config.get('data', {}),
                'metrics': self.settings.app_config.get('metrics', {}),
                'ui': self.settings.app_config.get('ui', {})
            })
        
        st.divider()
        
        # Reset configurations
        st.subheader("Reset Options")
        
        if st.button("**Reset All Configurations**", type="secondary", use_container_width=True):
            if st.checkbox("I understand this will reset everything to defaults"):
                with st.spinner("Resetting all configurations..."):
                    # Clear session state
                    st.session_state.universe = []
                    st.session_state.columns_config = []
                    st.session_state.metrics_data = pd.DataFrame()
                    st.session_state.last_refresh = None
                    
                    # Reset settings
                    self.settings.reset_all()
                    
                    # Reset counters
                    if 'collector' in st.session_state:
                        st.session_state.collector.reset_counters()
                    
                    st.success("All configurations reset!")
                    time.sleep(1)
                    st.rerun()
        
        st.divider()
        
        # Application info
        st.subheader("Application Info")
        st.info("""
        **Crypto Dashboard v2.0 - DuckDB Edition**
        
        Features:
        - DuckDB for fast local data storage
        - Async data collection for performance
        - Smart gap detection and filling
        - Automatic data staleness checking
        - Real-time price updates
        - Modular metric calculations
        
        **Data Sources:**
        - Binance Futures API
        - Hyperliquid Price API
        
        **Storage Backend:**
        - DuckDB (embedded analytical database)
        """)