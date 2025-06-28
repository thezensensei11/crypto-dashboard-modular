"""
Settings Tab - Clean version without emojis
"""

import streamlit as st
import pandas as pd
import shutil
import time
import logging

from config import Settings, CACHE_DIR

logger = logging.getLogger(__name__)

class SettingsTab:
    """Application settings and cache management"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def render(self):
        """Render the settings tab"""
        st.header("Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_cache_management()
        
        with col2:
            self._render_configuration_management()
    
    def _render_cache_management(self):
        """Render cache management section"""
        st.subheader("Cache Management")
        
        # Get cache statistics
        cache_stats = st.session_state.collector.get_cache_stats()
        
        st.info(f"""
        **Cache Statistics:**
        - Symbols: {cache_stats.get('total_symbols', 0)}
        - Files: {cache_stats.get('total_files', 0)}
        - Total Rows: {cache_stats.get('total_rows', 0):,}
        - Size: {cache_stats.get('total_size_mb', 0):.1f} MB
        - API Calls: {cache_stats.get('api_calls', 0)}
        - Cache Hits: {cache_stats.get('cache_hits', 0)}
        """)
        
        # Clear cache button
        if st.button("**Clear All Cache**", type="secondary", use_container_width=True):
            if st.checkbox("I understand this will delete all cached data"):
                with st.spinner("Clearing cache..."):
                    cache_dir = self.settings.config_dir / CACHE_DIR
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    
                    # Reinitialize data manager
                    st.session_state.data_manager.clear_all()
                    st.session_state.collector.reset_counters()
                    
                    st.success("Cache cleared successfully!")
                    time.sleep(1)
                    st.rerun()
        
        st.divider()
        
        # Clear specific symbol cache
        st.subheader("Clear Symbol Cache")
        
        if st.session_state.universe:
            symbol_to_clear = st.selectbox(
                "Select symbol to clear",
                [""] + st.session_state.universe,
                help="Clear cached data for a specific symbol"
            )
            
            if symbol_to_clear and st.button("**Clear Symbol Cache**", key="clear_symbol"):
                with st.spinner(f"Clearing cache for {symbol_to_clear}..."):
                    st.session_state.collector.clear_cache(symbol_to_clear)
                    st.success(f"Cleared cache for {symbol_to_clear}")
    
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
                    
                    st.success("All configurations reset!")
                    time.sleep(1)
                    st.rerun()
        
        st.divider()
        
        # Application info
        st.subheader("Application Info")
        st.info("""
        **Crypto Dashboard v2.0**
        - Modular architecture
        - Smart caching system
        - Real-time price updates
        - Extensible metrics engine
        
        **Data Sources:**
        - Binance Futures API
        - Hyperliquid Price API
        """)