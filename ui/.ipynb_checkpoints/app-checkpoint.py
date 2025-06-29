"""
Main Streamlit application - DuckDB-only version
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging

from crypto_dashboard_modular.config import get_settings, PAGE_TITLE, PAGE_ICON
from crypto_dashboard_modular.ui.styles.theme import apply_theme
from crypto_dashboard_modular.ui.components.sidebar import SidebarComponent
from crypto_dashboard_modular.ui.tabs import (
    UniverseTab, DashboardTab, ControlsTab, SettingsTab, 
    BacktesterTab, ShockTab
)
from crypto_dashboard_modular.data.duckdb_collector import BinanceDataCollector
from crypto_dashboard_modular.metrics.unified_engine import MetricsEngine

logger = logging.getLogger(__name__)


class DashboardApp:
    """Main dashboard application with DuckDB backend"""
    
    def __init__(self):
        self.settings = get_settings()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.universe = self.settings.universe
            st.session_state.columns_config = self.settings.columns_config
            st.session_state.metrics_data = pd.DataFrame()
            st.session_state.last_refresh = self.settings.get_last_refresh()
            
            # Initialize data components with DuckDB
            logger.info("Initializing DuckDB data collector...")
            
            # Create collector - always async internally, DuckDB backed
            st.session_state.collector = BinanceDataCollector()
            logger.info("✓ DuckDB async data collection initialized")
            
            # Create metrics engine
            st.session_state.engine = MetricsEngine(st.session_state.collector)
            
            # UI state
            st.session_state.price_data = {'BTC': 0, 'SOL': 0, 'HYPE': 0}
            st.session_state.symbols_loaded = False
            st.session_state.fetch_diagnostics = {}
            st.session_state.columns_expanded = False
            
            # Load saved metrics data if exists
            if self.settings.metrics_file.exists():
                try:
                    st.session_state.metrics_data = pd.read_parquet(self.settings.metrics_file)
                    logger.info(f"Loaded {len(st.session_state.metrics_data)} rows of metrics data")
                except Exception as e:
                    logger.error(f"Error loading metrics data: {e}")
    
    def run(self):
        """Run the dashboard application"""
        # Set page config
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply theme
        apply_theme()
        
        # Show database status in header
        st.markdown(
            '<div style="position: fixed; top: 10px; right: 10px; background: #00d4ff; color: #000; '
            'padding: 5px 10px; border-radius: 5px; font-size: 11px; font-weight: bold; z-index: 999;">'
            '🦆 DUCKDB</div>',
            unsafe_allow_html=True
        )
        
        # Render sidebar (simplified - no cache stats)
        sidebar = SidebarComponent()
        stats = st.session_state.collector.get_cache_stats()
        sidebar.render(
            universe_size=len(st.session_state.universe),
            columns_count=len(st.session_state.columns_config),
            db_stats=stats  # Changed from cache_stats
        )
        
        # Create tabs (removed Cache Inspector)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Metrics Dashboard",
            "Backtester",
            "Shock Analysis",
            "Universe Manager", 
            "Dashboard Controls",
            "Settings"
        ])
        
        # Initialize tab components
        dashboard_tab = DashboardTab(self.settings)
        backtester_tab = BacktesterTab(self.settings)
        shock_tab = ShockTab(self.settings)
        universe_tab = UniverseTab(self.settings)
        controls_tab = ControlsTab(self.settings)
        settings_tab = SettingsTab(self.settings)
        
        # Render tabs
        with tab1:
            dashboard_tab.render()
        
        with tab2:
            backtester_tab.render()
        
        with tab3:
            shock_tab.render()
        
        with tab4:
            universe_tab.render()
        
        with tab5:
            controls_tab.render()
        
        with tab6:
            settings_tab.render()


def run_app():
    """Entry point for the Streamlit app"""
    app = DashboardApp()
    app.run()