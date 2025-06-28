"""
Main Streamlit application
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging

from config import get_settings, PAGE_TITLE, PAGE_ICON
from ui.styles.theme import apply_theme
from ui.components.sidebar import SidebarComponent
from ui.tabs import UniverseTab, DashboardTab, ControlsTab, SettingsTab
from data.collector import BinanceDataCollector
from data.cache_manager import SmartDataManager
from metrics.engine import MetricsEngine

logger = logging.getLogger(__name__)

class DashboardApp:
    """Main dashboard application"""
    
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
            
            # Initialize data components
            st.session_state.collector = BinanceDataCollector()
            st.session_state.data_manager = SmartDataManager()
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
        
        # Main title
        st.markdown("<h1 style='text-align: center;'>Altcoin Dashboard</h1>", unsafe_allow_html=True)
        
        # Render sidebar
        sidebar = SidebarComponent()
        cache_stats = st.session_state.collector.get_cache_stats()
        sidebar.render(
            universe_size=len(st.session_state.universe),
            columns_count=len(st.session_state.columns_config),
            cache_stats=cache_stats
        )
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Universe Manager", 
            "Metrics Dashboard", 
            "Dashboard Controls", 
            "Settings"
        ])
        
        # Initialize tab components
        universe_tab = UniverseTab(self.settings)
        dashboard_tab = DashboardTab(self.settings)
        controls_tab = ControlsTab(self.settings)
        settings_tab = SettingsTab(self.settings)
        
        # Render tabs
        with tab1:
            universe_tab.render()
        
        with tab2:
            dashboard_tab.render()
        
        with tab3:
            controls_tab.render()
        
        with tab4:
            settings_tab.render()

def run_app():
    """Entry point for the Streamlit app"""
    app = DashboardApp()
    app.run()
