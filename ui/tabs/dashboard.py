"""
Metrics Dashboard Tab - Clean version without emojis
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import logging

from config import Settings
from ui.components.column_editor import ColumnEditor
from ui.components.metrics_table import MetricsTable
from metrics import MetricsEngine, PriceMetrics, CalculatedMetrics
from data.models import MetricConfig, FetchDiagnostics

logger = logging.getLogger(__name__)

class DashboardTab:
    """Main metrics dashboard tab"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.column_editor = ColumnEditor(settings)
        self.metrics_table = MetricsTable()
    
    def render(self):
        """Render the dashboard tab"""
        st.header("Metrics Dashboard")
        
        if not st.session_state.universe:
            # Better empty state since this is now the first tab users see
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.warning("**No symbols configured yet**")
                st.info("""
                To get started:
                1. Click on the **Universe Manager** tab above
                2. Add symbols you want to track
                3. Come back here to view metrics
                """)
            return
        
        # Column configuration section
        self.column_editor.render()
        
        # Display current columns (only visible when maximized)
        if st.session_state.columns_config:
            self.column_editor.render_column_list(
                st.session_state.columns_config,
                st.session_state.get('columns_expanded', False)
            )
        
        # Metrics calculation section
        if st.session_state.columns_config:
            st.divider()
            self._render_calculation_controls()
            
            # Display metrics table
            if not st.session_state.metrics_data.empty:
                self._render_metrics_table()
            else:
                self._render_no_data_message()
        else:
            st.info("Add some metric columns above to see data.")
    
    def _render_calculation_controls(self):
        """Render the calculation control buttons"""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if st.button("**Recalculate Metrics**", type="primary", use_container_width=True):
                self._calculate_metrics(force_cache=True)
        
        with col2:
            if st.button("**Fetch New Data & Calculate**", type="primary", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                self.settings.save_state(last_refresh=st.session_state.last_refresh.isoformat())
                self._calculate_metrics(force_cache=False)
        
        with col3:
            # Show last refresh info
            self._render_refresh_status()
    
    def _render_refresh_status(self):
        """Render the refresh status indicator"""
        if st.session_state.last_refresh:
            age = datetime.now() - st.session_state.last_refresh
            if age.total_seconds() < 3600:
                st.success(f"{int(age.total_seconds() / 60)}m ago")
            else:
                st.warning(f"{int(age.total_seconds() / 3600)}h ago")
        else:
            st.info("Ready")
    
    def _render_no_data_message(self):
        """Render message when no data is available"""
        cache_dir = self.settings.config_dir / "smart_cache"
        if not cache_dir.exists():
            st.warning("No cached data found. Click 'Fetch New Data & Calculate' to download data from Binance.")
        else:
            st.info("Click 'Recalculate Metrics' to calculate metrics using cached data, or 'Fetch New Data & Calculate' to get the latest data from Binance.")
    
    def _calculate_metrics(self, force_cache: bool = False):
        """Calculate all metrics for all symbols"""
        with st.spinner("Calculating metrics..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize tracking
            start_time = time.time()
            st.session_state.collector.reset_counters()
            
            # Get engine
            engine = st.session_state.engine
            
            # Prepare metric configs
            metric_configs = []
            calculated_configs = []
            
            for col_config in st.session_state.columns_config:
                if col_config.get('type') == 'metric':
                    metric_configs.append(MetricConfig.from_dict(col_config))
                elif col_config.get('type') == 'calculated':
                    calculated_configs.append(col_config)
            
            # Calculate metrics with progress callback
            def progress_callback(current, total, message):
                progress = current / total
                progress_bar.progress(progress)
                mode = "Cache Only" if force_cache else "Normal"
                status_text.text(f"{message} ({current}/{total}) - {mode}")
            
            # Calculate base metrics
            results_df = engine.calculate_batch(
                symbols=st.session_state.universe,
                metric_configs=metric_configs,
                force_cache=force_cache,
                progress_callback=progress_callback
            )
            
            # Add price data including ALL price metrics
            for idx, symbol in enumerate(st.session_state.universe):
                progress_callback(idx + 1, len(st.session_state.universe), f"Getting prices for {symbol}")
                
                # Get 3 days of data for volume comparison
                price_data = st.session_state.collector.get_price_data(
                    symbol=symbol,
                    interval='1h',
                    lookback_days=3,  # Need extra data for volume comparison
                    force_cache=force_cache
                )
                
                if price_data is not None and not price_data.empty:
                    # Calculate ALL price metrics including volume ratio
                    price_metrics = PriceMetrics.calculate_all(price_data, symbol)
                    
                    # Update results dataframe with ALL metrics
                    mask = results_df['Symbol'] == symbol
                    for key, value in price_metrics.items():
                        # Add all price-related columns
                        if key in ['Price', '24h Return %', '24h Volume Change %', '24h Vol/30d Avg']:
                            results_df.loc[mask, key] = value
            
            # Calculate formula columns
            if calculated_configs:
                results_df = CalculatedMetrics.calculate_batch(results_df, calculated_configs)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Update diagnostics
            end_time = time.time()
            st.session_state.fetch_diagnostics = FetchDiagnostics(
                start_time=start_time,
                end_time=end_time,
                total_api_calls=st.session_state.collector.api_call_count,
                total_cache_hits=st.session_state.collector.cache_hit_count,
                force_cache_mode=force_cache
            )
            
            # Save results
            st.session_state.metrics_data = results_df
            
            # Save to file
            try:
                results_df.to_parquet(self.settings.metrics_file, index=False)
                logger.info(f"Saved {len(results_df)} rows of metrics data")
            except Exception as e:
                logger.error(f"Error saving metrics data: {e}")
            
            # Save state
            self.settings.save_state()
            
            st.rerun()
    
    def _render_metrics_table(self):
        """Render the metrics table"""
        st.subheader("Metrics Table")
        
        # Use the metrics table component
        self.metrics_table.render(
            data=st.session_state.metrics_data,
            columns_config=st.session_state.columns_config
        )