"""
Dashboard Controls Tab - Clean version without emojis
"""

import streamlit as st
import os
from datetime import datetime
import logging

from config import Settings
from utils.formatting import format_duration

logger = logging.getLogger(__name__)

class ControlsTab:
    """Dashboard controls and diagnostics"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def render(self):
        """Render the controls tab"""
        st.header("Dashboard Controls")
        
        # Show fetch diagnostics
        self._render_diagnostics()
        
        # Save controls
        self._render_save_controls()
        
        # Export/Import controls
        self._render_export_import()
    
    def _render_diagnostics(self):
        """Render performance diagnostics"""
        if hasattr(st.session_state, 'fetch_diagnostics') and st.session_state.fetch_diagnostics:
            st.subheader("Last Operation Performance")
            
            diag = st.session_state.fetch_diagnostics
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Time Taken", format_duration(diag.duration))
                st.metric("API Calls", diag.total_api_calls)
            
            with col2:
                st.metric("Cache Hits", diag.total_cache_hits)
                st.metric("Cache Hit Rate", f"{diag.cache_efficiency:.0f}%")
            
            with col3:
                if diag.force_cache_mode:
                    st.info("Cache-only mode")
                else:
                    st.info("Normal mode")
            
            st.divider()
    
    def _render_save_controls(self):
        """Render save state controls"""
        st.subheader("Save Dashboard State")
        
        if st.button("**Save Dashboard State**", type="secondary", use_container_width=True):
            with st.spinner("Saving state..."):
                # Save all configurations
                self.settings.save_state(
                    universe=st.session_state.universe,
                    columns_config=st.session_state.columns_config,
                    last_refresh=st.session_state.last_refresh.isoformat() if st.session_state.last_refresh else None
                )
                
                # Verify save
                if self.settings.state_file.exists():
                    st.success("Dashboard state saved successfully!")
                    
                    # Show what was saved
                    with st.expander("Saved State Details", expanded=False):
                        st.json({
                            'symbols_count': len(st.session_state.universe),
                            'columns_count': len(st.session_state.columns_config),
                            'saved_at': datetime.now().isoformat(),
                            'files_updated': [
                                str(self.settings.universe_file.name),
                                str(self.settings.columns_file.name),
                                str(self.settings.state_file.name)
                            ]
                        })
                else:
                    st.error("Failed to save state!")
    
    def _render_export_import(self):
        """Render export/import controls"""
        st.subheader("Export/Import Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("**Export Configuration**", use_container_width=True):
                config = {
                    'universe': st.session_state.universe,
                    'columns': st.session_state.columns_config,
                    'exported_at': datetime.now().isoformat(),
                    'version': '2.0'  # Modular version
                }
                
                import json
                config_json = json.dumps(config, indent=2)
                
                st.download_button(
                    label="**Download Config JSON**",
                    data=config_json,
                    file_name=f"dashboard_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader(
                "Import Configuration",
                type=['json'],
                help="Upload a previously exported configuration file"
            )
            
            if uploaded_file is not None:
                try:
                    import json
                    config = json.load(uploaded_file)
                    
                    # Validate configuration
                    if 'universe' in config and 'columns' in config:
                        st.session_state.universe = config['universe']
                        st.session_state.columns_config = config['columns']
                        
                        # Save imported configuration
                        self.settings.save_universe(st.session_state.universe)
                        self.settings.save_columns(st.session_state.columns_config)
                        
                        st.success("Configuration imported successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid configuration file format")
                
                except Exception as e:
                    st.error(f"Error importing configuration: {e}")