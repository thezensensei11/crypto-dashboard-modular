"""
Metrics table display component
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from config import BASE_COLUMNS
from utils.formatting import format_price, format_percentage

class MetricsTable:
    """Component for displaying metrics data in a table"""
    
    def render(self, data: pd.DataFrame, columns_config: List[Dict[str, Any]]):
        """
        Render the metrics table
        
        Args:
            data: DataFrame with metrics data
            columns_config: List of column configurations
        """
        if data.empty:
            st.warning("No data to display")
            return
        
        # Prepare display columns
        display_columns = self._prepare_display_columns(data, columns_config)
        display_data = data[display_columns].copy()
        
        # Format data for display
        display_data = self._format_display_data(display_data)
        
        # Configure column display settings
        column_config = self._get_column_config(display_columns)
        
        # Calculate table height
        table_height = min(800, max(200, 60 + len(display_data) * 40))
        
        # Display the dataframe
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=table_height
        )
        
        # Download button
        self._render_download_button(display_data)
    
    def _prepare_display_columns(self, data: pd.DataFrame, columns_config: List[Dict[str, Any]]) -> List[str]:
        """Prepare the list of columns to display"""
        # Start with base columns that exist in data
        display_columns = [col for col in BASE_COLUMNS if col in data.columns]
        
        # Add configured columns
        configured_columns = [col['name'] for col in columns_config]
        display_columns.extend([col for col in configured_columns if col in data.columns])
        
        return display_columns
    
    def _format_display_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format data for display"""
        formatted = data.copy()
        
        # Format price column
        if 'Price' in formatted.columns:
            formatted['Price'] = formatted['Price'].apply(format_price)
        
        # Format percentage columns
        percentage_cols = [col for col in formatted.columns if '%' in col]
        for col in percentage_cols:
            formatted[col] = formatted[col].apply(format_percentage)
        
        return formatted
    
    def _get_column_config(self, columns: List[str]) -> Dict[str, Any]:
        """Get Streamlit column configuration"""
        config = {}
        
        for col in columns:
            if col == 'Symbol':
                config[col] = st.column_config.TextColumn(col)
            elif col == 'Price':
                config[col] = st.column_config.TextColumn(col)
            elif '%' in col:
                # Percentage columns - already formatted as strings
                config[col] = st.column_config.TextColumn(col, width="medium")
            else:
                # Numeric columns
                config[col] = st.column_config.NumberColumn(col, format="%.4f", width="medium")
        
        return config
    
    def _render_download_button(self, data: pd.DataFrame):
        """Render the download button"""
        csv = data.to_csv(index=False)
        
        col1, col2 = st.columns([4, 1])
        with col2:
            st.download_button(
                label="**Download CSV**",
                data=csv,
                file_name=f"crypto_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
