"""
Metrics table display component with proper coloring
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from config import BASE_COLUMNS
from utils.formatting import format_price, format_percentage, format_ratio

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
        
        # Prepare display columns - include ALL base columns that exist
        display_columns = self._prepare_display_columns(data, columns_config)
        display_data = data[display_columns].copy()
        
        # Format data for display but keep numeric for styling
        display_data_formatted = self._format_display_data(display_data.copy())
        
        # Configure column display settings
        column_config = self._get_column_config(display_columns)
        
        # Calculate table height - DOUBLED for more entries
        # Min: 600px (from 200px), Max: 1600px (from 800px), with 40px per row + 120px base
        # This allows viewing approximately 15-35 rows at once instead of 7-18
        table_height = min(1600, max(600, 120 + len(display_data) * 40))
        
        # Apply styling to percentage columns
        styled_data = self._apply_percentage_styling(display_data_formatted, display_data)
        
        # Display the dataframe with increased height
        st.dataframe(
            styled_data,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=table_height  # This will now be 600-1600px instead of 200-800px
        )
        
        # Download button
        self._render_download_button(display_data_formatted)
    
    def _prepare_display_columns(self, data: pd.DataFrame, columns_config: List[Dict[str, Any]]) -> List[str]:
        """Prepare the list of columns to display"""
        # Updated BASE_COLUMNS to include volume ratio
        base_columns_with_ratio = ['Symbol', 'Price', '24h Return %', '24h Volume Change %', '24h Vol/30d Avg']
        
        # Start with base columns that exist in data
        display_columns = [col for col in base_columns_with_ratio if col in data.columns]
        
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
        
        # Format percentage columns (but keep numeric values for styling)
        percentage_cols = [col for col in formatted.columns if '%' in col and col != '24h Vol/30d Avg']
        for col in percentage_cols:
            formatted[col] = formatted[col].apply(lambda x: format_percentage(x) if pd.notna(x) else "N/A")
        
        # Format volume ratio column
        if '24h Vol/30d Avg' in formatted.columns:
            formatted['24h Vol/30d Avg'] = formatted['24h Vol/30d Avg'].apply(
                lambda x: format_ratio(x) if pd.notna(x) else "N/A"
            )
        
        return formatted
    
    def _apply_percentage_styling(self, formatted_data: pd.DataFrame, numeric_data: pd.DataFrame) -> pd.DataFrame:
        """Apply color styling to percentage and ratio columns"""
        style = formatted_data.style
        
        # Apply styling to percentage columns
        percentage_cols = ['24h Return %', '24h Volume Change %']
        
        for col in percentage_cols:
            if col in formatted_data.columns:
                # Apply style based on the sign in the formatted string
                style = style.applymap(
                    lambda val: self._get_percentage_style(val),
                    subset=[col]
                )
        
        # Apply styling to volume ratio column
        if '24h Vol/30d Avg' in formatted_data.columns:
            style = style.applymap(
                lambda val: self._get_ratio_style(val),
                subset=['24h Vol/30d Avg']
            )
        
        return style
    
    def _get_percentage_style(self, val):
        """Get style for percentage value based on formatted string"""
        if pd.isna(val) or val == "N/A":
            return ''
        
        val_str = str(val)
        if val_str.startswith('+') or (val_str[0].isdigit() and float(val_str.replace('%', '')) > 0):
            return 'color: #26c987; font-weight: 600; background-color: rgba(38, 201, 135, 0.1);'
        elif val_str.startswith('-'):
            return 'color: #ff4b4b; font-weight: 600; background-color: rgba(255, 75, 75, 0.1);'
        else:
            return 'color: #888888;'
    
    def _get_ratio_style(self, val):
        """Get style for ratio value (e.g., 1.23x)"""
        if pd.isna(val) or val == "N/A":
            return ''
        
        val_str = str(val)
        try:
            # Extract numeric value from "1.23x" format
            numeric_val = float(val_str.rstrip('x'))
            
            if numeric_val > 1.0:
                # Above average volume - green
                return 'color: #26c987; font-weight: 600; background-color: rgba(38, 201, 135, 0.1);'
            elif numeric_val < 1.0:
                # Below average volume - red
                return 'color: #ff4b4b; font-weight: 600; background-color: rgba(255, 75, 75, 0.1);'
            else:
                # Exactly 1.0x - gray
                return 'color: #888888;'
        except:
            return 'color: #888888;'
    
    def _get_column_config(self, columns: List[str]) -> Dict[str, Any]:
        """Get Streamlit column configuration with consistent left alignment"""
        config = {}
        
        for col in columns:
            if col == 'Symbol':
                # Symbol column - text, left aligned
                config[col] = st.column_config.TextColumn(
                    col,
                    width="medium",
                    help=None,
                    disabled=False,
                    required=True,
                )
            elif col == 'Price':
                # Price column - text (already formatted), left aligned
                config[col] = st.column_config.TextColumn(
                    col,
                    width="medium"
                )
            elif col in ['24h Return %', '24h Volume Change %']:
                # Percentage columns - text (already formatted with colors), left aligned
                config[col] = st.column_config.TextColumn(
                    col, 
                    width="medium"
                )
            elif col == '24h Vol/30d Avg':
                # Volume ratio column - text, left aligned
                config[col] = st.column_config.TextColumn(
                    col, 
                    width="medium"
                )
            else:
                # Other numeric columns - still numeric but will be left-aligned by CSS
                config[col] = st.column_config.NumberColumn(
                    col, 
                    format="%.4f", 
                    width="medium"
                )
        
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