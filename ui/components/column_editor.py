"""
Column configuration editor component - Clean version without emojis
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from config import METRIC_TYPES, INTERVALS, DEFAULT_INTERVAL
from data.models import MetricConfig, CalculatedColumn
from utils.validators import validate_formula, extract_dependencies, validate_metric_name

logger = logging.getLogger(__name__)

class ColumnEditor:
    """Component for adding and editing dashboard columns"""
    
    def __init__(self, settings):
        self.settings = settings
    
    def render(self):
        """Render the column editor"""
        with st.expander("Add New Column", expanded=False):
            tab1, tab2 = st.tabs(["Metric Column", "Calculated Column"])
            
            with tab1:
                self._render_metric_column_editor()
            
            with tab2:
                self._render_calculated_column_editor()
    
    def _render_metric_column_editor(self):
        """Render the metric column editor"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            column_name = st.text_input(
                "Column Name", 
                placeholder="e.g., Upside Beta (7d)", 
                key="metric_name"
            )
            
            metric_type = st.selectbox(
                "Metric Type",
                METRIC_TYPES,
                key="metric_type"
            )
        
        with col2:
            interval = st.selectbox(
                "Candle Interval",
                INTERVALS,
                index=INTERVALS.index(DEFAULT_INTERVAL),
                key="metric_interval"
            )
            
            date_range_type = st.radio(
                "Date Range Type",
                ["Lookback Days", "Date Range"],
                key="date_range_type"
            )
        
        with col3:
            if date_range_type == "Lookback Days":
                lookback_days = st.number_input(
                    "Lookback Days", 
                    min_value=1, 
                    max_value=365, 
                    value=30, 
                    key="metric_lookback"
                )
                start_date = None
                end_date = None
            else:
                col3a, col3b = st.columns(2)
                with col3a:
                    start_date = st.date_input(
                        "Start Date", 
                        value=datetime.now() - timedelta(days=30), 
                        key="metric_start"
                    )
                with col3b:
                    end_date = st.date_input(
                        "End Date", 
                        value=datetime.now(), 
                        key="metric_end"
                    )
                lookback_days = None
            
            st.write("Additional Parameters")
            params = {}
            if metric_type in ['sharpe_ratio', 'sortino_ratio']:
                params['risk_free_rate'] = st.number_input(
                    "Risk-Free Rate", 
                    0.0, 0.1, 0.0, 0.001, 
                    key="metric_rf"
                )
        
        with col4:
            st.write("")  # Spacing
            st.write("")  # More spacing
            st.write("")  # Even more spacing
            
            if st.button("**Add Metric Column**", type="primary", use_container_width=True):
                is_valid, error_msg = validate_metric_name(column_name)
                if not is_valid:
                    st.error(error_msg)
                    return
                
                metric_config = MetricConfig(
                    name=column_name,
                    metric=metric_type,
                    interval=interval,
                    lookback_days=lookback_days,
                    start_date=start_date,
                    end_date=end_date,
                    params=params,
                    type='metric'
                )
                
                st.session_state.columns_config.append(metric_config.to_dict())
                self.settings.save_columns(st.session_state.columns_config)
                st.success(f"Added metric column: {column_name}")
                st.rerun()
    
    def _render_calculated_column_editor(self):
        """Render the calculated column editor"""
        st.write("Create a column using arithmetic operations on existing columns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            calc_column_name = st.text_input(
                "Column Name", 
                placeholder="e.g., Beta Ratio", 
                key="calc_name"
            )
            
            existing_columns = [col['name'] for col in st.session_state.columns_config]
            
            if not existing_columns:
                st.warning("Add some metric columns first to use in calculations")
                return
            
            st.subheader("Formula Builder")
            
            formula = st.text_area(
                "Enter formula using column names",
                placeholder="Examples:\n[upside_beta] / [downside_beta]\n([beta] + [correlation]) / 2\n[volatility] * 100",
                help="Use [column_name] to reference columns. Supports +, -, *, /, (, ), and numbers.",
                height=100,
                key="calc_formula"
            )
            
            st.write("**Available columns:**")
            cols_display = ", ".join([f"[{col}]" for col in existing_columns])
            st.info(cols_display)
        
        with col2:
            st.write("**Formula Preview**")
            
            if formula:
                preview = formula
                for col in existing_columns:
                    preview = preview.replace(f"[{col}]", f"**{col}**")
                st.markdown(f"Formula: {preview}")
                
                is_valid, error_msg = validate_formula(formula, existing_columns)
                if is_valid:
                    st.success("Formula is valid")
                else:
                    st.error(f"Invalid formula: {error_msg}")
            
            st.write("")  # Spacing
            
            if st.button("**Add Calculated Column**", type="primary", use_container_width=True):
                if not calc_column_name or not formula:
                    st.error("Please enter both column name and formula")
                    return
                
                is_valid, error_msg = validate_metric_name(calc_column_name)
                if not is_valid:
                    st.error(error_msg)
                    return
                
                is_valid, error_msg = validate_formula(formula, existing_columns)
                if not is_valid:
                    st.error(f"Invalid formula: {error_msg}")
                    return
                
                calc_column = CalculatedColumn(
                    name=calc_column_name,
                    formula=formula,
                    dependencies=extract_dependencies(formula),
                    type='calculated'
                )
                
                st.session_state.columns_config.append(calc_column.to_dict())
                self.settings.save_columns(st.session_state.columns_config)
                st.success(f"Added calculated column: {calc_column_name}")
                st.rerun()
    
    def render_column_list(self, columns: List[Dict[str, Any]], expanded: bool = False):
        """Render the list of configured columns - only visible when maximized"""
        if not columns:
            return
        
        # Show header with maximize/minimize button
        col1, col2 = st.columns([3, 1])
        with col1:
            if expanded:
                st.subheader("Current Columns")
        with col2:
            if st.button(
                f"**{'Minimize' if expanded else 'View Details'}**",
                use_container_width=True
            ):
                st.session_state.columns_expanded = not expanded
                st.rerun()
        
        # Only show column list when expanded
        if expanded:
            self._render_detailed_columns(columns)
    
    def _render_detailed_columns(self, columns: List[Dict[str, Any]]):
        """Render detailed column view"""
        for i, col_config in enumerate(columns):
            with st.container():
                col1, col2, col3 = st.columns([3, 4, 1])
                
                with col1:
                    # Navy blue circle bullet point
                    st.markdown(f"""
                    <div style="display: flex; align-items: center;">
                        <span style="color: #1a365d; font-size: 10px; margin-right: 8px;">●</span>
                        <span style="font-size: 18px; font-weight: 600;">{col_config['name']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if col_config.get('type') == 'metric':
                        details = self._format_metric_details(col_config)
                    else:
                        details = self._format_calculated_details(col_config)
                    st.markdown(details, unsafe_allow_html=True)
                
                with col3:
                    if st.button("**Delete**", key=f"del_{col_config['name']}_{i}"):
                        self._delete_column(i)
                
                st.divider()
    
    def _format_metric_details(self, config: Dict[str, Any]) -> str:
        """Format metric column details"""
        details = f"**Type:** {config['metric']}<br>"
        details += f"**Interval:** {config['interval']}<br>"
        
        if config.get('lookback_days'):
            details += f"**Lookback:** {config['lookback_days']} days<br>"
        elif config.get('start_date') and config.get('end_date'):
            start = datetime.fromisoformat(config['start_date']).strftime('%Y-%m-%d')
            end = datetime.fromisoformat(config['end_date']).strftime('%Y-%m-%d')
            details += f"**Date Range:** {start} to {end}<br>"
        
        if config.get('params'):
            for param, value in config['params'].items():
                details += f"**{param.replace('_', ' ').title()}:** {value}<br>"
        
        return details
    
    def _format_calculated_details(self, config: Dict[str, Any]) -> str:
        """Format calculated column details"""
        details = f"**Formula:**<br>`{config.get('formula', '')}`<br>"
        if config.get('dependencies'):
            details += f"**Dependencies:** {', '.join(config['dependencies'])}"
        return details
    
    def _delete_column(self, index: int):
        """Delete a column"""
        if 0 <= index < len(st.session_state.columns_config):
            deleted_name = st.session_state.columns_config[index]['name']
            st.session_state.columns_config.pop(index)
            self.settings.save_columns(st.session_state.columns_config)
            st.success(f"Deleted column: {deleted_name}")
            st.rerun()