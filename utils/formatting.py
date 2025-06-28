"""
Formatting utilities for the dashboard
"""

import pandas as pd
from typing import Union

def format_price(price: float, symbol: str = None) -> str:
    """Format price with appropriate precision"""
    if pd.isna(price):
        return "N/A"
    
    # Determine appropriate decimal places based on price magnitude
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 10:
        return f"${price:.3f}"
    elif price >= 1:
        return f"${price:.4f}"
    elif price >= 0.01:
        return f"${price:.5f}"
    else:
        # For very small prices, use scientific notation or more decimals
        formatted = f"{price:.16g}"
        return f"${formatted}"

def format_percentage(value: float) -> str:
    """Format percentage with appropriate decimals and sign"""
    if pd.isna(value):
        return "N/A"
    
    # Format with sign
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"

def format_volume(volume: float) -> str:
    """Format volume with appropriate units"""
    if pd.isna(volume):
        return "N/A"
    
    if volume >= 1e9:
        return f"{volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"{volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"{volume/1e3:.2f}K"
    else:
        return f"{volume:.0f}"

def format_ratio(ratio: float) -> str:
    """Format ratio values"""
    if pd.isna(ratio):
        return "N/A"
    return f"{ratio:.2f}x"

def format_metric_value(value: float, metric_type: str) -> str:
    """Format metric value based on type"""
    if pd.isna(value):
        return "N/A"
    
    # Different formatting for different metric types
    if metric_type in ['volatility']:
        return f"{value:.2%}"
    elif metric_type in ['sharpe_ratio', 'sortino_ratio']:
        return f"{value:.3f}"
    else:
        return f"{value:.4f}"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def get_percentage_color(value: float) -> str:
    """Get color for percentage value"""
    if pd.isna(value):
        return "#888888"
    elif value > 0:
        return "#26c987"  # Green
    elif value < 0:
        return "#ff4b4b"  # Red
    else:
        return "#888888"  # Gray

def style_dataframe_cell(val: Union[str, float], column_type: str = None) -> str:
    """Return CSS style for dataframe cell"""
    if pd.isna(val):
        return ''
    
    # For percentage columns
    if isinstance(val, (int, float)) and column_type == 'percentage':
        if val > 0:
            return 'color: #26c987; font-weight: 600;'
        elif val < 0:
            return 'color: #ff4b4b; font-weight: 600;'
    
    return ''
