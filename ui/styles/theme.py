"""
Theme and styling for the dashboard - Original colors with JetBrains Mono font
"""

from crypto_dashboard_modular.config.constants import COLORS

def get_dashboard_css() -> str:
    """Get the complete CSS theme for the dashboard"""
    return f"""
    <style>
        /* Import JetBrains Mono font */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
        
        /* Global font and size adjustments */
        html, body, [class*="css"] {{
            font-family: 'JetBrains Mono', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 14px;
        }}
        
        /* Dark theme background */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        
        /* Reduce top padding since no title */
        .main .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 2rem;
            max-width: 95%;
        }}
        
        /* Tab styling */
        .stTabs {{
            margin-top: -10px !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            background-color: #111111;
            padding: 0;
            border-radius: 10px 10px 0 0;
            border-bottom: 2px solid {COLORS['primary']};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 40px !important;
            padding: 0 24px;
            background-color: {COLORS['background_light']};
            color: {COLORS['text_muted']};
            border-radius: 10px 10px 0 0;
            margin-right: 4px;
            font-size: 16px !important;
            font-weight: 500;
            transition: all 0.3s;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: #222;
            color: #ccc;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['background']} !important;
            color: {COLORS['primary']} !important;
            border-bottom: 3px solid {COLORS['primary']};
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: #111111;
            border-right: 1px solid {COLORS['border']};
            padding-top: 1rem;
        }}
        
       

        /* Sidebar live prices - custom styling */
        .price-container {{
            background: linear-gradient(135deg, {COLORS['background_light']} 0%, #151515 100%);
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            transition: all 0.3s;
            font-family: 'JetBrains Mono', monospace !important;
        }}
        
        .price-container:hover {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 10px rgba(45, 225, 154, 0.1);
        }}
        
        .price-symbol {{
            font-size: 12px !important;
            color: {COLORS['text_muted']};
            font-weight: 500;
            margin-bottom: 4px;
            font-family: 'JetBrains Mono', monospace !important;
            letter-spacing: 0.5px;
        }}
        
        .price-value {{
            font-size: 20px !important;
            font-weight: 600;
            margin-bottom: 0;
            font-family: 'JetBrains Mono', monospace !important;
            letter-spacing: -0.5px;
        }}
        
        .price-btc {{
            color: #f7931a;
        }}
        
        .price-eth {{
            color: #627eea;
        }}
        
        .price-sol {{
            color: #14f195;
        }}
        
        .price-hype {{
            color: {COLORS['primary']};
        }}
        
        
        
        /* Sidebar specific font adjustments */
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label {{
            font-size: 13px !important;
        }}
        
        /* Headers */
        h1, h2, h3 {{
            font-weight: 500;
            letter-spacing: -0.02em;
            color: {COLORS['text']};
        }}
        
        h2 {{
            font-size: 20px !important;
        }}
        
        h3 {{
            font-size: 16px !important;
        }}
        
        /* Regular text */
        p, .stMarkdown {{
            font-size: 14px;
            line-height: 1.5;
        }}
        
        /* Hyperliquid green buttons */
        .stButton > button {{
            background-color: {COLORS['primary']} !important;
            color: {COLORS['background']} !important;
            border: none !important;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 16px;
            transition: all 0.2s;
            border-radius: 6px;
        }}
        
        .stButton > button:hover {{
            background-color: {COLORS['primary_dark']} !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(45, 225, 154, 0.25);
        }}
        
        /* Secondary buttons */
        [data-testid="baseButton-secondary"] {{
            background-color: {COLORS['background_light']} !important;
            color: {COLORS['text']} !important;
            border: 1px solid #333 !important;
            font-weight: 600;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stMultiSelect > div > div > div {{
            background-color: {COLORS['background_light']};
            color: {COLORS['text']};
            border: 1px solid #333;
            font-size: 14px;
        }}
        
        /* Metrics */
        [data-testid="metric-container"] {{
            background-color: {COLORS['background_light']};
            border: 1px solid #333;
            padding: 12px;
            border-radius: 8px;
        }}
        
        /* DataFrames - enhanced styling with left alignment */
        .dataframe {{
            font-size: 13px;
            border: 1px solid {COLORS['border']} !important;
            background-color: {COLORS['background']} !important;
            width: 100% !important;
            max-height: none !important;  /* Allow taller tables */
        }}
        
        /* DataFrame headers - BRIGHT WHITE and LEFT ALIGNED */
        .dataframe thead th {{
            background-color: {COLORS['background_light']} !important;
            color: {COLORS['header_text']} !important;  /* Bright white from constants */
            font-weight: 700 !important;
            text-transform: uppercase;
            font-size: 12px !important;
            letter-spacing: 0.5px;
            padding: 12px 8px !important;
            border-bottom: 2px solid {COLORS['primary']} !important;
            text-align: left !important;  /* Left align headers */
        }}
        
        /* DataFrame rows */
        .dataframe tbody tr {{
            border-bottom: 1px solid {COLORS['background_light']} !important;
            transition: all 0.2s;
        }}
        
        .dataframe tbody tr:hover {{
            background-color: #151515 !important;
        }}
        
        /* DataFrame cells - ALL LEFT ALIGNED */
        .dataframe tbody td {{
            padding: 10px 8px !important;
            color: {COLORS['text']} !important;
            text-align: left !important;  /* Left align all data */
        }}
        
        /* Alternate row coloring */
        .dataframe tbody tr:nth-child(even) {{
            background-color: #111111 !important;
        }}
        
        /* First column (Symbol) styling - bright white */
        .dataframe tbody td:first-child {{
            font-weight: 600;
            color: {COLORS['header_text']} !important;  /* Bright white for symbols */
            text-align: left !important;
        }}
        
        /* Remove Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {COLORS['background_light']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #333;
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS['primary']};
        }}

        /* Percentage column styling */
        .dataframe td:has(> div > div:contains('%')) {{
            font-weight: 600 !important;
        }}
        
        /* Positive percentage values */
        [data-testid="stDataFrame"] td:has(> div > div:contains('+')) {{
            color: {COLORS['positive']} !important;
            background-color: rgba(38, 201, 135, 0.05) !important;
        }}
        
        /* Negative percentage values */
        [data-testid="stDataFrame"] td:has(> div > div:contains('-')) {{
            color: {COLORS['negative']} !important;
            background-color: rgba(255, 75, 75, 0.05) !important;
        }}
        
        /* Orange circle for page icon */
        .stApp > header {{
            background-color: transparent !important;
        }}
        
        /* Override Streamlit's default alignment for specific columns */
        [data-testid="stDataFrame"] {{
            text-align: left !important;
        }}
        
        [data-testid="stDataFrame"] th {{
            text-align: left !important;
            color: {COLORS['header_text']} !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stDataFrame"] td {{
            text-align: left !important;
        }}
        
        /* Allow dataframe to expand vertically */
        [data-testid="stDataFrame"] > div {{
            max-height: none !important;
        }}
        
        div[data-testid="stDataFrameContainer"] {{
            max-height: none !important;
        }}
        
        /* Ensure the metrics table section has enough space */
        .element-container:has([data-testid="stDataFrame"]) {{
            max-height: none !important;
            overflow: visible !important;
        }}
    </style>
    """

def apply_theme():
    """Apply the dashboard theme"""
    import streamlit as st
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)