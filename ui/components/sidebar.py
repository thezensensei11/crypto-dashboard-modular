"""
Sidebar component for the dashboard
"""

import streamlit as st
import requests
from datetime import datetime
import logging
from typing import Dict

from config.constants import HYPERLIQUID_API_URL

logger = logging.getLogger(__name__)

class SidebarComponent:
    """Manages the sidebar content"""
    
    def __init__(self):
        self.price_data = {'BTC': 0, 'SOL': 0, 'HYPE': 0}
    
    def render(self, universe_size: int, columns_count: int, cache_stats: Dict):
        """Render the complete sidebar"""
        with st.sidebar:
            # Live prices fragment
            self._render_live_prices()
            
            st.divider()
            
            # Dashboard stats
            self._render_stats(universe_size, columns_count, cache_stats)
    
    def _render_live_prices(self):
        """Render live price display"""
        @st.fragment(run_every=2)
        def display_live_prices():
            self._fetch_prices()
            
            btc_price = self.price_data.get('BTC', 0)
            sol_price = self.price_data.get('SOL', 0)
            hype_price = self.price_data.get('HYPE', 0)
            
            # Display prices using separate containers
            st.markdown(f"""
            <div class="price-container">
                <div class="price-symbol">BTC</div>
                <div class="price-value price-btc">
                    {"Loading..." if btc_price == 0 else f"${btc_price:,.2f}"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="price-container">
                <div class="price-symbol">SOL</div>
                <div class="price-value price-sol">
                    {"Loading..." if sol_price == 0 else f"${sol_price:,.2f}"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="price-container">
                <div class="price-symbol">HYPE</div>
                <div class="price-value price-hype">
                    {"Loading..." if hype_price == 0 else f"${hype_price:,.4f}"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="text-align: center; font-size: 11px; color: #666; margin-top: 8px;">
                Updated: {datetime.now().strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        
        display_live_prices()
    
    def _fetch_prices(self):
        """Fetch live prices from Hyperliquid API"""
        try:
            request_data = {"type": "allMids"}
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(HYPERLIQUID_API_URL, json=request_data, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    for symbol in ['BTC', 'SOL', 'HYPE']:
                        if symbol in data:
                            self.price_data[symbol] = float(data[symbol])
                            
        except Exception as e:
            logger.error(f"Error fetching Hyperliquid prices: {e}")
    
    def _render_stats(self, universe_size: int, columns_count: int, cache_stats: Dict):
        """Render dashboard statistics"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Universe Size", universe_size)
            st.metric("Columns", columns_count)
        
        with col2:
            st.metric("Cached Symbols", cache_stats.get('total_symbols', 0))
            st.metric("Cache Size", f"{cache_stats.get('total_size_mb', 0):.1f} MB")
