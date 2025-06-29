"""
Sidebar component for the dashboard - DuckDB version
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, Optional
import asyncio

from crypto_dashboard_modular.config.constants import HYPERLIQUID_API_URL

logger = logging.getLogger(__name__)


class SidebarComponent:
    """Manages the sidebar content - DuckDB version"""
    
    def __init__(self):
        self.price_data = {'BTC': 0, 'ETH': 0, 'SOL': 0, 'HYPE': 0}
        self.btc_refresh_interval_minutes = 15
        self.dubai_offset_hours = 4
    
    def render(self, universe_size: int, columns_count: int, db_stats: Dict):
        """Render the complete sidebar with database stats"""
        with st.sidebar:
            # Live prices fragment
            self._render_live_prices()
            
            st.divider()
            
            # BTC metrics
            self._render_btc_metrics()
            
            st.divider()
            
            # Dashboard stats with DuckDB info
            self._render_stats(universe_size, columns_count, db_stats)
    
    def _render_live_prices(self):
        """Render live price display"""
        price_container = st.container()
        
        with price_container:
            @st.fragment(run_every=2)
            def display_live_prices():
                self._fetch_prices()
                
                btc_price = self.price_data.get('BTC', 0)
                eth_price = self.price_data.get('ETH', 0)
                sol_price = self.price_data.get('SOL', 0)
                hype_price = self.price_data.get('HYPE', 0)
                
                # Display prices
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
                    <div class="price-symbol">ETH</div>
                    <div class="price-value price-eth">
                        {"Loading..." if eth_price == 0 else f"${eth_price:,.2f}"}
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
                        {"Loading..." if hype_price == 0 else f"${hype_price:,.2f}"}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            display_live_prices()
    
    def _render_btc_metrics(self):
        """Render BTC metrics fragment"""
        metrics_container = st.container()
        
        with metrics_container:
            @st.fragment(run_every=self.btc_refresh_interval_minutes * 60)
            def display_btc_metrics():
                if 'collector' in st.session_state:
                    try:
                        # Fetch BTC data
                        btc_data = st.session_state.collector.get_price_data(
                            'BTCUSDT', '15m', lookback_days=1
                        )
                        
                        if not btc_data.empty:
                            # Calculate metrics
                            current_price = btc_data.iloc[-1]['close']
                            
                            # 1h change
                            h1_data = btc_data[btc_data['timestamp'] >= datetime.now(timezone.utc) - timedelta(hours=1)]
                            h1_change = ((current_price / h1_data.iloc[0]['close']) - 1) * 100 if len(h1_data) > 1 else 0
                            
                            # 4h change  
                            h4_data = btc_data[btc_data['timestamp'] >= datetime.now(timezone.utc) - timedelta(hours=4)]
                            h4_change = ((current_price / h4_data.iloc[0]['close']) - 1) * 100 if len(h4_data) > 1 else 0
                            
                            # Dubai time calculations
                            utc_now = datetime.now(timezone.utc)
                            dubai_now = utc_now + timedelta(hours=self.dubai_offset_hours)
                            dubai_9am = dubai_now.replace(hour=9, minute=0, second=0, microsecond=0)
                            
                            if dubai_now.hour < 9:
                                dubai_9am -= timedelta(days=1)
                            
                            dubai_9am_utc = dubai_9am - timedelta(hours=self.dubai_offset_hours)
                            dubai_data = btc_data[btc_data['timestamp'] >= dubai_9am_utc]
                            dubai_change = ((current_price / dubai_data.iloc[0]['close']) - 1) * 100 if len(dubai_data) > 1 else 0
                            
                            # Display metrics
                            st.markdown("**BTC Metrics**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("1h", f"{h1_change:+.2f}%", delta=None)
                                st.metric("Dubai 9AM", f"{dubai_change:+.2f}%", delta=None)
                            with col2:
                                st.metric("4h", f"{h4_change:+.2f}%", delta=None)
                    except Exception as e:
                        logger.error(f"Error calculating BTC metrics: {e}")
                        st.info("BTC metrics loading...")
            
            display_btc_metrics()
    
    def _render_stats(self, universe_size: int, columns_count: int, db_stats: Dict):
        """Render dashboard statistics with DuckDB info"""
        st.markdown("**Dashboard Stats**")
        
        # Dashboard configuration
        st.info(f"""
        📊 **Configuration**
        - Symbols: {universe_size}
        - Columns: {columns_count}
        """)
        
        # Database statistics
        st.info(f"""
        🦆 **DuckDB Storage**
        - Symbols: {db_stats.get('total_symbols', 0)}
        - Total Rows: {db_stats.get('total_rows', 0):,}
        - DB Size: {db_stats.get('total_size_mb', 0):.1f} MB
        - API Calls: {db_stats.get('api_calls', 0)}
        - DB Hits: {db_stats.get('cache_hits', 0)}
        """)
    
    def _fetch_prices(self):
        """Fetch live prices from exchanges"""
        try:
            # Fetch from Binance
            binance_url = "https://fapi.binance.com/fapi/v1/ticker/price"
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            
            for symbol in symbols:
                response = requests.get(binance_url, params={'symbol': symbol}, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    price = float(data['price'])
                    
                    if symbol == 'BTCUSDT':
                        self.price_data['BTC'] = price
                    elif symbol == 'ETHUSDT':
                        self.price_data['ETH'] = price
                    elif symbol == 'SOLUSDT':
                        self.price_data['SOL'] = price
            
            # Fetch HYPE from Hyperliquid
            hl_response = requests.post(
                HYPERLIQUID_API_URL,
                json={"type": "allMids"},
                timeout=2
            )
            if hl_response.status_code == 200:
                hl_data = hl_response.json()
                if 'HYPE' in hl_data:
                    self.price_data['HYPE'] = float(hl_data['HYPE'])
                    
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")