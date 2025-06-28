"""
Universe Manager Tab
"""

import streamlit as st
from typing import List
import logging

from config import Settings, POPULAR_SYMBOLS
from data.collector import BinanceDataCollector

logger = logging.getLogger(__name__)

class UniverseTab:
    """Manages the symbol universe configuration"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.collector = st.session_state.get('collector', BinanceDataCollector())
    
    def render(self):
        """Render the universe manager tab"""
        st.header("Symbol Universe Manager")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_add_symbols()
        
        with col2:
            self._render_current_universe()
    
    def _render_add_symbols(self):
        """Render the add symbols section"""
        st.subheader("Add Symbols")
        
        # Load symbols from Binance
        if st.button("**Load Symbols from Binance**", use_container_width=True):
            with st.spinner("Fetching symbols from Binance..."):
                symbols = self._fetch_available_symbols()
                if symbols:
                    st.session_state.symbols_loaded = True
                    st.session_state.all_symbols = symbols
                    st.success(f"Loaded {len(symbols)} symbols")
        
        # Symbol selector
        if st.session_state.get('symbols_loaded', False) and st.session_state.get('all_symbols'):
            all_symbols = st.session_state.all_symbols
            current_universe = st.session_state.universe
            available_symbols = [s for s in all_symbols if s not in current_universe]
            
            symbols_to_add = st.multiselect(
                "Select symbols to add:",
                available_symbols,
                placeholder="Choose symbols..."
            )
            
            if st.button("**Add Selected**", use_container_width=True):
                if symbols_to_add:
                    self._add_symbols(symbols_to_add)
        else:
            st.info("Click 'Load Symbols from Binance' to fetch available symbols, or add manually below.")
        
        st.divider()
        
        # Manual symbol input
        st.subheader("Add Symbols Manually")
        manual_symbol = st.text_input("Enter symbol (e.g., BTCUSDT):")
        if st.button("**Add Symbol**", use_container_width=True):
            if manual_symbol:
                self._add_symbols([manual_symbol.upper()])
        
        # Quick add popular symbols
        st.divider()
        st.subheader("Quick Add Popular Symbols")
        
        if st.button("**Add Top 5 Symbols**", use_container_width=True):
            new_symbols = [s for s in POPULAR_SYMBOLS if s not in st.session_state.universe]
            if new_symbols:
                self._add_symbols(new_symbols)
    
    def _render_current_universe(self):
        """Render the current universe display"""
        st.subheader("Current Universe")
        
        universe = st.session_state.universe
        
        if universe:
            search = st.text_input("🔍 Search symbols", placeholder="Type to filter...")
            
            displayed_symbols = universe
            if search:
                displayed_symbols = [s for s in displayed_symbols if search.upper() in s]
            
            st.caption(f"Showing {len(displayed_symbols)} of {len(universe)} symbols")
            
            # Display symbols in a grid
            container = st.container()
            with container:
                for i in range(0, len(displayed_symbols), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(displayed_symbols):
                            symbol = displayed_symbols[i + j]
                            with col:
                                if st.button(f"**× {symbol}**", key=f"remove_{symbol}", use_container_width=True):
                                    self._remove_symbol(symbol)
        else:
            st.info("No symbols in universe. Add some symbols to get started!")
    
    def _fetch_available_symbols(self) -> List[str]:
        """Fetch available symbols from Binance"""
        @st.cache_data(ttl=3600)
        def get_symbols():
            try:
                return self.collector.get_available_symbols()
            except Exception as e:
                logger.error(f"Error fetching symbols: {e}")
                st.error(f"Error fetching symbols: {e}")
                return []
        
        return get_symbols()
    
    def _add_symbols(self, symbols: List[str]):
        """Add symbols to universe"""
        added = []
        for symbol in symbols:
            if symbol not in st.session_state.universe:
                st.session_state.universe.append(symbol)
                added.append(symbol)
        
        if added:
            st.session_state.universe = sorted(list(set(st.session_state.universe)))
            self.settings.save_universe(st.session_state.universe)
            st.success(f"Added {len(added)} symbols")
            st.rerun()
    
    def _remove_symbol(self, symbol: str):
        """Remove a symbol from universe"""
        if symbol in st.session_state.universe:
            st.session_state.universe.remove(symbol)
            self.settings.save_universe(st.session_state.universe)
            st.rerun()
