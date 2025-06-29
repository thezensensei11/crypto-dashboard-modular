"""
Shock Analysis Tab - Portfolio stress testing based on BTC movements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

from crypto_dashboard_modular.config import Settings, INTERVALS
from crypto_dashboard_modular.data.models import MetricConfig
from crypto_dashboard_modular.utils.formatting import format_price, format_percentage

logger = logging.getLogger(__name__)

class ShockTab:
    """Tab for shock analysis and portfolio stress testing"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.metric_options = {
            'beta': 'Beta',
            'upside_beta': 'Upside Beta',
            'downside_beta': 'Downside Beta'
        }
        
    def render(self):
        """Render the shock analysis tab"""
        st.header("Shock Analysis")
        
        st.markdown("""
        Simulate portfolio performance under various BTC price shocks.
        Uses beta metrics to estimate position movements based on historical relationships.
        """)
        
        # Check prerequisites
        if not self._check_prerequisites():
            return
            
        st.divider()
        
        # Portfolio configuration
        portfolio_config = self._render_portfolio_config()
        
        if portfolio_config and len(portfolio_config['positions']) > 0:
            # Metric configuration
            metric_config = self._render_metric_config()
            
            # Shock controls
            shock_range = self._render_shock_controls()
            
            # Run analysis button
            if st.button("**Run Shock Analysis**", type="primary", use_container_width=True):
                self._run_shock_analysis(portfolio_config, metric_config, shock_range)
    
    def _check_prerequisites(self):
        """Check if prerequisites are met"""
        if 'engine' not in st.session_state:
            st.error("Metrics engine not initialized. Please refresh the page.")
            return False
            
        if 'collector' not in st.session_state:
            st.error("Data collector not initialized. Please refresh the page.")
            return False
            
        return True
    
    def _render_portfolio_config(self) -> Dict:
        """Render portfolio configuration section"""
        st.subheader("Portfolio Configuration")
        
        # Initialize portfolio state
        if 'shock_portfolio' not in st.session_state:
            st.session_state.shock_portfolio = []
        
        # Add position form
        with st.expander("Add Position", expanded=True):
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            
            with col1:
                # Get available symbols
                available_symbols = sorted(set(
                    st.session_state.get('universe', []) + 
                    ['ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT']
                ))
                
                symbol = st.selectbox(
                    "Symbol",
                    options=[""] + available_symbols,
                    key="shock_symbol"
                )
            
            with col2:
                direction = st.radio(
                    "Direction",
                    ["Long", "Short"],
                    key="shock_direction"
                )
            
            with col3:
                amount = st.number_input(
                    "Exposure (USD)",
                    min_value=100.0,
                    max_value=10000000.0,
                    value=10000.0,
                    step=1000.0,
                    key="shock_amount"
                )
            
            with col4:
                st.write("")  # Spacing
                if st.button("**Add Position**", use_container_width=True):
                    if symbol and symbol != 'BTCUSDT':  # Don't allow BTC in portfolio
                        position = {
                            'symbol': symbol,
                            'direction': direction.lower(),
                            'amount': amount
                        }
                        st.session_state.shock_portfolio.append(position)
                        st.rerun()
                    elif symbol == 'BTCUSDT':
                        st.error("Cannot add BTC to portfolio (it's the shock source)")
        
        # Display current portfolio
        if st.session_state.shock_portfolio:
            st.subheader("Current Portfolio")
            
            # Portfolio summary
            total_long = sum(p['amount'] for p in st.session_state.shock_portfolio if p['direction'] == 'long')
            total_short = sum(p['amount'] for p in st.session_state.shock_portfolio if p['direction'] == 'short')
            total_exposure = total_long + total_short
            net_exposure = total_long - total_short
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Long Exposure", f"${total_long:,.0f}")
            with col2:
                st.metric("Short Exposure", f"${total_short:,.0f}")
            with col3:
                st.metric("Total Exposure", f"${total_exposure:,.0f}")
            with col4:
                st.metric("Net Exposure", f"${net_exposure:,.0f}")
            
            # Position table
            positions_data = []
            for idx, position in enumerate(st.session_state.shock_portfolio):
                positions_data.append({
                    'Symbol': position['symbol'],
                    'Direction': position['direction'].upper(),
                    'Exposure': f"${position['amount']:,.0f}",
                    'Action': idx
                })
            
            # Display positions
            for idx, position in enumerate(st.session_state.shock_portfolio):
                col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
                
                with col1:
                    st.write(f"**{position['symbol']}**")
                with col2:
                    color = "green" if position['direction'] == 'long' else "red"
                    st.markdown(f"<span style='color: {color};'>{position['direction'].upper()}</span>", unsafe_allow_html=True)
                with col3:
                    st.write(f"${position['amount']:,.0f}")
                with col4:
                    if st.button("Remove", key=f"shock_remove_{idx}"):
                        st.session_state.shock_portfolio.pop(idx)
                        st.rerun()
            
            # Clear button
            if st.button("**Clear All Positions**", use_container_width=True):
                st.session_state.shock_portfolio = []
                st.rerun()
            
            return {
                'positions': st.session_state.shock_portfolio,
                'total_exposure': total_exposure,
                'net_exposure': net_exposure
            }
        else:
            st.info("Add positions to create a portfolio for shock analysis.")
            return None
    
    def _render_metric_config(self) -> Dict:
        """Render metric configuration section"""
        st.divider()
        st.subheader("Beta Calculation Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Metric selection
            metric_type = st.selectbox(
                "Reference Metric",
                options=list(self.metric_options.keys()),
                format_func=lambda x: self.metric_options[x],
                help="Which beta metric to use for shock calculations",
                key="shock_metric"
            )
            
            # Date range type
            date_range_type = st.radio(
                "Date Range",
                ["Lookback Days", "Custom Range"],
                key="shock_date_type"
            )
        
        with col2:
            # Interval selection
            interval = st.selectbox(
                "Candle Interval",
                options=INTERVALS,
                index=INTERVALS.index('15m'),
                help="Data interval for beta calculation",
                key="shock_interval"
            )
            
            # Date configuration
            if date_range_type == "Lookback Days":
                lookback_days = st.number_input(
                    "Lookback Days",
                    min_value=7,
                    max_value=365,
                    value=30,
                    key="shock_lookback"
                )
                start_date = None
                end_date = None
            else:
                end_date = datetime.now().date()
                start_date = st.date_input(
                    "Start Date",
                    value=end_date - timedelta(days=30),
                    key="shock_start"
                )
                end_date = st.date_input(
                    "End Date",
                    value=end_date,
                    key="shock_end"
                )
                lookback_days = None
        
        with col3:
            st.write("**Calculation Info**")
            st.info(f"""
            **{self.metric_options[metric_type]}**
            
            Measures how the asset moves 
            relative to BTC movements.
            
            β = 1.5 means 1.5x BTC moves
            """)
        
        return {
            'metric': metric_type,
            'interval': interval,
            'lookback_days': lookback_days,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def _render_shock_controls(self) -> Tuple[float, float]:
        """Render shock simulation controls"""
        st.divider()
        st.subheader("Shock Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Shock range slider
            shock_range = st.slider(
                "BTC Price Shock Range (%)",
                min_value=-10.0,
                max_value=10.0,
                value=(-5.0, 5.0),
                step=0.5,
                help="Simulate BTC price movements within this range"
            )
            
            # Quick presets
            st.write("**Quick Scenarios**")
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                if st.button("Crash (-10%)", use_container_width=True):
                    shock_range = (-10.0, -5.0)
            with col1b:
                if st.button("Normal (±5%)", use_container_width=True):
                    shock_range = (-5.0, 5.0)
            with col1c:
                if st.button("Rally (+10%)", use_container_width=True):
                    shock_range = (5.0, 10.0)
        
        with col2:
            # Display options
            st.write("**Display Options**")
            
            show_position_details = st.checkbox(
                "Show position-level details",
                value=True,
                key="shock_show_positions"
            )
            
            show_beta_table = st.checkbox(
                "Show beta values table",
                value=True,
                key="shock_show_betas"
            )
            
            # Info box
            st.info("""
            **How it works:**
            1. Calculates beta for each position vs BTC
            2. Applies BTC shock: Position Move = Beta × BTC Shock
            3. Shows portfolio P&L across shock range
            """)
        
        return shock_range
    
    def _run_shock_analysis(self, portfolio_config: Dict, metric_config: Dict, shock_range: Tuple[float, float]):
        """Run the shock analysis simulation"""
        with st.spinner("Running shock analysis..."):
            try:
                # Calculate betas for all positions
                betas = self._calculate_betas(portfolio_config['positions'], metric_config)
                
                if not betas:
                    st.error("Failed to calculate betas. Please check data availability.")
                    return
                
                # Generate shock scenarios
                shock_results = self._simulate_shocks(
                    portfolio_config['positions'],
                    betas,
                    shock_range
                )
                
                # Display results
                self._display_results(
                    shock_results,
                    betas,
                    portfolio_config,
                    shock_range
                )
                
            except Exception as e:
                st.error(f"Error running shock analysis: {str(e)}")
                logger.error(f"Shock analysis error: {e}", exc_info=True)
    
    def _calculate_betas(self, positions: List[Dict], metric_config: Dict) -> Dict[str, float]:
        """Calculate beta metrics for all positions"""
        betas = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        unique_symbols = list(set(p['symbol'] for p in positions))
        
        for idx, symbol in enumerate(unique_symbols):
            progress = (idx + 1) / len(unique_symbols)
            progress_bar.progress(progress)
            status_text.text(f"Calculating {metric_config['metric']} for {symbol}...")
            
            try:
                # Create metric config
                config = MetricConfig(
                    name=f"{symbol}_{metric_config['metric']}",
                    metric=metric_config['metric'],
                    interval=metric_config['interval'],
                    lookback_days=metric_config['lookback_days'],
                    start_date=metric_config['start_date'],
                    end_date=metric_config['end_date']
                )
                
                # Calculate beta
                beta_value = st.session_state.engine.calculate_metric(
                    symbol=symbol,
                    metric_config=config,
                    force_cache=False
                )
                
                betas[symbol] = beta_value if not pd.isna(beta_value) else 0.0
                
            except Exception as e:
                logger.error(f"Error calculating beta for {symbol}: {e}")
                betas[symbol] = 0.0
        
        progress_bar.empty()
        status_text.empty()
        
        return betas
    
    def _simulate_shocks(self, positions: List[Dict], betas: Dict[str, float], 
                        shock_range: Tuple[float, float]) -> pd.DataFrame:
        """Simulate portfolio performance across shock range"""
        # Generate shock scenarios
        shock_points = np.linspace(shock_range[0], shock_range[1], 21)
        
        results = []
        
        for shock in shock_points:
            portfolio_pnl = 0.0
            position_pnls = {}
            
            for position in positions:
                symbol = position['symbol']
                beta = betas.get(symbol, 0.0)
                
                # Calculate expected move based on beta
                expected_move = beta * shock / 100  # Convert percentage to decimal
                
                # Calculate P&L based on direction
                if position['direction'] == 'long':
                    position_pnl = position['amount'] * expected_move
                else:  # short
                    position_pnl = -position['amount'] * expected_move
                
                position_pnls[f"{symbol}_{position['direction']}"] = position_pnl
                portfolio_pnl += position_pnl
            
            results.append({
                'btc_shock': shock,
                'portfolio_pnl': portfolio_pnl,
                'portfolio_return': (portfolio_pnl / position['amount']) * 100 if position['amount'] > 0 else 0,
                **position_pnls
            })
        
        return pd.DataFrame(results)
    
    def _display_results(self, shock_results: pd.DataFrame, betas: Dict[str, float],
                        portfolio_config: Dict, shock_range: Tuple[float, float]):
        """Display shock analysis results"""
        st.divider()
        st.subheader("Shock Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Find worst and best case scenarios
        worst_idx = shock_results['portfolio_pnl'].idxmin()
        best_idx = shock_results['portfolio_pnl'].idxmax()
        
        worst_case = shock_results.iloc[worst_idx]
        best_case = shock_results.iloc[best_idx]
        
        with col1:
            st.metric(
                "Worst Case P&L",
                f"${worst_case['portfolio_pnl']:,.0f}",
                f"at BTC {worst_case['btc_shock']:+.1f}%"
            )
        
        with col2:
            st.metric(
                "Best Case P&L",
                f"${best_case['portfolio_pnl']:,.0f}",
                f"at BTC {best_case['btc_shock']:+.1f}%"
            )
        
        with col3:
            # P&L at BTC flat
            flat_pnl = shock_results[shock_results['btc_shock'] == 0.0]['portfolio_pnl'].iloc[0] if 0.0 in shock_results['btc_shock'].values else 0
            st.metric(
                "P&L at BTC Flat",
                f"${flat_pnl:,.0f}"
            )
        
        with col4:
            # Portfolio beta (weighted average)
            portfolio_beta = self._calculate_portfolio_beta(portfolio_config['positions'], betas)
            st.metric(
                "Portfolio Beta",
                f"{portfolio_beta:.3f}",
                "vs BTC"
            )
        
        # Main chart - P&L curve
        st.subheader("Portfolio P&L Curve")
        fig = self._create_pnl_chart(shock_results, portfolio_config)
        st.plotly_chart(fig, use_container_width=True)
        
        # Beta values table
        if st.session_state.get('shock_show_betas', True):
            st.subheader("Position Beta Values")
            self._display_beta_table(portfolio_config['positions'], betas)
        
        # Position details
        if st.session_state.get('shock_show_positions', True):
            st.subheader("Position-Level Impact")
            self._display_position_impacts(shock_results, portfolio_config['positions'])
        
        # Scenario table
        st.subheader("Scenario Analysis")
        self._display_scenario_table(shock_results)
    
    def _create_pnl_chart(self, shock_results: pd.DataFrame, portfolio_config: Dict) -> go.Figure:
        """Create P&L curve chart"""
        fig = go.Figure()
        
        # Portfolio P&L curve
        fig.add_trace(go.Scatter(
            x=shock_results['btc_shock'],
            y=shock_results['portfolio_pnl'],
            mode='lines+markers',
            name='Portfolio P&L',
            line=dict(color='#2de19a', width=3),
            marker=dict(size=8),
            hovertemplate='BTC: %{x:+.1f}%<br>P&L: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add shaded regions for profit/loss
        fig.add_hrect(y0=0, y1=shock_results['portfolio_pnl'].max() * 1.1,
                     fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=shock_results['portfolio_pnl'].min() * 1.1, y1=0,
                     fillcolor="red", opacity=0.1, line_width=0)
        
        # Layout
        fig.update_layout(
            title='Portfolio P&L vs BTC Price Shock',
            xaxis_title='BTC Price Change (%)',
            yaxis_title='Portfolio P&L ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        # Add annotations for key points
        worst_idx = shock_results['portfolio_pnl'].idxmin()
        best_idx = shock_results['portfolio_pnl'].idxmax()
        
        fig.add_annotation(
            x=shock_results.iloc[worst_idx]['btc_shock'],
            y=shock_results.iloc[worst_idx]['portfolio_pnl'],
            text="Worst",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ff4b4b"
        )
        
        fig.add_annotation(
            x=shock_results.iloc[best_idx]['btc_shock'],
            y=shock_results.iloc[best_idx]['portfolio_pnl'],
            text="Best",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#2de19a"
        )
        
        return fig
    
    def _display_beta_table(self, positions: List[Dict], betas: Dict[str, float]):
        """Display beta values table"""
        beta_data = []
        
        for position in positions:
            symbol = position['symbol']
            beta = betas.get(symbol, 0.0)
            
            beta_data.append({
                'Symbol': symbol,
                'Direction': position['direction'].upper(),
                'Exposure': f"${position['amount']:,.0f}",
                'Beta': f"{beta:.3f}",
                'Effective Beta': f"{beta if position['direction'] == 'long' else -beta:.3f}"
            })
        
        beta_df = pd.DataFrame(beta_data)
        
        # Style the dataframe
        def style_direction(val):
            return f'color: {"#26c987" if val == "LONG" else "#ff4b4b"}; font-weight: bold;'
        
        def style_beta(val):
            try:
                beta_val = float(val)
                if beta_val > 0:
                    return 'color: #26c987;'
                elif beta_val < 0:
                    return 'color: #ff4b4b;'
                return ''
            except:
                return ''
        
        styled_df = beta_df.style.applymap(
            style_direction, subset=['Direction']
        ).applymap(
            style_beta, subset=['Effective Beta']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    def _display_position_impacts(self, shock_results: pd.DataFrame, positions: List[Dict]):
        """Display position-level impacts"""
        # Create position impact chart
        fig = go.Figure()
        
        # Add trace for each position
        for position in positions:
            pos_key = f"{position['symbol']}_{position['direction']}"
            if pos_key in shock_results.columns:
                fig.add_trace(go.Scatter(
                    x=shock_results['btc_shock'],
                    y=shock_results[pos_key],
                    mode='lines',
                    name=f"{position['symbol']} ({position['direction'].upper()})",
                    hovertemplate='%{x:+.1f}%<br>$%{y:,.0f}<extra></extra>'
                ))
        
        # Layout
        fig.update_layout(
            title='Individual Position P&L',
            xaxis_title='BTC Price Change (%)',
            yaxis_title='Position P&L ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_scenario_table(self, shock_results: pd.DataFrame):
        """Display key scenario results"""
        # Select key scenarios
        scenarios = [-10, -5, -2, 0, 2, 5, 10]
        scenario_data = []
        
        for scenario in scenarios:
            # Find closest shock value
            idx = (shock_results['btc_shock'] - scenario).abs().idxmin()
            row = shock_results.iloc[idx]
            
            scenario_data.append({
                'BTC Shock': f"{row['btc_shock']:+.1f}%",
                'Portfolio P&L': f"${row['portfolio_pnl']:,.0f}",
                'Portfolio Return': f"{row['portfolio_return']:+.2f}%"
            })
        
        scenario_df = pd.DataFrame(scenario_data)
        
        # Style the dataframe
        def style_pnl(val):
            try:
                # Extract number from formatted string
                num_str = val.replace('$', '').replace(',', '')
                if '-' in num_str:
                    return 'color: #ff4b4b; font-weight: bold;'
                elif float(num_str) > 0:
                    return 'color: #26c987; font-weight: bold;'
                return ''
            except:
                return ''
        
        styled_df = scenario_df.style.applymap(
            style_pnl, subset=['Portfolio P&L', 'Portfolio Return']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    def _calculate_portfolio_beta(self, positions: List[Dict], betas: Dict[str, float]) -> float:
        """Calculate weighted portfolio beta"""
        total_exposure = sum(p['amount'] for p in positions)
        if total_exposure == 0:
            return 0.0
        
        weighted_beta = 0.0
        for position in positions:
            symbol = position['symbol']
            beta = betas.get(symbol, 0.0)
            weight = position['amount'] / total_exposure
            
            # Adjust for direction
            if position['direction'] == 'long':
                weighted_beta += beta * weight
            else:  # short
                weighted_beta -= beta * weight
        
        return weighted_beta