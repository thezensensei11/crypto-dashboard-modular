"""
Backtester Tab - Historical portfolio performance simulation with enhanced data handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

from crypto_dashboard_modular.config import Settings
from crypto_dashboard_modular.utils.formatting import format_price, format_percentage, format_duration

logger = logging.getLogger(__name__)

class BacktesterTab:
    """Tab for backtesting portfolio strategies"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.timeframe_options = {
            '15m': '15 minutes',
            '1h': '1 hour', 
            '4h': '4 hours',
            '1d': '1 day',
            '1w': '1 week'
        }
        
    def render(self):
        """Render the backtester tab"""
        st.header("Portfolio Backtester")
        
        st.markdown("""
        Test historical performance of a portfolio with long and short positions.
        Data will be automatically fetched if not available in cache.
        """)
        
        # Check prerequisites
        if not self._check_prerequisites():
            return
            
        st.divider()
        
        # Portfolio configuration section
        portfolio_config = self._render_portfolio_config()
        
        if portfolio_config and len(portfolio_config['positions']) > 0:
            # Backtest controls
            backtest_params = self._render_backtest_controls()
            
            if st.button("**Run Backtest**", type="primary", use_container_width=True):
                self._run_backtest(portfolio_config, backtest_params)
    
    def _check_prerequisites(self):
        """Check if prerequisites are met"""
        if 'data_manager' not in st.session_state:
            st.error("Data manager not initialized. Please refresh the page.")
            return False
            
        if 'collector' not in st.session_state:
            st.error("Data collector not initialized. Please refresh the page.")
            return False
            
        return True
    
    def _render_portfolio_config(self) -> Dict:
        """Render portfolio configuration section"""
        st.subheader("Portfolio Configuration")
        
        # Initialize portfolio state
        if 'backtest_portfolio' not in st.session_state:
            st.session_state.backtest_portfolio = []
        
        # Get available symbols - combine universe with common symbols
        available_symbols = sorted(set(
            st.session_state.get('universe', []) + 
            ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'HYPEUSDT']
        ))
        
        # Add position form
        with st.expander("Add Position", expanded=True):
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            
            with col1:
                # Allow any symbol input
                input_method = st.radio(
                    "Symbol Input",
                    ["Select from list", "Enter manually"],
                    horizontal=True,
                    key="bt_input_method"
                )
                
                if input_method == "Select from list":
                    symbol = st.selectbox(
                        "Symbol",
                        options=[""] + available_symbols,
                        key="bt_symbol_select"
                    )
                else:
                    symbol = st.text_input(
                        "Enter Symbol (e.g., BTCUSDT)",
                        key="bt_symbol_manual"
                    ).upper()
            
            with col2:
                direction = st.radio(
                    "Direction",
                    ["Long", "Short"],
                    key="bt_direction"
                )
            
            with col3:
                amount = st.number_input(
                    "Amount ($)",
                    min_value=100.0,
                    max_value=10000000.0,
                    value=10000.0,
                    step=1000.0,
                    key="bt_amount"
                )
            
            with col4:
                st.write("")  # Spacing
                if st.button("**Add to Portfolio**", use_container_width=True):
                    if symbol and symbol.endswith('USDT'):
                        position = {
                            'symbol': symbol,
                            'direction': direction.lower(),
                            'amount': amount
                        }
                        st.session_state.backtest_portfolio.append(position)
                        st.rerun()
                    elif symbol:
                        st.error("Symbol must end with USDT")
        
        # Display current portfolio
        if st.session_state.backtest_portfolio:
            st.subheader("Current Portfolio")
            
            # Portfolio summary
            total_long = sum(p['amount'] for p in st.session_state.backtest_portfolio if p['direction'] == 'long')
            total_short = sum(p['amount'] for p in st.session_state.backtest_portfolio if p['direction'] == 'short')
            total_capital = total_long + total_short
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Long", f"${total_long:,.0f}")
            with col2:
                st.metric("Total Short", f"${total_short:,.0f}")
            with col3:
                st.metric("Total Capital", f"${total_capital:,.0f}")
            
            # Position table
            for idx, position in enumerate(st.session_state.backtest_portfolio):
                col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
                
                with col1:
                    st.write(f"**{position['symbol']}**")
                with col2:
                    color = "green" if position['direction'] == 'long' else "red"
                    st.markdown(f"<span style='color: {color};'>{position['direction'].upper()}</span>", unsafe_allow_html=True)
                with col3:
                    st.write(f"${position['amount']:,.0f}")
                with col4:
                    if st.button("Remove", key=f"remove_{idx}"):
                        st.session_state.backtest_portfolio.pop(idx)
                        st.rerun()
            
            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("**Clear Portfolio**", use_container_width=True):
                    st.session_state.backtest_portfolio = []
                    st.rerun()
            
            return {
                'positions': st.session_state.backtest_portfolio,
                'total_capital': total_capital
            }
        else:
            st.info("Add positions to create a portfolio for backtesting.")
            return None
    
    def _render_backtest_controls(self) -> Dict:
        """Render backtest control parameters"""
        st.divider()
        st.subheader("Backtest Parameters")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Date range
            end_date = datetime.now().date()
            start_date = st.date_input(
                "Start Date",
                value=end_date - timedelta(days=30),
                max_value=end_date,
                key="bt_start_date"
            )
            
            end_date = st.date_input(
                "End Date",
                value=end_date,
                min_value=start_date,
                max_value=datetime.now().date(),
                key="bt_end_date"
            )
        
        with col2:
            # Benchmark selection - allow any symbol
            st.write("**Benchmarks**")
            
            # Get all unique symbols
            all_symbols = sorted(set(
                [p['symbol'] for p in st.session_state.backtest_portfolio] +
                st.session_state.get('universe', []) +
                ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'HYPEUSDT']
            ))
            
            # First benchmark
            benchmark1 = st.selectbox(
                "Primary Benchmark",
                options=all_symbols,
                index=all_symbols.index('BTCUSDT') if 'BTCUSDT' in all_symbols else 0,
                key="bt_benchmark1"
            )
            
            # Second benchmark (optional)
            use_second_benchmark = st.checkbox("Add second benchmark", key="bt_use_bench2")
            benchmark2 = None
            if use_second_benchmark:
                benchmark2 = st.selectbox(
                    "Secondary Benchmark",
                    options=[s for s in all_symbols if s != benchmark1],
                    key="bt_benchmark2"
                )
        
        with col3:
            # Chart options
            chart_type = st.radio(
                "Chart Type",
                ["Line", "Candlestick"],
                key="bt_chart_type"
            )
            
            # Timeframe for candlestick aggregation
            if chart_type == "Candlestick":
                candle_timeframe = st.selectbox(
                    "Candle Timeframe",
                    options=list(self.timeframe_options.keys()),
                    format_func=lambda x: self.timeframe_options[x],
                    index=2,  # Default to 4h
                    key="bt_candle_tf"
                )
            else:
                candle_timeframe = '1d'
            
            # Additional options
            include_fees = st.checkbox(
                "Include Trading Fees (0.05%)",
                value=False
            )
            
            show_drawdown = st.checkbox(
                "Show Drawdown Chart",
                value=True
            )
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'benchmark1': benchmark1,
            'benchmark2': benchmark2,
            'chart_type': chart_type,
            'candle_timeframe': candle_timeframe,
            'include_fees': include_fees,
            'show_drawdown': show_drawdown
        }
    
    def _run_backtest(self, portfolio_config: Dict, backtest_params: Dict):
        """Run the backtest simulation"""
        with st.spinner("Running backtest..."):
            try:
                # Collect all symbols needed
                all_symbols = [p['symbol'] for p in portfolio_config['positions']]
                if backtest_params['benchmark1']:
                    all_symbols.append(backtest_params['benchmark1'])
                if backtest_params['benchmark2']:
                    all_symbols.append(backtest_params['benchmark2'])
                all_symbols = list(set(all_symbols))
                
                # Fetch price data for all symbols with automatic API calls if needed
                price_data = self._fetch_price_data_smart(
                    all_symbols,
                    backtest_params['start_date'],
                    backtest_params['end_date']
                )
                
                if not price_data:
                    st.error("Failed to fetch price data.")
                    return
                
                # Calculate portfolio performance
                results = self._calculate_performance(
                    price_data,
                    portfolio_config,
                    backtest_params
                )
                
                # Display results
                self._display_results(results, backtest_params)
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                logger.error(f"Backtest error: {e}", exc_info=True)
    
    def _fetch_price_data_smart(self, symbols: List[str], 
                               start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch 15-minute price data with automatic API calls for missing data"""
        price_data = {}
        
        # Convert dates to datetime with timezone
        start_datetime = pd.Timestamp(start_date).tz_localize('UTC')
        end_datetime = pd.Timestamp(end_date).tz_localize('UTC') + pd.Timedelta(days=1)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate lookback days
        lookback_days = (end_date - start_date).days + 1
        
        for idx, symbol in enumerate(symbols):
            progress = (idx + 1) / len(symbols)
            progress_bar.progress(progress)
            status_text.text(f"Fetching data for {symbol}...")
            
            # First try to load from cache
            cached_data = st.session_state.data_manager.load_data(
                symbol=symbol,
                interval='15m',
                start_date=start_datetime,
                end_date=end_datetime
            )
            
            # Check if we have complete data
            if cached_data is not None and not cached_data.empty:
                # Verify we have data for the full range
                data_start = cached_data['timestamp'].min()
                data_end = cached_data['timestamp'].max()
                
                if data_start <= start_datetime and data_end >= end_datetime - pd.Timedelta(hours=1):
                    price_data[symbol] = cached_data
                    status_text.text(f"Loaded {symbol} from cache")
                    continue
            
            # Need to fetch from API
            status_text.text(f"Fetching {symbol} from API...")
            
            try:
                # Use the collector to fetch with smart caching
                api_data = st.session_state.collector.get_price_data(
                    symbol=symbol,
                    interval='15m',
                    lookback_days=lookback_days + 7,  # Extra buffer
                    force_cache=False  # Allow API calls
                )
                
                if api_data is not None and not api_data.empty:
                    # Filter to our date range
                    api_data = api_data[
                        (api_data['timestamp'] >= start_datetime) & 
                        (api_data['timestamp'] <= end_datetime)
                    ]
                    price_data[symbol] = api_data
                    status_text.text(f"Successfully fetched {symbol}")
                else:
                    st.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                st.error(f"Error fetching {symbol}: {str(e)}")
                logger.error(f"Error fetching {symbol}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        return price_data
    
    def _calculate_performance(self, price_data: Dict[str, pd.DataFrame], 
                              portfolio_config: Dict, backtest_params: Dict) -> Dict:
        """Calculate portfolio and benchmark performance"""
        # Align all price series to common timestamps
        aligned_prices = self._align_price_data(price_data)
        
        if aligned_prices.empty:
            raise ValueError("No overlapping data found for selected symbols")
        
        # Calculate portfolio value over time
        portfolio_values = pd.Series(0.0, index=aligned_prices.index)
        position_values = {}  # Track individual positions
        
        for position in portfolio_config['positions']:
            symbol = position['symbol']
            if symbol not in aligned_prices.columns:
                continue
                
            # Calculate position value
            initial_price = aligned_prices[symbol].iloc[0]
            shares = position['amount'] / initial_price
            
            if position['direction'] == 'long':
                position_val = shares * aligned_prices[symbol]
            else:  # short
                position_val = position['amount'] * 2 - (shares * aligned_prices[symbol])
            
            position_values[f"{symbol}_{position['direction']}"] = position_val
            portfolio_values += position_val
        
        # Apply trading fees if requested
        if backtest_params['include_fees']:
            fee_cost = portfolio_config['total_capital'] * 0.0005 * 2  # Entry and exit
            portfolio_values = portfolio_values - fee_cost
        
        # Calculate benchmark performances
        benchmark_values = {}
        
        # Primary benchmark
        if backtest_params['benchmark1'] in aligned_prices.columns:
            bench1_initial = aligned_prices[backtest_params['benchmark1']].iloc[0]
            bench1_shares = portfolio_config['total_capital'] / bench1_initial
            benchmark_values['benchmark1'] = bench1_shares * aligned_prices[backtest_params['benchmark1']]
        
        # Secondary benchmark
        if backtest_params.get('benchmark2') and backtest_params['benchmark2'] in aligned_prices.columns:
            bench2_initial = aligned_prices[backtest_params['benchmark2']].iloc[0]
            bench2_shares = portfolio_config['total_capital'] / bench2_initial
            benchmark_values['benchmark2'] = bench2_shares * aligned_prices[backtest_params['benchmark2']]
        
        # Calculate returns and metrics
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        metrics = {
            'portfolio': self._calculate_metrics_for_series(portfolio_values, portfolio_returns, "Portfolio"),
            'position_values': position_values
        }
        
        # Calculate metrics for each benchmark
        for bench_name, bench_values in benchmark_values.items():
            bench_returns = bench_values.pct_change().dropna()
            bench_label = backtest_params[bench_name]
            metrics[bench_name] = self._calculate_metrics_for_series(bench_values, bench_returns, bench_label)
        
        return {
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'metrics': metrics,
            'position_values': position_values
        }
    
    def _calculate_metrics_for_series(self, values: pd.Series, returns: pd.Series, label: str) -> Dict:
        """Calculate performance metrics for a value series"""
        # Annualization factor for 15-minute data
        periods_per_year = 365 * 24 * 4  # 4 periods per hour
        
        # Basic metrics
        total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(periods_per_year) * 100
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        max_drawdown = drawdown.min()
        
        # Win Rate
        winning_periods = (returns > 0).sum()
        total_periods = len(returns)
        win_rate = (winning_periods / total_periods * 100) if total_periods > 0 else 0
        
        # Calmar Ratio (return / max drawdown)
        calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'label': label,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar,
            'num_periods': total_periods,
            'returns': returns
        }
    
    def _align_price_data(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align all price series to common timestamps"""
        aligned_data = {}
        
        for symbol, data in price_data.items():
            if not data.empty:
                # Ensure timestamp is timezone-aware
                if data['timestamp'].dt.tz is None:
                    data['timestamp'] = data['timestamp'].dt.tz_localize('UTC')
                aligned_data[symbol] = data.set_index('timestamp')['close']
        
        if not aligned_data:
            return pd.DataFrame()
        
        # Combine all series
        combined = pd.DataFrame(aligned_data)
        
        # Forward fill missing values (but not too many)
        combined = combined.fillna(method='ffill', limit=4)
        
        # Drop rows with any remaining NaN values
        combined = combined.dropna()
        
        return combined
    
    def _display_results(self, results: Dict, backtest_params: Dict):
        """Display backtest results"""
        st.divider()
        st.subheader("Backtest Results")
        
        # Metrics comparison table
        self._display_metrics_comparison(results['metrics'], backtest_params)
        
        # Equity curves
        st.subheader("Performance Chart")
        
        if backtest_params['chart_type'] == "Line":
            fig = self._create_line_chart(results, backtest_params)
        else:
            fig = self._create_candlestick_chart(results, backtest_params)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        if backtest_params['show_drawdown']:
            st.subheader("Drawdown Analysis")
            drawdown_fig = self._create_drawdown_chart(results, backtest_params)
            st.plotly_chart(drawdown_fig, use_container_width=True)
        
        # Position breakdown
        self._display_position_breakdown(results)
    
    def _display_metrics_comparison(self, metrics: Dict, backtest_params: Dict):
        """Display metrics comparison table"""
        # Create comparison dataframe
        metrics_list = []
        
        # Add portfolio metrics
        port_metrics = metrics['portfolio']
        metrics_list.append({
            'Strategy': 'Portfolio',
            'Total Return (%)': f"{port_metrics['total_return']:.2f}",
            'Volatility (%)': f"{port_metrics['volatility']:.2f}",
            'Sharpe Ratio': f"{port_metrics['sharpe_ratio']:.3f}",
            'Sortino Ratio': f"{port_metrics['sortino_ratio']:.3f}",
            'Max Drawdown (%)': f"{port_metrics['max_drawdown']:.2f}",
            'Win Rate (%)': f"{port_metrics['win_rate']:.1f}",
            'Calmar Ratio': f"{port_metrics['calmar_ratio']:.3f}"
        })
        
        # Add benchmark metrics
        if 'benchmark1' in metrics:
            bench1_metrics = metrics['benchmark1']
            metrics_list.append({
                'Strategy': bench1_metrics['label'],
                'Total Return (%)': f"{bench1_metrics['total_return']:.2f}",
                'Volatility (%)': f"{bench1_metrics['volatility']:.2f}",
                'Sharpe Ratio': f"{bench1_metrics['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{bench1_metrics['sortino_ratio']:.3f}",
                'Max Drawdown (%)': f"{bench1_metrics['max_drawdown']:.2f}",
                'Win Rate (%)': f"{bench1_metrics['win_rate']:.1f}",
                'Calmar Ratio': f"{bench1_metrics['calmar_ratio']:.3f}"
            })
        
        if 'benchmark2' in metrics:
            bench2_metrics = metrics['benchmark2']
            metrics_list.append({
                'Strategy': bench2_metrics['label'],
                'Total Return (%)': f"{bench2_metrics['total_return']:.2f}",
                'Volatility (%)': f"{bench2_metrics['volatility']:.2f}",
                'Sharpe Ratio': f"{bench2_metrics['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{bench2_metrics['sortino_ratio']:.3f}",
                'Max Drawdown (%)': f"{bench2_metrics['max_drawdown']:.2f}",
                'Win Rate (%)': f"{bench2_metrics['win_rate']:.1f}",
                'Calmar Ratio': f"{bench2_metrics['calmar_ratio']:.3f}"
            })
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # Display with highlighting
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Quick summary
        if len(metrics_list) > 1:
            port_return = metrics['portfolio']['total_return']
            bench_return = metrics.get('benchmark1', {}).get('total_return', 0)
            outperformance = port_return - bench_return
            
            col1, col2, col3 = st.columns(3)
            with col1:
                delta_color = "normal" if outperformance >= 0 else "inverse"
                st.metric(
                    "Portfolio Return",
                    f"{port_return:.2f}%",
                    delta=f"{outperformance:.2f}% vs {backtest_params['benchmark1']}",
                    delta_color=delta_color
                )
            with col2:
                st.metric(
                    "Portfolio Sharpe",
                    f"{metrics['portfolio']['sharpe_ratio']:.3f}"
                )
            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{metrics['portfolio']['max_drawdown']:.2f}%"
                )
    
    def _create_line_chart(self, results: Dict, backtest_params: Dict) -> go.Figure:
        """Create line chart for equity curves"""
        fig = go.Figure()
        
        # Portfolio equity curve
        fig.add_trace(go.Scatter(
            x=results['portfolio_values'].index,
            y=results['portfolio_values'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#2de19a', width=2),
            hovertemplate='%{x}<br>Portfolio: $%{y:,.0f}<extra></extra>'
        ))
        
        # Benchmark curves
        colors = ['#f7931a', '#627eea']  # Orange for BTC, Blue for ETH-like
        for idx, (bench_name, bench_values) in enumerate(results['benchmark_values'].items()):
            bench_label = backtest_params[bench_name]
            fig.add_trace(go.Scatter(
                x=bench_values.index,
                y=bench_values,
                mode='lines',
                name=bench_label,
                line=dict(color=colors[idx], width=2),
                hovertemplate=f'%{{x}}<br>{bench_label}: $%{{y:,.0f}}<extra></extra>'
            ))
        
        # Layout
        fig.update_layout(
            title='Portfolio vs Benchmark Performance',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            height=500
        )
        
        return fig
    
    def _create_candlestick_chart(self, results: Dict, backtest_params: Dict) -> go.Figure:
        """Create interactive candlestick chart with selectable timeframe"""
        # Get the timeframe for aggregation
        timeframe = backtest_params['candle_timeframe']
        
        # Aggregate portfolio values to the selected timeframe
        portfolio_values = results['portfolio_values']
        
        # Resample based on timeframe
        resample_map = {
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        agg_portfolio = portfolio_values.resample(resample_map[timeframe]).agg(['first', 'max', 'min', 'last'])
        agg_portfolio = agg_portfolio.dropna()
        
        fig = go.Figure()
        
        # Portfolio candlestick
        fig.add_trace(go.Candlestick(
            x=agg_portfolio.index,
            open=agg_portfolio['first'],
            high=agg_portfolio['max'],
            low=agg_portfolio['min'],
            close=agg_portfolio['last'],
            name='Portfolio',
            increasing_line_color='#2de19a',
            decreasing_line_color='#ff4b4b'
        ))
        
        # Benchmark lines overlay
        colors = ['#f7931a', '#627eea']
        for idx, (bench_name, bench_values) in enumerate(results['benchmark_values'].items()):
            bench_label = backtest_params[bench_name]
            agg_benchmark = bench_values.resample(resample_map[timeframe]).last()
            
            fig.add_trace(go.Scatter(
                x=agg_benchmark.index,
                y=agg_benchmark,
                mode='lines',
                name=bench_label,
                line=dict(color=colors[idx], width=2),
                yaxis='y2' if idx == 0 else 'y3'
            ))
        
        # Layout with multiple y-axes
        layout_dict = {
            'title': f'Portfolio Performance ({self.timeframe_options[timeframe]} candles)',
            'xaxis_title': 'Date',
            'yaxis_title': 'Portfolio Value ($)',
            'template': 'plotly_dark',
            'showlegend': True,
            'height': 600,
            'hovermode': 'x unified'
        }
        
        # Add y-axes for benchmarks
        if results['benchmark_values']:
            layout_dict['yaxis2'] = dict(
                title=f'{backtest_params["benchmark1"]} ($)',
                overlaying='y',
                side='right',
                showgrid=False
            )
            
            if len(results['benchmark_values']) > 1:
                layout_dict['yaxis3'] = dict(
                    title=f'{backtest_params["benchmark2"]} ($)',
                    overlaying='y',
                    side='left',
                    position=0.05,
                    showgrid=False
                )
        
        # Add range selector to layout
        layout_dict['xaxis'] = dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
        
        fig.update_layout(**layout_dict)
        
        return fig
    
    def _create_drawdown_chart(self, results: Dict, backtest_params: Dict) -> go.Figure:
        """Create drawdown comparison chart"""
        fig = go.Figure()
        
        # Portfolio drawdown
        portfolio_returns = results['metrics']['portfolio']['returns']
        portfolio_dd = self._calculate_drawdown_series(portfolio_returns)
        
        fig.add_trace(go.Scatter(
            x=portfolio_dd.index,
            y=portfolio_dd,
            mode='lines',
            fill='tozeroy',
            name='Portfolio',
            line=dict(color='#ff4b4b', width=1),
            fillcolor='rgba(255, 75, 75, 0.3)'
        ))
        
        # Benchmark drawdowns
        colors = ['rgba(247, 147, 26, 0.3)', 'rgba(98, 126, 234, 0.3)']
        line_colors = ['#f7931a', '#627eea']
        
        for idx, bench_name in enumerate(['benchmark1', 'benchmark2']):
            if bench_name in results['metrics']:
                bench_returns = results['metrics'][bench_name]['returns']
                bench_dd = self._calculate_drawdown_series(bench_returns)
                bench_label = results['metrics'][bench_name]['label']
                
                fig.add_trace(go.Scatter(
                    x=bench_dd.index,
                    y=bench_dd,
                    mode='lines',
                    fill='tozeroy',
                    name=bench_label,
                    line=dict(color=line_colors[idx], width=1),
                    fillcolor=colors[idx]
                ))
        
        # Layout
        fig.update_layout(
            title='Drawdown Comparison',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            showlegend=True,
            height=300,
            hovermode='x unified'
        )
        
        return fig
    
    def _calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        return drawdown
    
    def _display_position_breakdown(self, results: Dict):
        """Display individual position performance"""
        if not results['position_values']:
            return
            
        st.subheader("Position Breakdown")
        
        position_metrics = []
        for pos_name, pos_values in results['position_values'].items():
            symbol, direction = pos_name.rsplit('_', 1)
            initial_val = pos_values.iloc[0]
            final_val = pos_values.iloc[-1]
            pos_return = (final_val / initial_val - 1) * 100
            
            position_metrics.append({
                'Symbol': symbol,
                'Direction': direction.upper(),
                'Initial Value': f"${initial_val:,.0f}",
                'Final Value': f"${final_val:,.0f}",
                'Return (%)': f"{pos_return:.2f}",
                'Contribution ($)': f"${final_val - initial_val:,.0f}"
            })
        
        pos_df = pd.DataFrame(position_metrics)
        
        # Style the dataframe
        def style_direction(val):
            color = '#26c987' if val == 'LONG' else '#ff4b4b'
            return f'color: {color}; font-weight: bold;'
        
        def style_return(val):
            try:
                num_val = float(val.replace('%', ''))
                color = '#26c987' if num_val >= 0 else '#ff4b4b'
                return f'color: {color}; font-weight: bold;'
            except:
                return ''
        
        styled_df = pos_df.style.applymap(
            style_direction, subset=['Direction']
        ).applymap(
            style_return, subset=['Return (%)']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)