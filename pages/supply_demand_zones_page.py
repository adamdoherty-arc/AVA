"""
Technical Indicators Dashboard
===============================

Comprehensive technical analysis with:
- RSI Oversold/Overbought Scanner
- Momentum Indicators (RSI, MACD, Stochastic)
- Volatility Indicators (Bollinger Bands, ATR, IV)
- Volume Indicators (OBV, VWAP, MFI, Volume Profile)
- Trend Indicators (EMAs, Ichimoku, ADX)
- Advanced Indicators (Fibonacci, Supply/Demand Zones, Order Flow)
- Options Indicators (IVR, Expected Move, Greeks)

Data Sources: Database Stocks, TradingView Watchlists, Robinhood Positions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv
import logging

# Load environment variables FIRST
load_dotenv()

# Add src to path
sys.path.insert(0, 'src')

# Import all indicator modules
from src.momentum_indicators import MomentumIndicators
from src.standard_indicators import StandardIndicators
from src.options_indicators import OptionsIndicators
from src.fibonacci_calculator import FibonacciCalculator
from src.advanced_technical_indicators import VolumeProfileCalculator, OrderFlowAnalyzer
from src.services import get_tradingview_manager

# Import consolidated analysis functions
from supply_demand_zones_page_consolidated import (
    show_consolidated_analysis_page,
    create_consolidated_chart,
    show_momentum_trend_details,
    show_volatility_volume_details,
    show_advanced_analysis_details,
    show_trading_recommendations
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Technical Indicators",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def get_momentum_indicators():
    """Get cached momentum indicators instance"""
    return MomentumIndicators()


@st.cache_resource
def get_standard_indicators():
    """Get cached standard indicators instance"""
    return StandardIndicators()


@st.cache_resource
def get_options_indicators():
    """Get cached options indicators instance"""
    return OptionsIndicators()


@st.cache_resource
def get_fibonacci_calc():
    """Get cached Fibonacci calculator instance"""
    return FibonacciCalculator()


@st.cache_resource
def get_volume_profile_calc():
    """Get cached volume profile calculator instance"""
    return VolumeProfileCalculator()


@st.cache_resource
def get_order_flow_analyzer():
    """Get cached order flow analyzer instance"""
    return OrderFlowAnalyzer()


@st.cache_resource
def get_tv_manager():
    """Get cached TradingView manager"""
    return get_tradingview_manager()


@st.cache_data(ttl=300)
def get_watchlists_cached(_tv_manager):
    """Get TradingView watchlists with 5-minute cache"""
    try:
        return _tv_manager.get_all_symbols_dict()
    except Exception as e:
        logger.error(f"Error loading watchlists: {e}")
        return {}


@st.cache_data(ttl=300)
def get_database_stocks():
    """Get stocks from database with 5-minute cache"""
    try:
        import psycopg2
        import os

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres123!'),
            database=os.getenv('DB_NAME', 'magnus')
        )

        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM stocks WHERE is_active = true ORDER BY symbol")
        symbols = [row[0] for row in cur.fetchall()]

        cur.close()
        conn.close()

        return symbols
    except Exception as e:
        logger.error(f"Error loading database stocks: {e}")
        return []


@st.cache_data(ttl=60)
def get_positions_symbols():
    """Get symbols from Robinhood positions with 1-minute cache"""
    try:
        from src.services.positions_connector import PositionsConnector

        connector = PositionsConnector()
        positions_data = connector.get_data()

        if 'positions' in positions_data:
            symbols = list(set([pos.get('underlying_symbol') for pos in positions_data['positions']
                              if pos.get('underlying_symbol')]))
            return sorted(symbols)

        return []
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
        return []


def get_stock_selection_sources():
    """Get all available stock selection sources"""
    sources = {
        'manual': {'name': 'Manual Entry', 'symbols': []},
    }

    # Add watchlists
    tv_manager = get_tv_manager()
    watchlists = get_watchlists_cached(tv_manager)
    if watchlists:
        for wl_name, symbols in watchlists.items():
            sources[f'watchlist_{wl_name}'] = {
                'name': f'Watchlist: {wl_name}',
                'symbols': symbols
            }

    # Add database stocks
    db_stocks = get_database_stocks()
    if db_stocks:
        sources['database'] = {
            'name': 'Database Stocks',
            'symbols': db_stocks
        }

    # Add positions
    positions = get_positions_symbols()
    if positions:
        sources['positions'] = {
            'name': 'My Positions',
            'symbols': positions
        }

    return sources


def show_supply_demand_zones():
    """Main technical indicators dashboard with consolidated charts"""

    st.title("📊 Technical Indicators Dashboard")
    st.caption("Comprehensive technical analysis with all indicators in one unified view")

    # Mode selector - Scanner vs Single Stock Analysis
    col1, col2 = st.columns([1, 3])

    with col1:
        analysis_mode = st.radio(
            "Analysis Mode",
            ["🔍 Scanner", "📈 Single Stock"],
            key='analysis_mode'
        )

    with col2:
        st.write("")  # Spacing

    st.markdown("---")

    if analysis_mode == "🔍 Scanner":
        show_rsi_scanner_page()
    else:
        show_consolidated_analysis_page()


def show_rsi_scanner_page():
    """RSI Oversold/Overbought Scanner"""

    st.header("🔍 RSI Oversold/Overbought Scanner")
    st.caption("Find stocks with extreme RSI values across all your sources")

    # Configuration
    col1, col2, col3 = st.columns(3)

    with col1:
        rsi_period = st.slider("RSI Period", 5, 30, 14, key='rsi_scanner_period')
    with col2:
        oversold_threshold = st.slider("Oversold Threshold", 10, 35, 30, key='rsi_scanner_oversold')
    with col3:
        overbought_threshold = st.slider("Overbought Threshold", 65, 90, 70, key='rsi_scanner_overbought')

    # Source selection
    st.subheader("📋 Select Stock Sources")

    sources = get_stock_selection_sources()

    col1, col2 = st.columns(2)

    with col1:
        # Default to database, watchlists, and positions if available
        default_sources = []
        if 'database' in sources:
            default_sources.append('database')
        for key in sources.keys():
            if key.startswith('watchlist_'):
                default_sources.append(key)
                break  # Just add first watchlist as default
        if 'positions' in sources:
            default_sources.append('positions')

        selected_sources = st.multiselect(
            "Data Sources",
            list(sources.keys()),
            default=default_sources,
            format_func=lambda x: sources[x]['name'],
            key='rsi_scanner_sources'
        )

    with col2:
        max_stocks = st.slider("Max Stocks to Scan", 10, 500, 100, key='rsi_scanner_max_stocks')

    # Collect symbols
    all_symbols = []
    for source_key in selected_sources:
        all_symbols.extend(sources[source_key]['symbols'])

    all_symbols = list(set(all_symbols))[:max_stocks]  # Remove duplicates and limit

    if not all_symbols:
        st.warning("⚠️ Please select at least one data source")
        return

    st.info(f"📊 Scanning {len(all_symbols)} symbols...")

    # Scan button - fixed sizing issue by removing use_container_width
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        scan_button = st.button("🔍 Scan for RSI Opportunities", type="primary", key='rsi_scanner_scan_button')

    if scan_button:
        momentum = get_momentum_indicators()

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, symbol in enumerate(all_symbols):
            try:
                status_text.text(f"Scanning {symbol} ({i+1}/{len(all_symbols)})...")

                # Fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='3mo', interval='1d')

                if df.empty:
                    continue

                df.columns = [col.lower() for col in df.columns]
                current_price = float(df['close'].iloc[-1])

                # Calculate RSI - fixed data format issue
                rsi_series = momentum.calculate_rsi(df['close'])
                rsi_value = rsi_series.iloc[-1]

                # Check if oversold or overbought
                if rsi_value <= oversold_threshold or rsi_value >= overbought_threshold:
                    signal = "OVERSOLD" if rsi_value <= oversold_threshold else "OVERBOUGHT"

                    # Get additional context
                    macd = momentum.calculate_macd(df['close'])
                    macd_signal = momentum.get_macd_signal(macd)

                    emas = momentum.calculate_emas(df['close'])
                    ema_alignment = momentum.get_ema_alignment(emas, current_price)

                    results.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'RSI': rsi_value,
                        'Signal': signal,
                        'MACD': macd_signal['signal'],
                        'Trend': ema_alignment['alignment'],
                        'Distance from EMA200': f"{ema_alignment['distance_from_ema200_pct']:.2f}%"
                    })

                progress_bar.progress((i + 1) / len(all_symbols))

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue

        progress_bar.empty()
        status_text.empty()

        if not results:
            st.info("✅ No extreme RSI values found. All stocks are in neutral territory.")
            return

        # Display results
        st.success(f"🎯 Found {len(results)} opportunities!")

        # Separate oversold and overbought
        oversold_df = pd.DataFrame([r for r in results if r['Signal'] == 'OVERSOLD'])
        overbought_df = pd.DataFrame([r for r in results if r['Signal'] == 'OVERBOUGHT'])

        # Tabs for oversold/overbought
        tab1, tab2 = st.tabs(["🔵 Oversold (Buy Opportunities)", "🔴 Overbought (Sell Opportunities)"])

        with tab1:
            if not oversold_df.empty:
                st.subheader(f"📊 {len(oversold_df)} Oversold Stocks")
                oversold_df = oversold_df.sort_values('RSI')
                st.dataframe(oversold_df, use_container_width=True, height=400)

                # Top 5 chart
                st.subheader("Top 5 Most Oversold")
                fig = go.Figure()
                top5 = oversold_df.head(5)

                fig.add_trace(go.Bar(
                    x=top5['Symbol'],
                    y=top5['RSI'],
                    text=top5['RSI'].apply(lambda x: f"{x:.1f}"),
                    textposition='auto',
                    marker_color='green'
                ))

                fig.add_hline(y=oversold_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"Oversold: {oversold_threshold}")

                fig.update_layout(
                    title="Top 5 Oversold Stocks by RSI",
                    xaxis_title="Symbol",
                    yaxis_title="RSI",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No oversold stocks found")

        with tab2:
            if not overbought_df.empty:
                st.subheader(f"📊 {len(overbought_df)} Overbought Stocks")
                overbought_df = overbought_df.sort_values('RSI', ascending=False)
                st.dataframe(overbought_df, use_container_width=True, height=400)

                # Top 5 chart
                st.subheader("Top 5 Most Overbought")
                fig = go.Figure()
                top5 = overbought_df.head(5)

                fig.add_trace(go.Bar(
                    x=top5['Symbol'],
                    y=top5['RSI'],
                    text=top5['RSI'].apply(lambda x: f"{x:.1f}"),
                    textposition='auto',
                    marker_color='red'
                ))

                fig.add_hline(y=overbought_threshold, line_dash="dash", line_color="green",
                            annotation_text=f"Overbought: {overbought_threshold}")

                fig.update_layout(
                    title="Top 5 Overbought Stocks by RSI",
                    xaxis_title="Symbol",
                    yaxis_title="RSI",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No overbought stocks found")


def show_multi_indicator_page(key_prefix='multi_indicator'):
    """Multi-indicator analysis for a single stock"""

    st.header("📈 Multi-Indicator Analysis")
    st.caption("Comprehensive technical analysis of a single stock")

    # Stock selection
    col1, col2 = st.columns([3, 1])

    with col1:
        sources = get_stock_selection_sources()
        source = st.selectbox(
            "Select Source",
            list(sources.keys()),
            format_func=lambda x: sources[x]['name'],
            key=f'{key_prefix}_source'
        )

    with col2:
        st.write("")  # Spacing
        st.write("")

    if source == 'manual':
        symbol = st.text_input("Enter Symbol", value="AAPL", key=f'{key_prefix}_symbol').upper()
    else:
        symbol = st.selectbox("Select Symbol", sources[source]['symbols'], key=f'{key_prefix}_symbol_select')

    # Timeframe selection
    col1, col2 = st.columns(2)

    with col1:
        period = st.selectbox("Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=2, key=f'{key_prefix}_period')

    with col2:
        interval = st.selectbox("Interval", ['1d', '1wk', '1h'], index=0, key=f'{key_prefix}_interval')

    if st.button("📊 Analyze", type="primary", key=f'{key_prefix}_analyze_button'):
        with st.spinner(f"Analyzing {symbol}..."):
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                st.error(f"No data available for {symbol}")
                return

            df.columns = [col.lower() for col in df.columns]
            current_price = float(df['close'].iloc[-1])

            # Initialize indicators
            momentum = get_momentum_indicators()
            standard = get_standard_indicators()

            # Calculate all indicators
            st.subheader(f"📊 {symbol} @ ${current_price:.2f}")

            # Tabs for different indicator categories
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Momentum",
                "📊 Volatility",
                "📉 Volume",
                "🎯 Trend"
            ])

            with tab1:
                show_momentum_indicators(df, current_price, momentum)

            with tab2:
                show_volatility_indicators(df, current_price, standard, momentum)

            with tab3:
                show_volume_indicators(df, current_price, standard, interval)

            with tab4:
                show_trend_indicators(df, current_price, momentum, standard)


def show_momentum_indicators(df, current_price, momentum):
    """Display momentum indicators"""

    st.subheader("📈 Momentum Indicators")

    # RSI
    rsi_data = momentum.calculate_rsi(df['close'])
    rsi_signal = momentum.get_rsi_signal(rsi_data)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RSI (14)", f"{rsi_signal['value']:.1f}", rsi_signal['signal'])

    # MACD
    macd = momentum.calculate_macd(df['close'])
    macd_signal = momentum.get_macd_signal(macd)

    with col2:
        st.metric("MACD", macd_signal['signal'], f"Histogram: {macd_signal['histogram']:.2f}")

    # Stochastic
    stoch = StandardIndicators().stochastic(df)
    stoch_signal = StandardIndicators().stochastic_signal(stoch)

    with col3:
        st.metric("Stochastic", f"{stoch_signal['zone']}", f"K: {stoch['k'].iloc[-1]:.1f}")

    # Chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{df.index[0].strftime("%Y-%m-%d")} Price', 'RSI')
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rsi_data['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=600, showlegend=False, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.info(f"**RSI:** {rsi_signal['recommendation']}")
    st.info(f"**MACD:** {macd_signal['recommendation']}")
    st.info(f"**Stochastic:** {stoch_signal['recommendation']}")


def show_volatility_indicators(df, current_price, standard, momentum):
    """Display volatility indicators"""

    st.subheader("📊 Volatility Indicators")

    # Bollinger Bands
    bbands = standard.bollinger_bands(df)
    bb_signal = standard.bollinger_signal(current_price, bbands)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("BB Position", bb_signal['position'], bb_signal['signal'])

    with col2:
        st.metric("Volatility", bb_signal['volatility_state'], f"Bandwidth: {bb_signal['bandwidth']:.4f}")

    # ATR
    atr_data = momentum.calculate_atr(df)

    with col3:
        st.metric("ATR", f"${atr_data['atr'].iloc[-1]:.2f}", f"{atr_data['atr_pct'].iloc[-1]:.2f}%")

    # Chart
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=bbands['upper'],
        mode='lines',
        name='Upper BB',
        line=dict(color='red', width=1, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=bbands['middle'],
        mode='lines',
        name='Middle BB',
        line=dict(color='blue', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=bbands['lower'],
        mode='lines',
        name='Lower BB',
        line=dict(color='green', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(0,100,255,0.1)'
    ))

    fig.update_layout(
        title="Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.info(f"**Bollinger Bands:** {bb_signal['recommendation']}")
    st.info(f"**Volatility:** {bb_signal['volatility_recommendation']}")


def show_volume_indicators(df, current_price, standard, interval='1d'):
    """Display volume indicators"""

    st.subheader("📉 Volume Indicators")

    # OBV
    obv = standard.obv(df)
    obv_signal = standard.obv_signal(obv, df['close'])

    # VWAP (only for intraday or daily data)
    if interval in ['1d', '1h', '5m', '15m', '30m']:
        vwap = standard.vwap(df)
        vwap_signal = standard.vwap_signal(current_price, vwap)

    # MFI
    mfi = standard.mfi(df)
    mfi_signal = standard.mfi_signal(mfi)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("OBV Signal", obv_signal['signal'], obv_signal['trend'])

    with col2:
        if 'vwap_signal' in locals():
            st.metric("VWAP", f"${vwap.iloc[-1]:.2f}", vwap_signal['signal'])
        else:
            st.metric("VWAP", "N/A", "Requires intraday data")

    with col3:
        st.metric("MFI", f"{mfi_signal['value']:.1f}", mfi_signal['signal'])

    # Volume Chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price with Volume', 'Money Flow Index')
    )

    # Price
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # MFI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=mfi,
            mode='lines',
            name='MFI',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )

    # MFI levels
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=600, showlegend=False, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.info(f"**OBV:** {obv_signal['recommendation']}")
    if 'vwap_signal' in locals():
        st.info(f"**VWAP:** {vwap_signal['recommendation']}")
    st.info(f"**MFI:** {mfi_signal['recommendation']}")


def show_trend_indicators(df, current_price, momentum, standard):
    """Display trend indicators"""

    st.subheader("🎯 Trend Indicators")

    # EMAs
    emas = momentum.calculate_emas(df['close'])
    ema_alignment = momentum.get_ema_alignment(emas, current_price)

    # ADX
    adx = standard.adx(df)
    adx_signal = standard.adx_signal(adx)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("EMA Alignment", ema_alignment['alignment'], ema_alignment['strength'])

    with col2:
        st.metric("ADX", f"{adx_signal['adx_value']:.1f}", adx_signal['trend_strength'])

    with col3:
        distance = ema_alignment['distance_from_ema200_pct']
        st.metric("Distance from EMA200", f"{distance:.2f}%",
                 "Above" if distance > 0 else "Below")

    # EMA Chart
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))

    # EMAs
    ema_colors = {
        'ema_9': 'yellow',
        'ema_21': 'orange',
        'ema_50': 'red',
        'ema_100': 'purple',
        'ema_200': 'blue'
    }

    for ema_name, color in ema_colors.items():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=emas[ema_name],
            mode='lines',
            name=ema_name.upper(),
            line=dict(color=color, width=1.5)
        ))

    fig.update_layout(
        title="Exponential Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    if ema_alignment['all_aligned']:
        st.success(f"✅ All EMAs aligned {ema_alignment['alignment']} - Strong trend!")

    st.info(f"**ADX:** {adx_signal['recommendation']}")


# Placeholder functions for other pages (same as original)
def show_bollinger_bands_page():
    st.info("Bollinger Bands detailed analysis - Coming from multi-indicator view")
    show_multi_indicator_page(key_prefix='bollinger')


def show_stochastic_page():
    st.info("Stochastic detailed analysis - Coming from multi-indicator view")
    show_multi_indicator_page(key_prefix='stochastic')


def show_fibonacci_page():
    st.info("Fibonacci analysis - Use the original implementation from supply_demand_zones_page.py")


def show_supply_demand_page():
    st.info("Supply/Demand Zones - Use the original implementation from supply_demand_zones_page.py")


def show_volume_profile_page():
    st.info("Volume Profile - Use the original implementation from supply_demand_zones_page.py")


def show_order_flow_page():
    st.info("Order Flow - Use the original implementation from supply_demand_zones_page.py")


def show_ichimoku_page():
    """Ichimoku Cloud Analysis"""
    st.header("🎨 Ichimoku Cloud Analysis")

    # Stock selection (reuse from multi-indicator)
    sources = get_stock_selection_sources()
    source = st.selectbox(
        "Select Source",
        list(sources.keys()),
        format_func=lambda x: sources[x]['name'],
        key='ichimoku_source'
    )

    if source == 'manual':
        symbol = st.text_input("Enter Symbol", value="AAPL", key='ichimoku_symbol').upper()
    else:
        symbol = st.selectbox("Select Symbol", sources[source]['symbols'], key='ichimoku_symbol_select')

    period = st.selectbox("Period", ['1mo', '3mo', '6mo', '1y'], index=2, key='ichimoku_period')

    if st.button("📊 Analyze Ichimoku", type="primary"):
        with st.spinner(f"Analyzing {symbol}..."):
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval='1d')

            if df.empty:
                st.error(f"No data available for {symbol}")
                return

            df.columns = [col.lower() for col in df.columns]
            current_price = float(df['close'].iloc[-1])

            # Calculate Ichimoku
            standard = get_standard_indicators()
            ichimoku = standard.ichimoku(df)
            ichimoku_signal = standard.ichimoku_signal(current_price, ichimoku)

            # Display metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Signal", ichimoku_signal['signal'], ichimoku_signal['strength'])

            with col2:
                st.metric("Cloud Position", ichimoku_signal['cloud_position'])

            with col3:
                st.metric("Current Price", f"${current_price:.2f}")

            # Ichimoku Chart
            fig = go.Figure()

            # Price
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))

            # Conversion and Base lines
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ichimoku['conversion'],
                mode='lines',
                name='Conversion (Tenkan)',
                line=dict(color='red', width=1)
            ))

            fig.add_trace(go.Scatter(
                x=df.index,
                y=ichimoku['base'],
                mode='lines',
                name='Base (Kijun)',
                line=dict(color='blue', width=1)
            ))

            # Cloud
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ichimoku['span_a'],
                mode='lines',
                name='Span A',
                line=dict(color='green', width=0.5),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=df.index,
                y=ichimoku['span_b'],
                mode='lines',
                name='Span B',
                line=dict(color='red', width=0.5),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)',
                showlegend=False
            ))

            fig.update_layout(
                title=f"{symbol} - Ichimoku Cloud",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Recommendation
            st.info(f"**Ichimoku Signal:** {ichimoku_signal['recommendation']}")


def show_options_analysis_page():
    """Options Analysis with IVR, Expected Move, Greeks"""
    st.header("📉 Options Analysis")
    st.caption("Implied Volatility Rank • Expected Move • Greeks")

    # Stock selection
    sources = get_stock_selection_sources()
    source = st.selectbox(
        "Select Source",
        list(sources.keys()),
        format_func=lambda x: sources[x]['name'],
        key='options_source'
    )

    if source == 'manual':
        symbol = st.text_input("Enter Symbol", value="AAPL", key='options_symbol').upper()
    else:
        symbol = st.selectbox("Select Symbol", sources[source]['symbols'], key='options_symbol_select')

    if st.button("📊 Analyze Options", type="primary"):
        with st.spinner(f"Analyzing {symbol} options..."):
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y', interval='1d')

            if df.empty:
                st.error(f"No data available for {symbol}")
                return

            df.columns = [col.lower() for col in df.columns]
            current_price = float(df['close'].iloc[-1])

            # Get IV data (simplified - would need real options data)
            # For now, use historical volatility as proxy
            returns = df['close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)

            # Mock IV (in reality, would fetch from options chain)
            current_iv = hist_vol * 1.2  # Assume IV is 20% higher than HV

            options_ind = get_options_indicators()

            # IVR
            iv_series = pd.Series([hist_vol] * len(df))  # Mock series
            ivr = options_ind.implied_volatility_rank(current_iv, iv_series)

            # Expected Move
            dte = 30  # 30 days
            expected_move = options_ind.expected_move(current_price, current_iv, dte)

            # Display
            st.subheader(f"📊 {symbol} @ ${current_price:.2f}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current IV", f"{current_iv*100:.1f}%")

            with col2:
                st.metric("IVR", f"{ivr['ivr']:.1f}", ivr['interpretation'])

            with col3:
                st.metric("Expected Move (30d)", f"${expected_move['expected_move']:.2f}")

            with col4:
                st.metric("Move %", f"{expected_move['move_pct']:.2f}%")

            # Expected Move Range
            st.subheader("📊 Expected Move Range (1 SD)")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Upper Bound", f"${expected_move['upper_bound']:.2f}")

            with col2:
                st.metric("Lower Bound", f"${expected_move['lower_bound']:.2f}")

            # Recommendation
            st.info(f"**IVR Strategy:** {ivr['recommendation']}")

            # Strategy Selector
            st.subheader("🎯 Recommended Strategy")

            momentum = get_momentum_indicators()
            emas = momentum.calculate_emas(df['close'])
            ema_alignment = momentum.get_ema_alignment(emas, current_price)

            strategy_rec = options_ind.option_strategy_recommendation(
                ivr=ivr['ivr'],
                trend=ema_alignment['alignment'],
                expected_move=expected_move
            )

            top_strategy = strategy_rec['top_recommendation']
            if top_strategy:
                st.success(f"**Recommended:** {top_strategy['strategy']}")
                st.info(f"**Reason:** {top_strategy['reason']}")
                st.write(f"- **Profit Potential:** {top_strategy['profit_potential']}")
                st.write(f"- **Risk:** {top_strategy['risk']}")
                st.write(f"- **Ideal For:** {top_strategy['ideal_for']}")

                # Show other strategies
                if len(strategy_rec['strategies']) > 1:
                    with st.expander("📋 Other Strategy Options"):
                        for i, strat in enumerate(strategy_rec['strategies'][1:], 1):
                            st.write(f"**{i}. {strat['strategy']}**")
                            st.write(f"  - Reason: {strat['reason']}")
                            st.write(f"  - Profit: {strat['profit_potential']}, Risk: {strat['risk']}")
            else:
                st.warning("No specific strategy recommendation available")


if __name__ == "__main__":
    show_supply_demand_zones()
