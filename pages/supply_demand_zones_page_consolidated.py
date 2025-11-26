"""
Consolidated Single Stock Technical Analysis
All indicators with charts in one unified view
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from typing import Dict


def show_consolidated_analysis_page():
    """Consolidated view showing all indicators with charts on one page"""

    st.header("📈 Consolidated Technical Analysis")
    st.caption("Complete technical view with all indicators and charts")

    # Stock selection
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        sources = get_stock_selection_sources()
        source = st.selectbox(
            "Select Source",
            list(sources.keys()),
            format_func=lambda x: sources[x]['name'],
            key='consolidated_source'
        )

    with col2:
        period = st.selectbox("Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=2, key='consolidated_period')

    with col3:
        interval = st.selectbox("Interval", ['1d', '1wk', '1h'], index=0, key='consolidated_interval')

    if source == 'manual':
        symbol = st.text_input("Enter Symbol", value="AAPL", key='consolidated_symbol').upper()
    else:
        symbol = st.selectbox("Select Symbol", sources[source]['symbols'], key='consolidated_symbol_select')

    # Centered analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze All Indicators", type="primary", key='consolidated_analyze')

    if analyze_button:
        with st.spinner(f"Analyzing {symbol} with all indicators..."):
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    st.error(f"No data available for {symbol}")
                    return

                # Normalize column names to lowercase
                df.columns = [col.lower() for col in df.columns]

                # Verify required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.write(f"Available columns: {list(df.columns)}")
                    return

                current_price = float(df['close'].iloc[-1])

            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

            # Initialize all indicators
            from src.momentum_indicators import MomentumIndicators
            from src.standard_indicators import StandardIndicators
            from src.options_indicators import OptionsIndicators
            from src.advanced_technical_indicators import VolumeProfileCalculator, OrderFlowAnalyzer

            momentum = MomentumIndicators()
            standard = StandardIndicators()
            options_ind = OptionsIndicators()
            vp_calc = VolumeProfileCalculator()
            of_analyzer = OrderFlowAnalyzer()

            # Header with current price
            st.markdown(f"## {symbol} @ ${current_price:.2f}")
            st.markdown("---")

            # Calculate all indicators upfront
            try:
                # Create a copy to avoid modifying original
                df_copy = df.copy()

                # Momentum
                rsi_series = momentum.calculate_rsi(df_copy['close'])
                rsi_signal = momentum.get_rsi_signal(rsi_series)

                macd = momentum.calculate_macd(df_copy['close'])
                macd_signal = momentum.get_macd_signal(macd)

                emas = momentum.calculate_emas(df_copy['close'])
                ema_alignment = momentum.get_ema_alignment(emas, current_price)

                atr_data = momentum.calculate_atr(df_copy)

                # Standard
                bbands = standard.bollinger_bands(df_copy)
                bb_signal = standard.bollinger_signal(current_price, bbands)

                stoch = standard.stochastic(df_copy)
                stoch_signal = standard.stochastic_signal(stoch)

                obv_series = standard.obv(df_copy)
                obv_signal = standard.obv_signal(obv_series, df_copy['close'])

                mfi_series = standard.mfi(df_copy)
                mfi_signal = standard.mfi_signal(mfi_series)

                adx_data = standard.adx(df_copy)
                adx_signal = standard.adx_signal(adx_data)

                ichimoku_data = standard.ichimoku(df_copy)
                ichimoku_signal = standard.ichimoku_signal(current_price, ichimoku_data)

                # Advanced
                volume_profile = vp_calc.calculate_volume_profile(df_copy, price_bins=30)
                vp_signals = vp_calc.get_trading_signals(current_price, volume_profile)

                df_with_cvd = df_copy.copy()
                df_with_cvd['cvd'] = of_analyzer.calculate_cvd(df_copy)
                cvd_divergences = of_analyzer.find_cvd_divergences(df_with_cvd, lookback=10)

            except Exception as e:
                st.error(f"Error calculating indicators: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

            # === UNIFIED CHARTS VIEW ===
            st.subheader("📊 Complete Chart Analysis")

            # Create comprehensive chart with all indicators
            fig = create_consolidated_chart(
                df=df_copy,
                current_price=current_price,
                rsi_series=rsi_series,
                macd=macd,
                emas=emas,
                bbands=bbands,
                stoch=stoch,
                mfi_series=mfi_series,
                volume_profile=volume_profile,
                cvd_series=df_with_cvd['cvd'],
                ichimoku=ichimoku_data,
                symbol=symbol
            )

            st.plotly_chart(fig, use_container_width=True)

            # === KEY METRICS GRID ===
            st.markdown("---")
            st.subheader("🎯 Key Metrics Summary")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("RSI", f"{rsi_signal['value']:.1f}", rsi_signal['signal'])
                st.caption(f"Zone: {rsi_signal['zone']}")

            with col2:
                st.metric("MACD", macd_signal['signal'][:7], f"H: {macd_signal['histogram']:.2f}")
                st.caption(f"{macd_signal['strength']}")

            with col3:
                st.metric("Trend", ema_alignment['alignment'], ema_alignment['strength'])
                st.caption(f"EMA: {ema_alignment['distance_from_ema200_pct']:.1f}%")

            with col4:
                st.metric("ADX", f"{adx_signal['adx']:.1f}", adx_signal['direction'])
                st.caption(f"{adx_signal['trend_strength']}")

            with col5:
                st.metric("BB Position", bb_signal['position'][:10], bb_signal['signal'])
                st.caption(f"Vol: {bb_signal['volatility_state']}")

            # Second row of metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Stochastic", stoch_signal['zone'], f"K: {stoch_signal['k']:.1f}")
                st.caption(f"{stoch_signal['strength']}")

            with col2:
                st.metric("MFI", f"{mfi_signal['mfi']:.1f}", mfi_signal['signal'])
                st.caption(f"Zone: {mfi_signal['zone']}")

            with col3:
                st.metric("OBV", obv_signal['trend'], obv_signal['signal'][:10])
                st.caption(f"Change: {obv_signal['obv_change']:,.0f}")

            with col4:
                st.metric("ATR", f"${atr_data['atr'].iloc[-1]:.2f}", f"{atr_data['atr_pct'].iloc[-1]:.2f}%")
                st.caption("Volatility")

            with col5:
                st.metric("Ichimoku", ichimoku_signal['cloud_position'][:10], ichimoku_signal['signal'])
                st.caption(f"{ichimoku_signal['strength']}")

            # === DETAILED ANALYSIS TABS ===
            st.markdown("---")
            st.subheader("📋 Detailed Analysis")

            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Momentum & Trend",
                "📊 Volatility & Volume",
                "🔮 Advanced Analysis",
                "💡 Trading Recommendations"
            ])

            with tab1:
                show_momentum_trend_details(
                    rsi_signal, macd_signal, ema_alignment, adx_signal, ichimoku_signal
                )

            with tab2:
                show_volatility_volume_details(
                    bb_signal, stoch_signal, mfi_signal, obv_signal, atr_data
                )

            with tab3:
                show_advanced_analysis_details(
                    volume_profile, vp_signals, cvd_divergences, df_with_cvd
                )

            with tab4:
                show_trading_recommendations(
                    symbol, current_price, rsi_signal, macd_signal, ema_alignment,
                    bb_signal, adx_signal, vp_signals
                )


def create_consolidated_chart(
    df, current_price, rsi_series, macd, emas, bbands, stoch, mfi_series,
    volume_profile, cvd_series, ichimoku, symbol
):
    """Create consolidated multi-panel chart with all key indicators"""

    # Create 6-panel layout
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.10, 0.10],
        subplot_titles=(
            f'{symbol} Price with EMAs & Bollinger Bands',
            'RSI (14)',
            'MACD',
            'Stochastic & MFI',
            'Volume',
            'CVD (Cumulative Volume Delta)'
        )
    )

    # === PANEL 1: PRICE WITH EMAs AND BOLLINGER BANDS ===
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            showlegend=True
        ),
        row=1, col=1
    )

    # EMAs
    ema_colors = {'ema_9': 'yellow', 'ema_21': 'orange', 'ema_50': 'red',
                  'ema_100': 'purple', 'ema_200': 'blue'}

    for ema_name, color in ema_colors.items():
        if ema_name in emas:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=emas[ema_name],
                    mode='lines', name=ema_name.upper(),
                    line=dict(color=color, width=1),
                    showlegend=True
                ),
                row=1, col=1
            )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index, y=bbands['upper'],
            mode='lines', name='BB Upper',
            line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=bbands['lower'],
            mode='lines', name='BB Lower',
            line=dict(color='rgba(0,255,0,0.3)', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(200,200,200,0.1)',
            showlegend=True
        ),
        row=1, col=1
    )

    # === PANEL 2: RSI ===
    fig.add_trace(
        go.Scatter(
            x=df.index, y=rsi_series,
            mode='lines', name='RSI',
            line=dict(color='purple', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # === PANEL 3: MACD ===
    fig.add_trace(
        go.Scatter(
            x=df.index, y=macd['macd'],
            mode='lines', name='MACD',
            line=dict(color='blue', width=1.5),
            showlegend=False
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=macd['signal'],
            mode='lines', name='Signal',
            line=dict(color='orange', width=1.5),
            showlegend=False
        ),
        row=3, col=1
    )

    # MACD Histogram
    colors = ['green' if val >= 0 else 'red' for val in macd['histogram']]
    fig.add_trace(
        go.Bar(
            x=df.index, y=macd['histogram'],
            name='Histogram',
            marker_color=colors,
            showlegend=False
        ),
        row=3, col=1
    )

    # === PANEL 4: STOCHASTIC & MFI ===
    # Stochastic
    fig.add_trace(
        go.Scatter(
            x=df.index, y=stoch['k'],
            mode='lines', name='Stoch %K',
            line=dict(color='blue', width=1.5),
            showlegend=False
        ),
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=stoch['d'],
            mode='lines', name='Stoch %D',
            line=dict(color='red', width=1.5),
            showlegend=False
        ),
        row=4, col=1
    )

    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.3, row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.3, row=4, col=1)

    # === PANEL 5: VOLUME ===
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['volume'],
            name='Volume',
            marker_color='rgba(100,100,250,0.5)',
            showlegend=False
        ),
        row=5, col=1
    )

    # === PANEL 6: CVD ===
    fig.add_trace(
        go.Scatter(
            x=df.index, y=cvd_series,
            mode='lines', name='CVD',
            line=dict(color='green', width=2),
            fill='tozeroy', fillcolor='rgba(0,255,0,0.1)',
            showlegend=False
        ),
        row=6, col=1
    )

    # Update layout
    fig.update_layout(
        height=1400,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Stoch", row=4, col=1)
    fig.update_yaxes(title_text="Volume", row=5, col=1)
    fig.update_yaxes(title_text="CVD", row=6, col=1)

    return fig


def show_momentum_trend_details(rsi_signal, macd_signal, ema_alignment, adx_signal, ichimoku_signal):
    """Show detailed momentum and trend analysis"""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Momentum Indicators")

        st.markdown(f"**RSI Analysis**")
        st.info(rsi_signal['recommendation'])
        st.write(f"- Value: {rsi_signal['value']:.1f}")
        st.write(f"- Zone: {rsi_signal['zone']}")
        st.write(f"- Signal: {rsi_signal['signal']}")

        st.markdown("---")

        st.markdown(f"**MACD Analysis**")
        st.info(macd_signal['recommendation'])
        st.write(f"- Signal: {macd_signal['signal']}")
        st.write(f"- Strength: {macd_signal['strength']}")
        st.write(f"- Histogram: {macd_signal['histogram']:.2f}")

    with col2:
        st.markdown("### 📈 Trend Indicators")

        st.markdown(f"**EMA Alignment**")
        if ema_alignment['all_aligned']:
            st.success(f"✅ All EMAs aligned {ema_alignment['alignment']} - Strong trend!")
        else:
            st.warning(f"EMAs not fully aligned - {ema_alignment['alignment']} bias")

        st.write(f"- Alignment: {ema_alignment['alignment']}")
        st.write(f"- Strength: {ema_alignment['strength']}")
        st.write(f"- Distance from EMA200: {ema_alignment['distance_from_ema200_pct']:.2f}%")

        st.markdown("---")

        st.markdown(f"**ADX Analysis**")
        st.info(adx_signal['recommendation'])
        st.write(f"- ADX Value: {adx_signal['adx']:.1f}")
        st.write(f"- Trend Strength: {adx_signal['trend_strength']}")
        st.write(f"- Direction: {adx_signal['direction']}")
        st.write(f"- Options Strategy: {adx_signal['options_strategy']}")


def show_volatility_volume_details(bb_signal, stoch_signal, mfi_signal, obv_signal, atr_data):
    """Show detailed volatility and volume analysis"""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Volatility Indicators")

        st.markdown(f"**Bollinger Bands**")
        st.info(bb_signal['recommendation'])
        st.write(f"- Position: {bb_signal['position']}")
        st.write(f"- Signal: {bb_signal['signal']}")
        st.write(f"- Volatility: {bb_signal['volatility_state']}")
        st.write(f"- Bandwidth: {bb_signal['bandwidth']:.4f}")

        st.markdown("---")

        st.markdown(f"**ATR (Average True Range)**")
        st.write(f"- Current ATR: ${atr_data['atr'].iloc[-1]:.2f}")
        st.write(f"- ATR %: {atr_data['atr_pct'].iloc[-1]:.2f}%")
        st.write(f"- Stop Loss Target: ${atr_data['stop_loss_target'].iloc[-1]:.2f}")
        st.write(f"- Take Profit Target: ${atr_data['take_profit_target'].iloc[-1]:.2f}")

    with col2:
        st.markdown("### 📉 Volume Indicators")

        st.markdown(f"**Stochastic Oscillator**")
        st.info(stoch_signal['recommendation'])
        st.write(f"- %K: {stoch_signal['k']:.1f}")
        st.write(f"- %D: {stoch_signal['d']:.1f}")
        st.write(f"- Zone: {stoch_signal['zone']}")
        st.write(f"- Signal: {stoch_signal['signal']}")

        st.markdown("---")

        st.markdown(f"**MFI (Money Flow Index)**")
        st.info(mfi_signal['recommendation'])
        st.write(f"- MFI Value: {mfi_signal['mfi']:.1f}")
        st.write(f"- Zone: {mfi_signal['zone']}")
        st.write(f"- Signal: {mfi_signal['signal']}")


def show_advanced_analysis_details(volume_profile, vp_signals, cvd_divergences, df_with_cvd):
    """Show advanced volume profile and order flow analysis"""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Volume Profile Analysis")

        st.markdown(f"**POC (Point of Control)**")
        st.write(f"- Price: ${volume_profile['poc']['price']:.2f}")
        st.write(f"- Volume: {volume_profile['poc']['volume']:,.0f}")
        st.write(f"- % of Total: {volume_profile['poc']['pct_of_total']:.1f}%")

        st.markdown(f"**Value Area (70% volume)**")
        st.write(f"- VAH: ${volume_profile['vah']:.2f}")
        st.write(f"- VAL: ${volume_profile['val']:.2f}")
        st.write(f"- Width: ${volume_profile['value_area_width']:.2f} ({volume_profile['value_area_width_pct']:.2f}%)")

        st.markdown(f"**Trading Position**")
        st.info(vp_signals['recommendation'])
        st.write(f"- Position: {vp_signals['position']}")
        st.write(f"- Bias: {vp_signals['bias']}")
        st.write(f"- Setup Quality: {vp_signals['setup_quality']}")

    with col2:
        st.markdown("### 💰 Order Flow (CVD) Analysis")

        st.markdown(f"**Current CVD**")
        st.write(f"- Latest CVD: {df_with_cvd['cvd'].iloc[-1]:,.0f}")
        st.write(f"- 5-Day Change: {df_with_cvd['cvd'].iloc[-1] - df_with_cvd['cvd'].iloc[-6]:,.0f}")

        cvd_trend = "BULLISH 📈" if df_with_cvd['cvd'].iloc[-1] > df_with_cvd['cvd'].iloc[-6] else "BEARISH 📉"
        st.write(f"- Trend: {cvd_trend}")

        st.markdown(f"**CVD Divergences**")
        if cvd_divergences:
            st.warning(f"⚠️ Found {len(cvd_divergences)} divergences!")
            for div in cvd_divergences:
                st.write(f"- {div['type']} on {div['date'].date()}")
                st.caption(div['signal'])
        else:
            st.success("✅ No CVD divergences detected")


def show_trading_recommendations(symbol, current_price, rsi_signal, macd_signal,
                                ema_alignment, bb_signal, adx_signal, vp_signals):
    """Generate comprehensive trading recommendations"""

    st.markdown("### 💡 Comprehensive Trading Analysis")

    # Calculate overall bias
    bullish_signals = 0
    bearish_signals = 0

    if rsi_signal['signal'] in ['BUY', 'STRONG_BUY']:
        bullish_signals += 1
    elif rsi_signal['signal'] in ['SELL', 'STRONG_SELL']:
        bearish_signals += 1

    if macd_signal['signal'] in ['BULLISH', 'STRONG_BULLISH']:
        bullish_signals += 1
    elif macd_signal['signal'] in ['BEARISH', 'STRONG_BEARISH']:
        bearish_signals += 1

    if ema_alignment['alignment'] == 'BULLISH':
        bullish_signals += 1
    elif ema_alignment['alignment'] == 'BEARISH':
        bearish_signals += 1

    if vp_signals['bias'] == 'BULLISH':
        bullish_signals += 1
    elif vp_signals['bias'] == 'BEARISH':
        bearish_signals += 1

    # Overall bias
    total_signals = bullish_signals + bearish_signals

    if total_signals == 0:
        overall_bias = "NEUTRAL"
        bias_color = "blue"
    else:
        bullish_pct = bullish_signals / total_signals * 100
        if bullish_pct >= 75:
            overall_bias = "STRONG BULLISH"
            bias_color = "green"
        elif bullish_pct >= 60:
            overall_bias = "BULLISH"
            bias_color = "lightgreen"
        elif bullish_pct >= 40:
            overall_bias = "NEUTRAL"
            bias_color = "blue"
        elif bullish_pct >= 25:
            overall_bias = "BEARISH"
            bias_color = "orange"
        else:
            overall_bias = "STRONG BEARISH"
            bias_color = "red"

    # Display overall bias
    st.markdown(f"**Overall Market Bias:** :{bias_color}[{overall_bias}]")
    st.write(f"- Bullish Signals: {bullish_signals}")
    st.write(f"- Bearish Signals: {bearish_signals}")
    st.write(f"- Bullish %: {bullish_signals / max(total_signals, 1) * 100:.0f}%")

    st.markdown("---")

    # Specific recommendations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 Options Strategies**")
        st.info(adx_signal['options_strategy'])

        if bb_signal['volatility_state'] == 'SQUEEZE':
            st.write("- Consider long options (straddles/strangles)")
        elif bb_signal['volatility_state'] == 'EXPANSION':
            st.write("- Consider selling premium (iron condors/credit spreads)")

        if overall_bias in ['STRONG BULLISH', 'BULLISH']:
            st.write("- Bullish directional plays (call spreads, cash-secured puts)")
        elif overall_bias in ['STRONG BEARISH', 'BEARISH']:
            st.write("- Bearish directional plays (put spreads, covered calls)")

    with col2:
        st.markdown("**🎯 Trade Setup**")

        if vp_signals['setup_quality'] == 'EXCELLENT':
            st.success(f"✅ {vp_signals['setup_quality']} setup quality!")
        elif vp_signals['setup_quality'] == 'GOOD':
            st.info(f"👍 {vp_signals['setup_quality']} setup quality")
        else:
            st.warning(f"⚠️ {vp_signals['setup_quality']} setup quality")

        st.write(f"- Entry Zone: Around ${current_price:.2f}")
        st.write(f"- Watch Volume Profile: {vp_signals['position']}")
        st.write(f"- Trend: {ema_alignment['alignment']}")


# Helper function to get stock sources - needs to be defined here to avoid circular import
def get_stock_selection_sources():
    """Get all available stock selection sources"""
    import sys
    import os
    import logging

    logger = logging.getLogger(__name__)

    sources = {
        'manual': {'name': 'Manual Entry', 'symbols': []},
    }

    try:
        # Add watchlists
        sys.path.insert(0, 'src')
        from src.services import get_tradingview_manager

        tv_manager = get_tradingview_manager()
        watchlists = tv_manager.get_all_symbols_dict() if tv_manager else {}

        if watchlists:
            for wl_name, symbols in watchlists.items():
                sources[f'watchlist_{wl_name}'] = {
                    'name': f'Watchlist: {wl_name}',
                    'symbols': symbols
                }
    except Exception as e:
        logger.error(f"Error loading watchlists: {e}")

    try:
        # Add database stocks
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres123!'),
            database=os.getenv('DB_NAME', 'magnus')
        )

        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM stocks WHERE is_active = true ORDER BY symbol")
        db_stocks = [row[0] for row in cur.fetchall()]

        cur.close()
        conn.close()

        if db_stocks:
            sources['database'] = {
                'name': 'Database Stocks',
                'symbols': db_stocks
            }
    except Exception as e:
        logger.error(f"Error loading database stocks: {e}")

    try:
        # Add positions
        sys.path.insert(0, 'src')
        from src.services.positions_connector import PositionsConnector

        connector = PositionsConnector()
        positions_data = connector.get_data()

        if 'positions' in positions_data:
            positions = list(set([pos.get('underlying_symbol') for pos in positions_data['positions']
                              if pos.get('underlying_symbol')]))

            if positions:
                sources['positions'] = {
                    'name': 'My Positions',
                    'symbols': sorted(positions)
                }
    except Exception as e:
        logger.error(f"Error loading positions: {e}")

    return sources
