"""
Enhanced Sector Analysis Page
Comprehensive sector analysis with advanced metrics, economic indicators, and rotation strategies
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import logging

from src.sector_metrics_calculator import SectorMetricsCalculator
from src.sector_etf_manager import SectorETFManager
from src.economic_indicators import EconomicIndicatorsManager

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Enhanced Sector Analysis", page_icon="🏭", layout="wide")


# ========================================================================
# INITIALIZATION
# ========================================================================

@st.cache_resource
def get_sector_calculator():
    """Get cached sector metrics calculator"""
    return SectorMetricsCalculator()

@st.cache_resource
def get_etf_manager():
    """Get cached ETF manager"""
    return SectorETFManager()

@st.cache_resource
def get_economic_manager():
    """Get cached economic indicators manager (auto-loads FRED API key from .env)"""
    return EconomicIndicatorsManager()  # Will auto-load from .env

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_etf_performance_data():
    """Get all ETF performance data (cached)"""
    etf_manager = get_etf_manager()
    return etf_manager.get_all_etf_performance()

@st.cache_data(ttl=3600)
def get_sector_rotation_signals():
    """Get sector rotation signals (cached)"""
    etf_manager = get_etf_manager()
    return etf_manager.get_sector_rotation_signals()

@st.cache_data(ttl=3600)
def get_economic_snapshot():
    """Get economic snapshot (cached)"""
    econ_manager = get_economic_manager()
    return econ_manager.get_economic_snapshot()


# ========================================================================
# MAIN PAGE
# ========================================================================

def display_sector_analysis_enhanced():
    """Display the enhanced sector analysis page"""

    st.title("🏭 Enhanced Sector Analysis")
    st.caption("Comprehensive GICS sector analysis with momentum, rotation signals, and economic indicators")

    # Initialize managers
    calculator = get_sector_calculator()
    etf_manager = get_etf_manager()
    econ_manager = get_economic_manager()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Sector Dashboard",
        "🔄 Sector Rotation",
        "🌍 Economic Indicators",
        "📈 ETF Performance",
        "💼 Sector Deep Dive",
        "📚 Reference Guide"
    ])

    # ====================================================================
    # TAB 1: SECTOR DASHBOARD
    # ====================================================================

    with tab1:
        st.header("Sector Performance Dashboard")

        # Fetch ETF performance data
        with st.spinner("Loading sector data..."):
            perf_df = get_etf_performance_data()

        if perf_df.empty:
            st.error("Unable to load sector data. Please check your internet connection.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Sectors", len(perf_df))

        with col2:
            avg_1m_return = perf_df['1M Return'].mean()
            st.metric("Avg 1M Return", f"{avg_1m_return:+.2f}%")

        with col3:
            best_sector = perf_df.loc[perf_df['1M Return'].idxmax(), 'Sector']
            st.metric("Top Sector (1M)", best_sector)

        with col4:
            worst_sector = perf_df.loc[perf_df['1M Return'].idxmin(), 'Sector']
            st.metric("Worst Sector (1M)", worst_sector)

        # Sector Heatmap
        st.markdown("### 🌡️ Sector Performance Heatmap")

        fig = px.treemap(
            perf_df,
            path=['Sector'],
            values=[1] * len(perf_df),  # Equal size
            color='1M Return',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            hover_data={
                '1M Return': ':.2f',
                '3M Return': ':.2f',
                '6M Return': ':.2f'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Performance comparison chart
        st.markdown("### 📊 Multi-Period Performance Comparison")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("1-Month Returns", "Year-to-Date Returns"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Sort by 1M return
        perf_sorted = perf_df.sort_values('1M Return', ascending=True)

        # 1M Returns
        fig.add_trace(
            go.Bar(
                y=perf_sorted['Ticker'],
                x=perf_sorted['1M Return'],
                orientation='h',
                marker_color=perf_sorted['1M Return'].apply(
                    lambda x: 'green' if x > 0 else 'red'
                ),
                text=perf_sorted['1M Return'].apply(lambda x: f"{x:+.1f}%"),
                textposition='outside',
                name='1M'
            ),
            row=1, col=1
        )

        # Sort by 1Y return
        perf_sorted_1y = perf_df.sort_values('1Y Return', ascending=True)

        fig.add_trace(
            go.Bar(
                y=perf_sorted_1y['Ticker'],
                x=perf_sorted_1y['1Y Return'],
                orientation='h',
                marker_color=perf_sorted_1y['1Y Return'].apply(
                    lambda x: 'green' if x > 0 else 'red'
                ),
                text=perf_sorted_1y['1Y Return'].apply(lambda x: f"{x:+.1f}%"),
                textposition='outside',
                name='1Y'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=600,
            showlegend=False
        )
        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Return (%)", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.markdown("### 📋 Sector Performance Table")

        display_df = perf_df[[
            'Ticker', 'Sector', 'Current Price', '1M Return', '3M Return',
            '6M Return', '1Y Return', 'Expense Ratio'
        ]].copy()

        # Calculate momentum score
        display_df['Momentum Score'] = (
            (display_df['1M Return'] * 0.5) +
            (display_df['3M Return'] * 0.3) +
            (display_df['6M Return'] * 0.2)
        ).round(2)

        # Sort by momentum
        display_df = display_df.sort_values('Momentum Score', ascending=False)

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("ETF", width=80),
                "Sector": st.column_config.TextColumn("Sector", width=200),
                "Current Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "1M Return": st.column_config.NumberColumn("1M %", format="%.2f%%"),
                "3M Return": st.column_config.NumberColumn("3M %", format="%.2f%%"),
                "6M Return": st.column_config.NumberColumn("6M %", format="%.2f%%"),
                "1Y Return": st.column_config.NumberColumn("1Y %", format="%.2f%%"),
                "Expense Ratio": st.column_config.NumberColumn("Expense", format="%.3f%%"),
                "Momentum Score": st.column_config.NumberColumn("Momentum", format="%.2f")
            }
        )

    # ====================================================================
    # TAB 2: SECTOR ROTATION
    # ====================================================================

    with tab2:
        st.header("🔄 Sector Rotation Strategy")

        signals = get_sector_rotation_signals()

        if not signals:
            st.error("Unable to load rotation signals")
            return

        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)

        # Summary
        st.markdown("### 📊 Rotation Signals Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            buy_count = len(signals_df[signals_df['Signal'] == 'BUY'])
            st.metric("🟢 Buy Signals", buy_count)

        with col2:
            hold_count = len(signals_df[signals_df['Signal'] == 'HOLD'])
            st.metric("🟡 Hold Signals", hold_count)

        with col3:
            sell_count = len(signals_df[signals_df['Signal'] == 'SELL'])
            st.metric("🔴 Sell Signals", sell_count)

        with col4:
            neutral_count = len(signals_df[signals_df['Signal'] == 'NEUTRAL'])
            st.metric("⚪ Neutral", neutral_count)

        # RRG Chart (Relative Rotation Graph simulation)
        st.markdown("### 🎯 Relative Rotation Graph (RRG)")

        # Calculate RS-Ratio and RS-Momentum from returns
        signals_df['RS-Ratio'] = 100 + signals_df['3M Return']  # Simplified
        signals_df['RS-Momentum'] = signals_df['1M Return'] - signals_df['3M Return']  # Simplified

        # Create quadrant chart
        fig = go.Figure()

        # Define colors by signal
        color_map = {
            'BUY': 'green',
            'HOLD': 'yellow',
            'SELL': 'red',
            'NEUTRAL': 'gray'
        }

        for signal_type in signals_df['Signal'].unique():
            subset = signals_df[signals_df['Signal'] == signal_type]

            fig.add_trace(go.Scatter(
                x=subset['RS-Ratio'],
                y=subset['RS-Momentum'],
                mode='markers+text',
                name=signal_type,
                text=subset['Ticker'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=color_map.get(signal_type, 'gray'),
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>%{text}</b><br>' +
                              'RS-Ratio: %{x:.1f}<br>' +
                              'RS-Momentum: %{y:.1f}<br>' +
                              '<extra></extra>'
            ))

        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.5)

        # Add quadrant labels
        fig.add_annotation(x=105, y=2, text="Leading", showarrow=False, font=dict(size=12, color="green"))
        fig.add_annotation(x=95, y=2, text="Improving", showarrow=False, font=dict(size=12, color="blue"))
        fig.add_annotation(x=95, y=-2, text="Lagging", showarrow=False, font=dict(size=12, color="red"))
        fig.add_annotation(x=105, y=-2, text="Weakening", showarrow=False, font=dict(size=12, color="orange"))

        fig.update_layout(
            title="Sector Rotation - RRG Style",
            xaxis_title="Relative Strength (RS-Ratio)",
            yaxis_title="Momentum (RS-Momentum)",
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Rotation signals table
        st.markdown("### 📋 Detailed Rotation Signals")

        st.dataframe(
            signals_df[[
                'Ticker', 'Sector', 'Signal', 'Strength', 'Momentum Score',
                'Rank', '1M Return', '3M Return', '6M Return'
            ]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("ETF", width=80),
                "Sector": st.column_config.TextColumn("Sector", width=180),
                "Signal": st.column_config.TextColumn("Signal", width=80),
                "Strength": st.column_config.TextColumn("Strength", width=100),
                "Momentum Score": st.column_config.NumberColumn("Momentum", format="%.2f"),
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "1M Return": st.column_config.NumberColumn("1M %", format="%.2f%%"),
                "3M Return": st.column_config.NumberColumn("3M %", format="%.2f%%"),
                "6M Return": st.column_config.NumberColumn("6M %", format="%.2f%%")
            }
        )

        # Faber's strategy
        st.markdown("### 🎯 Faber's Sector Rotation Strategy")

        st.info("""
        **Strategy Rules:**
        1. Rank sectors by 3-month momentum
        2. Buy top 3-4 sectors
        3. Rebalance monthly
        4. Stay invested when S&P 500 > 10-month SMA
        """)

        top_3 = signals_df.nsmallest(3, 'Rank')

        st.markdown("**Top 3 Sectors to Hold (Faber Strategy):**")
        for _, row in top_3.iterrows():
            st.write(f"✅ **{row['Ticker']}** - {row['Sector']} (Momentum: {row['Momentum Score']:.2f})")

    # ====================================================================
    # TAB 3: ECONOMIC INDICATORS
    # ====================================================================

    with tab3:
        st.header("🌍 Economic Indicators & Sector Recommendations")

        # Get economic snapshot
        with st.spinner("Fetching economic data..."):
            snapshot = get_economic_snapshot()

        # Economic cycle header
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### {snapshot['cycle_color']} Current Economic Cycle: **{snapshot['cycle']}**")

        with col2:
            st.caption(f"Updated: {snapshot['updated'].strftime('%Y-%m-%d %H:%M')}")

        # Key indicators
        st.markdown("### 📊 Key Economic Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pmi = snapshot['pmi']
            delta_color = "normal" if pmi['trend'] == 'Improving' else "inverse"
            st.metric(
                "Manufacturing PMI",
                pmi['value'],
                pmi['trend'],
                delta_color=delta_color
            )
            st.caption(pmi['interpretation'])

        with col2:
            gdp = snapshot['gdp']
            st.metric(
                "GDP Growth (YoY)",
                f"{gdp['value']}%",
                gdp['interpretation']
            )

        with col3:
            unemployment = snapshot['unemployment']
            delta_color = "inverse" if unemployment['trend'] == 'Improving' else "normal"
            st.metric(
                "Unemployment Rate",
                f"{unemployment['value']}%",
                unemployment['trend'],
                delta_color=delta_color
            )
            st.caption(unemployment['interpretation'])

        with col4:
            fed = snapshot['fed_funds']
            st.metric(
                "Fed Funds Rate",
                f"{fed['value']}%",
                fed['trend']
            )
            st.caption(fed['interpretation'])

        # Economic cycle recommendations
        st.markdown("### 💡 Sector Recommendations Based on Economic Cycle")

        recommendations = econ_manager.get_sector_recommendations_from_economy(snapshot)

        col1, col2 = st.columns(2)

        with col1:
            st.success("**🟢 Overweight (Favorable Sectors)**")
            for sector in recommendations['overweight']:
                etf_result = etf_manager.get_etf_by_sector(sector)
                if etf_result:
                    ticker, info = etf_result
                    st.write(f"✅ **{ticker}** - {sector}")
                else:
                    st.write(f"✅ {sector}")

        with col2:
            st.error("**🔴 Underweight (Unfavorable Sectors)**")
            for sector in recommendations['underweight']:
                etf_result = etf_manager.get_etf_by_sector(sector)
                if etf_result:
                    ticker, info = etf_result
                    st.write(f"❌ **{ticker}** - {sector}")
                else:
                    st.write(f"❌ {sector}")

        # Economic cycle explanation
        st.markdown("### 📖 Economic Cycle Analysis")

        if "Early" in snapshot['cycle']:
            st.info("""
            **Early Expansion Characteristics:**
            - PMI rising above 50
            - GDP growth accelerating
            - Unemployment declining
            - **Best Sectors:** Industrials, Materials, Technology
            - **Avoid:** Utilities, Consumer Staples (low growth)
            """)
        elif "Mid" in snapshot['cycle']:
            st.info("""
            **Mid Expansion Characteristics:**
            - PMI solidly above 50
            - GDP growing at moderate pace
            - Full employment conditions
            - **Best Sectors:** Technology, Consumer Discretionary, Industrials
            - **Avoid:** Utilities, Real Estate
            """)
        elif "Late" in snapshot['cycle'] or "Slowdown" in snapshot['cycle']:
            st.warning("""
            **Late Cycle / Slowdown Characteristics:**
            - PMI slowing, approaching 50
            - GDP growth decelerating
            - Possible inflation pressures
            - **Best Sectors:** Energy, Financials, Consumer Staples
            - **Avoid:** Industrials, Materials (early cycle plays)
            """)
        else:
            st.error("""
            **Recession / Contraction Characteristics:**
            - PMI below 50 (contraction)
            - GDP negative or very low
            - Rising unemployment
            - **Best Sectors:** Utilities, Consumer Staples, Health Care (defensive)
            - **Avoid:** Consumer Discretionary, Industrials, Materials
            """)

    # ====================================================================
    # TAB 4: ETF PERFORMANCE
    # ====================================================================

    with tab4:
        st.header("📈 Sector ETF Detailed Performance")

        # Performance metrics table
        st.markdown("### 📊 All Sector ETFs")

        st.dataframe(
            perf_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("ETF", width=80),
                "Sector": st.column_config.TextColumn("Sector", width=200),
                "Name": st.column_config.TextColumn("ETF Name", width=300),
                "Current Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "1M Return": st.column_config.NumberColumn("1M %", format="%.2f%%"),
                "3M Return": st.column_config.NumberColumn("3M %", format="%.2f%%"),
                "6M Return": st.column_config.NumberColumn("6M %", format="%.2f%%"),
                "1Y Return": st.column_config.NumberColumn("1Y %", format="%.2f%%"),
                "Expense Ratio": st.column_config.NumberColumn("Expense", format="%.4f")
            }
        )

        # Top holdings
        st.markdown("### 🏆 Top Holdings by Sector")

        selected_etf = st.selectbox(
            "Select Sector ETF:",
            options=perf_df['Ticker'].tolist(),
            format_func=lambda x: f"{x} - {perf_df[perf_df['Ticker']==x]['Sector'].values[0]}"
        )

        if selected_etf:
            etf_info = etf_manager.get_etf_info(selected_etf)
            holdings_df = etf_manager.get_top_holdings_df(selected_etf)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**{etf_info['name']}**")
                st.caption(etf_info['description'])

            with col2:
                st.metric("Expense Ratio", f"{etf_info['expense_ratio']*100:.2f}%")

            if not holdings_df.empty:
                st.dataframe(
                    holdings_df[['symbol', 'name', 'weight']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "symbol": st.column_config.TextColumn("Symbol", width=100),
                        "name": st.column_config.TextColumn("Company", width=250),
                        "weight": st.column_config.NumberColumn("Weight", format="%.2f%%")
                    }
                )

    # ====================================================================
    # TAB 5: SECTOR DEEP DIVE
    # ====================================================================

    with tab5:
        st.header("💼 Sector Deep Dive Analysis")

        # Select sector
        all_sectors = calculator.get_all_sectors()
        selected_sector = st.selectbox(
            "Select Sector for Deep Analysis:",
            options=all_sectors
        )

        if selected_sector:
            # Get sector profile
            profile = calculator.get_sector_profile(selected_sector)
            etf_ticker = calculator.get_sector_etf(selected_sector)
            etf_info = etf_manager.get_etf_info(etf_ticker)

            # Header
            st.markdown(f"## {selected_sector}")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Sector ETF", etf_ticker)
            with col2:
                st.metric("Typical Beta", f"{profile['typical_beta']:.2f}")
            with col3:
                st.metric("Cyclicality", profile['cyclicality'])
            with col4:
                st.metric("Avg Dividend Yield", f"{profile['avg_dividend_yield']:.1f}%")

            # Characteristics
            st.markdown("### 📊 Sector Characteristics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Defensive Sector:** {'Yes' if profile['defensive'] else 'No'}")
                st.markdown(f"**Optimal DTE for Wheel:** {profile['optimal_dte']} days")
                st.markdown(f"**Wheel Strategy Tier:** {profile['wheel_tier']} (1=Best, 3=ETF Better)")

            with col2:
                if etf_info:
                    st.markdown(f"**ETF Name:** {etf_info['name']}")
                    st.markdown(f"**Description:** {etf_info['description']}")

            # Top holdings
            st.markdown("### 🏆 Major Companies in This Sector")

            holdings_df = etf_manager.get_top_holdings_df(etf_ticker)
            if not holdings_df.empty:
                st.dataframe(
                    holdings_df[['symbol', 'name', 'weight']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "symbol": st.column_config.TextColumn("Symbol"),
                        "name": st.column_config.TextColumn("Company"),
                        "weight": st.column_config.NumberColumn("ETF Weight", format="%.2f%%")
                    }
                )

            # Trading recommendations
            st.markdown("### 💡 Trading Strategy Recommendations")

            if profile['wheel_tier'] == 1:
                st.success(f"""
                **Excellent for Wheel Strategy**
                - High premiums due to elevated IV
                - Optimal DTE: {profile['optimal_dte']} days
                - Focus on quality names with strong fundamentals
                - Monitor earnings calendars closely
                """)
            elif profile['wheel_tier'] == 2:
                st.info(f"""
                **Good for Selective Wheel Trades**
                - Moderate premiums
                - Optimal DTE: {profile['optimal_dte']} days
                - Choose stocks during elevated volatility
                - Consider sector rotation timing
                """)
            else:
                st.warning(f"""
                **Better via ETF or Long-DTE**
                - Low premiums on individual stocks
                - Consider sector ETF instead
                - If trading stocks, use {profile['optimal_dte']}+ day DTE
                - Focus on dividend capture strategies
                """)

    # ====================================================================
    # TAB 6: REFERENCE GUIDE
    # ====================================================================

    with tab6:
        st.header("📚 Sector Analysis Reference Guide")

        st.markdown("""
        ## The 11 GICS Sectors

        The Global Industry Classification Standard (GICS) is the definitive framework for sector classification.

        ### Sector Overview

        | Sector | ETF | Typical Weight | Characteristics |
        |--------|-----|----------------|-----------------|
        | Information Technology | XLK | ~28-30% | High growth, innovation-driven, cyclical |
        | Health Care | XLV | ~12-14% | Steady growth, defensive, demographic-driven |
        | Financials | XLF | ~12-13% | Cyclical, rate-sensitive, credit cycle dependent |
        | Communication Services | XLC | ~8-9% | Mixed (defensive telecom + growth media) |
        | Consumer Discretionary | XLY | ~10-11% | Highly cyclical, consumer spending driven |
        | Industrials | XLI | ~8-9% | Cyclical, early cycle indicator |
        | Consumer Staples | XLP | ~6-7% | Defensive, non-cyclical, steady dividends |
        | Energy | XLE | ~3-5% | Volatile, commodity-driven, cyclical |
        | Materials | XLB | ~2-3% | Cyclical, early cycle, commodity-sensitive |
        | Real Estate | XLRE | ~2-3% | Rate-sensitive, income-focused |
        | Utilities | XLU | ~2-3% | Defensive, low volatility, steady dividends |

        ### Economic Cycle Positioning

        **Early Expansion:**
        - Overweight: Industrials, Materials, Technology
        - Underweight: Utilities, Consumer Staples

        **Mid Expansion:**
        - Overweight: Technology, Consumer Discretionary, Industrials
        - Underweight: Utilities, Real Estate

        **Late Cycle/Slowdown:**
        - Overweight: Energy, Financials, Consumer Staples
        - Underweight: Industrials, Materials

        **Recession:**
        - Overweight: Utilities, Consumer Staples, Health Care
        - Underweight: Consumer Discretionary, Industrials, Materials

        ### Key Metrics Explained

        **Momentum Score:** Weighted average of 1M (50%), 3M (30%), 6M (20%) returns

        **RS-Ratio:** Relative Strength vs S&P 500 (>100 = outperforming)

        **RS-Momentum:** Change in RS-Ratio (positive = improving strength)

        **Beta:** Volatility vs market (>1 = more volatile, <1 = less volatile)

        **Sharpe Ratio:** Risk-adjusted returns ((Return - Risk Free) / Std Dev)

        ### Sector Rotation Quadrants

        - **Leading:** High RS-Ratio, Positive RS-Momentum → HOLD/BUY
        - **Improving:** Low RS-Ratio, Positive RS-Momentum → BUY
        - **Weakening:** High RS-Ratio, Negative RS-Momentum → REDUCE
        - **Lagging:** Low RS-Ratio, Negative RS-Momentum → SELL

        ### For More Details

        See the comprehensive sector guide in `docs/COMPREHENSIVE_SECTOR_ANALYSIS_GUIDE.md`
        """)


if __name__ == "__main__":
    display_sector_analysis_enhanced()
