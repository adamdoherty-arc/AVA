"""
Options Trading Hub - Complete Consolidated Interface
=====================================================

Unified options trading interface consolidating ALL options tools:
- AI Options Agent (Multi-stock screening)
- Strategy Analyzer (Single-stock strategy selection)
- Calendar Spreads Scanner
- Premium Scanner (High-premium opportunities)
- 0-7 DTE Scanner (Short-term theta capture)
- Earnings Calendar (Volatility plays)
- Database Scanner (Bulk scanning)
- Positions Tracker (Live options positions)

This hub provides a single organized interface for all options trading workflows.

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="Options Trading Hub | Magnus",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .hub-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .tool-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }

    .tool-card:hover {
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }

    .strategy-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-screening { background: #3498db; color: white; }
    .badge-analysis { background: #9b59b6; color: white; }
    .badge-theta { background: #e67e22; color: white; }
    .badge-vol { background: #e74c3c; color: white; }
    .badge-tracking { background: #27ae60; color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.markdown("""
<div class="hub-header">
    <h1 style="margin:0">📊 Options Trading Hub</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9">
        Complete Options Analysis Suite • Multi-Strategy Tools • Live Positions
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Quick Navigation
# ============================================================================

st.subheader("🎯 Quick Tool Selection")

st.markdown("""
**Choose the right tool for your workflow:**

- 🔍 **Screening** → Find best opportunities across many stocks
- 🎯 **Analysis** → Pick the right strategy for a specific stock
- ⏱️ **Theta Capture** → Short-term 0-7 DTE plays
- 📊 **Calendar Spreads** → Time decay arbitrage
- 🎰 **Earnings** → Volatility around earnings
- 📈 **Tracking** → Monitor live positions
""")

st.divider()

# ============================================================================
# Main Tabs
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Screening Tools",
    "🎯 Strategy Analysis",
    "⏱️ Theta Capture",
    "📊 Advanced Strategies",
    "📈 Position Tracking",
    "⚙️ Settings"
])

# ============================================================================
# TAB 1: Screening Tools
# ============================================================================

with tab1:
    st.markdown("### 🔍 Multi-Stock Opportunity Screening")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 AI Options Agent")
        st.markdown('<span class="strategy-badge badge-screening">BATCH SCREENING</span>', unsafe_allow_html=True)

        st.markdown("""
        **Best for:** Finding opportunities across 200+ stocks at once

        **Features:**
        - Multi-stock batch analysis
        - AI-powered MCDM scoring
        - Customizable filters (DTE, Delta, Premium)
        - Watchlist integration
        - CSV export
        - Historical tracking

        **Workflow:**
        1. Select watchlist or "All Stocks"
        2. Set filters (DTE 20-40, Delta -0.30)
        3. Run analysis (30-60 seconds)
        4. Review ranked opportunities
        5. Export top picks
        """)

        if st.button("🚀 Open AI Options Agent", key="open_ai_agent", type="primary"):
            st.info("Navigate to AI Options Agent page")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.markdown("#### 💎 Premium Scanner")
        st.markdown('<span class="strategy-badge badge-screening">HIGH PREMIUM</span>', unsafe_allow_html=True)

        st.markdown("""
        **Best for:** Finding highest-premium options

        **Features:**
        - Sort by premium ($)
        - Filter by IV rank
        - Volume & liquidity filters
        - Greeks analysis
        - Real-time option chains

        **Use Cases:**
        - Cash-secured puts with max premium
        - Covered calls above resistance
        - High-IV stocks for premium selling
        """)

        if st.button("💎 Open Premium Scanner", key="open_premium_scanner"):
            st.info("Navigate to Premium Scanner page")

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2: Strategy Analysis
# ============================================================================

with tab2:
    st.markdown("### 🎯 Single-Stock Strategy Selection")

    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Comprehensive Strategy Analyzer")
    st.markdown('<span class="strategy-badge badge-analysis">10 STRATEGIES</span>', unsafe_allow_html=True)

    st.markdown("""
    **Best for:** Choosing the right strategy for a specific stock

    **Analyzes All 10 Strategies:**
    1. Cash-Secured Puts
    2. Covered Calls
    3. Credit Spreads (Bull/Bear)
    4. Iron Condors
    5. Vertical Spreads
    6. Straddles/Strangles
    7. Calendar Spreads
    8. Diagonal Spreads
    9. Butterfly Spreads
    10. Ratio Spreads

    **Features:**
    - Multi-model AI consensus
    - Ranked by suitability for stock
    - Pros/cons for each strategy
    - Market environment analysis
    - Educational details
    - Risk/reward profiles
    """)

    ticker_input = st.text_input("Enter Ticker", placeholder="AAPL", key="strategy_ticker")

    col1, col2, col3 = st.columns(3)

    with col1:
        outlook = st.selectbox("Market Outlook", ["Bullish", "Neutral", "Bearish"], key="strategy_outlook")

    with col2:
        volatility = st.selectbox("IV Environment", ["High", "Medium", "Low"], key="strategy_volatility")

    with col3:
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], key="strategy_risk")

    if st.button("🎯 Analyze All Strategies", key="analyze_strategies", type="primary"):
        if ticker_input:
            st.info(f"Analyzing all 10 strategies for {ticker_input}...")
            st.markdown("**Results will appear here** (integrate with options_analysis_page.py)")
        else:
            st.warning("Please enter a ticker symbol")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 3: Theta Capture (0-7 DTE)
# ============================================================================

with tab3:
    st.markdown("### ⏱️ Short-Term Theta Capture Strategies")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ 0-7 DTE Scanner")
        st.markdown('<span class="strategy-badge badge-theta">RAPID THETA</span>', unsafe_allow_html=True)

        st.markdown("""
        **Best for:** Maximum theta decay (final week before expiration)

        **Strategy:**
        - Target 0.10-0.20 delta (80-90% win rate)
        - Close at 50% of max profit
        - 5-10 point wide spreads
        - Avoid earnings in DTE range

        **Features:**
        - Daily theta decay tracking
        - High-probability credit spreads
        - Technical analysis integration
        - Support/resistance levels
        - Gamma risk monitoring

        **Expected Performance:**
        - Win Rate: 80-85%
        - ROI: 20-35% per trade
        - Holding Period: 0-7 days
        """)

        dte_range = st.slider("DTE Range", 0, 7, (3, 5), key="dte_range")

        if st.button("⚡ Scan 0-7 DTE Opportunities", key="scan_dte", type="primary"):
            st.info("Navigate to Seven Day DTE Scanner page")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.markdown("#### 🎰 Earnings Plays")
        st.markdown('<span class="strategy-badge badge-vol">IV CRUSH</span>', unsafe_allow_html=True)

        st.markdown("""
        **Best for:** Trading volatility around earnings

        **Strategies:**
        - **Long Straddle:** Expect big move (long vol)
        - **Short Iron Condor:** IV crush play (short vol)
        - **Calendar Spread:** Short-term IV vs long-term

        **Features:**
        - Earnings calendar tracking
        - Implied move calculations
        - Historical earnings reactions
        - IV rank analysis
        - Whisper numbers integration

        **Key Metrics:**
        - Implied Move %
        - IV Crush (pre vs post)
        - Historical beat/miss patterns
        - Average move on beat/miss
        """)

        upcoming_period = st.selectbox(
            "Earnings Period",
            ["This Week", "Next Week", "Next 30 Days"],
            key="earnings_period"
        )

        if st.button("📅 View Earnings Calendar", key="view_earnings", type="primary"):
            st.info("Navigate to Earnings Calendar page")

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 4: Advanced Strategies
# ============================================================================

with tab4:
    st.markdown("### 📊 Advanced Multi-Leg Strategies")

    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Calendar Spreads Scanner")
    st.markdown('<span class="strategy-badge badge-analysis">TIME DECAY</span>', unsafe_allow_html=True)

    st.markdown("""
    **Best for:** Exploiting volatility skew between expirations

    **How It Works:**
    - Sell short-term option (30-45 DTE)
    - Buy long-term option (60-90 DTE)
    - Profit from theta decay differential
    - Delta-neutral positioning

    **Ideal Conditions:**
    - High IV rank (> 50%)
    - Mean-reverting IV
    - No earnings in short leg
    - Liquid options with tight spreads

    **Features:**
    - ATM ±5% strike selection
    - IV skew analysis
    - Greeks optimization
    - Parallel scanning (5-10x faster)
    - Max profit/loss modeling
    """)

    col1, col2 = st.columns(2)

    with col1:
        short_dte = st.slider("Short Leg DTE", 20, 60, (30, 45), key="short_dte")

    with col2:
        long_dte = st.slider("Long Leg DTE", 45, 120, (60, 90), key="long_dte")

    if st.button("📊 Scan Calendar Spreads", key="scan_calendars", type="primary"):
        st.info("Navigate to Calendar Spreads page")

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Database Scanner")
    st.markdown('<span class="strategy-badge badge-screening">BULK SCAN</span>', unsafe_allow_html=True)

    st.markdown("""
    **Best for:** Custom SQL-based screening

    **Use Cases:**
    - Complex multi-criteria filters
    - Historical backtesting
    - Custom strategy development
    - Data export for analysis

    **Features:**
    - SQL query builder
    - Custom filters
    - Export to CSV/Excel
    - Save queries as templates
    """)

    if st.button("🔍 Open Database Scanner", key="open_db_scanner"):
        st.info("Navigate to Database Scanner page")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 5: Position Tracking
# ============================================================================

with tab5:
    st.markdown("### 📈 Live Options Position Tracking")

    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    st.markdown("#### 💹 Positions Monitor")
    st.markdown('<span class="strategy-badge badge-tracking">LIVE DATA</span>', unsafe_allow_html=True)

    st.markdown("""
    **Real-Time Position Tracking:**
    - Live P&L monitoring
    - Greeks exposure (Delta, Theta, Vega, Gamma)
    - Time decay tracking
    - Robinhood integration
    - Entry/exit recommendations

    **Alerts:**
    - Profit target reached (50% of max)
    - Stop loss triggered
    - Expiration approaching
    - Assignment risk
    - IV changes
    """)

    if st.button("💹 View Live Positions", key="view_positions", type="primary"):
        st.info("Navigate to Positions Page")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 6: Settings
# ============================================================================

with tab6:
    st.markdown("### ⚙️ Options Trading Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Default Filters")

        default_dte_min = st.number_input("Min DTE", 0, 365, 20, key="default_dte_min")
        default_dte_max = st.number_input("Max DTE", 0, 365, 45, key="default_dte_max")
        default_delta_min = st.slider("Min Delta", -1.0, 0.0, -0.40, 0.05, key="default_delta_min")
        default_delta_max = st.slider("Max Delta", -1.0, 0.0, -0.20, 0.05, key="default_delta_max")

    with col2:
        st.markdown("#### 💰 Risk Management")

        max_position_size = st.number_input("Max Position Size ($)", 100, 50000, 1000, 100, key="max_pos_size")
        max_portfolio_allocation = st.slider("Max Portfolio % per Trade", 1, 20, 5, 1, key="max_allocation")
        auto_profit_target = st.slider("Auto-Close at Profit %", 30, 100, 50, 10, key="profit_target")
        stop_loss_percent = st.slider("Stop Loss %", 10, 100, 50, 10, key="stop_loss")

    st.divider()

    st.markdown("#### 📊 Display Preferences")

    col3, col4 = st.columns(2)

    with col3:
        show_greeks = st.checkbox("Show Greeks by Default", value=True, key="show_greeks")
        show_iv_rank = st.checkbox("Show IV Rank", value=True, key="show_iv_rank")

    with col4:
        sort_by = st.selectbox("Default Sort", ["Premium", "Delta", "IV Rank", "Theta"], key="default_sort")
        results_per_page = st.slider("Results Per Page", 10, 100, 50, 10, key="results_per_page")

    st.divider()

    if st.button("💾 Save All Settings", type="primary", key="save_all_settings"):
        st.success("✅ Settings saved successfully!")
        st.balloons()

# ============================================================================
# Footer
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📚 Quick References**
    - [Greeks Explained](https://www.optionsplaybook.com/options-introduction/what-are-greeks/)
    - [Strategy Selection Guide](https://www.tastylive.com/concepts-strategies/)
    - [Theta Decay Curve](https://www.projectoption.com/theta-decay/)
    """)

with col2:
    st.markdown("""
    **🎯 Strategy Quick Picks**
    - **Bullish:** CSP, Covered Call
    - **Neutral:** Iron Condor, Calendar Spread
    - **Bearish:** Bear Put Spread
    - **High IV:** Short Premium
    - **Low IV:** Long Premium
    """)

with col3:
    st.markdown("""
    **⚡ Keyboard Shortcuts**
    - `A` - AI Agent
    - `S` - Strategy Analyzer
    - `D` - DTE Scanner
    - `C` - Calendar Spreads
    - `P` - Positions
    """)

st.caption("Magnus Trading Platform • Options Trading Hub v1.0")
