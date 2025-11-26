"""AVA - Advanced Options Trading Platform"""

# Core imports only - Heavy imports moved to lazy loading for performance
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables at the very start
load_dotenv()

# PERFORMANCE FIX: Set global timeout for all external API calls (prevents page hangs)
import src.api_timeout_config  # Auto-configures 10-second timeout on import

# Optional date parser
try:
    from dateutil import parser
except ImportError:
    parser = None

# Import Omnipresent AVA (used in main flow)
from src.ava.omnipresent_ava_enhanced import show_enhanced_ava as show_omnipresent_ava

# Import Top Betting Picks Widget
try:
    from src.components.top_betting_picks_widget import display_top_picks_compact
    BETTING_WIDGET_AVAILABLE = True
except ImportError:
    BETTING_WIDGET_AVAILABLE = False
    display_top_picks_compact = None

# Lazy import helpers (loaded on-demand for better performance)
_plotly_go = None
_plotly_px = None
_yfinance = None
_redis_client = None
_agents_initialized = False

# Check if agent management page is available
AGENT_MANAGEMENT_AVAILABLE = os.path.exists('agent_management_page.py')

def get_plotly_go():
    """Lazy load plotly graph_objects"""
    global _plotly_go
    if _plotly_go is None:
        import plotly.graph_objects as go
        _plotly_go = go
    return _plotly_go

def get_plotly_px():
    """Lazy load plotly express"""
    global _plotly_px
    if _plotly_px is None:
        import plotly.express as px
        _plotly_px = px
    return _plotly_px

def get_yfinance():
    """Lazy load yfinance"""
    global _yfinance
    if _yfinance is None:
        import yfinance as yf
        _yfinance = yf
    return _yfinance

# Page config
st.set_page_config(
    page_title="AVA Trading Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PERFORMANCE: Cache Warming System
# ============================================================================

# PERFORMANCE FIX: Create executor at module level to prevent memory leaks
# Previously: executor was created inside @st.cache_resource, causing leak
# Now: Single executor shared across all cache warming operations
import atexit
from concurrent.futures import ThreadPoolExecutor
import logging

logger_warming = logging.getLogger(__name__)

# Global executor for background tasks (created once, reused forever)
_background_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache_warmer")

# Register cleanup function to shut down executor on exit
def _cleanup_executor():
    """Shutdown executor gracefully on application exit"""
    logger_warming.info("Shutting down background executor...")
    _background_executor.shutdown(wait=True)
    logger_warming.info("Background executor shutdown complete")

atexit.register(_cleanup_executor)


def background_warm():
    """Warm caches in background"""
    try:
        logger_warming.info("🔥 Starting cache warming...")

        # PERFORMANCE FIX: Corrected import paths for cache warming
        # Warm xtrades cache (most frequently accessed)
        try:
            from xtrades_watchlists_page import (
                get_closed_trades_cached,
                get_active_trades_cached,
                get_active_profiles_cached
            )
            get_closed_trades_cached(limit=100)
            get_active_trades_cached(limit=100)
            get_active_profiles_cached()
            logger_warming.info("✅ XTrades cache warmed")
        except Exception as e:
            logger_warming.warning(f"XTrades cache warming failed: {e}")

        # Warm Kalshi cache
        try:
            from kalshi_nfl_markets_page import KalshiMarketsApp
            app = KalshiMarketsApp()
            app.get_all_markets()  # Method call, not function
            logger_warming.info("✅ Kalshi cache warmed")
        except Exception as e:
            logger_warming.warning(f"Kalshi cache warming failed: {e}")

        logger_warming.info("🎉 Cache warming complete!")

    except Exception as e:
        logger_warming.error(f"Cache warming error: {e}")


@st.cache_resource
def warm_critical_caches():
    """
    Pre-populate critical caches on dashboard startup for instant page loads.
    Submits background task to global executor (no executor creation = no leak).
    """
    # Submit to global executor (no new executor created)
    _background_executor.submit(background_warm)
    return True

# Warm caches on startup (called once)
if 'caches_warmed' not in st.session_state:
    warm_critical_caches()
    st.session_state.caches_warmed = True

# Custom CSS
st.markdown("""
<style>
    .stMetric .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
    .danger-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }

    /* Clean sidebar navigation - extremely compact */
    section[data-testid="stSidebar"] .stButton button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0rem 0.3rem !important;
        margin: 0 !important;
        text-align: left !important;
        width: 100% !important;
        font-size: 0.7rem !important;
        line-height: 0.75 !important;
        height: auto !important;
        min-height: 0 !important;
        transition: all 0.2s ease;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(151, 166, 195, 0.45) !important;
        border-left: 8px solid #667eea !important;
        border-right: 3px solid #667eea !important;
        padding: 0.6rem 0.9rem !important;
        padding-left: 1.2rem !important;
        margin-left: -3px !important;
        transform: translateX(3px) scale(1.02);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3) !important;
    }

    section[data-testid="stSidebar"] .element-container {
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
    }

    section[data-testid="stSidebar"] {
        padding-top: 0.1rem;
    }

    /* Extremely tight section headers in sidebar */
    section[data-testid="stSidebar"] .stMarkdown h3 {
        margin-top: 0.1rem !important;
        margin-bottom: 0rem !important;
        padding: 0 !important;
        font-size: 0.7rem !important;
        line-height: 0.8 !important;
    }

    /* Remove extra spacing between buttons */
    section[data-testid="stSidebar"] [data-testid="column"] > div {
        gap: 0 !important;
    }

    /* Remove all spacing from dividers */
    section[data-testid="stSidebar"] hr {
        margin: 0rem !important;
        padding: 0rem !important;
    }

    /* Limit watchlist dropdown height for better UX */
    .stSelectbox [data-baseweb="select"] > div {
        max-height: 400px !important;
        overflow-y: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Redis connection (lazy - only when needed)
@st.cache_resource
def init_redis():
    import redis
    return redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True
    )

# Initialize agents (lazy - only when needed)
@st.cache_resource
def init_agents():
    from src.agents.runtime.market_data_agent import MarketDataAgent
    from src.agents.runtime.wheel_strategy_agent import WheelStrategyAgent
    from src.agents.runtime.risk_management_agent import RiskManagementAgent

    redis_client = init_redis()
    market_agent = MarketDataAgent(redis_client, max_price=50.0)
    strategy_agent = WheelStrategyAgent(redis_client)
    risk_agent = RiskManagementAgent(redis_client)
    return market_agent, strategy_agent, risk_agent

# Don't initialize immediately - let pages call init_agents() when needed
# This saves 1-2 seconds on initial dashboard load
def get_agents():
    """Get initialized agents (lazy loading)"""
    return init_agents()

# Sidebar - Add prominent AVA Platform header at very top (compact)
st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 4px;
    border-radius: 6px;
    margin-bottom: 10px;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
">
    <h1 style="color: white; margin: 0; font-size: 12px; font-weight: 700;">
        🤖 AVA PLATFORM
    </h1>
</div>
""", unsafe_allow_html=True)

# Navigation as buttons/links
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ==================== FINANCE ====================
st.sidebar.markdown("### 💰 Finance")
if st.sidebar.button("📈 Dashboard", width='stretch'):
    st.session_state.page = "Dashboard"
if st.sidebar.button("💼 Positions", width='stretch'):
    st.session_state.page = "Positions"
if st.sidebar.button("🏭 Sector Analysis", width='stretch'):
    st.session_state.page = "Sector Analysis"
if st.sidebar.button("📊 Watchlists", use_container_width=True):
    st.session_state.page = "TradingView Watchlists"
if st.sidebar.button("💎 Premium Scanner", width='stretch'):
    st.session_state.page = "Premium Scanner"
if st.sidebar.button("📅 Earnings Calendar", width='stretch'):
    st.session_state.page = "Earnings Calendar"
if st.sidebar.button("📱 XTrade Messages", width='stretch'):
    st.session_state.page = "XTrade Messages"
if st.sidebar.button("📊 Technical Indicators", width='stretch'):
    st.session_state.page = "Technical Indicators"
if st.sidebar.button("🎯 Options Analysis", width='stretch'):
    st.session_state.page = "Options Analysis"

st.sidebar.markdown("---")

# ==================== PREDICTION MARKETS ====================
st.sidebar.markdown("### 🎲 Prediction Markets")
if st.sidebar.button("🏟️ Sports Game Hub", width='stretch'):
    st.session_state.page = "Sports Game Hub"
# if st.sidebar.button("🎯 AI Sports Predictions", width='stretch'):
#     st.session_state.page = "AI Sports Predictions"
# NOTE: Page disabled - requires prediction_markets_enhanced.py
if st.sidebar.button("🎲 Kalshi Markets", width='stretch'):
    st.session_state.page = "Prediction Markets"

# ==================== SMART SUBSCRIPTIONS WIDGET ====================
try:
    from src.game_watchlist_manager import GameWatchlistManager
    from src.kalshi_db_manager import KalshiDBManager

    # Initialize user_id
    if 'user_id' not in st.session_state:
        st.session_state.user_id = os.getenv('TELEGRAM_USER_ID', 'default_user')

    # Get subscription stats
    db = KalshiDBManager()
    watchlist_mgr = GameWatchlistManager(db)
    stats = watchlist_mgr.get_watchlist_stats(st.session_state.user_id)

    # Only show widget if user has subscriptions
    if stats.get('total', 0) > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 My Subscriptions")

        # Show quick stats
        st.sidebar.metric("Total Games", stats.get('total', 0))

        # Sport breakdown
        sport_parts = []
        if stats.get('nfl', 0) > 0:
            sport_parts.append(f"🏈 {stats['nfl']}")
        if stats.get('ncaa', 0) > 0:
            sport_parts.append(f"🎓 {stats['ncaa']}")
        if stats.get('nba', 0) > 0:
            sport_parts.append(f"🏀 {stats['nba']}")

        if sport_parts:
            st.sidebar.caption(" | ".join(sport_parts))

        # Manage button
        if st.sidebar.button("⚙️ Manage Subscriptions", key="manage_subs_sidebar", use_container_width=True):
            st.session_state.page = "Sports Game Hub"
            st.rerun()

except Exception as e:
    # Gracefully handle if watchlist not available
    pass

st.sidebar.markdown("---")

# ==================== AVA MANAGEMENT ====================
st.sidebar.markdown("### 🤖 AVA Management")
if st.sidebar.button("🤖 Agent Management", width='stretch'):
    st.session_state.page = "Agent Management"
if st.sidebar.button("🔍 Cache Metrics", width='stretch'):
    st.session_state.page = "Cache Metrics"
if st.sidebar.button("⚙️ Settings", width='stretch'):
    st.session_state.page = "Settings"
if st.sidebar.button("🔧 Enhancement Agent", width='stretch'):
    st.session_state.page = "Enhancement Agent"
if st.sidebar.button("🚀 Enhancement Manager", width='stretch'):
    st.session_state.page = "Enhancement Manager"

page = st.session_state.page

# No connection status displayed

# Set default values for removed variables
max_price = 50
min_premium = 1.0
profit_target = 50
watchlist = []

# Show Omnipresent AVA at top of all pages
show_omnipresent_ava()

# Main content based on page selection
if page == "Dashboard":
    st.title("💰 AVA Performance & Forecasts")

    # Get account data and positions if connected
    account_data = {}
    positions = []

    if st.session_state.get('rh_connected') and 'rh_functions' in st.session_state:
        try:
            get_account = st.session_state['rh_functions']['get_account']
            get_options = st.session_state['rh_functions']['get_options']
            get_positions = st.session_state['rh_functions']['get_positions']
            identify_wheel = st.session_state['rh_functions']['identify_wheel']

            account_data = get_account()
            stocks = get_positions()
            options = get_options()
            wheel_positions = identify_wheel(stocks, options)

            # Convert to dashboard format
            for wp in wheel_positions:
                if wp['strategy'] == 'CSP':
                    positions.append({
                        'Symbol': wp['symbol'],
                        'Type': 'CSP',
                        'Strike': wp.get('strike', 0),
                        'Expiration': wp.get('expiration', ''),
                        'Premium': wp.get('premium', 0),
                        'Current Value': wp.get('current_value', 0),
                        'Days to Expiry': wp.get('days_to_expiry', 0),
                        'Quantity': wp.get('contracts', 1)
                    })
        except:
            pass

    # Current Status Section
    st.markdown("### 📊 Current Portfolio Status")

    # Check if we have real Robinhood data
    if account_data:
        col1, col2, col3, col4, col5 = st.columns(5)

        current_balance = float(account_data.get('portfolio_value', 0))
        buying_power = float(account_data.get('buying_power', 0))
        total_premium_collected = sum(p.get('Premium', 0) for p in positions)
        positions_at_risk = sum(p.get('Strike', 0) * 100 for p in positions if p.get('Type') == 'CSP')

        with col1:
            st.metric("Current Balance", f"${current_balance:,.0f}")
        with col2:
            st.metric("Buying Power", f"${buying_power:,.0f}")
        with col3:
            st.metric("Premium Collected", f"${total_premium_collected:,.0f}")
        with col4:
            st.metric("Capital at Risk", f"${positions_at_risk:,.0f}")
        with col5:
            st.metric("Active CSPs", len(positions))
    else:
        st.info("💡 Connect to Robinhood in the Settings tab to see your real portfolio data")

    # === TOP BETTING PICKS WIDGET ===
    # Show top betting opportunities in compact format
    if BETTING_WIDGET_AVAILABLE:
        display_top_picks_compact(max_picks=10)

    # Balance Forecast Timeline
    st.markdown("### 📅 Balance Forecast Timeline")

    if positions:
        # Group positions by expiration date
        from collections import defaultdict
        positions_by_date = defaultdict(list)

        for pos in positions:
            if pos.get('Expiration'):
                positions_by_date[pos['Expiration']].append(pos)

        # Sort dates
        sorted_dates = sorted(positions_by_date.keys())

        # Create forecast timeline
        forecast_data = []
        running_balance = current_balance

        for exp_date in sorted_dates:
            date_positions = positions_by_date[exp_date]

            # Calculate outcomes for this date
            premium_income = sum(p.get('Premium', 0) for p in date_positions)
            capital_deployed = sum(p.get('Strike', 0) * 100 for p in date_positions)

            # Best case: All CSPs expire worthless, keep premium
            best_case = running_balance + premium_income

            # Worst case: All CSPs assigned, deploy capital but keep premium
            worst_case = running_balance - capital_deployed + premium_income

            # Expected case: 70% expire worthless (based on typical OTM probabilities)
            expected_case = running_balance + (premium_income * 0.7) - (capital_deployed * 0.3)

            forecast_data.append({
                'Date': exp_date,
                'Positions': len(date_positions),
                'Premium Income': premium_income,
                'Capital at Risk': capital_deployed,
                'Best Case Balance': best_case,
                'Expected Balance': expected_case,
                'Worst Case Balance': worst_case
            })

            running_balance = expected_case  # Use expected for next calculation

        # Display forecast table
        if forecast_data:
            st.markdown("#### Expiration Date Projections")

            for forecast in forecast_data:
                with st.expander(f"📅 {forecast['Date']} - {forecast['Positions']} Position(s)"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Premium Income", f"${forecast['Premium Income']:,.0f}")
                        st.metric("Capital at Risk", f"${forecast['Capital at Risk']:,.0f}")

                    with col2:
                        st.metric("Best Case (All Expire)", f"${forecast['Best Case Balance']:,.0f}",
                                 f"+${forecast['Best Case Balance'] - current_balance:,.0f}")
                        st.metric("Expected (70% Expire)", f"${forecast['Expected Balance']:,.0f}",
                                 f"+${forecast['Expected Balance'] - current_balance:,.0f}")

                    with col3:
                        st.metric("Worst Case (All Assigned)", f"${forecast['Worst Case Balance']:,.0f}",
                                 f"{forecast['Worst Case Balance'] - current_balance:+,.0f}")

                        # Show positions for this date
                        st.markdown("**Positions Expiring:**")
                        for pos in positions_by_date[forecast['Date']]:
                            st.caption(f"• {pos['Symbol']} ${pos['Strike']} Put")

            # Summary Statistics
            st.markdown("#### 📈 Forecast Summary")
            col1, col2, col3, col4 = st.columns(4)

            total_premium = sum(f['Premium Income'] for f in forecast_data)
            max_capital_risk = max(f['Capital at Risk'] for f in forecast_data)
            final_best = forecast_data[-1]['Best Case Balance'] if forecast_data else current_balance
            final_expected = forecast_data[-1]['Expected Balance'] if forecast_data else current_balance

            with col1:
                st.metric("Total Premium Potential", f"${total_premium:,.0f}")
            with col2:
                st.metric("Max Capital at Risk", f"${max_capital_risk:,.0f}")
            with col3:
                st.metric("Expected Final Balance", f"${final_expected:,.0f}",
                         f"+{((final_expected/current_balance - 1) * 100):.1f}%")
            with col4:
                monthly_return = ((final_expected/current_balance - 1) * 100) / max(1, len(sorted_dates))
                st.metric("Avg Monthly Return", f"{monthly_return:.2f}%")

    # Detailed Position Forecast
    st.markdown("### 🎯 Individual Position Forecasts")

    if positions:
        # Create detailed forecast for each position
        for pos in positions:
            if pos.get('Type') == 'CSP' and pos.get('Days to Expiry', 0) > 0:
                with st.expander(f"{pos['Symbol']} - ${pos['Strike']} Put - Expires {pos['Expiration']}"):
                    col1, col2, col3, col4 = st.columns(4)

                    # Calculate probabilities (simplified Black-Scholes approximation)
                    days_left = pos['Days to Expiry']
                    premium = pos.get('Premium', 0)
                    strike = pos.get('Strike', 0)
                    current_value = abs(pos.get('Current Value', 0))

                    # Profit scenarios
                    profit_if_expires = premium
                    profit_if_closed_now = premium - current_value
                    daily_theta = profit_if_closed_now / max(1, days_left)

                    with col1:
                        st.markdown("**Position Details**")
                        st.metric("Strike Price", f"${strike:.2f}")
                        st.metric("Days to Expiry", days_left)
                        st.metric("Premium Collected", f"${premium:.2f}")

                    with col2:
                        st.markdown("**Current Status**")
                        st.metric("Current P&L", f"${profit_if_closed_now:.2f}",
                                 f"{(profit_if_closed_now/premium*100):.1f}%" if premium > 0 else "0%")
                        st.metric("Daily Theta", f"${daily_theta:.2f}")
                        st.metric("Cost to Close", f"${current_value:.2f}")

                    with col3:
                        st.markdown("**Forecast (If Expires OTM)**")
                        st.metric("Max Profit", f"${profit_if_expires:.2f}")
                        st.metric("Return on Risk", f"{(premium/(strike*100)*100):.2f}%")
                        st.metric("Annualized Return", f"{(premium/(strike*100)*365/max(30, days_left)*100):.1f}%")

                    with col4:
                        st.markdown("**Risk (If Assigned)**")
                        st.metric("Capital Required", f"${strike * 100:.2f}")
                        st.metric("Breakeven Price", f"${strike - premium/100:.2f}")

                        # Simple assignment probability based on moneyness
                        info = safe_get_info(pos['Symbol'], suppress_warnings=True)
                        if info:
                            current_stock_price = info.get('currentPrice', strike)
                            if current_stock_price < strike:
                                assignment_prob = min(90, 50 + ((strike - current_stock_price)/strike * 100))
                            else:
                                assignment_prob = max(10, 50 - ((current_stock_price - strike)/strike * 100))
                            st.metric("Assignment Prob", f"{assignment_prob:.0f}%")
                        else:
                            st.metric("Assignment Prob", "N/A")
    else:
        st.info("No active positions. Connect to Robinhood or open some positions to see forecasts.")

    # Historical Performance (if data available)
    st.markdown("### 📊 Historical Performance")

    # This would show actual historical data if available
    if st.session_state.get('rh_connected'):
        st.info("Historical performance data will be populated as you complete trades")
    else:
        st.info("Connect to Robinhood to track historical performance")

# Opportunities feature removed - premium analysis integrated into TradingView Watchlists and Premium Scanner

elif page == "Sports Game Hub":
    from sports_betting_hub_page import show_sports_betting_hub
    show_sports_betting_hub()

elif page == "Positions":
    # Use improved Positions page with all enhancements
    from positions_page_improved import show_positions_page
    show_positions_page()

# Premium Scanner page removed - functionality integrated into TradingView Watchlists and new unified Premium Scanner

elif page == "TradingView Watchlists":
    st.title("📊 Watchlists")
    
    # Sync status widget
    from src.components.sync_status_widget import SyncStatusWidget
    sync_widget = SyncStatusWidget()
    sync_widget.display(
        table_name="tradingview",
        title="Watchlist Sync",
        compact=True
    )

    # Import modules
    from src.tradingview_db_manager import TradingViewDBManager
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    import robin_stocks.robinhood as rh

    # Initialize DB sync
    tv_manager = TradingViewDBManager()

    # NOTE: Positions and Trade History have been moved to the "Positions" page
    # This page now focuses only on TradingView watchlist management

    # Check if we need pre-market sync (before 9:30 AM ET)
    import pytz
    et_tz = pytz.timezone('America/New_York')
    current_time_et = datetime.now(et_tz)
    market_open_time = current_time_et.replace(hour=9, minute=30, second=0, microsecond=0)

    # Auto-sync daily before market open
    last_sync_date = st.session_state.get('last_watchlist_sync_date')
    today = current_time_et.date()

    if last_sync_date != today and current_time_et < market_open_time:
        st.info("🔄 Running pre-market watchlist sync...")
        # Mark as synced for today
        st.session_state['last_watchlist_sync_date'] = today

    # Auto-sync on page load (if not synced recently - every 5 minutes during market hours)


    # Auto-sync on page load (if not synced recently)
    if 'last_sync' not in st.session_state or (datetime.now() - st.session_state.get('last_sync', datetime.min)).seconds > 300:
        with st.spinner("Loading watchlists from database..."):
            # Load watchlists from database
            watchlists_db = tv_manager.get_all_symbols_dict()

            if not watchlists_db:
                # First time - suggest syncing from TradingView
                st.info("💡 No watchlists found. Please sync from TradingView using: python src/tradingview_api_sync.py")

            st.session_state['watchlists_db'] = watchlists_db
            st.session_state['last_sync'] = datetime.now()

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔄 Auto-Sync", "📥 Import Watchlist", "📊 My Watchlist Analysis", "📈 Saved Watchlists", "🎯 Quick Scan", "📆 Calendar Spreads"])

    with tab1:
        # Refresh button (compact)
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button("🔄 Refresh Watchlists", use_container_width=True):
                with st.spinner("Loading watchlists from database..."):
                    # Load all watchlists from database
                    watchlists = tv_manager.get_all_symbols_dict()
                    st.session_state['watchlists_db'] = watchlists
                    st.session_state['last_sync'] = datetime.now()

                    if watchlists:
                        total_symbols = sum(len(symbols) for symbols in watchlists.values())
                        st.success(f"✅ Loaded {len(watchlists)} watchlists with {total_symbols} symbols")
                    else:
                        st.info("No watchlists found. Import symbols below to create watchlists.")
                    st.rerun()
        with col2:
            if 'last_sync' in st.session_state:
                time_since = (datetime.now() - st.session_state['last_sync']).seconds // 60
                st.caption(f"⏱️ {time_since}m")

        # Load watchlists on first run
        if 'watchlists_db' not in st.session_state:
            st.session_state['watchlists_db'] = tv_manager.get_all_symbols_dict()

        watchlists_db = st.session_state.get('watchlists_db', {})

        if watchlists_db:
            # Watchlist selector (no redundant subheader)
            selected_watchlist = st.selectbox(
                "Select Watchlist",
                list(watchlists_db.keys()),
                format_func=lambda x: f"{x} ({len(watchlists_db[x])} stocks)",
                help="Choose a watchlist to view premium opportunities"
            )

            if selected_watchlist:
                symbols = watchlists_db[selected_watchlist]

                # Filter to stock symbols only (no crypto)
                stock_symbols = [s for s in symbols if not any(x in s for x in ['USDT', 'USD', 'BTC', '.D', 'WETH'])]

                if not stock_symbols:
                    st.warning(f"No stock symbols found in {selected_watchlist}. This watchlist contains only crypto/non-stock symbols.")
                else:
                    # Sync section - compact 2-column layout
                    col_main, col_sync = st.columns([3, 1])
                    with col_main:
                        st.metric("Stocks", len(stock_symbols))
                    with col_sync:
                        if st.button("🔄 Sync", type="primary", use_container_width=True):
                            st.success("⚡ Syncing in background...")
                            # Start background sync (truly non-blocking)
                            import subprocess
                            subprocess.Popen([
                                "python", "src/watchlist_sync_service.py", selected_watchlist
                            ], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

                    # SIMPLE IMPLEMENTATION - Just show 30-day options with 0.25-0.40 delta
                    st.markdown("💵 **Cash-Secured Put Options** • 30 DTE • Δ 0.25-0.40")

                    # Compact filters
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        min_stock_price = st.number_input("Min Stock $", value=0.0, min_value=0.0, step=10.0, key="filter_min_stock")
                    with col_f2:
                        max_stock_price = st.number_input("Max Stock $", value=10000.0, min_value=10.0, step=50.0, key="filter_max_stock")
                    with col_f3:
                        min_premium = st.number_input("Min Premium $", value=0.0, min_value=0.0, step=1.0, key="filter_min_prem")

                    # Get 30-day options with ~0.3 delta
                    conn = tv_manager.get_connection()
                    cur = conn.cursor()

                    query = """
                        SELECT DISTINCT ON (sp.symbol)
                            sp.symbol,
                            sd.current_price as stock_price,
                            sp.strike_price,
                            sp.dte,
                            sp.premium,
                            sp.delta,
                            sp.monthly_return,
                            sp.implied_volatility as iv,
                            sp.bid,
                            sp.ask,
                            sp.volume,
                            sp.open_interest as oi
                        FROM stock_premiums sp
                        LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
                        WHERE sp.symbol = ANY(%s)
                            AND sp.dte BETWEEN 28 AND 32
                            AND ABS(sp.delta) BETWEEN 0.25 AND 0.40
                            AND sp.premium >= %s
                            AND sd.current_price BETWEEN %s AND %s
                        ORDER BY sp.symbol, sp.monthly_return DESC
                    """

                    cur.execute(query, (stock_symbols, min_premium, min_stock_price, max_stock_price))
                    rows = cur.fetchall()
                    cur.close()
                    conn.close()

                    # Display results
                    if not rows:
                        st.warning(f"⏳ No 30-day options found for {selected_watchlist}. Try adjusting filters or click 'Sync Prices & Premiums'.")
                    else:
                        # Create simple DataFrame
                        df = pd.DataFrame(rows, columns=[
                            'Symbol', 'Stock Price', 'Strike', 'DTE',
                            'Premium', 'Delta', 'Monthly %', 'IV',
                            'Bid', 'Ask', 'Volume', 'OI'
                        ])

                        # Convert to numeric types
                        numeric_cols = ['Stock Price', 'Strike', 'DTE', 'Premium', 'Delta',
                                       'Monthly %', 'IV', 'Bid', 'Ask', 'Volume', 'OI']
                        for col in numeric_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Options", len(df))
                        col2.metric("Avg Premium", f"${df['Premium'].mean():.2f}")
                        col3.metric("Monthly %", f"{df['Monthly %'].mean():.1f}%")
                        col4.metric("Avg IV", f"{df['IV'].mean():.0f}%")

                        # Results table
                        st.dataframe(
                            df,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                                "Stock Price": st.column_config.NumberColumn("Stock $", format="$%.2f"),
                                "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                "DTE": st.column_config.NumberColumn("DTE"),
                                "Premium": st.column_config.NumberColumn("Premium", format="$%.2f"),
                                "Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                                "Monthly %": st.column_config.NumberColumn("Monthly %", format="%.2f%%"),
                                "IV": st.column_config.NumberColumn("IV %", format="%.1f%%"),
                                "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                                "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                                "Volume": st.column_config.NumberColumn("Vol"),
                                "OI": st.column_config.NumberColumn("OI")
                            }
                        )

        else:
            st.warning("No watchlists found. Click 'Refresh Watchlists' to fetch from TradingView.")

    with tab2:
        st.subheader("📥 Import Watchlist")

        # Text area for importing symbols
        watchlist_text = st.text_area(
            "Enter symbols (comma or line separated):",
            placeholder="NVDA, AMD, AAPL, MSFT, TSLA, META, GOOGL",
            height=150,
            help="Paste from TradingView, Robinhood, or any source"
        )

        watchlist_name = st.text_input("Watchlist Name:", value="My Watchlist", max_chars=50)

        col1, col2 = st.columns(2)
        with col1:
            import_button = st.button("💾 Save & Analyze Watchlist", type="primary", use_container_width=True)
        with col2:
            # Load existing watchlist
            tv_integration = TradingViewDBManager()
            saved_lists = tv_integration.load_saved_watchlists()
            selected_list = st.selectbox(
                "Or load saved:",
                list(saved_lists.keys()),
                help="Load a previously saved watchlist"
            )

        if import_button and watchlist_text:
            # Import the symbols
            tv_integration = TradingViewDBManager()
            imported_symbols = tv_integration.import_from_text(watchlist_text, watchlist_name)

            if imported_symbols:
                st.success(f"✅ Imported {len(imported_symbols)} symbols to '{watchlist_name}'")
                st.session_state['current_watchlist'] = imported_symbols
                st.session_state['current_watchlist_name'] = watchlist_name

                # Display imported symbols
                st.markdown("### Imported Symbols:")
                st.write(", ".join(imported_symbols))
            else:
                st.error("No valid symbols found. Please check your input.")

        elif selected_list:
            # Load selected watchlist
            if st.button(f"Load '{selected_list}'"):
                st.session_state['current_watchlist'] = saved_lists[selected_list]
                st.session_state['current_watchlist_name'] = selected_list
                st.success(f"Loaded {len(saved_lists[selected_list])} symbols from '{selected_list}'")

    with tab3:
        st.subheader("📊 Watchlist Analysis")

        if not st.session_state.get('current_watchlist'):
            st.warning("👆 Import a watchlist first")
        else:
            symbols = st.session_state['current_watchlist']
            watchlist_name = st.session_state.get('current_watchlist_name', 'My Watchlist')

            st.caption(f"**{watchlist_name}** ({len(symbols)} stocks) • {', '.join(symbols[:15])}{'...' if len(symbols) > 15 else ''}")

            # Analyze button
            if st.button("📊 Analyze All Stocks & Premiums", type="primary", use_container_width=True):
                with st.spinner(f"Getting real-time data for {len(symbols)} stocks..."):

                    # Get comprehensive data with options
                    # Get comprehensive data with options using market data agent
                    market_agent = MarketDataAgent()
                    table_data = []
                    for symbol in symbols:
                        try:
                            data = market_agent.get_stock_data(symbol)
                            if data:
                                table_data.append(data)
                        except:
                            continue

                    if table_data:
                        st.session_state['watchlist_analysis'] = table_data
                        st.session_state['analysis_time'] = datetime.now()

            # Display results
            if st.session_state.get('watchlist_analysis'):
                results = st.session_state['watchlist_analysis']

                if 'analysis_time' in st.session_state:
                    st.caption(f"✅ {len(results)} stocks analyzed at {st.session_state['analysis_time'].strftime('%I:%M %p')}")

                # Create DataFrame
                df = pd.DataFrame(results)

                # Format columns
                for col in df.columns:
                    if 'Price' in col or 'Strike' in col:
                        df[col] = df[col].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and x > 0 else "")
                    elif 'Change' in col and '%' not in col:
                        df[col] = df[col].apply(lambda x: f"{x:+.2f}" if isinstance(x, (int, float)) else "")
                    elif '% Change' in col or 'Return%' in col:
                        df[col] = df[col].apply(lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) else "")
                    elif 'Premium' in col or 'Capital' in col:
                        df[col] = df[col].apply(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) and x > 0 else "")
                    elif 'Volume' in col:
                        df[col] = df[col].apply(lambda x: f"{x/1e6:.1f}M" if isinstance(x, (int, float)) and x > 1e6 else f"{x/1e3:.0f}K" if x > 1e3 else str(int(x)) if x > 0 else "")

                # Color code the % Change column
                def highlight_changes(val):
                    if isinstance(val, str) and '+' in val:
                        return 'color: green'
                    elif isinstance(val, str) and '-' in val:
                        return 'color: red'
                    return ''

                # Display table with color coding
                st.markdown("### 📈 Analysis Results")

                # Apply styling and display
                styled_df = df.style.map(highlight_changes, subset=[col for col in df.columns if '% Change' in col])

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )

                # Summary metrics
                st.markdown("### 📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_up = len([r for r in results if r.get('% Change', 0) > 0])
                    total_down = len([r for r in results if r.get('% Change', 0) < 0])
                    st.metric("🟢 Up Today", total_up)
                    st.metric("🔴 Down Today", total_down)

                with col2:
                    # Find best returns
                    best_returns = []
                    for row in results:
                        for key in row:
                            if 'Return%' in key:
                                try:
                                    val = row[key]
                                    if isinstance(val, (int, float)):
                                        best_returns.append(val)
                                except:
                                    continue

                    if best_returns:
                        st.metric("Best Return", f"{max(best_returns):.2f}%")
                        st.metric("Avg Return", f"{sum(best_returns)/len(best_returns):.2f}%")

                with col3:
                    # Market overview
                    avg_change = sum(r.get('% Change', 0) for r in results) / len(results) if results else 0
                    st.metric("Market Avg Today", f"{avg_change:+.2f}%")

                    # Count stocks with options
                    stocks_with_options = len([r for r in results if any('Strike' in k for k in r.keys())])
                    st.metric("With Options", stocks_with_options)

                with col4:
                    # Export button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Analysis CSV",
                        data=csv,
                        file_name=f"{watchlist_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

            else:
                st.info("👆 Click 'Analyze All Stocks & Premiums' to see complete analysis")

    with tab4:
        st.subheader("📈 Saved Watchlists")

        tv_integration = TradingViewDBManager()
        saved_lists = tv_integration.load_saved_watchlists()

        if saved_lists:
            st.info(f"You have {len(saved_lists)} saved watchlists")

            for list_name, symbols in saved_lists.items():
                with st.expander(f"{list_name} ({len(symbols)} symbols)"):
                    st.write(", ".join(symbols))

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"Load", key=f"saved_load_{list_name}"):
                            st.session_state['current_watchlist'] = symbols
                            st.session_state['current_watchlist_name'] = list_name
                            st.rerun()
                    with col2:
                        if st.button(f"Analyze", key=f"saved_analyze_{list_name}"):
                            st.session_state['current_watchlist'] = symbols
                            st.session_state['current_watchlist_name'] = list_name
                            st.info("Go to 'My Watchlist Analysis' tab")
                    with col3:
                        if st.button(f"Delete", key=f"saved_delete_{list_name}"):
                            # Remove from database
                            tv_integration = TradingViewDBManager()
                            tv_integration.delete_watchlist(list_name)
                            st.rerun()
        else:
            st.info("No saved watchlists. Import symbols in the first tab to get started.")

    with tab5:
        st.subheader("🎯 Quick Scan")
        st.info("📊 **Watchlist Premium Scanner** - Scans only symbols in selected watchlist (vs Premium Scanner which can scan all stocks)")

        st.info("No predefined lists available. Please use 'Import Watchlist' or 'Premium Scanner' to load stocks.")

    with tab6:
        st.subheader("📆 Calendar Spread Opportunities")
        st.caption("Find calendar spreads for symbols in your watchlists")

        # Info banner
        with st.expander("ℹ️ What are Calendar Spreads?"):
            st.markdown("""
            **Calendar Spreads** (also called time spreads or horizontal spreads) involve:
            - **Selling** a near-term option (30-45 days)
            - **Buying** a longer-term option (60-90 days)
            - **Same strike price** and option type

            ### How You Profit:
            - **Time Decay**: Short option decays faster than long option
            - **Theta Differential**: Earn the difference in decay rates
            - **Best When**: Stock stays near strike price at short expiration

            ### Risk Profile:
            - **Max Loss**: Net premium paid (limited risk)
            - **Max Profit**: Depends on long option value when short expires
            - **Ideal Conditions**: Low IV (<30%), range-bound markets

            ### Our AI Scoring (0-100):
            - Theta differential (30%)
            - IV level (25%)
            - Moneyness/ATM proximity (20%)
            - Timing (15%)
            - Liquidity (10%)
            """)

        # Login to Robinhood if needed
        if not st.session_state.get('rh_calendar_logged_in'):
            with st.spinner("Connecting to Robinhood..."):
                try:
                    import robin_stocks.robinhood as rh
                    # SECURITY FIX: Use environment variables instead of hardcoded credentials
                    import os
                    rh_username = os.getenv('ROBINHOOD_USERNAME')
                    rh_password = os.getenv('ROBINHOOD_PASSWORD')
                    if not rh_username or not rh_password:
                        raise ValueError("ROBINHOOD_USERNAME or ROBINHOOD_PASSWORD not found in environment")
                    rh.login(rh_username, rh_password)
                    st.session_state['rh_calendar_logged_in'] = True
                except Exception as e:
                    st.error(f"Failed to connect to Robinhood: {e}")
                    st.info("Calendar Spreads require Robinhood connection for options data.")

        if st.session_state.get('rh_calendar_logged_in'):
            # Get available watchlists
            watchlists_db = st.session_state.get('watchlists_db', {})

            if not watchlists_db:
                st.warning("No watchlists found. Please use the 'Auto-Sync' tab to load watchlists first.")
            else:
                # Watchlist selector
                selected_watchlist = st.selectbox(
                    "Select Watchlist",
                    list(watchlists_db.keys()),
                    format_func=lambda x: f"{x} ({len(watchlists_db[x])} stocks)",
                    key="calendar_spread_watchlist",
                    help="Choose a watchlist to find calendar spread opportunities"
                )

                if selected_watchlist:
                    symbols = watchlists_db[selected_watchlist]

                    # Filter to stock symbols only (no crypto)
                    stock_symbols = [s for s in symbols if not any(x in s for x in ['USDT', 'USD', 'BTC', '.D', 'WETH'])]

                    if not stock_symbols:
                        st.warning(f"No stock symbols found in {selected_watchlist}")
                    else:
                        # Strategy parameters
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            max_symbols = st.number_input(
                                "Max Symbols to Analyze",
                                min_value=1,
                                max_value=50,
                                value=10,
                                help="More symbols = longer analysis time",
                                key="calendar_max_symbols"
                            )

                        with col2:
                            min_score = st.slider(
                                "Minimum AI Score",
                                min_value=0,
                                max_value=100,
                                value=60,
                                help="Only show spreads with score >= this value",
                                key="calendar_min_score"
                            )

                        with col3:
                            spread_type = st.selectbox(
                                "Spread Type",
                                ["Both", "Call Calendars Only", "Put Calendars Only"],
                                key="calendar_spread_type",
                                help="Filter by calendar spread type"
                            )

                        # Analyze button
                        if st.button("🔍 Find Calendar Spreads", type="primary", key="find_calendar_spreads_btn"):
                            st.markdown("### 🎯 Analysis Results")

                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            all_opportunities = []
                            symbols_to_analyze = stock_symbols[:max_symbols]

                            # PERFORMANCE: Parallel analysis with ThreadPoolExecutor (5-10x faster!)
                            from src.calendar_spread_analyzer import CalendarSpreadAnalyzer
                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            import robin_stocks.robinhood as rh

                            analyzer = CalendarSpreadAnalyzer()

                            def analyze_single_symbol(symbol):
                                """Analyze a single symbol (thread-safe)"""
                                try:
                                    quote = rh.get_latest_price(symbol)
                                    if not quote or not quote[0]:
                                        return []

                                    stock_price = float(quote[0])
                                    opportunities = analyzer.analyze_symbol(symbol, stock_price)

                                    # Filter by score and type
                                    filtered = []
                                    for opp in opportunities:
                                        if opp['score'] >= min_score:
                                            if spread_type == "Both":
                                                filtered.append(opp)
                                            elif spread_type == "Call Calendars Only" and "Call" in opp['type']:
                                                filtered.append(opp)
                                            elif spread_type == "Put Calendars Only" and "Put" in opp['type']:
                                                filtered.append(opp)

                                    return filtered

                                except Exception as e:
                                    st.warning(f"Error analyzing {symbol}: {e}")
                                    return []

                            # Parallel execution (5 workers for optimal throughput)
                            with ThreadPoolExecutor(max_workers=5) as executor:
                                # Submit all tasks
                                future_to_symbol = {
                                    executor.submit(analyze_single_symbol, symbol): symbol
                                    for symbol in symbols_to_analyze
                                }

                                # Process results as they complete
                                completed = 0
                                for future in as_completed(future_to_symbol):
                                    symbol = future_to_symbol[future]
                                    completed += 1
                                    status_text.text(f"Analyzed {symbol}... ({completed}/{len(symbols_to_analyze)})")
                                    progress_bar.progress(completed / len(symbols_to_analyze))

                                    try:
                                        opportunities = future.result()
                                        all_opportunities.extend(opportunities)
                                    except Exception as e:
                                        st.warning(f"Error processing {symbol}: {e}")

                            progress_bar.empty()
                            status_text.empty()

                            if not all_opportunities:
                                st.info("No calendar spread opportunities found matching your criteria. Try lowering the minimum score or analyzing more symbols.")
                            else:
                                # Sort by score
                                all_opportunities.sort(key=lambda x: x['score'], reverse=True)

                                # Cache in session state
                                st.session_state['calendar_opportunities'] = all_opportunities

                        # Display results if available
                        if st.session_state.get('calendar_opportunities'):
                            opportunities = st.session_state['calendar_opportunities']

                            if opportunities:
                                # Display summary metrics
                                st.markdown("### 📈 Summary")
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Opportunities Found", len(opportunities))
                                with col2:
                                    avg_score = sum(o['score'] for o in opportunities) / len(opportunities)
                                    st.metric("Avg AI Score", f"{avg_score:.1f}")
                                with col3:
                                    avg_profit_pot = sum(o['profit_potential'] for o in opportunities) / len(opportunities)
                                    st.metric("Avg Profit Potential", f"{avg_profit_pot:.0f}%")
                                with col4:
                                    avg_iv = sum(o['avg_iv'] for o in opportunities) / len(opportunities)
                                    st.metric("Avg IV", f"{avg_iv:.1f}%")

                                # Build DataFrame
                                import pandas as pd
                                df = pd.DataFrame(opportunities)

                                # Display main table
                                st.markdown("### 📋 Calendar Spread Opportunities")
                                st.caption("Click column headers to sort. Spreads are ranked by AI score.")

                                # Format for display
                                display_df = df[[
                                    'symbol', 'type', 'score', 'recommendation',
                                    'strike', 'stock_price',
                                    'short_dte', 'long_dte',
                                    'net_debit', 'max_loss', 'max_profit_estimate', 'profit_potential',
                                    'avg_iv', 'theta_differential'
                                ]].copy()

                                display_df.columns = [
                                    'Symbol', 'Type', 'AI Score', 'Recommendation',
                                    'Strike', 'Stock Price',
                                    'Short DTE', 'Long DTE',
                                    'Net Debit', 'Max Loss', 'Est Max Profit', 'Profit %',
                                    'Avg IV %', 'Theta Diff'
                                ]

                                # Format numbers
                                display_df['Strike'] = display_df['Strike'].apply(lambda x: f"${x:.2f}")
                                display_df['Stock Price'] = display_df['Stock Price'].apply(lambda x: f"${x:.2f}")
                                display_df['Net Debit'] = display_df['Net Debit'].apply(lambda x: f"${x:.0f}")
                                display_df['Max Loss'] = display_df['Max Loss'].apply(lambda x: f"${x:.0f}")
                                display_df['Est Max Profit'] = display_df['Est Max Profit'].apply(lambda x: f"${x:.0f}")
                                display_df['Profit %'] = display_df['Profit %'].apply(lambda x: f"{x:.0f}%")
                                display_df['Avg IV %'] = display_df['Avg IV %'].apply(lambda x: f"{x:.1f}%")
                                display_df['Theta Diff'] = display_df['Theta Diff'].apply(lambda x: f"{x:.4f}")

                                st.dataframe(
                                    display_df,
                                    hide_index=True,
                                    width='stretch',
                                    height=600
                                )

                                # Detailed view for top opportunities
                                st.markdown("### 🔍 Detailed Analysis - Top 5 Opportunities")

                                top_5 = opportunities[:5]

                                for opp in top_5:
                                    with st.expander(f"⭐ {opp['symbol']} - {opp['type']} | Score: {opp['score']} | Strike: ${opp['strike']:.2f}"):
                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.markdown("**📊 Spread Details**")
                                            st.write(f"**Symbol:** {opp['symbol']}")
                                            st.write(f"**Type:** {opp['type']}")
                                            st.write(f"**Strike:** ${opp['strike']:.2f}")
                                            st.write(f"**Stock Price:** ${opp['stock_price']:.2f}")
                                            st.write(f"**AI Score:** {opp['score']}/100")

                                        with col2:
                                            st.markdown("**📅 Expiration Details**")
                                            st.write(f"**Short Leg:** {opp['short_exp']} ({opp['short_dte']} DTE)")
                                            st.write(f"**Long Leg:** {opp['long_exp']} ({opp['long_dte']} DTE)")
                                            st.write(f"**Time Spread:** {opp['long_dte'] - opp['short_dte']} days")

                                        with col3:
                                            st.markdown("**💰 Cost & P/L**")
                                            st.write(f"**Short Premium:** ${opp['short_premium']:.2f}")
                                            st.write(f"**Long Premium:** ${opp['long_premium']:.2f}")
                                            st.write(f"**Net Debit:** ${opp['net_debit']:.2f}")
                                            st.write(f"**Max Loss:** ${opp['max_loss']:.2f}")
                                            st.write(f"**Est Max Profit:** ${opp['max_profit_estimate']:.2f}")
                                            st.write(f"**Profit Potential:** {opp['profit_potential']:.0f}%")

                                        # Greeks and metrics
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            st.markdown("**📐 Greeks & Metrics**")
                                            st.write(f"**Avg IV:** {opp['avg_iv']:.1f}%")
                                            st.write(f"**Theta Differential:** {opp['theta_differential']:.4f}")
                                            st.write(f"**Short Delta:** {opp['short_delta']:.3f}")

                                        with col2:
                                            st.markdown("**💧 Liquidity**")
                                            st.write(f"**Short Volume:** {opp['short_volume']}")
                                            st.write(f"**Long Volume:** {opp['long_volume']}")
                                            st.write(f"**Short OI:** {opp['short_oi']}")
                                            st.write(f"**Long OI:** {opp['long_oi']}")

                                        # Recommendation
                                        st.markdown(f"**✅ Recommendation:** {opp['recommendation']}")

                                        # Entry instructions
                                        st.markdown("**📝 How to Enter This Spread:**")
                                        st.code(f"""
1. SELL TO OPEN: {opp['symbol']} {opp['short_exp']} ${opp['strike']:.2f} {opp['type'].split()[0]}
   - Collect: ${opp['short_premium']:.2f}

2. BUY TO OPEN: {opp['symbol']} {opp['long_exp']} ${opp['strike']:.2f} {opp['type'].split()[0]}
   - Pay: ${opp['long_premium']:.2f}

Net Debit: ${opp['net_debit']:.2f}
                                        """, language="text")

                                # Download CSV
                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name=f"calendar_spreads_{selected_watchlist}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    key="calendar_download_csv"
                                )

elif page == "Risk Analysis":
    st.title("🛡️ Risk Analysis")
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_score = 45
        st.metric("Risk Score", f"{risk_score}/100")
        if risk_score < 40:
            st.success("Low Risk")
        elif risk_score < 70:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")
    
    with col2:
        st.metric("Portfolio Delta", "+125", "▲5")
    
    with col3:
        st.metric("Max Drawdown", "-12.5%", "")
    
    with col4:
        st.metric("VaR (95%)", "$2,450", "▼10%")
    
    
    # Sector allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sector Allocation")
        
        sectors = pd.DataFrame({
            'Sector': ['Technology', 'Finance', 'Consumer', 'Energy', 'Healthcare'],
            'Allocation': [35, 25, 20, 15, 5]
        })
        
        fig = px.bar(
            sectors,
            x='Allocation',
            y='Sector',
            orientation='h',
            color='Allocation',
            color_continuous_scale='RdYlGn_r'
        )
        
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("⚠️ Risk Alerts")
        
        alerts = [
            {'Level': 'Warning', 'Message': 'Technology sector approaching 40% limit'},
            {'Level': 'Info', 'Message': 'Consider diversifying into defensive sectors'},
            {'Level': 'Success', 'Message': 'Position sizing within limits'},
        ]
        
        for alert in alerts:
            if alert['Level'] == 'Warning':
                st.warning(alert['Message'])
            elif alert['Level'] == 'Info':
                st.info(alert['Message'])
            else:
                st.success(alert['Message'])
    
    # Recommendations
    st.subheader("💡 Risk Management Recommendations")
    
    recommendations = [
        "Consider reducing Technology exposure below 30%",
        "Add defensive positions in Consumer Staples or Utilities",
        "Current cash reserves (40%) provide good assignment coverage",
        "VaR is within acceptable range for portfolio size"
    ]
    
    for rec in recommendations:
        st.write(f"• {rec}")

elif page == "Premium Scanner":
    # Import and run the new Premium Scanner page
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("premium_scanner_page", "premium_scanner_page.py")
    premium_scanner_module = importlib.util.module_from_spec(spec)
    sys.modules["premium_scanner_page"] = premium_scanner_module
    spec.loader.exec_module(premium_scanner_module)

elif page == "Database Scan (Legacy)":
    st.title("🗄️ Database Stock Scanner (Legacy)")
    st.markdown("Scan PostgreSQL database for stocks and analyze option premiums")
    st.warning("⚠️ This page is deprecated. Please use 'Premium Scanner' instead.")

    from src.database_scanner import DatabaseScanner
    import pytz

    scanner = DatabaseScanner()

    # Auto-update DATABASE stocks (separate from TradingView)
    # Runs AFTER market opens to get fresh options data
    et_tz = pytz.timezone('America/New_York')
    current_time_et = datetime.now(et_tz)
    market_open_time = current_time_et.replace(hour=9, minute=30, second=0, microsecond=0)
    sync_start_time = current_time_et.replace(hour=10, minute=0, second=0, microsecond=0)  # 10 AM ET

    # Check if we need daily database sync
    last_db_sync_date = st.session_state.get('last_db_options_sync_date')
    today = current_time_et.date()

    # Auto-sync options data AFTER market opens (10 AM ET) if not done today
    if last_db_sync_date != today and current_time_et >= sync_start_time:
        st.info("🔄 Starting daily premium sync (after market open)...")
        st.info("⏱️ Syncing 30-day options premiums for ALL database stocks. Takes 3-5 minutes.")

        # Start background sync process
        import subprocess
        try:
            # Launch background sync script
            subprocess.Popen(
                ["python", "sync_database_stocks_daily.py"],
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            st.session_state['last_db_options_sync_date'] = today
            st.session_state['db_sync_start_time'] = datetime.now()
            st.success("✅ Daily database options sync started in background!")
        except Exception as e:
            st.warning(f"Could not start background sync: {e}")

    # Create tabs - NOW WITH PREMIUM SCANNER, CALENDAR SPREADS & AI RESEARCH
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Database Overview",
        "➕ Add Stocks",
        "🔍 Premium Scanner",  # NEW - Enhanced premium scanning
        "📆 Calendar Spreads",  # NEW - Calendar spread opportunities
        "🤖 AI Research",  # NEW - AI-powered analysis
        "📈 Analytics"
    ])

    with tab1:
        st.subheader("Database Overview")

        if scanner.connect():
            # Get actual sync status from database
            try:
                scanner.cursor.execute("""
                    SELECT 
                        MAX(last_updated) as last_sync,
                        COUNT(*) as total_stocks,
                        COUNT(CASE WHEN last_updated > NOW() - INTERVAL '24 hours' THEN 1 END) as synced_today,
                        COUNT(CASE WHEN last_updated > NOW() - INTERVAL '7 days' THEN 1 END) as synced_this_week,
                        MIN(last_updated) as oldest_sync
                    FROM stock_data
                """)
                sync_status = scanner.cursor.fetchone()
                
                if sync_status and sync_status['last_sync']:
                    last_sync = sync_status['last_sync']
                    if isinstance(last_sync, str):
                        from dateutil import parser
                        last_sync = parser.parse(last_sync)
                    
                    # Calculate freshness
                    hours_ago = (datetime.now() - last_sync.replace(tzinfo=None)).total_seconds() / 3600
                    
                    if hours_ago < 1:
                        status_color = "🟢"
                        status_text = f"Fresh ({int(hours_ago * 60)} minutes ago)"
                    elif hours_ago < 24:
                        status_color = "🟡"
                        status_text = f"Recent ({int(hours_ago)} hours ago)"
                    else:
                        status_color = "🔴"
                        days_ago = int(hours_ago / 24)
                        status_text = f"Stale ({days_ago} day{'s' if days_ago > 1 else ''} ago)"
                    
                    # Display sync status metrics
                    col_sync1, col_sync2, col_sync3, col_sync4 = st.columns(4)
                    with col_sync1:
                        st.metric("Last Sync Status", f"{status_color} {status_text}")
                    with col_sync2:
                        st.metric("Synced Today", f"{sync_status['synced_today']}/{sync_status['total_stocks']}")
                    with col_sync3:
                        st.metric("Synced This Week", f"{sync_status['synced_this_week']}/{sync_status['total_stocks']}")
                    with col_sync4:
                        if last_sync:
                            st.caption(f"Last sync: {last_sync.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                else:
                    st.warning("⚠️ No sync data found. Run sync to populate stock data.")
            except Exception as e:
                st.warning(f"⚠️ Could not fetch sync status: {e}")
                # Fallback to session state
                if 'last_db_update_time' in st.session_state:
                    time_since = (datetime.now() - st.session_state['last_db_update_time']).seconds // 60
                    if time_since < 60:
                        st.info(f"📅 Prices last updated: {time_since} minutes ago (from session)")
                    else:
                        hours_since = time_since // 60
                        st.info(f"📅 Prices last updated: {hours_since} hours ago (from session)")
                else:
                    st.warning("⚠️ Prices have not been updated today. Click 'Update All Prices' below or wait for pre-market auto-update.")
            try:
                # Create tables if needed
                scanner.create_tables()

                # Get all stocks
                stocks = scanner.get_all_stocks()

                if stocks and len(stocks) > 0:
                    st.success(f"📊 Found {len(stocks)} stocks in database")

                    # Show summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Stocks", len(stocks))

                    with col2:
                        # Count stocks by sector
                        df_temp = pd.DataFrame(stocks)
                        sectors = df_temp['sector'].nunique() if 'sector' in df_temp.columns else 0
                        st.metric("Unique Sectors", sectors)

                    with col3:
                        # Count stocks with prices
                        if 'current_price' in df_temp.columns:
                            stocks_with_prices = len(df_temp[df_temp['current_price'] > 0])
                            st.metric("With Prices", stocks_with_prices)
                        else:
                            st.metric("With Prices", "N/A")

                    with col4:
                        # Average price
                        if 'current_price' in df_temp.columns:
                            avg_price = df_temp[df_temp['current_price'] > 0]['current_price'].mean()
                            st.metric("Avg Price", f"${avg_price:.2f}" if avg_price > 0 else "N/A")
                        else:
                            st.metric("Avg Price", "N/A")


                    # Display stocks table
                    df = pd.DataFrame(stocks)
                    if not df.empty:
                        # Add last synced column with relative time formatting
                        def format_relative_time(dt):
                            """Format datetime as relative time"""
                            if not dt or pd.isna(dt):
                                return "Never"
                            
                            if isinstance(dt, str):
                                from dateutil import parser
                                dt = parser.parse(dt)
                            
                            now = datetime.now()
                            if hasattr(dt, 'tzinfo') and dt.tzinfo:
                                dt = dt.replace(tzinfo=None)
                            
                            diff = now - dt
                            hours = diff.total_seconds() / 3600
                            
                            if hours < 1:
                                minutes = int(diff.total_seconds() / 60)
                                return f"{minutes}m ago"
                            elif hours < 24:
                                return f"{int(hours)}h ago"
                            else:
                                days = int(hours / 24)
                                return f"{days}d ago"
                        
                        # Get last_updated from stock_data table for each stock
                        if 'symbol' in df.columns:
                            try:
                                # Join with stock_data to get actual last_updated
                                symbols_list = df['symbol'].tolist()
                                placeholders = ','.join(['%s'] * len(symbols_list))
                                scanner.cursor.execute(f"""
                                    SELECT symbol, last_updated
                                    FROM stock_data
                                    WHERE symbol IN ({placeholders})
                                """, symbols_list)
                                stock_data_times = {row['symbol']: row['last_updated'] for row in scanner.cursor.fetchall()}
                                
                                # Add last_synced column
                                df['last_synced'] = df['symbol'].apply(
                                    lambda s: format_relative_time(stock_data_times.get(s))
                                )
                            except Exception as e:
                                # Fallback to stocks table last_updated
                                if 'last_updated' in df.columns:
                                    df['last_synced'] = df['last_updated'].apply(format_relative_time)
                                else:
                                    df['last_synced'] = "Unknown"
                        
                        # Create a display dataframe with proper column selection
                        display_columns = []
                        if 'symbol' in df.columns:
                            display_columns.append('symbol')
                        if 'name' in df.columns:
                            display_columns.append('name')
                        if 'sector' in df.columns:
                            display_columns.append('sector')
                        if 'current_price' in df.columns:
                            display_columns.append('current_price')
                            # Format price column
                            df['current_price'] = df['current_price'].apply(
                                lambda x: f"${x:.2f}" if x and x > 0 else "$0.00"
                            )
                        if 'market_cap' in df.columns:
                            display_columns.append('market_cap')
                            # Format market cap
                            df['market_cap'] = df['market_cap'].apply(
                                lambda x: f"${x/1e9:.2f}B" if x and x > 1e9 else f"${x/1e6:.2f}M" if x and x > 0 else "N/A"
                            )
                        if 'avg_volume' in df.columns:
                            display_columns.append('avg_volume')
                            # Format volume
                            df['avg_volume'] = df['avg_volume'].apply(
                                lambda x: f"{x/1e6:.2f}M" if x and x > 0 else "N/A"
                            )
                        if 'last_synced' in df.columns:
                            display_columns.append('last_synced')

                        if display_columns:
                            st.dataframe(df[display_columns], width='stretch', height=400)
                        else:
                            st.warning("No displayable columns found in stocks data")
                            st.json(stocks[0] if stocks else {})

                    # Refresh view button
                    st.info("💡 Stock prices update automatically during daily premium sync at 10 AM ET. Manual sync available in Premium Scanner tab.")
                    if st.button("🔄 Refresh View", type="primary"):
                        st.rerun()
                else:
                    st.warning("No stocks in database yet")
                    st.info("👉 Go to 'Add Stocks' tab to add symbols to the database")

            except Exception as e:
                st.error(f"Error loading database overview: {str(e)}")
                st.info("This may be due to database schema mismatch. Please check the database configuration.")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())

            finally:
                scanner.disconnect()
        else:
            st.error("Failed to connect to database")
            st.info("Please check database configuration in .env file")

    with tab2:
        st.subheader("Add Stocks to Database")

        # Input for symbols
        symbols_input = st.text_area(
            "Enter stock symbols (comma or line separated):",
            placeholder="AAPL, MSFT, GOOGL\nTSLA, META",
            height=150
        )

        col1, col2 = st.columns(2)
        with col1:
            fetch_data = st.checkbox("Fetch stock data from Yahoo Finance", value=True)

        with col2:
            if st.button("➕ Add Stocks", type="primary"):
                if symbols_input:
                    # Parse symbols
                    symbols = [s.strip().upper() for s in symbols_input.replace('\n', ',').split(',') if s.strip()]

                    if scanner.connect():
                        scanner.create_tables()
                        with st.spinner(f"Adding {len(symbols)} stocks..."):
                            added = 0
                            for symbol in symbols:
                                if scanner.add_stock(symbol, fetch_data):
                                    added += 1

                        st.success(f"✅ Added {added} stocks to database")
                        scanner.disconnect()
                else:
                    st.error("Please enter at least one symbol")

    with tab3:
        st.subheader("🔍 Premium Scanner")
        st.caption("Advanced premium scanning for all database stocks - filter and sort to find best opportunities")

        # Show last update time and stock count info
        col_info1, col_info2 = st.columns([3, 1])
        with col_info1:
            # Get count of stocks with options data
            from src.tradingview_db_manager import TradingViewDBManager
            tv_manager_temp = TradingViewDBManager()
            conn_temp = tv_manager_temp.get_connection()
            cur_temp = conn_temp.cursor()
            cur_temp.execute("SELECT COUNT(*) FROM stocks")
            total_in_db = cur_temp.fetchone()[0]
            cur_temp.execute("SELECT COUNT(DISTINCT symbol) FROM stock_premiums")
            with_options = cur_temp.fetchone()[0]
            cur_temp.close()
            conn_temp.close()

            if 'last_db_options_sync_date' in st.session_state:
                st.info(f"💡 Showing {with_options} of {total_in_db} database stocks. 30-day premiums sync daily at 10 AM ET.")
            else:
                st.warning(f"⚠️ Showing {with_options} of {total_in_db} database stocks. Daily premium sync runs at 10 AM ET. Click 'Sync Now' for immediate sync.")
        with col_info2:
            if 'last_db_update_time' in st.session_state:
                time_since = (datetime.now() - st.session_state['last_db_update_time']).seconds // 60
                if time_since < 60:
                    st.success(f"📅 Updated {time_since}m ago")
                else:
                    hours_since = time_since // 60
                    st.warning(f"📅 Updated {hours_since}h ago")
            else:
                st.warning("⚠️ Not updated today")

        # Manual sync button
        col_sync1, col_sync2 = st.columns([1, 4])
        with col_sync1:
            if st.button("🔄 Sync Now", type="primary", help="Manually trigger options data sync for all database stocks", key="manual_db_sync"):
                st.info("⏱️ Starting background sync for all database stocks...")
                import subprocess
                try:
                    subprocess.Popen(
                        ["python", "sync_database_stocks_daily.py"],
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    )
                    st.session_state['last_db_options_sync_date'] = datetime.now(pytz.timezone('America/New_York')).date()
                    st.session_state['db_sync_start_time'] = datetime.now()
                    st.session_state['show_sync_progress'] = True
                    st.success("✅ Sync started in background! Monitor progress below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error starting sync: {e}")
        with col_sync2:
            if 'db_sync_start_time' in st.session_state:
                minutes_ago = (datetime.now() - st.session_state['db_sync_start_time']).seconds // 60
                st.caption(f"Last sync started: {minutes_ago} minutes ago")

        # Show real-time progress if sync is running
        if st.session_state.get('show_sync_progress', False):
            import json
            import os

            progress_file = 'database_sync_progress.json'

            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)

                    # Check if sync is still active (updated within last 10 seconds)
                    from datetime import datetime
                    last_update = datetime.fromisoformat(progress['last_updated'])
                    seconds_since_update = (datetime.now() - last_update).total_seconds()

                    if seconds_since_update < 30:  # Still active
                        st.markdown("### 🔄 Sync Progress")

                        # Progress bar
                        progress_bar = st.progress(progress['percent'] / 100)

                        # Stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Progress", f"{progress['current']}/{progress['total']}")
                        with col2:
                            st.metric("Complete", f"{progress['percent']:.1f}%")
                        with col3:
                            elapsed_min = progress['elapsed_seconds'] // 60
                            elapsed_sec = progress['elapsed_seconds'] % 60
                            st.metric("Elapsed", f"{elapsed_min}m {elapsed_sec}s")
                        with col4:
                            remaining_min = progress['remaining_seconds'] // 60
                            remaining_sec = progress['remaining_seconds'] % 60
                            st.metric("Remaining", f"{remaining_min}m {remaining_sec}s")

                        st.caption(f"Current: {progress['current_symbol']} | Rate: {progress['rate_per_second']} stocks/sec")

                        # Auto-refresh button
                        if st.button("🔄 Refresh Progress", key="refresh_progress"):
                            st.rerun()

                        st.info("💡 Page will auto-refresh progress. Click 'Refresh Progress' for latest status.")

                    else:
                        # Sync completed or stopped
                        if progress.get('completed', False):
                            st.markdown("### ✅ Sync Complete!")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Stocks", progress['total'])
                            with col2:
                                st.metric("Successfully Synced", progress.get('stats', {}).get('successful', 0))
                            with col3:
                                st.metric("Failed/No Options", progress.get('stats', {}).get('failed', 0))

                            duration_min = progress['elapsed_seconds'] // 60
                            duration_sec = progress['elapsed_seconds'] % 60
                            st.success(f"✅ Completed in {duration_min}m {duration_sec}s. Refresh page to see {progress.get('stats', {}).get('successful', 0)} stocks with fresh options data!")

                            if st.button("🔄 Refresh Dashboard", key="refresh_after_sync"):
                                st.session_state['show_sync_progress'] = False
                                st.rerun()
                        else:
                            st.info("⏳ Sync may have stopped. Check database_sync.log for details.")
                            st.session_state['show_sync_progress'] = False

                except Exception as e:
                    st.warning(f"Could not read progress: {e}")
            else:
                st.info("⏳ Waiting for sync to start... Click 'Refresh Progress' below to check status.")
                # Manual refresh button instead of automatic rerun
                if st.button("🔄 Refresh Progress", key="refresh_progress_waiting"):
                    st.rerun()


        # Filters (30-day options only)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_stock_price = st.number_input("Min Stock Price ($)", value=0.0, min_value=0.0, step=10.0, key="db_min_stock")
        with col_f2:
            max_stock_price = st.number_input("Max Stock Price ($)", value=10000.0, min_value=10.0, step=50.0, key="db_max_stock")
        with col_f3:
            min_premium = st.number_input("Min Premium ($)", value=0.0, min_value=0.0, step=1.0, key="db_min_prem")

        # Fixed at 30-day options
        dte_filter = 30

        # Query stock_premiums table for ALL stocks with options
        from src.tradingview_db_manager import TradingViewDBManager
        tv_manager = TradingViewDBManager()

        conn = tv_manager.get_connection()
        cur = conn.cursor()

        query = """
            SELECT DISTINCT ON (sp.symbol)
                sp.symbol,
                sd.current_price as stock_price,
                sp.strike_price,
                sp.dte,
                sp.premium,
                sp.delta,
                sp.monthly_return,
                sp.implied_volatility as iv,
                sp.bid,
                sp.ask,
                sp.volume,
                sp.open_interest as oi,
                s.company_name,
                s.sector
            FROM stock_premiums sp
            LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
            LEFT JOIN stocks s ON sp.symbol = s.symbol
            WHERE sp.dte BETWEEN %s AND %s
                AND ABS(sp.delta) BETWEEN 0.25 AND 0.40
                AND sp.premium >= %s
                AND (sd.current_price BETWEEN %s AND %s OR sd.current_price IS NULL)
                AND sp.strike_price < sd.current_price
            ORDER BY sp.symbol, sp.monthly_return DESC
        """

        cur.execute(query, (dte_filter - 2, dte_filter + 2, min_premium, min_stock_price, max_stock_price))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if rows:
            # Create DataFrame
            df = pd.DataFrame(rows, columns=[
                'Symbol', 'Stock Price', 'Strike', 'DTE',
                'Premium', 'Delta', 'Monthly %', 'IV',
                'Bid', 'Ask', 'Volume', 'OI', 'Name', 'Sector'
            ])

            # Convert to numeric
            numeric_cols = ['Stock Price', 'Strike', 'DTE', 'Premium', 'Delta',
                           'Monthly %', 'IV', 'Bid', 'Ask', 'Volume', 'OI']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Options Found", len(df))
            with col2:
                st.metric("Avg Premium", f"${df['Premium'].mean():.2f}")
            with col3:
                st.metric("Avg Monthly Return", f"{df['Monthly %'].mean():.2f}%")
            with col4:
                st.metric("Unique Stocks", df['Symbol'].nunique())

            # Display sortable table
            st.markdown("#### 📊 All Stocks with 30-Day Options (Click headers to sort)")
            st.dataframe(
                df,
                hide_index=True,
                width='stretch',
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Stock Price": st.column_config.NumberColumn("Stock $", format="$%.2f"),
                    "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                    "DTE": st.column_config.NumberColumn("DTE"),
                    "Premium": st.column_config.NumberColumn("Premium", format="$%.2f"),
                    "Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                    "Monthly %": st.column_config.NumberColumn("Monthly %", format="%.2f%%"),
                    "IV": st.column_config.NumberColumn("IV %", format="%.1f%%"),
                    "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                    "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                    "Volume": st.column_config.NumberColumn("Vol"),
                    "OI": st.column_config.NumberColumn("OI"),
                    "Name": st.column_config.TextColumn("Name", width="medium"),
                    "Sector": st.column_config.TextColumn("Sector", width="medium")
                }
            )
        else:
            st.warning("No options found for 30-day expiration with current filters")
            st.info("Run the premium sync to populate database with 30-day options for all stocks.")

        # AI Research Integration for Premium Scanner results
        if rows:
            from src.components.ai_research_widget import display_consolidated_ai_research_section
            symbols = df['Symbol'].unique().tolist()[:20]  # Limit to top 20 for performance
            display_consolidated_ai_research_section(symbols, key_prefix="db_premium")

    with tab4:
        st.subheader("📆 Calendar Spread Opportunities")
        st.caption("Find calendar spreads for stocks in your database")

        # Info banner
        with st.expander("ℹ️ What are Calendar Spreads?"):
            st.markdown("""
            **Calendar Spreads** (also called time spreads or horizontal spreads) involve:
            - **Selling** a near-term option (30-45 days)
            - **Buying** a longer-term option (60-90 days)
            - **Same strike price** and option type

            ### How You Profit:
            - **Time Decay**: Short option decays faster than long option
            - **Theta Differential**: Earn the difference in decay rates
            - **Best When**: Stock stays near strike price at short expiration

            ### Risk Profile:
            - **Max Loss**: Net premium paid (limited risk)
            - **Max Profit**: Depends on long option value when short expires
            - **Ideal Conditions**: Low IV (<30%), range-bound markets

            ### Our AI Scoring (0-100):
            - Theta differential (30%)
            - IV level (25%)
            - Moneyness/ATM proximity (20%)
            - Timing (15%)
            - Liquidity (10%)
            """)

        # Login to Robinhood if needed
        if not st.session_state.get('rh_calendar_logged_in'):
            with st.spinner("Connecting to Robinhood..."):
                try:
                    import robin_stocks.robinhood as rh
                    # SECURITY FIX: Use environment variables instead of hardcoded credentials
                    import os
                    rh_username = os.getenv('ROBINHOOD_USERNAME')
                    rh_password = os.getenv('ROBINHOOD_PASSWORD')
                    if not rh_username or not rh_password:
                        raise ValueError("ROBINHOOD_USERNAME or ROBINHOOD_PASSWORD not found in environment")
                    rh.login(rh_username, rh_password)
                    st.session_state['rh_calendar_logged_in'] = True
                except Exception as e:
                    st.error(f"Failed to connect to Robinhood: {e}")
                    st.info("Calendar Spreads require Robinhood connection for options data.")

        if st.session_state.get('rh_calendar_logged_in'):
            # Get stocks from database
            from src.tradingview_db_manager import TradingViewDBManager
            tv_manager = TradingViewDBManager()
            conn = tv_manager.get_connection()
            cur = conn.cursor()

            # Get distinct symbols with options data
            cur.execute("""
                SELECT DISTINCT symbol
                FROM stock_premiums
                WHERE premium > 0
                ORDER BY symbol
                LIMIT 100
            """)
            stock_symbols = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()

            if not stock_symbols:
                st.warning("No stocks with options data found in database. Please sync data first.")
            else:
                st.info(f"Found {len(stock_symbols)} stocks with options data in database")

                # Configuration
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_symbols = st.number_input(
                        "Max Symbols to Analyze",
                        min_value=5,
                        max_value=100,
                        value=20,
                        help="Limit analysis to avoid timeouts",
                        key="db_calendar_max_symbols"
                    )

                with col2:
                    min_score = st.slider(
                        "Minimum AI Score",
                        min_value=0,
                        max_value=100,
                        value=60,
                        help="Only show spreads with score >= this value",
                        key="db_calendar_min_score"
                    )

                with col3:
                    spread_type = st.selectbox(
                        "Spread Type",
                        ["Both", "Call Calendars Only", "Put Calendars Only"],
                        key="db_calendar_spread_type"
                    )

                # Analyze button
                if st.button("🔍 Find Calendar Spreads", type="primary", key="db_find_calendar_spreads_btn"):
                    st.markdown("### 🎯 Analysis Results")

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    all_opportunities = []
                    symbols_to_analyze = stock_symbols[:max_symbols]

                    # Import analyzer
                    from src.calendar_spread_analyzer import CalendarSpreadAnalyzer
                    analyzer = CalendarSpreadAnalyzer()

                    for idx, symbol in enumerate(symbols_to_analyze):
                        status_text.text(f"Analyzing {symbol}... ({idx + 1}/{len(symbols_to_analyze)})")
                        progress_bar.progress((idx + 1) / len(symbols_to_analyze))

                        try:
                            # Get stock price
                            import robin_stocks.robinhood as rh
                            quote = rh.get_latest_price(symbol)
                            if not quote or not quote[0]:
                                continue

                            stock_price = float(quote[0])

                            # Analyze for calendar spreads
                            opportunities = analyzer.analyze_symbol(symbol, stock_price)

                            # Filter by score and type
                            for opp in opportunities:
                                if opp['score'] >= min_score:
                                    if spread_type == "Both":
                                        all_opportunities.append(opp)
                                    elif spread_type == "Call Calendars Only" and "Call" in opp['type']:
                                        all_opportunities.append(opp)
                                    elif spread_type == "Put Calendars Only" and "Put" in opp['type']:
                                        all_opportunities.append(opp)

                        except Exception as e:
                            st.warning(f"Error analyzing {symbol}: {e}")
                            continue

                    progress_bar.empty()
                    status_text.empty()

                    if not all_opportunities:
                        st.info("No calendar spread opportunities found matching your criteria. Try lowering the minimum score or analyzing more symbols.")
                    else:
                        # Sort by score
                        all_opportunities.sort(key=lambda x: x['score'], reverse=True)

                        # Cache in session state
                        st.session_state['db_calendar_opportunities'] = all_opportunities

                # Display results if available
                if st.session_state.get('db_calendar_opportunities'):
                    opportunities = st.session_state['db_calendar_opportunities']

                    if opportunities:
                        # Display summary metrics
                        st.markdown("### 📈 Summary")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Opportunities Found", len(opportunities))
                        with col2:
                            avg_score = sum(o['score'] for o in opportunities) / len(opportunities)
                            st.metric("Avg AI Score", f"{avg_score:.1f}")
                        with col3:
                            avg_profit_pot = sum(o['profit_potential'] for o in opportunities) / len(opportunities)
                            st.metric("Avg Profit Potential", f"{avg_profit_pot:.0f}%")
                        with col4:
                            avg_iv = sum(o['avg_iv'] for o in opportunities) / len(opportunities)
                            st.metric("Avg IV", f"{avg_iv:.1f}%")

                        # Build DataFrame
                        import pandas as pd
                        df_cal = pd.DataFrame(opportunities)

                        # Display main table
                        st.markdown("### 📋 Calendar Spread Opportunities")
                        st.caption("Click column headers to sort. Spreads are ranked by AI score.")

                        # Format for display
                        display_df = df_cal[[
                            'symbol', 'type', 'score', 'recommendation',
                            'strike', 'stock_price',
                            'short_dte', 'long_dte',
                            'net_debit', 'max_loss', 'max_profit_estimate', 'profit_potential',
                            'avg_iv', 'theta_differential'
                        ]].copy()

                        display_df.columns = [
                            'Symbol', 'Type', 'AI Score', 'Recommendation',
                            'Strike', 'Stock Price',
                            'Short DTE', 'Long DTE',
                            'Net Debit', 'Max Loss', 'Est Max Profit', 'Profit %',
                            'Avg IV %', 'Theta Diff'
                        ]

                        # Format numbers
                        display_df['Strike'] = display_df['Strike'].apply(lambda x: f"${x:.2f}")
                        display_df['Stock Price'] = display_df['Stock Price'].apply(lambda x: f"${x:.2f}")
                        display_df['Net Debit'] = display_df['Net Debit'].apply(lambda x: f"${x:.0f}")
                        display_df['Max Loss'] = display_df['Max Loss'].apply(lambda x: f"${x:.0f}")
                        display_df['Est Max Profit'] = display_df['Est Max Profit'].apply(lambda x: f"${x:.0f}")
                        display_df['Profit %'] = display_df['Profit %'].apply(lambda x: f"{x:.0f}%")
                        display_df['Avg IV %'] = display_df['Avg IV %'].apply(lambda x: f"{x:.1f}%")
                        display_df['Theta Diff'] = display_df['Theta Diff'].apply(lambda x: f"{x:.4f}")

                        st.dataframe(
                            display_df,
                            hide_index=True,
                            width='stretch',
                            height=600
                        )

                        # AI Research for top calendar spread opportunities
                        from src.components.ai_research_widget import display_consolidated_ai_research_section
                        top_symbols = [opp['symbol'] for opp in opportunities[:10]]
                        display_consolidated_ai_research_section(top_symbols, key_prefix="db_calendar")

    with tab5:
        st.subheader("🤖 AI Research for Database Stocks")
        st.caption("Get AI-powered analysis for any stock in your database")

        # Get stocks from database
        from src.tradingview_db_manager import TradingViewDBManager
        tv_manager = TradingViewDBManager()
        conn = tv_manager.get_connection()
        cur = conn.cursor()

        # Get all distinct symbols
        cur.execute("SELECT DISTINCT symbol FROM stocks ORDER BY symbol LIMIT 500")
        all_symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()

        if not all_symbols:
            st.warning("No stocks found in database. Please add stocks first.")
        else:
            st.info(f"📊 {len(all_symbols)} stocks available for AI analysis")

            # Multi-select for symbols
            selected_symbols = st.multiselect(
                "Select symbols for AI analysis:",
                options=all_symbols,
                default=[],
                key="db_ai_symbols",
                help="Select up to 20 symbols for detailed AI research"
            )

            if selected_symbols:
                if len(selected_symbols) > 20:
                    st.warning("⚠️ Please select 20 or fewer symbols for optimal performance")
                else:
                    from src.components.ai_research_widget import display_consolidated_ai_research_section
                    display_consolidated_ai_research_section(selected_symbols, key_prefix="db_ai")

    with tab6:
        st.subheader("📈 Database Analytics")

        if scanner.connect():
            stocks = scanner.get_all_stocks()

            if stocks:
                df = pd.DataFrame(stocks)

                col1, col2 = st.columns(2)

                with col1:
                    # Sector distribution
                    st.subheader("Stocks by Sector")
                    sector_counts = df['sector'].value_counts()
                    if len(sector_counts) > 0 and not sector_counts.empty:
                        # Use plotly to avoid Vega-Lite warnings
                        import plotly.express as px
                        fig = px.bar(x=sector_counts.index, y=sector_counts.values, labels={'x': 'Sector', 'y': 'Count'})
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("No sector data available")

                with col2:
                    # Price distribution
                    st.subheader("Price Distribution")
                    if 'current_price' in df.columns and len(df) > 0:
                        price_ranges = pd.cut(df['current_price'],
                                            bins=[0, 10, 25, 50, 100, 500, float('inf')],
                                            labels=['<$10', '$10-25', '$25-50', '$50-100', '$100-500', '>$500'])
                        price_dist = price_ranges.value_counts().sort_index()
                        if len(price_dist) > 0 and not price_dist.empty:
                            # Use plotly to avoid Vega-Lite warnings
                            import plotly.express as px
                            fig = px.bar(x=price_dist.index.astype(str), y=price_dist.values, labels={'x': 'Price Range', 'y': 'Count'})
                            fig.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.info("No price data available")
                    else:
                        st.info("No price data available")

                # Top movers would go here if we track price changes
                st.subheader("Database Stats")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Stocks", len(stocks))

                with col2:
                    avg_price = df['current_price'].mean()
                    st.metric("Avg Price", f"${avg_price:.2f}")

                with col3:
                    sectors = df['sector'].nunique()
                    st.metric("Sectors", sectors)

                with col4:
                    under_50 = len(df[df['current_price'] <= 50])
                    st.metric("Under $50", under_50)

            scanner.disconnect()
        else:
            st.error("Failed to connect to database")

elif page == "Earnings Calendar":
    from earnings_calendar_page import show_earnings_calendar
    show_earnings_calendar()

# Calendar Spreads moved to TradingView Watchlists page
# elif page == "Calendar Spreads":
#     from calendar_spreads_page import show_calendar_spreads
#     show_calendar_spreads()

elif page == "XTrade Messages":
    from discord_messages_page import main as show_xtrade_messages
    show_xtrade_messages()

# elif page == "AI Sports Predictions":
#     from prediction_markets_enhanced import show_prediction_markets_enhanced
#     show_prediction_markets_enhanced()
# NOTE: This page requires prediction_markets_enhanced.py to be created

elif page == "Prediction Markets":
    from prediction_markets_page import show_prediction_markets
    show_prediction_markets()

elif page == "Sports Game Hub":
    from game_cards_visual_page import show_game_cards
    show_game_cards()

elif page == "Technical Indicators":
    from supply_demand_zones_page import show_supply_demand_zones
    show_supply_demand_zones()

elif page == "Options Analysis":
    # Unified Options Analysis page (combines AI Options Agent + Comprehensive Strategy)
    from options_analysis_page import render_options_analysis_page
    render_options_analysis_page()

elif page == "AVA Chatbot":
    from ava_chatbot_page import show_ava_chatbot_page
    show_ava_chatbot_page()

elif page == "Agent Management":
    try:
        # Import and run agent management page
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from agent_management_page import main
        main()
    except Exception as e:
        st.error(f"❌ Error loading Agent Management page: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        st.info("Please ensure agent_management_page.py exists and all dependencies are installed")

elif page == "Settings":
    st.title("⚙️ Settings")
    
    # Strategy parameters
    st.subheader("🎯 Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Min DTE (Days to Expiration)", value=21, min_value=7, max_value=60)
        st.number_input("Max DTE", value=45, min_value=7, max_value=90)
        st.slider("Target Delta", 0.1, 0.5, 0.3, 0.05)
    
    with col2:
        st.number_input("Max Position Size (%)", value=5.0, min_value=1.0, max_value=20.0)
        st.number_input("Max Sector Exposure (%)", value=30.0, min_value=10.0, max_value=50.0)
        st.slider("Min Options Volume", 10, 1000, 100, 10)
    
    
    # Alert settings
    st.subheader("🔔 Alert Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Email Alerts", value=False)
        st.checkbox("Discord Alerts", value=True)
        st.checkbox("SMS Alerts", value=False)
    
    with col2:
        st.selectbox("Alert Frequency", ["Real-time", "Every 5 min", "Every 15 min", "Hourly"])
        st.multiselect(
            "Alert Types",
            ["Price Movements", "Assignment Warnings", "Risk Alerts"],
            default=["Assignment Warnings", "Risk Alerts"]
        )
    

    # TradingView Integration
    st.subheader("📈 TradingView Integration")

    col1, col2 = st.columns(2)

    with col1:
        tv_enabled = st.checkbox("Enable TradingView Integration", value=True)
        if tv_enabled:
            tv_username = st.text_input(
                "TradingView Username",
                value=os.getenv('TRADINGVIEW_USERNAME', ''),
                help="Your TradingView account username"
            )

    with col2:
        if tv_enabled:
            st.info("💡 TradingView Features:")
            st.markdown("""
            - Import watchlists
            - Sync with your lists
            - Real-time screeners
            - Pre-built wheel lists
            """)

    if tv_enabled:
        st.markdown("**Available Watchlists:**")
        from src.tradingview_integration import TradingViewClient
        tv_client = TradingViewClient()

        # Show available lists
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🔥 **High IV Stocks**")
            st.caption("Best for premium collection")
        with col2:
            st.markdown("🎯 **Popular Wheel**")
            st.caption("Most traded wheel stocks")
        with col3:
            st.markdown("💵 **Dividend Stocks**")
            st.caption("Income + premiums")


    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")

elif page == "Sector Analysis":
    from sector_analysis_page_enhanced import display_sector_analysis_enhanced
    display_sector_analysis_enhanced()

elif page == "Enhancement Agent":
    from enhancement_agent_page import show_enhancement_agent
    show_enhancement_agent()

elif page == "Enhancement Manager":
    from enhancement_manager_page import render_enhancement_manager_page
    render_enhancement_manager_page()

elif page == "Cache Metrics":
    from cache_metrics_page import show_cache_metrics
    show_cache_metrics()

# Footer
st.markdown(
    "<div style='text-align: center; color: #888;'>Magnus Trading Platform v1.0 | "
    "Data updates every 5 minutes | Last update: " +
    datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</div>",
    unsafe_allow_html=True
)

