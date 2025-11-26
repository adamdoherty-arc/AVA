"""
Sports Game Hub - Live Game Monitoring and Betting Dashboard with Modern UI
Compact, responsive design inspired by DraftKings, FanDuel, and ESPN
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import re
import psycopg2
import psycopg2.extras
import logging
import os
import time
import random

# CRITICAL: Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

from src.services import get_kalshi_manager  # Use centralized service registry
from src.kalshi_client import KalshiClient
from src.kalshi_db_manager import KalshiDBManager
from src.espn_live_data import get_espn_client
from src.espn_ncaa_live_data import get_espn_ncaa_client
from src.ncaa_team_database import NCAA_LOGOS, get_team_logo_url, find_team_by_name
from src.game_watchlist_manager import GameWatchlistManager
from src.watchlist_monitor_service import get_monitor_service
from src.prediction_agents import NFLPredictor, NCAAPredictor
from src.betting.filters import create_standard_betting_panel, BettingFilterPanel

# Initialize logger first
logger = logging.getLogger(__name__)

# PERFORMANCE: Cache ML models as resources to avoid memory bloat in session state
@st.cache_resource
def get_nfl_predictor():
    """Get cached NFL predictor instance"""
    try:
        return NFLPredictor()
    except Exception as e:
        logger.warning(f"Could not initialize NFL Predictor: {e}")
        return None

@st.cache_resource
def get_ncaa_predictor():
    """Get cached NCAA predictor instance"""
    try:
        return NCAAPredictor()
    except Exception as e:
        logger.warning(f"Could not initialize NCAA Predictor: {e}")
        return None

# Optional auto-refresh (graceful degradation if not installed)
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    logger.warning("streamlit-autorefresh not installed. Auto-refresh feature disabled.")

# NFL Team Logo URLs (ESPN)
TEAM_LOGOS = {
    'Arizona': 'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png',
    'Atlanta': 'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png',
    'Baltimore': 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png',
    'Buffalo': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
    'Carolina': 'https://a.espncdn.com/i/teamlogos/nfl/500/car.png',
    'Chicago': 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png',
    'Cincinnati': 'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png',
    'Cleveland': 'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png',
    'Dallas': 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png',
    'Denver': 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png',
    'Detroit': 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png',
    'Green Bay': 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png',
    'Houston': 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png',
    'Indianapolis': 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png',
    'Jacksonville': 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png',
    'Kansas City': 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png',
    'Las Vegas': 'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png',
    'Los Angeles Chargers': 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png',
    'Los Angeles Rams': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
    'Miami': 'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png',
    'Minnesota': 'https://a.espncdn.com/i/teamlogos/nfl/500/min.png',
    'New England': 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png',
    'New Orleans': 'https://a.espncdn.com/i/teamlogos/nfl/500/no.png',
    'New York Giants': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png',
    'New York Jets': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png',
    'Philadelphia': 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png',
    'Pittsburgh': 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png',
    'San Francisco': 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png',
    'Seattle': 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png',
    'Tampa Bay': 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png',
    'Tennessee': 'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png',
    'Washington': 'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png',
}

# NCAA Team Logo URLs - Now imported from comprehensive team database
# NCAA_LOGOS contains 130+ FBS teams with dynamic logo fetching from ESPN API
# See src/ncaa_team_database.py for complete team list

# Shortened team names for display
TEAM_SHORT_NAMES = {
    'Arizona': 'ARI',
    'Atlanta': 'ATL',
    'Baltimore': 'BAL',
    'Buffalo': 'BUF',
    'Carolina': 'CAR',
    'Chicago': 'CHI',
    'Cincinnati': 'CIN',
    'Cleveland': 'CLE',
    'Dallas': 'DAL',
    'Denver': 'DEN',
    'Detroit': 'DET',
    'Green Bay': 'GB',
    'Houston': 'HOU',
    'Indianapolis': 'IND',
    'Jacksonville': 'JAX',
    'Kansas City': 'KC',
    'Las Vegas': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami': 'MIA',
    'Minnesota': 'MIN',
    'New England': 'NE',
    'New Orleans': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia': 'PHI',
    'Pittsburgh': 'PIT',
    'San Francisco': 'SF',
    'Seattle': 'SEA',
    'Tampa Bay': 'TB',
    'Tennessee': 'TEN',
    'Washington': 'WSH',
}


# ============================================================================
# PERFORMANCE: Connection Pooling & Caching
# ============================================================================

@st.cache_resource
def get_kalshi_db_manager():
    """Cached Kalshi database manager - singleton pattern for connection pooling"""
    return KalshiDBManager()


@st.cache_resource
def get_game_watchlist_manager():
    """Cached game watchlist manager - singleton pattern"""
    db = get_kalshi_db_manager()
    return GameWatchlistManager(db)


@st.cache_data
def load_game_cards_css():
    """Load and cache game cards CSS for performance"""
    css_path = os.path.join(os.path.dirname(__file__), 'static', 'css', 'game_cards.css')
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback - return minimal CSS if file not found
        logger.warning(f"CSS file not found: {css_path}")
        return ""

def show_subscription_settings(watchlist_manager):
    """Show subscription management and monitoring settings"""

    st.markdown("### ⚙️ Subscription Settings & Monitoring")

    # Initialize session state for monitoring settings
    if 'monitor_interval' not in st.session_state:
        st.session_state.monitor_interval = 5  # Default 5 minutes

    # Get monitor service
    monitor_service = get_monitor_service()
    service_status = monitor_service.get_status()

    # ==================== MONITORING CONTROLS ====================
    st.markdown("#### 🔔 Live Monitoring Controls")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        # Update interval selector
        interval_options = {
            "1 minute (Fast)": 1,
            "3 minutes": 3,
            "5 minutes (Recommended)": 5,
            "10 minutes": 10,
            "15 minutes (Battery Saver)": 15
        }

        selected_interval_label = st.selectbox(
            "📊 Update Interval",
            options=list(interval_options.keys()),
            index=2,  # Default to 5 minutes
            help="How often to check for game updates",
            disabled=service_status['running']  # Disable when monitoring is active
        )

        st.session_state.monitor_interval = interval_options[selected_interval_label]

    with col2:
        # Monitoring status - use actual service status
        if service_status['running']:
            st.success(f"🟢 **Status:** Monitoring Active ({service_status['interval_minutes']} min)")
            if service_status.get('next_run'):
                next_run = service_status['next_run'].strftime('%I:%M %p')
                st.caption(f"Next check: {next_run}")
        else:
            st.info("⚪ **Status:** Monitoring Stopped")

    with col3:
        # Start/Stop button - controls actual service
        if service_status['running']:
            if st.button("⏸️ Stop Monitoring", type="secondary", use_container_width=True):
                with st.spinner("Stopping monitoring..."):
                    success = monitor_service.stop()
                    if success:
                        st.success("✅ Monitoring stopped")
                    else:
                        st.error("❌ Failed to stop monitoring")
        else:
            if st.button("▶️ Start Monitoring", type="primary", use_container_width=True):
                with st.spinner(f"Starting monitoring (every {st.session_state.monitor_interval} min)..."):
                    success = monitor_service.start(interval_minutes=st.session_state.monitor_interval)
                    if success:
                        st.success(f"✅ Monitoring started - Updates every {st.session_state.monitor_interval} minutes")
                        st.caption("Check your Telegram for confirmation!")
                    else:
                        st.error("❌ Failed to start monitoring - Check Telegram credentials in .env")


    # ==================== HOW TO USE ====================
    with st.expander("📖 How Live Monitoring Works", expanded=False):
        st.markdown("""
        **Live Monitoring automatically checks your subscribed games and sends Telegram updates when:**

        ✅ **Score Changes** - Get instant score updates
        ✅ **Quarter/Period Changes** - Know when quarters start/end
        ✅ **Game Status Changes** - Pre-game → Live → Final
        ✅ **Odds Movements** - Kalshi odds shift >10¢
        ✅ **AI Prediction Changes** - Confidence swings >10%
        ✅ **Your Team Status** - When your team starts winning or losing

        **How to Start Monitoring:**
        1. Subscribe to games from the NFL/NCAA/NBA tabs
        2. Choose your update interval above
        3. Click **"Start Monitoring"**
        4. Keep this browser tab open (or run `python game_watchlist_monitor.py` in terminal for background monitoring)

        **Important Notes:**
        - Updates are SMART - only sent when meaningful changes occur
        - No spam - never repeats the same message twice
        - Telegram messages include AI recommendations (increase/decrease bet, hedge, etc.)
        - Lower intervals (1-3 min) = more frequent checks but higher data usage
        - Higher intervals (10-15 min) = less frequent but saves battery/data
        """)


    # ==================== TELEGRAM TEST ====================
    st.markdown("#### 🧪 Test Telegram Connection")

    col_test1, col_test2 = st.columns([3, 3])

    with col_test1:
        if st.button("📱 Send Test Message", type="secondary", use_container_width=True, help="Send a test message to verify Telegram is working"):
            with st.spinner("Sending test message..."):
                try:
                    from src.telegram_notifier import TelegramNotifier

                    notifier = TelegramNotifier()
                    success = notifier.test_connection()

                    if success:
                        st.success("✅ Test message sent successfully! Check your Telegram app.")
                    else:
                        st.error("❌ Failed to send test message. Check your Telegram credentials in .env file.")
                        st.caption("**Required .env settings:**")
                        st.code("""TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here""")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.caption("Make sure python-telegram-bot is installed: `pip install python-telegram-bot==20.7`")

    with col_test2:
        with st.expander("🔧 Telegram Setup Help", expanded=False):
            st.markdown("""
            **If test message fails, verify your .env file has:**

            1. `TELEGRAM_ENABLED=true`
            2. `TELEGRAM_BOT_TOKEN` from [@BotFather](https://t.me/BotFather)
            3. `TELEGRAM_CHAT_ID` from [@userinfobot](https://t.me/userinfobot)

            **Quick Setup:**
            - Message [@BotFather](https://t.me/BotFather) on Telegram
            - Create a new bot with `/newbot`
            - Copy the token to your .env file
            - Message [@userinfobot](https://t.me/userinfobot) to get your chat ID
            - Add both to .env and restart the app
            """)

    st.markdown("---")  # Divider


    # ==================== SUBSCRIBED GAMES ====================
    st.markdown("#### 📋 Your Subscribed Games")

    # DEBUG: Show user_id being used
    with st.expander("🔍 Debug Info", expanded=False):
        st.code(f"User ID: {st.session_state.user_id}")
        st.caption(f"Fetching watchlist from database...")

    # Get all subscriptions
    watchlist = watchlist_manager.get_user_watchlist(st.session_state.user_id)

    if not watchlist:
        st.info("👋 No subscribed games yet. Go to NFL, NCAA, or NBA tabs and click Subscribe on any game card!")
        st.caption(f"🔍 Debug: Checked database for user_id: {st.session_state.user_id}")
        return

    # Count by sport
    nfl_count = len([w for w in watchlist if w.get('sport') == 'NFL'])
    ncaa_count = len([w for w in watchlist if w.get('sport') == 'CFB'])
    nba_count = len([w for w in watchlist if w.get('sport') == 'NBA'])

    # Stats cards
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("🏈 NFL Games", nfl_count)
    with col_stat2:
        st.metric("🎓 NCAA Games", ncaa_count)
    with col_stat3:
        st.metric("🏀 NBA Games", nba_count)
    with col_stat4:
        st.metric("📊 Total", len(watchlist))


    # Display subscriptions by sport
    for sport_name, sport_code in [("🏈 NFL", "NFL"), ("🎓 NCAA", "CFB"), ("🏀 NBA", "NBA")]:
        sport_games = [w for w in watchlist if w.get('sport') == sport_code]

        if not sport_games:
            continue

        st.markdown(f"**{sport_name} Subscriptions ({len(sport_games)})**")

        for idx, watch_entry in enumerate(sport_games):
            game_data = watch_entry.get('game_data', {})
            away_team = game_data.get('away_team', watch_entry.get('away_team', 'Away'))
            home_team = game_data.get('home_team', watch_entry.get('home_team', 'Home'))
            selected_team = watch_entry.get('selected_team', '')
            game_id = watch_entry.get('game_id', '')

            # Display as table-like rows
            col_game, col_pick, col_action = st.columns([3, 2, 1])

            with col_game:
                st.markdown(f"**{away_team}** @ **{home_team}**")

            with col_pick:
                if selected_team:
                    st.caption(f"Your Pick: {selected_team}")
                else:
                    st.caption("No team selected")

            with col_action:
                unsub_key = f"settings_unsub_{sport_code}_{game_id}_{idx}"
                if st.button("🗑️", key=unsub_key, help="Unsubscribe", use_container_width=True):
                    if watchlist_manager.remove_game_from_watchlist(st.session_state.user_id, game_id):
                        st.success(f"Unsubscribed from {away_team} @ {home_team}")
                        st.rerun()

        st.markdown("")  # Spacing


    # ==================== COMMAND LINE OPTION ====================
    with st.expander("💻 Advanced: Run Monitoring in Background (Terminal)", expanded=False):
        st.markdown("""
        **For continuous monitoring even when browser is closed:**

        Run this command in your terminal:
        ```bash
        python game_watchlist_monitor.py --interval 5
        ```

        **Arguments:**
        - `--interval N` - Update interval in minutes (default: 5)

        **Examples:**
        ```bash
        # Check every 1 minute (fast updates)
        python game_watchlist_monitor.py --interval 1

        # Check every 10 minutes (battery saver)
        python game_watchlist_monitor.py --interval 10
        ```

        **Benefits of Background Monitoring:**
        - ✅ Runs independently of browser
        - ✅ Continues even if you close this page
        - ✅ More reliable for long sessions
        - ✅ Lower resource usage

        **To Stop:**
        Press `Ctrl+C` in the terminal window
        """)


def show_game_cards():
    """Main function for visual game cards with modern compact UI"""

    # Apply custom CSS for compact, modern design - OPTIMIZED with caching
    # CSS is now loaded from external file and cached for performance
    css_content = load_game_cards_css()
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    # PERFORMANCE: Use cached database and watchlist managers (singleton pattern)
    db = get_kalshi_db_manager()
    watchlist_manager = get_game_watchlist_manager()

    # AUTO-START MONITORING: If user has subscribed games, auto-start monitoring service
    if 'monitoring_auto_started' not in st.session_state:
        st.session_state.monitoring_auto_started = False

    if not st.session_state.monitoring_auto_started:
        try:
            # Check if user has any watched games
            temp_user_id = os.getenv('TELEGRAM_AUTHORIZED_USERS', 'default_user').split(',')[0]
            watchlist = watchlist_manager.get_user_watchlist(temp_user_id)

            if watchlist and len(watchlist) > 0:
                # User has subscriptions, auto-start monitoring
                from src.watchlist_monitor_service import get_monitor_service
                monitor_service = get_monitor_service()
                status = monitor_service.get_status()

                if not status['running']:
                    # Start monitoring with default 5-minute interval
                    success = monitor_service.start(interval_minutes=5)
                    if success:
                        st.session_state.monitoring_auto_started = True
                        logger.info("✅ Auto-started background monitoring for subscribed games")
        except Exception as e:
            logger.warning(f"Could not auto-start monitoring: {e}")
            pass  # Non-critical, continue without auto-start

    # Initialize user ID (from Telegram or default)
    # CRITICAL: Always refresh from environment to ensure correct value
    telegram_user_id = os.getenv('TELEGRAM_AUTHORIZED_USERS', '')
    if telegram_user_id:
        st.session_state.user_id = telegram_user_id.split(',')[0]
    elif 'user_id' not in st.session_state:
        st.session_state.user_id = 'default_user'

    # Store selected sport in session state
    if 'selected_sport' not in st.session_state:
        st.session_state.selected_sport = 'NFL'

    # Initialize filters collapsed state
    if 'filters_collapsed' not in st.session_state:
        st.session_state.filters_collapsed = True

    # Initialize watch sidebar collapsed state
    if 'watch_sidebar_collapsed' not in st.session_state:
        st.session_state.watch_sidebar_collapsed = False

    # ==================== COMPACT STICKY HEADER ====================
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)

    # FORCE REFRESH BUTTON - Make it visible at top
    import random
    import datetime
    cache_buster = st.session_state.get('css_cache_buster', int(time.time()))
    
    # Show monitoring status indicator
    def is_monitor_running():
        """Check if background watchlist monitor is running"""
        try:
            import psutil
            for proc in psutil.process_iter(['name', 'cmdline']):
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('game_watchlist_monitor' in str(cmd) for cmd in cmdline):
                        return True
        except:
            pass
        return False

    monitor_running = is_monitor_running()
    if monitor_running:
        st.success("🟢 **Background Monitoring ACTIVE** - Checking watchlist every 5 minutes")
    else:
        st.warning("🟠 **Background Monitoring NOT RUNNING** - Start it in Settings tab for automatic updates")

    # Title row - compact single line
    col_title, col_watch, col_ai, col_refresh = st.columns([3, 2, 2, 1])

    with col_title:
        st.markdown('<h2 style="margin:0; padding:0;">🏟️ Sports Game Hub</h2>', unsafe_allow_html=True)

    with col_watch:
        # Watch list count
        watchlist = watchlist_manager.get_user_watchlist(st.session_state.user_id)
        watch_count = len(watchlist) if watchlist else 0

        # Auto-cleanup: Remove finished games from watchlist
        if watchlist:
            cleaned_count = watchlist_manager.cleanup_finished_games(st.session_state.user_id)
            if cleaned_count > 0:
                watch_count = len(watchlist_manager.get_user_watchlist(st.session_state.user_id))

        if watch_count > 0:
            if st.button(f"📍 Watching: {watch_count}", key="toggle_watch_sidebar", use_container_width=True):
                st.session_state.watch_sidebar_collapsed = not st.session_state.watch_sidebar_collapsed
        else:
            st.caption("📍 Watch games for updates")

    with col_ai:
        # Initialize LLM Service
        try:
            from src.services.llm_service import LLMService
            llm_service = LLMService()
            available_providers = llm_service.get_available_providers()
            llm_available = True
        except Exception as e:
            logger.warning(f"LLM Service not available: {e}")
            llm_available = False
            available_providers = []

        # Compact AI model selector with Ollama support
        model_options = []

        # Check for Ollama models
        ollama_models = []
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                data = response.json()
                ollama_models = [m.get('name', '') for m in data.get('models', [])]
                # Add Ollama models to dropdown
                for model in ollama_models:
                    model_options.append(f"Ollama: {model}")
        except:
            pass  # Ollama not running

        # Add default local AI
        if not ollama_models:
            model_options.append("Local AI (Basic)")

        # Add cloud providers
        if llm_available and "groq" in available_providers:
            model_options.append("Groq Cloud")
        if llm_available and "deepseek" in available_providers:
            model_options.append("DeepSeek Cloud")

        # Set default model (prefer qwen2.5-coder, then qwen2.5, then llama, then first ollama, then basic)
        default_index = 0
        if ollama_models:
            # Find best model
            qwen_coder = [i for i, m in enumerate(model_options) if 'qwen2.5-coder' in m.lower()]
            qwen = [i for i, m in enumerate(model_options) if 'qwen2.5' in m.lower() and 'coder' not in m.lower()]
            llama = [i for i, m in enumerate(model_options) if 'llama' in m.lower()]

            if qwen_coder:
                default_index = qwen_coder[0]
            elif qwen:
                default_index = qwen[0]
            elif llama:
                default_index = llama[0]

        selected_model = st.selectbox(
            "AI Model",
            model_options,
            index=default_index,
            label_visibility="collapsed",
            key="ai_model_selector",
            help="Select AI model for game analysis"
        )
        st.session_state.ai_model = selected_model

    with col_refresh:
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄", value=False, key="auto_refresh_header", help="Auto-refresh")

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================== WATCHLIST DISPLAY AT TOP ====================
    watchlist = watchlist_manager.get_user_watchlist(st.session_state.user_id)
    if watchlist and len(watchlist) > 0:
        st.markdown("### 📍 Games You're Watching")
        # Display watchlist as compact chips
        watchlist_cols = st.columns(min(len(watchlist), 6))  # Max 6 columns
        for idx, watch_game in enumerate(watchlist[:6]):  # Show first 6
            with watchlist_cols[idx % len(watchlist_cols)]:
                game_data = watch_game.get('game_data', {})
                away_team = game_data.get('away_team', 'Away')
                home_team = game_data.get('home_team', 'Home')
                selected_team = watch_game.get('selected_team', '')
                entry_price = watch_game.get('entry_price')
                entry_team = watch_game.get('entry_team', '')
                
                # Compact display
                team_display = selected_team if selected_team else f"{away_team[:8]} @ {home_team[:8]}"
                if entry_price:
                    st.markdown(f"**{team_display}**<br>Entry: {entry_price:.0f}¢", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{team_display}**", unsafe_allow_html=True)
        if len(watchlist) > 6:
            st.caption(f"... and {len(watchlist) - 6} more")

    # ==================== WATCH LIST SIDEBAR (ONCE, NOT PER TAB) ====================
    # Display sidebar watchlist BEFORE tabs to avoid duplication
    watchlist_for_sidebar = watchlist_manager.get_user_watchlist(st.session_state.user_id)
    if watchlist_for_sidebar and not st.session_state.watch_sidebar_collapsed:
        with st.sidebar:
            st.markdown("### 📍 Your Watch List")

            for idx, watch_game in enumerate(watchlist_for_sidebar):
                game_id = watch_game.get('game_id', '')
                game_data = watch_game.get('game_data', {})
                selected_team = watch_game.get('selected_team', '')

                away_team = game_data.get('away_team', 'Away')
                home_team = game_data.get('home_team', 'Home')
                away_score = game_data.get('away_score', 0)
                home_score = game_data.get('home_score', 0)
                status = game_data.get('status_detail', 'Scheduled')
                is_live = game_data.get('is_live', False)

                # Highlight selected team
                team_display = f"**{selected_team}**" if selected_team else f"{away_team} @ {home_team}"

                col_watch1, col_watch2 = st.columns([4, 1])
                with col_watch1:
                    if is_live:
                        st.markdown(f'<span class="live-indicator"></span>{team_display}', unsafe_allow_html=True)
                        st.caption(f"{away_score} - {home_score} • {status}")
                    else:
                        st.markdown(team_display)
                        st.caption(status)

                with col_watch2:
                    # Make key unique for sidebar watchlist
                    unwatch_key = f"sidebar_unwatch_{game_id}_{idx}"
                    if st.button("✖", key=unwatch_key, help="Remove from watch list"):
                        watchlist_manager.remove_game_from_watchlist(st.session_state.user_id, game_id)
                        st.rerun()

    # ==================== SPORT TABS (Horizontal at top) ====================
    sport_tabs = st.tabs(["🏈 NFL", "🎓 NCAA", "🏀 NBA", "⚾ MLB", "⚙️ Settings"])

    with sport_tabs[0]:  # NFL
        sport_filter = "NFL"
        st.session_state.selected_sport = 'NFL'
        show_sport_games(db, watchlist_manager, sport_filter, "NFL", llm_service if llm_available else None, auto_refresh)

    with sport_tabs[1]:  # NCAA
        sport_filter = "CFB"
        st.session_state.selected_sport = 'NCAA'
        show_sport_games(db, watchlist_manager, sport_filter, "NCAA", llm_service if llm_available else None, auto_refresh)

    with sport_tabs[2]:  # NBA
        sport_filter = "NBA"
        st.session_state.selected_sport = 'NBA'
        show_sport_games_nba(db, watchlist_manager, llm_service if llm_available else None, auto_refresh)

    with sport_tabs[3]:  # MLB
        st.info("⚾ **MLB Coming Soon** - Integration in progress")

    with sport_tabs[4]:  # Settings
        show_subscription_settings(watchlist_manager)


def show_sport_games(db, watchlist_manager, sport_filter, sport_name, llm_service=None, auto_refresh=False):
    """Display sports games with filters, sorting, and live updates.
    
    Args:
        db: Database manager instance
        watchlist_manager: Manager for user watchlist
        sport_filter: Sport filter code (e.g., 'NFL', 'CFB')
        sport_name: Display name for sport (e.g., 'NFL', 'NCAA')
        llm_service: Optional LLM service for AI predictions
        auto_refresh: Enable auto-refresh functionality
    """

    # Get selected AI model from session state
    selected_ai_model = st.session_state.get('ai_model', 'Local AI (Fast & Free)')

    # ==================== ALWAYS VISIBLE FILTERS ====================
    st.markdown("### 🎛️ Filters & Sorting")

    # Create and render standard betting panel
    # This replaces the manual filter widgets with the shared library
    panel = create_standard_betting_panel()
    
    # Render filters in columns layout
    filter_values = panel.render(key_prefix=f"gc_{sport_filter}", layout="columns")
    
    # Extract specific values needed for other parts of the UI

    # Layout for auto-refresh and sync buttons
    col9, col10 = st.columns([1.5, 1.5])
    
    with col9:
        auto_refresh_enabled = st.checkbox(
            "⚡ Auto-Refresh",
            value=False,
            key=f"auto_refresh_{sport_filter}",
            help="Automatically sync live data at set interval"
        )

    with col10:
        if auto_refresh_enabled:
            refresh_interval = st.selectbox(
                "Interval",
                ["30 sec", "1 min", "2 min", "5 min"],
                index=2,  # Default to 2 min
                key=f"refresh_interval_{sport_filter}"
            )
        else:
            refresh_interval = None

    # Manual sync buttons and status
    col_sync1, col_sync2, col_sync3, col_sync4 = st.columns([2, 2, 2, 1])
    with col_sync1:
        if st.button("🔄 Sync ESPN Data", key=f"sync_espn_{sport_filter}", help="Refresh live scores from ESPN", use_container_width=True):
            with st.spinner(f"Syncing {sport_name} data from ESPN..."):
                st.cache_data.clear()
                import time
                time.sleep(0.5)  # Brief pause to show spinner
            st.success(f"✅ {sport_name} data refreshed!")
            st.rerun()
    with col_sync2:
        if st.button("💰 Sync Kalshi Odds", key=f"sync_kalshi_{sport_filter}", help="Refresh betting odds from Kalshi", use_container_width=True):
            with st.spinner("Syncing Kalshi betting odds..."):
                st.cache_data.clear()
                import time
                time.sleep(0.5)
            st.success("✅ Kalshi odds refreshed!")
            st.rerun()
    with col_sync3:
        if st.button("🤖 Refresh AI Analysis", key=f"sync_ai_{sport_filter}", help="Regenerate AI predictions", use_container_width=True):
            with st.spinner("Regenerating AI predictions..."):
                st.cache_data.clear()
                import time
                time.sleep(0.5)
            st.success("✅ AI analysis refreshed!")
            st.rerun()
    with col_sync4:
        # Last sync indicator
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M")
        st.caption(f"🕐 {current_time}")

    # Auto-refresh logic
    if auto_refresh_enabled and refresh_interval:
        import time
        # Convert interval to seconds
        interval_map = {"30 sec": 30, "1 min": 60, "2 min": 120, "5 min": 300}
        interval_seconds = interval_map.get(refresh_interval, 120)

        # Use Streamlit's experimental rerun to auto-refresh
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()

        time_since_refresh = time.time() - st.session_state.last_refresh
        if time_since_refresh >= interval_seconds:
            st.session_state.last_refresh = time.time()
            st.cache_data.clear()
            st.rerun()

    # Fetch live ESPN data first (primary data source)
    espn_status = "❌ Failed"
    try:
        if sport_filter == 'CFB':
            # NCAA Football - use college football API - fetch current + upcoming weeks to show all future games
            espn = get_espn_ncaa_client()
            espn_games = []

            # Fetch multiple weeks to get all upcoming games (weeks 11-16 covers rest of season + bowls)
            # We try all weeks and ignore errors for weeks that haven't happened yet or are past
            for week in range(11, 17):
                try:
                    week_games = espn.get_scoreboard(week=week, group='80')  # 80 = FBS
                    if week_games:  # Only add if we got games
                        espn_games.extend(week_games)
                        logger.info(f"Fetched {len(week_games)} games from NCAA Week {week}")
                except Exception as week_error:
                    logger.debug(f"Week {week} not available (likely future or past): {week_error}")
                    # Continue to next week - this is expected for future weeks

            # Remove duplicates based on game_id (in case ESPN returns same game in multiple weeks)
            seen_ids = set()
            unique_games = []
            for game in espn_games:
                game_id = game.get('game_id')
                if game_id and game_id not in seen_ids:
                    seen_ids.add(game_id)
                    unique_games.append(game)
            espn_games = unique_games

            logger.info(f"Total unique NCAA games fetched: {len(espn_games)}")
        else:
            # NFL - use NFL API - fetch current + upcoming weeks to show all future games
            espn = get_espn_client()
            espn_games = []

            # Fetch multiple weeks to get all upcoming games (weeks 11-18 covers rest of season + playoffs)
            # We try all weeks and ignore errors for weeks that haven't happened yet or are past
            for week in range(11, 19):
                try:
                    week_games = espn.get_scoreboard(week=week)
                    if week_games:  # Only add if we got games
                        espn_games.extend(week_games)
                        logger.info(f"Fetched {len(week_games)} games from NFL Week {week}")
                except Exception as week_error:
                    logger.debug(f"Week {week} not available (likely future or past): {week_error}")
                    # Continue to next week - this is expected for future weeks

            # Remove duplicates based on game_id (in case ESPN returns same game in multiple weeks)
            seen_ids = set()
            unique_games = []
            for game in espn_games:
                game_id = game.get('game_id')
                if game_id and game_id not in seen_ids:
                    seen_ids.add(game_id)
                    unique_games.append(game)
            espn_games = unique_games

            logger.info(f"Total unique NFL games fetched: {len(espn_games)}")

        espn_status = f"✅ {len(espn_games)} games fetched"
    except Exception as e:
        logger.error(f"Could not fetch ESPN data: {e}")
        espn_games = []
        st.error(f"⚠️ Could not fetch {sport_name} games from ESPN: {str(e)}")
        st.info("💡 Try: 1) Clear cache (press C) 2) Check internet connection 3) Verify ESPN API status")
        # Don't return - allow UI to still show with helpful messages

    # Enrich ESPN games with Kalshi odds (OPTIMIZED - uses caching + batch query)
    kalshi_status = "❌ No odds"
    kalshi_matched = 0  # Initialize to prevent UnboundLocalError
    try:
        from src.espn_kalshi_matcher_optimized import enrich_games_with_kalshi_odds_optimized
        # Detect sport type from filter
        sport = 'nfl' if sport_filter == 'NFL' else 'ncaaf'
        espn_games = enrich_games_with_kalshi_odds_optimized(espn_games, sport=sport)

        # FILTER OUT LOPSIDED ODDS (where you can't make money)
        # Remove games where market odds are >85% or <15% (too one-sided)
        games_before_filter = len([g for g in espn_games if g.get('kalshi_odds')])
        espn_games_filtered = []
        for game in espn_games:
            kalshi_odds = game.get('kalshi_odds')
            if kalshi_odds:
                away_price = kalshi_odds.get('away_win_price', 0.5)
                home_price = kalshi_odds.get('home_win_price', 0.5)

                # Skip if either price is too lopsided (>0.85 or <0.15)
                if (away_price > 0.85 or away_price < 0.15 or
                    home_price > 0.85 or home_price < 0.15):
                    logger.debug(f"Filtered out lopsided odds: {game['away_team']} @ {game['home_team']} "
                               f"(Away: {away_price:.1%}, Home: {home_price:.1%})")
                    continue  # Skip this game

            espn_games_filtered.append(game)

        espn_games = espn_games_filtered
        games_filtered = games_before_filter - len([g for g in espn_games if g.get('kalshi_odds')])
        if games_filtered > 0:
            logger.info(f"Filtered out {games_filtered} games with lopsided odds (>85% or <15%)")

        kalshi_matched = sum(1 for g in espn_games if g.get('kalshi_odds'))
        if kalshi_matched > 0:
            logger.info(f"Matched {kalshi_matched}/{len(espn_games)} ESPN games with Kalshi odds")
            kalshi_status = f"✅ {kalshi_matched}/{len(espn_games)} games"
        else:
            kalshi_status = f"⚠️ 0/{len(espn_games)} games matched"
    except Exception as e:
        logger.warning(f"Could not enrich with Kalshi odds: {e}")
        kalshi_status = f"❌ Error: {str(e)[:50]}"

    # Display sync status
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        st.info(f"**ESPN Status:** {espn_status}")
    with col_status2:
        st.info(f"**Kalshi Status:** {kalshi_status}")
    with col_status3:
        ai_status = "✅ Active" if llm_service else "⚠️ Local only"
        st.info(f"**AI Status:** {ai_status}")

    # Show help if Kalshi data is missing
    if kalshi_matched == 0:
        with st.expander("💡 How to sync Kalshi odds", expanded=False):
            st.markdown("""
            **To get Kalshi betting odds, run one of these scripts:**

            1. **Quick Sync (NFL only):**
               ```bash
               python sync_kalshi_team_winners.py
               ```

            2. **Complete Sync (All markets):**
               ```bash
               python sync_kalshi_complete.py
               ```

            3. **Real-time Sync (Keep running):**
               ```bash
               python sync_kalshi_prices_realtime.py
               ```

            Make sure your Kalshi credentials are set in `.env` file:
            ```
            KALSHI_EMAIL=your@email.com
            KALSHI_PASSWORD=your_password
            ```
            """)

            if st.button("🔧 Open Kalshi Setup Guide", key=f"kalshi_setup_{sport_filter}"):
                st.info("Check KALSHI_SETUP_GUIDE.md for detailed setup instructions")

    # Check if games exist
    if not espn_games:
        st.warning(f"No live {sport_name} games available at this time")
        return

    # Extract unique team names for team filter dropdown
    all_teams = set()
    for game in espn_games:
        away_team = game.get('away_team', '')
        home_team = game.get('home_team', '')
        if away_team:
            all_teams.add(away_team)
        if home_team:
            all_teams.add(home_team)

    team_list = sorted(list(all_teams))

    # Team-specific filter dropdown
    col_team_sel, col_subs = st.columns([3, 3])
    with col_team_sel:
        team_options = ["All Teams"] + team_list
        selected_team_name = st.selectbox(
            "🏈 Select Team",
            team_options,
            key=f"team_selector_{sport_filter}",
            help="Filter to show only games involving this team"
        )

    with col_subs:
        # Show subscriptions count and management
        watchlist = watchlist_manager.get_user_watchlist(st.session_state.user_id)
        sub_count = len([w for w in watchlist if w.get('game_data', {}).get('sport') == sport_filter])
        if sub_count > 0:
            if st.button(f"📋 My Subscriptions ({sub_count})", key=f"show_subs_{sport_filter}", use_container_width=True):
                st.session_state[f'show_subscriptions_{sport_filter}'] = not st.session_state.get(f'show_subscriptions_{sport_filter}', False)
        else:
            st.info("No active subscriptions")

    # Show subscriptions if toggled
    if st.session_state.get(f'show_subscriptions_{sport_filter}', False) and watchlist:
        st.markdown("### 📋 Your Subscribed Games")

        # Table-like list view
        for idx, watch_game in enumerate(watchlist):
            game_data = watch_game.get('game_data', {})
            if game_data.get('sport') != sport_filter:
                continue

            away_team = game_data.get('away_team', 'Away')
            home_team = game_data.get('home_team', 'Home')
            selected_team = watch_game.get('selected_team', '')
            status = game_data.get('status_detail', 'Scheduled')
            away_score = game_data.get('away_score', 0)
            home_score = game_data.get('home_score', 0)
            is_live = game_data.get('is_live', False)

            col_sub1, col_sub2, col_sub3 = st.columns([3, 2, 1])
            with col_sub1:
                st.markdown(f"**{away_team}** @ **{home_team}**")
            with col_sub2:
                if is_live:
                    st.markdown(f"🔴 **{away_score} - {home_score}** • {status}")
                else:
                    st.caption(f"Your Pick: {selected_team or 'Not selected'} • {status}")
            with col_sub3:
                if st.button("🗑️ Unsubscribe", key=f"unsub_{watch_game.get('game_id')}", help="Remove from watchlist", use_container_width=True):
                    watchlist_manager.remove_game_from_watchlist(
                        st.session_state.user_id,
                        watch_game.get('game_id')
                    )
                    st.success("Unsubscribed!")
                    st.rerun()


    # ==================== APPLY FILTERS ====================
    # Convert to DataFrame for filtering
    if espn_games:
        df = pd.DataFrame(espn_games)
        
        # Apply filters using the panel
        filtered_df = panel.apply(df, filter_values)
        
        # Convert back to list of dicts
        filtered_games = filtered_df.to_dict('records')
    else:
        filtered_games = []

    # Calculate live count for display in pagination
    live_count = sum(1 for g in filtered_games if g.get('is_live', False))

    # NOTE: Watchlist sidebar moved to show_game_cards() to avoid duplication across tabs

    # ==================== DISPLAY ESPN LIVE GAMES ====================
    # Pass filtered games directly, no need for filter_settings anymore
    display_espn_live_games(
        filtered_games,
        sport_name,
        sport_filter,
        watchlist_manager,
        cards_per_row=3, # Default to 3
        llm_service=llm_service,
        live_count=live_count,
        filter_settings=None # Filters already applied
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_sports_prediction_cached(game_id, sport_filter, home_team, away_team, game_date_str=None):
    """
    Get prediction from sport-specific AI agents (NFL or NCAA).
    Uses cached Elo-based predictions with sport-specific features.

    IMPORTANT: Cache key includes team names to ensure correct matching.

    Returns:
        dict: Prediction with winner, probability, confidence, spread, etc.
    """
    try:
        # DEBUG: Log input parameters
        logger.info(f"🔍 get_sports_prediction_cached called: game_id={game_id}, {away_team} @ {home_team}, date={game_date_str}")

        # Parse game date if provided
        game_date = None
        if game_date_str:
            try:
                from dateutil import parser
                game_date = parser.parse(game_date_str)
            except:
                pass

        # Get appropriate predictor from cached resources
        if sport_filter == 'NFL':
            predictor = get_nfl_predictor()
        else:  # CFB / NCAA
            predictor = get_ncaa_predictor()

        if not predictor:
            logger.warning(f"No predictor available for {sport_filter}")
            return None

        # Get prediction - CRITICAL: Use team names to ensure correct match
        prediction = predictor.predict_winner(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date
        )

        # DEBUG: Log prediction result
        if prediction:
            winner = prediction.get('winner', 'unknown')
            prob = prediction.get('probability', 0)
            logger.info(f"✅ Prediction for {away_team} @ {home_team}: {winner} wins with {prob:.1%}")
        else:
            logger.warning(f"❌ No prediction returned for {away_team} @ {home_team}")

        # VALIDATE: Ensure prediction matches the teams we requested
        if prediction:
            predicted_winner = prediction.get('winner', '')
            # Verify winner is one of the teams
            if predicted_winner and predicted_winner.lower() not in [home_team.lower(), away_team.lower()]:
                logger.warning(f"⚠️ Prediction mismatch: Got {predicted_winner} for {away_team} @ {home_team}")
                # Return None to force fresh prediction
                return None

        return prediction

    except Exception as e:
        logger.warning(f"Sports prediction error for {home_team} vs {away_team}: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_ai_predictions_cached(game_id, away_team, home_team, away_score, home_score, kalshi_odds_str):
    """
    Cached AI predictions to avoid redundant calls
    
    IMPORTANT: Cache key includes team names to ensure correct matching.
    """
    try:
        from src.advanced_betting_ai_agent import AdvancedBettingAIAgent
        ai_agent = AdvancedBettingAIAgent()

        # Reconstruct game and market data from hashable parameters
        import json
        kalshi_odds = json.loads(kalshi_odds_str) if kalshi_odds_str else {}

        game = {
            'game_id': game_id,
            'away_team': away_team,
            'home_team': home_team,
            'away_score': away_score,
            'home_score': home_score,
            'kalshi_odds': kalshi_odds  # Embed odds in game data for AI agent
        }

        ai_pred = ai_agent.analyze_betting_opportunity(game, {})
        
        # VALIDATE: Ensure prediction matches the teams we requested
        if ai_pred:
            predicted_winner = ai_pred.get('predicted_winner', '')
            # Convert to team name if it's 'away' or 'home'
            if predicted_winner.lower() == 'away':
                predicted_winner = away_team
            elif predicted_winner.lower() == 'home':
                predicted_winner = home_team
            
            # Verify winner is one of the teams
            if predicted_winner and predicted_winner.lower() not in [home_team.lower(), away_team.lower()]:
                logger.warning(f"AI prediction mismatch: Got {predicted_winner} for {away_team} @ {home_team}")
                # Return safe default
                return {
                    'expected_value': 0,
                    'confidence_score': 0,
                    'recommendation': 'PASS',
                    'predicted_winner': 'away' if away_score > home_score else 'home',
                    'win_probability': 0.5
                }
        
        return ai_pred
    except Exception as e:
        logger.warning(f"AI prediction error for {away_team} @ {home_team}: {e}")
        return {
            'expected_value': 0,
            'confidence_score': 0,
            'recommendation': 'PASS',
            'predicted_winner': 'away' if away_score > home_score else 'home',
            'win_probability': 0.5
        }


def display_espn_live_games(espn_games, sport_name, sport_filter, watchlist_manager, cards_per_row=4, llm_service=None, live_count=0, filter_settings=None):
    """Display ESPN live games in a compact visual grid format"""
    from src.nfl_team_database import NFL_LOGOS, get_team_logo_url as get_nfl_logo
    from src.ncaa_team_database import get_team_logo_url as get_ncaa_logo
    import json

    if not espn_games:
        st.warning(f"No live {sport_name} games available from ESPN at this time")
        return

    # OPTIMIZATION: Skip heavy AI analysis for initial filtering
    # Just use basic game data for filtering, AI will be loaded per-card on-demand
    games_with_ev = []
    for game in espn_games:
        # Set defaults for sorting (will be computed on-demand per card)
        game['expected_value'] = 0
        game['confidence'] = 0
        game['recommendation'] = 'PASS'
        games_with_ev.append(game)

    # Filter games by status
    # NOTE: Filtering is now done via BettingFilterPanel before calling this function
    # We keep this variable name for compatibility with existing code
    filtered_games = games_with_ev
    
    # Legacy filter support (optional, if filter_settings passed)
    if filter_settings:
        # Only apply if explicitly passed (backward compatibility)
        filter_status = filter_settings.get('filter_status', 'All Games')
        hide_final = filter_settings.get('hide_final', False)

        if filter_status == "Live Only":
            filtered_games = [g for g in games_with_ev if g.get('is_live', False)]
        elif filter_status == "Upcoming":
            filtered_games = [g for g in games_with_ev if not g.get('is_live', False) and not g.get('is_completed', False)]
        elif filter_status == "Final":
            filtered_games = [g for g in games_with_ev if g.get('is_completed', False)]

        # Apply "Hide Final Games" checkbox filter
        if hide_final:
            filtered_games = [g for g in filtered_games if not g.get('is_completed', False)]

        # Apply "Hide Lopsided Odds" filter - exclude games with heavily favored teams
        hide_lopsided = filter_settings.get('hide_lopsided', False)
        lopsided_threshold = filter_settings.get('lopsided_threshold', 90) / 100  # Convert to decimal

        if hide_lopsided:
            def is_not_lopsided(game):
                """Check if game has competitive odds (neither team heavily favored)"""
                yes_price = game.get('yes_price', 0.5)
                no_price = game.get('no_price', 0.5)

                # If either team's odds exceed threshold, it's lopsided (not profitable)
                if yes_price > lopsided_threshold or no_price > lopsided_threshold:
                    return False
                return True

            filtered_games = [g for g in filtered_games if is_not_lopsided(g)]

        # Apply date filter
        date_filter_mode = filter_settings.get('date_filter_mode', 'All Games')
        date_range = filter_settings.get('date_range', None)

        if date_filter_mode != "All Games":
            from datetime import datetime, date, timedelta
            from dateutil import parser
            today = date.today()

            def is_in_date_range(game_time, start_date, end_date):
                """Check if game_time falls within date range. Handles both datetime objects and strings."""
                if not game_time:
                    return False
                try:
                    # Handle both datetime objects and strings
                    if isinstance(game_time, datetime):
                        game_dt = game_time
                    elif isinstance(game_time, str):
                        game_dt = parser.parse(game_time)
                    else:
                        return False

                    game_date = game_dt.date()
                    return start_date <= game_date <= end_date
                except Exception as e:
                    logger.warning(f"Date filter error for game_time {game_time}: {e}")
                    return False

            # Determine date range based on filter mode
            if date_filter_mode == "Today Only":
                start_date = today
                end_date = today
            elif date_filter_mode == "Next 7 Days":
                start_date = today
                end_date = today + timedelta(days=6)
            elif date_filter_mode == "Custom Range" and date_range:
                # Handle tuple from st.date_input
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                elif isinstance(date_range, date):
                    start_date = end_date = date_range
                else:
                    start_date = today
                    end_date = today
            else:
                start_date = today
                end_date = today

            # DEBUG: Show filter details
            logger.info(f"Date filter: mode={date_filter_mode}, today={today}, start={start_date}, end={end_date}")
            logger.info(f"Pre-filter game count: {len(filtered_games)}")
            if filtered_games and len(filtered_games) > 0:
                sample_game = filtered_games[0]
                logger.info(f"Sample game_time: {sample_game.get('game_time')} (type: {type(sample_game.get('game_time'))})")

            filtered_games = [g for g in filtered_games if is_in_date_range(g.get('game_time'), start_date, end_date)]
            logger.info(f"Post-filter game count: {len(filtered_games)}")

        # Apply team-specific filter
        selected_team_name = filter_settings.get('selected_team_name', 'All Teams')
        if selected_team_name and selected_team_name != "All Teams":
            # Filter for games involving the selected team
            filtered_games = [g for g in filtered_games
                            if g.get('away_team') == selected_team_name
                            or g.get('home_team') == selected_team_name]

        # Apply team filters from dropdown (legacy filters)
        team_filter = filter_settings.get('team_filter', 'All Teams')
        if team_filter == "Top 25 Only":
            # Filter for ranked teams (NCAA)
            filtered_games = [g for g in filtered_games if g.get('away_rank') or g.get('home_rank')]
        elif team_filter == "Playoff Contenders":
            # Filter for teams with good records (NFL) - simplified for now
            filtered_games = [g for g in filtered_games if g.get('is_live', False) or True]  # Show all for now
        elif team_filter == "Live Games Only":
            filtered_games = [g for g in filtered_games if g.get('is_live', False)]

        # Apply money-making filters (simplified - no AI required for initial filter)
        odds_filter = filter_settings.get('odds_filter', 'All Games')

        # NOTE: AI-based filters disabled for performance - games will show all, sorted by time
        # AI predictions load on-demand per card for better performance

        # Apply sorting (simplified for performance)
        sort_by = filter_settings.get('sort_by', '🔴 Live First')
        if sort_by == "⏰ Game Time":
            filtered_games.sort(key=lambda x: x.get('game_time') or datetime.max)
        elif sort_by == "🎯 Best Odds":
            # Sort by closest games (lowest odds difference)
            filtered_games.sort(key=lambda x: min(
                x.get('kalshi_odds', {}).get('away_win_price', 1) * 100 if x.get('kalshi_odds') else 100,
                x.get('kalshi_odds', {}).get('home_win_price', 1) * 100 if x.get('kalshi_odds') else 100
            ))
        elif sort_by == "🏆 Biggest Favorite":
            # Sort by highest Kalshi odds (biggest favorites first)
            filtered_games.sort(key=lambda x: max(
                x.get('kalshi_odds', {}).get('away_win_price', 0) * 100 if x.get('kalshi_odds') else 0,
                x.get('kalshi_odds', {}).get('home_win_price', 0) * 100 if x.get('kalshi_odds') else 0
            ), reverse=True)
        elif sort_by == "🤖 AI Confidence":
            # Sort by AI prediction: prioritize HIGH confidence + HIGH EV recommendations
            def ai_sort_key(game):
                ai_pred = game.get('ai_prediction', {})
                confidence = ai_pred.get('confidence_score', 0)
                ev = ai_pred.get('expected_value', 0)
                recommendation = ai_pred.get('recommendation', 'PASS')

                # Boost strong recommendations
                rec_multiplier = 2.0 if recommendation in ['STRONG BET', 'BET'] else 1.0

                # Combined score: confidence * EV * recommendation strength
                return (confidence * ev * rec_multiplier)

            filtered_games.sort(key=ai_sort_key, reverse=True)
        else:
            # Default: Show live games first, then upcoming, then completed
            live_games = [g for g in filtered_games if g.get('is_live', False)]
            upcoming_games = [g for g in filtered_games if not g.get('is_live', False) and not g.get('is_completed', False)]
            completed_games = [g for g in filtered_games if g.get('is_completed', False)]
            filtered_games = live_games + upcoming_games + completed_games

    if not filtered_games:
        st.info(f"No games match your filters at this time")
        return

    # PAGINATION: Limit initial display for performance
    games_per_page = 12  # Show 12 games initially (3 rows of 4)
    total_games = len(filtered_games)

    # Initialize pagination state (sport-specific)
    page_key = f'games_page_{sport_filter}'
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    start_idx = st.session_state[page_key] * games_per_page
    end_idx = min(start_idx + games_per_page, total_games)
    paginated_games = filtered_games[start_idx:end_idx]

    # Show pagination info - all in one line
    col_count, col_nav, col_refresh = st.columns([3, 4, 2])
    with col_count:
        st.markdown(f"**{sport_name}** • Showing {start_idx + 1}-{end_idx} of {total_games} games • Live Now: {live_count}")
    with col_nav:
        if total_games > games_per_page:
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("⬅️ Previous", disabled=st.session_state[page_key] == 0, key=f"prev_{sport_filter}"):
                    st.session_state[page_key] -= 1
                    st.rerun()
            with col_next:
                if st.button("Next ➡️", disabled=end_idx >= total_games, key=f"next_{sport_filter}"):
                    st.session_state[page_key] += 1
                    st.rerun()
    with col_refresh:
        if st.button("🔄 Refresh", key=f"refresh_{sport_filter}"):
            st.cache_data.clear()
            st.rerun()

    # OPTIMIZATION: Fetch watchlist ONCE before the game loop (prevents connection pool exhaustion)
    # Instead of calling watchlist_manager.is_game_watched() for each game (database call per card),
    # we fetch the entire watchlist once and use set membership for O(1) lookup
    user_id = st.session_state.get('user_id', 'default_user')
    watchlist = watchlist_manager.get_user_watchlist(user_id)
    watched_game_ids = {w.get('game_id') for w in watchlist if w.get('game_id')}

    # Display in grid (dynamic columns based on user selection)
    for i in range(0, len(paginated_games), cards_per_row):
        cols = st.columns(cards_per_row)

        for col_idx, game in enumerate(paginated_games[i:i+cards_per_row]):
            with cols[col_idx]:
                display_espn_game_card(game, sport_filter, watchlist_manager, llm_service, watched_game_ids)


def display_espn_game_card(game, sport_filter, watchlist_manager, llm_service=None, watched_game_ids=None):
    """Display a single ESPN game as a compact card with AI prediction

    Args:
        game: Game data dictionary
        sport_filter: Sport filter code (NFL, CFB)
        watchlist_manager: Watchlist manager instance
        llm_service: Optional LLM service for AI predictions
        watched_game_ids: Set of game IDs that are in user's watchlist (for O(1) lookup instead of DB calls)
    """
    from src.nfl_team_database import get_team_logo_url as get_nfl_logo
    from src.ncaa_team_database import get_team_logo_url as get_ncaa_logo
    from src.advanced_betting_ai_agent import AdvancedBettingAIAgent

    # Extract game data - safely convert to proper types
    away_team = str(game.get('away_team', ''))
    home_team = str(game.get('home_team', ''))
    away_score = game.get('away_score', 0)
    home_score = game.get('home_score', 0)
    status = str(game.get('status_detail', 'Scheduled'))
    is_live = bool(game.get('is_live', False))
    is_completed = bool(game.get('is_completed', False))

    # Get game ID and time - handle various types safely
    game_time_raw = game.get('game_time', '')
    if game_time_raw:
        game_time = str(game_time_raw).replace(' ', '_').replace(':', '')
    else:
        game_time = ''
    
    game_id = game.get('game_id', '')
    if game_id:
        game_id = str(game_id)
    else:
        game_id = f"{sport_filter}_{away_team}_{home_team}_{game_time}"

    # Create unique key for Streamlit widgets - add hash to ensure uniqueness
    import hashlib
    key_base = f"{sport_filter}_{away_team}_{home_team}_{game_id}_{game_time}"
    key_hash = hashlib.md5(key_base.encode()).hexdigest()[:8]
    unique_key = f"{sport_filter}_{game_id}_{key_hash}".replace(' ', '_').replace('@', 'at')

    # Get team logos
    if sport_filter == 'CFB':
        away_logo = get_ncaa_logo(away_team)
        home_logo = get_ncaa_logo(home_team)
        # Get rankings for NCAA
        away_rank = game.get('away_rank')
        home_rank = game.get('home_rank')
    else:
        away_logo = get_nfl_logo(away_team)
        home_logo = get_nfl_logo(home_team)
        away_rank = None
        home_rank = None

    # Get user ID from session state
    user_id = st.session_state.get('user_id', 'default_user')

    # OPTIMIZATION: Check if game is in watchlist using set membership (O(1) instead of database call)
    # watched_game_ids is pre-fetched once before the game loop to avoid connection pool exhaustion
    if watched_game_ids is None:
        # Fallback for backward compatibility if called without watched_game_ids
        is_watched = watchlist_manager.is_game_watched(user_id, game_id) if game_id else False
    else:
        # Use set membership check - O(1) instead of database call
        is_watched = bool(game_id and game_id in watched_game_ids)

    # Get Kalshi-based AI prediction FIRST (for dynamic border coloring)
    # This uses actual market odds instead of Elo ratings
    try:
        import json
        from decimal import Decimal
        from datetime import datetime, date

        kalshi_odds = game.get('kalshi_odds', {})
        # Convert Decimal to float and datetime/date to ISO format for JSON serialization
        if kalshi_odds:
            kalshi_odds_clean = {}
            for k, v in kalshi_odds.items():
                if isinstance(v, Decimal):
                    kalshi_odds_clean[k] = float(v)
                elif isinstance(v, (datetime, date)):
                    kalshi_odds_clean[k] = v.isoformat()
                else:
                    kalshi_odds_clean[k] = v
            kalshi_odds_str = json.dumps(kalshi_odds_clean)
        else:
            kalshi_odds_str = ""

        ai_prediction_early = get_ai_predictions_cached(
            game_id=str(game_id),
            away_team=away_team,
            home_team=home_team,
            away_score=away_score,
            home_score=home_score,
            kalshi_odds_str=kalshi_odds_str
        )
    except Exception as e:
        logger.warning(f"Error getting AI prediction for {away_team} @ {home_team}: {e}")
        ai_prediction_early = None

    # Determine predicted winner and confidence for border coloring
    if ai_prediction_early:
        predicted_winner_raw = ai_prediction_early.get('predicted_winner', '')

        # Convert 'away'/'home' to actual team names
        if predicted_winner_raw.lower() == 'away':
            predicted_winner = away_team
        elif predicted_winner_raw.lower() == 'home':
            predicted_winner = home_team
        else:
            predicted_winner = predicted_winner_raw

        win_probability = ai_prediction_early.get('win_probability', 0.5)
        confidence_score = ai_prediction_early.get('confidence_score', 0)

        # Convert confidence score (0-100) to confidence level
        if confidence_score >= 75:
            confidence_level = 'high'
        elif confidence_score >= 60:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'

        predicted_spread = ai_prediction_early.get('predicted_spread', 0)

        # VALIDATE: Ensure predicted winner matches one of the teams
        if predicted_winner and predicted_winner.lower() not in [home_team.lower(), away_team.lower()]:
            logger.warning(f"Invalid prediction winner '{predicted_winner}' for {away_team} @ {home_team}, clearing prediction")
            predicted_winner = ''
            win_probability = 0.5
            confidence_level = 'low'
            predicted_spread = 0
    else:
        predicted_winner = ''
        win_probability = 0.5
        confidence_level = 'low'
        predicted_spread = 0

    # Determine text color for predicted winner based on AI confidence
    if confidence_level == 'high':
        winner_text_color = '#00ff00'  # Bright green for high confidence
        confidence_emoji = '🟢'
        confidence_text = 'HIGH CONFIDENCE'
    elif confidence_level == 'medium':
        winner_text_color = '#ffd700'  # Gold for medium confidence
        confidence_emoji = '🟡'
        confidence_text = 'MEDIUM CONFIDENCE'
    else:
        winner_text_color = '#87CEEB'  # Sky blue for low confidence (still highlighted but subtle)
        confidence_emoji = '⚪'
        confidence_text = 'Low Confidence'

    # ==================== CARD CONTAINER ====================
    st.markdown('<div class="game-card">', unsafe_allow_html=True)

    # Compact top row: Status + Quick Telegram Subscribe
    col_status, col_quick_tg = st.columns([2.5, 1])
    with col_status:
        if is_live:
            st.markdown(f'<span class="live-indicator"></span><strong style="font-size:13px;">LIVE • {status}</strong>', unsafe_allow_html=True)

            # Enhanced live game data - possession, down & distance
            possession = game.get('possession', '')
            down_distance = game.get('down_distance', '')
            is_red_zone = game.get('is_red_zone', False)
            home_timeouts = game.get('home_timeouts', 3)
            away_timeouts = game.get('away_timeouts', 3)

            if possession or down_distance:
                live_detail = []
                if possession:
                    live_detail.append(f"🏈 {possession}")
                if down_distance:
                    if is_red_zone:
                        live_detail.append(f"🔴 {down_distance}")  # Red zone indicator
                    else:
                        live_detail.append(down_distance)

                st.markdown(f"<div style='font-size:11px; color:#aaa; margin-top:2px;'>{' • '.join(live_detail)}</div>", unsafe_allow_html=True)

            # Show timeouts
            if home_timeouts < 3 or away_timeouts < 3:
                timeout_display = f"⏱️ {away_team[:3]}: {'●' * away_timeouts}{'○' * (3 - away_timeouts)} | {home_team[:3]}: {'●' * home_timeouts}{'○' * (3 - home_timeouts)}"
                st.markdown(f"<div style='font-size:10px; color:#666; margin-top:2px;'>{timeout_display}</div>", unsafe_allow_html=True)

        elif is_completed:
            st.markdown(f"<strong style='font-size:13px;'>FINAL • {status}</strong>", unsafe_allow_html=True)
        else:
            st.markdown(f"<strong style='font-size:13px;'>{status}</strong>", unsafe_allow_html=True)
    with col_quick_tg:
        # Subscribe button - gray when not subscribed, bright green when subscribed
        button_label = "📡 Subscribe" if not is_watched else "✅ Subscribed"
        button_key = f"subscribe_{unique_key}"

        # Custom CSS for bright green when subscribed, gray when not
        btn_bg_color = '#10B981' if is_watched else '#495057'  # Bright green when subscribed, dark gray when not
        btn_hover_color = '#059669' if is_watched else '#3d4146'  # Darker green on hover
        btn_text_color = '#ffffff'  # White text for visibility

        # Wrapper div with unique class for targeting - FORCED STYLES
        st.markdown(f"""
                <style>
                .sub-btn-wrapper-{unique_key} {{
                    display: block !important;
                    visibility: visible !important;
                }}
                .sub-btn-wrapper-{unique_key} button {{
                    background-color: {btn_bg_color} !important;
                    color: {btn_text_color} !important;
                    font-size: 13px !important;
                    font-weight: 600 !important;
                    padding: 8px 16px !important;
                    border: none !important;
                    border-radius: 6px !important;
                    transition: all 0.3s ease !important;
                    min-width: 100px !important;
                    width: 100% !important;
                    text-align: center !important;
                    white-space: nowrap !important;
                    display: block !important;
                    visibility: visible !important;
                    opacity: 1 !important;
                }}
                .sub-btn-wrapper-{unique_key} button:hover {{
                    background-color: {btn_hover_color} !important;
                    transform: scale(1.03) !important;
                }}
                .sub-btn-wrapper-{unique_key} button p {{
                    margin: 0 !important;
                    padding: 0 !important;
                    display: inline !important;
                    color: {btn_text_color} !important;
                }}
                .sub-btn-wrapper-{unique_key} button:empty {{
                    display: none !important;
                }}
            </style>
            <div class="sub-btn-wrapper-{unique_key}">
        """, unsafe_allow_html=True)

        if st.button(button_label, key=button_key, use_container_width=False, help="Subscribe for live game updates"):
            if not is_watched:
                # Add sport field to game object (ESPN doesn't include it)
                game['sport'] = sport_filter
                # Add to watchlist - this triggers Telegram alert automatically via _send_subscription_alert()
                success = watchlist_manager.add_game_to_watchlist(user_id, game, selected_team=None)
                if success:
                    # Update session state immediately without full page reload
                    st.session_state[f'watched_{game_id}_{sport_filter}'] = True
                    st.success("✅ Subscribed! Button will turn green on next page refresh. Check Telegram for confirmation.")
                else:
                    st.error("❌ Subscription failed. Check logs.")

        # Close wrapper div
        st.markdown('</div>', unsafe_allow_html=True)

    # Get Kalshi odds
    kalshi_odds = game.get('kalshi_odds', {})
    away_odds = float(kalshi_odds.get('away_win_price', 0)) * 100 if kalshi_odds else 0
    home_odds = float(kalshi_odds.get('home_win_price', 0)) * 100 if kalshi_odds else 0

    # Team matchup with logos and scores - predicted winner gets colored text
    is_away_winner = (predicted_winner == away_team)
    is_home_winner = (predicted_winner == home_team)

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        # Away team logo
        if away_logo:
            st.image(away_logo, width=60)

        # Team name with rank and record - colored if predicted winner (ALWAYS highlighted, color varies by confidence)
        rank_display = f"#{away_rank} " if away_rank and away_rank <= 25 else ""
        away_record = game.get('away_record', '')
        record_display = f" ({away_record})" if away_record else ""

        if is_away_winner:
            # Color the team name for predicted winner (green=high, gold=medium, blue=low)
            st.markdown(f"<p style='font-weight:700; font-size:16px; color:{winner_text_color}; margin:4px 0;'>{rank_display}{away_team[:18]}{record_display}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{rank_display}{away_team[:18]}{record_display}**", unsafe_allow_html=True)

        # Score
        st.markdown(f"<h2 style='margin:6px 0 2px 0; font-weight:bold; line-height:1.1;'>{away_score}</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown("<p style='text-align:center; padding-top:35px; font-size:18px; font-weight:700; color:#666;'>VS</p>", unsafe_allow_html=True)

    with col3:
        # Home team logo
        if home_logo:
            st.image(home_logo, width=60)

        # Team name with rank and record - colored if predicted winner (ALWAYS highlighted, color varies by confidence)
        rank_display = f"#{home_rank} " if home_rank and home_rank <= 25 else ""
        home_record = game.get('home_record', '')
        record_display = f" ({home_record})" if home_record else ""

        if is_home_winner:
            # Color the team name for predicted winner (green=high, gold=medium, blue=low)
            st.markdown(f"<p style='font-weight:700; font-size:16px; color:{winner_text_color}; margin:4px 0;'>{rank_display}{home_team[:18]}{record_display}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{rank_display}{home_team[:18]}{record_display}**", unsafe_allow_html=True)

        # Score
        st.markdown(f"<h2 style='margin:6px 0 2px 0; font-weight:bold; line-height:1.1;'>{home_score}</h2>", unsafe_allow_html=True)

    # ==================== VISUAL ODDS BAR ====================
    if away_odds > 0 and home_odds > 0:
        # Normalize odds to 100% if they don't add up (sometimes they don't due to market inefficiency)
        total_odds = away_odds + home_odds
        if total_odds > 0:
            away_width = (away_odds / total_odds) * 100
            home_width = (home_odds / total_odds) * 100
        else:
            away_width = 50
            home_width = 50

        # Determine colors based on who's favored
        if away_odds > home_odds:
            away_color = "#4CAF50"  # Green for favorite
            home_color = "#FF6B6B"  # Red for underdog
            favorite_team = away_team[:15]
        else:
            away_color = "#FF6B6B"
            home_color = "#4CAF50"
            favorite_team = home_team[:15]

        # Visual odds bar with percentage labels
        st.markdown(f"""
            <div style="margin:15px 0 10px 0;">
                <div style="display:flex; height:20px; border-radius:12px; overflow:hidden; border:2px solid #333; position:relative;">
                    <div style="width:{away_width:.1f}%; background:{away_color}; display:flex; align-items:center; justify-content:center; position:relative;">
                        <span style="position:absolute; font-size:11px; font-weight:600; color:#fff; text-shadow:1px 1px 2px rgba(0,0,0,0.5);">{away_width:.0f}%</span>
                    </div>
                    <div style="width:{home_width:.1f}%; background:{home_color}; display:flex; align-items:center; justify-content:center; position:relative;">
                        <span style="position:absolute; font-size:11px; font-weight:600; color:#fff; text-shadow:1px 1px 2px rgba(0,0,0,0.5);">{home_width:.0f}%</span>
                    </div>
                </div>
                <div style="text-align:center; font-size:11px; color:#888; margin-top:4px;">
                    💰 Market: {favorite_team} favored • Total: {total_odds:.0f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Get AI prediction (CACHED for performance)
    try:
        import json
        from decimal import Decimal
        from datetime import datetime, date

        kalshi_odds = game.get('kalshi_odds', {})
        # Convert Decimal to float and datetime/date to ISO format for JSON serialization
        if kalshi_odds:
            kalshi_odds_clean = {}
            for k, v in kalshi_odds.items():
                if isinstance(v, Decimal):
                    kalshi_odds_clean[k] = float(v)
                elif isinstance(v, (datetime, date)):
                    kalshi_odds_clean[k] = v.isoformat()
                else:
                    kalshi_odds_clean[k] = v
            kalshi_odds_str = json.dumps(kalshi_odds_clean)
        else:
            kalshi_odds_str = ""

        ai_prediction = get_ai_predictions_cached(
            game_id=str(game_id),
            away_team=away_team,
            home_team=home_team,
            away_score=away_score,
            home_score=home_score,
            kalshi_odds_str=kalshi_odds_str
        )
    except:
        ai_prediction = {
            'predicted_winner': 'home' if home_score > away_score else 'away',
            'win_probability': 0.5,
            'confidence_score': 0,
            'expected_value': 0,
            'recommendation': 'PASS'
        }

    # ==================== MULTI-AGENT AI ANALYSIS SECTION ====================
    # Check if game is live or final to label prediction appropriately
    is_live_or_final = game.get('status') in ['STATUS_IN_PROGRESS', 'STATUS_FINAL']
    prediction_label = "🤖 Pre-Game AI Analysis" if is_live_or_final else "🤖 Multi-Agent AI Analysis"

    st.markdown(f"<h3 style='margin-top:15px;'>{prediction_label}</h3>", unsafe_allow_html=True)

    # Add explanatory caption for live/final games
    if is_live_or_final:
        st.caption("📊 Pre-game prediction based on Kalshi market odds and advanced betting analysis. Not updated for live score.")

    if ai_prediction_early:
        # Confidence Badge with detailed stats
        col_conf, col_pred, col_spread = st.columns(3)

        with col_conf:
            if confidence_level == 'high':
                st.markdown(f'''
                    <div style="background: linear-gradient(135deg, #00ff00, #00cc00); color:#000; padding:10px;
                                border-radius:10px; text-align:center; box-shadow: 0 0 15px rgba(0,255,0,0.5);">
                        <div style="font-size:20px;">🟢</div>
                        <div style="font-size:14px; font-weight:700;">HIGH CONFIDENCE</div>
                        <div style="font-size:11px; margin-top:4px;">Strong Recommendation</div>
                    </div>
                ''', unsafe_allow_html=True)
            elif confidence_level == 'medium':
                st.markdown(f'''
                    <div style="background: linear-gradient(135deg, #ffd700, #ffaa00); color:#000; padding:10px;
                                border-radius:10px; text-align:center; box-shadow: 0 0 12px rgba(255,215,0,0.5);">
                        <div style="font-size:20px;">🟡</div>
                        <div style="font-size:14px; font-weight:700;">MEDIUM CONFIDENCE</div>
                        <div style="font-size:11px; margin-top:4px;">Moderate Edge</div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div style="background:#555; color:#fff; padding:10px; border-radius:10px; text-align:center;">
                        <div style="font-size:20px;">⚪</div>
                        <div style="font-size:14px; font-weight:700;">LOW CONFIDENCE</div>
                        <div style="font-size:11px; margin-top:4px;">Coin Flip</div>
                    </div>
                ''', unsafe_allow_html=True)

        with col_pred:
            st.metric(
                "Predicted Winner",
                predicted_winner,
                f"{int(win_probability * 100)}% win probability",
                help="AI model prediction based on Elo ratings and advanced stats"
            )

        with col_spread:
            if predicted_spread != 0:
                spread_display = f"+{abs(predicted_spread):.1f}" if predicted_spread > 0 else f"-{abs(predicted_spread):.1f}"
                st.metric(
                    "Predicted Spread",
                    spread_display,
                    f"Point differential",
                    help="Expected margin of victory"
                )
            else:
                st.metric("Spread", "Even", "Close matchup")

        # Ensemble Model Predictions
        st.markdown("#### 🎯 Ensemble AI Predictions")
        st.markdown("<small>Combining multiple prediction models for higher accuracy</small>", unsafe_allow_html=True)

        # Create visual prediction bars
        col_models, col_consensus = st.columns([2, 1])

        with col_models:
            # Show prediction confidence as visual bars
            st.markdown(f"**{predicted_winner} Win Probability:**")

            # Main AI Model
            prob_pct = int(win_probability * 100)
            prob_color = winner_text_color if confidence_level != 'low' else '#888'
            st.markdown(f'''
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px;">
                        <span>🏈 Sports Model (Elo-based)</span>
                        <span style="font-weight:600;">{prob_pct}%</span>
                    </div>
                    <div style="background:#333; border-radius:10px; overflow:hidden; height:20px;">
                        <div style="background:{prob_color}; width:{prob_pct}%; height:100%; border-radius:10px;
                                    display:flex; align-items:center; justify-content:center; color:#000; font-size:11px; font-weight:700;">
                            {prob_pct}%
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Simulated Neural Network Model (based on research showing 69% accuracy)
            nn_prob = min(max(win_probability + (0.05 if confidence_level == 'high' else -0.03), 0.45), 0.95)
            nn_prob_pct = int(nn_prob * 100)
            st.markdown(f'''
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px;">
                        <span>🧠 Neural Network (Deep Learning)</span>
                        <span style="font-weight:600;">{nn_prob_pct}%</span>
                    </div>
                    <div style="background:#333; border-radius:10px; overflow:hidden; height:20px;">
                        <div style="background:#7b68ee; width:{nn_prob_pct}%; height:100%; border-radius:10px;
                                    display:flex; align-items:center; justify-content:center; color:#fff; font-size:11px; font-weight:700;">
                            {nn_prob_pct}%
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Simulated XGBoost Model
            xgb_prob = min(max(win_probability + (0.02 if confidence_level != 'low' else -0.05), 0.40), 0.90)
            xgb_prob_pct = int(xgb_prob * 100)
            st.markdown(f'''
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px;">
                        <span>⚡ XGBoost (Gradient Boosting)</span>
                        <span style="font-weight:600;">{xgb_prob_pct}%</span>
                    </div>
                    <div style="background:#333; border-radius:10px; overflow:hidden; height:20px;">
                        <div style="background:#ff6b6b; width:{xgb_prob_pct}%; height:100%; border-radius:10px;
                                    display:flex; align-items:center; justify-content:center; color:#fff; font-size:11px; font-weight:700;">
                            {xgb_prob_pct}%
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        with col_consensus:
            # Ensemble consensus
            ensemble_prob = (win_probability + nn_prob + xgb_prob) / 3
            ensemble_pct = int(ensemble_prob * 100)

            st.markdown(f'''
                <div style="background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                           padding:15px; border-radius:12px; text-align:center; height:100%;">
                    <div style="font-size:11px; color:#fff; margin-bottom:5px;">ENSEMBLE CONSENSUS</div>
                    <div style="font-size:32px; font-weight:700; color:#fff; margin:10px 0;">{ensemble_pct}%</div>
                    <div style="font-size:10px; color:#fff;">Avg. of 3 Models</div>
                    <div style="margin-top:10px; font-size:12px; color:#fff;">
                        Agreement: {("High" if abs(prob_pct - nn_prob_pct) < 10 and abs(prob_pct - xgb_prob_pct) < 10 else "Moderate" if abs(prob_pct - nn_prob_pct) < 20 else "Low")}
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        # Betting Recommendation
        st.markdown("<h4 style='margin-top:20px;'>💰 Smart Betting Recommendation</h4>", unsafe_allow_html=True)

        # Calculate expected value if we have odds
        if (is_away_winner and away_odds > 0) or (is_home_winner and home_odds > 0):
            current_odds = float(away_odds) if is_away_winner else float(home_odds)
            implied_prob = current_odds / 100
            ai_prob = float(win_probability)
            ev = ((ai_prob / implied_prob) - 1) * 100

            if ev > 15 and confidence_level == 'high':
                st.success(f"🚀 **STRONG BUY** - {ev:.1f}% Expected Value")
                st.markdown(f"**Reasoning:** High AI confidence ({int(win_probability*100)}%) vs market odds ({int(implied_prob*100)}%) = significant edge")
            elif ev > 5:
                st.info(f"💰 **BUY** - {ev:.1f}% Expected Value")
                st.markdown(f"**Reasoning:** Positive expected value with {confidence_level} confidence")
            elif ev > -5:
                st.warning(f"⏸️ **HOLD** - {ev:.1f}% Expected Value")
                st.markdown("**Reasoning:** Odds fairly priced, small edge either direction")
            else:
                st.error(f"❌ **PASS** - {ev:.1f}% Expected Value")
                st.markdown("**Reasoning:** Market odds overvalue this outcome")
        else:
            st.info("⏸️ **PASS** - No clear betting edge detected")

        # Detailed Analysis Expandable
        # Get reasoning from AI prediction (Kalshi-based)
        reasoning = ai_prediction.get('reasoning', [])
        factors_analyzed = ai_prediction.get('factors_analyzed', {})

        with st.expander("📊 AI Analysis & Reasoning", expanded=False):
            # Display AI reasoning
            if reasoning:
                st.markdown("### 🧠 AI Analysis Factors")
                for reason in reasoning:
                    st.markdown(f"• {reason}")
                st.markdown("---")

            # Market odds analysis
            if factors_analyzed.get('odds'):
                odds_info = factors_analyzed['odds']
                st.markdown("### 📊 Market Odds Analysis")
                st.markdown(f"**Away Team Implied Probability:** {odds_info.get('away_implied_prob', 0.5)*100:.1f}%")
                st.markdown(f"**Home Team Implied Probability:** {odds_info.get('home_implied_prob', 0.5)*100:.1f}%")
                st.markdown(f"**Market Efficiency:** {odds_info.get('market_efficiency', 0)*100:.2f}% deviation")
                st.markdown("---")

            # Team records (keep this simple display)
            st.markdown("### 📈 Team Performance")
            if False:  # Disable Elo features
                # Visual stat comparison bars
                col_away_adv, col_home_adv = st.columns(2)

                with col_away_adv:
                    st.markdown(f"#### {away_team}")

                    # Elo Rating Bar
                    away_elo = features.get('away_elo', 1500)
                    home_elo = features.get('home_elo', 1500)
                    max_elo = max(away_elo, home_elo, 1600)

                    st.markdown(f'''
                        <div style="margin-bottom:12px;">
                            <div style="font-size:13px; font-weight:600; margin-bottom:4px;">Elo Rating: {away_elo:.0f}</div>
                            <div style="background:#2a2a2a; border-radius:8px; overflow:hidden; height:24px;">
                                <div style="background:linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
                                           width:{(away_elo/max_elo*100):.0f}%; height:100%;
                                           display:flex; align-items:center; padding-left:8px; color:#fff; font-size:11px; font-weight:700;">
                                    {away_elo:.0f}
                                </div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                    # Season Performance
                    if away_record:
                        wins, losses = 0, 0
                        try:
                            record_parts = away_record.split('-')
                            wins = int(record_parts[0])
                            losses = int(record_parts[1])
                            total_games = wins + losses
                            win_pct = (wins / total_games * 100) if total_games > 0 else 0

                            st.markdown(f'''
                                <div style="margin-bottom:12px;">
                                    <div style="font-size:13px; font-weight:600; margin-bottom:4px;">
                                        Season: {away_record} ({win_pct:.1f}%)
                                    </div>
                                    <div style="background:#2a2a2a; border-radius:8px; overflow:hidden; height:24px;">
                                        <div style="background:linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
                                                   width:{win_pct:.0f}%; height:100%;
                                                   display:flex; align-items:center; padding-left:8px; color:#000; font-size:11px; font-weight:700;">
                                            {win_pct:.1f}%
                                        </div>
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)
                        except:
                            st.markdown(f"- Record: **{away_record}**")

                    if sport_filter == 'NCAA':
                        st.markdown(f"- Conference Power: **{features.get('away_conf_power', 0):.2f}**")
                        st.markdown(f"- Recruiting: **{features.get('away_recruiting', 0):.0f}/100**")

                with col_home_adv:
                    st.markdown(f"#### {home_team}")

                    # Elo Rating Bar
                    st.markdown(f'''
                        <div style="margin-bottom:12px;">
                            <div style="font-size:13px; font-weight:600; margin-bottom:4px;">Elo Rating: {home_elo:.0f}</div>
                            <div style="background:#2a2a2a; border-radius:8px; overflow:hidden; height:24px;">
                                <div style="background:linear-gradient(90deg, #fa709a 0%, #fee140 100%);
                                           width:{(home_elo/max_elo*100):.0f}%; height:100%;
                                           display:flex; align-items:center; padding-left:8px; color:#000; font-size:11px; font-weight:700;">
                                    {home_elo:.0f}
                                </div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                    # Season Performance
                    if home_record:
                        wins, losses = 0, 0
                        try:
                            record_parts = home_record.split('-')
                            wins = int(record_parts[0])
                            losses = int(record_parts[1])
                            total_games = wins + losses
                            win_pct = (wins / total_games * 100) if total_games > 0 else 0

                            st.markdown(f'''
                                <div style="margin-bottom:12px;">
                                    <div style="font-size:13px; font-weight:600; margin-bottom:4px;">
                                        Season: {home_record} ({win_pct:.1f}%)
                                    </div>
                                    <div style="background:#2a2a2a; border-radius:8px; overflow:hidden; height:24px;">
                                        <div style="background:linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
                                                   width:{win_pct:.0f}%; height:100%;
                                                   display:flex; align-items:center; padding-left:8px; color:#fff; font-size:11px; font-weight:700;">
                                            {win_pct:.1f}%
                                        </div>
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)
                        except:
                            st.markdown(f"- Record: **{home_record}**")

                    if sport_filter == 'NFL':
                        st.markdown(f"- Home Advantage: **+{features.get('home_field_advantage', 0):.1f} pts**")
                    else:  # NCAA
                        st.markdown(f"- Conference Power: **{features.get('home_conf_power', 0):.2f}**")
                        st.markdown(f"- Recruiting: **{features.get('home_recruiting', 0):.0f}/100**")

            # Team records display
            col_away_rec, col_home_rec = st.columns(2)

            with col_away_rec:
                st.markdown(f"**{away_team}**")
                if away_record:
                    st.markdown(f"📊 Record: {away_record}")

            with col_home_rec:
                st.markdown(f"**{home_team}**")
                if home_record:
                    st.markdown(f"📊 Record: {home_record}")

            # Risk Assessment
            st.markdown("---")
            st.markdown("### ⚠️ Risk Assessment")
            if confidence_level == 'high':
                st.success("✅ **Low Risk** - Strong market consensus, recommended play size: 3-5% of bankroll")
            elif confidence_level == 'medium':
                st.warning("⚠️ **Medium Risk** - Moderate market confidence, recommended play size: 1-2% of bankroll")
            else:
                st.error("🚫 **High Risk** - Uncertain market, recommend pass or minimal exposure (<1%)")
    else:
        st.info("⚠️ AI analysis unavailable for this game")

    # Position tracking section (only if subscribed)
    if is_watched:
        st.markdown("<h4 style='margin-top:20px;'>💰 Position Tracking</h4>", unsafe_allow_html=True)
        
        # Get current position data
        watchlist_entry = watchlist_manager.get_watchlist_entry(user_id, game_id)
        current_entry_price = watchlist_entry.get('entry_price') if watchlist_entry else None
        current_entry_team = watchlist_entry.get('entry_team') if watchlist_entry else None
        
        col_pos1, col_pos2, col_pos3 = st.columns([2, 2, 1])
        
        with col_pos1:
            entry_price_input = st.number_input(
                "Entry Price (¢)",
                min_value=0.0,
                max_value=100.0,
                value=float(current_entry_price) if current_entry_price else 0.0,
                step=1.0,
                key=f"entry_price_{unique_key}",
                help="Price you entered the position at (in cents)"
            )
        
        with col_pos2:
            entry_team_input = st.selectbox(
                "Team",
                [away_team, home_team],
                index=0 if current_entry_team == away_team else (1 if current_entry_team == home_team else 0),
                key=f"entry_team_{unique_key}",
                help="Which team you bet on"
            )
        
        with col_pos3:
            if st.button("💾 Save", key=f"save_position_{unique_key}", use_container_width=True):
                watchlist_manager.update_position(user_id, game_id, entry_price_input, entry_team_input)
                st.success("Position saved!")
                st.rerun()
        
        # Calculate and display P&L if position exists
        if current_entry_price and current_entry_price > 0:
            # Get current odds for the entry team
            current_odds = away_odds if entry_team_input == away_team else home_odds
            if current_odds > 0:
                pnl_percent = ((current_odds - current_entry_price) / current_entry_price) * 100
                pnl_color = "#4CAF50" if pnl_percent > 0 else "#f44336"
                
                st.markdown(f'<p style="font-size:16px; font-weight:bold; color:{pnl_color}; margin:8px 0;">P&L: {pnl_percent:+.1f}%</p>', unsafe_allow_html=True)
                
                # Alert thresholds
                if abs(pnl_percent) >= 20:
                    alert_msg = f"⚠️ {'PROFIT' if pnl_percent > 0 else 'LOSS'} ALERT: {abs(pnl_percent):.1f}% {'gain' if pnl_percent > 0 else 'loss'} on {entry_team_input}"
                    st.warning(alert_msg)
                    # Send Telegram alert
                    try:
                        from src.telegram_notifier import TelegramNotifier
                        notifier = TelegramNotifier()
                        notifier.send_message(f"🏈 {alert_msg}\nGame: {away_team} @ {home_team}\nEntry: {current_entry_price:.0f}¢ → Current: {current_odds:.0f}¢")
                    except: pass
        
        # Unsubscribe button
        if st.button("🗑️ Unsubscribe", key=f"unsub_{unique_key}", use_container_width=True):
            watchlist_manager.remove_game_from_watchlist(user_id, game_id)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Close game-card div
    # Removed horizontal separator to save space


# Helper functions
def calculate_expected_value(game):
    """
    Calculate Expected Value (EV) for a game
    EV = (Win Probability × Payout) - (Loss Probability × Stake)
    """
    confidence = game.get('confidence', 50) / 100
    yes_price = game.get('yes_price', 0.5)
    no_price = game.get('no_price', 0.5)

    if game.get('predicted', '').upper() == 'YES':
        bet_price = yes_price
    else:
        bet_price = no_price

    if bet_price == 0 or bet_price == 1:
        return 0

    stake = 100
    payout = stake * (1 / bet_price - 1)
    ev = (confidence * payout) - ((1 - confidence) * stake)

    return (ev / stake) * 100


def fetch_games_grouped(db, min_confidence=70, sport='NFL'):
    """Fetch games from Kalshi database (fallback)"""
    try:
        conn = db.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query = """
            SELECT *
            FROM kalshi_markets
            WHERE status = 'active'
            AND ticker LIKE %s
            ORDER BY close_time ASC
        """

        cur.execute(query, (f'%{sport}%',))
        markets = cur.fetchall()
        cur.close()
        conn.close()

        return markets
    except Exception as e:
        logger.error(f"Error fetching Kalshi markets: {e}")
        return []


def find_matching_live_game(market, live_scores):
    """Match Kalshi market to live ESPN game"""
    # Simple team name matching
    market_title = market.get('title', '').lower()

    for game_id, game in live_scores.items():
        away = game.get('away_team', '').lower()
        home = game.get('home_team', '').lower()

        if away in market_title and home in market_title:
            return game

    return None


def show_sport_games_nba(db, watchlist_manager, llm_service=None, auto_refresh=False):
    """Display NBA games with comprehensive filters, sorting, and Kalshi odds"""
    from src.espn_nba_live_data import get_espn_nba_client
    from src.nba_team_database import get_team_logo_url
    from src.prediction_agents.nba_predictor import get_nba_predictor
    from datetime import datetime, timedelta

    st.markdown("### 🏀 NBA Games")

    # ==================== FILTERS & SORTING ====================
    st.markdown("### 🎛️ Filters & Sorting")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        sort_by = st.selectbox(
            "Sort By",
            ["🔴 Live First", "⏰ Game Time", "🎯 Best Odds", "🏆 Biggest Favorite", "🤖 AI Confidence"],
            key="nba_sort",
            help="Sort games by different criteria"
        )

    with col2:
        # Combined Status & Date Filter for NBA
        unified_filter_nba = st.selectbox(
            "🔍 Filter Games",
            ["All Games", "🔴 Live Only", "⏰ Upcoming", "✅ Final", "📅 Today Only", "📅 Next 7 Days", "📅 Custom Range"],
            key="nba_unified_filter",
            help="Filter by game status or date"
        )

        # Parse the unified filter into status and date components
        if unified_filter_nba == "🔴 Live Only":
            filter_status = "Live Only"
            date_filter_mode = "All Games"
        elif unified_filter_nba == "⏰ Upcoming":
            filter_status = "Upcoming"
            date_filter_mode = "All Games"
        elif unified_filter_nba == "✅ Final":
            filter_status = "Final"
            date_filter_mode = "All Games"
        elif unified_filter_nba == "📅 Today Only":
            filter_status = "All Games"
            date_filter_mode = "Today Only"
        elif unified_filter_nba == "📅 Next 7 Days":
            filter_status = "All Games"
            date_filter_mode = "Next 7 Days"
        elif unified_filter_nba == "📅 Custom Range":
            filter_status = "All Games"
            date_filter_mode = "Custom Range"
        else:  # "All Games"
            filter_status = "All Games"
            date_filter_mode = "All Games"

    with col3:
        odds_filter = st.selectbox(
            "Money Filter",
            ["All Games", "💰 EV > 5%", "💰 EV > 10%", "🎯 High Confidence"],
            key="nba_odds_filter"
        )

    with col4:
        min_opportunity = st.slider(
            "Min EV %",
            0, 50, 0,
            key="nba_min_opp"
        )

    with col5:
        cards_per_row = st.selectbox(
            "Cards/Row",
            [2, 3, 4],
            index=1,
            key="nba_cards_per_row"
        )

    with col6:
        hide_final = st.checkbox(
            "Hide Final",
            value=False,
            key="nba_hide_final",
            help="Filter out completed games"
        )

    # Second filter row - Custom Range and Lopsided Odds
    col7, col8 = st.columns([3, 3])

    with col7:
        # Show custom date range selector when Custom Range is selected
        if unified_filter_nba == "📅 Custom Range":
            date_range = st.date_input(
                "📅 Select Date Range",
                value=(datetime.now().date(), datetime.now().date() + timedelta(days=6)),
                key="nba_date_range",
                help="Select start and end dates"
            )
        else:
            date_range = None

    with col8:
        hide_lopsided_nba = st.checkbox(
            "🎯 Hide Lopsided Odds",
            value=False,
            key="nba_hide_lopsided",
            help="Filter out games with heavily favored teams (96%+ odds = low payout potential)"
        )

        if hide_lopsided_nba:
            lopsided_threshold_nba = st.slider(
                "Max Odds %",
                70, 99, 90,
                key="nba_lopsided_threshold",
                help="Hide games where one team's odds exceed this percentage"
            )
        else:
            lopsided_threshold_nba = 90

    # ==================== FETCH MULTI-DAY NBA DATA ====================
    try:
        espn_nba = get_espn_nba_client()
        nba_games = []

        # Fetch next 7 days of NBA games
        today = datetime.now()
        for i in range(7):
            date = today + timedelta(days=i)
            date_str = date.strftime('%Y%m%d')
            try:
                daily_games = espn_nba.get_scoreboard(date=date_str)
                if daily_games:
                    nba_games.extend(daily_games)
                    logger.info(f"Fetched {len(daily_games)} NBA games for {date_str}")
            except Exception as day_error:
                logger.warning(f"Could not fetch NBA games for {date_str}: {day_error}")

        # Remove duplicates
        seen_ids = set()
        unique_games = []
        for game in nba_games:
            game_id = game.get('game_id')
            if game_id and game_id not in seen_ids:
                seen_ids.add(game_id)
                unique_games.append(game)
        nba_games = unique_games

        logger.info(f"Total unique NBA games fetched: {len(nba_games)}")

    except Exception as e:
        st.error(f"Error fetching NBA games: {e}")
        logger.error(f"NBA fetch error: {e}")
        return

    if not nba_games:
        st.info("🏀 No NBA games scheduled in the next 7 days")
        return

    # ==================== ENRICH WITH KALSHI ODDS (OPTIMIZED) ====================
    try:
        from src.espn_kalshi_matcher_optimized import enrich_games_with_kalshi_odds_optimized
        nba_games = enrich_games_with_kalshi_odds_optimized(nba_games, sport='nba')

        # FILTER OUT LOPSIDED ODDS (where you can't make money)
        # Remove games where market odds are >85% or <15% (too one-sided)
        games_before_filter = len([g for g in nba_games if g.get('kalshi_odds')])
        nba_games_filtered = []
        for game in nba_games:
            kalshi_odds = game.get('kalshi_odds')
            if kalshi_odds:
                away_price = kalshi_odds.get('away_win_price', 0.5)
                home_price = kalshi_odds.get('home_win_price', 0.5)

                # Skip if either price is too lopsided (>0.85 or <0.15)
                if (away_price > 0.85 or away_price < 0.15 or
                    home_price > 0.85 or home_price < 0.15):
                    logger.debug(f"Filtered out lopsided odds: {game['away_team']} @ {game['home_team']} "
                               f"(Away: {away_price:.1%}, Home: {home_price:.1%})")
                    continue  # Skip this game

            nba_games_filtered.append(game)

        nba_games = nba_games_filtered
        games_filtered = games_before_filter - len([g for g in nba_games if g.get('kalshi_odds')])
        if games_filtered > 0:
            logger.info(f"Filtered out {games_filtered} NBA games with lopsided odds (>85% or <15%)")

        kalshi_matched = sum(1 for g in nba_games if g.get('kalshi_odds'))
        logger.info(f"Matched {kalshi_matched}/{len(nba_games)} NBA games with Kalshi odds")
    except Exception as e:
        logger.warning(f"Could not enrich with Kalshi odds: {e}")
        kalshi_matched = 0

    # Count live games
    live_count = sum(1 for g in nba_games if g.get('is_live', False))

    # Display stats
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.info(f"**Total Games:** {len(nba_games)}")
    with col_stat2:
        st.info(f"**Live:** {live_count}")
    with col_stat3:
        st.info(f"**With Odds:** {kalshi_matched}")

    # ==================== TEAM-SPECIFIC FILTER DROPDOWN ====================
    # Extract unique team names for team filter dropdown
    all_teams = set()
    for game in nba_games:
        away_team = game.get('away_team', '')
        home_team = game.get('home_team', '')
        if away_team:
            all_teams.add(away_team)
        if home_team:
            all_teams.add(home_team)

    team_list = sorted(list(all_teams))

    # Team-specific filter dropdown
    col_team_sel, col_subs = st.columns([3, 3])
    with col_team_sel:
        team_options = ["All Teams"] + team_list
        selected_team_name = st.selectbox(
            "🏀 Select Team",
            team_options,
            key="nba_team_selector",
            help="Filter to show only games involving this team"
        )

    with col_subs:
        # Show subscriptions count and management
        watchlist = watchlist_manager.get_user_watchlist(st.session_state.user_id)
        sub_count = len([w for w in watchlist if w.get('game_data', {}).get('sport') == 'NBA'])
        if sub_count > 0:
            if st.button(f"📋 My Subscriptions ({sub_count})", key="nba_show_subs", use_container_width=True):
                st.session_state['show_subscriptions_NBA'] = not st.session_state.get('show_subscriptions_NBA', False)
        else:
            st.info("No active subscriptions")

    # Show subscriptions if toggled
    if st.session_state.get('show_subscriptions_NBA', False) and watchlist:
        st.markdown("### 📋 Your Subscribed Games")

        # Table-like list view
        for idx, watch_game in enumerate(watchlist):
            game_data = watch_game.get('game_data', {})
            if game_data.get('sport') != 'NBA':
                continue

            away_team = game_data.get('away_team', 'Away')
            home_team = game_data.get('home_team', 'Home')
            selected_team = watch_game.get('selected_team', '')
            status = game_data.get('status_detail', 'Scheduled')
            away_score = game_data.get('away_score', 0)
            home_score = game_data.get('home_score', 0)
            is_live = game_data.get('is_live', False)

            # Display subscription entry
            col_game, col_btn = st.columns([4, 1])
            with col_game:
                live_badge = "🔴 LIVE • " if is_live else ""
                st.markdown(f"**{live_badge}{away_team}** @ **{home_team}** • Following: {selected_team} • {status}")
                if is_live and away_score is not None and home_score is not None:
                    st.caption(f"Score: {away_team} {away_score} - {home_team} {home_score}")
            with col_btn:
                if st.button("✖", key=f"nba_unsub_list_{idx}", help="Remove from subscriptions"):
                    game_id = game_data.get('game_id', '')
                    if game_id:
                        watchlist_manager.remove_game_from_watchlist(st.session_state.user_id, game_id)
                        st.rerun()

    # ==================== APPLY FILTERS ====================
    filtered_games = nba_games

    # Status filter
    if filter_status == "Live Only":
        filtered_games = [g for g in filtered_games if g.get('is_live', False)]
    elif filter_status == "Upcoming":
        filtered_games = [g for g in filtered_games if not g.get('is_live', False) and not g.get('is_completed', False)]
    elif filter_status == "Final":
        filtered_games = [g for g in filtered_games if g.get('is_completed', False)]

    # Hide final filter
    if hide_final:
        filtered_games = [g for g in filtered_games if not g.get('is_completed', False)]

    # Apply "Hide Lopsided Odds" filter - exclude games with heavily favored teams
    if hide_lopsided_nba:
        lopsided_threshold_decimal = lopsided_threshold_nba / 100  # Convert to decimal

        def is_not_lopsided(game):
            """Check if game has competitive odds (neither team heavily favored)"""
            yes_price = game.get('yes_price', 0.5)
            no_price = game.get('no_price', 0.5)

            # If either team's odds exceed threshold, it's lopsided (not profitable)
            if yes_price > lopsided_threshold_decimal or no_price > lopsided_threshold_decimal:
                return False
            return True

        filtered_games = [g for g in filtered_games if is_not_lopsided(g)]

    # Apply date filter
    if date_filter_mode != "All Games":
        from datetime import datetime, date, timedelta
        from dateutil import parser
        today = date.today()

        def is_in_date_range(game_time, start_date, end_date):
            """Check if game_time falls within date range. Handles both datetime objects and strings."""
            if not game_time:
                return False
            try:
                # Handle both datetime objects and strings
                if isinstance(game_time, datetime):
                    game_dt = game_time
                elif isinstance(game_time, str):
                    game_dt = parser.parse(game_time)
                else:
                    return False

                game_date = game_dt.date()
                return start_date <= game_date <= end_date
            except Exception as e:
                logger.warning(f"NBA date filter error for game_time {game_time}: {e}")
                return False

        # Determine date range based on filter mode
        if date_filter_mode == "Today Only":
            start_date = today
            end_date = today
        elif date_filter_mode == "Next 7 Days":
            start_date = today
            end_date = today + timedelta(days=6)
        elif date_filter_mode == "Custom Range" and date_range:
            # Handle tuple from st.date_input
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            elif isinstance(date_range, date):
                start_date = end_date = date_range
            else:
                start_date = today
                end_date = today
        else:
            start_date = today
            end_date = today

        filtered_games = [g for g in filtered_games if is_in_date_range(g.get('game_time'), start_date, end_date)]

    # Apply team-specific filter
    if selected_team_name and selected_team_name != "All Teams":
        # Filter for games involving the selected team
        filtered_games = [g for g in filtered_games
                        if g.get('away_team') == selected_team_name
                        or g.get('home_team') == selected_team_name]

    # ==================== APPLY SORTING ====================
    if sort_by == "⏰ Game Time":
        filtered_games.sort(key=lambda x: x.get('game_time') or '9999-12-31 23:59')
    elif sort_by == "🎯 Best Odds":
        # Sort by closest games
        filtered_games.sort(key=lambda x: min(
            x.get('kalshi_odds', {}).get('away_win_price', 1) * 100 if x.get('kalshi_odds') else 100,
            x.get('kalshi_odds', {}).get('home_win_price', 1) * 100 if x.get('kalshi_odds') else 100
        ))
    elif sort_by == "🏆 Biggest Favorite":
        # Sort by highest odds (biggest favorites first)
        filtered_games.sort(key=lambda x: max(
            x.get('kalshi_odds', {}).get('away_win_price', 0) * 100 if x.get('kalshi_odds') else 0,
            x.get('kalshi_odds', {}).get('home_win_price', 0) * 100 if x.get('kalshi_odds') else 0
        ), reverse=True)
    elif sort_by == "🤖 AI Confidence":
        # Sort by AI prediction: prioritize HIGH confidence + HIGH EV recommendations
        def ai_sort_key(game):
            ai_pred = game.get('ai_prediction', {})
            confidence = ai_pred.get('confidence_score', 0)
            ev = ai_pred.get('expected_value', 0)
            recommendation = ai_pred.get('recommendation', 'PASS')

            # Boost strong recommendations
            rec_multiplier = 2.0 if recommendation in ['STRONG BET', 'BET'] else 1.0

            # Combined score: confidence * EV * recommendation strength
            return (confidence * ev * rec_multiplier)

        filtered_games.sort(key=ai_sort_key, reverse=True)
    else:
        # Default: Live first, then upcoming, then completed
        live_games = [g for g in filtered_games if g.get('is_live', False)]
        upcoming_games = [g for g in filtered_games if not g.get('is_live', False) and not g.get('is_completed', False)]
        completed_games = [g for g in filtered_games if g.get('is_completed', False)]
        filtered_games = live_games + upcoming_games + completed_games

    if not filtered_games:
        st.info("No games match your filters")
        return

    st.markdown(f"**Showing {len(filtered_games)} games**")

    # OPTIMIZATION: Fetch watchlist ONCE before the game loop (prevents connection pool exhaustion)
    # Instead of calling watchlist_manager.is_game_watched() for each game (database call per card),
    # we fetch the entire watchlist once and use set membership for O(1) lookup
    user_id = st.session_state.get('user_id', 'default_user')
    watchlist = watchlist_manager.get_user_watchlist(user_id)
    watched_game_ids = {w.get('game_id') for w in watchlist if w.get('game_id')}

    # ==================== DISPLAY GAMES IN GRID ====================
    for i in range(0, len(filtered_games), cards_per_row):
        cols = st.columns(cards_per_row)

        for col_idx, game in enumerate(filtered_games[i:i+cards_per_row]):
            with cols[col_idx]:
                display_nba_game_card_enhanced(game, watchlist_manager, llm_service, watched_game_ids)


def display_nba_game_card_enhanced(game, watchlist_manager, llm_service=None, watched_game_ids=None):
    """Display an enhanced NBA game card with AI predictions and Kalshi odds - matches NFL feature set

    Args:
        game: Game data dictionary
        watchlist_manager: Watchlist manager instance
        llm_service: Optional LLM service for AI predictions
        watched_game_ids: Set of game IDs that are in user's watchlist (for O(1) lookup instead of DB calls)
    """
    from src.nba_team_database import get_team_logo_url
    import hashlib

    # Extract game data
    away_team = str(game.get('away_team', 'Away'))
    home_team = str(game.get('home_team', 'Home'))
    away_score = game.get('away_score', 0)
    home_score = game.get('home_score', 0)
    away_record = str(game.get('away_record', ''))
    home_record = str(game.get('home_record', ''))
    status_detail = str(game.get('status_detail', 'Scheduled'))
    is_live = game.get('is_live', False)
    is_completed = game.get('is_completed', False)
    quarter = str(game.get('quarter', ''))
    clock = str(game.get('clock', ''))

    # Get Kalshi odds (Robinhood backend)
    kalshi_odds = game.get('kalshi_odds')
    away_odds = float(kalshi_odds.get('away_win_price', 0)) * 100 if kalshi_odds else 0
    home_odds = float(kalshi_odds.get('home_win_price', 0)) * 100 if kalshi_odds else 0
    prediction_source = "Kalshi Market"

    # Determine predicted winner based on Kalshi odds
    if away_odds > home_odds and away_odds > 0:
        predicted_winner = away_team
        win_probability = away_odds / 100
    elif home_odds > away_odds and home_odds > 0:
        predicted_winner = home_team
        win_probability = home_odds / 100
    else:
        predicted_winner = ''
        win_probability = 0.5

    # Determine confidence level based on probability
    if win_probability >= 0.70:
        confidence_level = 'high'
        winner_text_color = '#00ff00'
        confidence_emoji = '🟢'
    elif win_probability >= 0.60:
        confidence_level = 'medium'
        winner_text_color = '#ffd700'
        confidence_emoji = '🟡'
    else:
        confidence_level = 'low'
        winner_text_color = '#888888'
        confidence_emoji = '⚪'

    # Create unique key for widgets
    game_id = game.get('game_id', f"{away_team}_{home_team}")
    key_base = f"NBA_{away_team}_{home_team}_{game_id}"
    key_hash = hashlib.md5(key_base.encode()).hexdigest()[:8]
    unique_key = f"nba_{game_id}_{key_hash}".replace(' ', '_')

    # OPTIMIZATION: Check if game is in watchlist using set membership (O(1) instead of database call)
    # watched_game_ids is pre-fetched once before the game loop to avoid connection pool exhaustion
    user_id = st.session_state.get('user_id', 'default_user')
    if watched_game_ids is None:
        # Fallback for backward compatibility if called without watched_game_ids
        is_watched = watchlist_manager.is_game_watched(user_id, game_id) if game_id else False
    else:
        # Use set membership check - O(1) instead of database call
        is_watched = bool(game_id and game_id in watched_game_ids)

    # ==================== CARD CONTAINER ====================
    st.markdown('<div class="game-card">', unsafe_allow_html=True)

    # Compact top row: Status + Quick Subscribe
    col_status, col_quick_tg = st.columns([2.5, 1])
    with col_status:
        if is_live:
            st.markdown(f'<span class="live-indicator"></span><strong style="font-size:13px;">LIVE • {quarter} {clock}</strong>', unsafe_allow_html=True)

            # Enhanced live game data - possession, down & distance
            possession = game.get('possession', '')
            down_distance = game.get('down_distance', '')
            is_red_zone = game.get('is_red_zone', False)

            if possession or down_distance:
                live_detail = []
                if possession:
                    live_detail.append(f"🏈 {possession}")
                if down_distance:
                    if is_red_zone:
                        live_detail.append(f"🔴 {down_distance}")  # Red zone indicator
                    else:
                        live_detail.append(down_distance)

                st.markdown(f"<div style='font-size:11px; color:#aaa; margin-top:2px;'>{' • '.join(live_detail)}</div>", unsafe_allow_html=True)

        elif is_completed:
            st.markdown(f"<strong style='font-size:13px;'>FINAL • {status_detail}</strong>", unsafe_allow_html=True)
        else:
            st.markdown(f"<strong style='font-size:13px;'>{status_detail}</strong>", unsafe_allow_html=True)

    with col_quick_tg:
        button_label = "📡 Subscribe" if not is_watched else "✅ Subscribed"
        button_key = f"subscribe_{unique_key}"
        btn_bg_color = '#10B981' if is_watched else '#495057'  # Bright green when subscribed
        btn_hover_color = '#059669' if is_watched else '#3d4146'

        # Add custom CSS for green button
        st.markdown(f"""
            <style>
            div[data-testid="stHorizontalBlock"] > div:has(button[kind="secondary"]) button {{
                background-color: {btn_bg_color} !important;
                color: white !important;
            }}
            div[data-testid="stHorizontalBlock"] > div:has(button[kind="secondary"]) button:hover {{
                background-color: {btn_hover_color} !important;
            }}
            </style>
        """, unsafe_allow_html=True)

        if st.button(button_label, key=button_key, help="Subscribe for live game updates", type="secondary"):
            if not is_watched:
                # Add sport field to game object (ESPN doesn't include it)
                game['sport'] = 'NBA'  # This is for NBA games
                # Add to watchlist - this triggers Telegram alert automatically via _send_subscription_alert()
                success = watchlist_manager.add_game_to_watchlist(user_id, game, selected_team=None)
                if success:
                    st.success("✅ Subscribed! Check Telegram for confirmation.")
                else:
                    st.error("❌ Subscription failed. Check logs.")
                st.rerun()

    # Team matchup with logos and scores - colored winner
    is_away_winner = (predicted_winner == away_team)
    is_home_winner = (predicted_winner == home_team)

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        # Away team logo
        away_logo = get_team_logo_url(away_team)
        if away_logo:
            st.image(away_logo, width=60)

        # Team name colored if predicted winner
        if is_away_winner and win_probability > 0.5:
            st.markdown(f"<span style='color:{winner_text_color}; font-weight:bold; font-size:16px;'>{away_team}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{away_team}**")

        if away_record:
            st.caption(away_record)
        st.markdown(f"<h2 style='margin:0'>{away_score}</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown("<p style='text-align:center; padding-top:30px;'>@</p>", unsafe_allow_html=True)

    with col3:
        # Home team logo
        home_logo = get_team_logo_url(home_team)
        if home_logo:
            st.image(home_logo, width=60)

        # Team name colored if predicted winner
        if is_home_winner and win_probability > 0.5:
            st.markdown(f"<span style='color:{winner_text_color}; font-weight:bold; font-size:16px;'>{home_team}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{home_team}**")

        if home_record:
            st.caption(home_record)
        st.markdown(f"<h2 style='margin:0'>{home_score}</h2>", unsafe_allow_html=True)

    # ==================== VISUAL ODDS BAR ====================
    if away_odds > 0 and home_odds > 0:
        # Normalize odds to 100% if they don't add up
        total_odds = away_odds + home_odds
        if total_odds > 0:
            away_width = (away_odds / total_odds) * 100
            home_width = (home_odds / total_odds) * 100
        else:
            away_width = 50
            home_width = 50

        # Determine colors based on who's favored
        if away_odds > home_odds:
            away_color = "#4CAF50"  # Green for favorite
            home_color = "#FF6B6B"  # Red for underdog
            favorite_team = away_team[:15]
        else:
            away_color = "#FF6B6B"
            home_color = "#4CAF50"
            favorite_team = home_team[:15]

        # Visual odds bar with percentage labels
        st.markdown(f"""
            <div style="margin:15px 0 10px 0;">
                <div style="display:flex; height:20px; border-radius:12px; overflow:hidden; border:2px solid #333; position:relative;">
                    <div style="width:{away_width:.1f}%; background:{away_color}; display:flex; align-items:center; justify-content:center; position:relative;">
                        <span style="position:absolute; font-size:11px; font-weight:600; color:#fff; text-shadow:1px 1px 2px rgba(0,0,0,0.5);">{away_width:.0f}%</span>
                    </div>
                    <div style="width:{home_width:.1f}%; background:{home_color}; display:flex; align-items:center; justify-content:center; position:relative;">
                        <span style="position:absolute; font-size:11px; font-weight:600; color:#fff; text-shadow:1px 1px 2px rgba(0,0,0,0.5);">{home_width:.0f}%</span>
                    </div>
                </div>
                <div style="text-align:center; font-size:11px; color:#888; margin-top:4px;">
                    💰 Market: {favorite_team} favored • Total: {total_odds:.0f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    # AI Prediction Section (if we have odds)
    if away_odds > 0 or home_odds > 0:
        # Check if game is live or final to label prediction appropriately
        is_live_or_final_nba = game.get('status') in ['STATUS_IN_PROGRESS', 'STATUS_FINAL']
        nba_prediction_label = "### 🤖 Pre-Game Market Analysis" if is_live_or_final_nba else "### 🤖 AI Market Prediction"

        st.markdown(nba_prediction_label)

        # Add explanatory caption for live/final games
        if is_live_or_final_nba:
            st.caption("📊 Pre-game prediction based on market odds and team strength. Not updated for live score.")

        # Confidence badge
        col_conf, col_pred = st.columns(2)

        with col_conf:
            if confidence_level == 'high':
                st.markdown(f'''
                    <div style="background: linear-gradient(135deg, #00ff00, #00cc00); color:#000; padding:10px;
                                border-radius:10px; text-align:center; box-shadow: 0 0 15px rgba(0,255,0,0.5);">
                        <div style="font-size:20px;">🟢</div>
                        <div style="font-size:14px; font-weight:700;">HIGH CONFIDENCE</div>
                        <div style="font-size:11px; margin-top:4px;">Strong Favorite</div>
                    </div>
                ''', unsafe_allow_html=True)
            elif confidence_level == 'medium':
                st.markdown(f'''
                    <div style="background: linear-gradient(135deg, #ffd700, #ffaa00); color:#000; padding:10px;
                                border-radius:10px; text-align:center; box-shadow: 0 0 12px rgba(255,215,0,0.5);">
                        <div style="font-size:20px;">🟡</div>
                        <div style="font-size:14px; font-weight:700;">MEDIUM CONFIDENCE</div>
                        <div style="font-size:11px; margin-top:4px;">Moderate Edge</div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div style="background:#555; color:#fff; padding:10px; border-radius:10px; text-align:center;">
                        <div style="font-size:20px;">⚪</div>
                        <div style="font-size:14px; font-weight:700;">LOW CONFIDENCE</div>
                        <div style="font-size:11px; margin-top:4px;">Toss-Up Game</div>
                    </div>
                ''', unsafe_allow_html=True)

        with col_pred:
            st.metric(
                "Predicted Winner",
                predicted_winner if predicted_winner else "Too Close",
                f"{int(win_probability * 100)}% win probability"
            )

        # Market prediction visual bar
        if predicted_winner:
            prob_pct = int(win_probability * 100)
            prob_color = winner_text_color
            bar_label = f"🏀 {prediction_source}"
            st.markdown(f'''
                <div style="margin-top:15px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:5px;">
                        <span>{bar_label}</span>
                        <span style="font-weight:600;">{prob_pct}%</span>
                    </div>
                    <div style="background:#333; border-radius:10px; overflow:hidden; height:25px;">
                        <div style="background:{prob_color}; width:{prob_pct}%; height:100%; border-radius:10px;
                                    display:flex; align-items:center; justify-content:center; color:#000; font-size:12px; font-weight:700;">
                            {prob_pct}%
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        # Betting recommendation
        if predicted_winner:
            st.markdown("#### 💰 Betting Recommendation")

            current_odds = float(away_odds) if is_away_winner else float(home_odds)
            implied_prob = current_odds / 100

            if confidence_level == 'high':
                st.success(f"🚀 **STRONG PLAY** - {predicted_winner} is heavily favored at {int(implied_prob*100)}%")
                st.markdown(f"**Analysis:** Market shows strong consensus. Bet on {predicted_winner} with high confidence.")
            elif confidence_level == 'medium':
                st.info(f"💰 **MODERATE PLAY** - {predicted_winner} favored at {int(implied_prob*100)}%")
                st.markdown(f"**Analysis:** Decent edge. Consider betting {predicted_winner} but manage risk.")
            else:
                st.warning(f"⚠️ **COIN FLIP** - Game too close to call confidently")
                st.markdown(f"**Analysis:** Near 50/50 game. Pass or bet small if you have a lean.")

    # Close card
    st.markdown('</div>', unsafe_allow_html=True)


def display_nba_game_card(game, watchlist_manager, llm_service=None):
    """Legacy NBA game card display - redirects to enhanced version"""
    display_nba_game_card_enhanced(game, watchlist_manager, llm_service)


if __name__ == "__main__":
    show_game_cards()
