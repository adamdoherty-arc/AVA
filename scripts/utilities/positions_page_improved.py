"""
Improved Positions Page with:
- Auto-refresh controls
- Color-coded P/L (green/red)
- TradingView chart links
- Complete trade history with P/L calculations
- Performance analytics by time period
- Database caching for fast load times
"""

import streamlit as st
import robin_stocks.robinhood as rh
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from src.trade_history_sync import TradeHistorySyncService
from src.theta_forecast_display import display_theta_forecasts
from src.services.rate_limiter import rate_limit

load_dotenv()

# PERFORMANCE FIX: Set global timeout for all external API calls (prevents page hangs)
import src.api_timeout_config  # Auto-configures 10-second timeout on import
from src.components.ai_research_widget import (
    display_consolidated_ai_research_section,
    display_quick_links_section
)
from src.components.expert_position_advisory import display_expert_position_advisory
from src.yfinance_utils import safe_get_history, safe_get_current_price
from src.recovery_strategies_tab import display_recovery_strategies_tab
from src.portfolio_balance_display import display_portfolio_balance_dashboard

# PERFORMANCE FIX: Rate-limited Robinhood API wrappers to prevent account bans
@rate_limit("robinhood", tokens=1, timeout=30)
def get_open_stock_positions_rate_limited():
    """Rate-limited wrapper for rh.get_open_stock_positions()"""
    return rh.get_open_stock_positions()

@rate_limit("robinhood", tokens=1, timeout=30)
def get_instrument_by_url_rate_limited(url):
    """Rate-limited wrapper for rh.get_instrument_by_url()"""
    return rh.get_instrument_by_url(url)

@rate_limit("robinhood", tokens=1, timeout=30)
def get_latest_price_rate_limited(symbol):
    """Rate-limited wrapper for rh.get_latest_price()"""
    return rh.get_latest_price(symbol)

@rate_limit("robinhood", tokens=1, timeout=30)
def get_open_option_positions_rate_limited():
    """Rate-limited wrapper for rh.get_open_option_positions()"""
    return rh.get_open_option_positions()

@rate_limit("robinhood", tokens=1, timeout=30)
def get_option_instrument_data_rate_limited(option_id):
    """Rate-limited wrapper for rh.get_option_instrument_data_by_id()"""
    return rh.get_option_instrument_data_by_id(option_id)

@rate_limit("robinhood", tokens=1, timeout=30)
def get_option_market_data_rate_limited(option_id):
    """Rate-limited wrapper for rh.get_option_market_data_by_id()"""
    return rh.get_option_market_data_by_id(option_id)

@rate_limit("robinhood", tokens=1, timeout=30)
def get_quotes_rate_limited(symbol):
    """Rate-limited wrapper for rh.get_quotes()"""
    return rh.get_quotes(symbol)
from src.auto_balance_recorder import AutoBalanceRecorder
from src.components.position_summary_dashboard import (
    display_position_summary,
    display_quick_glance_table,
    display_actionable_alerts,
    display_position_grouping_by_symbol
)

logger = logging.getLogger(__name__)


def display_news_section(symbols):
    """
    Display news section for all position symbols

    Args:
        symbols: List of stock ticker symbols to fetch news for
    """
    from src.news_service import NewsService

    if not symbols:
        return

    with st.expander("📰 Latest Market News", expanded=False):
        # Symbol selector
        selected_symbol = st.selectbox(
            "Select symbol for news:",
            options=symbols,
            key="news_symbol_selector"
        )

        if selected_symbol:
            news_service = NewsService()

            with st.spinner(f"Loading news for {selected_symbol}..."):
                news_articles = news_service.get_combined_news(selected_symbol)

            if news_articles:
                st.caption(f"Found {len(news_articles)} recent articles from Finnhub and Polygon APIs")

                for article in news_articles:
                    # Calculate time ago
                    from datetime import timezone
                    try:
                        # Use UTC now for comparison
                        if article.published_at.tzinfo:
                            time_diff = datetime.now(timezone.utc) - article.published_at
                        else:
                            time_diff = datetime.now(timezone.utc) - article.published_at.replace(tzinfo=timezone.utc)
                    except Exception as e:
                        time_diff = datetime.now(timezone.utc) - article.published_at

                    if time_diff.days > 0:
                        time_ago = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
                    elif time_diff.seconds // 3600 > 0:
                        hours = time_diff.seconds // 3600
                        time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
                    else:
                        minutes = time_diff.seconds // 60
                        time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"

                    # Display article
                    with st.expander(f"📄 {article.headline} ({time_ago})", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"**Source:** {article.source}")
                            if article.summary:
                                st.write(article.summary)
                        with col2:
                            st.link_button("Read Full Article", article.url, width='stretch')
            else:
                st.info(f"No recent news found for {selected_symbol}")


def ensure_rh_login():
    """Ensure Robinhood is logged in, re-login if session is lost"""
    username = os.getenv('ROBINHOOD_USERNAME')
    password = os.getenv('ROBINHOOD_PASSWORD')

    if not username or not password:
        logger.error("Robinhood credentials not found in environment variables")
        return False

    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Test if we're logged in
            rh.profiles.load_account_profile()
            return True
        except Exception as e:
            logger.warning(f"Session test failed (attempt {attempt + 1}): {e}")
            try:
                rh.logout()
            except:
                pass

            try:
                rh.login(username=username, password=password, store_session=True)
                logger.info("Re-logged in to Robinhood")
            except Exception as login_error:
                logger.error(f"Re-login failed: {login_error}")
                if attempt == max_retries - 1:
                    return False
    return False


def show_positions_page():
    """Main positions page with all improvements"""

    st.title("💼 Active Positions")
    st.caption("Live option positions from Robinhood with auto-refresh")

    # Auto-Refresh Controls and Navigation
    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-Refresh", value=False, key="pos_auto_refresh")
    with col2:
        refresh_freq = st.selectbox(
            "Frequency",
            ["30s", "1m", "2m", "5m", "10m"],
            index=2,
            key="pos_refresh_freq",
            label_visibility="collapsed"
        )
    with col3:
        if st.button("🔄 Refresh Now", type="primary"):
            st.rerun()
    with col4:
        if st.button("🔍 Find More Opportunities", type="secondary", help="Jump to Options Analysis to find new trades"):
            st.session_state.page = "Options Analysis"
            st.session_state.options_analysis_mode = "scan"
            st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        freq_map = {"30s": 30, "1m": 60, "2m": 120, "5m": 300, "10m": 600}
        st.markdown(
            f'<meta http-equiv="refresh" content="{freq_map[refresh_freq]}">',
            unsafe_allow_html=True
        )

    # === LOGIN TO ROBINHOOD (once for entire page) ===
    username = os.getenv('ROBINHOOD_USERNAME')
    password = os.getenv('ROBINHOOD_PASSWORD')

    if not username or not password:
        st.error("❌ Robinhood credentials not configured")
        st.info("Please set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in your .env file")
        return

    rh_session = None
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Force logout first to clear any stale session
            try:
                rh.logout()
            except:
                pass  # Ignore logout errors

            # Fresh login
            login_result = rh.login(username=username, password=password, store_session=True)

            # Verify we're actually logged in by testing the session
            test_profile = rh.profiles.load_account_profile()
            if test_profile:
                rh_session = rh  # Store the logged-in session
                logger.info("Successfully logged into Robinhood")
                break
            else:
                raise Exception("Login succeeded but session test failed")

        except Exception as e:
            retry_count += 1
            logger.error(f"Robinhood login attempt {retry_count} failed: {e}")

            if retry_count >= max_retries:
                st.error(f"❌ Failed to connect to Robinhood after {max_retries} attempts")
                st.error(f"Error: {str(e)}")
                st.info("💡 If you see a device approval prompt, check your Robinhood mobile app to approve this device")
                st.info("Then refresh this page")
                return

            # Wait before retrying
            import time
            time.sleep(2)

    if not rh_session:
        st.error("❌ Could not establish Robinhood connection")
        return

    # === ACTIVE POSITIONS ===
    st.markdown("### 🎯 Your Current Option Positions")

    # Initialize position lists at function scope for consolidated AI Research
    stock_positions_data = []
    csp_positions = []
    cc_positions = []
    long_call_positions = []
    long_put_positions = []

    try:
        # Ensure we're logged in before proceeding
        if not ensure_rh_login():
            st.error("❌ Could not establish or maintain Robinhood session")
            st.info("Please refresh the page to try again")
            return

        # Get total account value
        account_profile = rh.profiles.load_account_profile()
        portfolio = rh.profiles.load_portfolio_profile()
        total_equity = float(portfolio.get('equity', 0)) if portfolio else 0

        # Get buying power early
        try:
            buying_power = float(account_profile.get('buying_power', 0))
        except:
            buying_power = 0

        # === STOCK POSITIONS ===
        try:
            # PERFORMANCE: Rate-limited API call
            stock_positions_raw = get_open_stock_positions_rate_limited()
            stock_positions_data = []

            for stock_pos in stock_positions_raw:
                quantity = float(stock_pos.get('quantity', 0))
                if quantity == 0:
                    continue

                # Get instrument URL and extract symbol
                instrument_url = stock_pos.get('instrument')
                if instrument_url:
                    # PERFORMANCE: Rate-limited API call
                    instrument_data = get_instrument_by_url_rate_limited(instrument_url)
                    symbol = instrument_data.get('symbol', 'Unknown')
                else:
                    continue

                # Get average buy price
                avg_buy_price = float(stock_pos.get('average_buy_price', 0))

                # Get current stock price
                try:
                    # PERFORMANCE: Rate-limited API call
                    stock_quote = get_latest_price_rate_limited(symbol)
                    current_price = float(stock_quote[0]) if stock_quote else 0
                except:
                    current_price = 0

                # Calculate values
                cost_basis = avg_buy_price * quantity
                current_value = current_price * quantity
                pl = current_value - cost_basis
                pl_pct = (pl / cost_basis * 100) if cost_basis > 0 else 0

                # TradingView link
                tv_link = f"https://www.tradingview.com/chart/?symbol={symbol}"

                stock_positions_data.append({
                    'Symbol': symbol,
                    'Shares': int(quantity),
                    'Avg Buy Price': avg_buy_price,
                    'Current Price': current_price,
                    'Cost Basis': cost_basis,
                    'Current Value': current_value,
                    'P/L': pl,
                    'P/L %': pl_pct,
                    'Chart': tv_link,
                    'symbol_raw': symbol,  # For AI research lookup
                    'pl_raw': pl
                })

            # Display stock positions if any
            if stock_positions_data:
                with st.expander(f"📊 Stock Positions ({len(stock_positions_data)})", expanded=True):
                    # Add refresh button for stock positions
                    col_refresh1, col_refresh2 = st.columns([5, 1])
                    with col_refresh1:
                        st.caption(f"{len(stock_positions_data)} stock position{'s' if len(stock_positions_data) > 1 else ''}")
                    with col_refresh2:
                        if st.button("🔄", key="refresh_stock_positions", help="Refresh stock positions data"):
                            # Clear any cached stock data
                            if "stock_positions_cache" in st.session_state:
                                del st.session_state["stock_positions_cache"]
                            st.rerun()

                    df_stocks = pd.DataFrame(stock_positions_data)

                    # Format display
                    display_stocks = df_stocks.copy()
                    display_stocks['Avg Buy Price'] = display_stocks['Avg Buy Price'].apply(lambda x: f'${x:.2f}')
                    display_stocks['Current Price'] = display_stocks['Current Price'].apply(lambda x: f'${x:.2f}')
                    display_stocks['Cost Basis'] = display_stocks['Cost Basis'].apply(lambda x: f'${x:,.2f}')
                    display_stocks['Current Value'] = display_stocks['Current Value'].apply(lambda x: f'${x:,.2f}')

                    # Store raw P/L for coloring
                    stock_pl_vals = display_stocks['P/L'].copy()

                    display_stocks['P/L'] = display_stocks['P/L'].apply(lambda x: f'${x:,.2f}')
                    display_stocks['P/L %'] = display_stocks['P/L %'].apply(lambda x: f'{x:.1f}%')
                    display_stocks = display_stocks.drop(columns=['pl_raw', 'symbol_raw'])

                    # Color coding function
                    def highlight_stock_pl(row):
                        idx = row.name
                        pl_val = stock_pl_vals.iloc[idx] if idx < len(stock_pl_vals) else 0
                        styles = [''] * len(row)
                        pl_idx = list(display_stocks.columns).index('P/L')
                        pl_pct_idx = list(display_stocks.columns).index('P/L %')
                        if pl_val > 0:
                            styles[pl_idx] = 'color: #00AA00; font-weight: bold'
                            styles[pl_pct_idx] = 'color: #00AA00; font-weight: bold'
                        elif pl_val < 0:
                            styles[pl_idx] = 'color: #DD0000; font-weight: bold'
                            styles[pl_pct_idx] = 'color: #DD0000; font-weight: bold'
                        return styles

                    styled_stocks = display_stocks.style.apply(highlight_stock_pl, axis=1)

                    st.dataframe(
                        styled_stocks,
                        hide_index=True,
                        width='stretch',
                        column_config={
                            "Chart": st.column_config.LinkColumn(
                                "Chart",
                                help="Click to view TradingView chart",
                                display_text="📈"
                            )
                        },
                        key="stock_positions_table"
                    )

        except Exception as e:
            st.warning(f"Could not load stock positions: {e}")

        # Get all open option positions
        # PERFORMANCE: Rate-limited API call
        positions_raw = get_open_option_positions_rate_limited()

        if positions_raw:
            positions_data = []

            for pos in positions_raw:
                # Get option details
                opt_id = pos.get('option_id')
                if not opt_id:
                    continue

                # PERFORMANCE: Rate-limited API call
                opt_data = get_option_instrument_data_rate_limited(opt_id)
                symbol = opt_data.get('chain_symbol', 'Unknown')
                strike = float(opt_data.get('strike_price', 0))
                exp_date = opt_data.get('expiration_date', 'Unknown')
                opt_type = opt_data.get('type', 'unknown')

                # Position details
                position_type = pos.get('type', 'unknown')
                quantity = float(pos.get('quantity', 0))
                avg_price = abs(float(pos.get('average_price', 0)))

                # Calculate values
                total_premium = avg_price * quantity

                # Get current market price
                # adjusted_mark_price is in dollars per share, need to multiply by 100 (shares per contract)
                # PERFORMANCE: Rate-limited API call
                market_data = get_option_market_data_rate_limited(opt_id)
                if market_data and len(market_data) > 0:
                    current_price_per_share = float(market_data[0].get('adjusted_mark_price', 0))  # dollars per share
                    current_value = current_price_per_share * 100 * quantity  # dollars total
                else:
                    current_price_per_share = 0
                    current_value = 0

                # Calculate P/L
                if position_type == 'short':
                    pl = total_premium - current_value
                else:
                    pl = current_value - total_premium

                # Calculate DTE
                if exp_date != 'Unknown':
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt - datetime.now()).days
                else:
                    dte = 0

                # Determine strategy type
                if position_type == 'short' and opt_type == 'put':
                    strategy = 'CSP'
                elif position_type == 'short' and opt_type == 'call':
                    strategy = 'CC'
                elif position_type == 'long' and opt_type == 'call':
                    strategy = 'Long Call'
                elif position_type == 'long' and opt_type == 'put':
                    strategy = 'Long Put'
                else:
                    strategy = 'Other'

                # Get stock prices (regular close and after-hours)
                stock_price = 0  # Regular market close
                after_hours_price = None  # Current after-hours price

                try:
                    # Use get_quotes which includes both regular and extended hours data
                    # PERFORMANCE: Rate-limited API call
                    quote_data = get_quotes_rate_limited(symbol)
                    if quote_data and len(quote_data) > 0:
                        quote = quote_data[0]

                        # Get regular hours close price (last trade during regular hours)
                        last_trade = quote.get('last_trade_price')
                        if last_trade:
                            stock_price = float(last_trade)

                        # Get extended hours price - always show if available
                        extended_price = quote.get('last_extended_hours_trade_price')
                        if extended_price:
                            after_hours_price = float(extended_price)
                except Exception as e:
                    logger.debug(f"Could not get prices for {symbol}: {e}")
                    # Fallback to get_latest_price
                    try:
                        # PERFORMANCE: Rate-limited API call
                        stock_quote = get_latest_price_rate_limited(symbol)
                        stock_price = float(stock_quote[0]) if stock_quote else 0
                    except:
                        stock_price = 0

                # Create TradingView link (will display as clickable icon)
                tv_link = f"https://www.tradingview.com/chart/?symbol={symbol}"

                # Store per-contract prices for better display (in dollars)
                per_contract_cost = avg_price  # This is already per-contract in dollars
                per_contract_current = current_price_per_share * 100  # Convert per-share to per-contract

                # Calculate capital secured for CSPs (money that must be held in reserve)
                capital_secured = None
                if position_type == 'short' and opt_type == 'put':
                    # For CSPs: strike price * 100 * number of contracts
                    capital_secured = strike * 100 * quantity

                # Calculate estimated after-hours option value
                after_hours_value = None
                after_hours_pl = None
                if after_hours_price and stock_price and stock_price > 0:
                    # Calculate stock price change
                    stock_change = after_hours_price - stock_price

                    # Estimate delta based on moneyness
                    # Delta represents how much option price changes per $1 change in stock
                    if opt_type == 'put':
                        # For puts: OTM (stock > strike) = lower delta, ITM (stock < strike) = higher delta
                        if stock_price > strike:
                            # OTM put - delta around -0.30
                            estimated_delta = -0.30
                        elif stock_price < strike:
                            # ITM put - delta around -0.70
                            estimated_delta = -0.70
                        else:
                            # ATM put - delta around -0.50
                            estimated_delta = -0.50

                        # For puts, delta is negative (stock up = put value down)
                        option_price_change = estimated_delta * stock_change * 100  # dollars per contract (100 shares)
                    else:  # call
                        # For calls: OTM (stock < strike) = lower delta, ITM (stock > strike) = higher delta
                        if stock_price < strike:
                            # OTM call - delta around 0.30
                            estimated_delta = 0.30
                        elif stock_price > strike:
                            # ITM call - delta around 0.70
                            estimated_delta = 0.70
                        else:
                            # ATM call - delta around 0.50
                            estimated_delta = 0.50

                        # For calls, delta is positive (stock up = call value up)
                        option_price_change = estimated_delta * stock_change * 100  # dollars per contract (100 shares)

                    # Calculate after-hours option value
                    # option_price_change is per contract in dollars, multiply by quantity for total dollars
                    after_hours_value = max(0, current_value + (option_price_change * quantity))

                    # Calculate P/L at after-hours prices
                    if position_type == 'short':
                        after_hours_pl = total_premium - after_hours_value
                    else:
                        after_hours_pl = after_hours_value - total_premium

                positions_data.append({
                    'Symbol': symbol,
                    'Stock Price': stock_price,
                    'After-Hours': after_hours_price,
                    'Strategy': strategy,
                    'Strike': strike,
                    'Expiration': exp_date,
                    'DTE': dte,
                    'Contracts': int(quantity),
                    'Premium': total_premium,  # Total premium collected/paid
                    'Value': current_value,  # Current market value
                    'Capital Secured': capital_secured,  # NEW: For CSPs only
                    'AH Value': after_hours_value,  # After-hours estimated value
                    'P/L': pl,
                    'AH P/L': after_hours_pl,  # After-hours P/L
                    'P/L %': (pl/total_premium*100) if total_premium > 0 else 0,
                    'Chart': tv_link,  # Plain URL - will be styled by column_config
                    'symbol_raw': symbol,  # For AI research lookup
                    'pl_raw': pl  # For color coding
                })

            if positions_data:
                # Get buying power
                try:
                    account_profile = rh.profiles.load_account_profile()
                    buying_power = float(account_profile.get('buying_power', 0))
                except:
                    buying_power = 0

                # Display metrics
                total_pl = sum([p['P/L'] for p in positions_data])
                total_premium = sum([p['Premium'] for p in positions_data])

                # Calculate after-hours totals
                total_ah_pl = sum([p.get('AH P/L', p['P/L']) for p in positions_data if p.get('AH P/L') is not None])
                has_ah_data = any(p.get('AH Value') is not None for p in positions_data)

                # Calculate after-hours account value
                # The account value changes by the P/L change (after-hours P/L - regular P/L)
                if has_ah_data:
                    total_pl_change = total_ah_pl - total_pl
                    ah_account_value = total_equity + total_pl_change
                else:
                    ah_account_value = None

                # Auto-record daily balance (once per day)
                try:
                    auto_recorder = AutoBalanceRecorder()
                    options_val = sum([p['Value'] for p in positions_data])
                    stock_val = sum([p['Current Value'] for p in stock_positions_data]) if stock_positions_data else 0

                    result = auto_recorder.auto_record_balance(
                        total_equity=total_equity,
                        buying_power=buying_power,
                        options_value=options_val,
                        stock_value=stock_val,
                        total_positions=len(positions_data)
                    )

                    # Store status for display
                    balance_status = auto_recorder.get_recording_status()
                except Exception as e:
                    logger.error(f"Error auto-recording balance: {e}")
                    balance_status = "⚠️ Auto-record error"

                # === NEW: POSITION SUMMARY DASHBOARD ===
                display_position_summary(
                    stock_positions=stock_positions_data,
                    option_positions=positions_data,
                    total_equity=total_equity,
                    buying_power=buying_power
                )

                # === NEW: ACTIONABLE ALERTS ===
                display_actionable_alerts(
                    stock_positions=stock_positions_data,
                    option_positions=positions_data
                )

                # === NEW: QUICK GLANCE TABLE ===
                display_quick_glance_table(
                    stock_positions=stock_positions_data,
                    option_positions=positions_data
                )

                # Display balance recording status
                st.caption(f"📊 {balance_status}")

                # Separate positions by strategy type
                csp_positions = [p for p in positions_data if p['Strategy'] == 'CSP']
                cc_positions = [p for p in positions_data if p['Strategy'] == 'CC']
                long_call_positions = [p for p in positions_data if p['Strategy'] == 'Long Call']
                long_put_positions = [p for p in positions_data if p['Strategy'] == 'Long Put']

                # Helper function to display a strategy table
                def display_strategy_table(title, emoji, positions, section_key, expanded=False):
                    """Display positions table for a specific strategy"""
                    if not positions:
                        return

                    with st.expander(f"{emoji} {title} ({len(positions)})", expanded=expanded):
                        # Add refresh button for this specific table
                        col_refresh1, col_refresh2 = st.columns([5, 1])
                        with col_refresh1:
                            st.caption(f"{len(positions)} active position{'s' if len(positions) > 1 else ''}")
                        with col_refresh2:
                            if st.button("🔄", key=f"refresh_{section_key}", help=f"Refresh {title} data"):
                                # Clear any cached data for this section
                                cache_key = f"positions_cache_{section_key}"
                                if cache_key in st.session_state:
                                    del st.session_state[cache_key]
                                st.rerun()

                        df = pd.DataFrame(positions)

                        # Format display columns
                        display_df = df.copy()

                        # Rename Contracts to # to save space
                        display_df = display_df.rename(columns={'Contracts': '#'})

                        display_df['Stock Price'] = display_df['Stock Price'].apply(lambda x: f'${x:.2f}')
                        # Handle After-Hours with proper NaN/None checking
                        display_df['After-Hours'] = display_df['After-Hours'].apply(
                            lambda x: f'${x:.2f}' if (x is not None and pd.notna(x)) else '-'
                        )
                        display_df['Strike'] = display_df['Strike'].apply(lambda x: f'${x:.2f}')

                        # Format value columns
                        display_df['Premium'] = display_df['Premium'].apply(lambda x: f'${x:,.2f}')
                        display_df['Value'] = display_df['Value'].apply(lambda x: f'${x:,.2f}')

                        # Format Capital Secured for CSPs (will be None for other strategies)
                        if 'Capital Secured' in display_df.columns:
                            display_df['Capital Secured'] = display_df['Capital Secured'].apply(
                                lambda x: f'${x:,.0f}' if (x is not None and pd.notna(x)) else '-'
                            )

                        # Store raw P/L values for coloring before formatting
                        pl_vals = display_df['P/L'].copy()

                        # Store raw after-hours P/L for coloring if it exists
                        ah_pl_vals = None
                        if 'AH P/L' in display_df.columns:
                            ah_pl_vals = display_df['AH P/L'].copy()

                        # Format after-hours columns if they exist
                        if 'AH Value' in display_df.columns:
                            display_df['AH Value'] = display_df['AH Value'].apply(
                                lambda x: f'${x:,.2f}' if (x is not None and pd.notna(x)) else '-'
                            )
                        if 'AH P/L' in display_df.columns:
                            display_df['AH P/L'] = display_df['AH P/L'].apply(
                                lambda x: f'${x:,.2f}' if (x is not None and pd.notna(x)) else '-'
                            )

                        display_df['P/L'] = display_df['P/L'].apply(lambda x: f'${x:,.2f}')
                        display_df['P/L %'] = display_df['P/L %'].apply(lambda x: f'{x:.1f}%')

                        # Drop helper columns
                        display_df = display_df.drop(columns=['pl_raw', 'symbol_raw'])

                        # Function to apply text color
                        def highlight_pl(row):
                            """Apply row-wise styling based on P/L value"""
                            idx = row.name
                            pl_val = pl_vals.iloc[idx] if idx < len(pl_vals) else 0

                            styles = [''] * len(row)

                            # Find P/L and P/L % column indices
                            pl_idx = list(display_df.columns).index('P/L')
                            pl_pct_idx = list(display_df.columns).index('P/L %')

                            # Profit (positive P/L) = GREEN TEXT, Loss (negative P/L) = RED TEXT
                            if pl_val > 0:
                                styles[pl_idx] = 'color: #00AA00; font-weight: bold'  # Green text for profit
                                styles[pl_pct_idx] = 'color: #00AA00; font-weight: bold'
                            elif pl_val < 0:
                                styles[pl_idx] = 'color: #DD0000; font-weight: bold'  # Red text for loss
                                styles[pl_pct_idx] = 'color: #DD0000; font-weight: bold'
                            # else pl_val == 0, no styling (neutral)

                            # Color after-hours P/L if column exists
                            if ah_pl_vals is not None and 'AH P/L' in display_df.columns:
                                try:
                                    ah_pl_idx = list(display_df.columns).index('AH P/L')
                                    if idx < len(ah_pl_vals):
                                        ah_pl_val = ah_pl_vals.iloc[idx]
                                        if pd.notna(ah_pl_val):
                                            if ah_pl_val > 0:
                                                styles[ah_pl_idx] = 'color: #00AA00; font-weight: bold'
                                            elif ah_pl_val < 0:
                                                styles[ah_pl_idx] = 'color: #DD0000; font-weight: bold'
                                except:
                                    pass

                            return styles

                        # Apply styling
                        styled_df = display_df.style.apply(highlight_pl, axis=1)

                        # Display table
                        st.dataframe(
                            styled_df,
                            hide_index=True,
                            width='stretch',
                            column_config={
                                "Chart": st.column_config.LinkColumn(
                                    "Chart",
                                    help="Click to view TradingView chart",
                                    display_text="📈"
                                )
                            },
                            key=f"positions_table_{section_key}"
                        )

                        # Add Quick Actions - Analyze buttons for each position
                        st.markdown("**Quick Actions:**")
                        action_cols = st.columns(min(len(positions), 5))  # Max 5 columns
                        for idx, position in enumerate(positions):
                            col_idx = idx % 5  # Wrap to next row if more than 5
                            with action_cols[col_idx]:
                                symbol = position.get('symbol_raw', position.get('Symbol', 'N/A'))
                                if st.button(
                                    f"🔍 {symbol}",
                                    key=f"analyze_pos_{section_key}_{idx}",
                                    help=f"Analyze {symbol} in Options Analysis page",
                                    use_container_width=True
                                ):
                                    # Store position data in session state
                                    st.session_state.options_analysis_symbol = symbol
                                    st.session_state.options_analysis_mode = "position"
                                    st.session_state.options_analysis_position = position
                                    # Navigate to Options Analysis page
                                    st.session_state.page = "Options Analysis"
                                    st.rerun()

                # Display each strategy section (smart defaults: expand if has positions)
                display_strategy_table("Cash-Secured Puts", "💰", csp_positions, "csp", expanded=len(csp_positions) > 0)
                display_strategy_table("Covered Calls", "📞", cc_positions, "cc", expanded=len(cc_positions) > 0)
                display_strategy_table("Long Calls", "📈", long_call_positions, "long_calls", expanded=len(long_call_positions) > 0)
                display_strategy_table("Long Puts", "📉", long_put_positions, "long_puts", expanded=len(long_put_positions) > 0)

                # Display Next CSP Opportunities (30-day, ~0.3 delta)
                if csp_positions:
                    from src.csp_opportunities_finder import CSPOpportunitiesFinder

                    with st.expander("🎯 Next CSP Opportunities (30 Days, Delta ~0.3)", expanded=False):
                        st.caption("Next optimal CSP trades for your current position symbols")

                        try:
                            finder = CSPOpportunitiesFinder()
                            opportunities_df = finder.find_opportunities_for_current_positions(csp_positions)

                            if not opportunities_df.empty:
                                # Show summary metrics
                                metrics = finder.get_summary_metrics(opportunities_df)
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Opportunities", metrics['total_opportunities'])
                                with col2:
                                    st.metric("Avg Premium", f"${metrics['avg_premium']:.2f}")
                                with col3:
                                    st.metric("Avg Monthly %", f"{metrics['avg_monthly_return']:.2f}%")
                                with col4:
                                    st.metric("Total Premium", f"${metrics['total_premium']:.2f}")

                                # Display opportunities table (TradingView watchlist style)
                                st.markdown("#### 📊 30-Day CSP Opportunities (Click headers to sort)")
                                st.dataframe(
                                    opportunities_df[['Symbol', 'Stock Price', 'Strike', 'DTE', 'Premium',
                                                     'Delta', 'Monthly %', 'Annual %', 'IV', 'Breakeven',
                                                     'Bid', 'Ask', 'Volume', 'OI']],
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
                                        "Annual %": st.column_config.NumberColumn("Annual %", format="%.1f%%"),
                                        "IV": st.column_config.NumberColumn("IV %", format="%.1f%%"),
                                        "Breakeven": st.column_config.NumberColumn("Breakeven", format="$%.2f"),
                                        "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                                        "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                                        "Volume": st.column_config.NumberColumn("Vol"),
                                        "OI": st.column_config.NumberColumn("OI")
                                    }
                                )

                                st.caption("💡 Tip: Click any symbol to open in Robinhood options chain")
                            else:
                                st.info("⏳ No 30-day opportunities found. Options data may need syncing for these symbols.")
                                st.caption("Add these symbols to a TradingView watchlist and sync to get options data.")

                        except Exception as e:
                            st.error(f"Error loading CSP opportunities: {e}")
                            logger.error(f"CSP opportunities error: {e}")

                # === PORTFOLIO BALANCE HISTORY ===
                display_portfolio_balance_dashboard(days_back=90, expanded=False)

                # Display Theta Decay Forecasts for CSP, Covered Calls, and Long Call positions
                if csp_positions or cc_positions or long_call_positions:
                    with st.expander("📉 Theta Decay Forecasts", expanded=False):
                        st.caption("Day-by-day profit projections showing how much premium you'll earn/lose as time passes")

                        # Build list of available position types
                        available_types = []
                        if csp_positions:
                            available_types.append("Cash-Secured Puts")
                        if cc_positions:
                            available_types.append("Covered Calls")
                        if long_call_positions:
                            available_types.append("Long Calls")

                        # Add option to select position type
                        position_type = st.radio(
                            "Select Position Type:",
                            available_types,
                            horizontal=True,
                            key="theta_position_type"
                        )

                        if position_type == "Cash-Secured Puts" and csp_positions:
                            display_theta_forecasts(csp_positions)
                        elif position_type == "Covered Calls" and cc_positions:
                            display_theta_forecasts(cc_positions)
                        elif position_type == "Long Calls" and long_call_positions:
                            display_theta_forecasts(long_call_positions)
                        else:
                            st.info(f"No {position_type} positions to display")

                # === POSITION GROUPING BY SYMBOL ===
                display_position_grouping_by_symbol(
                    stock_positions=stock_positions_data,
                    option_positions=positions_data
                )

                # === EXPERT POSITION ADVISORY ===
                # Comprehensive AI-powered trade analysis for all positions
                with st.expander("💼 Expert Position Advisory - AI Trade Analysis", expanded=False):
                    st.markdown("""
**Get AI-powered analysis for any position:**
- Current position assessment with risk level
- Scenario analysis (Hold, Close, or Roll)
- Expert recommendation with clear reasoning
- Risk management guidance
                    """)

                    # Display advisory for all positions combined
                    all_positions = csp_positions + cc_positions + long_call_positions + long_put_positions
                    if all_positions:
                        display_expert_position_advisory(all_positions)
                    else:
                        st.info("No positions available for analysis")

                # === RECOVERY STRATEGIES TAB ===
                # Check if there are any losing CSP positions
                losing_csp_positions = []
                for pos in csp_positions:
                    if pos.get('pl_raw', 0) < 0:
                        # Add additional data needed for recovery analysis
                        pos['symbol'] = pos.get('Symbol', pos.get('symbol_raw', ''))  # Fix: add lowercase 'symbol' key
                        pos['option_type'] = 'put'
                        pos['position_type'] = 'short'

                        # Calculate current price per contract from Value (Value / quantity / 100)
                        value = pos.get('Value', 0)
                        if isinstance(value, str):
                            value = float(value.replace('$', '').replace(',', ''))
                        quantity = abs(pos.get('Contracts', 1))
                        pos['current_price'] = value / quantity / 100 if quantity > 0 else 0

                        pos['strike_price'] = pos.get('Strike', 0)  # Fix: use 'strike_price' not 'current_strike'
                        pos['current_strike'] = pos.get('Strike', 0)
                        if isinstance(pos['current_strike'], str):
                            pos['current_strike'] = float(pos['current_strike'].replace('$', '').replace(',', ''))
                        if isinstance(pos['strike_price'], str):
                            pos['strike_price'] = float(pos['strike_price'].replace('$', '').replace(',', ''))
                        pos['current_loss'] = abs(pos.get('pl_raw', 0))
                        pos['loss_percentage'] = abs(pos.get('P/L %', 0))
                        if isinstance(pos['loss_percentage'], str):
                            pos['loss_percentage'] = float(pos['loss_percentage'].replace('%', ''))
                        pos['days_to_expiry'] = pos.get('DTE', 0)
                        pos['expiration_date'] = pos.get('Expiration', '')  # Fix: use 'expiration_date' not 'expiration'
                        pos['expiration'] = pos.get('Expiration', '')
                        pos['quantity'] = quantity

                        premium = pos.get('Premium', 0)
                        if isinstance(premium, str):
                            premium = float(premium.replace('$', '').replace(',', ''))
                        # Premium is total for all contracts, convert to per-contract
                        pos['average_price'] = abs(premium) / quantity if quantity > 0 and premium else 0
                        pos['premium_collected'] = abs(premium)
                        losing_csp_positions.append(pos)

                if losing_csp_positions:
                    with st.expander("🎯 Recovery Strategies", expanded=False):
                        display_recovery_strategies_tab(losing_csp_positions, rh_session)

            else:
                st.info("No open option positions found in Robinhood")

        else:
            st.info("No open option positions found")

    except Exception as e:
        st.error(f"Error loading positions: {e}")

    # === CONSOLIDATED AI RESEARCH & QUICK LINKS ===
    # Display consolidated AI Research for all positions
    try:
        all_symbols = []

        # Collect from stock positions
        if stock_positions_data:
            all_symbols.extend([p['symbol_raw'] for p in stock_positions_data])

        # Collect from option positions
        for positions_list in [csp_positions, cc_positions, long_call_positions, long_put_positions]:
            if positions_list:
                all_symbols.extend([p.get('symbol_raw') for p in positions_list])

        if all_symbols:
            display_consolidated_ai_research_section(all_symbols, key_prefix="positions", use_expander=True, expanded=False)
            display_quick_links_section(all_symbols, key_prefix="positions", use_expander=True, expanded=False)

    except Exception as e:
        st.warning(f"Could not load AI Research: {e}")

    # === NEWS SECTION ===
    # Collect all unique symbols for news display
    all_news_symbols = set()
    for pos in stock_positions_data:
        all_news_symbols.add(pos['symbol_raw'])
    for pos in csp_positions:
        all_news_symbols.add(pos['symbol_raw'])
    for pos in cc_positions:
        all_news_symbols.add(pos['symbol_raw'])
    for pos in long_call_positions:
        all_news_symbols.add(pos['symbol_raw'])
    for pos in long_put_positions:
        all_news_symbols.add(pos['symbol_raw'])

    if all_news_symbols:
        try:
            display_news_section(sorted(list(all_news_symbols)))
        except Exception as e:
            st.warning(f"Could not load news section: {e}")

    # === TRADE HISTORY ===
    # Initialize sync service to get count for expander title
    sync_service_temp = TradeHistorySyncService()
    db_trades_temp = sync_service_temp.get_closed_trades_from_db(days_back=365)
    trade_count = len(db_trades_temp) if db_trades_temp else 0
    last_sync_temp = sync_service_temp.get_last_sync_time()
    sync_info = f" - Last sync: {last_sync_temp.strftime('%I:%M %p')}" if last_sync_temp else ""

    closed_trades = []  # Initialize outside for use in performance analytics

    with st.expander(f"📊 Trade History ({trade_count} trades{sync_info})", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("Closed trades with P/L calculations (loaded from database)")
        with col2:
            if st.button("🔄 Sync Now", key="sync_trades"):
                with st.spinner("Syncing trades from Robinhood..."):
                    try:
                        count = sync_service_temp.sync_trades_from_robinhood(rh_session)
                        st.success(f"✅ Synced {count} new trades")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sync failed: {e}")

        try:
            # Use fast database query instead of slow Robinhood API
            sync_service = TradeHistorySyncService()
            db_trades = sync_service.get_closed_trades_from_db(days_back=365)

            # Convert to format expected by display code
            from datetime import timezone
            closed_trades = []
            for trade in db_trades:
                # Convert close_date to timezone-aware timestamp for performance analytics
                if trade['close_date']:
                    if trade['close_date'].tzinfo is None:
                        close_timestamp = trade['close_date'].replace(tzinfo=timezone.utc)
                    else:
                        close_timestamp = trade['close_date']
                else:
                    close_timestamp = datetime.now(timezone.utc)

                closed_trades.append({
                    'Symbol': trade['symbol'],
                    'Strategy': 'CSP' if trade['strategy_type'] == 'cash_secured_put' else 'CC',
                    'Strike': trade['strike'],
                    'Open Premium': trade['premium_collected'],
                    'Close Cost': trade['close_price'],
                    'P/L': trade['profit_loss'],
                    'P/L %': (trade['profit_loss'] / trade['premium_collected'] * 100) if trade['premium_collected'] > 0 else 0,
                    'Days Held': trade['days_held'],
                    'Close Date': trade['close_date'].strftime('%Y-%m-%d') if trade['close_date'] else 'N/A',
                    'close_timestamp': close_timestamp  # For performance analytics filtering
                })

            # Fetch current stock prices for after-hours display
            if closed_trades:
                unique_symbols = list(set([t['Symbol'] for t in closed_trades]))
                current_prices = {}

                with st.spinner("Fetching current stock prices..."):
                    for symbol in unique_symbols:
                        # Use safe wrapper to handle delisted symbols gracefully
                        price = safe_get_current_price(symbol, suppress_warnings=True)
                        current_prices[symbol] = price

                # Add current prices to trades
                for trade in closed_trades:
                    trade['Current Price'] = current_prices.get(trade['Symbol'], None)

            if closed_trades:
                # Calculate summary metrics
                total_pl = sum([t['P/L'] for t in closed_trades])
                winning_trades = [t for t in closed_trades if t['P/L'] > 0]
                win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
                avg_pl = total_pl / len(closed_trades) if closed_trades else 0

                # Display summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Closed", len(closed_trades))
                with col2:
                    pl_color = "normal" if total_pl >= 0 else "inverse"
                    st.metric("Total P/L", f'${total_pl:,.2f}', delta_color=pl_color)
                with col3:
                    st.metric("Win Rate", f'{win_rate:.1f}%')
                with col4:
                    st.metric("Avg P/L", f'${avg_pl:.2f}')

                # Display trades table with color coding
                df_history = pd.DataFrame(closed_trades[:50])  # Show last 50

                # Add TradingView links
                df_history['TradingView'] = df_history['Symbol'].apply(
                    lambda x: f"https://www.tradingview.com/chart/?symbol={x}"
                )

                # Format display columns
                display_cols = ['Close Date', 'Symbol', 'Strategy', 'Strike', 'Current Price',
                              'Open Premium', 'Close Cost', 'P/L', 'P/L %', 'Days Held', 'TradingView']

                # Store raw P/L values before formatting
                pl_raw_vals = df_history['P/L'].copy()

                # Format numeric columns
                df_display = df_history[display_cols].copy()
                df_display['Current Price'] = df_display['Current Price'].apply(lambda x: f'${x:.2f}' if x and pd.notna(x) else '-')
                df_display['Open Premium'] = df_display['Open Premium'].apply(lambda x: f'${x:,.2f}')
                df_display['Close Cost'] = df_display['Close Cost'].apply(lambda x: f'${x:,.2f}')
                df_display['P/L'] = df_display['P/L'].apply(lambda x: f'${x:,.2f}')
                df_display['P/L %'] = df_display['P/L %'].apply(lambda x: f'{x:.1f}%')

                # Function to apply row-wise styling
                def highlight_history_pl(row):
                    """Apply row-wise styling based on P/L value"""
                    idx = row.name
                    pl_val = pl_raw_vals.iloc[idx] if idx < len(pl_raw_vals) else 0

                    styles = [''] * len(row)

                    # Find P/L and P/L % column indices
                    pl_idx = list(df_display.columns).index('P/L')
                    pl_pct_idx = list(df_display.columns).index('P/L %')

                    # Profit (positive P/L) = GREEN TEXT, Loss (negative P/L) = RED TEXT
                    if pl_val > 0:
                        styles[pl_idx] = 'color: #00AA00; font-weight: bold'  # Green text for profit
                        styles[pl_pct_idx] = 'color: #00AA00; font-weight: bold'
                    elif pl_val < 0:
                        styles[pl_idx] = 'color: #DD0000; font-weight: bold'  # Red text for loss
                        styles[pl_pct_idx] = 'color: #DD0000; font-weight: bold'
                    # else pl_val == 0, no styling (neutral)

                    return styles

                # Apply styling
                styled_history = df_display.style.apply(highlight_history_pl, axis=1)

                st.dataframe(
                    styled_history,
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "TradingView": st.column_config.LinkColumn(
                            "Chart",
                            display_text="📈"
                        )
                    }
                )

            else:
                st.info("No closed trades found")

        except Exception as e:
            st.error(f"Error loading trade history: {e}")

    # === PERFORMANCE ANALYTICS ===
    with st.expander("📈 Performance Analytics", expanded=False):
        st.caption("Profit breakdown by time period")

        try:
            if closed_trades:
                performance_data = calculate_performance_by_period(closed_trades)

                df_perf = pd.DataFrame(performance_data)

                # Apply color coding to Total P/L column
                def color_perf_pl(val):
                    """Apply text color to P/L cells"""
                    try:
                        if '$' in str(val):
                            num_val = float(str(val).replace('$', '').replace(',', ''))
                        else:
                            num_val = float(val)

                        if num_val > 0:
                            return 'color: #00AA00; font-weight: bold'  # Green text for profit
                        elif num_val < 0:
                            return 'color: #DD0000; font-weight: bold'  # Red text for loss
                        else:
                            return 'font-weight: bold'  # Neutral for zero
                    except:
                        return ''

                styled_perf = df_perf.style.map(
                    color_perf_pl,
                    subset=['Total P/L']
                )

                # Display styled table
                st.dataframe(
                    styled_perf,
                    hide_index=True,
                    width='stretch'
                )

            else:
                st.info("No performance data available yet")

        except Exception as e:
            st.error(f"Error calculating performance: {e}")


def get_closed_trades_with_pl(rh_session):
    """
    Get closed trades with full P/L calculations
    Matches opening and closing orders
    """
    all_orders = rh_session.get_all_option_orders()

    # Separate opens and closes
    open_orders = {}  # key: (symbol, strike, exp, type) -> order
    closed_trades = []

    for order in all_orders:
        if order.get('state') != 'filled':
            continue

        legs = order.get('legs', [])
        if not legs:
            continue

        leg = legs[0]
        side = leg.get('side')
        position_effect = leg.get('position_effect')

        opt_url = leg.get('option')
        if not opt_url:
            continue

        # Extract option ID
        if isinstance(opt_url, str) and 'options/instruments/' in opt_url:
            opt_id = opt_url.split('/')[-2]
        else:
            opt_id = opt_url

        # Get option details
        opt_data = rh_session.get_option_instrument_data_by_id(opt_id)
        if not opt_data:
            continue

        symbol = opt_data.get('chain_symbol', 'Unknown')
        strike = float(opt_data.get('strike_price', 0))
        exp_date = opt_data.get('expiration_date', 'Unknown')
        opt_type = opt_data.get('type', 'unknown')

        # Trade details
        quantity = float(order.get('quantity', 0))

        # Use processed_premium (total premium) with direction
        processed_premium = float(order.get('processed_premium', 0))
        premium_direction = order.get('processed_premium_direction', 'debit')

        # Credit = you received money (positive), Debit = you paid money (negative)
        if premium_direction == 'credit':
            price = (processed_premium / quantity) if quantity > 0 else 0
        else:  # debit
            price = -(processed_premium / quantity) if quantity > 0 else 0

        date_str = order.get('updated_at', '')

        if date_str:
            trade_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            # Make sure we use timezone-aware datetime
            from datetime import timezone
            trade_date = datetime.now(timezone.utc)

        # Key for matching
        trade_key = (symbol, strike, exp_date, opt_type)

        if position_effect == 'open':
            # Store opening order
            open_orders[trade_key] = {
                'open_price': price,  # Per-contract price
                'open_date': trade_date,
                'quantity': quantity,
                'side': side
            }
        elif position_effect == 'close':
            # Match with opening order
            if trade_key in open_orders:
                open_order = open_orders[trade_key]

                # Calculate P/L using signed values
                # open_price: positive for credit (sold), negative for debit (bought)
                # price: positive for credit (sold), negative for debit (bought)
                # P/L = open_price + price (because close price already has correct sign)

                open_amount = open_order['open_price'] * quantity
                close_amount = price * quantity
                pl = open_amount + close_amount  # Simply add (signs handle the math)

                # Determine strategy
                if side == 'buy':  # Buying to close a short position
                    strategy = 'CSP' if opt_type == 'put' else 'CC'
                else:  # Selling to close a long position
                    strategy = 'Long Call' if opt_type == 'call' else 'Long Put'

                # For percentage, use absolute value of the opening amount
                pl_pct = (pl / abs(open_amount) * 100) if abs(open_amount) > 0 else 0

                # Days held
                days_held = (trade_date - open_order['open_date']).days

                closed_trades.append({
                    'Close Date': trade_date.strftime('%Y-%m-%d'),
                    'Symbol': symbol,
                    'Strategy': strategy,
                    'Strike': f'${strike:.2f}',
                    'Open Premium': abs(open_amount),  # Show as positive for display
                    'Close Cost': abs(close_amount),   # Show as positive for display
                    'P/L': pl,
                    'P/L %': pl_pct,
                    'Days Held': days_held,
                    'close_timestamp': trade_date  # For sorting/filtering
                })

                # Remove from open orders
                del open_orders[trade_key]

    # Sort by close date (newest first)
    closed_trades.sort(key=lambda x: x['close_timestamp'], reverse=True)

    return closed_trades


def calculate_performance_by_period(closed_trades):
    """Calculate performance metrics for different time periods"""
    from datetime import timezone
    now = datetime.now(timezone.utc)

    periods = {
        'Last 7 Days': timedelta(days=7),
        'Last 30 Days': timedelta(days=30),
        'Last 3 Months': timedelta(days=90),
        'Last 6 Months': timedelta(days=180),
        'Last 1 Year': timedelta(days=365),
        'All Time': None
    }

    performance = []

    for period_name, delta in periods.items():
        if delta:
            cutoff = now - delta
            period_trades = [t for t in closed_trades if t['close_timestamp'] >= cutoff]
        else:
            period_trades = closed_trades

        if period_trades:
            total_pl = sum([t['P/L'] for t in period_trades])
            winning = [t for t in period_trades if t['P/L'] > 0]
            win_rate = (len(winning) / len(period_trades) * 100)
            avg_pl = total_pl / len(period_trades)
            best_trade = max([t['P/L'] for t in period_trades])

            # Simple ROI estimate (assumes $10k capital)
            roi = (total_pl / 10000 * 100)

            performance.append({
                'Period': period_name,
                'Trades': len(period_trades),
                'Total P/L': f'${total_pl:,.2f}',
                'Win Rate': f'{win_rate:.1f}%',
                'Avg P/L': f'${avg_pl:.2f}',
                'Best Trade': f'${best_trade:.2f}',
                'ROI': f'{roi:.1f}%'
            })
        else:
            performance.append({
                'Period': period_name,
                'Trades': 0,
                'Total P/L': '$0.00',
                'Win Rate': '0%',
                'Avg P/L': '$0.00',
                'Best Trade': '$0.00',
                'ROI': '0%'
            })

    return performance
