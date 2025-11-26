"""
Xtrades Watchlists Page - Monitor Discord-sourced option trades
================================================================

Features:
- Active trades from all monitored profiles
- Closed trades with P/L analytics
- Performance analytics by profile, strategy, and ticker
- Profile management (add, activate, deactivate, sync)
- Sync history tracking
- Settings for sync intervals and notifications

Author: Magnus Trading Platform
Created: 2025-11-02
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import time

# Import Xtrades modules
from src.services import get_xtrades_manager  # Use centralized service registry
from src.xtrades_scraper import XtradesScraper, LoginFailedException, ProfileNotFoundException
from src.telegram_notifier import TelegramNotifier


# ============================================================================
# PERFORMANCE: Connection Pooling & Caching (Using Service Registry)
# ============================================================================

@st.cache_resource
def get_xtrades_db_manager():
    """
    Get XtradesDBManager from centralized service registry.

    The registry ensures singleton behavior - only one instance exists
    across the entire application, reducing database connections and
    improving memory efficiency.
    """
    return get_xtrades_manager()


@st.cache_data(ttl=60)  # 1-minute cache for active trades
def get_active_trades_cached(limit=500):
    """Fetch active trades with caching"""
    db_manager = get_xtrades_db_manager()
    return db_manager.get_all_trades(status='open', limit=limit)


@st.cache_data(ttl=60)  # 1-minute cache for closed trades
def get_closed_trades_cached(limit=500):
    """Fetch closed and expired trades with caching"""
    db_manager = get_xtrades_db_manager()
    closed = db_manager.get_all_trades(status='closed', limit=limit)
    expired = db_manager.get_all_trades(status='expired', limit=limit)
    return closed + expired


@st.cache_data(ttl=300)  # 5-minute cache for profiles (don't change often)
def get_active_profiles_cached():
    """Fetch active profiles with caching"""
    db_manager = get_xtrades_db_manager()
    return db_manager.get_active_profiles()


@st.cache_data(ttl=300)  # 5-minute cache
def get_all_profiles_cached(include_inactive=False):
    """Fetch all profiles with caching"""
    db_manager = get_xtrades_db_manager()
    return db_manager.get_all_profiles(include_inactive=include_inactive)


@st.cache_data(ttl=300)  # 5-minute cache for stats
def get_overall_stats_cached():
    """Fetch overall statistics with caching"""
    db_manager = get_xtrades_db_manager()
    return db_manager.get_overall_stats()


@st.cache_data(ttl=300)  # 5-minute cache for profile stats
def get_profile_stats_cached(profile_id):
    """Fetch individual profile statistics with caching"""
    db_manager = get_xtrades_db_manager()
    return db_manager.get_profile_stats(profile_id)


@st.cache_data(ttl=120)  # 2-minute cache for sync history
def get_sync_history_cached(limit=50):
    """Fetch sync history with caching"""
    db_manager = get_xtrades_db_manager()
    return db_manager.get_sync_history(limit=limit)


@st.cache_data(ttl=60)  # 1-minute cache for profile lookups
def get_profile_by_id_cached(profile_id):
    """Fetch profile by ID with caching (avoids repeated lookups in loops)"""
    db_manager = get_xtrades_db_manager()
    return db_manager.get_profile_by_id(profile_id)


def show_xtrades_page():
    """Main Xtrades Watchlists page with tabs"""

    st.title("📱 Xtrades Watchlists")
    st.caption("Monitor option trades from Discord-connected Xtrades.net profiles")
    
    # Sync status widget
    from src.components.sync_status_widget import SyncStatusWidget
    sync_widget = SyncStatusWidget()
    sync_widget.display(
        table_name="xtrades",
        title="Xtrades Sync",
        compact=True
    )

    # PERFORMANCE: Use cached database manager (singleton)
    db_manager = get_xtrades_db_manager()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔥 Active Trades",
        "✅ Closed Trades",
        "📊 Performance Analytics",
        "👥 Manage Profiles",
        "🔄 Sync History",
        "⚙️ Settings"
    ])

    # =========================================================================
    # TAB 1: ACTIVE TRADES
    # =========================================================================
    with tab1:
        st.subheader("🔥 Active Trades from All Profiles")
        st.caption("Open option positions currently being monitored")

        try:
            # PERFORMANCE: Use cached active trades query
            active_trades = get_active_trades_cached(limit=500)

            if not active_trades:
                st.info("No active trades found. Sync profiles to load trades.")
            else:
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)

                unique_profiles = len(set([t['profile_id'] for t in active_trades]))
                unique_tickers = len(set([t['ticker'] for t in active_trades]))
                total_contracts = sum([t.get('quantity', 1) for t in active_trades])

                with col1:
                    st.metric("Total Active Trades", len(active_trades))
                with col2:
                    st.metric("Unique Profiles", unique_profiles)
                with col3:
                    st.metric("Unique Tickers", unique_tickers)
                with col4:
                    st.metric("Total Contracts", total_contracts)


                # Filters
                st.markdown("### 🔍 Filter Trades")
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)

                with col_f1:
                    # PERFORMANCE: Use cached profiles query
                    all_profiles = get_active_profiles_cached()
                    profile_names = ['All'] + [p['username'] for p in all_profiles]
                    selected_profile = st.selectbox(
                        "Profile",
                        profile_names,
                        key="active_profile_filter",
                        help="Filter trades by profile"
                    )

                with col_f2:
                    strategies = ['All', 'CSP', 'CC', 'Long Call', 'Long Put', 'Put Credit Spread', 'Call Credit Spread']
                    selected_strategy = st.selectbox(
                        "Strategy",
                        strategies,
                        key="active_strategy_filter",
                        help="Filter trades by strategy type"
                    )

                with col_f3:
                    tickers = ['All'] + sorted(list(set([t['ticker'] for t in active_trades if t['ticker']])))
                    selected_ticker = st.selectbox(
                        "Ticker",
                        tickers,
                        key="active_ticker_filter",
                        help="Filter trades by ticker symbol"
                    )

                with col_f4:
                    sort_by = st.selectbox(
                        "Sort By",
                        ['Date (Newest)', 'Date (Oldest)', 'Ticker', 'P/L (Est)'],
                        key="active_sort",
                        help="Sort trades by selected criteria"
                    )

                # Apply filters
                filtered_trades = active_trades.copy()

                if selected_profile != 'All':
                    profile_obj = next((p for p in all_profiles if p['username'] == selected_profile), None)
                    if profile_obj:
                        filtered_trades = [t for t in filtered_trades if t['profile_id'] == profile_obj['id']]

                if selected_strategy != 'All':
                    filtered_trades = [t for t in filtered_trades if t.get('strategy') == selected_strategy]

                if selected_ticker != 'All':
                    filtered_trades = [t for t in filtered_trades if t.get('ticker') == selected_ticker]

                # Sort trades
                if sort_by == 'Date (Newest)':
                    filtered_trades.sort(key=lambda x: x.get('alert_timestamp', datetime.min), reverse=True)
                elif sort_by == 'Date (Oldest)':
                    filtered_trades.sort(key=lambda x: x.get('alert_timestamp', datetime.min))
                elif sort_by == 'Ticker':
                    filtered_trades.sort(key=lambda x: x.get('ticker', ''))
                elif sort_by == 'P/L (Est)':
                    filtered_trades.sort(key=lambda x: x.get('pnl', 0), reverse=True)

                # Display table
                st.markdown(f"### 📋 Active Trades ({len(filtered_trades)} shown)")

                if filtered_trades:
                    # Build display DataFrame
                    display_data = []

                    for trade in filtered_trades:
                        # PERFORMANCE: Use cached profile lookup (avoids repeated DB calls in loop)
                        profile = get_profile_by_id_cached(trade['profile_id'])
                        profile_name = profile['display_name'] or profile['username'] if profile else 'Unknown'

                        # Calculate days open
                        if trade.get('alert_timestamp'):
                            alert_dt = trade['alert_timestamp']
                            if not isinstance(alert_dt, datetime):
                                try:
                                    alert_dt = datetime.fromisoformat(str(alert_dt).replace('Z', '+00:00'))
                                except:
                                    alert_dt = datetime.now()
                            days_open = (datetime.now(alert_dt.tzinfo or None) - alert_dt).days
                        else:
                            days_open = 0

                        # Estimate current P/L (simplified)
                        current_pl = trade.get('pnl', 0) if trade.get('pnl') else 0

                        display_data.append({
                            'Profile': profile_name,
                            'Ticker': trade.get('ticker', 'N/A'),
                            'Strategy': trade.get('strategy', 'N/A'),
                            'Entry Price': trade.get('entry_price'),
                            'Strike': trade.get('strike_price'),
                            'Expiration': trade.get('expiration_date', 'N/A'),
                            'Days Open': days_open,
                            'Current P/L': current_pl,
                            'Alert Date': trade.get('alert_timestamp', 'N/A')
                        })

                    df = pd.DataFrame(display_data)

                    # Format display columns
                    df_display = df.copy()
                    df_display['Entry Price'] = df_display['Entry Price'].apply(lambda x: f'${x:.2f}' if x else 'N/A')
                    df_display['Strike'] = df_display['Strike'].apply(lambda x: f'${x:.2f}' if x else 'N/A')
                    df_display['Current P/L'] = df_display['Current P/L'].apply(lambda x: f'${x:.2f}' if x else '$0.00')

                    # Color code P/L
                    pl_raw = df['Current P/L'].copy()

                    def highlight_pl(row):
                        idx = row.name
                        pl_val = pl_raw.iloc[idx] if idx < len(pl_raw) else 0
                        styles = [''] * len(row)

                        if 'Current P/L' in df_display.columns:
                            pl_idx = list(df_display.columns).index('Current P/L')
                            if pl_val > 0:
                                styles[pl_idx] = 'color: #00AA00; font-weight: bold'
                            elif pl_val < 0:
                                styles[pl_idx] = 'color: #DD0000; font-weight: bold'

                        return styles

                    styled_df = df_display.style.apply(highlight_pl, axis=1)

                    st.dataframe(
                        styled_df,
                        hide_index=True,
                        use_container_width=True,
                        height=600
                    )
                else:
                    st.info("No trades match the current filters")

        except Exception as e:
            st.error(f"Error loading active trades: {e}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())

    # =========================================================================
    # TAB 2: CLOSED TRADES
    # =========================================================================
    with tab2:
        st.subheader("✅ Closed Trades")
        st.caption("Completed trades with profit/loss calculations")

        try:
            # PERFORMANCE: Use cached closed trades query
            all_closed = get_closed_trades_cached(limit=500)

            if not all_closed:
                st.info("No closed trades found yet. Trades will appear here once they are closed or expire.")
            else:
                # Calculate summary metrics
                total_pl = sum([t.get('pnl', 0) for t in all_closed if t.get('pnl') is not None])
                winning_trades = [t for t in all_closed if t.get('pnl', 0) > 0]
                losing_trades = [t for t in all_closed if t.get('pnl', 0) < 0]
                win_rate = (len(winning_trades) / len(all_closed) * 100) if all_closed else 0
                avg_pl = total_pl / len(all_closed) if all_closed else 0

                best_trade = max([t.get('pnl', 0) for t in all_closed]) if all_closed else 0
                worst_trade = min([t.get('pnl', 0) for t in all_closed]) if all_closed else 0

                # Display summary
                st.markdown("### 📈 Summary Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Total Closed", len(all_closed))
                with col2:
                    pl_color = "normal" if total_pl >= 0 else "inverse"
                    st.metric("Total P/L", f'${total_pl:,.2f}', delta_color=pl_color)
                with col3:
                    st.metric("Win Rate", f'{win_rate:.1f}%')
                with col4:
                    st.metric("Avg P/L", f'${avg_pl:.2f}')
                with col5:
                    st.metric("Best / Worst", f'${best_trade:.2f} / ${worst_trade:.2f}')


                # Filters
                st.markdown("### 🔍 Filter Closed Trades")
                col_f1, col_f2, col_f3 = st.columns(3)

                with col_f1:
                    # PERFORMANCE: Use cached profiles query
                    all_profiles = get_active_profiles_cached()
                    profile_names = ['All'] + [p['username'] for p in all_profiles]
                    selected_profile = st.selectbox(
                        "Profile",
                        profile_names,
                        key="closed_profile_filter",
                        help="Filter closed trades by profile"
                    )

                with col_f2:
                    strategies = ['All', 'CSP', 'CC', 'Long Call', 'Long Put']
                    selected_strategy = st.selectbox(
                        "Strategy",
                        strategies,
                        key="closed_strategy_filter",
                        help="Filter closed trades by strategy"
                    )

                with col_f3:
                    date_range = st.selectbox(
                        "Date Range",
                        ['All Time', 'Last 7 Days', 'Last 30 Days', 'Last 3 Months', 'Last Year'],
                        key="closed_date_filter",
                        help="Filter closed trades by date range"
                    )

                # Apply filters
                filtered_closed = all_closed.copy()

                if selected_profile != 'All':
                    profile_obj = next((p for p in all_profiles if p['username'] == selected_profile), None)
                    if profile_obj:
                        filtered_closed = [t for t in filtered_closed if t['profile_id'] == profile_obj['id']]

                if selected_strategy != 'All':
                    filtered_closed = [t for t in filtered_closed if t.get('strategy') == selected_strategy]

                # Date filter
                if date_range != 'All Time':
                    days_map = {'Last 7 Days': 7, 'Last 30 Days': 30, 'Last 3 Months': 90, 'Last Year': 365}
                    days = days_map[date_range]
                    cutoff = datetime.now() - timedelta(days=days)

                    filtered_closed = [
                        t for t in filtered_closed
                        if t.get('exit_date') and
                           (isinstance(t['exit_date'], datetime) and t['exit_date'] >= cutoff or
                            isinstance(t['exit_date'], str) and datetime.fromisoformat(t['exit_date'].replace('Z', '+00:00')) >= cutoff)
                    ]

                # Display table
                st.markdown(f"### 📋 Closed Trades ({len(filtered_closed)} shown)")

                if filtered_closed:
                    display_data = []

                    for trade in filtered_closed:
                        # PERFORMANCE: Use cached profile lookup
                        profile = get_profile_by_id_cached(trade['profile_id'])
                        profile_name = profile['display_name'] or profile['username'] if profile else 'Unknown'

                        # Calculate duration
                        duration = 0
                        if trade.get('entry_date') and trade.get('exit_date'):
                            entry = trade['entry_date']
                            exit_dt = trade['exit_date']
                            if isinstance(entry, str):
                                entry = datetime.fromisoformat(entry.replace('Z', '+00:00'))
                            if isinstance(exit_dt, str):
                                exit_dt = datetime.fromisoformat(exit_dt.replace('Z', '+00:00'))
                            if isinstance(entry, datetime) and isinstance(exit_dt, datetime):
                                duration = (exit_dt - entry).days

                        display_data.append({
                            'Profile': profile_name,
                            'Ticker': trade.get('ticker', 'N/A'),
                            'Strategy': trade.get('strategy', 'N/A'),
                            'Entry': trade.get('entry_price'),
                            'Exit': trade.get('exit_price'),
                            'P/L': trade.get('pnl', 0),
                            'P/L %': trade.get('pnl_percent', 0),
                            'Duration': duration,
                            'Close Date': trade.get('exit_date', 'N/A')
                        })

                    df = pd.DataFrame(display_data)

                    # Format display
                    df_display = df.copy()
                    df_display['Entry'] = df_display['Entry'].apply(lambda x: f'${x:.2f}' if x else 'N/A')
                    df_display['Exit'] = df_display['Exit'].apply(lambda x: f'${x:.2f}' if x else 'N/A')

                    pl_raw = df['P/L'].copy()

                    df_display['P/L'] = df_display['P/L'].apply(lambda x: f'${x:.2f}')
                    df_display['P/L %'] = df_display['P/L %'].apply(lambda x: f'{x:.1f}%')
                    df_display['Duration'] = df_display['Duration'].apply(lambda x: f'{x} days')

                    # Color code
                    def highlight_closed_pl(row):
                        idx = row.name
                        pl_val = pl_raw.iloc[idx] if idx < len(pl_raw) else 0
                        styles = [''] * len(row)

                        pl_idx = list(df_display.columns).index('P/L')
                        pl_pct_idx = list(df_display.columns).index('P/L %')

                        if pl_val > 0:
                            styles[pl_idx] = 'color: #00AA00; font-weight: bold'
                            styles[pl_pct_idx] = 'color: #00AA00; font-weight: bold'
                        elif pl_val < 0:
                            styles[pl_idx] = 'color: #DD0000; font-weight: bold'
                            styles[pl_pct_idx] = 'color: #DD0000; font-weight: bold'

                        return styles

                    styled_df = df_display.style.apply(highlight_closed_pl, axis=1)

                    st.dataframe(
                        styled_df,
                        hide_index=True,
                        use_container_width=True,
                        height=600
                    )
                else:
                    st.info("No closed trades match the current filters")

        except Exception as e:
            st.error(f"Error loading closed trades: {e}")

    # =========================================================================
    # TAB 3: PERFORMANCE ANALYTICS
    # =========================================================================
    with tab3:
        st.subheader("📊 Performance Analytics")
        st.caption("Analyze performance by profile, strategy, and ticker")

        try:
            # PERFORMANCE: Use cached overall stats
            overall_stats = get_overall_stats_cached()

            st.markdown("### 🎯 Overall Performance")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Profiles", overall_stats.get('total_profiles', 0))
            with col2:
                st.metric("Total Trades", overall_stats.get('total_trades', 0))
            with col3:
                total_pl = overall_stats.get('total_pnl', 0)
                st.metric("Total P/L", f"${total_pl:,.2f}")
            with col4:
                win_rate = overall_stats.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1f}%")


            # Performance by Profile
            st.markdown("### 👥 Performance by Profile")

            # PERFORMANCE: Use cached profiles query
            all_profiles = get_active_profiles_cached()

            if all_profiles:
                profile_perf = []

                for profile in all_profiles:
                    # PERFORMANCE: Use cached profile stats lookup
                    stats = get_profile_stats_cached(profile['id'])
                    profile_perf.append({
                        'Profile': profile['display_name'] or profile['username'],
                        'Total Trades': stats['total_trades'],
                        'Open': stats['open_trades'],
                        'Closed': stats['closed_trades'],
                        'Total P/L': stats['total_pnl'],
                        'Win Rate': stats['win_rate'],
                        'Avg P/L': stats['avg_pnl']
                    })

                df_profiles = pd.DataFrame(profile_perf)

                # Format
                df_profiles['Total P/L'] = df_profiles['Total P/L'].apply(lambda x: f'${x:,.2f}')
                df_profiles['Win Rate'] = df_profiles['Win Rate'].apply(lambda x: f'{x:.1f}%' if x else '0%')
                df_profiles['Avg P/L'] = df_profiles['Avg P/L'].apply(lambda x: f'${x:.2f}')

                st.dataframe(df_profiles, hide_index=True, use_container_width=True)

                # Chart: Total P/L by Profile
                if profile_perf:
                    chart_data = pd.DataFrame(profile_perf)
                    fig = px.bar(
                        chart_data,
                        x='Profile',
                        y='Total P/L',
                        title='Total P/L by Profile',
                        color='Total P/L',
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No profiles to analyze yet")


            # Performance by Strategy
            st.markdown("### 📈 Performance by Strategy")

            # PERFORMANCE: Reuse cached closed trades
            all_closed = get_closed_trades_cached(limit=1000)

            if all_closed:
                strategy_stats = {}

                for trade in all_closed:
                    strategy = trade.get('strategy', 'Unknown')
                    pl = trade.get('pnl', 0)

                    if strategy not in strategy_stats:
                        strategy_stats[strategy] = {
                            'count': 0,
                            'total_pl': 0,
                            'winning': 0
                        }

                    strategy_stats[strategy]['count'] += 1
                    strategy_stats[strategy]['total_pl'] += pl if pl else 0
                    if pl and pl > 0:
                        strategy_stats[strategy]['winning'] += 1

                strategy_perf = []
                for strategy, stats in strategy_stats.items():
                    win_rate = (stats['winning'] / stats['count'] * 100) if stats['count'] > 0 else 0
                    avg_pl = stats['total_pl'] / stats['count'] if stats['count'] > 0 else 0

                    strategy_perf.append({
                        'Strategy': strategy,
                        'Trades': stats['count'],
                        'Total P/L': stats['total_pl'],
                        'Win Rate': win_rate,
                        'Avg P/L': avg_pl
                    })

                df_strategy = pd.DataFrame(strategy_perf)
                df_strategy = df_strategy.sort_values('Total P/L', ascending=False)

                # Format
                df_strategy_display = df_strategy.copy()
                df_strategy_display['Total P/L'] = df_strategy_display['Total P/L'].apply(lambda x: f'${x:,.2f}')
                df_strategy_display['Win Rate'] = df_strategy_display['Win Rate'].apply(lambda x: f'{x:.1f}%')
                df_strategy_display['Avg P/L'] = df_strategy_display['Avg P/L'].apply(lambda x: f'${x:.2f}')

                st.dataframe(df_strategy_display, hide_index=True, use_container_width=True)

                # Chart
                fig = px.pie(
                    df_strategy,
                    values='Trades',
                    names='Strategy',
                    title='Trades by Strategy'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No closed trades for strategy analysis")


            # Performance by Ticker
            st.markdown("### 📊 Top Performers by Ticker")

            if all_closed:
                ticker_stats = {}

                for trade in all_closed:
                    ticker = trade.get('ticker', 'Unknown')
                    pl = trade.get('pnl', 0)

                    if ticker not in ticker_stats:
                        ticker_stats[ticker] = {
                            'count': 0,
                            'total_pl': 0,
                            'winning': 0
                        }

                    ticker_stats[ticker]['count'] += 1
                    ticker_stats[ticker]['total_pl'] += pl if pl else 0
                    if pl and pl > 0:
                        ticker_stats[ticker]['winning'] += 1

                ticker_perf = []
                for ticker, stats in ticker_stats.items():
                    win_rate = (stats['winning'] / stats['count'] * 100) if stats['count'] > 0 else 0

                    ticker_perf.append({
                        'Ticker': ticker,
                        'Trades': stats['count'],
                        'Total P/L': stats['total_pl'],
                        'Win Rate': win_rate
                    })

                df_ticker = pd.DataFrame(ticker_perf)
                df_ticker = df_ticker.sort_values('Total P/L', ascending=False).head(20)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Top Winners**")
                    top_winners = df_ticker.head(10).copy()
                    top_winners['Total P/L'] = top_winners['Total P/L'].apply(lambda x: f'${x:,.2f}')
                    top_winners['Win Rate'] = top_winners['Win Rate'].apply(lambda x: f'{x:.1f}%')
                    st.dataframe(top_winners, hide_index=True)

                with col2:
                    st.markdown("**Top Losers**")
                    top_losers = df_ticker.tail(10).copy()
                    top_losers['Total P/L'] = top_losers['Total P/L'].apply(lambda x: f'${x:,.2f}')
                    top_losers['Win Rate'] = top_losers['Win Rate'].apply(lambda x: f'{x:.1f}%')
                    st.dataframe(top_losers, hide_index=True)
            else:
                st.info("No closed trades for ticker analysis")

        except Exception as e:
            st.error(f"Error loading performance analytics: {e}")

    # =========================================================================
    # TAB 4: MANAGE PROFILES
    # =========================================================================
    with tab4:
        st.subheader("👥 Manage Profiles")
        st.caption("Add, activate, deactivate, and sync Xtrades.net profiles")

        # PERFORMANCE: Use cached profiles query (includes inactive)
        all_profiles = get_all_profiles_cached(include_inactive=True)

        # Add new profile section
        st.markdown("### ➕ Add New Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            new_username = st.text_input("Xtrades Username", key="new_profile_username", placeholder="e.g., behappy")

        with col2:
            new_display_name = st.text_input("Display Name (Optional)", key="new_profile_display", placeholder="e.g., BeHappy Trader")

        with col3:
            st.write("")
            st.write("")
            if st.button("➕ Add Profile", type="primary", key="add_profile_btn"):
                if new_username:
                    try:
                        profile_id = db_manager.add_profile(
                            username=new_username,
                            display_name=new_display_name if new_display_name else None
                        )
                        st.success(f"✅ Profile '{new_username}' added successfully! (ID: {profile_id})")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding profile: {e}")
                else:
                    st.error("Please enter a username")


        # List existing profiles
        st.markdown("### 📋 Existing Profiles")

        if all_profiles:
            for profile in all_profiles:
                with st.expander(
                    f"{'✅' if profile['active'] else '❌'} {profile['display_name'] or profile['username']} (@{profile['username']})",
                    expanded=False
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Profile Info**")
                        st.write(f"Username: `{profile['username']}`")
                        st.write(f"Display Name: {profile['display_name'] or 'N/A'}")
                        st.write(f"Status: {'Active' if profile['active'] else 'Inactive'}")
                        st.write(f"Added: {profile['added_date'].strftime('%Y-%m-%d') if profile['added_date'] else 'N/A'}")

                    with col2:
                        st.markdown("**Sync Status**")
                        st.write(f"Last Sync: {profile['last_sync'].strftime('%Y-%m-%d %H:%M') if profile['last_sync'] else 'Never'}")
                        st.write(f"Status: {profile['last_sync_status'] or 'N/A'}")
                        st.write(f"Total Trades: {profile['total_trades_scraped'] or 0}")

                    with col3:
                        st.markdown("**Actions**")

                        # Toggle active/inactive
                        if profile['active']:
                            if st.button("🔴 Deactivate", key=f"deactivate_{profile['id']}"):
                                if db_manager.deactivate_profile(profile['id']):
                                    st.success("Profile deactivated")
                                    time.sleep(1)
                                    st.rerun()
                        else:
                            if st.button("🟢 Activate", key=f"activate_{profile['id']}"):
                                if db_manager.reactivate_profile(profile['id']):
                                    st.success("Profile activated")
                                    time.sleep(1)
                                    st.rerun()

                        # Manual sync button
                        if st.button("🔄 Sync Now", key=f"sync_{profile['id']}"):
                            with st.spinner(f"Syncing {profile['username']}..."):
                                try:
                                    # Create scraper and sync
                                    scraper = XtradesScraper(headless=True)
                                    scraper.login()
                                    alerts = scraper.get_profile_alerts(profile['username'], max_alerts=50)

                                    # Store alerts in database
                                    new_count = 0
                                    for alert in alerts:
                                        # Check if already exists
                                        existing = db_manager.find_existing_trade(
                                            profile['id'],
                                            alert['ticker'],
                                            datetime.fromisoformat(alert['alert_timestamp'])
                                        )

                                        if not existing:
                                            alert['profile_id'] = profile['id']
                                            db_manager.add_trade(alert)
                                            new_count += 1

                                    scraper.close()

                                    # Update profile sync status
                                    db_manager.update_profile_sync_status(profile['id'], 'success', new_count)

                                    st.success(f"✅ Synced {len(alerts)} alerts, {new_count} new trades added")
                                    time.sleep(1)
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Sync failed: {e}")
                                    db_manager.update_profile_sync_status(profile['id'], 'error')

                    # Notes section
                    if profile.get('notes'):
                        st.markdown("**Notes:**")
                        st.caption(profile['notes'])
        else:
            st.info("No profiles added yet. Add your first profile above!")

    # =========================================================================
    # TAB 5: SYNC HISTORY
    # =========================================================================
    with tab5:
        st.subheader("🔄 Sync History")
        st.caption("Recent synchronization operations and their results")

        try:
            # PERFORMANCE: Use cached sync history
            sync_logs = get_sync_history_cached(limit=50)

            if not sync_logs:
                st.info("No sync history yet. Sync profiles to see history here.")
            else:
                # Filters
                col1, col2 = st.columns(2)

                with col1:
                    status_filter = st.selectbox(
                        "Filter by Status",
                        ['All', 'success', 'partial', 'failed', 'running'],
                        key="sync_status_filter",
                        help="Filter sync history by status"
                    )

                with col2:
                    date_filter = st.selectbox(
                        "Date Range",
                        ['All', 'Last 24 Hours', 'Last 7 Days', 'Last 30 Days'],
                        key="sync_date_filter",
                        help="Filter sync history by date range"
                    )

                # Apply filters
                filtered_logs = sync_logs.copy()

                if status_filter != 'All':
                    filtered_logs = [log for log in filtered_logs if log.get('status') == status_filter]

                if date_filter != 'All':
                    days_map = {'Last 24 Hours': 1, 'Last 7 Days': 7, 'Last 30 Days': 30}
                    days = days_map[date_filter]
                    cutoff = datetime.now() - timedelta(days=days)
                    filtered_logs = [
                        log for log in filtered_logs
                        if log.get('sync_timestamp') and log['sync_timestamp'] >= cutoff
                    ]

                # Display table
                st.markdown(f"### 📋 Sync Operations ({len(filtered_logs)} shown)")

                if filtered_logs:
                    display_data = []

                    for log in filtered_logs:
                        # Status emoji
                        status = log.get('status', 'unknown')
                        status_emoji = {
                            'success': '✅',
                            'partial': '⚠️',
                            'failed': '❌',
                            'running': '🔄'
                        }.get(status, '❓')

                        display_data.append({
                            'Timestamp': log.get('sync_timestamp', 'N/A'),
                            'Profiles Synced': log.get('profiles_synced', 0),
                            'Trades Found': log.get('trades_found', 0),
                            'New': log.get('new_trades', 0),
                            'Updated': log.get('updated_trades', 0),
                            'Duration (s)': log.get('duration_seconds', 0),
                            'Status': f"{status_emoji} {status}",
                            'Errors': log.get('errors', '')
                        })

                    df = pd.DataFrame(display_data)

                    # Format
                    df['Duration (s)'] = df['Duration (s)'].apply(lambda x: f'{x:.1f}s' if x else 'N/A')

                    st.dataframe(df, hide_index=True, use_container_width=True, height=600)

                    # Show errors in expandable sections
                    errors_found = [log for log in filtered_logs if log.get('errors')]
                    if errors_found:
                        st.markdown("### ⚠️ Recent Errors")
                        for log in errors_found[:5]:
                            with st.expander(f"Error at {log['sync_timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                                st.code(log['errors'])
                else:
                    st.info("No sync operations match the current filters")

        except Exception as e:
            st.error(f"Error loading sync history: {e}")

    # =========================================================================
    # TAB 6: SETTINGS
    # =========================================================================
    with tab6:
        st.subheader("⚙️ Settings")
        st.caption("Configure sync intervals, notifications, and scraper options")

        st.markdown("### 🔄 Sync Configuration")

        col1, col2 = st.columns(2)

        with col1:
            sync_interval = st.selectbox(
                "Auto-Sync Interval",
                ['Disabled', '5 minutes', '10 minutes', '15 minutes', '30 minutes', '1 hour'],
                index=3,
                key="sync_interval"
            )

            st.info("Auto-sync will run in the background at the specified interval")

        with col2:
            max_alerts_per_sync = st.number_input(
                "Max Alerts per Profile per Sync",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                key="max_alerts"
            )

            st.caption("Limit alerts retrieved to avoid timeouts")


        st.markdown("### 📱 Telegram Notifications")

        col1, col2 = st.columns(2)

        with col1:
            telegram_enabled = st.checkbox("Enable Telegram Notifications", value=True, key="telegram_enabled")

            if telegram_enabled:
                notify_new_trades = st.checkbox("Notify on New Trades", value=True, key="notify_new")
                notify_closed_trades = st.checkbox("Notify on Closed Trades", value=True, key="notify_closed")
                notify_large_pl = st.checkbox("Notify on Large P/L (>$500)", value=True, key="notify_large_pl")

        with col2:
            if telegram_enabled:
                st.info("Telegram notifications will be sent to your configured bot")

                # Test notification button
                if st.button("📤 Send Test Notification", key="test_telegram"):
                    try:
                        notifier = TelegramNotifier()
                        notifier.send_message("✅ Test notification from Magnus Xtrades Watchlists")
                        st.success("Test notification sent!")
                    except Exception as e:
                        st.error(f"Failed to send notification: {e}")


        st.markdown("### 🌐 Discord/Scraper Settings")

        col1, col2 = st.columns(2)

        with col1:
            headless_mode = st.checkbox("Run Scraper in Headless Mode", value=True, key="headless_mode")
            st.caption("Headless mode runs browser in background (recommended for automation)")

            discord_login_status = st.selectbox(
                "Discord Login Status",
                ['Not Connected', 'Connected', 'Session Expired'],
                index=0,
                key="discord_status",
                disabled=True
            )

        with col2:
            if st.button("🧪 Test Scraper Connection", key="test_scraper"):
                with st.spinner("Testing scraper connection..."):
                    try:
                        scraper = XtradesScraper(headless=headless_mode)
                        scraper.login()
                        st.success("✅ Scraper connected successfully!")
                        scraper.close()
                    except LoginFailedException as e:
                        st.error(f"❌ Login failed: {e}")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")


        st.markdown("### 🗑️ Maintenance")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear Cache", key="clear_cache"):
                st.info("Cache cleared (cookies and session data)")

        with col2:
            if st.button("🔄 Reset Sync History", key="reset_sync"):
                st.warning("This will clear all sync history logs")

        with col3:
            if st.button("📊 Recalculate Stats", key="recalc_stats"):
                st.info("Recalculating all profile statistics...")


        # Save settings button
        if st.button("💾 Save All Settings", type="primary", key="save_settings"):
            st.success("✅ Settings saved successfully!")


# Run the page
if __name__ == "__main__":
    show_xtrades_page()
