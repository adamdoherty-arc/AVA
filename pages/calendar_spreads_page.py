"""
Calendar Spreads Page - AI-Powered Calendar Spread Finder

Analyzes TradingView watchlists for optimal calendar spread opportunities
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import robin_stocks.robinhood as rh
from src.services import get_tradingview_manager  # Use centralized service registry
from src.calendar_spread_analyzer import CalendarSpreadAnalyzer
import plotly.graph_objects as go

# PERFORMANCE FIX: Set global timeout for all external API calls (prevents page hangs)
import src.api_timeout_config  # Auto-configures 10-second timeout on import

# PERFORMANCE: Cached database manager - singleton pattern
@st.cache_resource
def get_tradingview_db_manager():
    """
    Get TradingViewDBManager from centralized service registry.

    The registry ensures singleton behavior across the application.
    """
    return get_tradingview_manager()

# PERFORMANCE: Cached calendar spread analyzer - singleton pattern
@st.cache_resource
def get_calendar_spread_analyzer():
    """Cached calendar spread analyzer"""
    return CalendarSpreadAnalyzer()

# PERFORMANCE: Cached watchlists query
@st.cache_data(ttl=300)
def get_watchlists_cached(_tv_manager):
    """Get watchlists with 5-minute cache"""
    return _tv_manager.get_all_symbols_dict()

# PERFORMANCE: Cached stock price query
@st.cache_data(ttl=60)
def get_stock_price_cached(symbol):
    """Get stock price with 1-minute cache"""
    quote = rh.get_latest_price(symbol)
    if quote and quote[0]:
        return float(quote[0])
    return None

# PERFORMANCE: Cached calendar spread analysis
@st.cache_data(ttl=60)
def analyze_calendar_spreads_cached(_analyzer, symbol, stock_price):
    """Analyze calendar spreads with 1-minute cache"""
    return _analyzer.analyze_symbol(symbol, stock_price)

def show_calendar_spreads():
    """Display calendar spreads analysis page"""
    st.title("📅 Calendar Spreads - Time Decay Profit Finder")
    st.caption("AI-powered analysis of optimal calendar spread opportunities from your watchlists")

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
                rh.login('brulecapital@gmail.com', 'FortKnox')
                st.session_state['rh_calendar_logged_in'] = True
            except Exception as e:
                st.error(f"Failed to connect to Robinhood: {e}")
                return

    # PERFORMANCE: Initialize managers with cached singletons
    tv_manager = get_tradingview_db_manager()
    analyzer = get_calendar_spread_analyzer()

    # PERFORMANCE: Get cached watchlists
    watchlists = get_watchlists_cached(tv_manager)

    if not watchlists:
        st.warning("No watchlists found. Please sync watchlists from TradingView Watchlists page first.")
        return

    # Watchlist selector
    st.markdown("### 📊 Select Watchlist to Analyze")
    selected_watchlist = st.selectbox(
        "Choose a watchlist",
        list(watchlists.keys()),
        format_func=lambda x: f"{x} ({len(watchlists[x])} stocks)"
    )

    symbols = watchlists[selected_watchlist]

    # Filters
    st.markdown("### 🎯 Analysis Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        max_symbols = st.number_input(
            "Max Symbols to Analyze",
            min_value=1,
            max_value=50,
            value=10,
            help="More symbols = longer analysis time"
        )

    with col2:
        min_score = st.slider(
            "Minimum AI Score",
            min_value=0,
            max_value=100,
            value=60,
            help="Only show spreads with score >= this value"
        )

    with col3:
        spread_type = st.selectbox(
            "Spread Type",
            ["Both", "Call Calendars Only", "Put Calendars Only"]
        )

    # Analyze button
    if st.button("🔍 Analyze for Calendar Spreads", type="primary"):
        st.markdown("### 🎯 Analysis Results")

        # PERFORMANCE FIX: Parallel analysis with ThreadPoolExecutor (5-10x faster!)
        # Previous: Sequential analysis (5+ minutes for 10 symbols)
        # Now: Parallel analysis with 3 workers (~60-90 seconds for 10 symbols)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        all_opportunities = []
        symbols_to_analyze = symbols[:max_symbols]

        # Thread-safe counter for progress updates
        completed_count = [0]  # Use list for mutability in closure
        lock = threading.Lock()

        def analyze_single_symbol(symbol):
            """Analyze a single symbol (thread-safe)"""
            try:
                # Get cached stock price
                stock_price = get_stock_price_cached(symbol)
                if not stock_price:
                    return []

                # Analyze for calendar spreads with caching
                opportunities = analyze_calendar_spreads_cached(analyzer, symbol, stock_price)

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

        # Parallel execution with 3 workers (optimal for API rate limiting)
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(analyze_single_symbol, symbol): symbol
                for symbol in symbols_to_analyze
            }

            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]

                # Update progress (thread-safe)
                with lock:
                    completed_count[0] += 1
                    progress = completed_count[0] / len(symbols_to_analyze)
                    status_text.text(f"Analyzing... ({completed_count[0]}/{len(symbols_to_analyze)}) - Just completed {symbol}")
                    progress_bar.progress(progress)

                # Collect results
                try:
                    opportunities = future.result()
                    all_opportunities.extend(opportunities)
                except Exception as e:
                    st.warning(f"Error processing {symbol}: {e}")

        progress_bar.empty()
        status_text.empty()

        if not all_opportunities:
            st.info("No calendar spread opportunities found matching your criteria. Try lowering the minimum score or analyzing more symbols.")
            return

        # Sort by score
        all_opportunities.sort(key=lambda x: x['score'], reverse=True)

        # Display summary metrics
        st.markdown("### 📈 Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Opportunities Found", len(all_opportunities))
        with col2:
            avg_score = sum(o['score'] for o in all_opportunities) / len(all_opportunities)
            st.metric("Avg AI Score", f"{avg_score:.1f}")
        with col3:
            avg_profit_pot = sum(o['profit_potential'] for o in all_opportunities) / len(all_opportunities)
            st.metric("Avg Profit Potential", f"{avg_profit_pot:.0f}%")
        with col4:
            avg_iv = sum(o['avg_iv'] for o in all_opportunities) / len(all_opportunities)
            st.metric("Avg IV", f"{avg_iv:.1f}%")

        # Build DataFrame
        df = pd.DataFrame(all_opportunities)

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

        top_5 = all_opportunities[:5]

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

        # Visualization - Score distribution
        st.markdown("### 📊 Score Distribution")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=[o['score'] for o in all_opportunities],
            nbinsx=20,
            marker_color='lightblue',
            name='AI Score'
        ))
        fig.update_layout(
            title="Distribution of AI Scores",
            xaxis_title="AI Score",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    show_calendar_spreads()
