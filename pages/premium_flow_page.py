"""
Premium Options Flow Page - Institutional Money Tracking

Comprehensive UI for tracking options flow, premium inflows/outflows,
and identifying the best trading opportunities for wheel strategy.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import logging
from typing import List, Dict, Any
import numpy as np

from src.options_flow_tracker import OptionsFlowTracker, get_popular_optionable_symbols
from src.ai_flow_analyzer import AIFlowAnalyzer
from src.tradingview_db_manager import TradingViewDBManager
from src.components.pagination_component import paginate_dataframe

logger = logging.getLogger(__name__)


# ========================================================================
# PERFORMANCE OPTIMIZATION: Cached Database Connection Manager
# ========================================================================

@st.cache_resource
def get_tv_manager():
    """
    Cached database manager to avoid creating multiple connections.
    Connection pooling for better performance.
    """
    return TradingViewDBManager()


@st.cache_data(ttl=300)  # 5-minute cache
def check_tables_exist():
    """
    Cached check for table existence to avoid repeated queries.
    """
    tv_manager = get_tv_manager()
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'options_flow'
        )
    """)
    tables_exist = cur.fetchone()[0]
    cur.close()
    conn.close()

    return tables_exist


@st.cache_data(ttl=60)  # 1-minute cache for flow summary
def get_market_flow_summary_cached(tracker):
    """Cached wrapper for market flow summary"""
    return tracker.get_market_flow_summary()


@st.cache_data(ttl=60)  # 1-minute cache for top symbols
def get_top_symbols_flow_cached():
    """Cached query for top 10 symbols by net flow"""
    tv_manager = get_tv_manager()
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            symbol,
            net_premium_flow,
            put_call_ratio,
            flow_sentiment,
            total_volume,
            call_premium,
            put_premium
        FROM options_flow
        WHERE flow_date = CURRENT_DATE
        ORDER BY ABS(net_premium_flow) DESC
        LIMIT 10
    """)

    top_symbols = cur.fetchall()
    cur.close()
    conn.close()

    return top_symbols


@st.cache_data(ttl=60)  # 1-minute cache for unusual activity
def get_unusual_activity_cached():
    """Cached query for unusual activity alerts"""
    tv_manager = get_tv_manager()
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            symbol,
            total_volume,
            net_premium_flow,
            flow_sentiment,
            put_call_ratio
        FROM options_flow
        WHERE flow_date = CURRENT_DATE
            AND unusual_activity = true
        ORDER BY total_volume DESC
        LIMIT 10
    """)

    unusual = cur.fetchall()
    cur.close()
    conn.close()

    return unusual


def display_premium_flow_page():
    """Main function to display Premium Options Flow page"""

    st.title("💸 Premium Options Flow - Institutional Money Tracking")
    st.caption("Track institutional options flow and identify high-probability wheel strategy opportunities")

    # Sync status widget
    from src.components.sync_status_widget import SyncStatusWidget
    sync_widget = SyncStatusWidget()
    sync_widget.display(
        table_name="stock_premiums",
        title="Options Flow Data Sync",
        compact=True
    )

    # Initialize components (use cached manager)
    tracker = OptionsFlowTracker()
    analyzer = AIFlowAnalyzer()
    tv_manager = get_tv_manager()  # PERFORMANCE: Use cached manager

    # Check if tables exist (PERFORMANCE: Use cached check)
    tables_exist = check_tables_exist()

    if not tables_exist:
        st.warning("⚠️ Premium Options Flow tables not yet created. Run migration first.")

        if st.button("🔧 Run Migration Now", type="primary"):
            with st.spinner("Creating premium options flow tables..."):
                try:
                    import os
                    migration_path = os.path.join(os.path.dirname(__file__), 'migrations', 'add_premium_options_flow.sql')

                    with open(migration_path, 'r') as f:
                        migration_sql = f.read()

                    conn = tv_manager.get_connection()
                    cur = conn.cursor()
                    cur.execute(migration_sql)
                    conn.commit()
                    cur.close()
                    conn.close()

                    st.success("✅ Migration complete! Refresh page to continue.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Migration failed: {e}")

        return

    # Refresh data controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if st.button("🔄 Refresh Flow Data", type="primary"):
            with st.spinner("Fetching latest options flow data..."):
                symbols = get_popular_optionable_symbols()[:20]  # Top 20 for quick refresh
                results = tracker.batch_update_flow(symbols, limit=20)
                st.success(f"✅ Updated {results['success']} symbols successfully")

    with col2:
        if st.button("🤖 Run AI Analysis"):
            with st.spinner("Running AI analysis on flow data..."):
                conn = tv_manager.get_connection()
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT symbol FROM options_flow WHERE flow_date = CURRENT_DATE")
                symbols = [row[0] for row in cur.fetchall()]
                cur.close()
                conn.close()

                analyzed = analyzer.batch_analyze(symbols[:30])
                st.success(f"✅ Analyzed {analyzed} symbols")

    with col3:
        # Last updated timestamp
        conn = tv_manager.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT MAX(last_updated) FROM options_flow")
        last_updated = cur.fetchone()[0]
        cur.close()
        conn.close()

        if last_updated:
            st.caption(f"Updated: {last_updated.strftime('%H:%M:%S')}")

    st.markdown("---")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Flow Overview",
        "🎯 Flow Opportunities",
        "🔍 Flow Analysis",
        "📚 Strategies"
    ])

    # TAB 1: Flow Overview
    with tab1:
        display_flow_overview(tracker, tv_manager)

    # TAB 2: Flow Opportunities
    with tab2:
        display_flow_opportunities(tracker, analyzer, tv_manager)

    # TAB 3: Flow Analysis
    with tab3:
        display_flow_analysis(tracker, analyzer, tv_manager)

    # TAB 4: Strategies
    with tab4:
        display_strategies()


def display_flow_overview(tracker: OptionsFlowTracker, tv_manager: TradingViewDBManager):
    """Display flow overview tab"""

    st.subheader("📊 Market-Wide Options Flow Overview")

    # Get market summary (PERFORMANCE: Use cached version)
    summary = get_market_flow_summary_cached(tracker)

    if not summary or summary.get('total_symbols', 0) == 0:
        st.info("📊 No flow data available yet. Click 'Refresh Flow Data' to fetch data.")
        return

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Symbols",
            f"{summary.get('total_symbols', 0):,}",
            help="Number of symbols with flow data today"
        )

    with col2:
        call_premium = summary.get('total_call_premium', 0)
        st.metric(
            "Call Premium",
            f"${call_premium/1e6:.1f}M",
            help="Total premium in call options"
        )

    with col3:
        put_premium = summary.get('total_put_premium', 0)
        st.metric(
            "Put Premium",
            f"${put_premium/1e6:.1f}M",
            help="Total premium in put options"
        )

    with col4:
        net_flow = summary.get('total_net_flow', 0)
        delta_color = "normal" if net_flow >= 0 else "inverse"
        st.metric(
            "Net Flow",
            f"${abs(net_flow)/1e6:.1f}M",
            delta="Calls" if net_flow >= 0 else "Puts",
            delta_color=delta_color,
            help="Net premium flow (calls - puts)"
        )

    with col5:
        avg_pc = summary.get('avg_put_call_ratio', 1.0)
        st.metric(
            "Avg P/C Ratio",
            f"{avg_pc:.2f}",
            help="Average put/call ratio across all symbols"
        )

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        # Call vs Put premium bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Call Premium',
            x=['Premium Flow'],
            y=[call_premium/1e6],
            marker_color='green'
        ))
        fig.add_trace(go.Bar(
            name='Put Premium',
            x=['Premium Flow'],
            y=[put_premium/1e6],
            marker_color='red'
        ))

        fig.update_layout(
            title="Call vs Put Premium (Millions $)",
            barmode='group',
            height=300,
            showlegend=True,
            yaxis_title="Premium ($M)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Flow sentiment pie chart
        bullish = summary.get('bullish_count', 0)
        bearish = summary.get('bearish_count', 0)
        neutral = summary.get('neutral_count', 0)

        fig = go.Figure(data=[go.Pie(
            labels=['Bullish', 'Bearish', 'Neutral'],
            values=[bullish, bearish, neutral],
            marker_colors=['green', 'red', 'gray'],
            hole=0.4
        )])

        fig.update_layout(
            title="Flow Sentiment Breakdown",
            height=300,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Top 10 symbols by net flow
    st.subheader("🔥 Top 10 Symbols by Net Premium Flow")

    # PERFORMANCE: Use cached query
    top_symbols = get_top_symbols_flow_cached()

    if top_symbols:
        df = pd.DataFrame(top_symbols, columns=[
            'Symbol', 'Net Flow', 'P/C Ratio', 'Sentiment',
            'Volume', 'Call Premium', 'Put Premium'
        ])

        df['Net Flow'] = df['Net Flow'].apply(lambda x: f"${x/1e6:.2f}M")
        df['P/C Ratio'] = df['P/C Ratio'].apply(lambda x: f"{x:.2f}")
        df['Volume'] = df['Volume'].apply(lambda x: f"{x:,}")
        df['Call Premium'] = df['Call Premium'].apply(lambda x: f"${x/1e6:.2f}M")
        df['Put Premium'] = df['Put Premium'].apply(lambda x: f"${x/1e6:.2f}M")

        # PERFORMANCE: Add pagination for large tables
        paginated_df = paginate_dataframe(df, page_size=50, key_prefix="premium_flow_top_symbols")
        st.dataframe(paginated_df, width=None, height=400)
    else:
        st.info("No data available")

    # Unusual activity alerts
    st.markdown("---")
    st.subheader("⚠️ Unusual Activity Alerts")

    # PERFORMANCE: Use cached query
    unusual = get_unusual_activity_cached()

    if unusual:
        cols = st.columns(min(len(unusual), 3))
        for i, (symbol, volume, net_flow, sentiment, pc_ratio) in enumerate(unusual[:6]):
            col_idx = i % 3
            with cols[col_idx]:
                sentiment_color = "#d4edda" if sentiment == 'Bullish' else "#f8d7da" if sentiment == 'Bearish' else "#fff3cd"
                st.markdown(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: {sentiment_color}; margin-bottom: 10px;'>
                    <h4 style='margin: 0;'>{symbol}</h4>
                    <p style='margin: 5px 0;'><strong>{sentiment}</strong> Flow</p>
                    <p style='margin: 5px 0;'>Volume: {volume:,}</p>
                    <p style='margin: 5px 0;'>P/C: {pc_ratio:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No unusual activity detected today")


def display_flow_opportunities(tracker: OptionsFlowTracker, analyzer: AIFlowAnalyzer,
                               tv_manager: TradingViewDBManager):
    """Display flow opportunities tab"""

    st.subheader("🎯 Top Flow-Based Trading Opportunities")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sentiment_filter = st.selectbox(
            "Sentiment",
            ["All", "Bullish", "Bearish", "Neutral"],
            help="Filter by flow sentiment"
        )

    with col2:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "Low", "Medium", "High"],
            help="Filter by risk assessment"
        )

    with col3:
        min_score = st.slider(
            "Min Score",
            0, 100, 50,
            help="Minimum opportunity score"
        )

    with col4:
        action_filter = st.selectbox(
            "Action",
            ["All", "SELL_PUT", "BUY_CALL", "WAIT"],
            help="Filter by recommended action"
        )

    # Get opportunities
    opportunities = tracker.get_top_flow_opportunities(limit=100)

    if not opportunities:
        st.info("📊 No opportunities available. Run AI Analysis to generate recommendations.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(opportunities)

    # Apply filters
    if sentiment_filter != "All":
        df = df[df['flow_sentiment'] == sentiment_filter]

    if risk_filter != "All":
        df = df[df['risk_level'] == risk_filter]

    if action_filter != "All":
        df = df[df['best_action'] == action_filter]

    df = df[df['opportunity_score'] >= min_score]

    if df.empty:
        st.warning("No opportunities match your filters")
        return

    st.info(f"Found {len(df)} opportunities matching your criteria")

    # Display opportunities table
    display_df = df[[
        'symbol', 'current_price', 'opportunity_score', 'net_flow_7d',
        'put_call_ratio', 'flow_sentiment', 'flow_trend_7d',
        'best_action', 'risk_level', 'confidence'
    ]].copy()

    display_df.columns = [
        'Symbol', 'Price', 'Score', 'Net Flow 7D',
        'P/C Ratio', 'Sentiment', 'Trend',
        'Action', 'Risk', 'Confidence'
    ]

    # Format columns
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "0")
    display_df['Net Flow 7D'] = display_df['Net Flow 7D'].apply(
        lambda x: f"${x/1e6:.2f}M" if pd.notna(x) else "N/A"
    )
    display_df['P/C Ratio'] = display_df['P/C Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    # Color code by score
    def color_score(val):
        try:
            score = float(val)
            if score >= 80:
                return 'background-color: #d4edda'
            elif score >= 60:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        except:
            return ''

    styled_df = display_df.style.applymap(color_score, subset=['Score'])

    # PERFORMANCE: Add pagination for opportunities table
    paginated_opp_df = paginate_dataframe(display_df, page_size=50, key_prefix="premium_flow_opportunities")
    st.dataframe(paginated_opp_df, width=None, height=500)

    # Detailed opportunity cards
    st.markdown("---")
    st.subheader("📋 Detailed Opportunity Analysis")

    # Show top 5 in expandable cards
    top_opportunities = df.nlargest(5, 'opportunity_score')

    for _, opp in top_opportunities.iterrows():
        with st.expander(f"**{opp['symbol']}** - Score: {opp['opportunity_score']:.1f}/100"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Flow Metrics**")
                st.write(f"• Current Price: ${opp.get('current_price', 0):.2f}")
                st.write(f"• 7D Net Flow: ${opp.get('net_flow_7d', 0)/1e6:.2f}M")
                st.write(f"• Put/Call Ratio: {opp.get('put_call_ratio', 0):.2f}")
                st.write(f"• Sentiment: {opp.get('flow_sentiment', 'N/A')}")

            with col2:
                st.markdown("**Recommendation**")
                st.write(f"• Action: **{opp.get('best_action', 'WAIT')}**")
                st.write(f"• Risk Level: {opp.get('risk_level', 'N/A')}")
                st.write(f"• Confidence: {opp.get('confidence', 0):.1%}")
                if opp.get('recommended_strike'):
                    st.write(f"• Target Strike: ${opp.get('recommended_strike', 0):.2f}")

            with col3:
                st.markdown("**Expected Outcome**")
                if opp.get('expected_premium'):
                    st.write(f"• Expected Premium: ${opp.get('expected_premium', 0):.2f}")
                if opp.get('win_probability'):
                    st.write(f"• Win Probability: {opp.get('win_probability', 0):.1%}")
                st.write(f"• Trend: {opp.get('flow_trend_7d', 'N/A')}")

            if opp.get('ai_recommendation'):
                st.markdown("**AI Analysis**")
                st.info(opp['ai_recommendation'])


def display_flow_analysis(tracker: OptionsFlowTracker, analyzer: AIFlowAnalyzer,
                          tv_manager: TradingViewDBManager):
    """Display flow analysis tab"""

    st.subheader("🔍 Detailed Flow Analysis by Symbol")

    # Symbol selector
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT symbol
        FROM options_flow
        ORDER BY symbol
    """)

    available_symbols = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    if not available_symbols:
        st.info("No symbols available. Refresh flow data first.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select Symbol",
            available_symbols,
            help="Choose a symbol to analyze"
        )

    with col2:
        days_back = st.selectbox(
            "Time Period",
            [7, 14, 30],
            help="Days of historical flow data"
        )

    if not selected_symbol:
        return

    # Get historical flow data
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            flow_date,
            call_volume,
            put_volume,
            call_premium,
            put_premium,
            net_premium_flow,
            put_call_ratio,
            flow_sentiment,
            total_volume
        FROM options_flow
        WHERE symbol = %s
            AND flow_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY flow_date ASC
    """, (selected_symbol, days_back))

    historical_data = cur.fetchall()
    cur.close()
    conn.close()

    if not historical_data:
        st.warning(f"No historical data for {selected_symbol}")
        return

    # Create DataFrame
    df_hist = pd.DataFrame(historical_data, columns=[
        'Date', 'Call Volume', 'Put Volume', 'Call Premium', 'Put Premium',
        'Net Flow', 'P/C Ratio', 'Sentiment', 'Total Volume'
    ])

    # Historical flow chart
    st.markdown("---")
    st.subheader(f"📈 {selected_symbol} Premium Flow History ({days_back} Days)")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{selected_symbol} Net Premium Flow', 'Put/Call Ratio'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    # Net flow chart
    colors = ['green' if x >= 0 else 'red' for x in df_hist['Net Flow']]
    fig.add_trace(
        go.Bar(
            x=df_hist['Date'],
            y=df_hist['Net Flow'] / 1e6,
            name='Net Flow',
            marker_color=colors
        ),
        row=1, col=1
    )

    # P/C ratio chart
    fig.add_trace(
        go.Scatter(
            x=df_hist['Date'],
            y=df_hist['P/C Ratio'],
            name='P/C Ratio',
            mode='lines+markers',
            line=dict(color='blue')
        ),
        row=2, col=1
    )

    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Net Flow ($M)", row=1, col=1)
    fig.update_yaxes(title_text="P/C Ratio", row=2, col=1)

    fig.update_layout(height=600, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

    # Call vs Put volume comparison
    st.markdown("---")
    st.subheader("📊 Call vs Put Volume Comparison")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_hist['Date'],
        y=df_hist['Call Volume'],
        name='Call Volume',
        fill='tozeroy',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=df_hist['Date'],
        y=df_hist['Put Volume'],
        name='Put Volume',
        fill='tozeroy',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f"{selected_symbol} - Call vs Put Volume",
        xaxis_title="Date",
        yaxis_title="Volume (Contracts)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Get AI analysis
    st.markdown("---")
    st.subheader("🤖 AI Insights & Recommendations")

    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            ai_recommendation,
            key_insights,
            opportunity_score,
            best_action,
            risk_level,
            confidence,
            recommended_strike,
            expected_premium,
            win_probability
        FROM options_flow_analysis
        WHERE symbol = %s
    """, (selected_symbol,))

    analysis = cur.fetchone()
    cur.close()
    conn.close()

    if analysis:
        ai_rec, insights, score, action, risk, conf, strike, premium, win_prob = analysis

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Opportunity Score", f"{score:.1f}/100")

        with col2:
            st.metric("Recommended Action", action)

        with col3:
            st.metric("Risk Level", risk)

        with col4:
            st.metric("Confidence", f"{conf:.1%}")

        if ai_rec:
            st.markdown("**AI Recommendation:**")
            st.info(ai_rec)

        if insights:
            st.markdown("**Key Insights:**")
            for insight in insights:
                st.write(f"• {insight}")

        if strike and premium:
            st.markdown("**Trade Setup:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Target Strike: **${strike:.2f}**")
            with col2:
                st.write(f"Expected Premium: **${premium:.2f}**")
            with col3:
                if win_prob:
                    st.write(f"Win Probability: **{win_prob:.1%}**")

    else:
        st.warning("No AI analysis available. Run AI Analysis to generate insights.")

    # Similar patterns
    st.markdown("---")
    st.subheader("🔄 Similar Flow Patterns")

    # Find symbols with similar flow characteristics
    current_flow = df_hist.iloc[-1] if not df_hist.empty else None

    if current_flow is not None:
        target_pc = current_flow['P/C Ratio']
        target_sentiment = current_flow['Sentiment']

        conn = tv_manager.get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                symbol,
                put_call_ratio,
                flow_sentiment,
                net_premium_flow,
                total_volume
            FROM options_flow
            WHERE flow_date = CURRENT_DATE
                AND symbol != %s
                AND ABS(put_call_ratio - %s) < 0.3
                AND flow_sentiment = %s
            ORDER BY ABS(put_call_ratio - %s)
            LIMIT 5
        """, (selected_symbol, target_pc, target_sentiment, target_pc))

        similar = cur.fetchall()
        cur.close()
        conn.close()

        if similar:
            df_similar = pd.DataFrame(similar, columns=[
                'Symbol', 'P/C Ratio', 'Sentiment', 'Net Flow', 'Volume'
            ])

            df_similar['P/C Ratio'] = df_similar['P/C Ratio'].apply(lambda x: f"{x:.2f}")
            df_similar['Net Flow'] = df_similar['Net Flow'].apply(lambda x: f"${x/1e6:.2f}M")
            df_similar['Volume'] = df_similar['Volume'].apply(lambda x: f"{x:,}")

            # PERFORMANCE: Add pagination for similar patterns
            paginated_similar = paginate_dataframe(df_similar, page_size=25, key_prefix="premium_flow_similar")
            st.dataframe(paginated_similar, width=None)
        else:
            st.info("No similar patterns found")


def display_strategies():
    """Display strategies tab"""

    st.subheader("📚 Options Flow Trading Strategies")

    st.markdown("""
    Options flow analysis tracks institutional money movement to identify high-probability trading opportunities.
    Here's how to use flow data for the wheel strategy:
    """)

    # Strategy 1
    with st.expander("✅ Strategy 1: Following Bullish Flow for CSPs", expanded=True):
        st.markdown("""
        **When to Use:** Strong bullish flow with low put/call ratios

        **Setup:**
        - Put/Call Ratio < 0.7
        - Net premium flow heavily favoring calls
        - Increasing flow trend over 7 days

        **Action:**
        - Sell cash-secured puts 5-10% below current price
        - Target 30-45 DTE
        - Delta ~0.30 for optimal probability

        **Why It Works:**
        - Institutional call buying suggests confidence
        - Lower put volume reduces downside risk
        - High premium collection with favorable odds

        **Example:**
        ```
        Stock: AAPL at $180
        Flow: P/C 0.65, $20M net call flow
        Trade: Sell $165 Put, 35 DTE, $300 premium
        Win Rate: ~70% based on flow
        ```
        """)

    # Strategy 2
    with st.expander("⚠️ Strategy 2: Interpreting Put/Call Ratios"):
        st.markdown("""
        **Understanding P/C Ratios:**

        | P/C Ratio | Sentiment | Action |
        |-----------|-----------|--------|
        | < 0.7 | Very Bullish | Aggressive CSPs |
        | 0.7 - 1.0 | Moderately Bullish | Standard CSPs |
        | 1.0 - 1.3 | Neutral | Conservative CSPs |
        | > 1.3 | Bearish | Avoid or Wait |

        **Key Insights:**
        - **Below 0.7:** Strong institutional confidence, good for CSPs
        - **0.7-1.0:** Healthy bullish flow, wheel strategy friendly
        - **1.0-1.3:** Balanced, use fundamental analysis
        - **Above 1.3:** Elevated risk, wait for stabilization

        **Important:** Always combine P/C ratio with net flow and trend direction
        """)

    # Strategy 3
    with st.expander("🚨 Strategy 3: Trading Unusual Activity"):
        st.markdown("""
        **Detecting Unusual Activity:**

        Unusual activity flags when:
        - Volume > 2x average
        - Volume > 2 standard deviations above mean
        - Significant premium flow spike

        **How to Trade:**

        **1. Verify the Signal:**
        - Check if unusual activity aligns with flow sentiment
        - Look for consistent direction (not mixed signals)
        - Confirm with 7-day trend

        **2. Position Sizing:**
        - **High Confidence:** Unusual + Bullish Flow → Standard position size
        - **Mixed Signals:** Unusual + Conflicting Flow → Reduce 50%
        - **Bearish Unusual:** Avoid or protective positions only

        **3. Entry Timing:**
        - Don't chase - wait for confirmation day
        - Enter on pullbacks if flow remains strong
        - Use limit orders, not market orders

        **Example Alert:**
        ```
        MSFT: Unusual Activity Detected
        - Volume: 85th percentile
        - P/C Ratio: 0.55 (very bullish)
        - Net Flow: +$15M calls

        Action: Sell $350 Put, collect $500 premium
        ```
        """)

    # Strategy 4
    with st.expander("💡 Strategy 4: Flow + Wheel Strategy Integration"):
        st.markdown("""
        **Combining Flow with Wheel Criteria:**

        **Standard Wheel Strategy:**
        - Delta ~0.30 strikes
        - 30-45 DTE
        - 1-2% weekly premium yield
        - Strong fundamentals

        **Enhanced with Flow:**
        - **Bullish Flow (P/C < 0.8):** Sell slightly closer to money (delta 0.35)
        - **Neutral Flow (P/C 0.8-1.2):** Standard delta 0.30
        - **Bearish Flow (P/C > 1.2):** More conservative delta 0.20 or wait

        **Flow-Based Adjustments:**

        | Flow Condition | Strike Adjustment | Premium Target |
        |----------------|------------------|----------------|
        | Strong Bullish | Closer to money | Higher premium |
        | Moderate Bullish | Standard | Standard |
        | Neutral | Farther OTM | Lower premium |
        | Bearish | Wait or skip | N/A |

        **Risk Management:**
        - Never override fundamentals for flow alone
        - Use flow as confirmation, not sole criteria
        - Reduce position size on conflicting signals
        """)

    # Strategy 5
    with st.expander("🎯 Strategy 5: Risk Management for Flow Trades"):
        st.markdown("""
        **Risk Assessment Framework:**

        **Low Risk:**
        - Consistent bullish flow 7+ days
        - P/C ratio stable < 0.8
        - Strong fundamentals
        - Low volatility

        **Medium Risk:**
        - Recent flow trend change
        - P/C ratio 0.8-1.2
        - Mixed fundamental picture
        - Moderate volatility

        **High Risk:**
        - Conflicting flow signals
        - Elevated P/C ratio > 1.3
        - Weak fundamentals
        - High volatility (IV Rank > 70)

        **Position Sizing:**
        ```
        Low Risk: 2-3% of portfolio per position
        Medium Risk: 1-2% of portfolio per position
        High Risk: 0.5-1% or avoid
        ```

        **Stop Loss Rules:**
        - Close CSP if stock drops 15% below strike
        - Exit if flow reverses dramatically (P/C +0.5)
        - Roll down if flow remains supportive
        """)

    # Strategy 6
    with st.expander("📊 Strategy 6: Case Studies - Successful Flow Trades"):
        st.markdown("""
        **Case Study 1: NVDA Bullish Flow (Example)**

        **Initial Setup:**
        - Date: Start of month
        - Price: $450
        - Flow: P/C 0.58, $50M net call premium
        - 7-Day Trend: Increasing

        **Trade:**
        - Sold $410 Put (delta 0.30)
        - 35 DTE
        - Premium: $800 per contract

        **Outcome:**
        - Stock stayed above $420
        - Put expired worthless
        - Return: 1.95% in 35 days (20% annualized)

        **Key Success Factor:** Flow remained bullish throughout

        ---

        **Case Study 2: Mixed Flow Signals (Example)**

        **Initial Setup:**
        - Stock: JPM at $150
        - Flow: P/C 1.1, but increasing call volume
        - Trend: Shifting from bearish to neutral

        **Trade Decision:**
        - Waited for confirmation
        - P/C dropped to 0.85 three days later
        - Then sold $140 Put (conservative)

        **Lesson:** Patience with mixed signals improved probability

        ---

        **Case Study 3: Avoiding Bearish Flow (Example)**

        **Setup:**
        - Stock XYZ at $75
        - Flow: P/C 1.6, heavy put buying
        - Unusual activity flag

        **Decision:**
        - Did NOT enter position
        - Stock dropped to $68 next week

        **Lesson:** Bearish flow warning prevented 10% loss
        """)

    st.markdown("---")
    st.info("""
    💡 **Pro Tip:** Always use options flow as a **confirmation tool** alongside fundamental analysis,
    technical indicators, and wheel strategy criteria. Flow shows institutional bias but doesn't
    guarantee outcomes. Combine multiple signals for highest probability trades.
    """)


if __name__ == "__main__":
    display_premium_flow_page()
