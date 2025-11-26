"""
AI Options Agent Page
Streamlit UI for AI-powered options analysis and recommendations
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

# Import AI Options Agent components
from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent
from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

# Import shared components
from src.ai_options_agent.shared.stock_selector import StockSelector
from src.ai_options_agent.shared.llm_config_ui import LLMConfigUI
from src.ai_options_agent.shared.data_fetchers import fetch_database_stocks
from src.ai_options_agent.shared.display_helpers import (
    display_score_gauge,
    display_recommendation_badge
)

# PERFORMANCE: Cached database manager - singleton pattern
@st.cache_resource
def get_ai_options_db_manager():
    """Cached AI Options database manager"""
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager
    return AIOptionsDBManager()

# PERFORMANCE: Cached top recommendations query
@st.cache_data(ttl=300)
def get_top_recommendations_cached(_agent, days=1, min_score=50):
    """Get top recommendations with 5-minute cache"""
    return _agent.get_top_recommendations(days=days, min_score=min_score)

# PERFORMANCE: Cached all agents performance query
@st.cache_data(ttl=300)
def get_all_agents_performance_cached(_db_manager):
    """Get all agents performance with 5-minute cache"""
    return _db_manager.get_all_agents_performance()

def render_ai_options_agent_page():
    """Main UI for AI Options Agent"""

    st.title("🤖 AI Options Agent")

    st.markdown("""
    **AI-Powered Options Analysis** using Multi-Criteria Decision Making (MCDM) + LLM Reasoning

    Scan hundreds of stocks at once to find the best cash-secured put opportunities using advanced multi-criteria scoring.
    """)
    
    # Sync status widget
    from src.components.sync_status_widget import SyncStatusWidget
    sync_widget = SyncStatusWidget()
    sync_widget.display(
        table_name="stock_premiums",
        title="Options Data Sync",
        compact=True
    )

    # Initialize LLM manager first
    if 'llm_manager' not in st.session_state:
        from src.ai_options_agent.llm_manager import get_llm_manager
        st.session_state.llm_manager = get_llm_manager()

    # Initialize agent with LLM manager
    if 'ai_agent' not in st.session_state:
        with st.spinner("Initializing AI Options Agent..."):
            st.session_state.ai_agent = OptionsAnalysisAgent(
                llm_manager=st.session_state.llm_manager
            )
            # PERFORMANCE: Use cached database manager
            st.session_state.db_manager = get_ai_options_db_manager()

    agent = st.session_state.ai_agent
    db_manager = st.session_state.db_manager
    llm_manager = st.session_state.llm_manager

    # === LLM PROVIDER SECTION ===
    llm_config = LLMConfigUI(llm_manager)
    selected_provider = llm_config.render_provider_selector(
        show_add_provider=True,
        allow_manual_selection=True
    )

    # === ANALYSIS SETTINGS ===
    st.subheader("⚙️ Analysis Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Analysis source
        analysis_source = st.radio(
            "Analyze:",
            ["All Stocks", "TradingView Watchlist"],
            help="Choose whether to analyze all stocks or a specific watchlist"
        )

        # Watchlist selection using shared component
        watchlist_name = None
        symbols = None
        if analysis_source == "TradingView Watchlist":
            selector = StockSelector()
            watchlist_name, symbols = selector.render_watchlist_selector()

    with col2:
        # DTE Range
        st.markdown("**Days to Expiration (DTE)**")
        min_dte = st.number_input("Min DTE", 1, 90, 20, 1)
        max_dte = st.number_input("Max DTE", 1, 90, 40, 1)

        # Min Premium
        min_premium = st.number_input("Min Premium ($)", 0.0, 1000.0, 100.0, 10.0)

    with col3:
        # Delta Range
        st.markdown("**Delta Range (for puts)**")
        min_delta = st.number_input("Min Delta", -0.50, -0.01, -0.45, 0.01)
        max_delta = st.number_input("Max Delta", -0.50, -0.01, -0.15, 0.01)

        # Max Results
        max_results = st.number_input("Max Results", 10, 1000, 200, 50,
                                     help="Maximum number of opportunities to analyze (default: 200, max: 1000)")

    # Display Filters
    st.markdown("**Display Filters**")
    col1, col2 = st.columns(2)
    with col1:
        min_score_display = st.slider("Min Score to Display", 0, 100, 50, 5,
                                      help="Only show results with score above this threshold")
    with col2:
        use_llm_reasoning = st.checkbox("🤖 Use LLM Reasoning", value=False,
                                       help="Add AI-generated reasoning (slower, uses API)")

    # Run Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "🏆 Top Picks", "📈 Performance"])

    # Tab 1: Analysis Results
    with tab1:
        # PERFORMANCE: Load cached analyses on page load (if not running new analysis)
        if 'current_analyses' not in st.session_state and not run_analysis:
            # Try to load recent analyses from database with caching
            cached_analyses = get_top_recommendations_cached(agent, days=1, min_score=min_score_display)
            if cached_analyses:
                st.session_state.current_analyses = cached_analyses
                st.info(f"📂 Loaded {len(cached_analyses)} cached analyses from database (last 24 hours)")

        if run_analysis:
            with st.spinner("🤖 AI Agent is analyzing opportunities..."):
                start_time = time.time()

                # Run analysis based on source
                if analysis_source == "TradingView Watchlist" and watchlist_name:
                    analyses = agent.analyze_watchlist(
                        watchlist_name=watchlist_name,
                        dte_range=(min_dte, max_dte),
                        delta_range=(min_delta, max_delta),
                        min_premium=min_premium,
                        limit=max_results,
                        use_llm=use_llm_reasoning,
                        llm_provider=selected_provider
                    )
                else:
                    analyses = agent.analyze_all_stocks(
                        dte_range=(min_dte, max_dte),
                        delta_range=(min_delta, max_delta),
                        min_premium=min_premium,
                        limit=max_results,
                        use_llm=use_llm_reasoning,
                        llm_provider=selected_provider
                    )

                elapsed_time = time.time() - start_time

                # Filter by min score
                analyses = [a for a in analyses if a['final_score'] >= min_score_display]

                # Store in session state
                st.session_state.current_analyses = analyses
                st.session_state.analysis_time = elapsed_time

            st.success(f"✅ Analysis complete! Analyzed {len(analyses)} opportunities in {elapsed_time:.1f}s")

        # Display results
        if 'current_analyses' in st.session_state and st.session_state.current_analyses:
            analyses = st.session_state.current_analyses

            st.subheader(f"📊 Found {len(analyses)} Opportunities")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            strong_buys = len([a for a in analyses if a['recommendation'] == 'STRONG_BUY'])
            buys = len([a for a in analyses if a['recommendation'] == 'BUY'])
            avg_score = sum(a['final_score'] for a in analyses) / len(analyses)

            col1.metric("STRONG BUY", strong_buys)
            col2.metric("BUY", buys)
            col3.metric("Average Score", f"{avg_score:.0f}/100")
            col4.metric("Total Analyzed", len(analyses))


            # Display each analysis
            for analysis in analyses:
                with st.expander(
                    f"{analysis['symbol']} - {display_recommendation_badge(analysis['recommendation'])} - Score: {analysis['final_score']}/100",
                    expanded=analysis.get('recommendation') in ['STRONG_BUY', 'BUY']
                ):
                    # Key Info
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Strike", f"${analysis.get('strike_price', 0):.2f}")
                    col2.metric("DTE", analysis.get('dte', 'N/A'))
                    col3.metric("Premium", f"${analysis.get('premium', 0)/100:.2f}")
                    col4.metric("Monthly Return", f"{analysis.get('monthly_return', 0):.2f}%")

                    # Scores breakdown
                    st.markdown("**Score Breakdown:**")
                    score_cols = st.columns(5)
                    score_cols[0].metric("📊 Fundamental", f"{analysis['fundamental_score']}")
                    score_cols[1].metric("📈 Technical", f"{analysis['technical_score']}")
                    score_cols[2].metric("🎯 Greeks", f"{analysis['greeks_score']}")
                    score_cols[3].metric("⚠️ Risk", f"{analysis['risk_score']}")
                    score_cols[4].metric("💭 Sentiment", f"{analysis['sentiment_score']}")

                    # Strategy & Confidence
                    st.markdown(f"**Strategy:** {analysis.get('strategy', 'N/A')}")
                    st.markdown(f"**Confidence:** {analysis.get('confidence', 0)}%")

                    # LLM Model Info
                    llm_model = analysis.get('llm_model', 'rule_based_v1')
                    if llm_model != 'rule_based_v1':
                        st.markdown(f"**🤖 AI Model:** `{llm_model}` ({analysis.get('llm_tokens_used', 0)} tokens)")

                    # Reasoning
                    st.markdown("**Analysis:**")
                    st.info(analysis.get('reasoning', 'No reasoning available'))

                    # Risks and Opportunities
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**⚠️ Key Risks:**")
                        st.warning(analysis.get('key_risks', 'None identified'))
                    with col2:
                        st.markdown("**✨ Key Opportunities:**")
                        st.success(analysis.get('key_opportunities', 'None identified'))

                    # Greeks detail
                    with st.expander("📊 Detailed Greeks"):
                        delta = analysis.get('delta')
                        iv = analysis.get('iv')

                        detail_cols = st.columns(3)
                        detail_cols[0].metric("Delta", f"{delta:.3f}" if delta else "N/A")
                        detail_cols[1].metric("IV", f"{iv*100:.1f}%" if iv else "N/A")
                        detail_cols[2].metric("Annual Return", f"{analysis.get('annual_return', 0):.1f}%")
        else:
            st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to begin")

    # Tab 2: Top Picks
    with tab2:
        st.subheader("🏆 Top AI-Recommended Picks")

        # Filters for top picks
        lookback_days = st.selectbox("Lookback Period", [1, 3, 7, 14, 30], index=2)

        # PERFORMANCE: Use cached query
        top_picks = get_top_recommendations_cached(agent, days=lookback_days, min_score=75)

        if top_picks:
            # Create DataFrame
            df_data = []
            for pick in top_picks[:20]:  # Show top 20
                df_data.append({
                    'Symbol': pick.get('symbol'),
                    'Score': pick.get('final_score'),
                    'Recommendation': pick.get('recommendation'),
                    'Strike': f"${pick.get('strike_price', 0):.2f}",
                    'DTE': pick.get('dte'),
                    'Premium': f"${pick.get('premium', 0)/100:.2f}",
                    'Monthly %': f"{pick.get('monthly_return', 0):.2f}%",
                    'Annual %': f"{pick.get('annual_return', 0):.1f}%",
                    'Confidence': f"{pick.get('confidence', 0)}%",
                    'Strategy': pick.get('strategy', 'CSP')
                })

            df = pd.DataFrame(df_data)

            # Style the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"No recommendations found in the last {lookback_days} days. Run an analysis to generate recommendations.")

    # Tab 3: Performance Tracking
    with tab3:
        st.subheader("📈 AI Agent Performance")
        st.markdown("""
        **Coming Soon:**
        - Accuracy tracking (predicted vs actual outcomes)
        - Win rate by score threshold
        - Average P&L by recommendation type
        - Agent learning and improvement metrics

        *Performance tracking will be enabled after you start executing trades based on AI recommendations.*
        """)

        # PERFORMANCE: Show cached agent performance if available
        all_agents_perf = get_all_agents_performance_cached(db_manager)
        if all_agents_perf:
            st.markdown("**Current Agent Performance:**")
            perf_data = []
            for agent_perf in all_agents_perf:
                perf_data.append({
                    'Agent': agent_perf.get('agent_name'),
                    'Total Predictions': agent_perf.get('total_predictions', 0),
                    'Correct': agent_perf.get('total_correct', 0),
                    'Accuracy': f"{agent_perf.get('avg_accuracy_rate', 0):.1f}%"
                })

            if perf_data:
                st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
        else:
            st.info("No performance data yet. Performance tracking begins after trade outcomes are recorded.")


# Run the page
if __name__ == "__main__":
    render_ai_options_agent_page()
