"""
Two-Mode Options Analysis Page
MODE 1: Batch Analysis - Scan & rank 100+ stocks with paginated results
MODE 2: Individual Stock Analysis - Deep dive into single stock with all strategies
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import time
from typing import Optional, Dict, Any, List

# Import AI Options Agent components
try:
    from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager
    from src.ai_options_agent.comprehensive_strategy_analyzer import ComprehensiveStrategyAnalyzer
    from src.ai_options_agent.shared.stock_selector import StockSelector
    from src.ai_options_agent.shared.llm_config_ui import LLMConfigUI
    from src.ai_options_agent.shared.display_helpers import (
        display_score_gauge,
        display_recommendation_badge
    )
    from src.components.paginated_table import PaginatedTable
    from src.components.stock_dropdown import StockDropdown, WatchlistSelector

    AI_AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"AI Options Agent components not found: {e}")
    AI_AGENT_AVAILABLE = False


def render_options_analysis_page():
    """Main function to display two-mode options analysis"""

    st.title("🎯 Options Analysis")

    if not AI_AGENT_AVAILABLE:
        st.error("Required components not available. Please check installation.")
        st.info("Run: `pip install -r requirements.txt`")
        return

    # ==================== INITIALIZATION ====================

    # Cached database manager
    @st.cache_resource
    def get_ai_options_db_manager():
        """Cached AI Options database manager"""
        return AIOptionsDBManager()

    # Initialize LLM manager
    if 'llm_manager' not in st.session_state:
        try:
            from src.ai_options_agent.llm_manager import get_llm_manager
            st.session_state.llm_manager = get_llm_manager()
        except Exception as e:
            st.warning(f"LLM manager not available: {e}")
            st.session_state.llm_manager = None

    # Initialize agent with LLM manager
    if 'ai_agent' not in st.session_state:
        with st.spinner("Initializing AI Options Agent..."):
            st.session_state.ai_agent = OptionsAnalysisAgent(
                llm_manager=st.session_state.llm_manager
            )
            st.session_state.db_manager = get_ai_options_db_manager()

    agent = st.session_state.ai_agent
    db_manager = st.session_state.db_manager
    llm_manager = st.session_state.llm_manager

    # ==================== MODE SELECTOR ====================

    mode = st.radio(
        "📋 Analysis Mode",
        ["🔍 Batch Analysis (Scan & Rank)", "📊 Individual Stock Deep Dive"],
        horizontal=True,
        help="Choose analysis mode: Batch scans 100+ stocks, Individual analyzes one stock in detail"
    )

    # ==================== MODE 1: BATCH ANALYSIS ====================

    if mode == "🔍 Batch Analysis (Scan & Rank)":
        render_batch_analysis_mode(agent, db_manager, llm_manager)

    # ==================== MODE 2: INDIVIDUAL STOCK ====================

    else:
        render_individual_stock_mode(agent, db_manager, llm_manager)


def render_batch_analysis_mode(agent, db_manager, llm_manager):
    """
    MODE 1: Batch Analysis - Scan multiple stocks and rank them

    Features:
    - Scan All Stocks or specific watchlist
    - Paginated results table (not expandable cards)
    - Sortable by any column
    - Export to CSV
    - View Details button for each row
    """

    st.subheader("🔍 Batch Analysis Mode")
    st.caption("Scan and rank 100+ stocks by AI score - View results in paginated table")

    # === SETTINGS SECTION ===

    with st.expander("⚙️ Analysis Settings", expanded=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Data Source**")

            analysis_source = st.radio(
                "Analyze:",
                ["All Stocks", "TradingView Watchlist"],
                help="Choose whether to analyze all stocks or a specific watchlist",
                key="batch_source"
            )

            # Watchlist selection
            watchlist_name = None
            symbols = None
            if analysis_source == "TradingView Watchlist":
                watchlist_selector = WatchlistSelector(db_manager)
                watchlist_name, symbols = watchlist_selector.render(key="batch_watchlist")

                if symbols:
                    st.info(f"📊 {len(symbols)} stocks in '{watchlist_name}'")

        with col2:
            st.markdown("**Options Filters**")

            min_dte = st.number_input("Min DTE", 1, 90, 20, 1, key="batch_min_dte")
            max_dte = st.number_input("Max DTE", 1, 90, 40, 1, key="batch_max_dte")
            min_premium = st.number_input(
                "Min Premium ($)",
                0.0, 1000.0, 100.0, 10.0,
                key="batch_min_premium"
            )

        with col3:
            st.markdown("**Greeks & Display**")

            min_delta = st.number_input(
                "Min Delta",
                -0.50, -0.01, -0.45, 0.01,
                key="batch_min_delta",
                help="Minimum delta for puts (negative value)"
            )
            max_delta = st.number_input(
                "Max Delta",
                -0.50, -0.01, -0.15, 0.01,
                key="batch_max_delta",
                help="Maximum delta for puts (negative value)"
            )

            max_results = st.number_input(
                "Max Results",
                10, 1000, 200, 50,
                key="batch_max_results",
                help="Maximum number of opportunities to analyze"
            )

        # Display Filters
        col1, col2 = st.columns(2)
        with col1:
            min_score_display = st.slider(
                "Min Score to Display",
                0, 100, 50, 5,
                key="batch_min_score",
                help="Only show results with score above this threshold"
            )
        with col2:
            use_llm_reasoning = st.checkbox(
                "🤖 Use LLM Reasoning",
                value=False,
                key="batch_use_llm",
                help="Add AI-generated reasoning (slower, uses API)",
                disabled=(llm_manager is None)
            )

    # === LLM PROVIDER SECTION ===

    selected_provider = None
    if use_llm_reasoning and llm_manager:
        with st.expander("🤖 LLM Provider Settings"):
            llm_config = LLMConfigUI(llm_manager)
            selected_provider = llm_config.render_provider_selector(
                show_add_provider=True,
                allow_manual_selection=True
            )

    # === RUN ANALYSIS BUTTON ===

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button(
            "🚀 Run Batch Analysis",
            type="primary",
            use_container_width=True,
            key="batch_run"
        )

    # === RESULTS SECTION ===


    if run_analysis:
        with st.spinner("🤖 AI Agent analyzing opportunities..."):
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
            st.session_state.batch_analyses = analyses
            st.session_state.batch_analysis_time = elapsed_time

        st.success(f"✅ Analysis complete! Found {len(analyses)} opportunities in {elapsed_time:.1f}s")

    # Display results
    if 'batch_analyses' in st.session_state and st.session_state.batch_analyses:
        analyses = st.session_state.batch_analyses

        st.subheader(f"📊 Batch Analysis Results ({len(analyses)} opportunities)")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        strong_buys = len([a for a in analyses if a['recommendation'] == 'STRONG_BUY'])
        buys = len([a for a in analyses if a['recommendation'] == 'BUY'])
        avg_score = sum(a['final_score'] for a in analyses) / len(analyses) if analyses else 0

        col1.metric("🎯 STRONG BUY", strong_buys)
        col2.metric("✅ BUY", buys)
        col3.metric("📊 Avg Score", f"{avg_score:.0f}/100")
        col4.metric("📈 Total", len(analyses))


        # Convert to DataFrame for paginated table
        df_data = []
        for analysis in analyses:
            df_data.append({
                'Symbol': analysis.get('symbol'),
                'Score': analysis.get('final_score'),
                'Recommendation': analysis.get('recommendation'),
                'Strike': f"${analysis.get('strike_price', 0):.2f}",
                'DTE': analysis.get('dte'),
                'Premium': f"${analysis.get('premium', 0)/100:.2f}",
                'Monthly %': f"{analysis.get('monthly_return', 0):.2f}%",
                'Annual %': f"{analysis.get('annual_return', 0):.1f}%",
                'Delta': f"{analysis.get('delta', 0):.3f}",
                'Confidence': f"{analysis.get('confidence', 0)}%"
            })

        df = pd.DataFrame(df_data)

        # Define action callback for View Details button
        def view_details_callback(row_data: Dict[str, Any]):
            """Callback when View Details is clicked"""
            symbol = row_data.get('Symbol')
            # Find the full analysis
            for analysis in analyses:
                if analysis.get('symbol') == symbol:
                    st.session_state.selected_batch_analysis = analysis
                    st.rerun()

        # Render paginated table
        table = PaginatedTable(
            df=df,
            key_prefix="batch_results",
            page_size=20,
            show_export=True,
            show_page_size_selector=True,
            sortable_columns=None,  # All columns sortable
            action_column={
                'label': 'Details',
                'button_label': '🔍 View',
                'callback': view_details_callback
            }
        )

        table.render()

    else:
        st.info("👈 Configure settings and click 'Run Batch Analysis' to begin")

    # Display selected analysis details in modal
    if 'selected_batch_analysis' in st.session_state:
        display_analysis_details(st.session_state.selected_batch_analysis)

        if st.button("✖️ Close Details", key="close_batch_details"):
            del st.session_state.selected_batch_analysis
            st.rerun()


def render_individual_stock_mode(agent, db_manager, llm_manager):
    """
    MODE 2: Individual Stock Analysis - Deep dive into one stock

    Features:
    - Select ONE stock from dropdown
    - Shows ALL 18 option strategies (10 core + 8 modern)
    - Display all 5 scorer breakdowns
    - Show reasoning, risks, opportunities
    - Show Greeks details
    - LLM reasoning optional
    """

    st.subheader("📊 Individual Stock Deep Dive")
    st.caption("Analyze all 18 option strategies for a single stock with detailed scoring breakdown")

    # === STOCK SELECTOR ===

    col1, col2 = st.columns([2, 1])

    with col1:
        stock_selector = StockDropdown(db_manager)
        selected_symbol = stock_selector.render(
            label="Select Stock to Analyze",
            key="individual_stock",
            show_metadata=True
        )

    with col2:
        if selected_symbol:
            st.metric("Selected", selected_symbol)
        else:
            if st.button("🔄 Refresh Stock List", key="refresh_stocks"):
                st.cache_data.clear()
                st.rerun()

    if not selected_symbol:
        return

    # === ANALYSIS SETTINGS ===

    with st.expander("⚙️ Analysis Settings", expanded=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**DTE Range**")
            min_dte = st.number_input("Min DTE", 1, 90, 20, 1, key="indiv_min_dte")
            max_dte = st.number_input("Max DTE", 1, 90, 40, 1, key="indiv_max_dte")

        with col2:
            st.markdown("**Delta Range**")
            min_delta = st.number_input(
                "Min Delta",
                -0.50, -0.01, -0.45, 0.01,
                key="indiv_min_delta"
            )
            max_delta = st.number_input(
                "Max Delta",
                -0.50, -0.01, -0.15, 0.01,
                key="indiv_max_delta"
            )

        with col3:
            st.markdown("**Other Filters**")
            min_premium = st.number_input(
                "Min Premium ($)",
                0.0, 1000.0, 50.0, 10.0,
                key="indiv_min_premium"
            )

            use_llm = st.checkbox(
                "🤖 Use LLM Reasoning",
                value=False,
                key="indiv_use_llm",
                disabled=(llm_manager is None)
            )

    # === LLM PROVIDER ===

    selected_provider = None
    if use_llm and llm_manager:
        with st.expander("🤖 LLM Provider"):
            llm_config = LLMConfigUI(llm_manager)
            selected_provider = llm_config.render_provider_selector()

    # === ANALYZE BUTTON ===

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            f"🔬 Analyze {selected_symbol}",
            type="primary",
            use_container_width=True,
            key="indiv_analyze"
        )

    # === RESULTS ===

    if analyze_button:
        with st.spinner(f"🤖 Analyzing all option strategies for {selected_symbol}..."):

            # Get stock and options data
            opportunities = db_manager.get_opportunities(
                symbols=[selected_symbol],
                dte_range=(min_dte, max_dte),
                delta_range=(min_delta, max_delta),
                min_premium=min_premium,
                limit=1  # Just need one to get stock data
            )

            if not opportunities:
                st.warning(f"No option data found for {selected_symbol} with current filters")
                return

            # Get first opportunity for data
            opp = opportunities[0]

            # Prepare stock data (convert decimals to float)
            current_price = float(opp.get('current_price', 0) or opp.get('stock_price', 100))
            stock_data = {
                'symbol': selected_symbol,
                'current_price': current_price,
                'iv': float(opp.get('iv', 0.35)),
                'price_52w_high': current_price * 1.2,  # Estimate if not available
                'price_52w_low': current_price * 0.8,
                'market_cap': float(opp.get('market_cap', 0) or 0),
                'pe_ratio': float(opp.get('pe_ratio', 25) or 25),
                'sector': opp.get('sector', 'Unknown')
            }

            # Prepare options data (convert decimals to float)
            options_data = {
                'strike_price': float(opp.get('strike_price', current_price * 0.95)),
                'dte': int(opp.get('dte', 30)),
                'delta': float(opp.get('delta', -0.30) or -0.30),
                'premium': float(opp.get('premium', 0) or 0) / 100  # Convert cents to dollars
            }

            # Run comprehensive strategy analysis
            strategy_analyzer = ComprehensiveStrategyAnalyzer()
            analysis = strategy_analyzer.analyze_stock(
                symbol=selected_symbol,
                stock_data=stock_data,
                options_data=options_data
            )

            st.session_state.comprehensive_analysis = analysis
            st.session_state.individual_symbol = selected_symbol

        st.success(f"✅ Analyzed all 18 strategies for {selected_symbol}")

    # Display comprehensive analysis results
    if 'comprehensive_analysis' in st.session_state:
        analysis = st.session_state.comprehensive_analysis
        symbol = st.session_state.individual_symbol

        st.subheader(f"📊 {symbol} - Comprehensive Strategy Analysis")

        # === MARKET ENVIRONMENT ===
        st.markdown("### 🌍 Market Environment")

        env = analysis.get('environment_analysis', {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Volatility Regime", env.get('volatility_regime', 'N/A').upper())
        col2.metric("Trend", env.get('trend', 'N/A').replace('_', ' ').title())
        iv_display = env.get('iv', 0.35) * 100
        col3.metric("Implied Volatility", f"{iv_display:.1f}%")
        col4.metric("Market Regime", env.get('market_regime', 'N/A').replace('_', ' ').title())


        # === TOP 3 STRATEGIES ===
        st.markdown("### 🏆 Top 3 Recommended Strategies")

        strategies = analysis.get('strategy_rankings', [])

        if strategies:
            for idx, strategy in enumerate(strategies[:3]):
                with st.expander(
                    f"#{idx+1}: {strategy.get('name', 'Unknown')} - Score: {strategy.get('score', 0)}/100",
                    expanded=(idx == 0)
                ):
                    # Strategy overview
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Score", f"{strategy.get('score', 0)}/100")
                    col2.metric("Type", strategy.get('type', 'N/A'))
                    col3.metric("Win Rate", strategy.get('win_rate', 'N/A'))

                    st.markdown(f"**Outlook:** {strategy.get('outlook', 'N/A')}")
                    st.markdown(f"**Best When:** {strategy.get('best_when', 'N/A')}")
                    st.markdown(f"**Risk Profile:** {strategy.get('risk_profile', 'N/A')}")

                    st.markdown("### 📋 Trade Details")

                    metrics = strategy.get('metrics', {})
                    legs = metrics.get('legs', [])

                    # Display trade legs
                    st.markdown("**Trade Legs:**")
                    for leg in legs:
                        action = leg.get('action', '')
                        leg_type = leg.get('type', '')
                        strike = leg.get('strike', 0)
                        premium = leg.get('premium', 0)
                        expiration = leg.get('expiration', 'N/A')

                        action_emoji = '🔴' if action == 'SELL' else '🟢' if action == 'BUY' else '⚪'

                        if premium > 0:
                            st.markdown(f"- {action_emoji} **{action}** {leg.get('quantity', 1)} {leg_type} @ ${strike:.2f} (${premium:.2f} premium) - Exp: {expiration}")
                        else:
                            st.markdown(f"- {action_emoji} **{action}** {leg.get('quantity', 1)} {leg_type} @ ${strike:.2f} - Exp: {expiration}")

                    # Risk/Reward metrics
                    st.markdown("**Risk/Reward:**")
                    col1, col2, col3 = st.columns(3)

                    max_profit = metrics.get('max_profit', 0)
                    max_loss = metrics.get('max_loss', 0)
                    capital_required = metrics.get('capital_required', 0)

                    with col1:
                        if isinstance(max_profit, (int, float)):
                            st.metric("Max Profit", f"${max_profit:.2f}")
                        else:
                            st.metric("Max Profit", str(max_profit))

                    with col2:
                        if isinstance(max_loss, (int, float)):
                            st.metric("Max Loss", f"${max_loss:.2f}")
                        else:
                            st.metric("Max Loss", str(max_loss))

                    with col3:
                        if isinstance(capital_required, (int, float)):
                            st.metric("Capital Required", f"${capital_required:.2f}")
                        else:
                            st.metric("Capital Required", str(capital_required))

                    # Breakeven
                    st.markdown("**Breakeven:**")
                    if 'breakeven_lower' in metrics and 'breakeven_upper' in metrics:
                        st.info(f"Lower: ${metrics['breakeven_lower']:.2f} | Upper: ${metrics['breakeven_upper']:.2f}")
                    elif 'breakeven' in metrics:
                        breakeven = metrics['breakeven']
                        if isinstance(breakeven, (int, float)):
                            st.info(f"${breakeven:.2f}")
                        else:
                            st.info(str(breakeven))

                    # Return on Capital
                    return_on_capital = metrics.get('return_on_capital', 0)
                    if isinstance(return_on_capital, (int, float)):
                        st.metric("Return on Capital", f"{return_on_capital:.2f}%")

                    # Probability of Profit
                    prob_profit = metrics.get('probability_profit', 0)
                    if prob_profit:
                        st.metric("Probability of Profit", f"~{prob_profit}%")


            # === ALL 18 STRATEGIES TABLE ===
            st.markdown("### 📊 All 18 Strategies Ranked")

            strategy_df = pd.DataFrame([
                {
                    'Rank': idx + 1,
                    'Strategy': s.get('name', 'Unknown'),
                    'Score': s.get('score', 0),
                    'Type': s.get('type', 'N/A'),
                    'Win Rate': s.get('win_rate', 'N/A'),
                    'Outlook': s.get('outlook', 'N/A')
                }
                for idx, s in enumerate(strategies)
            ])

            st.dataframe(strategy_df, use_container_width=True, hide_index=True)

        else:
            st.warning("No strategies analyzed")

    else:
        st.info("👆 Click 'Analyze' to begin comprehensive strategy analysis")


def display_analysis_details(analysis: Dict[str, Any]):
    """
    Display detailed analysis in a modal/expander

    Args:
        analysis: Analysis dictionary
    """

    with st.expander(f"🔍 Detailed Analysis: {analysis.get('symbol')}", expanded=True):

        # Key Info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strike", f"${analysis.get('strike_price', 0):.2f}")
        col2.metric("DTE", analysis.get('dte', 'N/A'))
        col3.metric("Premium", f"${analysis.get('premium', 0)/100:.2f}")
        col4.metric("Monthly %", f"{analysis.get('monthly_return', 0):.2f}%")

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
            detail_cols[2].metric("Annual %", f"{analysis.get('annual_return', 0):.1f}%")


# Main entry point
if __name__ == "__main__":
    render_options_analysis_page()
