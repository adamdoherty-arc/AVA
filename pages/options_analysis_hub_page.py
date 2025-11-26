"""
Options Analysis Hub Page
Landing page to help users choose the right options analysis tool
"""

import streamlit as st


def render_options_analysis_hub_page():
    """Main hub for options analysis tools"""

    st.title("📊 Options Analysis Hub")
    st.markdown("**Choose the right tool for your analysis needs**")


    # Introduction
    st.markdown("""
    ### Welcome to the Options Analysis Hub

    We provide **two specialized tools** for options analysis, each designed for different workflows.
    Choose the tool that best fits your current goal:
    """)


    # Side-by-side tool comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 AI Options Agent")
        st.markdown("**Best for: Finding opportunities across many stocks**")

        st.markdown("**What it does:**")
        st.markdown("""
        - Scans 200+ stocks at once
        - Filters by DTE, delta, premium
        - AI-powered MCDM scoring
        - Ranks all opportunities
        - Shows STRONG_BUY recommendations
        """)

        st.markdown("**When to use:**")
        st.success("""
        - "What are the best CSP opportunities today?"
        - "Show me high-premium puts across my watchlist"
        - "I want to screen for trades quickly"
        - "Find me 10 best opportunities to review"
        """)

        st.markdown("**Key Features:**")
        st.markdown("""
        - Batch analysis (1-1000 stocks)
        - MCDM scoring (5 criteria)
        - Optional LLM reasoning
        - Historical tracking (1-30 days)
        - CSV export
        - Performance monitoring
        """)

        st.markdown("**Typical workflow:**")
        st.code("""
1. Select watchlist or "All Stocks"
2. Set filters (DTE 20-40, Delta -0.30)
3. Click "Run Analysis"
4. Review 200 ranked opportunities
5. Pick top 3-5 to trade
6. Export to CSV
        """, language="text")

        st.metric("Analysis Time", "30-60 seconds")
        st.metric("Stocks Analyzed", "200+")

        if st.button("🚀 Open AI Options Agent", key="open_ai_agent", use_container_width=True, type="primary"):
            st.session_state.page = "AI Options Agent"
            st.rerun()

    with col2:
        st.markdown("### 🎯 Comprehensive Strategy")
        st.markdown("**Best for: Choosing the right strategy for a stock**")

        st.markdown("**What it does:**")
        st.markdown("""
        - Analyzes ALL 10 strategies
        - Multi-model AI consensus
        - Ranks strategies by suitability
        - Provides educational details
        - Shows pros/cons for each
        """)

        st.markdown("**When to use:**")
        st.success("""
        - "I'm bullish on AAPL - which strategy?"
        - "Should I do CSP or Iron Condor on TSLA?"
        - "What's the best strategy for this environment?"
        - "I want to learn about different strategies"
        """)

        st.markdown("**Key Features:**")
        st.markdown("""
        - All 10 strategies analyzed
        - Market environment analysis
        - Multi-model AI voting
        - Strategy education tabs
        - Win rates & complexity
        - JSON export
        """)

        st.markdown("**Typical workflow:**")
        st.code("""
1. Select stock (AAPL)
2. Review auto-populated data
3. Click "Analyze ALL Strategies"
4. See top 3 recommendations
5. Read why each is ranked
6. Learn about all 10 strategies
        """, language="text")

        st.metric("Analysis Time", "10-30 seconds")
        st.metric("Strategies Analyzed", "All 10")

        if st.button("🎯 Open Comprehensive Strategy", key="open_comp_strat", use_container_width=True, type="primary"):
            st.session_state.page = "Comprehensive Strategy Analysis"
            st.rerun()


    # Comparison table
    st.markdown("### 📋 Quick Comparison")

    comparison_data = {
        "Feature": [
            "Primary Focus",
            "Stocks Analyzed",
            "Strategies Analyzed",
            "Analysis Type",
            "Best Use Case",
            "Output",
            "AI Models",
            "Time Required",
            "Learning Curve"
        ],
        "🤖 AI Options Agent": [
            "Find best opportunities",
            "1-1000 stocks",
            "1 strategy (CSP)",
            "Breadth (many stocks)",
            "Trade discovery",
            "Ranked list",
            "1 model (optional)",
            "30-60 seconds",
            "Easy"
        ],
        "🎯 Comprehensive Strategy": [
            "Choose best strategy",
            "1 stock at a time",
            "All 10 strategies",
            "Depth (all strategies)",
            "Strategy selection",
            "Ranked strategies",
            "3 models (consensus)",
            "10-30 seconds",
            "Educational"
        ]
    }

    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


    # Recommended workflows
    st.markdown("### 🔄 Recommended Workflows")

    tab1, tab2, tab3 = st.tabs(["📅 Weekly Planning", "📈 Bullish Stock", "🔍 Deep Research"])

    with tab1:
        st.markdown("**Workflow: Weekly Options Planning**")
        st.markdown("""
        **Goal**: Find the best opportunities for the week ahead

        **Step-by-step:**
        1. **Monday Morning**: Open AI Options Agent
        2. Select your TradingView watchlist
        3. Set filters: DTE 20-45, Min Premium $100
        4. Run analysis → Get 200 opportunities
        5. **For each STRONG_BUY**:
           - Switch to Comprehensive Strategy
           - Analyze that specific stock
           - Confirm CSP is the best strategy
           - Review environment fit
        6. Execute top 3-5 trades

        **Why this works:**
        - AI Agent finds the opportunities
        - Comprehensive Strategy validates the approach
        - You get both breadth AND depth
        """)

    with tab2:
        st.markdown("**Workflow: I'm Bullish on a Specific Stock**")
        st.markdown("""
        **Goal**: Decide which bullish strategy to use

        **Step-by-step:**
        1. Open Comprehensive Strategy
        2. Enter your stock (e.g., "NVDA")
        3. Review auto-filled data
        4. Click "Analyze ALL Strategies"
        5. Review market environment:
           - High IV → Premium selling strategies ranked higher
           - Low IV → Long strategies ranked higher
        6. See top 3 bullish strategies:
           - Cash-Secured Put
           - Bull Put Spread
           - PMCC (Poor Man's Covered Call)
        7. Read pros/cons for each
        8. Choose based on your capital and risk tolerance

        **Why this works:**
        - You get personalized strategy recommendations
        - AI consensus validates the choice
        - Educational content helps you understand WHY
        """)

    with tab3:
        st.markdown("**Workflow: Deep Research Mode**")
        st.markdown("""
        **Goal**: Thoroughly analyze a stock before committing capital

        **Step-by-step:**
        1. **Start with Comprehensive Strategy**:
           - Enter your stock
           - Get all 10 strategies ranked
           - Understand market environment
           - Note top 3 recommendations

        2. **Validate with AI Options Agent**:
           - Select your stock's watchlist
           - See how it ranks vs peers
           - Check if it appears in top 20
           - Compare scores with similar stocks

        3. **Check other pages**:
           - Earnings Calendar: Avoid earnings weeks
           - Sector Analysis: Understand sector trends
           - Premium Flow: See institutional activity

        4. **Make informed decision**:
           - Combine insights from all tools
           - Choose strategy with highest confidence

        **Why this works:**
        - Multiple data sources reduce blind spots
        - Cross-validation increases confidence
        - Comprehensive view before risk capital
        """)


    # Tips section
    st.markdown("### 💡 Pro Tips")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **AI Options Agent Tips:**
        - Use "All Stocks" for maximum opportunities
        - Filter by Min Premium to ensure profitability
        - Check historical top picks for backtest validation
        - Enable LLM reasoning for high-stakes trades
        - Export CSV for record-keeping
        """)

    with col2:
        st.info("""
        **Comprehensive Strategy Tips:**
        - Try Manual Input mode for any ticker
        - Use multi-model consensus for validation
        - Read strategy education tabs to learn
        - Check environment fit before executing
        - Download JSON reports for documentation
        """)


    # FAQ
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        **Q: Can I use both tools for the same stock?**
        A: Absolutely! In fact, we recommend it. Use AI Options Agent to find opportunities, then validate with Comprehensive Strategy.

        **Q: Which tool is faster?**
        A: Both are fast. AI Agent takes 30-60s for 200 stocks, Comprehensive Strategy takes 10-30s for one stock.

        **Q: Do they use the same data sources?**
        A: Yes, both use the same database (stock_data, stock_premiums) and TradingView watchlists.

        **Q: Which tool should beginners use?**
        A: Start with Comprehensive Strategy - it's educational and helps you understand different strategies.

        **Q: Can I analyze my entire portfolio?**
        A: Yes! Create a TradingView watchlist with your holdings, then use AI Options Agent to analyze all at once.

        **Q: Do I need API keys?**
        A: Not required. LLM reasoning is optional in AI Options Agent, and Comprehensive Strategy uses free providers.
        """)


    # Footer
    st.markdown("### 🚀 Ready to get started?")
    st.markdown("Choose a tool above to begin your analysis journey!")


# Run the page
if __name__ == "__main__":
    render_options_analysis_hub_page()
