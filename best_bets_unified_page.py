"""
Best Bets Across All Sports - Unified Ranking System

Shows the best betting opportunities across ALL sports (NFL, NCAA, NBA, etc.)
ranked by profitability using the OpportunityScorer.

This answers the question: "Which bets have the best odds of making money?"
"""

import streamlit as st
import os
from datetime import datetime
from src.betting.best_bets_ranker import BestBetsRanker
from src.betting.filters import (
    BettingFilterPanel, 
    ConfidenceFilter, 
    ExpectedValueFilter, 
    SportFilter
)

# Page config
st.set_page_config(page_title="Best Bets Across All Sports", page_icon="🏆", layout="wide")

# Title
st.title("🏆 Best Bets Across All Sports")
st.markdown("**Unified ranking system to find the most profitable betting opportunities across all markets**")

# Database connection
DB_CONNECTION_STRING = f"dbname={os.getenv('DB_NAME', 'sports_trading')} " \
                      f"user={os.getenv('DB_USER', 'postgres')} " \
                      f"password={os.getenv('DB_PASSWORD', '')} " \
                      f"host={os.getenv('DB_HOST', 'localhost')} " \
                      f"port={os.getenv('DB_PORT', '5432')}"

# Initialize ranker
@st.cache_resource
def get_ranker():
    return BestBetsRanker(DB_CONNECTION_STRING)

try:
    ranker = get_ranker()

    # Sidebar filters
    st.sidebar.header("🎯 Filters")

    # Get sport summary
    sport_summary = ranker.get_sport_summary()

    if sport_summary:
        st.sidebar.markdown("**Available Opportunities:**")
        for sport, count in sport_summary.items():
            st.sidebar.markdown(f"- **{sport}**: {count} bets")
        st.sidebar.markdown("---")

    # Filter controls
    top_n = st.sidebar.slider(
        "Number of bets to show",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Show top N ranked opportunities"
    )

    # Create and render betting filter panel
    # Replaces manual EV, Confidence, and Sport widgets
    panel = BettingFilterPanel()
    
    # Add filters with custom settings to match original page
    panel.add_filter(ExpectedValueFilter(
        label="Minimum EV %", 
        min_val=0.0, 
        max_val=30.0, 
        default=5.0,
        step=1.0
    ))
    
    panel.add_filter(ConfidenceFilter(
        label="Minimum Confidence",
        min_val=0,
        max_val=100,
        default=60,
        step=5
    ))
    
    # Sport filter needs dynamic options from summary
    all_sports = list(sport_summary.keys()) if sport_summary else ['NFL', 'NCAA', 'NBA']
    panel.add_filter(SportFilter(
        label="Sports to include",
        options=all_sports,
        multi_select=True
    ))
    
    # Render panel in sidebar
    panel_values = panel.render(key_prefix="best_bets", layout="sidebar")

    max_age_hours = st.sidebar.slider(
        "Max odds age (hours)",
        min_value=1,
        max_value=48,
        value=24,
        step=1,
        help="Only show odds updated within this many hours"
    )

    # Fetch button
    if st.sidebar.button("🔄 Refresh Rankings", type="primary"):
        st.cache_resource.clear()
        st.rerun()

    # Fetch best bets
    with st.spinner("🔍 Analyzing opportunities across all sports..."):
        # Extract values from panel (using filter names as keys)
        # ExpectedValueFilter -> 'expectedvalue'
        # ConfidenceFilter -> 'confidence'
        # SportFilter -> 'sport'
        
        min_ev_pct = panel_values.get('expectedvalue', 5.0)
        min_confidence = panel_values.get('confidence', 60)
        sport_selection = panel_values.get('sport', all_sports)
        
        best_bets = ranker.get_best_bets(
            top_n=top_n,
            min_ev=min_ev_pct / 100.0,
            min_confidence=min_confidence,
            sports_filter=sport_selection if sport_selection else None,
            max_age_hours=max_age_hours
        )

    # Display results
    if not best_bets:
        st.warning("⚠️ No opportunities found matching your filters. Try relaxing the criteria.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Opportunities",
                len(best_bets),
                help="Number of bets meeting your criteria"
            )

        with col2:
            avg_ev = sum(b['score_details']['ev_percentage'] for b in best_bets) / len(best_bets)
            st.metric(
                "Avg Expected Value",
                f"{avg_ev:.1f}%",
                help="Average EV across all opportunities"
            )

        with col3:
            avg_score = sum(b['total_score'] for b in best_bets) / len(best_bets)
            st.metric(
                "Avg Opportunity Score",
                f"{avg_score:.1f}/100",
                help="Average overall quality score"
            )

        with col4:
            sports_represented = len(set(b['sport'] for b in best_bets))
            st.metric(
                "Sports Represented",
                sports_represented,
                help="Number of different sports in results"
            )

        st.markdown("---")

        # Display each opportunity
        for idx, bet in enumerate(best_bets, 1):
            score_details = bet['score_details']
            total_score = bet['total_score']
            rating = score_details['rating']

            # Color code by rating
            if rating == "Exceptional":
                color = "#00FF00"  # Green
                emoji = "🔥"
            elif rating == "Great":
                color = "#90EE90"  # Light green
                emoji = "✅"
            elif rating == "Good":
                color = "#FFD700"  # Gold
                emoji = "👍"
            elif rating == "Decent":
                color = "#FFA500"  # Orange
                emoji = "⚠️"
            else:
                color = "#FF6347"  # Red
                emoji = "❌"

            # Create expander for each bet
            with st.expander(
                f"#{idx} {emoji} **{bet['sport']}** | {bet['team']} | "
                f"Score: {total_score:.1f}/100 ({rating}) | "
                f"EV: {score_details['ev_percentage']:.1f}%",
                expanded=(idx <= 5)  # Auto-expand top 5
            ):
                # Two columns: Game info and Scoring details
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.markdown(f"### 🎮 Game Information")
                    st.markdown(f"**Sport:** {bet['sport']}")
                    st.markdown(f"**Game:** {bet['game_info']}")
                    st.markdown(f"**Betting on:** {bet['team']}")
                    st.markdown(f"**Market:** {bet.get('market_type', 'Winner')}")

                    if bet.get('game_time'):
                        st.markdown(f"**Game Time:** {bet['game_time'].strftime('%a %m/%d %I:%M %p')}")

                    if bet.get('last_updated'):
                        age_minutes = (datetime.now() - bet['last_updated']).total_seconds() / 60
                        st.markdown(f"**Odds Updated:** {age_minutes:.0f} min ago")

                    st.markdown("---")
                    st.markdown(f"### 💰 Betting Details")
                    st.markdown(f"**AI Win Probability:** {bet['ai_win_prob']*100:.1f}%")
                    st.markdown(f"**Market Price:** {bet['market_price']*100:.1f}¢")
                    st.markdown(f"**Expected Value:** <span style='color:{color};font-weight:bold;font-size:1.2em'>{score_details['ev_percentage']:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"**Edge (AI vs Market):** {score_details['edge_percentage']:.1f}%")

                    if bet.get('market_volume'):
                        st.markdown(f"**Market Volume:** ${bet['market_volume']:,.0f}")

                with col_right:
                    st.markdown(f"### 📊 Opportunity Score Breakdown")

                    # Overall score with color
                    st.markdown(
                        f"**Overall Score:** <span style='color:{color};font-weight:bold;font-size:1.5em'>"
                        f"{total_score:.1f}/100</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Rating:** {rating}")
                    st.markdown(f"**Recommendation:** {score_details['recommendation']}")

                    st.markdown("---")
                    st.markdown("**Component Scores:**")

                    # Progress bars for each component
                    components = [
                        ("EV Score (40%)", score_details['ev_score'], 40),
                        ("Confidence (25%)", score_details['confidence_score'], 25),
                        ("Edge Size (15%)", score_details['edge_score'], 15),
                        ("Liquidity (10%)", score_details['liquidity_score'], 10),
                        ("Recency (10%)", score_details['recency_score'], 10)
                    ]

                    for name, score, weight in components:
                        # Color code progress bar
                        if score >= 80:
                            bar_color = "green"
                        elif score >= 60:
                            bar_color = "orange"
                        else:
                            bar_color = "red"

                        st.markdown(f"**{name}:** {score:.1f}/100")
                        st.progress(score / 100, )

                # Add quick action button
                st.markdown("---")
                if bet.get('market_ticker'):
                    st.info(f"📍 **Kalshi Ticker:** `{bet['market_ticker']}`")

        st.markdown("---")
        st.markdown("### 📖 How Scoring Works")
        st.markdown("""
        Each opportunity is scored 0-100 based on:

        - **Expected Value (40%)**: How profitable the bet is
        - **Confidence (25%)**: How reliable the AI prediction is
        - **Edge Size (15%)**: How much the AI disagrees with the market
        - **Liquidity (10%)**: Market volume (higher = easier to place bet)
        - **Recency (10%)**: How fresh the odds are

        **Ratings:**
        - 🔥 **Exceptional (90-100)**: Top tier opportunities - Strong BUY
        - ✅ **Great (75-89)**: Highly recommended - BUY
        - 👍 **Good (60-74)**: Recommended - BUY
        - ⚠️ **Decent (40-59)**: Consider carefully
        - ❌ **Poor (0-39)**: Avoid
        """)

except Exception as e:
    st.error(f"❌ Error loading best bets: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown("*Rankings update based on latest AI predictions and market odds*")
