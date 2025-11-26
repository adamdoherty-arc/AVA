"""
Prediction Markets Page - AI-Powered Event Contract Opportunities
Displays non-sports prediction markets from Kalshi with AI ratings
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List
from src.kalshi_integration import KalshiIntegration
from src.general_market_evaluator import GeneralMarketEvaluator

# PERFORMANCE: Cached Kalshi client - singleton pattern (using public elections API)
@st.cache_resource
def get_kalshi_client():
    """Get cached Kalshi API client (public elections API, no auth required)"""
    return KalshiIntegration()

# PERFORMANCE: Cached AI evaluator - singleton pattern
@st.cache_resource
def get_market_evaluator():
    """Get cached general market evaluator"""
    return GeneralMarketEvaluator()

# PERFORMANCE: Cached markets fetch
@st.cache_data(ttl=300, show_spinner=False)
def fetch_markets_cached(_client):
    """Fetch non-sports markets with 5-minute cache"""
    # Get all markets from public API
    all_markets = _client.get_markets(limit=1000, status='active')

    # Filter and categorize non-sports markets
    return categorize_non_sports_markets(all_markets)


def categorize_non_sports_markets(markets: List[Dict]) -> Dict[str, List[Dict]]:
    """Filter out sports and categorize remaining markets by sector"""

    # Sports keywords to exclude
    sports_keywords = [
        'nfl', 'nba', 'mlb', 'nhl', 'ncaa', 'ncaaf', 'ncaab',
        'football', 'basketball', 'baseball', 'hockey', 'soccer',
        'super bowl', 'playoffs', 'championship', 'world series',
        'game', 'match', 'vs', 'vs.', 'versus',
        'quarterback', 'touchdown', 'field goal', 'yards',
        'points scored', 'to win', 'will win', 'to beat', 'beats',
        'spread', 'over/under', 'prop', 'team', 'player'
    ]

    # Categorize by sector
    markets_by_sector = {
        'Politics': [],
        'Economics': [],
        'Crypto': [],
        'Tech': [],
        'Climate': [],
        'World': [],
        'Other': []
    }

    for market in markets:
        title = market.get('title', '').lower()
        subtitle = market.get('subtitle', '').lower()
        ticker = market.get('ticker', '').lower()
        category = market.get('category', '').lower()

        combined_text = f"{title} {subtitle} {ticker} {category}"

        # Skip sports markets
        if any(keyword in combined_text for keyword in sports_keywords):
            continue

        # Categorize by keywords
        if any(kw in combined_text for kw in ['election', 'president', 'congress', 'senate', 'vote', 'political', 'campaign', 'democrat', 'republican']):
            markets_by_sector['Politics'].append(market)
        elif any(kw in combined_text for kw in ['economy', 'gdp', 'inflation', 'unemployment', 'fed', 'interest rate', 'stock market', 'recession', 'dow', 's&p']):
            markets_by_sector['Economics'].append(market)
        elif any(kw in combined_text for kw in ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth', 'blockchain', 'coin']):
            markets_by_sector['Crypto'].append(market)
        elif any(kw in combined_text for kw in ['tech', 'ai', 'apple', 'google', 'microsoft', 'tesla', 'amazon', 'meta', 'nvidia', 'artificial intelligence']):
            markets_by_sector['Tech'].append(market)
        elif any(kw in combined_text for kw in ['climate', 'temperature', 'weather', 'hurricane', 'wildfire', 'emission', 'global warming']):
            markets_by_sector['Climate'].append(market)
        elif any(kw in combined_text for kw in ['war', 'conflict', 'international', 'country', 'global', 'china', 'russia', 'ukraine']):
            markets_by_sector['World'].append(market)
        else:
            markets_by_sector['Other'].append(market)

    return markets_by_sector


def categorize_all_markets(markets: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize ALL markets including sports"""

    # Categorize by sector (including Sports)
    markets_by_sector = {
        'Sports': [],
        'Politics': [],
        'Economics': [],
        'Crypto': [],
        'Tech': [],
        'Climate': [],
        'World': [],
        'Other': []
    }

    # Sports keywords
    sports_keywords = [
        'nfl', 'nba', 'mlb', 'nhl', 'ncaa', 'ncaaf', 'ncaab',
        'football', 'basketball', 'baseball', 'hockey', 'soccer',
        'super bowl', 'playoffs', 'championship', 'world series',
        'game', 'match', 'vs', 'vs.', 'versus',
        'quarterback', 'touchdown', 'field goal', 'yards',
        'points scored', 'to win', 'will win', 'to beat', 'beats',
        'spread', 'over/under', 'prop', 'team', 'player'
    ]

    for market in markets:
        title = market.get('title', '').lower()
        subtitle = market.get('subtitle', '').lower()
        ticker = market.get('ticker', '').lower()
        category = market.get('category', '').lower()

        combined_text = f"{title} {subtitle} {ticker} {category}"

        # Check if sports market
        if any(keyword in combined_text for keyword in sports_keywords):
            markets_by_sector['Sports'].append(market)
            continue

        # Categorize non-sports by keywords
        if any(kw in combined_text for kw in ['election', 'president', 'congress', 'senate', 'vote', 'political', 'campaign', 'democrat', 'republican']):
            markets_by_sector['Politics'].append(market)
        elif any(kw in combined_text for kw in ['economy', 'gdp', 'inflation', 'unemployment', 'fed', 'interest rate', 'stock market', 'recession', 'dow', 's&p']):
            markets_by_sector['Economics'].append(market)
        elif any(kw in combined_text for kw in ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth', 'blockchain', 'coin']):
            markets_by_sector['Crypto'].append(market)
        elif any(kw in combined_text for kw in ['tech', 'ai', 'apple', 'google', 'microsoft', 'tesla', 'amazon', 'meta', 'nvidia', 'artificial intelligence']):
            markets_by_sector['Tech'].append(market)
        elif any(kw in combined_text for kw in ['climate', 'temperature', 'weather', 'hurricane', 'wildfire', 'emission', 'global warming']):
            markets_by_sector['Climate'].append(market)
        elif any(kw in combined_text for kw in ['war', 'conflict', 'international', 'country', 'global', 'china', 'russia', 'ukraine']):
            markets_by_sector['World'].append(market)
        else:
            markets_by_sector['Other'].append(market)

    return markets_by_sector

def show_prediction_markets():
    """Main function to display prediction markets page"""

    st.title("🎲 Prediction Markets")
    st.caption("AI-powered non-sports opportunities from Kalshi")

    # Initialize integrations with cached singletons
    client = get_kalshi_client()
    evaluator = get_market_evaluator()

    # Filters in sidebar-style columns
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

    # First get the sports checkbox value
    with col4:
        include_sports = st.checkbox("Include Sports", value=False, help="Show sports markets with AI ratings")

    with col1:
        # Sector filter - include Sports if checkbox is enabled
        sector_options = ["All", "Politics", "Economics", "Crypto", "Tech", "Climate", "World", "Other"]
        # Add Sports option at the beginning if sports are included
        if include_sports:
            sector_options.insert(1, "Sports")

        sector_filter = st.selectbox("Sector", sector_options, index=0)

    with col2:
        min_score = st.number_input("Min Score", min_value=0, max_value=100, value=65, step=5)

    with col3:
        max_days = st.number_input("Max Days", min_value=1, max_value=365, value=90, step=1)

    with col5:
        if st.button("🔄 Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # Fetch markets from Kalshi API
    with st.spinner("Fetching markets from Kalshi..."):
        try:
            if include_sports:
                # Fetch all markets without sports filtering
                all_markets = client.get_markets(limit=1000, status='active')
                # Categorize all markets (including sports as "Sports" sector)
                markets_by_sector = categorize_all_markets(all_markets)
            else:
                # Fetch and filter non-sports markets (original behavior)
                markets_by_sector = fetch_markets_cached(client)
        except Exception as e:
            st.error(f"Failed to fetch markets: {e}")
            markets_by_sector = {}

    if not markets_by_sector or all(len(markets) == 0 for markets in markets_by_sector.values()):
        if include_sports:
            st.error("⚠️ Failed to fetch markets from Kalshi")
            st.info("Check your network connection and try refreshing the page.")
        else:
            st.warning("⚠️ No non-sports markets currently available")

            st.info("""
            **No Politics, Economics, or Crypto markets found.**

            The public API is currently sports-focused. To see markets and AI ratings in action:
            1. ✅ **Check "Include Sports"** above to see all markets with AI evaluation
            2. Wait for non-sports markets to become available (elections, economic events, etc.)

            The AI evaluation system works with any market type!
            """)

        return

    # Select markets based on filter
    if sector_filter == "All":
        all_markets = []
        for sector_markets in markets_by_sector.values():
            all_markets.extend(sector_markets)
        markets_to_evaluate = all_markets
    else:
        markets_to_evaluate = markets_by_sector.get(sector_filter, [])

    if not markets_to_evaluate:
        st.warning(f"No {sector_filter} markets found.")
        return

    # Evaluate markets with AI
    with st.spinner("Analyzing markets with AI..."):
        evaluated_markets = evaluator.evaluate_markets(markets_to_evaluate)

    # Filter by score and timing
    filtered_markets = [
        m for m in evaluated_markets
        if m.get('overall_score', 0) >= min_score
        and calculate_days_to_close(m.get('close_time', '')) <= max_days
        and calculate_days_to_close(m.get('close_time', '')) > 0
    ]

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Markets", len(evaluated_markets))
    with col2:
        st.metric("Excellent (≥80)", len([m for m in filtered_markets if m.get('overall_score', 0) >= 80]))
    with col3:
        avg_score = sum(m.get('overall_score', 0) for m in filtered_markets) / len(filtered_markets) if filtered_markets else 0
        st.metric("Avg Score", f"{avg_score:.1f}")
    with col4:
        st.metric("Showing", len(filtered_markets))


    # Display markets
    if not filtered_markets:
        st.info(f"No markets found with score ≥ {min_score} and ≤ {max_days} days to close.")
        return

    # Show top 25 markets
    for market in filtered_markets[:25]:
        display_market_card(market)

def display_market_card(market):
    """Display a single market opportunity card with AI rating"""

    ticker = market.get('ticker', 'Unknown')
    title = market.get('title', 'Unknown Market')
    sector = market.get('sector', 'Other')
    yes_price = market.get('yes_price', 0)
    overall_score = market.get('overall_score', 0)
    reasoning = market.get('reasoning', 'No analysis available')
    recommendation = market.get('recommended_action', 'PASS')
    days_to_close = calculate_days_to_close(market.get('close_time', ''))

    # Component scores
    value_score = market.get('value_score', 0)
    liquidity_score = market.get('liquidity_score', 0)
    timing_score = market.get('timing_score', 0)
    clarity_score = market.get('clarity_score', 0)
    momentum_score = market.get('momentum_score', 0)

    # Score color coding
    if overall_score >= 85:
        score_emoji = "🔥"
        score_color = "#00C851"  # Green
    elif overall_score >= 75:
        score_emoji = "⭐"
        score_color = "#33B5E5"  # Blue
    elif overall_score >= 65:
        score_emoji = "👍"
        score_color = "#FFBB33"  # Yellow
    else:
        score_emoji = "⚠️"
        score_color = "#FF4444"  # Red

    # Recommendation styling
    rec_styles = {
        'BUY_YES': ('🟢', '#00C851'),
        'BUY_NO': ('🔵', '#33B5E5'),
        'WATCH': ('🟡', '#FFBB33'),
        'PASS': ('⚪', '#999999')
    }
    rec_emoji, rec_color = rec_styles.get(recommendation, ('⚪', '#999999'))

    # Create expandable card
    header = f"{score_emoji} **{overall_score:.0f}/100** | {sector} | {days_to_close}d | {title}"

    with st.expander(header, expanded=False):
        # Header row with key info
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.markdown(f"**Ticker:** `{ticker}`")
        with col2:
            st.markdown(f"**YES Price:** {yes_price*100:.1f}%")
        with col3:
            st.markdown(f"**Action:** {rec_emoji} {recommendation.replace('_', ' ')}")
        with col4:
            st.markdown(f"**Days Left:** {days_to_close}")


        # AI Reasoning
        st.markdown("**🤖 AI Analysis:**")
        st.info(reasoning)


        # Component Scores
        st.markdown("**📊 Score Breakdown:**")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Value", f"{value_score:.0f}/100")
        with col2:
            st.metric("Liquidity", f"{liquidity_score:.0f}/100")
        with col3:
            st.metric("Timing", f"{timing_score:.0f}/100")
        with col4:
            st.metric("Clarity", f"{clarity_score:.0f}/100")
        with col5:
            st.metric("Momentum", f"{momentum_score:.0f}/100")

        # Progress bar for overall score
        st.progress(overall_score / 100, f"Overall AI Rating: {overall_score:.1f}/100")

# Keep old fetch function for backwards compatibility (now unused)
@st.cache_data(ttl=300)
def fetch_and_score_markets_old(_db, _evaluator, category, limit=50):
    """Fetch markets from database with AI predictions"""
    try:
        # Fetch markets with predictions from database
        market_type = None
        if category == "Sports":
            market_type = "nfl"  # We only have NFL for now

        # PERFORMANCE: Get markets with predictions (already scored by AI)
        markets_with_predictions = _db.get_top_opportunities(limit=limit)

        if markets_with_predictions:
            # Convert to format expected by display
            scored_markets = []
            for m in markets_with_predictions:
                scored_market = {
                    'ticker': m.get('ticker', ''),
                    'title': m.get('title', ''),
                    'category': 'Sports' if m.get('market_type') == 'nfl' else 'Other',
                    'yes_price': float(m.get('yes_price', 0)) if m.get('yes_price') else 0,
                    'no_price': float(m.get('no_price', 0)) if m.get('no_price') else 0,
                    'volume_24h': float(m.get('volume', 0)) if m.get('volume') else 0,
                    'days_to_close': calculate_days_to_close(m.get('close_time')),
                    'ai_score': float(m.get('confidence_score', 0)) if m.get('confidence_score') else 0,
                    'ai_reasoning': m.get('reasoning', 'No analysis available'),
                    'recommended_position': m.get('recommended_action', 'pass').upper(),
                    'risk_level': get_risk_level(m.get('confidence_score', 0)),
                    'expected_value': float(m.get('edge_percentage', 0)) if m.get('edge_percentage') else 0,
                    'bid_ask_spread': 0,  # Not available in current data
                    'close_date': m.get('close_time', ''),
                    'description': m.get('title', '')
                }
                scored_markets.append(scored_market)
            return scored_markets

        # If no predictions, get raw markets and analyze them
        active_markets = _db.get_active_markets(market_type=market_type)

        if not active_markets:
            return []

        # Generate predictions if needed
        predictions = _evaluator.evaluate_markets(active_markets[:limit])

        # Store predictions in database
        if predictions:
            _db.store_predictions(predictions)

        # Convert to display format
        scored_markets = []
        for i, pred in enumerate(predictions):
            market = active_markets[i] if i < len(active_markets) else {}
            scored_market = {
                'ticker': pred.get('ticker', ''),
                'title': market.get('title', ''),
                'category': 'Sports',
                'yes_price': float(market.get('yes_price', 0)) if market.get('yes_price') else 0,
                'no_price': float(market.get('no_price', 0)) if market.get('no_price') else 0,
                'volume_24h': float(market.get('volume', 0)) if market.get('volume') else 0,
                'days_to_close': calculate_days_to_close(market.get('close_time')),
                'ai_score': float(pred.get('confidence_score', 0)),
                'ai_reasoning': pred.get('reasoning', ''),
                'recommended_position': pred.get('recommended_action', 'pass').upper(),
                'risk_level': get_risk_level(pred.get('confidence_score', 0)),
                'expected_value': float(pred.get('edge_percentage', 0)),
                'bid_ask_spread': 0,
                'close_date': market.get('close_time', ''),
                'description': market.get('title', '')
            }
            scored_markets.append(scored_market)

        return scored_markets

    except Exception as e:
        st.error(f"Error fetching markets: {e}")
        import traceback
        st.code(traceback.format_exc())
        return []

def calculate_days_to_close(close_time):
    """Calculate days until market closes"""
    if not close_time:
        return 0
    try:
        from datetime import datetime, timezone
        import re

        if isinstance(close_time, str):
            # Fix timezone format: -05 -> -05:00
            close_time_fixed = re.sub(r'([-+]\d{2})$', r'\1:00', close_time)
            # Also handle Z suffix
            close_time_fixed = close_time_fixed.replace('Z', '+00:00')
            close_dt = datetime.fromisoformat(close_time_fixed)
        else:
            close_dt = close_time

        # Convert to UTC for consistent calculation
        if close_dt.tzinfo is not None:
            close_dt_utc = close_dt.astimezone(timezone.utc)
        else:
            close_dt_utc = close_dt.replace(tzinfo=timezone.utc)

        now_utc = datetime.now(timezone.utc)
        delta = (close_dt_utc - now_utc).total_seconds() / 86400  # Convert to days
        return max(0, int(delta))
    except Exception as e:
        # Log error for debugging
        import traceback
        print(f"Error parsing close_time '{close_time}': {e}")
        traceback.print_exc()
        return 0

def get_risk_level(confidence):
    """Get risk level based on confidence score"""
    if confidence >= 80:
        return "Low"
    elif confidence >= 60:
        return "Medium"
    else:
        return "High"

