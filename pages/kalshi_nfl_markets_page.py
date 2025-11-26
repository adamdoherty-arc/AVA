"""
Kalshi NFL Prediction Markets - Modern Dashboard with Enhanced Game Cards
Feature-rich interface for NFL prediction market analysis and tracking
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.betting.filters import (
    BettingFilterPanel, 
    ConfidenceFilter, 
    ExpectedValueFilter, 
    DateRangeFilter
)

from src.services import get_kalshi_manager  # Use centralized service registry
from src.kalshi_ai_evaluator import KalshiAIEvaluator

# Import Telegram notifier
try:
    from src.telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    TelegramNotifier = None


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

def configure_page():
    """Configure Streamlit page settings and custom styling"""
    st.set_page_config(
        page_title="Kalshi NFL Markets",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for modern, professional look
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 0rem 1rem;
        }

        /* Metric cards styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            color: white;
            margin-bottom: 1rem;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }

        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Market card styling */
        .market-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .market-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }

        /* Game card header with team logos */
        .game-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .team-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Team logo button styling */
        .team-logo-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: 3px solid transparent;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 2.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .team-logo-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .team-logo-btn.user-pick {
            border-color: #10b981;
            background: #ecfdf5;
            box-shadow: 0 0 0 2px #10b981;
        }

        .team-logo-btn.ai-pick {
            border-color: #3b82f6;
            background: #eff6ff;
            box-shadow: 0 0 0 2px #3b82f6;
        }

        /* Live indicator */
        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .live-indicator.live {
            background: #ef4444;
            color: white;
            animation: pulse 2s ease-in-out infinite;
        }

        .live-indicator.scheduled {
            background: #f3f4f6;
            color: #6b7280;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Kalshi odds display */
        .kalshi-odds {
            display: flex;
            gap: 1rem;
            padding: 1rem;
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            border-radius: 8px;
            margin: 1rem 0;
        }

        .odds-item {
            flex: 1;
            text-align: center;
            padding: 0.75rem;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .odds-label {
            font-size: 0.75rem;
            color: #6b7280;
            text-transform: uppercase;
            font-weight: 600;
        }

        .odds-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f2937;
        }

        .odds-value.yes {
            color: #10b981;
        }

        .odds-value.no {
            color: #ef4444;
        }

        /* Pick legend */
        .pick-legend {
            display: flex;
            gap: 1rem;
            justify-content: center;
            padding: 0.75rem;
            background: #f9fafb;
            border-radius: 6px;
            margin: 1rem 0;
            font-size: 0.875rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .legend-dot.user {
            background: #10b981;
        }

        .legend-dot.ai {
            background: #3b82f6;
        }

        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            font-weight: 600;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }

        /* Score badge styling */
        .score-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .score-excellent {
            background-color: #10b981;
            color: white;
        }

        .score-good {
            background-color: #3b82f6;
            color: white;
        }

        .score-fair {
            background-color: #f59e0b;
            color: white;
        }

        .score-poor {
            background-color: #ef4444;
            color: white;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Responsive design */
        @media (max-width: 768px) {
            .metric-value {
                font-size: 1.8rem;
            }
            .team-logo-btn {
                width: 50px;
                height: 50px;
                font-size: 2rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize session state for user picks and Telegram"""
    if 'user_picks' not in st.session_state:
        st.session_state.user_picks = {}  # {market_ticker: team_name}

    if 'telegram_notifier' not in st.session_state and TELEGRAM_AVAILABLE:
        try:
            st.session_state.telegram_notifier = TelegramNotifier()
        except Exception as e:
            st.session_state.telegram_notifier = None

    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []


# ============================================================================
# TELEGRAM INTEGRATION
# ============================================================================

def send_team_selection_notification(team_name: str, opponent: str, game_time: str, ticker: str):
    """Send Telegram notification when user selects a team"""
    if not TELEGRAM_AVAILABLE or not st.session_state.get('telegram_notifier'):
        return False

    try:
        notifier = st.session_state.telegram_notifier
        message = f"📊 Watching {team_name} vs {opponent} - Game at {game_time}\n\nMarket: `{ticker}`"

        message_id = notifier.send_custom_message(message)
        return message_id is not None
    except Exception as e:
        st.warning(f"Could not send Telegram notification: {e}")
        return False


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

class MarketDataManager:
    """
    Manages market data fetching and caching.

    Uses centralized service registry for database manager to ensure
    singleton behavior and efficient resource utilization.
    """

    def __init__(self):
        self.db = get_kalshi_manager()  # Use centralized registry
        self.evaluator = KalshiAIEvaluator()

    @st.cache_data(ttl=300)
    def get_all_markets(_self) -> pd.DataFrame:
        """Fetch all NFL markets with predictions"""
        try:
            markets = _self.db.get_markets_with_predictions(market_type='nfl', limit=1000)

            if not markets:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(markets)

            # Add calculated fields
            df['days_to_close'] = df['close_time'].apply(_self._calculate_days_to_close)
            df['implied_probability'] = df['yes_price'] * 100
            df['edge_pct'] = df['edge_percentage'].fillna(0)
            df['confidence'] = df['confidence_score'].fillna(0)

            # Extract team names and bet types from title
            df['bet_type'] = df['title'].apply(_self._extract_bet_type)
            df['player_name'] = df['title'].apply(_self._extract_player_name)

            # Risk level
            df['risk_level'] = df['confidence'].apply(_self._get_risk_level)

            # Determine if game is live
            df['is_live'] = df['game_date'].apply(_self._is_game_live)

            return df

        except Exception as e:
            st.error(f"Error fetching markets: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=60)
    def get_price_history(_self, ticker: str) -> pd.DataFrame:
        """Fetch price history for a specific market"""
        try:
            conn = _self.db.get_connection()
            query = """
                SELECT snapshot_time, yes_price, no_price, volume
                FROM kalshi_price_history
                WHERE ticker = %s
                ORDER BY snapshot_time ASC
            """
            df = pd.read_sql(query, conn, params=(ticker,))
            conn.close()
            return df
        except Exception as e:
            return pd.DataFrame()

    @st.cache_data(ttl=600)
    def get_team_list(_self, df: pd.DataFrame) -> List[str]:
        """Extract unique team list from markets"""
        teams = set()
        for _, row in df.iterrows():
            if row['home_team']:
                teams.add(row['home_team'])
            if row['away_team']:
                teams.add(row['away_team'])
        return sorted(list(teams))

    @staticmethod
    def _calculate_days_to_close(close_time) -> int:
        """Calculate days until market closes"""
        if pd.isna(close_time):
            return 0
        try:
            if isinstance(close_time, str):
                close_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
            else:
                close_dt = close_time
            now = datetime.now(close_dt.tzinfo) if close_dt.tzinfo else datetime.now()
            delta = (close_dt - now).total_seconds() / 86400
            return max(0, int(delta))
        except:
            return 0

    @staticmethod
    def _is_game_live(game_date) -> bool:
        """Determine if game is currently live"""
        if pd.isna(game_date):
            return False
        try:
            if isinstance(game_date, str):
                game_dt = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
            else:
                game_dt = game_date
            now = datetime.now(game_dt.tzinfo) if game_dt.tzinfo else datetime.now()

            # Game is live if it started within the last 4 hours
            time_diff = (now - game_dt).total_seconds() / 3600
            return 0 <= time_diff <= 4
        except:
            return False

    @staticmethod
    def _extract_bet_type(title: str) -> str:
        """Extract bet type from market title"""
        title_lower = title.lower()
        if 'spread' in title_lower or 'cover' in title_lower:
            return 'Spread'
        elif 'total' in title_lower or 'over' in title_lower or 'under' in title_lower:
            return 'Total'
        elif 'moneyline' in title_lower or 'win' in title_lower:
            return 'Moneyline'
        elif 'parlay' in title_lower:
            return 'Parlay'
        elif 'touchdown' in title_lower or 'yards' in title_lower or 'reception' in title_lower:
            return 'Player Prop'
        else:
            return 'Other'

    @staticmethod
    def _extract_player_name(title: str) -> Optional[str]:
        """Extract player name from title (simple heuristic)"""
        # Look for common patterns like "Patrick Mahomes to throw..."
        words = title.split()
        for i, word in enumerate(words):
            if word[0].isupper() and i + 1 < len(words) and words[i+1][0].isupper():
                return f"{word} {words[i+1]}"
        return None

    @staticmethod
    def _get_risk_level(confidence: float) -> str:
        """Determine risk level based on confidence"""
        if confidence >= 80:
            return "Low"
        elif confidence >= 60:
            return "Medium"
        else:
            return "High"


# ============================================================================
# TEAM LOGO HELPERS
# ============================================================================

def get_team_emoji(team_name: str) -> str:
    """Get emoji representation for NFL team"""
    team_emojis = {
        # AFC East
        'Bills': '🦬', 'Dolphins': '🐬', 'Patriots': '🇺🇸', 'Jets': '✈️',
        # AFC North
        'Ravens': '🦅', 'Bengals': '🐅', 'Browns': '🟤', 'Steelers': '⚒️',
        # AFC South
        'Texans': '🤠', 'Colts': '🐴', 'Jaguars': '🐆', 'Titans': '⚔️',
        # AFC West
        'Chiefs': '👑', 'Chargers': '⚡', 'Raiders': '☠️', 'Broncos': '🐴',
        # NFC East
        'Cowboys': '⭐', 'Eagles': '🦅', 'Giants': '🏈', 'Commanders': '🎖️',
        # NFC North
        'Packers': '🧀', 'Vikings': '⚔️', 'Bears': '🐻', 'Lions': '🦁',
        # NFC South
        'Falcons': '🦅', 'Panthers': '🐆', 'Saints': '⚜️', 'Buccaneers': '🏴‍☠️',
        # NFC West
        '49ers': '🌉', 'Seahawks': '🦅', 'Rams': '🐏', 'Cardinals': '🦅',
    }

    # Try to match team name
    for team, emoji in team_emojis.items():
        if team.lower() in team_name.lower():
            return emoji

    return '🏈'  # Default


def format_game_time(game_date) -> str:
    """Format game date/time for display"""
    if pd.isna(game_date):
        return "TBD"

    try:
        if isinstance(game_date, str):
            game_dt = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
        else:
            game_dt = game_date

        return game_dt.strftime("%a %I:%M %p")
    except:
        return "TBD"


# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

class ChartBuilder:
    """Builds interactive charts for market visualization"""

    @staticmethod
    def create_odds_movement_chart(price_history: pd.DataFrame, title: str) -> go.Figure:
        """Create line chart showing odds movement over time"""
        fig = go.Figure()

        if not price_history.empty:
            fig.add_trace(go.Scatter(
                x=price_history['snapshot_time'],
                y=price_history['yes_price'] * 100,
                mode='lines+markers',
                name='Yes Price',
                line=dict(color='#10b981', width=3),
                marker=dict(size=6)
            ))

            fig.add_trace(go.Scatter(
                x=price_history['snapshot_time'],
                y=price_history['no_price'] * 100,
                mode='lines+markers',
                name='No Price',
                line=dict(color='#ef4444', width=3),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price (%)",
            height=350,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def create_volume_chart(df: pd.DataFrame) -> go.Figure:
        """Create bar chart showing volume by market"""
        top_volume = df.nlargest(10, 'volume')

        fig = go.Figure(data=[
            go.Bar(
                x=top_volume['title'].str[:40] + '...',
                y=top_volume['volume'],
                marker_color='#667eea',
                text=top_volume['volume'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside'
            )
        ])

        fig.update_layout(
            title="Top 10 Markets by Volume",
            xaxis_title="Market",
            yaxis_title="Volume ($)",
            height=400,
            template='plotly_white',
            xaxis_tickangle=-45
        )

        return fig

    @staticmethod
    def create_confidence_distribution(df: pd.DataFrame) -> go.Figure:
        """Create histogram of confidence scores"""
        fig = go.Figure(data=[
            go.Histogram(
                x=df['confidence'],
                nbinsx=20,
                marker_color='#764ba2',
                opacity=0.7
            )
        ])

        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Number of Markets",
            height=350,
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_opportunity_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create heatmap showing best opportunities by team and bet type"""
        # Aggregate by team and bet type
        heatmap_data = df[df['home_team'].notna()].groupby(['home_team', 'bet_type'])['edge_pct'].mean().unstack(fill_value=0)

        if heatmap_data.empty:
            return go.Figure()

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=heatmap_data.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Edge %")
        ))

        fig.update_layout(
            title="Opportunity Heatmap: Edge % by Team & Bet Type",
            xaxis_title="Bet Type",
            yaxis_title="Team",
            height=500,
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_edge_scatter(df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of edge vs confidence"""
        fig = go.Figure(data=[
            go.Scatter(
                x=df['confidence'],
                y=df['edge_pct'],
                mode='markers',
                marker=dict(
                    size=df['volume'] / 100,
                    color=df['confidence'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Confidence"),
                    line=dict(width=0.5, color='white')
                ),
                text=df['title'].str[:50],
                hovertemplate='<b>%{text}</b><br>Confidence: %{x:.1f}<br>Edge: %{y:.1f}%<extra></extra>'
            )
        ])

        fig.update_layout(
            title="Market Quality: Edge vs Confidence",
            xaxis_title="Confidence Score",
            yaxis_title="Edge Percentage",
            height=450,
            template='plotly_white'
        )

        return fig


# ============================================================================
# FILTER SIDEBAR
# ============================================================================

def render_filter_sidebar(df: pd.DataFrame, data_manager: MarketDataManager) -> Dict:
    """Render sidebar with comprehensive filtering options"""
    st.sidebar.header("🔍 Filters & Search")

    filters = {}

    # ========== Search ==========
    st.sidebar.subheader("🔎 Search")
    search_query = st.sidebar.text_input(
        "Search markets or players",
        placeholder="Enter team, player name, or keyword...",
        help="Search in market titles"
    )
    filters['search'] = search_query

    # ========== Team Filter ==========
    st.sidebar.subheader("🏈 Teams")
    team_list = data_manager.get_team_list(df)
    selected_teams = st.sidebar.multiselect(
        "Filter by teams",
        options=team_list,
        default=[],
        help="Select one or more teams"
    )
    filters['teams'] = selected_teams

    # ========== Bet Type Filter ==========
    st.sidebar.subheader("🎯 Bet Type")
    bet_types = df['bet_type'].unique().tolist()
    selected_bet_types = st.sidebar.multiselect(
        "Filter by bet type",
        options=bet_types,
        default=bet_types,
        help="Select bet types to include"
    )
    filters['bet_types'] = selected_bet_types

    # ========== Standard Filters (Confidence, Edge, Time) ==========
    # Use shared filter library for these standard filters
    panel = BettingFilterPanel()
    panel.add_filter(ConfidenceFilter(default=60))
    panel.add_filter(ExpectedValueFilter(default=0.0, min_value=-10.0, max_value=20.0))
    panel.add_filter(DateRangeFilter(presets=['All', 'Today', 'Tomorrow', 'This Week', 'This Month']))
    
    # Render panel in sidebar
    panel_values = panel.render(key_prefix="kalshi_sidebar", layout="sidebar")
    filters.update(panel_values)
    
    # Store panel in filters for use in apply_filters
    filters['_panel'] = panel

    # ========== Risk Level ==========
    st.sidebar.subheader("⚠️ Risk Level")
    risk_levels = st.sidebar.multiselect(
        "Filter by risk level",
        options=['Low', 'Medium', 'High'],
        default=['Low', 'Medium'],
        help="Based on AI confidence"
    )
    filters['risk_levels'] = risk_levels

    # ========== Actions ==========
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Filters", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    return filters


# ============================================================================
# APPLY FILTERS
# ============================================================================

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply selected filters to dataframe"""
    filtered_df = df.copy()

    # Search filter
    if filters.get('search'):
        search_term = filters['search'].lower()
        filtered_df = filtered_df[
            filtered_df['title'].str.lower().str.contains(search_term, na=False) |
            filtered_df['player_name'].str.lower().str.contains(search_term, na=False).fillna(False)
        ]

    # Team filter
    if filters.get('teams'):
        filtered_df = filtered_df[
            filtered_df['home_team'].isin(filters['teams']) |
            filtered_df['away_team'].isin(filters['teams'])
        ]

    # Bet type filter
    if filters.get('bet_types'):
        filtered_df = filtered_df[filtered_df['bet_type'].isin(filters['bet_types'])]

    # Apply standard filters using the panel
    if '_panel' in filters:
        panel = filters['_panel']
        filtered_df = panel.apply(filtered_df, filters)
    else:
        # Fallback if panel not present (shouldn't happen)
        pass

    # Risk level filter
    if filters.get('risk_levels'):
        filtered_df = filtered_df[filtered_df['risk_level'].isin(filters['risk_levels'])]

    return filtered_df


# ============================================================================
# DASHBOARD METRICS
# ============================================================================

def render_dashboard_metrics(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render top-level dashboard metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Markets",
            f"{len(df):,}",
            delta=f"{len(filtered_df)} shown"
        )

    with col2:
        high_confidence = len(filtered_df[filtered_df['confidence'] >= 80])
        st.metric(
            "High Confidence",
            f"{high_confidence:,}",
            delta=f"{high_confidence/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%"
        )

    with col3:
        avg_edge = filtered_df['edge_pct'].mean() if len(filtered_df) > 0 else 0
        st.metric(
            "Avg Edge",
            f"{avg_edge:.2f}%",
            delta="Positive" if avg_edge > 0 else "Negative"
        )

    with col4:
        total_volume = filtered_df['volume'].sum()
        st.metric(
            "Total Volume",
            f"${total_volume:,.0f}",
            delta=f"{len(filtered_df[filtered_df['volume'] > 1000])} liquid"
        )

    with col5:
        closing_today = len(filtered_df[filtered_df['days_to_close'] == 0])
        st.metric(
            "Closing Today",
            f"{closing_today:,}",
            delta="Urgent" if closing_today > 0 else "None"
        )


# ============================================================================
# ENHANCED MARKET CARDS WITH TEAM LOGOS
# ============================================================================

def render_enhanced_market_card(market: pd.Series, data_manager: MarketDataManager):
    """Render enhanced market card with clickable team logos and Kalshi odds"""

    # Determine score badge
    confidence = market['confidence']
    if confidence >= 80:
        badge_class = "score-excellent"
        badge_icon = "🔥"
    elif confidence >= 70:
        badge_class = "score-good"
        badge_icon = "⭐"
    elif confidence >= 60:
        badge_class = "score-fair"
        badge_icon = "👍"
    else:
        badge_class = "score-poor"
        badge_icon = "👎"

    # Create card header with game info
    header = f"{badge_icon} **{market['title'][:80]}{'...' if len(market['title']) > 80 else ''}**"

    with st.expander(header, expanded=False):
        # ========== GAME HEADER WITH TEAM LOGOS ==========
        if market['home_team'] and market['away_team']:
            # Live/Scheduled indicator
            if market['is_live']:
                live_html = '<span class="live-indicator live">🔴 LIVE</span>'
            else:
                game_time = format_game_time(market['game_date'])
                live_html = f'<span class="live-indicator scheduled">📅 {game_time}</span>'

            st.markdown(f"<div style='text-align: center; margin-bottom: 1rem;'>{live_html}</div>", unsafe_allow_html=True)

            # Team logos row with clickable buttons
            col1, col2, col3 = st.columns([1, 1, 1])

            away_team = market['away_team']
            home_team = market['home_team']
            away_emoji = get_team_emoji(away_team)
            home_emoji = get_team_emoji(home_team)
            ticker = market['ticker']

            # Get AI prediction
            ai_pick = None
            if market.get('predicted_outcome') == 'yes':
                # Determine which team AI picked (depends on market type)
                if 'win' in market['title'].lower():
                    if home_team in market['title']:
                        ai_pick = home_team
                    elif away_team in market['title']:
                        ai_pick = away_team

            with col1:
                st.markdown(f"<div style='text-align: center;'><strong>{away_team}</strong></div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div style='text-align: center; font-size: 1.2rem; font-weight: bold;'>VS</div>", unsafe_allow_html=True)

            with col3:
                st.markdown(f"<div style='text-align: center;'><strong>{home_team}</strong></div>", unsafe_allow_html=True)

            # Team selection buttons
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                # Away team button
                away_class = ""
                if st.session_state.user_picks.get(ticker) == away_team:
                    away_class = "user-pick"
                elif ai_pick == away_team:
                    away_class = "ai-pick"

                if st.button(
                    away_emoji,
                    key=f"away_{ticker}",
                    help=f"Select {away_team}",
                    use_container_width=True
                ):
                    # Toggle selection
                    if st.session_state.user_picks.get(ticker) == away_team:
                        del st.session_state.user_picks[ticker]
                    else:
                        st.session_state.user_picks[ticker] = away_team
                        # Send Telegram notification
                        game_time = format_game_time(market['game_date'])
                        send_team_selection_notification(away_team, home_team, game_time, ticker)
                    st.rerun()

            with col2:
                pass  # Empty middle column

            with col3:
                # Home team button
                home_class = ""
                if st.session_state.user_picks.get(ticker) == home_team:
                    home_class = "user-pick"
                elif ai_pick == home_team:
                    home_class = "ai-pick"

                if st.button(
                    home_emoji,
                    key=f"home_{ticker}",
                    help=f"Select {home_team}",
                    use_container_width=True
                ):
                    # Toggle selection
                    if st.session_state.user_picks.get(ticker) == home_team:
                        del st.session_state.user_picks[ticker]
                    else:
                        st.session_state.user_picks[ticker] = home_team
                        # Send Telegram notification
                        game_time = format_game_time(market['game_date'])
                        send_team_selection_notification(home_team, away_team, game_time, ticker)
                    st.rerun()

            # Pick legend
            st.markdown("""
                <div class="pick-legend">
                    <div class="legend-item">
                        <div class="legend-dot user"></div>
                        <span>🟢 Your Pick</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot ai"></div>
                        <span>🔵 AI Prediction</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ========== KALSHI ODDS DISPLAY ==========
        yes_price = market['yes_price'] * 100 if market['yes_price'] else 0
        no_price = market['no_price'] * 100 if market['no_price'] else 0

        st.markdown(f"""
            <div class="kalshi-odds">
                <div class="odds-item">
                    <div class="odds-label">Yes Price</div>
                    <div class="odds-value yes">{yes_price:.0f}¢</div>
                </div>
                <div class="odds-item">
                    <div class="odds-label">No Price</div>
                    <div class="odds-value no">{no_price:.0f}¢</div>
                </div>
                <div class="odds-item">
                    <div class="odds-label">Volume</div>
                    <div class="odds-value">${market['volume']:,.0f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)


        # ========== MARKET DETAILS ==========
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.markdown(f"**Ticker:** `{market['ticker']}`")
            st.markdown(f"**Type:** {market['bet_type']}")
            if market['player_name']:
                st.markdown(f"**Player:** {market['player_name']}")

        with col2:
            st.markdown(f"<div class='score-badge {badge_class}'>Score: {confidence:.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"**Risk:** {market['risk_level']}")

        with col3:
            st.markdown(f"**Edge:** {market['edge_pct']:.2f}%")
            st.markdown(f"**Closes:** {market['days_to_close']}d")

        with col4:
            st.markdown(f"**Action:** {market.get('recommended_action', 'N/A').upper()}")

        # ========== AI ANALYSIS ==========
        if market.get('reasoning'):
            st.markdown("**🤖 AI Analysis:**")
            st.info(market['reasoning'])

        # ========== PRICE HISTORY CHART ==========
        if st.checkbox(f"Show Price History", key=f"chart_{market['ticker']}"):
            price_history = data_manager.get_price_history(market['ticker'])
            if not price_history.empty:
                chart = ChartBuilder.create_odds_movement_chart(
                    price_history,
                    f"Price Movement - {market['ticker']}"
                )
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.caption("No price history available yet")

        # ========== ACTION BUTTONS ==========
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⭐ Add to Watchlist", key=f"watch_{market['ticker']}", use_container_width=True):
                add_to_watchlist(market['ticker'])
        with col2:
            kalshi_url = f"https://kalshi.com/markets/{market['ticker']}"
            st.link_button("📈 View on Kalshi", kalshi_url, use_container_width=True)
        with col3:
            if st.button("🔔 Set Alert", key=f"alert_{market['ticker']}", use_container_width=True):
                show_alert_config(market)


# ============================================================================
# WATCHLIST MANAGEMENT
# ============================================================================

def initialize_watchlist():
    """Initialize watchlist in session state"""
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []

def add_to_watchlist(ticker: str):
    """Add market to watchlist"""
    initialize_watchlist()
    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)
        st.success(f"Added {ticker} to watchlist!")
    else:
        st.info(f"{ticker} is already in your watchlist")

def remove_from_watchlist(ticker: str):
    """Remove market from watchlist"""
    initialize_watchlist()
    if ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker)
        st.success(f"Removed {ticker} from watchlist")

def render_watchlist_tab(df: pd.DataFrame, data_manager: MarketDataManager):
    """Render watchlist tab content"""
    initialize_watchlist()

    st.header("⭐ Your Watchlist")

    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add markets from the main view to track them here.")
        return

    # Filter dataframe to watchlist items
    watchlist_df = df[df['ticker'].isin(st.session_state.watchlist)]

    if watchlist_df.empty:
        st.warning("No watchlist markets found in current data. They may have closed or expired.")
        return

    st.metric("Watchlist Items", len(watchlist_df))

    # Display watchlist markets
    for _, market in watchlist_df.iterrows():
        with st.expander(f"📌 {market['title'][:60]}...", expanded=True):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**Confidence:** {market['confidence']:.0f} | **Edge:** {market['edge_pct']:.2f}%")
                st.markdown(f"**Current Prices:** YES {market['yes_price']*100:.1f}% | NO {market['no_price']*100:.1f}%")

            with col2:
                st.metric("Closes In", f"{market['days_to_close']}d")

            with col3:
                if st.button("🗑️ Remove", key=f"remove_{market['ticker']}", use_container_width=True):
                    remove_from_watchlist(market['ticker'])
                    st.rerun()


# ============================================================================
# ALERT CONFIGURATION
# ============================================================================

def show_alert_config(market: pd.Series):
    """Show alert configuration modal"""
    st.session_state.alert_market = market['ticker']
    st.info("Alert functionality coming soon! You'll be able to set price alerts and notifications.")


# ============================================================================
# COMPARISON TOOLS
# ============================================================================

def render_comparison_tab(df: pd.DataFrame):
    """Render market comparison tab"""
    st.header("⚖️ Market Comparison")

    st.markdown("Select markets to compare side-by-side:")

    # Multi-select for markets to compare
    market_options = df['ticker'].tolist()
    selected_markets = st.multiselect(
        "Choose markets to compare (2-4 recommended)",
        options=market_options,
        max_selections=4,
        format_func=lambda x: f"{x} - {df[df['ticker']==x]['title'].iloc[0][:50]}..."
    )

    if len(selected_markets) < 2:
        st.info("Select at least 2 markets to compare")
        return

    # Filter to selected markets
    comparison_df = df[df['ticker'].isin(selected_markets)]

    # Comparison table
    st.subheader("Key Metrics Comparison")

    comparison_metrics = comparison_df[[
        'ticker', 'title', 'confidence', 'edge_pct',
        'yes_price', 'volume', 'days_to_close', 'risk_level'
    ]].copy()

    comparison_metrics['yes_price'] = comparison_metrics['yes_price'] * 100
    comparison_metrics.columns = ['Ticker', 'Market', 'Confidence', 'Edge %', 'Yes Price %', 'Volume', 'Days to Close', 'Risk']

    st.dataframe(
        comparison_metrics,
        use_container_width=True,
        hide_index=True
    )

    # Visual comparison
    col1, col2 = st.columns(2)

    with col1:
        # Confidence comparison
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_df['ticker'],
                y=comparison_df['confidence'],
                marker_color='#667eea',
                text=comparison_df['confidence'].round(1),
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Confidence Comparison",
            yaxis_title="Confidence Score",
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Edge comparison
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_df['ticker'],
                y=comparison_df['edge_pct'],
                marker_color='#10b981',
                text=comparison_df['edge_pct'].round(2),
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Edge Percentage Comparison",
            yaxis_title="Edge %",
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# ANALYTICS TAB
# ============================================================================

def render_analytics_tab(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render analytics and insights tab"""
    st.header("📊 Market Analytics")

    # Create tabs for different analytics views
    analytics_tabs = st.tabs([
        "📈 Volume Trends",
        "🎯 Confidence Distribution",
        "🔥 Opportunity Heatmap",
        "💎 Edge Analysis"
    ])

    with analytics_tabs[0]:
        st.subheader("Volume Analysis")
        volume_chart = ChartBuilder.create_volume_chart(filtered_df)
        st.plotly_chart(volume_chart, use_container_width=True)

        # Volume statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Volume", f"${filtered_df['volume'].sum():,.0f}")
        with col2:
            st.metric("Avg Volume", f"${filtered_df['volume'].mean():,.0f}")
        with col3:
            st.metric("High Volume Markets", len(filtered_df[filtered_df['volume'] > 1000]))

    with analytics_tabs[1]:
        st.subheader("Confidence Score Distribution")
        confidence_chart = ChartBuilder.create_confidence_distribution(filtered_df)
        st.plotly_chart(confidence_chart, use_container_width=True)

        # Confidence breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High (80+)", len(filtered_df[filtered_df['confidence'] >= 80]))
        with col2:
            st.metric("Medium (60-80)", len(filtered_df[(filtered_df['confidence'] >= 60) & (filtered_df['confidence'] < 80)]))
        with col3:
            st.metric("Low (<60)", len(filtered_df[filtered_df['confidence'] < 60]))

    with analytics_tabs[2]:
        st.subheader("Opportunity Heatmap")
        heatmap_chart = ChartBuilder.create_opportunity_heatmap(filtered_df)
        if heatmap_chart.data:
            st.plotly_chart(heatmap_chart, use_container_width=True)
        else:
            st.info("Not enough data to generate heatmap. Need markets with team information.")

    with analytics_tabs[3]:
        st.subheader("Edge vs Confidence Analysis")
        scatter_chart = ChartBuilder.create_edge_scatter(filtered_df)
        st.plotly_chart(scatter_chart, use_container_width=True)

        # Edge statistics
        st.markdown("**Edge Statistics:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Edge", f"{filtered_df['edge_pct'].mean():.2f}%")
        with col2:
            st.metric("Max Edge", f"{filtered_df['edge_pct'].max():.2f}%")
        with col3:
            positive_edge = len(filtered_df[filtered_df['edge_pct'] > 0])
            st.metric("Positive Edge", f"{positive_edge} ({positive_edge/len(filtered_df)*100:.1f}%)")
        with col4:
            strong_edge = len(filtered_df[filtered_df['edge_pct'] > 5])
            st.metric("Strong Edge (>5%)", strong_edge)


# ============================================================================
# EXPORT FUNCTIONALITY
# ============================================================================

def render_export_section(df: pd.DataFrame):
    """Render export options"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Export Data")

    export_format = st.sidebar.selectbox(
        "Export format",
        options=['CSV', 'Excel', 'JSON']
    )

    if st.sidebar.button("📥 Export Markets", use_container_width=True):
        if export_format == 'CSV':
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"kalshi_markets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                use_container_width=True
            )
        elif export_format == 'Excel':
            # Note: Requires openpyxl
            st.sidebar.info("Excel export requires openpyxl package")
        elif export_format == 'JSON':
            json_data = df.to_json(orient='records', indent=2)
            st.sidebar.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"kalshi_markets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json',
                use_container_width=True
            )


# ============================================================================
# LIVE GAMES TAB
# ============================================================================

def render_live_games_tab(df: pd.DataFrame):
    """Render live games tracking tab"""
    st.header("🎮 Live Game Center")

    # Filter for live games
    live_games = df[df['is_live'] == True]

    if not live_games.empty:
        st.subheader(f"🔴 Live Games ({len(live_games)})")

        for _, game in live_games.iterrows():
            with st.expander(f"🔴 LIVE: {game['title'][:60]}...", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Teams:** {game['away_team']} @ {game['home_team']}")
                    st.markdown(f"**Current Odds:** YES {game['yes_price']*100:.0f}¢ | NO {game['no_price']*100:.0f}¢")

                with col2:
                    st.metric("AI Confidence", f"{game['confidence']:.0f}")
                    st.metric("Edge", f"{game['edge_pct']:.2f}%")
    else:
        st.info("No games currently live")

    # Show upcoming games
    st.subheader("📅 Upcoming Games")

    games_today = df[df['days_to_close'] == 0]
    if not games_today.empty:
        for _, game in games_today.head(5).iterrows():
            game_time = format_game_time(game['game_date'])
            st.markdown(f"**{game['title']}** - {game_time}")
    else:
        st.info("No games closing today")


# ============================================================================
# MAIN PAGE FUNCTION
# ============================================================================

def show_kalshi_nfl_markets():
    """Main function to render Kalshi NFL markets page"""

    configure_page()
    initialize_session_state()

    # Header
    st.title("🏈 Kalshi NFL Prediction Markets")
    st.caption("Modern dashboard for NFL prediction market analysis with AI-powered recommendations")

    # Initialize data manager
    data_manager = MarketDataManager()

    # Load data
    with st.spinner("Loading NFL markets..."):
        df = data_manager.get_all_markets()

    if df.empty:
        st.warning("No NFL markets found in database.")
        st.info("💡 Run the sync script to pull NFL markets: `python sync_kalshi_complete.py`")

        # Show database stats
        stats = data_manager.db.get_stats()
        st.json(stats)
        return

    # Render filters sidebar
    filters = render_filter_sidebar(df, data_manager)

    # Apply filters
    filtered_df = apply_filters(df, filters)

    # Sort by confidence (highest first)
    filtered_df = filtered_df.sort_values('confidence', ascending=False)

    # Render dashboard metrics
    render_dashboard_metrics(df, filtered_df)


    # Main content tabs
    main_tabs = st.tabs([
        "🏈 All Markets",
        "⭐ Watchlist",
        "⚖️ Compare",
        "📊 Analytics",
        "🎮 Live Games"
    ])

    # ========== Tab 1: All Markets ==========
    with main_tabs[0]:
        st.header(f"All Markets ({len(filtered_df)} shown)")

        # Sort options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                options=['Confidence', 'Edge %', 'Volume', 'Closing Soon'],
                index=0
            )
        with col2:
            items_per_page = st.selectbox(
                "Items per page",
                options=[10, 20, 50, 100],
                index=1
            )
        with col3:
            view_mode = st.selectbox(
                "View",
                options=['Cards', 'Table'],
                index=0
            )

        # Apply sorting
        if sort_by == 'Confidence':
            filtered_df = filtered_df.sort_values('confidence', ascending=False)
        elif sort_by == 'Edge %':
            filtered_df = filtered_df.sort_values('edge_pct', ascending=False)
        elif sort_by == 'Volume':
            filtered_df = filtered_df.sort_values('volume', ascending=False)
        elif sort_by == 'Closing Soon':
            filtered_df = filtered_df.sort_values('days_to_close', ascending=True)

        # Pagination
        total_items = len(filtered_df)
        total_pages = (total_items + items_per_page - 1) // items_per_page

        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0

        # Display markets
        if view_mode == 'Cards':
            start_idx = st.session_state.current_page * items_per_page
            end_idx = start_idx + items_per_page

            for idx, market in filtered_df.iloc[start_idx:end_idx].iterrows():
                render_enhanced_market_card(market, data_manager)
        else:
            # Table view
            display_cols = [
                'ticker', 'title', 'bet_type', 'confidence', 'edge_pct',
                'yes_price', 'volume', 'days_to_close', 'risk_level'
            ]
            st.dataframe(
                filtered_df[display_cols],
                use_container_width=True,
                hide_index=True
            )

        # Pagination controls
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0):
                    st.session_state.current_page -= 1
                    st.rerun()
            with col2:
                st.markdown(f"<center>Page {st.session_state.current_page + 1} of {total_pages}</center>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages - 1):
                    st.session_state.current_page += 1
                    st.rerun()

    # ========== Tab 2: Watchlist ==========
    with main_tabs[1]:
        render_watchlist_tab(df, data_manager)

    # ========== Tab 3: Compare ==========
    with main_tabs[2]:
        render_comparison_tab(filtered_df)

    # ========== Tab 4: Analytics ==========
    with main_tabs[3]:
        render_analytics_tab(df, filtered_df)

    # ========== Tab 5: Live Games ==========
    with main_tabs[4]:
        render_live_games_tab(filtered_df)

    # Export section
    render_export_section(filtered_df)

    # Footer
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Markets in database: {len(df)} | Showing: {len(filtered_df)}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def show():
    """Entry point for dashboard integration"""
    show_kalshi_nfl_markets()


if __name__ == "__main__":
    show_kalshi_nfl_markets()
