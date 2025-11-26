"""
XTrade Messages Page - Optimized & Modernized
View and analyze XTrade alerts and signals from Discord channels with world-class RAG integration
"""

import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="XTrade Messages", page_icon="📱", layout="wide")

import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import os
import re
import json
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple


# ==================== DATABASE MANAGER ====================

@st.cache_resource
def get_discord_db():
    """Get cached database connection manager"""
    return DiscordDB()


class DiscordDB:
    """Discord database manager with optimized connection pooling"""

    def __init__(self):
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'magnus')
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', '')

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            yield conn
        except psycopg2.Error as e:
            st.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @st.cache_data(ttl=60)
    def get_channels(_self):
        """Get all Discord channels with message counts (cached, optimized JOIN)"""
        try:
            with _self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            c.channel_id,
                            c.channel_name,
                            c.server_name,
                            c.last_sync,
                            COALESCE(COUNT(m.message_id), 0) as message_count
                        FROM discord_channels c
                        LEFT JOIN discord_messages m ON c.channel_id = m.channel_id
                        GROUP BY c.channel_id, c.channel_name, c.server_name, c.last_sync
                        ORDER BY c.last_sync DESC NULLS LAST
                    """)
                    channels = cur.fetchall()

                    # Ensure Unicode compatibility (handle emojis in channel names)
                    for ch in channels:
                        if ch.get('channel_name'):
                            ch['channel_name'] = str(ch['channel_name'])
                        if ch.get('server_name'):
                            ch['server_name'] = str(ch['server_name'])

                    return channels
        except Exception as e:
            st.error(f"Error fetching channels: {e}")
            import traceback
            st.error(traceback.format_exc())
            return []

    @st.cache_data(ttl=30)
    def get_rag_signals(_self, hours_back=168, min_confidence=40, limit=100):
        """Get RAG signals with quality scores and performance data (world-class integration)"""
        try:
            with _self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            s.id,
                            s.message_id,
                            s.author,
                            s.timestamp,
                            s.content,
                            s.tickers,
                            s.primary_ticker,
                            s.setup_type,
                            s.sentiment,
                            s.entry,
                            s.target,
                            s.stop_loss,
                            s.option_strike,
                            s.option_type,
                            s.option_expiration,
                            s.confidence as base_confidence,
                            c.channel_name,
                            c.server_name,
                            -- World-class RAG integration
                            q.composite_score,
                            q.recommendation,
                            q.reasoning,
                            q.rank,
                            q.author_credibility,
                            q.setup_success_rate,
                            a.win_rate as author_win_rate,
                            a.total_pnl_dollars as author_total_pnl,
                            a.trades_taken as author_trades,
                            sp.win_rate as setup_win_rate,
                            sp.avg_pnl_percent as setup_avg_return
                        FROM discord_trading_signals s
                        LEFT JOIN discord_channels c ON s.channel_id = c.channel_id
                        LEFT JOIN signal_quality_scores q ON s.id = q.signal_id
                        LEFT JOIN author_performance a ON s.author = a.author
                        LEFT JOIN setup_performance sp ON s.primary_ticker = sp.ticker AND s.setup_type = sp.setup_type
                        WHERE s.timestamp >= NOW() - INTERVAL '%s hours'
                        AND s.confidence >= %s
                        ORDER BY
                            COALESCE(q.composite_score, s.confidence) DESC,
                            s.timestamp DESC
                        LIMIT %s
                    """, (hours_back, min_confidence, limit))
                    return cur.fetchall()
        except Exception as e:
            st.error(f"Error fetching RAG signals: {e}")
            return []

    @st.cache_data(ttl=30)
    def get_messages(_self, channel_id=None, search_term=None, hours_back=24, limit=100, offset=0):
        """Get Discord messages with pagination (optimized)"""
        try:
            with _self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    query = """
                        SELECT
                            m.message_id,
                            m.content,
                            m.author_name,
                            m.timestamp,
                            c.channel_name,
                            c.server_name,
                            m.reactions,
                            m.attachments,
                            m.embeds
                        FROM discord_messages m
                        JOIN discord_channels c ON m.channel_id = c.channel_id
                        WHERE m.timestamp >= NOW() - INTERVAL '%s hours'
                    """
                    params = [hours_back]

                    if channel_id:
                        query += " AND m.channel_id = %s"
                        params.append(channel_id)

                    if search_term:
                        query += " AND m.content ILIKE %s"
                        params.append(f'%{search_term}%')

                    query += " ORDER BY m.timestamp DESC LIMIT %s OFFSET %s"
                    params.extend([limit, offset])

                    cur.execute(query, params)
                    return cur.fetchall()
        except Exception as e:
            st.error(f"Error fetching messages: {e}")
            return []

    @st.cache_data(ttl=120)
    def get_analytics_stats(_self):
        """Get aggregated analytics stats (cached longer)"""
        try:
            with _self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Top authors by credibility
                    cur.execute("""
                        SELECT
                            author,
                            credibility_score,
                            win_rate,
                            trades_taken,
                            total_pnl_dollars
                        FROM author_performance
                        WHERE trades_taken >= 5
                        ORDER BY credibility_score DESC
                        LIMIT 10
                    """)
                    top_authors = cur.fetchall()

                    # Best setups by ticker
                    cur.execute("""
                        SELECT
                            ticker,
                            setup_type,
                            win_rate,
                            trades_taken,
                            avg_pnl_percent
                        FROM setup_performance
                        WHERE trades_taken >= 3
                        ORDER BY success_score DESC
                        LIMIT 10
                    """)
                    best_setups = cur.fetchall()

                    # Signal quality distribution
                    cur.execute("""
                        SELECT
                            recommendation,
                            COUNT(*) as count,
                            AVG(composite_score) as avg_score
                        FROM signal_quality_scores
                        GROUP BY recommendation
                        ORDER BY
                            CASE recommendation
                                WHEN 'strong_buy' THEN 1
                                WHEN 'buy' THEN 2
                                WHEN 'hold' THEN 3
                                WHEN 'pass' THEN 4
                            END
                    """)
                    quality_dist = cur.fetchall()

                    return {
                        'top_authors': top_authors,
                        'best_setups': best_setups,
                        'quality_distribution': quality_dist
                    }
        except Exception as e:
            st.warning(f"Analytics not available: {e}")
            return {
                'top_authors': [],
                'best_setups': [],
                'quality_distribution': []
            }

    def add_channel(self, channel_id: int, channel_name: str, server_name: str, description: str = None):
        """Add a new Discord channel to track"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO discord_channels (channel_id, channel_name, server_name, description, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (channel_id)
                        DO UPDATE SET
                            channel_name = EXCLUDED.channel_name,
                            server_name = EXCLUDED.server_name,
                            description = EXCLUDED.description
                    """, (channel_id, channel_name, server_name, description))
                    conn.commit()
                    st.cache_data.clear()
                    return True
        except Exception as e:
            st.error(f"Error adding channel: {e}")
            return False

    def sync_channel(self, channel_id: int, days_back: int = 7):
        """Sync messages from Discord channel with progress updates"""
        try:
            from src.discord_message_sync import DiscordMessageSync
            from src.discord_signal_extractor import DiscordSignalExtractor
            from src.signal_performance_tracker import SignalPerformanceTracker

            sync = DiscordMessageSync()

            # Step 1: Export messages
            with st.status("📥 Syncing messages from Discord...", expanded=True) as status:
                st.write(f"Fetching {days_back} days of history...")
                json_file = sync.export_channel(str(channel_id), days_back)
                st.write("✅ Messages downloaded")

                # Step 2: Import to database
                st.write("💾 Importing to database...")
                count = sync.import_messages(json_file)
                st.write(f"✅ Imported {count} messages")

                # Step 3: Extract signals
                st.write("🔍 Extracting trading signals...")
                extractor = DiscordSignalExtractor()
                signals_count = extractor.process_all_messages()
                st.write(f"✅ Extracted {signals_count} signals")

                # Step 4: Update quality scores
                st.write("📊 Calculating quality scores...")
                tracker = SignalPerformanceTracker()
                tracker.update_author_performance()
                tracker.update_setup_performance()
                tracker.calculate_quality_scores()
                st.write("✅ Quality scores updated")

                status.update(label="✅ Sync complete!", state="complete")

            st.cache_data.clear()
            return count

        except Exception as e:
            st.error(f"Sync error: {e}")
            return None

    def remove_channel(self, channel_id: int):
        """Remove a Discord channel from tracking"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM discord_messages WHERE channel_id = %s", (channel_id,))
                    cur.execute("DELETE FROM discord_channels WHERE channel_id = %s", (channel_id,))
                    conn.commit()
                    st.cache_data.clear()
                    return True
        except Exception as e:
            st.error(f"Error removing channel: {e}")
            return False


# ==================== UI COMPONENTS ====================

def render_signal_card(signal: Dict, show_quality_score: bool = True):
    """Render a modern signal card with optional quality scoring"""

    # Determine which score to use
    if show_quality_score and signal.get('composite_score'):
        score = float(signal['composite_score'])
        score_label = "Quality Score"
        recommendation = signal.get('recommendation', 'hold')
    else:
        score = float(signal.get('base_confidence', signal.get('confidence', 0)))
        score_label = "Confidence"
        recommendation = 'hold'

    # Color scheme
    if score >= 75:
        border_color, bg_color, badge_color = '#00ff00', 'rgba(0, 255, 0, 0.1)', '🟢'
    elif score >= 60:
        border_color, bg_color, badge_color = '#ffd700', 'rgba(255, 215, 0, 0.1)', '🟡'
    else:
        border_color, bg_color, badge_color = '#ff9900', 'rgba(255, 153, 0, 0.1)', '🟠'

    # Sentiment badge
    sentiment = signal.get('sentiment', 'neutral')
    sentiment_badges = {
        'bullish': '📈 Bullish',
        'bearish': '📉 Bearish',
        'neutral': '➡️ Neutral'
    }

    # Recommendation badge
    rec_badges = {
        'strong_buy': '🚀 STRONG BUY',
        'buy': '✅ BUY',
        'hold': '⏸️ HOLD',
        'pass': '❌ PASS'
    }

    # Card header
    st.markdown(f"""
        <div style="border-left: 4px solid {border_color}; background: {bg_color};
                    padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <strong style="font-size: 1.1em;">{signal.get('primary_ticker', 'N/A')}</strong> •
                    <span style="color: #aaa;">{signal.get('author', 'Unknown')}</span>
                </div>
                <div style="text-align: right;">
                    <span style="color: {border_color}; font-weight: bold; font-size: 1.3em;">
                        {score:.0f}%
                    </span>
                </div>
            </div>
            <div style="font-size: 0.85em; color: #888;">
                {signal.get('server_name', '')} / {signal.get('channel_name', '')} •
                {signal.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Key metrics in columns
    cols = st.columns([2, 2, 2, 2])

    with cols[0]:
        st.markdown(f"**Setup:** {signal.get('setup_type', 'N/A').replace('_', ' ').title()}")

    with cols[1]:
        st.markdown(f"**Sentiment:** {sentiment_badges.get(sentiment, '➡️ Neutral')}")

    with cols[2]:
        if show_quality_score and recommendation:
            st.markdown(f"**Rec:** {rec_badges.get(recommendation, '⏸️ HOLD')}")
        else:
            st.markdown(f"**{score_label}:** {score:.0f}%")

    with cols[3]:
        # Show author credibility if available
        if signal.get('author_credibility'):
            cred = float(signal['author_credibility'])
            st.markdown(f"**Author:** {cred:.0f}/100")
        else:
            st.markdown(f"**Author:** New")

    # Expandable sections
    with st.expander("📄 Message Content"):
        st.write(signal.get('content', 'No content'))

    # Price levels
    if signal.get('entry') or signal.get('target') or signal.get('stop_loss'):
        with st.expander("💰 Price Levels"):
            price_cols = st.columns(3)

            with price_cols[0]:
                if signal.get('entry'):
                    st.metric("Entry", f"${signal['entry']:.2f}")

            with price_cols[1]:
                if signal.get('target'):
                    profit = ((signal['target'] - signal.get('entry', signal['target'])) / signal.get('entry', signal['target']) * 100) if signal.get('entry') else 0
                    st.metric("Target", f"${signal['target']:.2f}", f"+{profit:.1f}%")

            with price_cols[2]:
                if signal.get('stop_loss'):
                    loss = ((signal['stop_loss'] - signal.get('entry', signal['stop_loss'])) / signal.get('entry', signal['stop_loss']) * 100) if signal.get('entry') else 0
                    st.metric("Stop Loss", f"${signal['stop_loss']:.2f}", f"{loss:.1f}%")

    # Option details
    if signal.get('option_strike'):
        with st.expander("📋 Option Details"):
            opt_cols = st.columns(3)

            with opt_cols[0]:
                st.metric("Strike", f"${signal['option_strike']:.2f}")

            with opt_cols[1]:
                if signal.get('option_type'):
                    st.metric("Type", signal['option_type'].upper())

            with opt_cols[2]:
                if signal.get('option_expiration'):
                    st.metric("Expiration", signal['option_expiration'])

    # Performance context (if available)
    if show_quality_score and (signal.get('author_win_rate') or signal.get('setup_win_rate')):
        with st.expander("📊 Historical Performance"):
            perf_cols = st.columns(2)

            with perf_cols[0]:
                if signal.get('author_win_rate') and signal.get('author_trades'):
                    st.markdown(f"""
                    **Author Track Record:**
                    - Win Rate: {signal['author_win_rate']:.1f}%
                    - Trades: {signal['author_trades']}
                    - Total P&L: ${signal.get('author_total_pnl', 0):.0f}
                    """)

            with perf_cols[1]:
                if signal.get('setup_win_rate'):
                    st.markdown(f"""
                    **Setup Performance ({signal.get('primary_ticker', 'N/A')}):**
                    - Win Rate: {signal['setup_win_rate']:.1f}%
                    - Avg Return: {signal.get('setup_avg_return', 0):.1f}%
                    """)

        # Reasoning
        if signal.get('reasoning'):
            with st.expander("🧠 AI Reasoning"):
                st.info(signal['reasoning'])

    st.write("")  # Spacing between signals


def render_stats_header(channels: List[Dict]):
    """Render modern stats header with key metrics"""
    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "📡 Channels",
            len(channels) if channels else 0,
            help="Number of Discord channels being monitored"
        )

    with cols[1]:
        total_messages = sum(c.get('message_count', 0) for c in channels) if channels else 0
        st.metric(
            "💬 Messages",
            f"{total_messages:,}",
            help="Total messages in database"
        )

    with cols[2]:
        if channels and channels[0].get('last_sync'):
            last_sync = channels[0]['last_sync']
            time_ago = datetime.now() - last_sync
            if time_ago.total_seconds() < 3600:
                delta_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                delta_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
            st.metric("🔄 Last Sync", last_sync.strftime('%H:%M'), delta_str)
        else:
            st.metric("🔄 Last Sync", "Never")

    with cols[3]:
        # Get quality stats
        try:
            db = get_discord_db()
            stats = db.get_analytics_stats()
            strong_buys = next((d['count'] for d in stats['quality_distribution'] if d['recommendation'] == 'strong_buy'), 0)
            st.metric("🚀 Strong Buys", strong_buys, help="Signals with composite score ≥ 75%")
        except:
            st.metric("🎯 Signals", "N/A")


# ==================== MAIN APPLICATION ====================

def main():
    # Custom CSS for better sidebar navigation padding
    st.markdown("""
        <style>
        /* Increase padding for sidebar navigation links */
        [data-testid="stSidebarNav"] li {
            padding: 0.5rem 0rem !important;
        }

        [data-testid="stSidebarNav"] li a {
            padding: 0.75rem 1rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 0.5rem !important;
        }

        [data-testid="stSidebarNav"] li a:hover {
            background-color: rgba(151, 166, 195, 0.15) !important;
        }

        /* Active/selected link styling */
        [data-testid="stSidebarNav"] li a[aria-current="page"] {
            background-color: rgba(255, 75, 75, 0.1) !important;
            padding: 0.75rem 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📱 XTrade Messages")
    st.markdown("**World-Class RAG System** for Discord Trading Signals")

    # Get database manager
    db = get_discord_db()

    # Load channels
    with st.spinner("Loading channels..."):
        channels = db.get_channels()

    # Stats header
    render_stats_header(channels)

    # Main tabs
    tabs = st.tabs([
        "🎯 Top Signals (RAG)",
        "📊 Analytics",
        "📨 Messages",
        "⚙️ Channel Management"
    ])

    # ===== TAB 1: TOP SIGNALS (RAG) =====
    with tabs[0]:
        st.markdown("### 🎯 Top Trading Signals (World-Class RAG)")
        st.caption("Signals ranked by composite quality score (author credibility + setup performance + completeness)")

        # Inline filters
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])

        with filter_col1:
            hours_back = st.slider("Hours Back", 1, 720, 168, help="Look back period")

        with filter_col2:
            min_confidence = st.slider("Min Quality Score", 0, 100, 40, 5)

        with filter_col3:
            recommendation_filter = st.multiselect(
                "Recommendations",
                ["strong_buy", "buy", "hold", "pass"],
                default=["strong_buy", "buy", "hold"]
            )

        # Fetch RAG signals
        with st.spinner("Loading signals with quality scores..."):
            signals = db.get_rag_signals(
                hours_back=hours_back,
                min_confidence=min_confidence,
                limit=200
            )

        # Filter by recommendation
        if recommendation_filter:
            signals = [s for s in signals if s.get('recommendation') in recommendation_filter]

        if signals:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                unique_tickers = len(set(s['primary_ticker'] for s in signals if s.get('primary_ticker')))
                st.metric("Unique Tickers", unique_tickers)

            with col2:
                strong_buys = len([s for s in signals if s.get('recommendation') == 'strong_buy'])
                st.metric("🚀 Strong Buys", strong_buys)

            with col3:
                avg_score = sum(float(s.get('composite_score', 0)) for s in signals if s.get('composite_score')) / len(signals) if signals else 0
                st.metric("Avg Quality", f"{avg_score:.0f}%")

            with col4:
                bullish = len([s for s in signals if s.get('sentiment') == 'bullish'])
                st.metric("📈 Bullish", bullish)

            st.write("")  # Spacing

            # Display signals
            for signal in signals[:50]:  # Show top 50
                render_signal_card(signal, show_quality_score=True)

        else:
            st.info("No signals match your filters. Try adjusting the time range or lowering the minimum score.")

    # ===== TAB 2: ANALYTICS =====
    with tabs[1]:
        st.markdown("### 📊 Analytics & Performance")

        # Get analytics data
        with st.spinner("Loading analytics..."):
            analytics = db.get_analytics_stats()

        # Top Authors
        if analytics['top_authors']:
            st.markdown("#### 🏆 Top Performing Authors")
            st.caption("Authors ranked by credibility score (minimum 5 trades)")

            authors_data = []
            for author in analytics['top_authors']:
                authors_data.append({
                    'Author': author['author'],
                    'Credibility': f"{author['credibility_score']:.0f}/100",
                    'Win Rate': f"{author['win_rate']:.1f}%",
                    'Trades': author['trades_taken'],
                    'Total P&L': f"${author['total_pnl_dollars']:.0f}"
                })

            st.dataframe(
                pd.DataFrame(authors_data),
                use_container_width=True,
                hide_index=True
            )

        st.write("")  # Spacing

        # Best Setups
        if analytics['best_setups']:
            st.markdown("#### 🎯 Best Performing Setups")
            st.caption("Setup types ranked by success score (minimum 3 trades)")

            setups_data = []
            for setup in analytics['best_setups']:
                setups_data.append({
                    'Ticker': setup['ticker'],
                    'Setup': setup['setup_type'].replace('_', ' ').title(),
                    'Win Rate': f"{setup['win_rate']:.1f}%",
                    'Trades': setup['trades_taken'],
                    'Avg Return': f"{setup['avg_pnl_percent']:.1f}%"
                })

            st.dataframe(
                pd.DataFrame(setups_data),
                use_container_width=True,
                hide_index=True
            )

        st.write("")  # Spacing

        # Quality Distribution
        if analytics['quality_distribution']:
            st.markdown("#### 📈 Signal Quality Distribution")

            dist_cols = st.columns(len(analytics['quality_distribution']))

            rec_icons = {
                'strong_buy': '🚀',
                'buy': '✅',
                'hold': '⏸️',
                'pass': '❌'
            }

            for i, dist in enumerate(analytics['quality_distribution']):
                with dist_cols[i]:
                    icon = rec_icons.get(dist['recommendation'], '❓')
                    label = dist['recommendation'].replace('_', ' ').title()
                    st.metric(
                        f"{icon} {label}",
                        dist['count'],
                        f"Avg: {dist['avg_score']:.0f}%"
                    )

    # ===== TAB 3: MESSAGES =====
    with tabs[2]:
        st.markdown("### 📨 Recent Messages")

        # Inline filters
        msg_col1, msg_col2, msg_col3, msg_col4 = st.columns([2, 2, 2, 2])

        with msg_col1:
            # Channel selector
            if channels:
                channel_options = {
                    f"{c['server_name']} / {c['channel_name']}": c['channel_id']
                    for c in channels
                }
                channel_options = {"All Channels": None, **channel_options}
                selected_channel_name = st.selectbox("Channel", list(channel_options.keys()))
                selected_channel = channel_options[selected_channel_name]
            else:
                selected_channel = None

        with msg_col2:
            hours_back_msg = st.slider("Hours Back", 1, 720, 168, key="msg_hours")

        with msg_col3:
            search_term = st.text_input("🔍 Search", placeholder="Keywords...")

        with msg_col4:
            items_per_page = st.selectbox("Per Page", [25, 50, 100, 200], index=1)

        # Pagination
        page = st.number_input("Page", min_value=1, value=1, step=1)
        offset = (page - 1) * items_per_page

        # Fetch messages
        with st.spinner(f"Loading messages (page {page})..."):
            messages = db.get_messages(
                channel_id=selected_channel,
                search_term=search_term,
                hours_back=hours_back_msg,
                limit=items_per_page,
                offset=offset
            )

        if messages:
            st.caption(f"Showing {len(messages)} messages (page {page})")

            for msg in messages:
                # Time ago
                time_ago = datetime.now() - msg['timestamp']
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.seconds // 60}m ago"

                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"**{msg['author_name']}** • {msg['server_name']} / {msg['channel_name']}")
                        st.write(msg['content'][:500])

                    with col2:
                        st.caption(time_str)
                        st.caption(msg['timestamp'].strftime('%m/%d %H:%M'))

                    st.write("")  # Spacing
        else:
            st.info("No messages found")

    # ===== TAB 4: CHANNEL MANAGEMENT =====
    with tabs[3]:
        st.markdown("### ⚙️ Channel Management")

        # Add channel form
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ➕ Add New Channel")
            with st.form("add_channel"):
                new_channel_id = st.text_input("Channel ID", placeholder="1234567890")
                new_channel_name = st.text_input("Channel Name", placeholder="alerts")
                new_server_name = st.text_input("Server Name", placeholder="XTrades")
                new_description = st.text_area("Description", placeholder="Stock/options signals")

                if st.form_submit_button("➕ Add Channel", use_container_width=True):
                    if not new_channel_id or not new_channel_name or not new_server_name:
                        st.error("All fields required!")
                    elif not new_channel_id.isdigit():
                        st.error("Channel ID must be numeric")
                    else:
                        if db.add_channel(int(new_channel_id), new_channel_name, new_server_name, new_description):
                            st.success(f"✅ Added: {new_server_name} / {new_channel_name}")
                            st.rerun()

        with col2:
            st.markdown("#### 🔍 How to Find Channel IDs")
            st.info("""
            1. Enable **Developer Mode** in Discord (Settings → Advanced)
            2. Right-click on channel → **Copy ID**
            3. Paste ID into form

            📝 **Note:** Stock/options channels only
            """)

        st.write("")  # Spacing

        # Existing channels
        st.markdown("#### 📋 Tracked Channels")

        if channels:
            st.caption(f"Currently tracking {len(channels)} channel(s)")

            for i, channel in enumerate(channels):
                if i > 0:
                    st.write("")  # Spacing between channels

                with st.container():
                    cols = st.columns([3, 2, 1.5, 0.5])

                    with cols[0]:
                        st.markdown(f"**{channel['server_name']} / {channel['channel_name']}**")
                        st.caption(f"ID: {channel['channel_id']}")

                    with cols[1]:
                        st.metric("Messages", f"{channel.get('message_count', 0):,}")
                        if channel.get('last_sync'):
                            st.caption(f"Last: {channel['last_sync'].strftime('%m/%d %H:%M')}")
                        else:
                            st.caption("Never synced")

                    with cols[2]:
                        sync_options = {
                            "7 days": 7,
                            "30 days": 30,
                            "90 days": 90,
                            "180 days": 180,
                            "All History": 9999
                        }
                        sync_label = st.selectbox(
                            "Range",
                            list(sync_options.keys()),
                            key=f"sync_{channel['channel_id']}",
                            label_visibility="collapsed"
                        )

                        if st.button("🔄", key=f"btn_{channel['channel_id']}", use_container_width=True):
                            count = db.sync_channel(channel['channel_id'], sync_options[sync_label])
                            if count is not None:
                                st.toast(f"✅ Synced {count} messages!")
                                st.rerun()

                    with cols[3]:
                        if st.button("🗑️", key=f"del_{channel['channel_id']}", use_container_width=True):
                            if db.remove_channel(channel['channel_id']):
                                st.toast(f"Removed: {channel['channel_name']}")
                                st.rerun()

        else:
            st.warning("No channels configured")


if __name__ == "__main__":
    main()
