"""
AVA Discord Knowledge Integration
Provides AVA with access to Discord trading signals and messages
"""

import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

load_dotenv()


class DiscordKnowledge:
    """Provides AVA with Discord message analysis capabilities"""

    def __init__(self) -> None:
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'magnus')
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', '')

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def get_recent_signals(self, hours_back: int = 24, limit: int = 20) -> List[Dict]:
        """
        Get recent trading signals from Discord

        Args:
            hours_back: How many hours back to search
            limit: Maximum number of signals to return

        Returns:
            List of trading signals with analysis
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Keywords for trading signals
            trading_keywords = [
                'buy', 'sell', 'call', 'put', 'strike', 'expiry', 'expiration',
                'bullish', 'bearish', 'target', 'entry', 'stop', 'alert'
            ]

            search_conditions = ' OR '.join(['content ILIKE %s' for _ in trading_keywords])
            params = [hours_back] + [f'%{kw}%' for kw in trading_keywords] + [limit]

            query = f"""
                SELECT
                    m.message_id,
                    m.content,
                    m.author_name,
                    m.timestamp,
                    c.channel_name,
                    c.server_name
                FROM discord_messages m
                JOIN discord_channels c ON m.channel_id = c.channel_id
                WHERE m.timestamp >= NOW() - INTERVAL '%s hours'
                AND ({search_conditions})
                ORDER BY m.timestamp DESC
                LIMIT %s
            """

            cur.execute(query, params)
            results = cur.fetchall()

            cur.close()
            conn.close()

            return [dict(row) for row in results]

        except Exception as e:
            print(f"Error fetching Discord signals: {e}")
            return []

    def get_signals_by_ticker(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """
        Get all Discord messages mentioning a specific ticker

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            days_back: How many days back to search

        Returns:
            List of messages mentioning the ticker
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Search for ticker with word boundaries
            cur.execute("""
                SELECT
                    m.message_id,
                    m.content,
                    m.author_name,
                    m.timestamp,
                    c.channel_name,
                    c.server_name
                FROM discord_messages m
                JOIN discord_channels c ON m.channel_id = c.channel_id
                WHERE m.timestamp >= NOW() - INTERVAL '%s days'
                AND (
                    m.content ILIKE %s
                    OR m.content ~* %s
                )
                ORDER BY m.timestamp DESC
                LIMIT 50
            """, (days_back, f'%{ticker}%', f'\\$?{ticker}\\b'))

            results = cur.fetchall()

            cur.close()
            conn.close()

            return [dict(row) for row in results]

        except Exception as e:
            print(f"Error fetching ticker signals: {e}")
            return []

    def get_channel_summary(self, hours_back: int = 24) -> Dict:
        """
        Get summary statistics of Discord activity

        Args:
            hours_back: Hours back to analyze

        Returns:
            Dictionary with summary stats
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get overall stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT m.channel_id) as active_channels,
                    COUNT(DISTINCT m.author_name) as unique_authors
                FROM discord_messages m
                WHERE m.timestamp >= NOW() - INTERVAL '%s hours'
            """, (hours_back,))

            overall_stats = dict(cur.fetchone())

            # Get top channels
            cur.execute("""
                SELECT
                    c.channel_name,
                    c.server_name,
                    COUNT(*) as message_count
                FROM discord_messages m
                JOIN discord_channels c ON m.channel_id = c.channel_id
                WHERE m.timestamp >= NOW() - INTERVAL '%s hours'
                GROUP BY c.channel_name, c.server_name
                ORDER BY message_count DESC
                LIMIT 5
            """, (hours_back,))

            top_channels = [dict(row) for row in cur.fetchall()]

            # Get top authors
            cur.execute("""
                SELECT
                    author_name,
                    COUNT(*) as message_count
                FROM discord_messages
                WHERE timestamp >= NOW() - INTERVAL '%s hours'
                GROUP BY author_name
                ORDER BY message_count DESC
                LIMIT 5
            """, (hours_back,))

            top_authors = [dict(row) for row in cur.fetchall()]

            cur.close()
            conn.close()

            return {
                'overall': overall_stats,
                'top_channels': top_channels,
                'top_authors': top_authors,
                'hours_analyzed': hours_back
            }

        except Exception as e:
            print(f"Error getting channel summary: {e}")
            return {}

    def search_messages(self, query: str, hours_back: int = 168, limit: int = 20) -> List[Dict]:
        """
        Search Discord messages for specific content

        Args:
            query: Search query
            hours_back: Hours back to search
            limit: Maximum results

        Returns:
            List of matching messages
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute("""
                SELECT
                    m.message_id,
                    m.content,
                    m.author_name,
                    m.timestamp,
                    c.channel_name,
                    c.server_name
                FROM discord_messages m
                JOIN discord_channels c ON m.channel_id = c.channel_id
                WHERE m.timestamp >= NOW() - INTERVAL '%s hours'
                AND m.content ILIKE %s
                ORDER BY m.timestamp DESC
                LIMIT %s
            """, (hours_back, f'%{query}%', limit))

            results = cur.fetchall()

            cur.close()
            conn.close()

            return [dict(row) for row in results]

        except Exception as e:
            print(f"Error searching messages: {e}")
            return []

    def format_signal_for_ava(self, signal: Dict) -> str:
        """
        Format a Discord signal for AVA to analyze

        Args:
            signal: Discord message dictionary

        Returns:
            Formatted string for AVA context
        """
        timestamp = signal.get('timestamp', datetime.now())
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
        else:
            timestamp_str = str(timestamp)

        return f"""
Discord Trading Signal:
- Source: {signal.get('server_name', 'Unknown')} / {signal.get('channel_name', 'Unknown')}
- Author: {signal.get('author_name', 'Unknown')}
- Time: {timestamp_str}
- Message: {signal.get('content', '')}
"""

    def get_ava_context(self, ticker: Optional[str] = None, hours_back: int = 24) -> str:
        """
        Get formatted Discord context for AVA

        Args:
            ticker: Optional ticker to focus on
            hours_back: Hours back to analyze

        Returns:
            Formatted context string for AVA
        """
        if ticker:
            signals = self.get_signals_by_ticker(ticker, days_back=hours_back // 24)
            context_header = f"Recent Discord signals for ${ticker}:"
        else:
            signals = self.get_recent_signals(hours_back=hours_back, limit=10)
            context_header = "Recent Discord trading signals:"

        if not signals:
            return "No recent Discord trading signals found."

        # Format signals
        formatted = [context_header, ""]
        for signal in signals[:10]:  # Limit to 10 for context window
            formatted.append(self.format_signal_for_ava(signal))
            formatted.append("-" * 60)

        # Add summary
        summary = self.get_channel_summary(hours_back=hours_back)
        if summary:
            formatted.append("\nActivity Summary:")
            formatted.append(f"Total messages: {summary.get('overall', {}).get('total_messages', 0)}")
            formatted.append(f"Active channels: {summary.get('overall', {}).get('active_channels', 0)}")
            formatted.append(f"Unique authors: {summary.get('overall', {}).get('unique_authors', 0)}")

        return "\n".join(formatted)


# Singleton instance for easy import
_discord_knowledge = None


def get_discord_knowledge() -> DiscordKnowledge:
    """Get singleton DiscordKnowledge instance"""
    global _discord_knowledge
    if _discord_knowledge is None:
        _discord_knowledge = DiscordKnowledge()
    return _discord_knowledge


if __name__ == "__main__":
    # Test Discord knowledge
    dk = DiscordKnowledge()

    print("\n" + "=" * 70)
    print("Discord Knowledge Test")
    print("=" * 70)

    # Test recent signals
    print("\n1. Recent Trading Signals:")
    signals = dk.get_recent_signals(hours_back=24, limit=5)
    print(f"   Found {len(signals)} signals")
    if signals:
        print(f"\n   Latest signal:")
        print(f"   {dk.format_signal_for_ava(signals[0])}")

    # Test summary
    print("\n2. Channel Summary:")
    summary = dk.get_channel_summary(hours_back=24)
    if summary:
        print(f"   Total messages: {summary.get('overall', {}).get('total_messages', 0)}")
        print(f"   Active channels: {summary.get('overall', {}).get('active_channels', 0)}")

    # Test AVA context
    print("\n3. AVA Context (sample):")
    context = dk.get_ava_context(hours_back=24)
    print(context[:500] + "..." if len(context) > 500 else context)

    print("\n" + "=" * 70)
