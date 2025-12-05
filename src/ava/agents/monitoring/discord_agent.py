"""
Discord Integration Agent - Monitor and analyze Discord messages
Uses XTrades Discord channel (990343144241500232) as primary source
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras
import os

from ...core.agent_base import BaseAgent, AgentState
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# XTrades Discord channel ID - Primary source
XTRADES_CHANNEL_ID = 990343144241500232


def _get_db_connection():
    """Get database connection with correct settings"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'magnus'),  # Fixed: was 'trading'
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '')
    )


@tool
def get_discord_messages_tool(hours_back: int = 24, channel_id: Optional[str] = None, limit: int = 50) -> str:
    """
    Get recent Discord messages (XTrades trader signals)

    Args:
        hours_back: How many hours back to fetch (default 24)
        channel_id: Specific channel ID to filter (optional, defaults to XTrades channel)
        limit: Maximum number of messages to return (default 50)

    Returns:
        JSON string with Discord messages
    """
    conn = None
    cursor = None
    try:
        conn = _get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        since_time = datetime.now() - timedelta(hours=hours_back)
        target_channel = int(channel_id) if channel_id else XTRADES_CHANNEL_ID

        query = """
            SELECT
                m.message_id,
                m.channel_id,
                c.channel_name,
                m.author_name,
                m.content,
                m.timestamp,
                m.attachments,
                m.embeds
            FROM discord_messages m
            LEFT JOIN discord_channels c ON m.channel_id = c.channel_id
            WHERE m.timestamp >= %s
            AND m.channel_id = %s
            AND m.content IS NOT NULL
            AND m.content != ''
            ORDER BY m.timestamp DESC
            LIMIT %s
        """
        cursor.execute(query, (since_time, target_channel, limit))
        messages = cursor.fetchall()

        if not messages:
            return f"No Discord messages found in the last {hours_back} hours for channel {target_channel}"

        # Convert to serializable format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'message_id': str(msg['message_id']),
                'channel_id': str(msg['channel_id']),
                'channel_name': msg['channel_name'],
                'author': msg['author_name'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat() if msg['timestamp'] else None
            })

        result = {
            'count': len(formatted_messages),
            'time_range': f'Last {hours_back} hours',
            'channel_id': str(target_channel),
            'messages': formatted_messages
        }

        return str(result)

    except Exception as e:
        logger.error(f"Error fetching Discord messages: {e}")
        return f"Error: {str(e)}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@tool
def search_discord_alerts_tool(keywords: str, days_back: int = 7) -> str:
    """
    Search Discord messages for specific keywords (e.g., ticker symbols, alerts)

    Args:
        keywords: Keywords to search for (space-separated)
        days_back: How many days back to search (default 7)

    Returns:
        JSON string with matching messages
    """
    conn = None
    cursor = None
    try:
        conn = _get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        since_time = datetime.now() - timedelta(days=days_back)

        # Use full-text search or LIKE for keyword matching
        keyword_list = keywords.strip().split()
        search_pattern = '%' + '%'.join(keyword_list) + '%'

        query = """
            SELECT
                m.message_id,
                m.channel_id,
                c.channel_name,
                m.author_name,
                m.content,
                m.timestamp
            FROM discord_messages m
            LEFT JOIN discord_channels c ON m.channel_id = c.channel_id
            WHERE m.timestamp >= %s
            AND m.content ILIKE %s
            ORDER BY m.timestamp DESC
            LIMIT 100
        """

        cursor.execute(query, (since_time, search_pattern))
        messages = cursor.fetchall()

        if not messages:
            return f"No messages found matching '{keywords}' in the last {days_back} days"

        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'message_id': str(msg['message_id']),
                'channel_name': msg['channel_name'],
                'author': msg['author_name'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat() if msg['timestamp'] else None
            })

        result = {
            'keywords': keywords,
            'count': len(formatted_messages),
            'time_range': f'Last {days_back} days',
            'matches': formatted_messages
        }

        return str(result)

    except Exception as e:
        logger.error(f"Error searching Discord messages: {e}")
        return f"Error: {str(e)}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@tool
def filter_trader_messages_tool(trader_name: str, hours_back: int = 48) -> str:
    """
    Filter Discord messages by specific trader/author

    Args:
        trader_name: Trader/author username to filter
        hours_back: How many hours back to search (default 48)

    Returns:
        JSON string with trader's messages
    """
    conn = None
    cursor = None
    try:
        conn = _get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        since_time = datetime.now() - timedelta(hours=hours_back)

        query = """
            SELECT
                m.message_id,
                m.channel_id,
                c.channel_name,
                m.author_name,
                m.content,
                m.timestamp,
                m.attachments
            FROM discord_messages m
            LEFT JOIN discord_channels c ON m.channel_id = c.channel_id
            WHERE m.timestamp >= %s
            AND m.author_name ILIKE %s
            ORDER BY m.timestamp DESC
            LIMIT 50
        """

        cursor.execute(query, (since_time, f'%{trader_name}%'))
        messages = cursor.fetchall()

        if not messages:
            return f"No messages found from trader '{trader_name}' in the last {hours_back} hours"

        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'message_id': str(msg['message_id']),
                'channel_name': msg['channel_name'],
                'author': msg['author_name'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat() if msg['timestamp'] else None
            })

        result = {
            'trader': trader_name,
            'count': len(formatted_messages),
            'time_range': f'Last {hours_back} hours',
            'messages': formatted_messages
        }

        return str(result)

    except Exception as e:
        logger.error(f"Error filtering trader messages: {e}")
        return f"Error: {str(e)}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


class DiscordAgent(BaseAgent):
    """
    Discord Integration Agent - Monitor and analyze Discord messages

    Capabilities:
    - Get recent Discord messages (XTrades trader signals)
    - Search messages by keywords (tickers, alerts)
    - Filter by specific traders/authors
    - Analyze trader activity patterns
    - Track signal frequency and timing
    """

    def __init__(self, use_huggingface: bool = False):
        """Initialize Discord Integration Agent"""
        tools = [
            get_discord_messages_tool,
            search_discord_alerts_tool,
            filter_trader_messages_tool
        ]

        super().__init__(
            name="discord_agent",
            description="Monitors and analyzes Discord messages from XTrades trader channels",
            tools=tools,
            use_huggingface=use_huggingface
        )

        self.metadata['capabilities'] = [
            'get_discord_messages',
            'search_trader_alerts',
            'filter_by_trader',
            'analyze_trader_activity',
            'track_signal_patterns',
            'discord_message_analysis'
        ]

    async def execute(self, state: AgentState) -> AgentState:
        """Execute Discord agent"""
        try:
            input_text = state.get('input', '')
            context = state.get('context', {})

            # Extract parameters from input
            hours_back = context.get('hours_back', 24)
            keywords = context.get('keywords')
            trader_name = context.get('trader_name')

            result = {
                'agent': 'discord_agent',
                'timestamp': datetime.now().isoformat()
            }

            # Determine which operation to perform
            if trader_name:
                # Filter by trader
                messages = filter_trader_messages_tool.invoke({
                    'trader_name': trader_name,
                    'hours_back': hours_back
                })
                result['operation'] = 'filter_by_trader'
                result['data'] = messages

            elif keywords:
                # Search by keywords
                messages = search_discord_alerts_tool.invoke({
                    'keywords': keywords,
                    'days_back': hours_back // 24 or 1
                })
                result['operation'] = 'search_keywords'
                result['data'] = messages

            else:
                # Get recent messages
                messages = get_discord_messages_tool.invoke({
                    'hours_back': hours_back,
                    'limit': 50
                })
                result['operation'] = 'get_recent_messages'
                result['data'] = messages

            state['result'] = result
            return state

        except Exception as e:
            logger.error(f"DiscordAgent error: {e}")
            state['error'] = str(e)
            return state
