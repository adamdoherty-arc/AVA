"""
Discord Router - API endpoints for Discord integration
NO MOCK DATA - All endpoints use real database

Pulls from:
- discord_channels table
- discord_messages table
- discord_trading_signals table
"""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime, timedelta
import logging

from backend.infrastructure.database import get_database

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/discord",
    tags=["discord"]
)


@router.get("/channels")
async def get_discord_channels():
    """
    Get all configured Discord channels from database.
    Used by DiscordMessages page for channel selection.
    """
    try:
        db = await get_database()

        channels = await db.fetch("""
            SELECT
                c.channel_id,
                c.channel_name,
                c.server_name,
                c.description,
                c.last_sync,
                c.created_at,
                (SELECT COUNT(*) FROM discord_messages WHERE channel_id = c.channel_id) as message_count,
                (SELECT COUNT(*) FROM discord_messages
                 WHERE channel_id = c.channel_id
                 AND timestamp >= NOW() - INTERVAL '24 hours') as recent_count
            FROM discord_channels c
            ORDER BY c.created_at DESC
        """)

        formatted_channels = []
        total_unread = 0

        for ch in channels:
            recent = ch['recent_count'] or 0
            total_unread += recent
            formatted_channels.append({
                "id": str(ch['channel_id']),
                "name": ch['channel_name'],
                "server": ch['server_name'],
                "description": ch['description'],
                "type": "text",
                "unread": recent,
                "total_messages": ch['message_count'] or 0,
                "last_sync": ch['last_sync'].isoformat() if ch['last_sync'] else None
            })

        # Get connection status (check if we have any recent messages)
        has_recent = await db.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM discord_messages
                WHERE timestamp >= NOW() - INTERVAL '7 days'
            )
        """)

        return {
            "channels": formatted_channels,
            "total_channels": len(formatted_channels),
            "total_unread": total_unread,
            "connected": bool(has_recent),
            "last_sync": datetime.now().isoformat(),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching Discord channels: {e}")
        return {
            "channels": [],
            "total_channels": 0,
            "total_unread": 0,
            "connected": False,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/messages")
async def get_discord_messages(
    channel_id: Optional[str] = Query(None, description="Channel ID to filter"),
    limit: int = Query(50, description="Maximum messages to return"),
    days_back: int = Query(7, description="Days to look back")
):
    """
    Get Discord messages from database.
    If channel_id provided, filter by that channel; otherwise return from all channels.
    """
    try:
        db = await get_database()
        cutoff_date = datetime.now() - timedelta(days=days_back)

        if channel_id:
            messages = await db.fetch("""
                SELECT
                    m.message_id,
                    m.channel_id,
                    m.content,
                    m.author_name,
                    m.timestamp,
                    m.attachments,
                    m.embeds,
                    m.reactions,
                    c.channel_name,
                    c.server_name
                FROM discord_messages m
                LEFT JOIN discord_channels c ON m.channel_id = c.channel_id
                WHERE m.channel_id = $1
                AND m.timestamp >= $2
                ORDER BY m.timestamp DESC
                LIMIT $3
            """, int(channel_id), cutoff_date, limit)
        else:
            messages = await db.fetch("""
                SELECT
                    m.message_id,
                    m.channel_id,
                    m.content,
                    m.author_name,
                    m.timestamp,
                    m.attachments,
                    m.embeds,
                    m.reactions,
                    c.channel_name,
                    c.server_name
                FROM discord_messages m
                LEFT JOIN discord_channels c ON m.channel_id = c.channel_id
                WHERE m.timestamp >= $1
                ORDER BY m.timestamp DESC
                LIMIT $2
            """, cutoff_date, limit)

        formatted_messages = []
        for msg in messages:
            # Determine message type based on content
            content = msg['content'] or ''
            msg_type = 'general'
            if any(word in content.lower() for word in ['earnings', 'er play', 'report']):
                msg_type = 'earnings'
            elif any(word in content.lower() for word in ['alert', 'signal', 'buy', 'sell', 'call', 'put']):
                msg_type = 'alert'
            elif any(word in content.lower() for word in ['options', 'strike', 'expiry', 'dte']):
                msg_type = 'options'

            formatted_messages.append({
                "id": str(msg['message_id']),
                "channel_id": str(msg['channel_id']),
                "channel": msg['channel_name'] or "Unknown",
                "server": msg['server_name'],
                "author": msg['author_name'] or "Unknown",
                "content": content,
                "timestamp": msg['timestamp'].isoformat() if msg['timestamp'] else None,
                "type": msg_type,
                "has_attachments": bool(msg['attachments']),
                "has_embeds": bool(msg['embeds']),
                "reactions": msg['reactions']
            })

        # Check if there are more messages
        total_count = await db.fetchval("""
            SELECT COUNT(*) FROM discord_messages
            WHERE timestamp >= $1
        """, cutoff_date)

        return {
            "messages": formatted_messages,
            "channel_id": channel_id,
            "total": len(formatted_messages),
            "total_available": total_count or 0,
            "has_more": (total_count or 0) > limit,
            "days_back": days_back,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching Discord messages: {e}")
        return {
            "messages": [],
            "channel_id": channel_id,
            "total": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/signals")
async def get_discord_trading_signals(
    limit: int = Query(50, description="Maximum signals to return"),
    days_back: int = Query(7, description="Days to look back"),
    min_confidence: int = Query(40, description="Minimum signal confidence")
):
    """
    Get parsed trading signals from Discord messages.
    Uses the discord_trading_signals table which has pre-extracted tickers, prices, etc.
    """
    try:
        db = await get_database()
        cutoff_date = datetime.now() - timedelta(days=days_back)

        signals = await db.fetch("""
            SELECT
                s.message_id,
                s.channel_id,
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
                s.confidence,
                c.channel_name
            FROM discord_trading_signals s
            LEFT JOIN discord_channels c ON s.channel_id = c.channel_id
            WHERE s.timestamp >= $1
            AND s.confidence >= $2
            ORDER BY s.timestamp DESC
            LIMIT $3
        """, cutoff_date, min_confidence, limit)

        formatted_signals = []
        for sig in signals:
            formatted_signals.append({
                "message_id": str(sig['message_id']),
                "channel_id": str(sig['channel_id']),
                "channel_name": sig['channel_name'],
                "author": sig['author'],
                "timestamp": sig['timestamp'].isoformat() if sig['timestamp'] else None,
                "content": sig['content'],
                "tickers": sig['tickers'] or [],
                "primary_ticker": sig['primary_ticker'],
                "setup_type": sig['setup_type'],
                "sentiment": sig['sentiment'],
                "entry": float(sig['entry']) if sig['entry'] else None,
                "target": float(sig['target']) if sig['target'] else None,
                "stop_loss": float(sig['stop_loss']) if sig['stop_loss'] else None,
                "option_strike": float(sig['option_strike']) if sig['option_strike'] else None,
                "option_type": sig['option_type'],
                "option_expiration": sig['option_expiration'],
                "confidence": sig['confidence']
            })

        return {
            "signals": formatted_signals,
            "total": len(formatted_signals),
            "days_back": days_back,
            "min_confidence": min_confidence,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching Discord signals: {e}")
        return {
            "signals": [],
            "total": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/status")
async def get_discord_status():
    """
    Get Discord integration status - connection health, sync status, message counts.
    """
    try:
        db = await get_database()

        # Get total channels
        channel_count = await db.fetchval("SELECT COUNT(*) FROM discord_channels")

        # Get total messages
        message_count = await db.fetchval("SELECT COUNT(*) FROM discord_messages")

        # Get messages in last 24 hours
        messages_today = await db.fetchval("""
            SELECT COUNT(*) FROM discord_messages
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """)

        # Get most recent message
        newest_message = await db.fetchrow("""
            SELECT timestamp, author_name, channel_id
            FROM discord_messages
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        # Get last sync time
        last_sync = await db.fetchval("""
            SELECT MAX(last_sync) FROM discord_channels
        """)

        # Calculate connection status
        hours_since_last = None
        if newest_message and newest_message['timestamp']:
            hours_since_last = (datetime.now() - newest_message['timestamp'].replace(tzinfo=None)).total_seconds() / 3600

        connected = hours_since_last is not None and hours_since_last < 24

        return {
            "connected": connected,
            "status": "healthy" if connected else "stale",
            "channels_configured": channel_count or 0,
            "total_messages": message_count or 0,
            "messages_last_24h": messages_today or 0,
            "last_message": newest_message['timestamp'].isoformat() if newest_message and newest_message['timestamp'] else None,
            "last_sync": last_sync.isoformat() if last_sync else None,
            "hours_since_last_message": round(hours_since_last, 1) if hours_since_last else None,
            "needs_sync": not connected,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting Discord status: {e}")
        return {
            "connected": False,
            "status": "error",
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/stats")
async def get_discord_stats():
    """
    Get detailed Discord statistics - message counts by channel, signal extraction rates, etc.
    """
    try:
        db = await get_database()

        # Messages by channel
        channel_stats = await db.fetch("""
            SELECT
                c.channel_id,
                c.channel_name,
                c.server_name,
                COUNT(m.message_id) as message_count,
                MAX(m.timestamp) as last_message,
                COUNT(CASE WHEN m.timestamp >= NOW() - INTERVAL '7 days' THEN 1 END) as messages_7d
            FROM discord_channels c
            LEFT JOIN discord_messages m ON c.channel_id = m.channel_id
            GROUP BY c.channel_id, c.channel_name, c.server_name
            ORDER BY message_count DESC
        """)

        # Signal extraction stats
        signal_stats = await db.fetchrow("""
            SELECT
                COUNT(*) as total_signals,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN sentiment = 'bullish' THEN 1 END) as bullish_count,
                COUNT(CASE WHEN sentiment = 'bearish' THEN 1 END) as bearish_count,
                COUNT(CASE WHEN sentiment = 'neutral' THEN 1 END) as neutral_count
            FROM discord_trading_signals
            WHERE timestamp >= NOW() - INTERVAL '30 days'
        """)

        # Top tickers mentioned
        top_tickers = await db.fetch("""
            SELECT primary_ticker, COUNT(*) as mention_count
            FROM discord_trading_signals
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            AND primary_ticker IS NOT NULL
            GROUP BY primary_ticker
            ORDER BY mention_count DESC
            LIMIT 10
        """)

        formatted_channels = []
        for ch in channel_stats:
            formatted_channels.append({
                "channel_id": str(ch['channel_id']),
                "channel_name": ch['channel_name'],
                "server_name": ch['server_name'],
                "total_messages": ch['message_count'] or 0,
                "messages_7d": ch['messages_7d'] or 0,
                "last_message": ch['last_message'].isoformat() if ch['last_message'] else None
            })

        return {
            "channels": formatted_channels,
            "signals": {
                "total_30d": signal_stats['total_signals'] or 0,
                "avg_confidence": round(float(signal_stats['avg_confidence'] or 0), 1),
                "bullish": signal_stats['bullish_count'] or 0,
                "bearish": signal_stats['bearish_count'] or 0,
                "neutral": signal_stats['neutral_count'] or 0
            },
            "top_tickers": [{"ticker": t['primary_ticker'], "count": t['mention_count']} for t in top_tickers],
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting Discord stats: {e}")
        return {
            "channels": [],
            "signals": {},
            "top_tickers": [],
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }
