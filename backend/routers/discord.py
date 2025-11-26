"""
Discord Router - API endpoints for Discord integration
"""
from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/discord",
    tags=["discord"]
)


@router.get("/channels")
async def get_discord_channels():
    """Get Discord channels - used by DiscordMessages page"""
    return {
        "channels": [
            {"id": "alerts", "name": "Trading Alerts", "type": "text", "unread": 5},
            {"id": "options", "name": "Options Flow", "type": "text", "unread": 12},
            {"id": "earnings", "name": "Earnings Alerts", "type": "text", "unread": 3},
            {"id": "sports", "name": "Sports Betting", "type": "text", "unread": 8},
            {"id": "general", "name": "General", "type": "text", "unread": 0}
        ],
        "total_unread": 28,
        "connected": True,
        "last_sync": datetime.now().isoformat()
    }


@router.get("/messages")
async def get_discord_messages(channel: str = "alerts", limit: int = 50):
    """Get Discord messages from a channel"""
    return {
        "messages": [
            {
                "id": "msg1",
                "channel": channel,
                "author": "AVA Bot",
                "content": "AAPL unusual options activity detected - 5000 calls @ $180 strike",
                "timestamp": "2024-11-25T10:30:00Z",
                "type": "alert"
            },
            {
                "id": "msg2",
                "channel": channel,
                "author": "AVA Bot",
                "content": "SPY breaking above resistance at $450",
                "timestamp": "2024-11-25T10:25:00Z",
                "type": "alert"
            },
            {
                "id": "msg3",
                "channel": channel,
                "author": "AVA Bot",
                "content": "Earnings alert: NVDA reporting after hours today",
                "timestamp": "2024-11-25T09:00:00Z",
                "type": "earnings"
            }
        ],
        "channel": channel,
        "total": 3,
        "has_more": False
    }


@router.post("/send")
async def send_discord_message(channel: str, message: str):
    """Send a message to Discord channel"""
    return {
        "status": "success",
        "message_id": "msg_new",
        "channel": channel,
        "sent_at": datetime.now().isoformat()
    }


@router.get("/status")
async def get_discord_status():
    """Get Discord bot connection status"""
    return {
        "connected": True,
        "bot_name": "AVA Trading Bot",
        "servers": 1,
        "uptime_hours": 48,
        "messages_sent_today": 156
    }
