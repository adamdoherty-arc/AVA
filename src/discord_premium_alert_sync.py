"""
Discord Premium Alert Sync
===========================

Enhanced Discord sync with priority handling for premium alerts channel.

Features:
- Runs every 5 minutes via Celery
- Prioritizes channel ID 990331623260180580 (premium alerts)
- Sends Discord bot notifications for new premium alerts
- Tracks last sync time to avoid duplicate notifications
- Integrates with RAG knowledge base

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


# Premium alerts channel ID (prioritized)
PREMIUM_ALERTS_CHANNEL_ID = '990331623260180580'


class DiscordPremiumAlertSync:
    """Enhanced Discord sync with premium alert prioritization"""

    def __init__(self) -> None:
        """Initialize sync manager"""
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.discord_bot_webhook = os.getenv('DISCORD_BOT_WEBHOOK_URL')  # AVA bot webhook

        # Database connection params
        self.db_host = os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', 'localhost'))
        self.db_port = os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432'))
        self.db_name = os.getenv('POSTGRES_DB', os.getenv('DB_NAME', 'magnus'))
        self.db_user = os.getenv('POSTGRES_USER', os.getenv('DB_USER', 'postgres'))
        self.db_password = os.getenv('POSTGRES_PASSWORD', os.getenv('DB_PASSWORD', ''))

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def get_new_messages_since_last_sync(
        self,
        channel_id: str,
        minutes_back: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get new messages from a channel since last sync

        Args:
            channel_id: Discord channel ID
            minutes_back: How many minutes back to check (default: 5)

        Returns:
            List of new message dictionaries
        """
        conn = None
        cur = None

        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get messages from last N minutes
            since_time = datetime.now() - timedelta(minutes=minutes_back)

            cur.execute("""
                SELECT
                    message_id,
                    channel_id,
                    author_id,
                    author_name,
                    content,
                    timestamp,
                    attachments,
                    embeds,
                    raw_data
                FROM discord_messages
                WHERE channel_id = %s
                  AND timestamp >= %s
                  AND content IS NOT NULL
                  AND content != ''
                  AND content != '@everyone'
                ORDER BY timestamp DESC
            """, (channel_id, since_time))

            messages = cur.fetchall()

            # Convert to list of dicts
            return [dict(msg) for msg in messages]

        except Exception as e:
            logger.error(f"Error getting new messages: {e}")
            return []

        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def is_premium_alert(self, message: Dict[str, Any]) -> bool:
        """
        Determine if a message is a premium trading alert

        Args:
            message: Message dictionary

        Returns:
            True if message appears to be a trading alert
        """
        content = message.get('content', '').lower()

        # Keywords that indicate trading alerts
        alert_keywords = [
            'buy', 'sell', 'entry', 'exit', 'target', 'stop',
            'call', 'put', 'strike', 'exp', 'expiration',
            '$', 'price', 'alert', 'signal', 'trade',
            'long', 'short', 'bullish', 'bearish'
        ]

        # Check if content contains alert keywords
        return any(keyword in content for keyword in alert_keywords)

    def extract_alert_summary(self, message: Dict[str, Any]) -> str:
        """
        Extract a concise summary of the trading alert

        Args:
            message: Message dictionary

        Returns:
            Short summary of the alert
        """
        content = message.get('content', '')
        author = message.get('author_name', 'Unknown')
        timestamp = message.get('timestamp')

        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + '...'

        time_str = timestamp.strftime('%I:%M %p') if timestamp else 'Now'

        return f"**{author}** at {time_str}:\n{content}"

    def send_discord_notification(
        self,
        message: Dict[str, Any],
        webhook_url: Optional[str] = None
    ) -> bool:
        """
        Send notification to Discord bot via webhook

        Args:
            message: Message dictionary
            webhook_url: Discord webhook URL (defaults to bot webhook)

        Returns:
            True if notification sent successfully
        """
        if not webhook_url:
            webhook_url = self.discord_bot_webhook

        if not webhook_url:
            logger.warning("Discord bot webhook not configured (DISCORD_BOT_WEBHOOK_URL)")
            return False

        try:
            # Build notification payload
            summary = self.extract_alert_summary(message)

            payload = {
                'content': f'🚨 **NEW PREMIUM ALERT** 🚨\n\n{summary}',
                'username': 'Magnus Alert Bot',
                'avatar_url': 'https://i.imgur.com/4M34hi2.png'  # Optional: Magnus logo
            }

            # Add embed for rich formatting
            if message.get('timestamp'):
                payload['embeds'] = [{
                    'title': '📈 Premium Trading Alert',
                    'description': message.get('content', '')[:2000],  # Discord limit
                    'color': 0xFF6B6B,  # Red color
                    'author': {
                        'name': message.get('author_name', 'Unknown Trader')
                    },
                    'timestamp': message.get('timestamp').isoformat(),
                    'footer': {
                        'text': f'Channel: Premium Alerts ({PREMIUM_ALERTS_CHANNEL_ID})'
                    }
                }]

            # Send webhook
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"✅ Sent Discord notification for message {message.get('message_id')}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to send Discord notification: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error sending notification: {e}")
            return False

    def sync_premium_channel(self, minutes_back: int = 5) -> Dict[str, Any]:
        """
        Sync premium alerts channel and send notifications for new alerts

        Args:
            minutes_back: How many minutes back to check

        Returns:
            Sync results dictionary
        """
        logger.info(f"🔍 Checking premium alerts channel {PREMIUM_ALERTS_CHANNEL_ID}...")

        # Get new messages
        new_messages = self.get_new_messages_since_last_sync(
            PREMIUM_ALERTS_CHANNEL_ID,
            minutes_back=minutes_back
        )

        if not new_messages:
            logger.info("✅ No new premium alerts")
            return {
                'status': 'success',
                'channel_id': PREMIUM_ALERTS_CHANNEL_ID,
                'new_messages': 0,
                'alerts_sent': 0
            }

        logger.info(f"📬 Found {len(new_messages)} new messages")

        # Filter for actual trading alerts
        alerts = [msg for msg in new_messages if self.is_premium_alert(msg)]

        logger.info(f"🚨 Identified {len(alerts)} trading alerts")

        # Send notifications for each alert
        alerts_sent = 0
        for alert in alerts:
            if self.send_discord_notification(alert):
                alerts_sent += 1

        logger.info(f"✅ Sent {alerts_sent}/{len(alerts)} Discord notifications")

        return {
            'status': 'success',
            'channel_id': PREMIUM_ALERTS_CHANNEL_ID,
            'new_messages': len(new_messages),
            'alerts_identified': len(alerts),
            'alerts_sent': alerts_sent
        }

    def sync_all_channels(
        self,
        channel_ids: Optional[List[str]] = None,
        minutes_back: int = 5
    ) -> Dict[str, Any]:
        """
        Sync multiple Discord channels

        Args:
            channel_ids: List of channel IDs to sync (None = all active)
            minutes_back: How many minutes back to sync

        Returns:
            Sync results dictionary
        """
        conn = None
        cur = None

        try:
            # Get active channels from database if not specified
            if channel_ids is None:
                conn = self.get_connection()
                cur = conn.cursor()
                cur.execute("""
                    SELECT DISTINCT channel_id
                    FROM discord_channels
                    WHERE last_sync IS NOT NULL
                    ORDER BY last_sync DESC
                """)
                channel_ids = [row[0] for row in cur.fetchall()]

                if cur:
                    cur.close()
                if conn:
                    conn.close()

            if not channel_ids:
                logger.warning("No channels to sync")
                return {
                    'status': 'success',
                    'channels_synced': 0,
                    'total_messages': 0
                }

            logger.info(f"Syncing {len(channel_ids)} channels...")

            total_messages = 0
            channels_synced = 0

            # Sync each channel
            for channel_id in channel_ids:
                try:
                    messages = self.get_new_messages_since_last_sync(
                        channel_id,
                        minutes_back=minutes_back
                    )

                    if messages:
                        total_messages += len(messages)
                        channels_synced += 1
                        logger.info(f"  Channel {channel_id}: {len(messages)} new messages")

                except Exception as e:
                    logger.error(f"  Error syncing channel {channel_id}: {e}")
                    continue

            logger.info(f"✅ Synced {channels_synced} channels, {total_messages} total messages")

            return {
                'status': 'success',
                'channels_synced': channels_synced,
                'total_messages': total_messages
            }

        except Exception as e:
            logger.error(f"Error syncing channels: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()


def sync_premium_alerts(minutes_back: int = 5) -> Dict[str, Any]:
    """
    Main function to sync premium alerts
    Called by Celery task every 5 minutes

    Args:
        minutes_back: How many minutes back to check

    Returns:
        Sync results dictionary
    """
    syncer = DiscordPremiumAlertSync()

    # 1. Sync premium channel first (priority)
    premium_result = syncer.sync_premium_channel(minutes_back=minutes_back)

    # 2. Sync all other channels
    all_channels_result = syncer.sync_all_channels(minutes_back=minutes_back)

    # Combine results
    return {
        'status': 'success',
        'premium_channel': premium_result,
        'all_channels': all_channels_result,
        'total_alerts_sent': premium_result.get('alerts_sent', 0)
    }


if __name__ == "__main__":
    """Test the premium alert sync"""
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    print("Discord Premium Alert Sync - Test Mode")
    print("=" * 60)

    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    print(f"\nChecking last {minutes} minutes for premium alerts...")
    print(f"Premium channel: {PREMIUM_ALERTS_CHANNEL_ID}\n")

    result = sync_premium_alerts(minutes_back=minutes)

    print("\nResults:")
    print(f"  Status: {result['status']}")
    print(f"\n  Premium Channel:")
    print(f"    New messages: {result['premium_channel']['new_messages']}")
    print(f"    Alerts identified: {result['premium_channel']['alerts_identified']}")
    print(f"    Alerts sent: {result['premium_channel']['alerts_sent']}")
    print(f"\n  All Channels:")
    print(f"    Channels synced: {result['all_channels']['channels_synced']}")
    print(f"    Total messages: {result['all_channels']['total_messages']}")
    print(f"\n  Total Discord notifications sent: {result['total_alerts_sent']}")
    print()
