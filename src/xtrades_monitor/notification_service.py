"""
Telegram Notification Service
==============================

Sends high-quality trade alerts via Telegram with:
- Rate limiting (5 alerts per hour)
- Priority queue
- Retry logic
- Rich formatting with markdown
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import requests
from enum import Enum

# Import connection pool
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from db_connection_pool import get_db_pool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class NotificationStatus(Enum):
    """Notification status"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    CANCELLED = "cancelled"


class TelegramNotificationService:
    """
    Telegram notification service with rate limiting and retry logic.

    Features:
    - Max 5 notifications per hour
    - Priority queue (high score = high priority)
    - Auto-retry on failure (max 3 retries)
    - Rich markdown formatting
    """

    def __init__(self) -> None:
        """Initialize Telegram notification service with connection pooling"""
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment")

        # Get connection pool instance
        self.pool = get_db_pool()

        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.bot_token_backup = os.getenv("TELEGRAM_BOT_TOKEN_BACKUP")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token:
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN not found in environment")

        if not self.chat_id or self.chat_id == "YOUR_CHAT_ID_HERE":
            logger.warning("⚠️ TELEGRAM_CHAT_ID not configured. Please update .env file.")

        self.telegram_api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        logger.info("✅ Telegram notification service initialized with connection pooling")

    def can_send_notification(self) -> bool:
        """
        Check if we can send a notification (rate limit check).

        Returns:
            True if under rate limit, False otherwise
        """
        try:
            with self.pool.get_cursor() as cursor:
                # Call database function (parameterized - safe from SQL injection)
                cursor.execute("SELECT can_send_notification()")
                can_send = cursor.fetchone()[0]
                return can_send

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False

    def queue_notification(self, alert_id: int, evaluation: Dict[str, Any]) -> int:
        """
        Add notification to queue with transaction safety.

        Args:
            alert_id: Database ID of the alert
            evaluation: Evaluation results with trade details

        Returns:
            notification_id: Database ID of queued notification
        """
        try:
            with self.pool.get_connection() as conn:
                with conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        # Get trade details (parameterized query - safe from SQL injection)
                        cursor.execute("""
                            SELECT a.*, t.ticker, t.action, t.strike_price,
                                   t.expiration_date, t.entry_price, t.quantity,
                                   prof.username as trader_username
                            FROM xtrades_alerts a
                            JOIN xtrades_trades t ON a.trade_id = t.id
                            JOIN xtrades_profiles prof ON t.profile_id = prof.id
                            WHERE a.id = %s
                        """, (alert_id,))

                        alert_data = cursor.fetchone()

                        if not alert_data:
                            logger.error(f"Alert {alert_id} not found")
                            return None

                        # Generate notification message
                        message = self._format_notification_message(dict(alert_data))

                        # Calculate priority (higher score = higher priority)
                        consensus_score = evaluation.get('consensus_score', 0)
                        priority = 10 - int(consensus_score / 10)  # 90-100 score = priority 1

                        ticker = alert_data.get('ticker', 'UNKNOWN')

                        # Insert into queue (all parameters sanitized)
                        cursor.execute("""
                            INSERT INTO xtrades_notification_queue (
                                alert_id, trade_id, notification_type, priority,
                                message_title, message_body, status
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            RETURNING id
                        """, (
                            alert_id,
                            alert_data['trade_id'],
                            'telegram',
                            priority,
                            f"🔔 HIGH-QUALITY TRADE ALERT: {ticker}",
                            message,
                            NotificationStatus.PENDING.value
                        ))

                        notification_id = cursor.fetchone()['id']

                        # Transaction commits here automatically

                logger.info(f"📬 Queued notification {notification_id} for alert {alert_id} (priority={priority})")
                return notification_id

        except Exception as e:
            logger.error(f"Error queuing notification: {e}", exc_info=True)
            # Transaction automatically rolled back by context manager
            return None

    def _format_notification_message(self, alert_data: Dict[str, Any]) -> str:
        """
        Format notification message with rich markdown.

        Creates a comprehensive, actionable alert message.
        """
        ticker = alert_data.get('ticker', 'UNKNOWN')
        trader = alert_data.get('trader_username', 'Unknown')
        strategy = alert_data.get('strategy_name', 'Unknown')
        score = alert_data.get('consensus_score', 0)
        recommendation = alert_data.get('recommendation', 'HOLD')

        action = alert_data.get('action', '')
        strike = alert_data.get('strike_price', 0)
        premium = alert_data.get('entry_price', 0)
        expiry = alert_data.get('expiration_date', '')

        ai_reasoning = alert_data.get('ai_reasoning', 'No reasoning available')
        key_risk = alert_data.get('key_risk', 'Standard options risk')

        # Calculate DTE
        if expiry:
            try:
                if isinstance(expiry, str):
                    expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                else:
                    expiry_date = expiry
                dte = (expiry_date - datetime.now().date()).days
            except:
                dte = 0
        else:
            dte = 0

        # Calculate max profit/loss (for CSP)
        max_profit = float(premium) * 100 * alert_data.get('quantity', 1)
        max_loss = (float(strike) * 100 - max_profit) * alert_data.get('quantity', 1)

        message = f"""🔔 **HIGH-QUALITY TRADE ALERT**

**Symbol:** {ticker}
**Trader:** @{trader}
**Strategy:** {strategy}
**Score:** {score}/100 {'🔥' if score >= 90 else '⭐' if score >= 80 else ''}

**Entry Details:**
• Action: {action} ${strike:.2f} {'PUT' if 'PUT' in strategy.upper() or 'CSP' in strategy.upper() else 'CALL'} @ ${premium:.2f}
• Expiry: {expiry} ({dte} DTE)
• Win Rate: {self._get_win_rate_by_strategy(strategy)}

**AI Analysis:**
{ai_reasoning}

**⚠️ Key Risk:**
{key_risk}

**💰 Profit/Loss:**
• Max Profit: ${max_profit:,.0f}
• Max Loss: ${max_loss:,.0f}
• Return on Capital: {(max_profit / max_loss * 100):.1f}%

**Recommendation:** {self._format_recommendation(recommendation)}

---
🤖 *AI-Powered Analysis by Magnus Trading Platform*
"""

        return message

    def _get_win_rate_by_strategy(self, strategy: str) -> str:
        """Get typical win rate for strategy"""
        win_rates = {
            'CSP': '60-70%',
            'Cash-Secured Put': '60-70%',
            'Iron Condor': '75-85%',
            'Bull Put Spread': '60-70%',
            'Bear Call Spread': '60-70%',
            'Covered Call': '60-70%',
            'PMCC': '55-65%',
            'Calendar Spread': '60-70%',
            'Diagonal Spread': '55-65%',
            'Long Straddle': '30-40%',
            'Short Strangle': '70-80%'
        }

        for key, rate in win_rates.items():
            if key.lower() in strategy.lower():
                return rate

        return '50-60%'

    def _format_recommendation(self, recommendation: str) -> str:
        """Format recommendation with emoji"""
        emoji_map = {
            'STRONG_BUY': '🚀 STRONG BUY',
            'BUY': '✅ BUY',
            'HOLD': '⚠️ HOLD',
            'AVOID': '❌ AVOID'
        }
        return emoji_map.get(recommendation, recommendation)

    def send_pending_notifications(self) -> Dict[str, int]:
        """
        Process pending notifications in queue with transaction safety.

        Respects rate limits and priority order.

        Returns:
            Dict with counts: sent, rate_limited, failed
        """
        stats = {'sent': 0, 'rate_limited': 0, 'failed': 0}

        try:
            with self.pool.get_connection() as conn:
                with conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        # Get pending notifications (ordered by priority) - parameterized query
                        cursor.execute("""
                            SELECT *
                            FROM xtrades_notification_queue
                            WHERE status = %s
                            ORDER BY priority ASC, created_at ASC
                            LIMIT %s
                        """, (NotificationStatus.PENDING.value, 10))

                        pending = cursor.fetchall()

                        logger.info(f"📬 Processing {len(pending)} pending notifications...")

                        for notification in pending:
                            # Check rate limit
                            if not self.can_send_notification():
                                # Mark as rate limited (parameterized)
                                cursor.execute("""
                                    UPDATE xtrades_notification_queue
                                    SET status = %s,
                                        next_retry_at = NOW() + INTERVAL '1 hour'
                                    WHERE id = %s
                                """, (NotificationStatus.RATE_LIMITED.value, notification['id']))

                                stats['rate_limited'] += 1
                                logger.info(f"⏸️ Notification {notification['id']} rate limited")
                                break  # Stop processing - at rate limit

                            # Try to send
                            success = self._send_telegram_message(
                                notification['message_title'],
                                notification['message_body']
                            )

                            if success:
                                # Mark as sent (parameterized)
                                cursor.execute("""
                                    UPDATE xtrades_notification_queue
                                    SET status = %s,
                                        sent_at = NOW(),
                                        telegram_chat_id = %s
                                    WHERE id = %s
                                """, (NotificationStatus.SENT.value, self.chat_id, notification['id']))

                                # Record in rate limiter (parameterized)
                                cursor.execute("SELECT record_notification_sent(%s)", (notification['id'],))

                                stats['sent'] += 1
                                logger.info(f"✅ Sent notification {notification['id']}")

                            else:
                                # Increment retry count
                                new_retry_count = notification['retry_count'] + 1

                                if new_retry_count >= notification['max_retries']:
                                    # Max retries reached - mark as failed (parameterized)
                                    cursor.execute("""
                                        UPDATE xtrades_notification_queue
                                        SET status = %s,
                                            failed_at = NOW(),
                                            retry_count = %s,
                                            error_message = %s
                                        WHERE id = %s
                                    """, (NotificationStatus.FAILED.value, new_retry_count, 'Max retries reached', notification['id']))

                                    stats['failed'] += 1
                                    logger.error(f"❌ Notification {notification['id']} failed (max retries)")

                                else:
                                    # Schedule retry (parameterized)
                                    cursor.execute("""
                                        UPDATE xtrades_notification_queue
                                        SET retry_count = %s,
                                            next_retry_at = NOW() + INTERVAL '5 minutes'
                                        WHERE id = %s
                                    """, (new_retry_count, notification['id']))

                                    logger.warning(f"🔄 Notification {notification['id']} will retry (attempt {new_retry_count})")

                        # Transaction commits here automatically

                logger.info(
                    f"📊 Notification batch complete: "
                    f"{stats['sent']} sent, {stats['rate_limited']} rate limited, "
                    f"{stats['failed']} failed"
                )

                return stats

        except Exception as e:
            logger.error(f"Error processing notifications: {e}", exc_info=True)
            # Transaction automatically rolled back by context manager
            return stats

    def _send_telegram_message(self, title: str, message: str) -> bool:
        """
        Send message via Telegram Bot API.

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.bot_token or not self.chat_id or self.chat_id == "YOUR_CHAT_ID_HERE":
            logger.warning("⚠️ Telegram not configured - skipping send")
            return False

        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }

            response = requests.post(self.telegram_api_url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info("✅ Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")

                # Try backup token if primary fails
                if self.bot_token_backup and response.status_code == 401:
                    logger.info("🔄 Trying backup bot token...")
                    backup_url = f"https://api.telegram.org/bot{self.bot_token_backup}/sendMessage"
                    response = requests.post(backup_url, json=payload, timeout=10)

                    if response.status_code == 200:
                        logger.info("✅ Sent via backup token")
                        return True

                return False

        except requests.exceptions.Timeout:
            logger.error("❌ Telegram API timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            return False


if __name__ == "__main__":
    # Test notification service
    print("🧪 Testing Telegram Notification Service...")

    service = TelegramNotificationService()

    # Test rate limit check
    can_send = service.can_send_notification()
    print(f"\n✅ Can send notification: {can_send}")

    # Test message formatting
    test_alert = {
        'ticker': 'AAPL',
        'trader_username': 'behappy',
        'strategy_name': 'Cash-Secured Put',
        'consensus_score': 87,
        'recommendation': 'STRONG_BUY',
        'action': 'BTO',
        'strike_price': 170.00,
        'entry_price': 2.50,
        'expiration_date': '2025-12-06',
        'quantity': 1,
        'ai_reasoning': 'High IV (35%) + Bullish trend provides excellent premium. Similar trades have 80% success rate with avg +12% return.',
        'key_risk': 'Watch for earnings volatility next week'
    }

    message = service._format_notification_message(test_alert)
    print(f"\n📱 Sample Notification Message:")
    print("=" * 60)
    print(message)
    print("=" * 60)

    # Test sending (if configured)
    if service.bot_token and service.chat_id and service.chat_id != "YOUR_CHAT_ID_HERE":
        print(f"\n📤 Sending test notification...")
        success = service._send_telegram_message("Test Alert", message)
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    else:
        print(f"\n⚠️ Telegram not fully configured - skipping actual send test")
        print(f"To enable:")
        print(f"1. Message your bot: /start")
        print(f"2. Get chat ID from: https://api.telegram.org/bot{service.bot_token}/getUpdates")
        print(f"3. Update .env: TELEGRAM_CHAT_ID=your_chat_id")
