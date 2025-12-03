"""
Discord to Telegram Alert System
Detects important trading signals from Discord and sends Telegram alerts
"""

import os
import re
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

# Import existing Telegram notifier
try:
    from src.notifications.telegram_notifier import TelegramNotifier
except ImportError:
    # Fallback to direct import
    from src.telegram_notifier import TelegramNotifier

load_dotenv()
logger = logging.getLogger(__name__)


class DiscordTelegramAlerts:
    """Detect important Discord messages and send Telegram alerts"""

    def __init__(self) -> None:
        self.telegram = TelegramNotifier()

        # Database connection
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'magnus')
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', '')

        # Alert criteria
        self.min_confidence_score = 70  # Minimum confidence for alerts
        self.high_value_keywords = ['strong buy', 'lock', 'max confidence', 'high conviction']
        self.ticker_patterns = [r'\$([A-Z]{1,5})', r'\b([A-Z]{2,5})\b']

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def analyze_message_importance(self, message: Dict) -> Optional[Dict]:
        """
        Analyze if a message is important enough for an alert

        Returns:
            Dict with alert details if important, None otherwise
        """
        content = message.get('content', '').lower()

        # Calculate importance score
        score = 0
        reasons = []

        # Check for tickers
        tickers = []
        for pattern in self.ticker_patterns:
            matches = re.findall(pattern, message.get('content', '').upper())
            tickers.extend(matches)

        if tickers:
            score += 30
            reasons.append(f"Tickers: {', '.join(set(tickers[:3]))}")

        # Check for action keywords
        if any(word in content for word in ['buy', 'long', 'call']):
            score += 20
            reasons.append("Bullish action")
        elif any(word in content for word in ['sell', 'short', 'put']):
            score += 20
            reasons.append("Bearish action")

        # Check for price targets
        price_pattern = r'\$?(\d+(?:\.\d{1,2})?)'
        prices = re.findall(price_pattern, message.get('content', ''))
        if len(prices) >= 2:
            score += 15
            reasons.append("Price targets included")

        # Check for high-value keywords
        for keyword in self.high_value_keywords:
            if keyword in content:
                score += 25
                reasons.append(f"High confidence: '{keyword}'")
                break

        # Check for options-specific content
        if any(word in content for word in ['strike', 'expiration', 'expiry', 'dte']):
            score += 15
            reasons.append("Options trade")

        # Check for urgency
        if any(word in content for word in ['now', 'immediately', 'urgent', 'breaking']):
            score += 10
            reasons.append("Urgent")

        # Only alert if score meets threshold
        if score >= self.min_confidence_score:
            return {
                'message': message,
                'score': score,
                'reasons': reasons,
                'tickers': list(set(tickers))[:3]  # Max 3 tickers
            }

        return None

    def format_alert_message(self, alert: Dict) -> str:
        """Format alert for Telegram"""
        msg = alert['message']

        # Build alert text
        lines = []
        lines.append("🚨 <b>Trading Alert</b>")
        lines.append("")

        # Tickers
        if alert['tickers']:
            lines.append(f"📊 <b>Tickers:</b> {', '.join(alert['tickers'])}")

        # Source
        channel_name = msg.get('channel_name', 'Unknown')
        author_name = msg.get('author_name', 'Unknown')
        lines.append(f"📱 <b>Source:</b> {channel_name}")
        lines.append(f"👤 <b>Author:</b> {author_name}")

        # Confidence
        lines.append(f"⭐ <b>Confidence:</b> {alert['score']}/100")

        # Reasons
        if alert['reasons']:
            lines.append("")
            lines.append("<b>Why this is important:</b>")
            for reason in alert['reasons']:
                lines.append(f"  • {reason}")

        # Message content
        lines.append("")
        lines.append("<b>Message:</b>")
        lines.append(f"<code>{msg.get('content', '')[:500]}</code>")

        # Timestamp
        timestamp = msg.get('timestamp')
        if timestamp:
            lines.append("")
            lines.append(f"🕐 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def check_and_alert(self, hours_back: int = 1):
        """
        Check recent messages and send alerts for important ones

        Args:
            hours_back: How many hours back to check (default 1)
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get recent messages that haven't been alerted on
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
                AND NOT EXISTS (
                    SELECT 1 FROM discord_alerts
                    WHERE message_id = m.message_id
                )
                ORDER BY m.timestamp DESC
            """, (hours_back,))

            messages = cur.fetchall()

            if not messages:
                logger.info(f"No new messages in last {hours_back} hour(s)")
                cur.close()
                conn.close()
                return

            logger.info(f"Analyzing {len(messages)} messages for alerts...")

            alerts_sent = 0

            for msg in messages:
                # Analyze message
                alert = self.analyze_message_importance(dict(msg))

                if alert:
                    # Format and send alert
                    alert_text = self.format_alert_message(alert)

                    try:
                        self.telegram.send_message(alert_text, parse_mode='HTML')
                        logger.info(f"Alert sent for message {msg['message_id']} (score: {alert['score']})")
                        alerts_sent += 1

                        # Record that we alerted on this message
                        cur.execute("""
                            INSERT INTO discord_alerts (message_id, alert_sent_at, confidence_score)
                            VALUES (%s, NOW(), %s)
                            ON CONFLICT (message_id) DO NOTHING
                        """, (msg['message_id'], alert['score']))
                        conn.commit()

                    except Exception as e:
                        logger.error(f"Failed to send alert: {e}")

            logger.info(f"Sent {alerts_sent} alert(s) out of {len(messages)} messages")

            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error checking for alerts: {e}", exc_info=True)

    def test_alert(self) -> None:
        """Send a test alert to verify Telegram is working"""
        test_message = """
🚨 <b>Test Alert</b>

This is a test alert from Magnus Discord Alert System.

📊 <b>System Status:</b> ✅ Working
📱 <b>Telegram:</b> ✅ Connected

If you see this message, your alert system is configured correctly!
        """

        try:
            self.telegram.send_message(test_message, parse_mode='HTML')
            logger.info("Test alert sent successfully")
            return True
        except Exception as e:
            logger.error(f"Test alert failed: {e}")
            return False


def create_alerts_table():
    """Create table to track which messages have been alerted on"""
    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '')
    )

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS discord_alerts (
            message_id BIGINT PRIMARY KEY REFERENCES discord_messages(message_id),
            alert_sent_at TIMESTAMP NOT NULL,
            confidence_score INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_discord_alerts_sent_at
        ON discord_alerts(alert_sent_at DESC);
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("[OK] discord_alerts table created")


if __name__ == "__main__":
    # Create alerts table if running standalone
    create_alerts_table()

    # Test the alert system
    alerts = DiscordTelegramAlerts()

    print("\nTesting Telegram alert system...")
    if alerts.test_alert():
        print("[OK] Test alert sent successfully!")
        print("\nNow checking for recent important messages...")
        alerts.check_and_alert(hours_back=24)
    else:
        print("[ERROR] Test alert failed!")
        print("Check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
