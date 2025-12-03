"""
AVA Alert Service - Intelligent Alert System for Passive Income Advisor

This service provides:
- Alert creation with automatic deduplication
- Multi-channel delivery (Telegram + Email)
- Priority-based routing
- Rate limiting per channel
- Delivery tracking with retry support

Usage:
    from src.services.alert_service import AlertService, AlertCategory, AlertPriority

    alert_service = AlertService()
    alert_service.create_alert(
        category=AlertCategory.OPPORTUNITY_CSP,
        priority=AlertPriority.IMPORTANT,
        title="High-Quality CSP Opportunity",
        message="NVDA $120P Dec 20 - 4.2% monthly return",
        symbol="NVDA",
        metadata={"strike": 120, "expiration": "2024-12-20", "return": 4.2}
    )

Author: AVA Trading Platform
Created: 2025-11-28
"""

import os
import logging
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
import json

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Import existing Telegram notifier
try:
    from src.telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None

load_dotenv()
logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels"""
    URGENT = "urgent"           # Immediate delivery, bypass some rate limits
    IMPORTANT = "important"      # Standard delivery
    INFORMATIONAL = "informational"  # Low priority, respects quiet hours


class AlertCategory(Enum):
    """Alert categories"""
    ASSIGNMENT_RISK = "assignment_risk"
    EARNINGS_PROXIMITY = "earnings_proximity"
    OPPORTUNITY_CSP = "opportunity_csp"
    OPPORTUNITY_CC = "opportunity_cc"
    IV_SPIKE = "iv_spike"
    XTRADES_NEW = "xtrades_new"
    MARGIN_WARNING = "margin_warning"
    THETA_DECAY = "theta_decay"
    EXPIRATION_REMINDER = "expiration_reminder"
    GOAL_PROGRESS = "goal_progress"
    REPORT_READY = "report_ready"
    INFORMATIONAL = "informational"


@dataclass
class Alert:
    """Alert data structure"""
    id: Optional[int] = None
    category: AlertCategory = AlertCategory.INFORMATIONAL
    priority: AlertPriority = AlertPriority.INFORMATIONAL
    title: str = ""
    message: str = ""
    symbol: Optional[str] = None
    metadata: Optional[Dict] = None
    fingerprint: Optional[str] = None
    created_at: Optional[datetime] = None


class AlertService:
    """
    Comprehensive alert service for AVA trading platform.

    Features:
    - Automatic deduplication via fingerprinting
    - Multi-channel delivery (Telegram, Email)
    - Priority-based routing
    - Rate limiting
    - Delivery tracking
    """

    # Default rate limits per hour by channel
    DEFAULT_RATE_LIMITS = {
        "telegram": 10,
        "email": 5
    }

    # Priority-based rate limit multipliers
    PRIORITY_RATE_MULTIPLIERS = {
        AlertPriority.URGENT: 2.0,       # 2x normal rate limit
        AlertPriority.IMPORTANT: 1.0,    # Normal rate limit
        AlertPriority.INFORMATIONAL: 0.5  # Half rate limit
    }

    # Category to emoji mapping for Telegram
    CATEGORY_EMOJIS = {
        AlertCategory.ASSIGNMENT_RISK: "\u26A0\uFE0F",      # Warning
        AlertCategory.EARNINGS_PROXIMITY: "\U0001F4C5",    # Calendar
        AlertCategory.OPPORTUNITY_CSP: "\U0001F4B0",       # Money bag
        AlertCategory.OPPORTUNITY_CC: "\U0001F4B5",        # Dollar
        AlertCategory.IV_SPIKE: "\U0001F4C8",              # Chart up
        AlertCategory.XTRADES_NEW: "\U0001F195",           # New
        AlertCategory.MARGIN_WARNING: "\U0001F6A8",        # Alert
        AlertCategory.THETA_DECAY: "\u23F3",               # Hourglass
        AlertCategory.EXPIRATION_REMINDER: "\u23F0",       # Alarm
        AlertCategory.GOAL_PROGRESS: "\U0001F3AF",         # Target
        AlertCategory.REPORT_READY: "\U0001F4CA",          # Chart
    }

    def __init__(
        self,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        user_id: str = "default_user"
    ):
        """
        Initialize the AlertService.

        Args:
            db_*: Database connection parameters (defaults to env vars)
            user_id: User identifier for preferences and rate limiting
        """
        self.user_id = user_id

        # Database configuration
        self.db_config = {
            "host": db_host or os.getenv("DB_HOST", "localhost"),
            "port": db_port or int(os.getenv("DB_PORT", "5432")),
            "database": db_name or os.getenv("DB_NAME", "wheel_strategy"),
            "user": db_user or os.getenv("DB_USER", "postgres"),
            "password": db_password or os.getenv("DB_PASSWORD", "")
        }

        # Initialize Telegram notifier
        self.telegram = TelegramNotifier() if TelegramNotifier else None

        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("EMAIL_FROM", self.smtp_username)
        self.to_email = os.getenv("EMAIL_TO", "")
        self.email_enabled = bool(self.smtp_username and self.smtp_password and self.to_email)

        logger.info(f"AlertService initialized (Telegram: {self.telegram is not None and self.telegram.enabled}, Email: {self.email_enabled})")

    def _get_db_connection(self):
        """Get a database connection."""
        return psycopg2.connect(**self.db_config)

    def create_alert(
        self,
        category: AlertCategory,
        priority: AlertPriority,
        title: str,
        message: str,
        symbol: Optional[str] = None,
        metadata: Optional[Dict] = None,
        position_id: Optional[str] = None,
        expires_in_hours: int = 24
    ) -> Optional[int]:
        """
        Create a new alert with automatic deduplication.

        Args:
            category: Alert category
            priority: Alert priority level
            title: Short title for the alert
            message: Full alert message
            symbol: Related stock symbol (optional)
            metadata: Additional data (optional)
            position_id: Related position UUID (optional)
            expires_in_hours: Hours until alert expires

        Returns:
            Alert ID if created, None if duplicate or error
        """
        try:
            # Generate fingerprint for deduplication
            fingerprint = self._generate_fingerprint(category, symbol, metadata)

            # Check if duplicate exists
            if self._is_duplicate(fingerprint):
                logger.debug(f"Duplicate alert detected, fingerprint: {fingerprint[:16]}...")
                return None

            # Insert alert into database
            alert_id = self._insert_alert(
                category=category,
                priority=priority,
                title=title,
                message=message,
                symbol=symbol,
                metadata=metadata,
                fingerprint=fingerprint,
                position_id=position_id,
                expires_in_hours=expires_in_hours
            )

            if alert_id:
                # Queue for delivery
                self._queue_delivery(alert_id, category, priority)
                logger.info(f"Alert created: {category.value} - {title} (ID: {alert_id})")

            return alert_id

        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return None

    def _generate_fingerprint(
        self,
        category: AlertCategory,
        symbol: Optional[str],
        metadata: Optional[Dict]
    ) -> str:
        """
        Generate unique fingerprint for alert deduplication.

        Different categories have different deduplication strategies.
        """
        metadata = metadata or {}

        if category == AlertCategory.ASSIGNMENT_RISK:
            # Dedupe by symbol + strike + expiration
            key = f"{category.value}:{symbol or ''}:{metadata.get('strike', '')}:{metadata.get('expiration', '')}"

        elif category in [AlertCategory.OPPORTUNITY_CSP, AlertCategory.OPPORTUNITY_CC]:
            # Dedupe by symbol + strike + current hour (allow new alerts hourly)
            hour_bucket = datetime.now().strftime("%Y%m%d%H")
            key = f"{category.value}:{symbol or ''}:{metadata.get('strike', '')}:{hour_bucket}"

        elif category == AlertCategory.XTRADES_NEW:
            # Dedupe by trade_id
            key = f"{category.value}:{metadata.get('trade_id', '')}"

        elif category == AlertCategory.EARNINGS_PROXIMITY:
            # Dedupe by symbol + earnings_date
            key = f"{category.value}:{symbol or ''}:{metadata.get('earnings_date', '')}"

        elif category == AlertCategory.IV_SPIKE:
            # Dedupe by symbol + current day
            day_bucket = datetime.now().strftime("%Y%m%d")
            key = f"{category.value}:{symbol or ''}:{day_bucket}"

        elif category == AlertCategory.REPORT_READY:
            # Dedupe by report_type + date
            key = f"{category.value}:{metadata.get('report_type', '')}:{metadata.get('report_date', '')}"

        else:
            # Default: symbol + hour bucket
            hour_bucket = datetime.now().strftime("%Y%m%d%H")
            key = f"{category.value}:{symbol or ''}:{hour_bucket}"

        return hashlib.md5(key.encode()).hexdigest()

    def _is_duplicate(self, fingerprint: str) -> bool:
        """Check if an alert with this fingerprint already exists and is active."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) FROM ava_alerts
                        WHERE fingerprint = %s
                          AND is_active = TRUE
                          AND (expires_at IS NULL OR expires_at > NOW())
                    """, (fingerprint,))
                    count = cur.fetchone()[0]
                    return count > 0
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False

    def _insert_alert(
        self,
        category: AlertCategory,
        priority: AlertPriority,
        title: str,
        message: str,
        symbol: Optional[str],
        metadata: Optional[Dict],
        fingerprint: str,
        position_id: Optional[str],
        expires_in_hours: int
    ) -> Optional[int]:
        """Insert alert into database."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ava_alerts (
                            category, priority, title, message, symbol,
                            metadata, fingerprint, position_id, expires_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW() + INTERVAL '%s hours')
                        RETURNING id
                    """, (
                        category.value,
                        priority.value,
                        title,
                        message,
                        symbol,
                        json.dumps(metadata) if metadata else '{}',
                        fingerprint,
                        position_id,
                        expires_in_hours
                    ))
                    alert_id = cur.fetchone()[0]
                    conn.commit()
                    return alert_id
        except Exception as e:
            logger.error(f"Error inserting alert: {e}")
            return None

    def _queue_delivery(
        self,
        alert_id: int,
        category: AlertCategory,
        priority: AlertPriority
    ) -> None:
        """Queue alert for delivery to configured channels."""
        try:
            # Get user preferences for this category
            prefs = self._get_user_preferences(category)

            if not prefs.get("enabled", True):
                logger.debug(f"Alert category {category.value} disabled for user")
                return

            # Check priority threshold
            if not self._meets_priority_threshold(priority, prefs.get("priority_threshold", "informational")):
                logger.debug(f"Alert priority {priority.value} below threshold")
                return

            # Check quiet hours
            if self._is_quiet_hours(prefs):
                if priority != AlertPriority.URGENT:  # Urgent bypasses quiet hours
                    logger.debug("Quiet hours active, delaying non-urgent alert")
                    return

            # Get configured channels
            channels = prefs.get("channels", ["telegram"])

            # Deliver to each channel
            for channel in channels:
                if channel == "telegram" and self._check_rate_limit("telegram", priority):
                    self._deliver_telegram(alert_id)
                elif channel == "email" and self._check_rate_limit("email", priority):
                    self._deliver_email(alert_id)

        except Exception as e:
            logger.error(f"Error queueing delivery: {e}")

    def _get_user_preferences(self, category: AlertCategory) -> Dict:
        """Get user preferences for a specific alert category."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT enabled, priority_threshold, channels,
                               quiet_hours_enabled, quiet_hours_start, quiet_hours_end,
                               max_per_hour
                        FROM ava_alert_preferences
                        WHERE user_id = %s AND category = %s
                    """, (self.user_id, category.value))
                    row = cur.fetchone()

                    if row:
                        return dict(row)

                    # Return defaults
                    return {
                        "enabled": True,
                        "priority_threshold": "informational",
                        "channels": ["telegram"],
                        "quiet_hours_enabled": False,
                        "max_per_hour": 10
                    }
        except Exception as e:
            logger.error(f"Error getting preferences: {e}")
            return {"enabled": True, "channels": ["telegram"]}

    def _meets_priority_threshold(self, priority: AlertPriority, threshold: str) -> bool:
        """Check if alert priority meets or exceeds threshold."""
        priority_order = [AlertPriority.URGENT, AlertPriority.IMPORTANT, AlertPriority.INFORMATIONAL]
        threshold_enum = AlertPriority(threshold) if isinstance(threshold, str) else threshold

        return priority_order.index(priority) <= priority_order.index(threshold_enum)

    def _is_quiet_hours(self, prefs: Dict) -> bool:
        """Check if currently in quiet hours."""
        if not prefs.get("quiet_hours_enabled", False):
            return False

        start = prefs.get("quiet_hours_start")
        end = prefs.get("quiet_hours_end")

        if not start or not end:
            return False

        now = datetime.now().time()
        start_time = datetime.strptime(str(start), "%H:%M:%S").time() if isinstance(start, str) else start
        end_time = datetime.strptime(str(end), "%H:%M:%S").time() if isinstance(end, str) else end

        # Handle overnight quiet hours (e.g., 22:00 - 07:00)
        if start_time > end_time:
            return now >= start_time or now <= end_time
        else:
            return start_time <= now <= end_time

    def _check_rate_limit(self, channel: str, priority: AlertPriority) -> bool:
        """
        Check rate limit for channel and increment counter.

        Returns True if within limit, False if throttled.
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Get base limit and apply priority multiplier
                    base_limit = self.DEFAULT_RATE_LIMITS.get(channel, 10)
                    multiplier = self.PRIORITY_RATE_MULTIPLIERS.get(priority, 1.0)
                    effective_limit = int(base_limit * multiplier)

                    # Use database function
                    cur.execute("""
                        SELECT check_and_increment_rate_limit(%s, %s::alert_channel, %s)
                    """, (self.user_id, channel, effective_limit))

                    result = cur.fetchone()[0]
                    conn.commit()
                    return result

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error

    def _deliver_telegram(self, alert_id: int) -> bool:
        """Deliver alert via Telegram."""
        if not self.telegram or not self.telegram.enabled:
            self._record_delivery(alert_id, "telegram", "failed", "Telegram not configured")
            return False

        try:
            # Get alert data
            alert = self._get_alert(alert_id)
            if not alert:
                return False

            # Format message
            message = self._format_telegram_message(alert)

            # Send via Telegram notifier
            message_id = self.telegram.send_custom_message(message)

            if message_id:
                self._record_delivery(alert_id, "telegram", "sent", None, message_id)
                return True
            else:
                self._record_delivery(alert_id, "telegram", "failed", "Send failed")
                return False

        except Exception as e:
            logger.error(f"Error delivering Telegram alert: {e}")
            self._record_delivery(alert_id, "telegram", "failed", str(e))
            return False

    def _deliver_email(self, alert_id: int) -> bool:
        """Deliver alert via Email."""
        if not self.email_enabled:
            self._record_delivery(alert_id, "email", "failed", "Email not configured")
            return False

        try:
            # Get alert data
            alert = self._get_alert(alert_id)
            if not alert:
                return False

            # Format email
            subject = f"[AVA Alert] {alert['title']}"
            html_body = self._format_email_body(alert)

            # Create email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = self.to_email

            # Add HTML content
            html_part = MIMEText(html_body, "html")
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            self._record_delivery(alert_id, "email", "sent")
            return True

        except Exception as e:
            logger.error(f"Error delivering email alert: {e}")
            self._record_delivery(alert_id, "email", "failed", str(e))
            return False

    def _get_alert(self, alert_id: int) -> Optional[Dict]:
        """Get alert by ID."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, category, priority, title, message,
                               symbol, metadata, created_at
                        FROM ava_alerts
                        WHERE id = %s
                    """, (alert_id,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting alert: {e}")
            return None

    def _format_telegram_message(self, alert: Dict) -> str:
        """Format alert for Telegram."""
        category = AlertCategory(alert["category"])
        priority = AlertPriority(alert["priority"])
        emoji = self.CATEGORY_EMOJIS.get(category, "\U0001F514")

        # Priority indicator
        priority_indicator = ""
        if priority == AlertPriority.URGENT:
            priority_indicator = "\U0001F6A8 *URGENT* "
        elif priority == AlertPriority.IMPORTANT:
            priority_indicator = "\u2757 "

        # Build message
        message = f"{emoji} {priority_indicator}*{alert['title']}*\n\n"
        message += f"{alert['message']}\n"

        # Add symbol if present
        if alert.get("symbol"):
            message += f"\n\U0001F4C8 Symbol: *{alert['symbol']}*"

        # Add metadata highlights
        metadata = alert.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        if metadata:
            if metadata.get("strike"):
                message += f"\n\U0001F3AF Strike: `${metadata['strike']}`"
            if metadata.get("expiration"):
                message += f"\n\U0001F4C5 Expiration: `{metadata['expiration']}`"
            if metadata.get("return"):
                message += f"\n\U0001F4B0 Return: `{metadata['return']}%`"
            if metadata.get("score"):
                message += f"\n\u2B50 Score: `{metadata['score']}/100`"

        # Timestamp
        message += f"\n\n\U0001F553 {alert['created_at'].strftime('%Y-%m-%d %I:%M %p') if alert.get('created_at') else 'Now'}"

        return message

    def _format_email_body(self, alert: Dict) -> str:
        """Format alert as HTML email."""
        category = AlertCategory(alert["category"])
        priority = AlertPriority(alert["priority"])

        # Priority colors
        priority_colors = {
            AlertPriority.URGENT: "#dc3545",      # Red
            AlertPriority.IMPORTANT: "#fd7e14",   # Orange
            AlertPriority.INFORMATIONAL: "#17a2b8"  # Blue
        }
        color = priority_colors.get(priority, "#17a2b8")

        metadata = alert.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: {color}; color: white; padding: 20px; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .header .priority {{ opacity: 0.9; font-size: 14px; text-transform: uppercase; }}
                .content {{ padding: 20px; }}
                .message {{ font-size: 16px; line-height: 1.6; color: #333; }}
                .details {{ margin-top: 20px; background: #f8f9fa; padding: 15px; border-radius: 4px; }}
                .detail-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e9ecef; }}
                .detail-row:last-child {{ border-bottom: none; }}
                .detail-label {{ font-weight: bold; color: #666; }}
                .detail-value {{ color: #333; }}
                .footer {{ padding: 15px 20px; background: #f8f9fa; font-size: 12px; color: #666; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="priority">{priority.value.upper()}</div>
                    <h1>{alert['title']}</h1>
                </div>
                <div class="content">
                    <div class="message">{alert['message']}</div>
                    <div class="details">
        """

        if alert.get("symbol"):
            html += f'<div class="detail-row"><span class="detail-label">Symbol</span><span class="detail-value">{alert["symbol"]}</span></div>'

        if metadata.get("strike"):
            html += f'<div class="detail-row"><span class="detail-label">Strike</span><span class="detail-value">${metadata["strike"]}</span></div>'

        if metadata.get("expiration"):
            html += f'<div class="detail-row"><span class="detail-label">Expiration</span><span class="detail-value">{metadata["expiration"]}</span></div>'

        if metadata.get("return"):
            html += f'<div class="detail-row"><span class="detail-label">Return</span><span class="detail-value">{metadata["return"]}%</span></div>'

        if metadata.get("score"):
            html += f'<div class="detail-row"><span class="detail-label">Score</span><span class="detail-value">{metadata["score"]}/100</span></div>'

        timestamp = alert['created_at'].strftime('%Y-%m-%d %I:%M %p') if alert.get('created_at') else datetime.now().strftime('%Y-%m-%d %I:%M %p')

        html += f"""
                    </div>
                </div>
                <div class="footer">
                    AVA Trading Platform | {timestamp}
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _record_delivery(
        self,
        alert_id: int,
        channel: str,
        status: str,
        error_message: Optional[str] = None,
        external_message_id: Optional[str] = None
    ) -> None:
        """Record delivery attempt in database."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ava_alert_deliveries (
                            alert_id, channel, status, sent_at, error_message, external_message_id
                        )
                        VALUES (%s, %s::alert_channel, %s, %s, %s, %s)
                    """, (
                        alert_id,
                        channel,
                        status,
                        datetime.now() if status == "sent" else None,
                        error_message,
                        external_message_id
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error recording delivery: {e}")

    # ===== Convenience Methods for Common Alert Types =====

    def alert_assignment_risk(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        current_price: float,
        dte: int,
        itm_pct: float
    ) -> Optional[int]:
        """Create assignment risk alert."""
        message = (
            f"Your {symbol} ${strike}P expiring {expiration} is ITM by {itm_pct:.1f}%.\n"
            f"Current price: ${current_price:.2f}, DTE: {dte} days.\n"
            f"Consider rolling or closing before assignment."
        )

        return self.create_alert(
            category=AlertCategory.ASSIGNMENT_RISK,
            priority=AlertPriority.URGENT if dte <= 1 else AlertPriority.IMPORTANT,
            title=f"Assignment Risk: {symbol} ${strike}P",
            message=message,
            symbol=symbol,
            metadata={
                "strike": strike,
                "expiration": expiration,
                "current_price": current_price,
                "dte": dte,
                "itm_pct": itm_pct
            }
        )

    def alert_earnings_proximity(
        self,
        symbol: str,
        earnings_date: str,
        days_away: int,
        position_type: str
    ) -> Optional[int]:
        """Create earnings proximity alert."""
        message = (
            f"You have a {position_type} position in {symbol}.\n"
            f"Earnings announcement in {days_away} days ({earnings_date}).\n"
            f"Consider closing or adjusting before the event."
        )

        return self.create_alert(
            category=AlertCategory.EARNINGS_PROXIMITY,
            priority=AlertPriority.IMPORTANT if days_away <= 3 else AlertPriority.INFORMATIONAL,
            title=f"Earnings Alert: {symbol} in {days_away} days",
            message=message,
            symbol=symbol,
            metadata={
                "earnings_date": earnings_date,
                "days_away": days_away,
                "position_type": position_type
            }
        )

    def alert_opportunity_csp(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        premium: float,
        monthly_return: float,
        score: int
    ) -> Optional[int]:
        """Create CSP opportunity alert."""
        message = (
            f"High-quality Cash-Secured Put opportunity found!\n"
            f"${strike} strike expiring {expiration}\n"
            f"Premium: ${premium:.2f} ({monthly_return:.1f}% monthly return)\n"
            f"Quality Score: {score}/100"
        )

        return self.create_alert(
            category=AlertCategory.OPPORTUNITY_CSP,
            priority=AlertPriority.IMPORTANT if score >= 80 else AlertPriority.INFORMATIONAL,
            title=f"CSP Opportunity: {symbol} ${strike}P",
            message=message,
            symbol=symbol,
            metadata={
                "strike": strike,
                "expiration": expiration,
                "premium": premium,
                "return": monthly_return,
                "score": score
            }
        )

    def alert_opportunity_cc(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        premium: float,
        monthly_return: float,
        score: int
    ) -> Optional[int]:
        """Create Covered Call opportunity alert."""
        message = (
            f"High-quality Covered Call opportunity found!\n"
            f"${strike} strike expiring {expiration}\n"
            f"Premium: ${premium:.2f} ({monthly_return:.1f}% monthly return)\n"
            f"Quality Score: {score}/100"
        )

        return self.create_alert(
            category=AlertCategory.OPPORTUNITY_CC,
            priority=AlertPriority.IMPORTANT if score >= 80 else AlertPriority.INFORMATIONAL,
            title=f"CC Opportunity: {symbol} ${strike}C",
            message=message,
            symbol=symbol,
            metadata={
                "strike": strike,
                "expiration": expiration,
                "premium": premium,
                "return": monthly_return,
                "score": score
            }
        )

    def alert_xtrades_new_trade(
        self,
        profile_username: str,
        symbol: str,
        strategy: str,
        action: str,
        entry_price: float,
        trade_id: int
    ) -> Optional[int]:
        """Create XTrades new trade alert."""
        message = (
            f"New trade from {profile_username}!\n"
            f"{action} {symbol} {strategy} @ ${entry_price:.2f}"
        )

        return self.create_alert(
            category=AlertCategory.XTRADES_NEW,
            priority=AlertPriority.IMPORTANT,
            title=f"XTrades: {profile_username} - {symbol}",
            message=message,
            symbol=symbol,
            metadata={
                "profile_username": profile_username,
                "strategy": strategy,
                "action": action,
                "entry_price": entry_price,
                "trade_id": trade_id
            }
        )

    def alert_report_ready(
        self,
        report_type: str,
        report_date: str,
        summary: str
    ) -> Optional[int]:
        """Create report ready alert."""
        return self.create_alert(
            category=AlertCategory.REPORT_READY,
            priority=AlertPriority.INFORMATIONAL,
            title=f"{report_type.replace('_', ' ').title()} Ready",
            message=summary,
            metadata={
                "report_type": report_type,
                "report_date": report_date
            }
        )

    def alert_goal_progress(
        self,
        goal_name: str,
        current_value: float,
        target_value: float,
        progress_pct: float
    ) -> Optional[int]:
        """Create goal progress alert."""
        status = "exceeded" if progress_pct >= 100 else "on track" if progress_pct >= 75 else "behind"

        message = (
            f"Goal '{goal_name}': ${current_value:,.2f} of ${target_value:,.2f} ({progress_pct:.1f}%)\n"
            f"Status: {status.upper()}"
        )

        return self.create_alert(
            category=AlertCategory.GOAL_PROGRESS,
            priority=AlertPriority.INFORMATIONAL,
            title=f"Goal Progress: {progress_pct:.0f}%",
            message=message,
            metadata={
                "goal_name": goal_name,
                "current_value": current_value,
                "target_value": target_value,
                "progress_pct": progress_pct,
                "status": status
            }
        )

    # ===== Query Methods =====

    def get_recent_alerts(
        self,
        limit: int = 50,
        category: Optional[AlertCategory] = None,
        priority: Optional[AlertPriority] = None
    ) -> List[Dict]:
        """Get recent alerts with optional filtering."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT id, category, priority, title, message,
                               symbol, metadata, is_read, created_at
                        FROM ava_alerts
                        WHERE is_active = TRUE
                    """
                    params = []

                    if category:
                        query += " AND category = %s"
                        params.append(category.value)

                    if priority:
                        query += " AND priority = %s"
                        params.append(priority.value)

                    query += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)

                    cur.execute(query, params)
                    return [dict(row) for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    def mark_alert_read(self, alert_id: int) -> bool:
        """Mark an alert as read."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ava_alerts
                        SET is_read = TRUE, read_at = NOW()
                        WHERE id = %s
                    """, (alert_id,))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error marking alert read: {e}")
            return False

    def get_delivery_status(self, alert_id: int) -> List[Dict]:
        """Get delivery status for an alert."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT channel, status, sent_at, error_message, retry_count
                        FROM ava_alert_deliveries
                        WHERE alert_id = %s
                        ORDER BY created_at DESC
                    """, (alert_id,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting delivery status: {e}")
            return []


# Convenience function
def create_alert_service(**kwargs) -> AlertService:
    """Create and return a configured AlertService instance."""
    return AlertService(**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Alert Service - Example Usage\n")

    service = AlertService()

    # Test creating different types of alerts
    print("Creating test alerts...")

    # Assignment risk alert
    alert_id = service.alert_assignment_risk(
        symbol="NVDA",
        strike=120.0,
        expiration="2024-12-20",
        current_price=118.50,
        dte=2,
        itm_pct=1.25
    )
    print(f"Assignment risk alert created: {alert_id}")

    # CSP opportunity alert
    alert_id = service.alert_opportunity_csp(
        symbol="AAPL",
        strike=180.0,
        expiration="2024-12-27",
        premium=2.50,
        monthly_return=3.8,
        score=85
    )
    print(f"CSP opportunity alert created: {alert_id}")

    # Get recent alerts
    print("\nRecent alerts:")
    alerts = service.get_recent_alerts(limit=5)
    for alert in alerts:
        print(f"  - {alert['title']} ({alert['priority']})")
