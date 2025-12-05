"""
Telegram Notifier for QA System

Sends QA alerts and summaries via AVA's Telegram bot.
Wraps the existing TelegramNotifier from src/telegram_notifier.py
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# Add project root to path for imports
# Use absolute path based on actual file location
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Import from src.telegram_notifier explicitly to avoid circular import
# (this file is also named telegram_notifier.py)
try:
    from src.telegram_notifier import TelegramNotifier as BaseTelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError as e:
    # Log the actual error for debugging
    logging.getLogger(__name__).error(f"Failed to import TelegramNotifier: {e}")
    logging.getLogger(__name__).error(f"Project root: {_PROJECT_ROOT}")
    TELEGRAM_AVAILABLE = False
    BaseTelegramNotifier = None

logger = logging.getLogger(__name__)


class QATelegramNotifier:
    """
    Telegram notifier specialized for QA alerts.

    Sends:
    - Critical failure alerts (immediate)
    - Daily enhancement summaries
    - Health score warnings

    Features:
    - Alert deduplication (suppresses repeated alerts within cooldown period)
    - Persistent alert tracking across restarts
    """

    # Cooldown period for duplicate alerts (in hours)
    ALERT_COOLDOWN_HOURS = 2

    def __init__(self, enabled: bool = True):
        """Initialize the QA Telegram notifier."""
        self.enabled = enabled

        # Alert deduplication tracking
        self._sent_alerts: Dict[str, datetime] = {}
        self._alert_history_file = Path(__file__).parent / "data" / "alert_history.json"

        # Load previous alert history
        self._load_alert_history()

        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "Base TelegramNotifier not available. "
                "QA Telegram notifications disabled."
            )
            self._notifier = None
            return

        try:
            self._notifier = BaseTelegramNotifier()

            # Enhanced diagnostic logging
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            tg_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower()

            logger.info(
                f"Telegram config diagnostic: "
                f"TELEGRAM_ENABLED={tg_enabled}, "
                f"BOT_TOKEN={'set' if bot_token else 'MISSING'}, "
                f"CHAT_ID={'set' if chat_id else 'MISSING'}, "
                f"base_notifier_enabled={self._notifier.enabled if self._notifier else 'N/A'}"
            )

            if self._notifier.enabled:
                logger.info("QA Telegram notifier initialized successfully - notifications ACTIVE")
            else:
                if not bot_token:
                    logger.warning("Telegram disabled: TELEGRAM_BOT_TOKEN not set in .env")
                elif not chat_id:
                    logger.warning("Telegram disabled: TELEGRAM_CHAT_ID not set in .env")
                elif tg_enabled != 'true':
                    logger.warning(f"Telegram disabled: TELEGRAM_ENABLED={tg_enabled} (set to 'true' to enable)")
                else:
                    logger.warning("Telegram disabled for unknown reason - check .env configuration")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram notifier: {e}", exc_info=True)
            self._notifier = None

    def _load_alert_history(self) -> None:
        """Load alert history from persistent storage."""
        try:
            if self._alert_history_file.exists():
                with open(self._alert_history_file, 'r') as f:
                    data = json.load(f)
                    for key, timestamp_str in data.items():
                        self._sent_alerts[key] = datetime.fromisoformat(timestamp_str)
                logger.info(f"Loaded {len(self._sent_alerts)} alert records from history")
        except Exception as e:
            logger.warning(f"Could not load alert history: {e}")
            self._sent_alerts = {}

    def _save_alert_history(self) -> None:
        """Save alert history to persistent storage."""
        try:
            # Ensure directory exists
            self._alert_history_file.parent.mkdir(parents=True, exist_ok=True)

            # Clean up old alerts before saving
            self._cleanup_old_alerts()

            # Convert datetimes to strings for JSON
            data = {key: ts.isoformat() for key, ts in self._sent_alerts.items()}
            with open(self._alert_history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save alert history: {e}")

    def _cleanup_old_alerts(self) -> None:
        """Remove alerts older than 24 hours from tracking."""
        cutoff = datetime.now() - timedelta(hours=24)
        self._sent_alerts = {
            key: ts for key, ts in self._sent_alerts.items()
            if ts > cutoff
        }

    def _get_alert_key(self, message: str, module: str = "") -> str:
        """Generate a unique key for an alert based on content."""
        # Create a hash of the message content (normalized)
        content = f"{module}:{message}".lower().strip()
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _is_duplicate_alert(self, message: str, module: str = "") -> bool:
        """
        Check if this alert was recently sent.

        Returns True if the same alert was sent within ALERT_COOLDOWN_HOURS.
        """
        key = self._get_alert_key(message, module)

        if key in self._sent_alerts:
            last_sent = self._sent_alerts[key]
            cooldown = timedelta(hours=self.ALERT_COOLDOWN_HOURS)

            if datetime.now() - last_sent < cooldown:
                logger.info(f"Suppressing duplicate alert (sent {datetime.now() - last_sent} ago): {message[:50]}...")
                return True

        return False

    def _record_alert_sent(self, message: str, module: str = "") -> None:
        """Record that an alert was sent."""
        key = self._get_alert_key(message, module)
        self._sent_alerts[key] = datetime.now()
        self._save_alert_history()

    def mark_issue_resolved(self, message: str, module: str = "") -> None:
        """
        Mark an issue as resolved, allowing new alerts for this issue.

        Call this when an issue is fixed to reset the deduplication cooldown.
        """
        key = self._get_alert_key(message, module)
        if key in self._sent_alerts:
            del self._sent_alerts[key]
            self._save_alert_history()
            logger.info(f"Marked issue as resolved: {message[:50]}...")

    def get_suppressed_alerts_count(self) -> int:
        """Get the count of alerts currently being suppressed."""
        self._cleanup_old_alerts()
        return len(self._sent_alerts)

    def get_alert_status(self) -> Dict[str, Any]:
        """Get status of alert deduplication system."""
        self._cleanup_old_alerts()
        return {
            'active_suppressions': len(self._sent_alerts),
            'cooldown_hours': self.ALERT_COOLDOWN_HOURS,
            'history_file': str(self._alert_history_file),
        }

    def is_available(self) -> bool:
        """Check if Telegram notifications are available."""
        return self.enabled and self._notifier is not None and self._notifier.enabled

    def send_critical_alert(self, message: str, module: str = "",
                            details: Dict = None, force: bool = False) -> Optional[str]:
        """
        Send a critical alert via Telegram.

        Args:
            message: The alert message
            module: Which QA module generated the alert
            details: Additional details
            force: If True, bypass deduplication and send anyway

        Returns:
            Message ID if successful, None otherwise
        """
        if not self.is_available():
            logger.warning(f"Critical alert (not sent): {message}")
            return None

        # Check for duplicate alerts (unless forced)
        if not force and self._is_duplicate_alert(message, module):
            logger.info(f"Skipping duplicate critical alert: {message[:50]}...")
            return None

        formatted_message = self._format_critical_alert(message, module, details)
        result = self._notifier.send_custom_message(formatted_message)

        # Record that we sent this alert
        if result:
            self._record_alert_sent(message, module)

        return result

    def send_qa_cycle_summary(self, summary: Dict[str, Any]) -> Optional[str]:
        """
        Send QA cycle completion summary.

        Args:
            summary: Dict with keys:
                - run_id
                - duration_seconds
                - checks_performed
                - issues_found
                - issues_fixed
                - health_score
                - critical_failures

        Returns:
            Message ID if successful, None otherwise
        """
        if not self.is_available():
            return None

        formatted_message = self._format_cycle_summary(summary)
        return self._notifier.send_custom_message(formatted_message)

    def send_daily_summary(self, accomplishments: List[Dict],
                           health_trend: Dict = None) -> Optional[str]:
        """
        Send daily enhancement summary.

        Args:
            accomplishments: List of recent accomplishments
            health_trend: Health score trend data

        Returns:
            Message ID if successful, None otherwise
        """
        if not self.is_available():
            return None

        formatted_message = self._format_daily_summary(accomplishments, health_trend)
        return self._notifier.send_custom_message(formatted_message)

    def send_health_warning(self, current_score: float, previous_score: float,
                            threshold: float = 70.0, force: bool = False) -> Optional[str]:
        """
        Send warning if health score drops below threshold.

        Args:
            current_score: Current health score
            previous_score: Previous health score
            threshold: Warning threshold
            force: If True, bypass deduplication

        Returns:
            Message ID if successful, None otherwise
        """
        if not self.is_available():
            return None

        if current_score >= threshold:
            return None  # No warning needed

        # Create a message key based on score range (within 5 points = same alert)
        score_bucket = int(current_score / 5) * 5
        message = f"Health score below threshold: {score_bucket}-{score_bucket+5}"

        # Check for duplicate (unless forced)
        if not force and self._is_duplicate_alert(message, "health_warning"):
            logger.info(f"Skipping duplicate health warning for score {current_score}")
            return None

        formatted_message = self._format_health_warning(
            current_score, previous_score, threshold
        )
        result = self._notifier.send_custom_message(formatted_message)

        if result:
            self._record_alert_sent(message, "health_warning")

        return result

    def send_enhancement_notification(self, enhancement: Dict) -> Optional[str]:
        """
        Send notification about a proactive enhancement.

        Args:
            enhancement: Dict with keys:
                - message
                - files
                - impact_score
                - category

        Returns:
            Message ID if successful, None otherwise
        """
        if not self.is_available():
            return None

        formatted_message = self._format_enhancement(enhancement)
        return self._notifier.send_custom_message(formatted_message)

    # =========================================================================
    # Message Formatting
    # =========================================================================

    def _escape_markdown(self, text: str) -> str:
        """Escape special Markdown characters to prevent parse errors."""
        if not text:
            return text
        # Escape characters that break Telegram Markdown: _ * ` [ ]
        for char in ['_', '*', '`', '[', ']']:
            text = text.replace(char, '\\' + char)
        return text

    def _format_critical_alert(self, message: str, module: str,
                               details: Dict = None) -> str:
        """Format a critical alert message."""
        # Escape user-provided content to prevent Markdown parse errors
        safe_message = self._escape_markdown(message)
        safe_module = self._escape_markdown(module) if module else ""

        text = (
            f"\U0001F6A8 *MAGNUS QA CRITICAL ALERT* \U0001F6A8\n\n"
            f"\U0001F4A5 *Issue:* {safe_message}\n"
        )

        if module:
            text += f"\U0001F4E6 *Module:* {safe_module}\n"

        text += f"\U0001F553 *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        if details:
            text += "\n\U0001F4CB *Details:*\n"
            for key, value in details.items():
                safe_key = self._escape_markdown(str(key))
                safe_value = self._escape_markdown(str(value))
                text += f"  - {safe_key}: {safe_value}\n"

        text += "\n_Immediate attention required!_"

        return text

    def _format_cycle_summary(self, summary: Dict[str, Any]) -> str:
        """Format a QA cycle summary message."""
        run_id = self._escape_markdown(str(summary.get('run_id', 'Unknown')))
        duration = summary.get('duration_seconds', 0)
        checks = summary.get('checks_performed', 0)
        found = summary.get('issues_found', 0)
        fixed = summary.get('issues_fixed', 0)
        health = summary.get('health_score', 0)
        critical = summary.get('critical_failures', 0)

        # Choose emoji based on results
        if critical > 0:
            status_emoji = "\U0001F534"  # Red
            status_text = "CRITICAL ISSUES"
        elif found > fixed:
            status_emoji = "\U0001F7E1"  # Yellow
            status_text = "NEEDS ATTENTION"
        else:
            status_emoji = "\U0001F7E2"  # Green
            status_text = "HEALTHY"

        text = (
            f"{status_emoji} *Magnus QA Cycle Complete*\n\n"
            f"Run: {run_id}\n"
            f"Status: {status_text}\n\n"
            f"*Results:*\n"
            f"  - Checks: {checks}\n"
            f"  - Issues Found: {found}\n"
            f"  - Issues Fixed: {fixed}\n"
            f"  - Critical: {critical}\n\n"
            f"Health Score: {health:.1f}/100\n"
            f"Duration: {duration:.1f}s\n"
        )

        return text

    def _format_daily_summary(self, accomplishments: List[Dict],
                              health_trend: Dict = None) -> str:
        """Format a daily summary message."""
        text = (
            f"\U0001F4CA *Magnus Daily QA Summary*\n"
            f"{datetime.now().strftime('%Y-%m-%d')}\n\n"
        )

        # Count by category
        by_category = {}
        for a in accomplishments:
            cat = a.get('category', 'other')
            by_category[cat] = by_category.get(cat, 0) + 1

        text += "*Activity:*\n"
        category_emojis = {
            'auto_fix': '\U0001F527',
            'enhancement': '\U0001F680',
            'issue_found': '\U0001F50D',
            'learning': '\U0001F4D6',
        }

        for cat, count in sorted(by_category.items()):
            emoji = category_emojis.get(cat, '-')
            safe_cat = self._escape_markdown(cat.replace('_', ' ').title())
            text += f"  {emoji} {safe_cat}: {count}\n"

        # Recent accomplishments
        recent = accomplishments[-5:]  # Last 5
        if recent:
            text += "\n*Recent Accomplishments:*\n"
            for a in recent:
                safe_msg = self._escape_markdown(str(a.get('message', 'N/A')))[:100]
                text += f"  - {safe_msg}\n"

        # Health trend
        if health_trend:
            current = health_trend.get('current', 0)
            previous = health_trend.get('previous', 0)
            trend = health_trend.get('trend', 'stable')

            trend_emoji = {
                'improving': '\U0001F4C8',
                'stable': '\U0001F4CA',
                'declining': '\U0001F4C9',
            }.get(trend, '\U0001F4CA')

            text += f"\nHealth: {current:.1f} {trend_emoji} (was {previous:.1f})\n"

        return text

    def _format_health_warning(self, current: float, previous: float,
                               threshold: float) -> str:
        """Format a health warning message."""
        drop = previous - current

        text = (
            f"\u26A0\uFE0F *Magnus Health Warning*\n\n"
            f"\U0001F4C9 Health score dropped below threshold!\n\n"
            f"  \u2022 Current: `{current:.1f}/100`\n"
            f"  \u2022 Previous: `{previous:.1f}/100`\n"
            f"  \u2022 Drop: `{drop:.1f}` points\n"
            f"  \u2022 Threshold: `{threshold:.1f}`\n\n"
            f"_Review recent changes and QA logs._"
        )

        return text

    def _format_enhancement(self, enhancement: Dict) -> str:
        """Format an enhancement notification."""
        message = enhancement.get('message', 'Enhancement applied')
        files = enhancement.get('files', [])
        impact = enhancement.get('impact_score', 0)
        category = enhancement.get('category', 'enhancement')

        text = (
            f"\U0001F680 *Magnus Enhancement Applied*\n\n"
            f"\U0001F4DD *Action:* {message}\n"
            f"\U0001F3AF *Impact:* `{impact}/10`\n"
        )

        if files:
            text += f"\U0001F4C1 *Files:*\n"
            for f in files[:5]:  # Max 5 files
                text += f"  \u2022 `{f}`\n"
            if len(files) > 5:
                text += f"  _...and {len(files) - 5} more_\n"

        return text


# Singleton instance
_qa_notifier_instance: Optional[QATelegramNotifier] = None


def get_qa_notifier() -> QATelegramNotifier:
    """Get the singleton QA notifier instance."""
    global _qa_notifier_instance
    if _qa_notifier_instance is None:
        _qa_notifier_instance = QATelegramNotifier()
    return _qa_notifier_instance


def send_critical_alert(message: str, module: str = "",
                        details: Dict = None) -> Optional[str]:
    """Convenience function to send critical alert."""
    return get_qa_notifier().send_critical_alert(message, module, details)


def send_qa_summary(summary: Dict[str, Any]) -> Optional[str]:
    """Convenience function to send QA cycle summary."""
    return get_qa_notifier().send_qa_cycle_summary(summary)
