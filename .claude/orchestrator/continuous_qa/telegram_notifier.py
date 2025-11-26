"""
Telegram Notifier for QA System

Sends QA alerts and summaries via AVA's Telegram bot.
Wraps the existing TelegramNotifier from src/telegram_notifier.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add src to path to import existing notifier
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from telegram_notifier import TelegramNotifier as BaseTelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
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
    """

    def __init__(self, enabled: bool = True):
        """Initialize the QA Telegram notifier."""
        self.enabled = enabled

        if not TELEGRAM_AVAILABLE:
            logger.warning(
                "Base TelegramNotifier not available. "
                "QA Telegram notifications disabled."
            )
            self._notifier = None
            return

        try:
            self._notifier = BaseTelegramNotifier()
            if self._notifier.enabled:
                logger.info("QA Telegram notifier initialized")
            else:
                logger.info("Telegram notifications disabled in config")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram notifier: {e}")
            self._notifier = None

    def is_available(self) -> bool:
        """Check if Telegram notifications are available."""
        return self.enabled and self._notifier is not None and self._notifier.enabled

    def send_critical_alert(self, message: str, module: str = "",
                            details: Dict = None) -> Optional[str]:
        """
        Send a critical alert via Telegram.

        Args:
            message: The alert message
            module: Which QA module generated the alert
            details: Additional details

        Returns:
            Message ID if successful, None otherwise
        """
        if not self.is_available():
            logger.warning(f"Critical alert (not sent): {message}")
            return None

        formatted_message = self._format_critical_alert(message, module, details)
        return self._notifier.send_custom_message(formatted_message)

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
                            threshold: float = 70.0) -> Optional[str]:
        """
        Send warning if health score drops below threshold.

        Args:
            current_score: Current health score
            previous_score: Previous health score
            threshold: Warning threshold

        Returns:
            Message ID if successful, None otherwise
        """
        if not self.is_available():
            return None

        if current_score >= threshold:
            return None  # No warning needed

        formatted_message = self._format_health_warning(
            current_score, previous_score, threshold
        )
        return self._notifier.send_custom_message(formatted_message)

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

    def _format_critical_alert(self, message: str, module: str,
                               details: Dict = None) -> str:
        """Format a critical alert message."""
        text = (
            f"\U0001F6A8 *MAGNUS QA CRITICAL ALERT* \U0001F6A8\n\n"
            f"\U0001F4A5 *Issue:* {message}\n"
        )

        if module:
            text += f"\U0001F4E6 *Module:* `{module}`\n"

        text += f"\U0001F553 *Time:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"

        if details:
            text += "\n\U0001F4CB *Details:*\n"
            for key, value in details.items():
                text += f"  \u2022 {key}: `{value}`\n"

        text += "\n_Immediate attention required!_"

        return text

    def _format_cycle_summary(self, summary: Dict[str, Any]) -> str:
        """Format a QA cycle summary message."""
        run_id = summary.get('run_id', 'Unknown')
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
            f"\U0001F3F7 *Run:* `{run_id}`\n"
            f"\U0001F4CA *Status:* {status_text}\n\n"
            f"\U0001F4C8 *Results:*\n"
            f"  \u2022 Checks: `{checks}`\n"
            f"  \u2022 Issues Found: `{found}`\n"
            f"  \u2022 Issues Fixed: `{fixed}`\n"
            f"  \u2022 Critical: `{critical}`\n\n"
            f"\U0001F3AF *Health Score:* `{health:.1f}/100`\n"
            f"\U0001F553 *Duration:* `{duration:.1f}s`\n"
        )

        return text

    def _format_daily_summary(self, accomplishments: List[Dict],
                              health_trend: Dict = None) -> str:
        """Format a daily summary message."""
        text = (
            f"\U0001F4CA *Magnus Daily QA Summary*\n"
            f"\U0001F4C5 `{datetime.now().strftime('%Y-%m-%d')}`\n\n"
        )

        # Count by category
        by_category = {}
        for a in accomplishments:
            cat = a.get('category', 'other')
            by_category[cat] = by_category.get(cat, 0) + 1

        text += "\U0001F4CB *Activity:*\n"
        category_emojis = {
            'auto_fix': '\U0001F527',
            'enhancement': '\U0001F680',
            'issue_found': '\U0001F50D',
            'learning': '\U0001F4D6',
        }

        for cat, count in sorted(by_category.items()):
            emoji = category_emojis.get(cat, '\u2022')
            text += f"  {emoji} {cat.replace('_', ' ').title()}: `{count}`\n"

        # Recent accomplishments
        recent = accomplishments[-5:]  # Last 5
        if recent:
            text += "\n\U0001F3C6 *Recent Accomplishments:*\n"
            for a in recent:
                text += f"  \u2713 {a.get('message', 'N/A')}\n"

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

            text += f"\n\U0001F3AF *Health:* `{current:.1f}` {trend_emoji} (was `{previous:.1f}`)\n"

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
