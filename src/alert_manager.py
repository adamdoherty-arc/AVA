"""
Alert Manager for Supply/Demand Zones
Sends Telegram notifications for zone events
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from zone_database_manager import ZoneDatabaseManager

# Load environment
load_dotenv()

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    logger.warning("python-telegram-bot not installed. Alerts will be logged only.")
    TELEGRAM_AVAILABLE = False


class AlertManager:
    """
    Manages alerts for supply/demand zone events

    Sends Telegram notifications when:
    - Price enters demand zone (buying opportunity)
    - Price enters supply zone (selling opportunity)
    - Zone is tested (bounce or break)
    - New high-quality zones are detected
    """

    def __init__(
        self,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        db_manager: Optional[ZoneDatabaseManager] = None,
        enable_telegram: bool = True
    ):
        """
        Initialize alert manager

        Args:
            telegram_bot_token: Telegram bot token (from BotFather)
            telegram_chat_id: Chat ID to send alerts to
            db_manager: Database manager instance
            enable_telegram: Enable Telegram notifications
        """

        self.db = db_manager or ZoneDatabaseManager()

        # Telegram configuration
        self.telegram_enabled = enable_telegram and TELEGRAM_AVAILABLE

        if self.telegram_enabled:
            self.bot_token = telegram_bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
            self.chat_id = telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')

            if not self.bot_token or not self.chat_id:
                logger.warning("Telegram credentials not configured. Alerts will be logged only.")
                self.telegram_enabled = False
            else:
                self.bot = Bot(token=self.bot_token)
                logger.info("Telegram alerts enabled")
        else:
            logger.info("Telegram alerts disabled")

    async def send_zone_event_alert(self, event: Dict) -> bool:
        """
        Send alert for zone event

        Args:
            event: Event dictionary from PriceMonitor

        Returns:
            True if alert sent successfully
        """

        try:
            # Format alert message
            message = self._format_zone_event(event)

            # Log alert
            logger.info(f"Alert: {event['event_type']} - {event['symbol']} @ ${event['current_price']:.2f}")

            # Send to Telegram if enabled
            if self.telegram_enabled:
                telegram_message_id = await self._send_telegram_message(message, event['priority'])
            else:
                telegram_message_id = None
                print(f"\n{message}\n")  # Console output if Telegram disabled

            # Save alert to database
            alert_data = {
                'zone_id': event['zone_id'],
                'ticker': event['symbol'],
                'alert_type': event['event_type'],
                'alert_price': event['current_price'],
                'zone_type': event['zone_type'],
                'distance_to_zone': event.get('distance_pct'),
                'zone_strength': event.get('strength_score'),
                'setup_quality': self._get_setup_quality(event),
                'telegram_message_id': telegram_message_id,
                'status': 'sent'
            }

            self.db.save_alert(alert_data)

            return True

        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False

    def send_zone_event_alert_sync(self, event: Dict) -> bool:
        """Synchronous version of send_zone_event_alert"""
        return asyncio.run(self.send_zone_event_alert(event))

    async def send_batch_alerts(self, events: List[Dict]) -> int:
        """
        Send multiple alerts in batch

        Args:
            events: List of event dictionaries

        Returns:
            Number of alerts sent successfully
        """

        sent_count = 0

        for event in events:
            success = await self.send_zone_event_alert(event)
            if success:
                sent_count += 1

        return sent_count

    def send_batch_alerts_sync(self, events: List[Dict]) -> int:
        """Synchronous version of send_batch_alerts"""
        return asyncio.run(self.send_batch_alerts(events))

    async def send_daily_summary(self, summary_data: Dict):
        """
        Send daily summary of zones and opportunities

        Args:
            summary_data: Dictionary with summary statistics
        """

        message = self._format_daily_summary(summary_data)

        if self.telegram_enabled:
            await self._send_telegram_message(message, priority='LOW')
        else:
            print(f"\n{message}\n")

    def _format_zone_event(self, event: Dict) -> str:
        """
        Format zone event as Telegram message

        Args:
            event: Event dictionary

        Returns:
            Formatted message string
        """

        symbol = event['symbol']
        event_type = event['event_type']
        current_price = event['current_price']
        zone_type = event['zone_type']
        zone_bottom = event['zone_bottom']
        zone_top = event['zone_top']
        strength = event.get('strength_score', 50)
        status = event.get('status', 'FRESH')
        priority = event['priority']

        # Emoji based on event type
        if event_type == 'PRICE_AT_DEMAND':
            emoji = '🟢'
            title = 'BUY OPPORTUNITY'
            description = f'Price INSIDE demand zone'
        elif event_type == 'PRICE_ENTERING_DEMAND':
            emoji = '⚡'
            title = 'APPROACHING DEMAND'
            description = f'Price approaching demand zone'
        elif event_type == 'PRICE_AT_SUPPLY':
            emoji = '🔴'
            title = 'SELL OPPORTUNITY'
            description = f'Price INSIDE supply zone'
        elif event_type == 'PRICE_ENTERING_SUPPLY':
            emoji = '⚠️'
            title = 'APPROACHING SUPPLY'
            description = f'Price approaching supply zone'
        elif event_type == 'ZONE_BOUNCE':
            emoji = '💹'
            title = 'ZONE BOUNCE'
            description = f'Price bounced from {zone_type.lower()} zone'
        elif event_type == 'ZONE_BREAK':
            emoji = '❌'
            title = 'ZONE BROKEN'
            description = f'{zone_type} zone broken'
        else:
            emoji = 'ℹ️'
            title = event_type
            description = ''

        # Build message
        message = f"""
{emoji} **{title}**

**{symbol}** @ **${current_price:.2f}**

{description}

**Zone Details:**
• Type: {zone_type}
• Range: ${zone_bottom:.2f} - ${zone_top:.2f}
• Strength: {strength}/100
• Status: {status}
• Priority: {priority}

**Action:**
"""

        # Add recommended action
        if event_type == 'PRICE_AT_DEMAND':
            message += f"✅ Consider BUYING at ${zone_bottom:.2f}-${zone_top:.2f}\n"
            message += f"🎯 Target: ${zone_top * 1.05:.2f} (+5%)\n"
            message += f"🛑 Stop: ${zone_bottom * 0.98:.2f} (-2%)"
        elif event_type == 'PRICE_AT_SUPPLY':
            message += f"✅ Consider SELLING/SHORTING at ${zone_bottom:.2f}-${zone_top:.2f}\n"
            message += f"🎯 Target: ${zone_bottom * 0.95:.2f} (-5%)\n"
            message += f"🛑 Stop: ${zone_top * 1.02:.2f} (+2%)"
        elif event_type == 'PRICE_ENTERING_DEMAND':
            message += f"⏳ WATCH for entry at ${zone_bottom:.2f}"
        elif event_type == 'PRICE_ENTERING_SUPPLY':
            message += f"⏳ WATCH for entry at ${zone_top:.2f}"
        elif event_type == 'ZONE_BREAK':
            message += f"⚠️ Zone invalidated - avoid trading this zone"

        # Add timestamp
        timestamp = event.get('timestamp', datetime.now())
        message += f"\n\n🕐 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

        return message

    def _format_daily_summary(self, summary_data: Dict) -> str:
        """Format daily summary message"""

        message = """
📊 **Daily Supply/Demand Zone Summary**

**Active Zones:**
"""

        message += f"• Total: {summary_data.get('total_zones', 0)}\n"
        message += f"• Fresh: {summary_data.get('fresh_zones', 0)}\n"
        message += f"• Demand: {summary_data.get('demand_zones', 0)}\n"
        message += f"• Supply: {summary_data.get('supply_zones', 0)}\n\n"

        message += f"**Today's Activity:**\n"
        message += f"• Alerts Sent: {summary_data.get('alerts_sent', 0)}\n"
        message += f"• Zones Detected: {summary_data.get('zones_detected', 0)}\n"
        message += f"• Zones Tested: {summary_data.get('zones_tested', 0)}\n\n"

        # Top opportunities
        top_opportunities = summary_data.get('top_opportunities', [])
        if top_opportunities:
            message += f"**Top Opportunities:**\n"
            for opp in top_opportunities[:5]:
                message += f"• {opp['symbol']}: {opp['zone_type']} @ ${opp['zone_midpoint']:.2f} (Strength: {opp['strength_score']}/100)\n"

        return message

    async def _send_telegram_message(self, message: str, priority: str = 'MEDIUM') -> Optional[str]:
        """
        Send message to Telegram

        Args:
            message: Message text
            priority: Alert priority (HIGH, MEDIUM, LOW)

        Returns:
            Message ID if sent successfully
        """

        if not self.telegram_enabled:
            return None

        try:
            # Add priority indicator
            if priority == 'HIGH':
                message = f"🔔 **URGENT**\n\n{message}"

            # Send message
            sent_message = await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )

            return str(sent_message.message_id)

        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return None

    def _get_setup_quality(self, event: Dict) -> str:
        """
        Determine setup quality from event

        Returns:
            'HIGH', 'MEDIUM', or 'LOW'
        """

        strength = event.get('strength_score', 50)
        status = event.get('status', 'TESTED')
        priority = event.get('priority', 'MEDIUM')

        if priority == 'HIGH' and strength >= 75 and status == 'FRESH':
            return 'HIGH'
        elif priority == 'HIGH' or (strength >= 60 and status in ['FRESH', 'TESTED']):
            return 'MEDIUM'
        else:
            return 'LOW'

    async def send_test_alert(self) -> None:
        """Send test alert to verify Telegram connection"""

        test_message = """
🧪 **Test Alert**

This is a test message from the Supply/Demand Zone Alert System.

If you're seeing this, Telegram alerts are working correctly!

✅ System Status: Operational
🕐 {timestamp}
"""f-string.strftime('%Y-%m-%d %H:%M:%S'))

        if self.telegram_enabled:
            await self._send_telegram_message(test_message, priority='LOW')
            logger.info("Test alert sent to Telegram")
        else:
            print(f"\n{test_message}\n")
            logger.info("Test alert displayed (Telegram disabled)")

    def send_test_alert_sync(self) -> None:
        """Synchronous version of send_test_alert"""
        asyncio.run(self.send_test_alert())


# Convenience function for quick alerts
def send_quick_alert(symbol: str, message: str):
    """
    Send a quick alert without full event structure

    Args:
        symbol: Stock ticker
        message: Alert message
    """

    alert_manager = AlertManager()

    quick_message = f"""
💬 **Quick Alert: {symbol}**

{message}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    if alert_manager.telegram_enabled:
        asyncio.run(alert_manager._send_telegram_message(quick_message))
    else:
        print(f"\n{quick_message}\n")


if __name__ == "__main__":
    # Test alert manager
    logging.basicConfig(level=logging.INFO)

    print("Testing AlertManager...")

    # Check Telegram configuration
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if bot_token and chat_id and bot_token != "YOUR_BOT_TOKEN_HERE":
        print(f"✅ Telegram configured (Chat ID: {chat_id[:10]}...)")

        alert_manager = AlertManager()

        # Send test alert
        print("\nSending test alert...")
        alert_manager.send_test_alert_sync()
        print("✅ Test alert sent")

    else:
        print("❌ Telegram not configured")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")

        # Test without Telegram (console output)
        alert_manager = AlertManager(enable_telegram=False)

        # Sample event
        sample_event = {
            'zone_id': 1,
            'symbol': 'AAPL',
            'zone_type': 'DEMAND',
            'event_type': 'PRICE_AT_DEMAND',
            'current_price': 178.50,
            'zone_top': 180.50,
            'zone_bottom': 178.00,
            'zone_midpoint': 179.25,
            'strength_score': 85,
            'status': 'FRESH',
            'priority': 'HIGH',
            'distance_pct': 0.0,
            'timestamp': datetime.now()
        }

        print("\nSample alert (console output):")
        alert_manager.send_zone_event_alert_sync(sample_event)
