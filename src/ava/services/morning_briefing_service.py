"""
Morning Briefing Service - Daily proactive intelligence delivery

Generates comprehensive morning briefings covering:
- Goal progress and daily targets
- Open positions and risk status
- Earnings calendar for the week
- XTrades trader activity
- Market conditions and opportunities
- Recommended actions for the day
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.database.connection_pool import get_db_connection
from src.ava.services.goal_tracking_service import GoalTrackingService, GoalStatus
from src.services.alert_service import AlertService, AlertCategory, AlertPriority

logger = logging.getLogger(__name__)


@dataclass
class BriefingSection:
    """A section of the morning briefing"""
    title: str
    content: str
    priority: int = 0  # Lower = more important
    emoji: str = ""


@dataclass
class MorningBriefing:
    """Complete morning briefing"""
    generated_at: datetime
    user_id: str
    greeting: str
    sections: List[BriefingSection] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    alerts_count: int = 0

    def to_telegram_message(self) -> str:
        """Format briefing for Telegram delivery"""
        lines = [
            f"**{self.greeting}**",
            f"_Generated {self.generated_at.strftime('%I:%M %p')} EST_",
            ""
        ]

        # Sort sections by priority
        for section in sorted(self.sections, key=lambda s: s.priority):
            emoji = f"{section.emoji} " if section.emoji else ""
            lines.append(f"**{emoji}{section.title}**")
            lines.append(section.content)
            lines.append("")

        if self.action_items:
            lines.append("**Action Items:**")
            for i, item in enumerate(self.action_items[:5], 1):
                lines.append(f"{i}. {item}")

        return "\n".join(lines)

    def to_email_html(self) -> str:
        """Format briefing for email delivery"""
        html = [
            "<html><body style='font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;'>",
            f"<h1 style='color: #2563eb;'>{self.greeting}</h1>",
            f"<p style='color: #6b7280; font-size: 12px;'>Generated {self.generated_at.strftime('%I:%M %p')} EST</p>",
            "<hr>"
        ]

        for section in sorted(self.sections, key=lambda s: s.priority):
            emoji = f"{section.emoji} " if section.emoji else ""
            html.append(f"<h2 style='color: #1f2937;'>{emoji}{section.title}</h2>")
            html.append(f"<p style='white-space: pre-line;'>{section.content}</p>")

        if self.action_items:
            html.append("<h2 style='color: #dc2626;'>Action Items</h2>")
            html.append("<ol>")
            for item in self.action_items[:5]:
                html.append(f"<li>{item}</li>")
            html.append("</ol>")

        html.append("</body></html>")
        return "\n".join(html)


class MorningBriefingService:
    """
    Service for generating and delivering daily morning briefings.

    The briefing aggregates intelligence from multiple sources to give
    a comprehensive view of the trading day ahead.
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.goal_service = GoalTrackingService(user_id)
        self.alert_service = AlertService(user_id)

    async def generate_briefing(self) -> MorningBriefing:
        """Generate a complete morning briefing"""
        now = datetime.now()

        # Determine greeting based on time
        hour = now.hour
        if hour < 12:
            greeting = "Good Morning! Here's Your Trading Briefing"
        elif hour < 17:
            greeting = "Good Afternoon! Here's Your Market Update"
        else:
            greeting = "Good Evening! Here's Your End-of-Day Summary"

        briefing = MorningBriefing(
            generated_at=now,
            user_id=self.user_id,
            greeting=greeting
        )

        # Gather all sections concurrently
        sections = []
        action_items = []

        # 1. Goal Progress Section
        goal_section, goal_actions = self._build_goal_section()
        if goal_section:
            sections.append(goal_section)
            action_items.extend(goal_actions)

        # 2. Open Positions Section
        positions_section, pos_actions = await self._build_positions_section()
        if positions_section:
            sections.append(positions_section)
            action_items.extend(pos_actions)

        # 3. Earnings Calendar Section
        earnings_section, earn_actions = self._build_earnings_section()
        if earnings_section:
            sections.append(earnings_section)
            action_items.extend(earn_actions)

        # 4. XTrades Activity Section
        xtrades_section, xt_actions = self._build_xtrades_section()
        if xtrades_section:
            sections.append(xtrades_section)
            action_items.extend(xt_actions)

        # 5. Opportunities Section
        opps_section, opp_actions = self._build_opportunities_section()
        if opps_section:
            sections.append(opps_section)
            action_items.extend(opp_actions)

        # 6. Alerts Summary
        alerts_section = self._build_alerts_section()
        if alerts_section:
            sections.append(alerts_section)

        briefing.sections = sections
        briefing.action_items = action_items[:10]  # Top 10 action items

        return briefing

    def _build_goal_section(self) -> tuple[Optional[BriefingSection], List[str]]:
        """Build the goal progress section"""
        actions = []

        try:
            goal = self.goal_service.get_monthly_income_goal()

            if not goal:
                return None, []

            status_emoji = {
                GoalStatus.ACHIEVED: "🎉",
                GoalStatus.AHEAD: "📈",
                GoalStatus.ON_TRACK: "✅",
                GoalStatus.BEHIND: "⚠️",
                GoalStatus.AT_RISK: "🚨"
            }

            lines = [
                f"Target: ${goal.target_amount:,.0f}/month",
                f"Current: ${goal.current_amount:,.0f} ({goal.progress_percent:.1f}%)",
                f"Status: {status_emoji.get(goal.status, '')} {goal.status.value.replace('_', ' ').title()}",
                f"Days remaining: {goal.days_remaining}"
            ]

            if goal.daily_target > 0:
                lines.append(f"Daily target: ${goal.daily_target:,.0f}")
                lines.append(f"Your daily avg: ${goal.daily_actual:,.0f}")

            if goal.status in [GoalStatus.BEHIND, GoalStatus.AT_RISK]:
                remaining = goal.target_amount - goal.current_amount
                lines.append(f"\nNeed ${remaining:,.0f} more to hit target")
                if goal.opportunities_needed > 0:
                    lines.append(f"≈ {goal.opportunities_needed} trades at ${goal.avg_premium_needed:,.0f} avg")

                actions.append(f"Find {goal.opportunities_needed} premium opportunities to get back on track")

            return BriefingSection(
                title="Monthly Income Goal",
                content="\n".join(lines),
                priority=0,
                emoji="🎯"
            ), actions

        except Exception as e:
            logger.error(f"Error building goal section: {e}")
            return None, []

    async def _build_positions_section(self) -> tuple[Optional[BriefingSection], List[str]]:
        """Build the open positions section"""
        actions = []

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get open positions summary
                cursor.execute("""
                    SELECT
                        symbol,
                        strike_price,
                        expiration_date,
                        strategy_type,
                        premium_collected,
                        contracts,
                        (expiration_date - CURRENT_DATE) as dte
                    FROM trade_history
                    WHERE status = 'open'
                    ORDER BY expiration_date ASC
                    LIMIT 10
                """)

                rows = cursor.fetchall()

                if not rows:
                    return BriefingSection(
                        title="Open Positions",
                        content="No open positions. Time to find opportunities!",
                        priority=1,
                        emoji="📊"
                    ), ["Review scanner for new CSP opportunities"]

                lines = []
                urgent_positions = []

                for row in rows:
                    symbol = row[0]
                    strike = float(row[1]) if row[1] else 0
                    exp_date = row[2]
                    strategy = row[3] or 'option'
                    premium = float(row[4]) if row[4] else 0
                    contracts = row[5] or 1
                    dte = row[6] if row[6] else 0

                    # Format expiration
                    if isinstance(exp_date, str):
                        exp_str = exp_date
                    else:
                        exp_str = exp_date.strftime('%m/%d') if exp_date else 'N/A'

                    line = f"• {symbol} ${strike:.0f} {strategy[:3].upper()} - {exp_str} ({dte}d)"
                    lines.append(line)

                    # Flag urgent positions
                    if dte is not None and dte <= 3:
                        urgent_positions.append(f"{symbol} expires in {dte} days - decide: close, roll, or let expire")

                summary = f"{len(rows)} open position(s)\n\n" + "\n".join(lines)

                if urgent_positions:
                    actions.extend(urgent_positions)

                return BriefingSection(
                    title="Open Positions",
                    content=summary,
                    priority=1,
                    emoji="📊"
                ), actions

        except Exception as e:
            logger.error(f"Error building positions section: {e}")
            return None, []

    def _build_earnings_section(self) -> tuple[Optional[BriefingSection], List[str]]:
        """Build the earnings calendar section"""
        actions = []

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get upcoming earnings for held positions or watchlist
                cursor.execute("""
                    SELECT DISTINCT
                        e.symbol,
                        e.report_date,
                        e.time_of_day,
                        th.id as has_position
                    FROM earnings_calendar e
                    LEFT JOIN trade_history th ON th.symbol = e.symbol AND th.status = 'open'
                    WHERE e.report_date >= CURRENT_DATE
                      AND e.report_date <= CURRENT_DATE + INTERVAL '7 days'
                    ORDER BY e.report_date ASC
                    LIMIT 10
                """)

                rows = cursor.fetchall()

                if not rows:
                    return None, []

                lines = []
                position_earnings = []

                for row in rows:
                    symbol = row[0]
                    report_date = row[1]
                    time_of_day = row[2] or 'TBD'
                    has_position = row[3] is not None

                    if isinstance(report_date, str):
                        date_str = report_date
                    else:
                        date_str = report_date.strftime('%m/%d') if report_date else 'TBD'

                    position_indicator = "⚠️ " if has_position else ""
                    lines.append(f"{position_indicator}{symbol} - {date_str} ({time_of_day})")

                    if has_position:
                        position_earnings.append(f"Review {symbol} position before earnings on {date_str}")

                summary = "\n".join(lines)

                if position_earnings:
                    actions.extend(position_earnings)

                return BriefingSection(
                    title="Earnings This Week",
                    content=summary,
                    priority=2,
                    emoji="📅"
                ), actions

        except Exception as e:
            logger.error(f"Error building earnings section: {e}")
            return None, []

    def _build_xtrades_section(self) -> tuple[Optional[BriefingSection], List[str]]:
        """Build the XTrades activity section"""
        actions = []

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get recent XTrades alerts (last 24 hours)
                cursor.execute("""
                    SELECT
                        symbol,
                        action,
                        trader_name,
                        created_at,
                        strike_price,
                        expiration_date
                    FROM xtrades_alerts
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC
                    LIMIT 5
                """)

                rows = cursor.fetchall()

                if not rows:
                    return None, []

                lines = []
                for row in rows:
                    symbol = row[0]
                    action = row[1] or 'trade'
                    trader = row[2] or 'Unknown'
                    strike = float(row[4]) if row[4] else None

                    strike_info = f" ${strike:.0f}" if strike else ""
                    lines.append(f"• {trader}: {action.upper()} {symbol}{strike_info}")

                    # If top trader opened a position we don't have, suggest reviewing
                    if 'open' in action.lower() or 'buy' in action.lower():
                        actions.append(f"Review {symbol} - {trader} opened position")

                summary = f"Last 24 hours:\n" + "\n".join(lines)

                return BriefingSection(
                    title="XTrades Activity",
                    content=summary,
                    priority=3,
                    emoji="👥"
                ), actions[:2]  # Limit to 2 action items from XTrades

        except Exception as e:
            logger.error(f"Error building XTrades section: {e}")
            return None, []

    def _build_opportunities_section(self) -> tuple[Optional[BriefingSection], List[str]]:
        """Build the opportunities section from scanner results"""
        actions = []

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get recent high-quality scanner results
                cursor.execute("""
                    SELECT
                        symbol,
                        strike,
                        expiration_date,
                        premium,
                        annual_return,
                        delta,
                        scan_type
                    FROM scanner_results
                    WHERE scanned_at >= CURRENT_DATE
                      AND annual_return >= 20
                    ORDER BY annual_return DESC
                    LIMIT 5
                """)

                rows = cursor.fetchall()

                if not rows:
                    return BriefingSection(
                        title="Today's Opportunities",
                        content="No high-yield opportunities found yet. Run the scanner!",
                        priority=4,
                        emoji="🔍"
                    ), ["Run premium scanner to find today's opportunities"]

                lines = []
                for row in rows:
                    symbol = row[0]
                    strike = float(row[1]) if row[1] else 0
                    exp_date = row[2]
                    premium = float(row[3]) if row[3] else 0
                    annual_return = float(row[4]) if row[4] else 0
                    delta = float(row[5]) if row[5] else 0
                    scan_type = row[6] or 'CSP'

                    if isinstance(exp_date, str):
                        exp_str = exp_date
                    else:
                        exp_str = exp_date.strftime('%m/%d') if exp_date else 'N/A'

                    lines.append(
                        f"• {symbol} ${strike:.0f} {scan_type} {exp_str} - "
                        f"${premium:.0f} ({annual_return:.0f}% ann, δ{delta:.2f})"
                    )

                summary = "\n".join(lines)
                actions.append(f"Review top opportunity: {rows[0][0]} at ${float(rows[0][1]):.0f}")

                return BriefingSection(
                    title="Today's Opportunities",
                    content=summary,
                    priority=4,
                    emoji="💰"
                ), actions

        except Exception as e:
            logger.error(f"Error building opportunities section: {e}")
            return None, []

    def _build_alerts_section(self) -> Optional[BriefingSection]:
        """Build unread alerts summary"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Count unread alerts by category
                cursor.execute("""
                    SELECT category, COUNT(*)
                    FROM ava_alerts
                    WHERE user_id = %s
                      AND created_at >= CURRENT_DATE
                      AND (metadata->>'read' IS NULL OR metadata->>'read' = 'false')
                    GROUP BY category
                """, (self.user_id,))

                rows = cursor.fetchall()

                if not rows:
                    return None

                lines = []
                total = 0
                for row in rows:
                    category = row[0]
                    count = row[1]
                    total += count
                    lines.append(f"• {category}: {count}")

                summary = f"{total} unread alert(s):\n" + "\n".join(lines)

                return BriefingSection(
                    title="Pending Alerts",
                    content=summary,
                    priority=5,
                    emoji="🔔"
                )

        except Exception as e:
            logger.error(f"Error building alerts section: {e}")
            return None

    async def send_briefing(
        self,
        via_telegram: bool = True,
        via_email: bool = True
    ) -> bool:
        """Generate and send the morning briefing"""
        try:
            briefing = await self.generate_briefing()

            # Send via alert service
            if via_telegram or via_email:
                telegram_msg = briefing.to_telegram_message()

                self.alert_service.create_alert(
                    category=AlertCategory.REPORT,
                    priority=AlertPriority.NORMAL,
                    title="Morning Briefing",
                    message=telegram_msg,
                    metadata={
                        'briefing_type': 'morning',
                        'sections_count': len(briefing.sections),
                        'action_items_count': len(briefing.action_items)
                    }
                )

            # Store briefing in database
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO ava_generated_reports (
                        user_id, report_type, title, content, generated_at
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    self.user_id,
                    'morning_briefing',
                    briefing.greeting,
                    briefing.to_telegram_message(),
                    briefing.generated_at
                ))

                conn.commit()

            logger.info(f"Morning briefing sent for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"Error sending morning briefing: {e}")
            return False


# Convenience function for scheduled task
async def send_morning_briefing(user_id: str = "default_user") -> bool:
    """Quick helper to send morning briefing"""
    service = MorningBriefingService(user_id)
    return await service.send_briefing()


# Convenience function to get briefing without sending
async def get_morning_briefing(user_id: str = "default_user") -> MorningBriefing:
    """Quick helper to generate briefing for display"""
    service = MorningBriefingService(user_id)
    return await service.generate_briefing()
