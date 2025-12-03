"""
Goal Tracking Service - Track progress toward passive income goals

Provides comprehensive tracking of the $2,500/month target and adaptive
advice based on performance patterns, current positions, and market conditions.
"""
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)


class GoalStatus(str, Enum):
    """Goal progress status levels"""
    ON_TRACK = "on_track"
    AHEAD = "ahead"
    BEHIND = "behind"
    AT_RISK = "at_risk"
    ACHIEVED = "achieved"


@dataclass
class GoalProgress:
    """Represents current goal progress and projections"""
    goal_id: int
    goal_name: str
    goal_type: str
    target_amount: float
    current_amount: float
    period_start: datetime
    period_end: datetime

    # Calculated fields
    progress_percent: float = 0.0
    days_elapsed: int = 0
    days_remaining: int = 0
    daily_target: float = 0.0
    daily_actual: float = 0.0
    projected_amount: float = 0.0
    status: GoalStatus = GoalStatus.ON_TRACK

    # Breakdown
    premium_income: float = 0.0
    realized_gains: float = 0.0
    unrealized_gains: float = 0.0

    # Advice
    advice: List[str] = field(default_factory=list)
    opportunities_needed: int = 0
    avg_premium_needed: float = 0.0


@dataclass
class IncomeBreakdown:
    """Breakdown of income sources"""
    csp_premium: float = 0.0
    cc_premium: float = 0.0
    other_premium: float = 0.0
    realized_gains: float = 0.0
    dividend_income: float = 0.0
    total: float = 0.0

    by_symbol: Dict[str, float] = field(default_factory=dict)
    by_week: Dict[str, float] = field(default_factory=dict)


class GoalTrackingService:
    """
    Service for tracking income goals and providing adaptive advice.

    Key capabilities:
    - Track progress toward monthly/weekly/yearly income targets
    - Calculate income from multiple sources (premium, realized gains)
    - Project month-end income based on current trajectory
    - Provide actionable advice to stay on track
    - Identify gaps and recommend specific actions
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id

    def get_active_goals(self) -> List[Dict[str, Any]]:
        """Get all active goals for the user"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, user_id, goal_name, goal_type, target_value,
                           current_value, end_date, period_type,
                           start_date, status,
                           created_at, updated_at
                    FROM ava_user_goals
                    WHERE user_id = %s AND status = 'active'
                    ORDER BY end_date ASC NULLS LAST
                """, (self.user_id,))

                rows = cursor.fetchall()

                goals = []
                for row in rows:
                    goals.append({
                        'id': row[0],
                        'user_id': row[1],
                        'goal_name': row[2],
                        'goal_type': row[3],
                        'target_amount': float(row[4]) if row[4] else 0,
                        'current_amount': float(row[5]) if row[5] else 0,
                        'target_date': row[6],
                        'period_type': row[7],
                        'period_start': row[8],
                        'status': row[9],
                        'created_at': row[10],
                        'updated_at': row[11]
                    })

                return goals

        except Exception as e:
            logger.error(f"Error getting active goals: {e}")
            return []

    def get_monthly_income_goal(self) -> Optional[GoalProgress]:
        """Get the primary monthly income goal with full progress tracking"""
        try:
            now = datetime.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # Get next month for period end
            if now.month == 12:
                month_end = now.replace(year=now.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = now.replace(month=now.month + 1, day=1) - timedelta(days=1)
            month_end = month_end.replace(hour=23, minute=59, second=59)

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get or create monthly income goal
                cursor.execute("""
                    SELECT id, goal_name, goal_type, target_value, current_value,
                           start_date, end_date
                    FROM ava_user_goals
                    WHERE user_id = %s
                      AND goal_type = 'monthly_income'
                      AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (self.user_id,))

                row = cursor.fetchone()

                if not row:
                    # Create default $2,500/month goal
                    cursor.execute("""
                        INSERT INTO ava_user_goals (
                            user_id, goal_name, goal_type, target_value,
                            target_unit, period_type, start_date, end_date, status
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active')
                        RETURNING id, goal_name, goal_type, target_value, current_value,
                                  start_date, end_date
                    """, (
                        self.user_id,
                        'Monthly Passive Income Target',
                        'monthly_income',
                        2500.00,
                        'USD',
                        'monthly',
                        month_start.date(),
                        month_end.date()
                    ))
                    row = cursor.fetchone()
                    conn.commit()

                goal_id = row[0]
                goal_name = row[1]
                goal_type = row[2]
                target_amount = float(row[3]) if row[3] else 2500.0
                period_start = datetime.combine(row[5], datetime.min.time()) if row[5] else month_start
                period_end = datetime.combine(row[6], datetime.max.time()) if row[6] else month_end

                # Calculate actual income for this period
                income = self._calculate_period_income(cursor, period_start, period_end)

                # Calculate time metrics
                total_days = (period_end - period_start).days + 1
                days_elapsed = (now - period_start).days + 1
                days_remaining = max(0, (period_end - now).days)

                # Calculate targets and projections
                daily_target = target_amount / total_days if total_days > 0 else 0
                daily_actual = income.total / days_elapsed if days_elapsed > 0 else 0
                projected_amount = daily_actual * total_days if daily_actual > 0 else 0

                # Determine status
                progress_percent = (income.total / target_amount * 100) if target_amount > 0 else 0
                expected_progress = (days_elapsed / total_days * 100) if total_days > 0 else 0

                if progress_percent >= 100:
                    status = GoalStatus.ACHIEVED
                elif progress_percent >= expected_progress + 10:
                    status = GoalStatus.AHEAD
                elif progress_percent >= expected_progress - 10:
                    status = GoalStatus.ON_TRACK
                elif progress_percent >= expected_progress - 25:
                    status = GoalStatus.BEHIND
                else:
                    status = GoalStatus.AT_RISK

                # Calculate what's needed to hit goal
                remaining_needed = max(0, target_amount - income.total)

                # Estimate opportunities needed (assuming $150 avg premium per trade)
                avg_premium_assumption = 150.0
                opportunities_needed = int(remaining_needed / avg_premium_assumption) + 1 if remaining_needed > 0 else 0

                # Generate advice
                advice = self._generate_goal_advice(
                    status=status,
                    remaining_needed=remaining_needed,
                    days_remaining=days_remaining,
                    daily_target=daily_target,
                    daily_actual=daily_actual,
                    income_breakdown=income
                )

                # Update goal with current progress
                cursor.execute("""
                    UPDATE ava_user_goals
                    SET current_value = %s, progress_pct = %s, updated_at = NOW()
                    WHERE id = %s
                """, (income.total, progress_percent, goal_id))

                # Record progress history
                cursor.execute("""
                    INSERT INTO ava_goal_progress_history (
                        goal_id, snapshot_date, period_value, cumulative_value,
                        progress_pct, premium_collected, total_pnl, notes
                    ) VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (goal_id, snapshot_date) DO UPDATE
                    SET period_value = EXCLUDED.period_value,
                        cumulative_value = EXCLUDED.cumulative_value,
                        progress_pct = EXCLUDED.progress_pct,
                        premium_collected = EXCLUDED.premium_collected,
                        total_pnl = EXCLUDED.total_pnl,
                        notes = EXCLUDED.notes
                """, (
                    goal_id,
                    income.csp_premium + income.cc_premium + income.other_premium,  # period_value (premium only)
                    income.total,  # cumulative_value (all income)
                    progress_percent,
                    income.csp_premium + income.cc_premium + income.other_premium,  # premium_collected
                    income.realized_gains,  # total_pnl
                    f"Daily tracking - {status.value}"
                ))

                conn.commit()

                return GoalProgress(
                    goal_id=goal_id,
                    goal_name=goal_name,
                    goal_type=goal_type,
                    target_amount=target_amount,
                    current_amount=income.total,
                    period_start=period_start,
                    period_end=period_end,
                    progress_percent=progress_percent,
                    days_elapsed=days_elapsed,
                    days_remaining=days_remaining,
                    daily_target=daily_target,
                    daily_actual=daily_actual,
                    projected_amount=projected_amount,
                    status=status,
                    premium_income=income.csp_premium + income.cc_premium + income.other_premium,
                    realized_gains=income.realized_gains,
                    unrealized_gains=0,  # TODO: Calculate from open positions
                    advice=advice,
                    opportunities_needed=opportunities_needed,
                    avg_premium_needed=remaining_needed / days_remaining if days_remaining > 0 else 0
                )

        except Exception as e:
            logger.error(f"Error getting monthly income goal: {e}")
            return None

    def _calculate_period_income(
        self,
        cursor,
        period_start: datetime,
        period_end: datetime
    ) -> IncomeBreakdown:
        """Calculate total income from all sources for a period"""
        breakdown = IncomeBreakdown()

        try:
            # Get premium income from trade_history (CSPs and CCs)
            cursor.execute("""
                SELECT
                    strategy_type,
                    SUM(premium_collected * contracts) as total_premium,
                    symbol
                FROM trade_history
                WHERE open_date >= %s AND open_date <= %s
                GROUP BY strategy_type, symbol
            """, (period_start.date(), period_end.date()))

            rows = cursor.fetchall()
            for row in rows:
                strategy = row[0] or 'other'
                premium = float(row[1]) if row[1] else 0
                symbol = row[2]

                if 'put' in strategy.lower() or 'csp' in strategy.lower():
                    breakdown.csp_premium += premium
                elif 'call' in strategy.lower() or 'cc' in strategy.lower():
                    breakdown.cc_premium += premium
                else:
                    breakdown.other_premium += premium

                # Track by symbol
                if symbol:
                    breakdown.by_symbol[symbol] = breakdown.by_symbol.get(symbol, 0) + premium

            # Get realized gains from trade_journal
            cursor.execute("""
                SELECT
                    SUM(realized_pnl) as total_pnl,
                    DATE_TRUNC('week', closed_at) as week_start
                FROM trade_journal
                WHERE closed_at >= %s AND closed_at <= %s
                  AND realized_pnl IS NOT NULL
                GROUP BY DATE_TRUNC('week', closed_at)
            """, (period_start, period_end))

            rows = cursor.fetchall()
            for row in rows:
                pnl = float(row[0]) if row[0] else 0
                breakdown.realized_gains += pnl

                # Track by week
                if row[1]:
                    week_key = row[1].strftime('%Y-W%W')
                    breakdown.by_week[week_key] = breakdown.by_week.get(week_key, 0) + pnl

            # Also get premium-based P&L from trade_history (closed trades)
            cursor.execute("""
                SELECT SUM(profit_loss)
                FROM trade_history
                WHERE close_date >= %s AND close_date <= %s
                  AND status IN ('closed', 'assigned')
                  AND profit_loss IS NOT NULL
            """, (period_start.date(), period_end.date()))

            row = cursor.fetchone()
            if row and row[0]:
                breakdown.realized_gains += float(row[0])

            breakdown.total = (
                breakdown.csp_premium +
                breakdown.cc_premium +
                breakdown.other_premium +
                breakdown.realized_gains +
                breakdown.dividend_income
            )

        except Exception as e:
            logger.error(f"Error calculating period income: {e}")

        return breakdown

    def _generate_goal_advice(
        self,
        status: GoalStatus,
        remaining_needed: float,
        days_remaining: int,
        daily_target: float,
        daily_actual: float,
        income_breakdown: IncomeBreakdown
    ) -> List[str]:
        """Generate actionable advice based on goal status"""
        advice = []

        if status == GoalStatus.ACHIEVED:
            advice.append("Congratulations! You've hit your monthly target!")
            advice.append("Consider banking gains or rolling into next month's positions")
            return advice

        if status == GoalStatus.AHEAD:
            advice.append(f"Great progress! You're ahead of schedule")
            if remaining_needed > 0:
                advice.append(f"${remaining_needed:,.0f} more to reach your goal")

        elif status == GoalStatus.ON_TRACK:
            advice.append("You're on pace to hit your monthly target")
            if days_remaining > 0:
                daily_needed = remaining_needed / days_remaining
                advice.append(f"Maintain ${daily_needed:,.0f}/day average to stay on track")

        elif status == GoalStatus.BEHIND:
            advice.append("You're falling behind - time to accelerate")
            if days_remaining > 0:
                daily_needed = remaining_needed / days_remaining
                advice.append(f"Need ${daily_needed:,.0f}/day to catch up (vs ${daily_actual:,.0f} current)")

            # Specific recommendations
            if income_breakdown.csp_premium < income_breakdown.cc_premium:
                advice.append("Look for more CSP opportunities - they typically have higher premiums")

        elif status == GoalStatus.AT_RISK:
            advice.append("Goal is at risk - aggressive action needed")
            if days_remaining > 0:
                daily_needed = remaining_needed / days_remaining
                advice.append(f"Need ${daily_needed:,.0f}/day - consider increasing position sizes")
            advice.append("Review scanner for high-premium opportunities")
            advice.append("Consider weekly expiry trades for faster premium capture")

        # Add general tips based on breakdown
        if income_breakdown.csp_premium == 0 and income_breakdown.cc_premium == 0:
            advice.append("No premium collected yet - open your first CSP position today!")

        if remaining_needed > 500 and days_remaining < 7:
            advice.append("Consider 0-3 DTE trades for rapid premium capture")

        return advice

    def get_weekly_progress(self) -> List[Dict[str, Any]]:
        """Get week-by-week progress for the current month"""
        try:
            now = datetime.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get weekly income data
                cursor.execute("""
                    SELECT
                        DATE_TRUNC('week', closed_at) as week_start,
                        SUM(realized_pnl) as pnl,
                        COUNT(*) as trade_count
                    FROM trade_journal
                    WHERE closed_at >= %s
                      AND realized_pnl IS NOT NULL
                    GROUP BY DATE_TRUNC('week', closed_at)
                    ORDER BY week_start
                """, (month_start,))

                rows = cursor.fetchall()

                weeks = []
                for row in rows:
                    weeks.append({
                        'week_start': row[0].strftime('%Y-%m-%d') if row[0] else '',
                        'income': float(row[1]) if row[1] else 0,
                        'trade_count': int(row[2]) if row[2] else 0
                    })

                return weeks

        except Exception as e:
            logger.error(f"Error getting weekly progress: {e}")
            return []

    def get_goal_summary_for_chat(self) -> str:
        """Get a formatted summary of goal progress for chat context"""
        goal = self.get_monthly_income_goal()

        if not goal:
            return "No active income goal set."

        lines = [
            f"**{goal.goal_name}**",
            f"Target: ${goal.target_amount:,.0f}/month",
            f"Current: ${goal.current_amount:,.0f} ({goal.progress_percent:.1f}%)",
            f"Status: {goal.status.value.replace('_', ' ').title()}",
            f"Days remaining: {goal.days_remaining}",
            ""
        ]

        if goal.advice:
            lines.append("**Advice:**")
            for tip in goal.advice[:3]:  # Top 3 tips
                lines.append(f"- {tip}")

        if goal.status in [GoalStatus.BEHIND, GoalStatus.AT_RISK]:
            lines.append("")
            lines.append(f"**Action needed:** {goal.opportunities_needed} trades at ~${goal.avg_premium_needed:,.0f} avg premium")

        return "\n".join(lines)

    def record_income(
        self,
        amount: float,
        source: str,
        symbol: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Manually record income toward goal (for tracking external income)"""
        try:
            goal = self.get_monthly_income_goal()
            if not goal:
                return False

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Update goal current value
                cursor.execute("""
                    UPDATE ava_user_goals
                    SET current_value = current_value + %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (amount, goal.goal_id))

                # Record in history - use upsert for today's record
                cursor.execute("""
                    INSERT INTO ava_goal_progress_history (
                        goal_id, snapshot_date, period_value, cumulative_value,
                        progress_pct, notes
                    ) VALUES (%s, CURRENT_DATE, %s, %s, %s, %s)
                    ON CONFLICT (goal_id, snapshot_date) DO UPDATE
                    SET period_value = ava_goal_progress_history.period_value + EXCLUDED.period_value,
                        cumulative_value = ava_goal_progress_history.cumulative_value + EXCLUDED.period_value,
                        notes = EXCLUDED.notes
                """, (
                    goal.goal_id,
                    amount,
                    goal.current_amount + amount,
                    ((goal.current_amount + amount) / goal.target_amount * 100) if goal.target_amount > 0 else 0,
                    f"{source}: {symbol or ''} - {notes or ''}"
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error recording income: {e}")
            return False

    def update_goal_target(self, new_target: float) -> bool:
        """Update the monthly income target"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE ava_user_goals
                    SET target_value = %s, updated_at = NOW()
                    WHERE user_id = %s
                      AND goal_type = 'monthly_income'
                      AND status = 'active'
                """, (new_target, self.user_id))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating goal target: {e}")
            return False

    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over recent months"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get monthly totals for last 6 months
                cursor.execute("""
                    SELECT
                        DATE_TRUNC('month', closed_at) as month,
                        SUM(realized_pnl) as monthly_pnl,
                        COUNT(*) as trade_count,
                        AVG(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as win_rate
                    FROM trade_journal
                    WHERE closed_at >= CURRENT_DATE - INTERVAL '6 months'
                      AND realized_pnl IS NOT NULL
                    GROUP BY DATE_TRUNC('month', closed_at)
                    ORDER BY month DESC
                """)

                rows = cursor.fetchall()

                months = []
                for row in rows:
                    months.append({
                        'month': row[0].strftime('%Y-%m') if row[0] else '',
                        'income': float(row[1]) if row[1] else 0,
                        'trades': int(row[2]) if row[2] else 0,
                        'win_rate': float(row[3]) if row[3] else 0
                    })

                # Calculate trends
                if len(months) >= 2:
                    recent_avg = sum(m['income'] for m in months[:3]) / min(3, len(months))
                    older_avg = sum(m['income'] for m in months[3:]) / max(1, len(months) - 3)
                    trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                else:
                    trend = "insufficient_data"

                return {
                    'monthly_history': months,
                    'trend': trend,
                    'average_monthly': sum(m['income'] for m in months) / max(1, len(months)),
                    'best_month': max(months, key=lambda m: m['income']) if months else None,
                    'worst_month': min(months, key=lambda m: m['income']) if months else None
                }

        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {
                'monthly_history': [],
                'trend': 'error',
                'average_monthly': 0,
                'best_month': None,
                'worst_month': None
            }


# Convenience function for quick access
def get_goal_progress(user_id: str = "default_user") -> Optional[GoalProgress]:
    """Quick helper to get current goal progress"""
    service = GoalTrackingService(user_id)
    return service.get_monthly_income_goal()


def get_goal_summary(user_id: str = "default_user") -> str:
    """Quick helper to get goal summary for chat"""
    service = GoalTrackingService(user_id)
    return service.get_goal_summary_for_chat()
