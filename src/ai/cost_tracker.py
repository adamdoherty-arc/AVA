"""
AI API Cost Tracking and Budget Management
Monitors and controls API spending across all AI models
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import psycopg2
import psycopg2.extras
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIUsage:
    """API usage record"""
    model_name: str
    input_tokens: int
    output_tokens: int
    cost: float
    request_type: str
    timestamp: datetime


@dataclass
class BudgetAlert:
    """Budget alert notification"""
    alert_type: str  # 'warning', 'critical', 'exceeded'
    current_spend: float
    budget_limit: float
    period: str
    message: str


class CostTracker:
    """
    Tracks AI API costs and enforces budget limits

    Features:
    - Real-time cost tracking
    - Budget alerts (daily, weekly, monthly)
    - Cost optimization recommendations
    - Usage analytics and reporting
    """

    # Cost per 1M tokens (as of 2025-01)
    COST_PER_1M_TOKENS = {
        'gpt4': {
            'input': 10.00,
            'output': 30.00
        },
        'gpt4-turbo': {
            'input': 10.00,
            'output': 30.00
        },
        'claude': {
            'input': 3.00,
            'output': 15.00
        },
        'claude-3-5-sonnet': {
            'input': 3.00,
            'output': 15.00
        },
        'gemini': {
            'input': 0.50,
            'output': 1.50
        },
        'gemini-pro': {
            'input': 0.50,
            'output': 1.50
        },
        'llama3': {
            'input': 0.00,
            'output': 0.00
        }
    }

    # Default budget limits (USD)
    DEFAULT_BUDGETS = {
        'daily': 150.00,
        'weekly': 700.00,
        'monthly': 2500.00
    }

    def __init__(self) -> None:
        """Initialize cost tracker with database connection"""
        self.db_config = {
            'host': 'localhost',
            'port': '5432',
            'database': 'magnus',
            'user': 'postgres',
            'password': os.getenv('DB_PASSWORD')
        }

        self._ensure_schema()

    def _get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _ensure_schema(self) -> None:
        """Create cost tracking tables if they don't exist"""
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            # API usage log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kalshi_ai_usage (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(50) NOT NULL,
                    request_type VARCHAR(50),
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    cost DECIMAL(10,4) DEFAULT 0,
                    market_ticker VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Budget settings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kalshi_ai_budgets (
                    id SERIAL PRIMARY KEY,
                    period VARCHAR(20) NOT NULL,
                    budget_limit DECIMAL(10,2) NOT NULL,
                    alert_threshold DECIMAL(5,2) DEFAULT 80.0,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(period)
                )
            """)

            # Insert default budgets
            for period, limit in self.DEFAULT_BUDGETS.items():
                cur.execute("""
                    INSERT INTO kalshi_ai_budgets (period, budget_limit)
                    VALUES (%s, %s)
                    ON CONFLICT (period) DO NOTHING
                """, (period, limit))

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ai_usage_timestamp
                ON kalshi_ai_usage(timestamp DESC)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ai_usage_model
                ON kalshi_ai_usage(model_name)
            """)

            conn.commit()
            logger.info("Cost tracking schema initialized")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating schema: {e}")
        finally:
            cur.close()
            conn.close()

    def log_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str = 'market_analysis',
        market_ticker: Optional[str] = None
    ) -> float:
        """
        Log API usage and calculate cost

        Args:
            model_name: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_type: Type of request
            market_ticker: Associated market ticker (optional)

        Returns:
            Cost in USD
        """
        # Calculate cost
        cost = self.calculate_cost(model_name, input_tokens, output_tokens)

        # Store in database
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO kalshi_ai_usage (
                    model_name, request_type, input_tokens,
                    output_tokens, cost, market_ticker
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                model_name, request_type, input_tokens,
                output_tokens, cost, market_ticker
            ))

            conn.commit()

            logger.debug(
                f"Logged usage: {model_name} - "
                f"{input_tokens}in/{output_tokens}out - ${cost:.4f}"
            )

        except Exception as e:
            conn.rollback()
            logger.error(f"Error logging usage: {e}")
        finally:
            cur.close()
            conn.close()

        # Check budget limits
        self._check_budget_alerts()

        return cost

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for API usage

        Args:
            model_name: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        # Normalize model name
        model_key = model_name.lower().replace('-', '')

        # Find matching cost structure
        costs = None
        for key, value in self.COST_PER_1M_TOKENS.items():
            if key in model_key or model_key in key:
                costs = value
                break

        if not costs:
            logger.warning(f"Unknown model {model_name}, assuming GPT-4 pricing")
            costs = self.COST_PER_1M_TOKENS['gpt4']

        input_cost = (input_tokens / 1_000_000) * costs['input']
        output_cost = (output_tokens / 1_000_000) * costs['output']

        return input_cost + output_cost

    def get_spending(
        self,
        period: str = 'daily',
        model_name: Optional[str] = None
    ) -> float:
        """
        Get total spending for a period

        Args:
            period: 'daily', 'weekly', or 'monthly'
            model_name: Filter by model (optional)

        Returns:
            Total spending in USD
        """
        # Calculate time range
        now = datetime.now()
        if period == 'daily':
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'weekly':
            start_time = now - timedelta(days=7)
        elif period == 'monthly':
            start_time = now - timedelta(days=30)
        else:
            raise ValueError(f"Invalid period: {period}")

        conn = self._get_connection()
        cur = conn.cursor()

        try:
            if model_name:
                cur.execute("""
                    SELECT COALESCE(SUM(cost), 0)
                    FROM kalshi_ai_usage
                    WHERE timestamp >= %s
                    AND model_name = %s
                """, (start_time, model_name))
            else:
                cur.execute("""
                    SELECT COALESCE(SUM(cost), 0)
                    FROM kalshi_ai_usage
                    WHERE timestamp >= %s
                """, (start_time,))

            total = cur.fetchone()[0]
            return float(total)

        finally:
            cur.close()
            conn.close()

    def _check_budget_alerts(self) -> List[BudgetAlert]:
        """
        Check if spending exceeds budget thresholds

        Returns:
            List of budget alerts
        """
        alerts = []

        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            # Get all budget settings
            cur.execute("SELECT * FROM kalshi_ai_budgets")
            budgets = cur.fetchall()

            for budget in budgets:
                period = budget['period']
                limit = float(budget['budget_limit'])
                threshold = float(budget['alert_threshold'])

                # Get current spending
                current = self.get_spending(period)

                # Calculate percentage
                pct = (current / limit) * 100 if limit > 0 else 0

                # Generate alerts
                if pct >= 100:
                    alerts.append(BudgetAlert(
                        alert_type='exceeded',
                        current_spend=current,
                        budget_limit=limit,
                        period=period,
                        message=f"BUDGET EXCEEDED: {period} spending (${current:.2f}) "
                               f"exceeds limit (${limit:.2f})"
                    ))
                elif pct >= threshold:
                    alerts.append(BudgetAlert(
                        alert_type='critical' if pct >= 95 else 'warning',
                        current_spend=current,
                        budget_limit=limit,
                        period=period,
                        message=f"{period.title()} budget at {pct:.1f}% "
                               f"(${current:.2f} / ${limit:.2f})"
                    ))

            # Log critical alerts
            for alert in alerts:
                if alert.alert_type in ['critical', 'exceeded']:
                    logger.warning(alert.message)

        finally:
            cur.close()
            conn.close()

        return alerts

    def get_usage_stats(self, days: int = 7) -> Dict:
        """
        Get usage statistics for the past N days

        Args:
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        start_time = datetime.now() - timedelta(days=days)

        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            # Total spending
            cur.execute("""
                SELECT
                    COALESCE(SUM(cost), 0) as total_cost,
                    COUNT(*) as total_requests,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens
                FROM kalshi_ai_usage
                WHERE timestamp >= %s
            """, (start_time,))

            totals = cur.fetchone()

            # By model
            cur.execute("""
                SELECT
                    model_name,
                    COALESCE(SUM(cost), 0) as cost,
                    COUNT(*) as requests,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens
                FROM kalshi_ai_usage
                WHERE timestamp >= %s
                GROUP BY model_name
                ORDER BY cost DESC
            """, (start_time,))

            by_model = cur.fetchall()

            # Daily breakdown
            cur.execute("""
                SELECT
                    DATE(timestamp) as date,
                    COALESCE(SUM(cost), 0) as cost,
                    COUNT(*) as requests
                FROM kalshi_ai_usage
                WHERE timestamp >= %s
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (start_time,))

            daily = cur.fetchall()

            return {
                'period_days': days,
                'total_cost': float(totals['total_cost']),
                'total_requests': totals['total_requests'],
                'total_input_tokens': totals['total_input_tokens'],
                'total_output_tokens': totals['total_output_tokens'],
                'avg_cost_per_request': (
                    float(totals['total_cost']) / totals['total_requests']
                    if totals['total_requests'] > 0 else 0
                ),
                'by_model': [dict(row) for row in by_model],
                'daily_breakdown': [dict(row) for row in daily]
            }

        finally:
            cur.close()
            conn.close()

    def get_optimization_recommendations(self) -> List[str]:
        """
        Analyze usage patterns and suggest cost optimizations

        Returns:
            List of recommendation strings
        """
        recommendations = []

        stats = self.get_usage_stats(days=7)

        # Check if expensive models are overused
        for model_usage in stats['by_model']:
            model = model_usage['model_name']
            cost = model_usage['cost']
            requests = model_usage['requests']

            if 'gpt4' in model.lower() and requests > 100:
                savings = cost * 0.4  # Could save 40% by using cheaper models
                recommendations.append(
                    f"Consider using Gemini for routine analysis instead of GPT-4. "
                    f"Potential savings: ${savings:.2f}/week"
                )

            if cost > 50 and 'llama3' not in model.lower():
                recommendations.append(
                    f"High usage of {model}. Consider using local Llama3 model "
                    f"for non-critical analysis to reduce costs."
                )

        # Check daily budget pace
        daily_avg = stats['total_cost'] / stats['period_days']
        monthly_projection = daily_avg * 30

        if monthly_projection > self.DEFAULT_BUDGETS['monthly']:
            recommendations.append(
                f"Current pace (${daily_avg:.2f}/day) projects to "
                f"${monthly_projection:.2f}/month, exceeding monthly budget. "
                f"Consider switching to 'cost' or 'fast' ensemble mode."
            )

        # Check for inefficient usage patterns
        if stats['total_requests'] > 1000:
            recommendations.append(
                "High request volume detected. Consider implementing response "
                "caching to reduce duplicate analyses."
            )

        if not recommendations:
            recommendations.append("Usage is within optimal parameters. No changes needed.")

        return recommendations

    def set_budget(self, period: str, limit: float):
        """
        Set budget limit for a period

        Args:
            period: 'daily', 'weekly', or 'monthly'
            limit: Budget limit in USD
        """
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO kalshi_ai_budgets (period, budget_limit)
                VALUES (%s, %s)
                ON CONFLICT (period) DO UPDATE
                SET budget_limit = EXCLUDED.budget_limit,
                    updated_at = NOW()
            """, (period, limit))

            conn.commit()
            logger.info(f"Set {period} budget to ${limit:.2f}")

        finally:
            cur.close()
            conn.close()

    def generate_report(self, days: int = 7) -> str:
        """
        Generate usage and cost report

        Args:
            days: Number of days to include

        Returns:
            Formatted report string
        """
        stats = self.get_usage_stats(days)
        alerts = self._check_budget_alerts()
        recommendations = self.get_optimization_recommendations()

        report = f"""
{'='*80}
KALSHI AI COST REPORT
{'='*80}
Period: Last {days} days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Cost: ${stats['total_cost']:.2f}
Total Requests: {stats['total_requests']:,}
Avg Cost/Request: ${stats['avg_cost_per_request']:.4f}
Total Tokens: {stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out

BUDGET STATUS
-------------
"""

        for period in ['daily', 'weekly', 'monthly']:
            current = self.get_spending(period)
            limit = self.DEFAULT_BUDGETS[period]
            pct = (current / limit) * 100 if limit > 0 else 0

            status = "OK" if pct < 80 else "WARNING" if pct < 95 else "CRITICAL"
            report += f"{period.title():8} ${current:8.2f} / ${limit:8.2f} ({pct:5.1f}%) [{status}]\n"

        report += f"\nUSAGE BY MODEL\n{'-'*80}\n"
        for model in stats['by_model']:
            report += (
                f"{model['model_name']:20} "
                f"${model['cost']:8.2f}  "
                f"{model['requests']:6,} requests  "
                f"{model['input_tokens']:10,} in / {model['output_tokens']:10,} out\n"
            )

        if alerts:
            report += f"\nBUDGET ALERTS\n{'-'*80}\n"
            for alert in alerts:
                report += f"[{alert.alert_type.upper()}] {alert.message}\n"

        report += f"\nOPTIMIZATION RECOMMENDATIONS\n{'-'*80}\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"

        report += f"\n{'='*80}\n"

        return report


# ============================================================================
# Testing
# ============================================================================

def test_cost_tracker():
    """Test cost tracking functionality"""

    tracker = CostTracker()

    print("\n" + "="*80)
    print("COST TRACKER TEST")
    print("="*80)

    # Log some usage
    print("\nLogging test usage...")
    tracker.log_usage('gpt4', 2000, 500, 'market_analysis', 'NFL-KC-BUF-001')
    tracker.log_usage('claude', 1800, 450, 'market_analysis', 'NFL-KC-BUF-002')
    tracker.log_usage('gemini', 1900, 400, 'market_analysis', 'NFL-KC-BUF-003')

    # Get spending
    print("\nCurrent spending:")
    for period in ['daily', 'weekly', 'monthly']:
        spending = tracker.get_spending(period)
        print(f"  {period}: ${spending:.2f}")

    # Get stats
    print("\nUsage statistics:")
    stats = tracker.get_usage_stats(days=1)
    print(f"  Total cost: ${stats['total_cost']:.2f}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Avg cost/request: ${stats['avg_cost_per_request']:.4f}")

    # Get recommendations
    print("\nOptimization recommendations:")
    for rec in tracker.get_optimization_recommendations():
        print(f"  - {rec}")

    # Generate report
    print("\n" + "="*80)
    print("FULL REPORT")
    print("="*80)
    print(tracker.generate_report(days=7))


if __name__ == "__main__":
    test_cost_tracker()
