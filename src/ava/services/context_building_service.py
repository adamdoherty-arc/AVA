"""
AVA Context Building Service - Assembles Comprehensive Context for Chat

This service builds rich context for every chat interaction, including:
- Current portfolio positions with Greeks
- Active goals and progress
- Recent trade history patterns
- XTrades activity from followed traders
- Earnings calendar for portfolio symbols
- RAG knowledge base retrieval

The assembled context is injected into every chat to enable truly intelligent responses.

Usage:
    from src.ava.services.context_building_service import ContextBuildingService

    context_service = ContextBuildingService()
    context = await context_service.build_context(
        user_query="Should I sell covered calls on NVDA?",
        user_id="default_user"
    )

Author: AVA Trading Platform
Created: 2025-11-28
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import json

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """Context priority levels for token budget management"""
    CRITICAL = 1    # Always include (positions at risk, urgent alerts)
    HIGH = 2        # Include if space allows (active goals, recent trades)
    MEDIUM = 3      # Include for relevant queries (historical patterns)
    LOW = 4         # Include if explicitly relevant (RAG background)


@dataclass
class ContextItem:
    """A single item of context with priority and source"""
    content: str
    priority: ContextPriority
    source: str
    token_estimate: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class AssembledContext:
    """The final assembled context ready for injection"""
    portfolio_summary: Optional[str] = None
    positions_context: Optional[str] = None
    goals_context: Optional[str] = None
    earnings_context: Optional[str] = None
    xtrades_context: Optional[str] = None
    alerts_context: Optional[str] = None
    rag_context: Optional[str] = None
    user_preferences: Optional[Dict] = None

    total_tokens: int = 0
    items_included: int = 0

    def to_prompt_string(self, max_tokens: int = 2000) -> str:
        """Convert to a formatted string for LLM prompt injection"""
        sections = []

        if self.portfolio_summary:
            sections.append(f"## Your Portfolio\n{self.portfolio_summary}")

        if self.positions_context:
            sections.append(f"## Current Positions\n{self.positions_context}")

        if self.goals_context:
            sections.append(f"## Your Goals\n{self.goals_context}")

        if self.earnings_context:
            sections.append(f"## Earnings Calendar\n{self.earnings_context}")

        if self.xtrades_context:
            sections.append(f"## XTrades Activity\n{self.xtrades_context}")

        if self.alerts_context:
            sections.append(f"## Active Alerts\n{self.alerts_context}")

        if self.rag_context:
            sections.append(f"## Relevant Knowledge\n{self.rag_context}")

        if not sections:
            return ""

        context_str = "\n\n".join(sections)

        # Truncate if needed
        if len(context_str) > max_tokens * 4:  # Rough char to token estimate
            context_str = context_str[:max_tokens * 4] + "\n...[truncated]"

        return f"---\n# CONTEXT FOR THIS CONVERSATION\n\n{context_str}\n---\n"


class ContextBuildingService:
    """
    Builds comprehensive context for chat interactions.

    Assembles context from multiple sources:
    - Portfolio positions (Robinhood)
    - User goals and progress
    - Earnings calendar
    - XTrades followed traders
    - Active alerts
    - RAG knowledge base
    """

    def __init__(
        self,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        max_context_tokens: int = 2000
    ):
        """
        Initialize the ContextBuildingService.

        Args:
            db_*: Database connection parameters
            max_context_tokens: Maximum tokens for context (default 2000)
        """
        self.db_config = {
            "host": db_host or os.getenv("DB_HOST", "localhost"),
            "port": db_port or int(os.getenv("DB_PORT", "5432")),
            "database": db_name or os.getenv("DB_NAME", "wheel_strategy"),
            "user": db_user or os.getenv("DB_USER", "postgres"),
            "password": db_password or os.getenv("DB_PASSWORD", "")
        }
        self.max_context_tokens = max_context_tokens

        # Try to import portfolio service
        try:
            from backend.services.portfolio_service import PortfolioService
            self.portfolio_service = PortfolioService()
        except ImportError:
            logger.warning("Portfolio service not available")
            self.portfolio_service = None

        # Try to import RAG
        try:
            from src.rag import get_rag
            self.rag = get_rag()
        except ImportError:
            logger.warning("RAG service not available")
            self.rag = None

        logger.info(f"ContextBuildingService initialized (Portfolio: {self.portfolio_service is not None}, RAG: {self.rag is not None})")

    def _get_db_connection(self):
        """Get a database connection."""
        return psycopg2.connect(**self.db_config)

    async def build_context(
        self,
        user_query: str,
        user_id: str = "default_user",
        include_portfolio: bool = True,
        include_goals: bool = True,
        include_earnings: bool = True,
        include_xtrades: bool = True,
        include_alerts: bool = True,
        include_rag: bool = True
    ) -> AssembledContext:
        """
        Build comprehensive context for a chat interaction.

        Args:
            user_query: The user's query (used for RAG retrieval)
            user_id: User identifier
            include_*: Flags to include/exclude context sources

        Returns:
            AssembledContext with all relevant context
        """
        context = AssembledContext()
        context_items: List[ContextItem] = []

        # Gather context from all sources in parallel
        tasks = []

        if include_portfolio:
            tasks.append(self._get_portfolio_context())

        if include_goals:
            tasks.append(self._get_goals_context(user_id))

        if include_earnings:
            tasks.append(self._get_earnings_context())

        if include_xtrades:
            tasks.append(self._get_xtrades_context())

        if include_alerts:
            tasks.append(self._get_alerts_context())

        if include_rag and self.rag:
            tasks.append(self._get_rag_context(user_query))

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        result_idx = 0

        if include_portfolio:
            result = results[result_idx]
            if result and not isinstance(result, Exception):
                context.portfolio_summary = result.get("summary")
                context.positions_context = result.get("positions")
            result_idx += 1

        if include_goals:
            result = results[result_idx]
            if result and not isinstance(result, Exception):
                context.goals_context = result
            result_idx += 1

        if include_earnings:
            result = results[result_idx]
            if result and not isinstance(result, Exception):
                context.earnings_context = result
            result_idx += 1

        if include_xtrades:
            result = results[result_idx]
            if result and not isinstance(result, Exception):
                context.xtrades_context = result
            result_idx += 1

        if include_alerts:
            result = results[result_idx]
            if result and not isinstance(result, Exception):
                context.alerts_context = result
            result_idx += 1

        if include_rag and self.rag:
            result = results[result_idx]
            if result and not isinstance(result, Exception):
                context.rag_context = result
            result_idx += 1

        return context

    async def _get_portfolio_context(self) -> Optional[Dict[str, str]]:
        """Get portfolio context from Robinhood positions."""
        if not self.portfolio_service:
            return None

        try:
            # Run sync method in executor
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(
                None,
                lambda: asyncio.run(self.portfolio_service.get_positions())
            )

            if not positions:
                return None

            # Build summary
            summary = positions.get("summary", {})
            summary_text = (
                f"Total Equity: ${summary.get('total_equity', 0):,.2f}\n"
                f"Buying Power: ${summary.get('buying_power', 0):,.2f}\n"
                f"Total Positions: {summary.get('total_positions', 0)}"
            )

            # Build positions text
            positions_parts = []

            # Stock positions
            stocks = positions.get("stocks", [])
            if stocks:
                positions_parts.append("**Stock Positions:**")
                for stock in stocks[:5]:  # Limit to 5
                    pl_indicator = "+" if stock.get("pl", 0) >= 0 else ""
                    positions_parts.append(
                        f"- {stock['symbol']}: {stock['quantity']} shares @ ${stock.get('current_price', 0):.2f} "
                        f"(P/L: {pl_indicator}${stock.get('pl', 0):.2f})"
                    )

            # Option positions
            options = positions.get("options", [])
            if options:
                positions_parts.append("\n**Option Positions:**")
                for opt in options[:10]:  # Limit to 10
                    pl_indicator = "+" if opt.get("pl", 0) >= 0 else ""
                    positions_parts.append(
                        f"- {opt.get('symbol')} ${opt.get('strike_price')} {opt.get('option_type')} "
                        f"exp {opt.get('expiration_date')} (x{opt.get('quantity', 1)}) "
                        f"Delta: {opt.get('delta', 'N/A')}, Theta: ${opt.get('theta', 'N/A')}/day"
                    )

            positions_text = "\n".join(positions_parts) if positions_parts else None

            return {
                "summary": summary_text,
                "positions": positions_text
            }

        except Exception as e:
            logger.error(f"Error getting portfolio context: {e}")
            return None

    async def _get_goals_context(self, user_id: str) -> Optional[str]:
        """Get active goals and progress."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT goal_name, goal_type, target_value, target_unit,
                               current_value, progress_pct, period_type, status
                        FROM ava_user_goals
                        WHERE user_id = %s AND status = 'active'
                        ORDER BY progress_pct DESC
                        LIMIT 5
                    """, (user_id,))

                    goals = cur.fetchall()

                    if not goals:
                        return None

                    parts = []
                    for goal in goals:
                        progress_status = "exceeded" if goal["progress_pct"] >= 100 else \
                                         "on track" if goal["progress_pct"] >= 75 else \
                                         "moderate" if goal["progress_pct"] >= 50 else "behind"

                        parts.append(
                            f"- **{goal['goal_name']}** ({goal['period_type'].title()}): "
                            f"${goal['current_value']:,.2f} / ${goal['target_value']:,.2f} "
                            f"({goal['progress_pct']:.1f}% - {progress_status})"
                        )

                    return "\n".join(parts)

        except Exception as e:
            logger.error(f"Error getting goals context: {e}")
            return None

    async def _get_earnings_context(self) -> Optional[str]:
        """Get upcoming earnings for portfolio symbols."""
        try:
            # First get portfolio symbols
            portfolio_symbols = set()
            if self.portfolio_service:
                try:
                    loop = asyncio.get_event_loop()
                    positions = await loop.run_in_executor(
                        None,
                        lambda: asyncio.run(self.portfolio_service.get_positions())
                    )
                    for stock in positions.get("stocks", []):
                        portfolio_symbols.add(stock.get("symbol"))
                    for opt in positions.get("options", []):
                        portfolio_symbols.add(opt.get("symbol"))
                except Exception:
                    pass

            if not portfolio_symbols:
                return None

            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Check if earnings table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'earnings_events'
                        )
                    """)
                    if not cur.fetchone()['exists']:
                        return None

                    placeholders = ','.join(['%s'] * len(portfolio_symbols))
                    cur.execute(f"""
                        SELECT symbol, earnings_date, time_of_day, eps_estimate
                        FROM earnings_events
                        WHERE symbol IN ({placeholders})
                          AND earnings_date >= CURRENT_DATE
                          AND earnings_date <= CURRENT_DATE + INTERVAL '14 days'
                        ORDER BY earnings_date ASC
                        LIMIT 10
                    """, tuple(portfolio_symbols))

                    earnings = cur.fetchall()

                    if not earnings:
                        return None

                    parts = ["**Upcoming earnings affecting your positions:**"]
                    for e in earnings:
                        days_away = (e['earnings_date'] - datetime.now().date()).days
                        parts.append(
                            f"- {e['symbol']}: {e['earnings_date']} ({e['time_of_day']}) - "
                            f"{days_away} days away"
                        )

                    return "\n".join(parts)

        except Exception as e:
            logger.error(f"Error getting earnings context: {e}")
            return None

    async def _get_xtrades_context(self) -> Optional[str]:
        """Get recent XTrades activity from followed traders."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Check if xtrades tables exist
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'xtrades_trades'
                        )
                    """)
                    if not cur.fetchone()['exists']:
                        return None

                    cur.execute("""
                        SELECT t.ticker, t.strategy, t.action, t.entry_price,
                               p.username as profile_username,
                               t.created_at
                        FROM xtrades_trades t
                        JOIN xtrades_profiles p ON t.profile_id = p.id
                        WHERE p.is_active = TRUE
                          AND t.created_at >= NOW() - INTERVAL '24 hours'
                        ORDER BY t.created_at DESC
                        LIMIT 5
                    """)

                    trades = cur.fetchall()

                    if not trades:
                        return None

                    parts = ["**Recent trades from followed XTrades profiles (last 24h):**"]
                    for trade in trades:
                        parts.append(
                            f"- {trade['profile_username']}: {trade['action']} {trade['ticker']} "
                            f"{trade['strategy']} @ ${trade['entry_price']:.2f}"
                        )

                    return "\n".join(parts)

        except Exception as e:
            logger.error(f"Error getting XTrades context: {e}")
            return None

    async def _get_alerts_context(self) -> Optional[str]:
        """Get active urgent/important alerts."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Check if alerts table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'ava_alerts'
                        )
                    """)
                    if not cur.fetchone()['exists']:
                        return None

                    cur.execute("""
                        SELECT category, priority, title, symbol, created_at
                        FROM ava_alerts
                        WHERE is_active = TRUE
                          AND is_read = FALSE
                          AND (expires_at IS NULL OR expires_at > NOW())
                          AND priority IN ('urgent', 'important')
                        ORDER BY
                            CASE priority
                                WHEN 'urgent' THEN 1
                                WHEN 'important' THEN 2
                                ELSE 3
                            END,
                            created_at DESC
                        LIMIT 5
                    """)

                    alerts = cur.fetchall()

                    if not alerts:
                        return None

                    parts = ["**Active alerts requiring attention:**"]
                    for alert in alerts:
                        priority_emoji = "\U0001F6A8" if alert['priority'] == 'urgent' else "\u2757"
                        parts.append(
                            f"- {priority_emoji} [{alert['category']}] {alert['title']}"
                            + (f" ({alert['symbol']})" if alert['symbol'] else "")
                        )

                    return "\n".join(parts)

        except Exception as e:
            logger.error(f"Error getting alerts context: {e}")
            return None

    async def _get_rag_context(self, user_query: str) -> Optional[str]:
        """Get relevant context from RAG knowledge base."""
        if not self.rag:
            return None

        try:
            context = self.rag.get_context_for_query(
                query_text=user_query,
                n_results=3,
                max_context_length=500
            )

            if context and context.strip():
                return context

            return None

        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return None

    def get_quick_context(self, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Get a quick synchronous context snapshot (for non-async contexts).

        Returns minimal context without async operations.
        """
        context = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "goals": None,
            "alerts_count": 0
        }

        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get primary goal
                    cur.execute("""
                        SELECT goal_name, target_value, current_value, progress_pct
                        FROM ava_user_goals
                        WHERE user_id = %s AND status = 'active' AND goal_type = 'monthly_income'
                        LIMIT 1
                    """, (user_id,))
                    goal = cur.fetchone()
                    if goal:
                        context["goals"] = dict(goal)

                    # Get unread alert count
                    cur.execute("""
                        SELECT COUNT(*) as count
                        FROM ava_alerts
                        WHERE is_active = TRUE AND is_read = FALSE
                          AND (expires_at IS NULL OR expires_at > NOW())
                    """)
                    result = cur.fetchone()
                    context["alerts_count"] = result["count"] if result else 0

        except Exception as e:
            logger.error(f"Error getting quick context: {e}")

        return context


# Convenience function
def get_context_service() -> ContextBuildingService:
    """Get singleton instance of context building service."""
    if not hasattr(get_context_service, '_instance'):
        get_context_service._instance = ContextBuildingService()
    return get_context_service._instance


if __name__ == "__main__":
    import asyncio

    async def test():
        print("Context Building Service - Test\n")

        service = ContextBuildingService()

        # Build full context
        print("Building context for query: 'Should I sell covered calls on NVDA?'\n")

        context = await service.build_context(
            user_query="Should I sell covered calls on NVDA?",
            user_id="default_user"
        )

        print("=" * 50)
        print(context.to_prompt_string())
        print("=" * 50)

        # Quick context
        print("\nQuick context:")
        quick = service.get_quick_context()
        print(json.dumps(quick, indent=2, default=str))

    asyncio.run(test())
