"""
Prediction Service - Fetches prediction markets data

Updated: 2025-12-04 - Migrated to async database pattern
"""

from typing import List, Optional
import structlog

from backend.infrastructure.database import get_database
from backend.models.market import Market

logger = structlog.get_logger(__name__)


class PredictionService:
    """Async service for prediction markets data."""

    async def get_prediction_markets(self, sector: Optional[str] = None, limit: int = 50) -> List[Market]:
        """
        Get non-sports prediction markets.
        """
        try:
            db = await get_database()

            rows = await db.fetch("""
                SELECT
                    m.id,
                    m.ticker,
                    m.title,
                    m.market_type,
                    m.home_team,
                    m.away_team,
                    m.game_date,
                    m.yes_price,
                    m.no_price,
                    m.volume,
                    m.close_time,
                    p.predicted_outcome,
                    p.confidence_score,
                    p.edge_percentage,
                    p.overall_rank,
                    p.recommended_action,
                    p.recommended_stake_pct,
                    p.reasoning
                FROM kalshi_markets m
                LEFT JOIN kalshi_predictions p ON m.id = p.market_id
                WHERE m.status IN ('open', 'active')
                AND m.market_type != 'nfl'
                ORDER BY p.overall_rank ASC NULLS LAST
                LIMIT $1
            """, limit)

            return [Market(**dict(row)) for row in rows]

        except Exception as e:
            logger.error("Error fetching prediction markets", error=str(e))
            return []


# Singleton instance
prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    """Get the prediction service singleton."""
    return prediction_service
