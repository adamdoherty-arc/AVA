from typing import List, Optional
import logging
from psycopg2.extras import RealDictCursor
from backend.database.connection import db_pool
from backend.models.market import Market

logger = logging.getLogger(__name__)

class PredictionService:
    def get_prediction_markets(self, sector: Optional[str] = None, limit: int = 50) -> List[Market]:
        """
        Get non-sports prediction markets.
        """
        with db_pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                try:
                    query = """
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
                    """
                    
                    params = []
                    # Note: Sector filtering would require more complex logic or a 'sector' column
                    # For now, we return all non-NFL markets
                    
                    query += " ORDER BY p.overall_rank ASC NULLS LAST LIMIT %s"
                    params.append(limit)
                    
                    cur.execute(query, tuple(params))
                    results = cur.fetchall()
                    
                    return [Market(**row) for row in results]
                    
                except Exception as e:
                    logger.error(f"Error fetching prediction markets: {e}")
                    return []

prediction_service = PredictionService()
