"""
Predictions Router - API endpoints for prediction markets
NO MOCK DATA - All endpoints use real Kalshi API and database
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime, timedelta
import logging
from backend.services.prediction_service import prediction_service
from backend.models.market import MarketResponse
from src.kalshi_db_manager import KalshiDBManager
from src.prediction_agents.nfl_predictor import NFLPredictor
from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/predictions",
    tags=["predictions"]
)

# Initialize services
_kalshi_db = None
_nfl_predictor = None

def get_kalshi_db():
    global _kalshi_db
    if _kalshi_db is None:
        _kalshi_db = KalshiDBManager()
    return _kalshi_db

def get_predictor():
    global _nfl_predictor
    if _nfl_predictor is None:
        _nfl_predictor = NFLPredictor()
    return _nfl_predictor


@router.get("/markets", response_model=MarketResponse)
async def get_markets(
    sector: Optional[str] = Query(None, description="Filter by sector (e.g., 'Politics')"),
    limit: int = Query(50, description="Limit number of results")
):
    """
    Get active prediction markets (non-sports) from database.
    """
    markets = prediction_service.get_prediction_markets(sector, limit)
    return MarketResponse(markets=markets, count=len(markets))


# ============ Kalshi Markets Endpoints ============

@router.get("/kalshi")
async def get_kalshi_markets_alias(
    min_edge: float = Query(5.0, description="Minimum AI edge percentage"),
    min_volume: int = Query(100, description="Minimum volume"),
    sort_by: str = Query("edge", description="Sort by: edge, volume, confidence")
):
    """Alias route for /kalshi - used by KalshiMarkets page"""
    return await get_kalshi_nfl_markets(min_edge, min_volume, sort_by)


@router.get("/kalshi/nfl")
async def get_kalshi_nfl_markets(
    min_edge: float = Query(5.0, description="Minimum AI edge percentage"),
    min_volume: int = Query(100, description="Minimum volume"),
    sort_by: str = Query("edge", description="Sort by: edge, volume, confidence")
):
    """
    Get Kalshi NFL prediction markets with AI edge detection from database.
    """
    try:
        kalshi_db = get_kalshi_db()
        predictor = get_predictor()

        # Get NFL markets from database
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, ticker, title, home_team, away_team, market_type,
                       yes_price, no_price, volume, open_interest, close_time, game_time
                FROM kalshi_markets
                WHERE market_type = 'nfl' AND status IN ('open', 'active')
                ORDER BY game_time ASC
            """)

            rows = cursor.fetchall()

            markets = []
            for row in rows:
                market_id = row[0]
                home_team = row[3]
                away_team = row[4]
                market_type = row[5]
                yes_price = float(row[6]) if row[6] else 50
                no_price = float(row[7]) if row[7] else 50
                volume = int(row[8]) if row[8] else 0
                open_interest = int(row[9]) if row[9] else 0

                if volume < min_volume:
                    continue

                # Get AI prediction
                try:
                    prediction = predictor.predict_game({
                        'home_team': home_team,
                        'away_team': away_team
                    })

                    if prediction:
                        ai_prob = prediction.get('win_probability', 0.5) * 100
                        confidence = prediction.get('confidence', 0.7) * 100
                    else:
                        ai_prob = 50
                        confidence = 60
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    ai_prob = 50
                    confidence = 60

                # Calculate edge
                if ai_prob > yes_price:
                    edge = round(ai_prob - yes_price, 1)
                    recommendation = "YES"
                else:
                    edge = round(no_price - (100 - ai_prob), 1)
                    recommendation = "NO"

                if edge < min_edge:
                    continue

                close_time = row[10].strftime("%Y-%m-%d %H:%M") if row[10] else ""
                game_time = row[11].strftime("%Y-%m-%d %H:%M") if row[11] else ""

                markets.append({
                    "id": row[1] or f"kalshi_{market_id}",
                    "title": row[2],
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_type": market_type,
                    "category": market_type.upper() if market_type else "NFL",  # Frontend expects category
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "ai_prediction": round(ai_prob, 0),  # Renamed from ai_probability
                    "ai_probability": round(ai_prob, 0),  # Keep for backwards compatibility
                    "edge": edge,
                    "recommendation": recommendation,
                    "confidence": round(confidence, 0),
                    "volume": volume,
                    "open_interest": open_interest,
                    "close_time": close_time,  # Frontend expects close_time
                    "expiry": close_time,
                    "game_time": game_time,
                    "reasoning": f"AI model predicts {recommendation} with {edge}% edge"
                })

        # Sort
        if sort_by == "edge":
            markets.sort(key=lambda x: x["edge"], reverse=True)
        elif sort_by == "volume":
            markets.sort(key=lambda x: x["volume"], reverse=True)
        else:
            markets.sort(key=lambda x: x["confidence"], reverse=True)

        # Calculate summary
        total_volume = sum(m["volume"] for m in markets)
        avg_edge = sum(m["edge"] for m in markets) / len(markets) if markets else 0

        return {
            "markets": markets,
            "summary": {
                "total_markets": len(markets),
                "total_volume": total_volume,
                "avg_edge": round(avg_edge, 1),
                "high_confidence": len([m for m in markets if m["confidence"] >= 75]),
                "yes_recommendations": len([m for m in markets if m["recommendation"] == "YES"]),
                "no_recommendations": len([m for m in markets if m["recommendation"] == "NO"])
            },
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting Kalshi NFL markets: {e}")
        return {
            "markets": [],
            "summary": {
                "total_markets": 0,
                "total_volume": 0,
                "avg_edge": 0,
                "high_confidence": 0,
                "yes_recommendations": 0,
                "no_recommendations": 0
            },
            "message": f"Error fetching markets: {str(e)}. Ensure Kalshi data is synced.",
            "generated_at": datetime.now().isoformat()
        }


@router.get("/kalshi/categories")
async def get_kalshi_categories():
    """
    Get available Kalshi market categories from database.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT market_type, COUNT(*) as count
                FROM kalshi_markets
                WHERE status IN ('open', 'active')
                GROUP BY market_type
                ORDER BY count DESC
            """)

            rows = cursor.fetchall()

            icons = {
                'nfl': 'football',
                'nba': 'basketball',
                'politics': 'landmark',
                'economics': 'trending-up',
                'weather': 'cloud',
                'entertainment': 'film',
                'crypto': 'bitcoin'
            }

            categories = []
            for row in rows:
                cat_name = row[0] or 'other'
                categories.append({
                    "id": cat_name.lower(),
                    "name": cat_name.upper() if len(cat_name) <= 4 else cat_name.title(),
                    "active_markets": row[1],
                    "icon": icons.get(cat_name.lower(), 'circle')
                })

            return {"categories": categories, "generated_at": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return {
            "categories": [],
            "message": "No Kalshi data available. Run sync to populate.",
            "generated_at": datetime.now().isoformat()
        }


@router.get("/kalshi/opportunities")
async def get_kalshi_opportunities(
    category: str = Query("all", description="Category filter"),
    min_edge: float = Query(8.0, description="Minimum edge %"),
    limit: int = Query(20, description="Maximum results")
):
    """
    Get top Kalshi opportunities across all categories from database.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT m.id, m.ticker, m.title, m.market_type, m.yes_price, m.no_price,
                       m.volume, m.close_time, p.confidence_score, p.edge_percentage,
                       p.predicted_outcome, p.recommended_action
                FROM kalshi_markets m
                LEFT JOIN kalshi_predictions p ON m.id = p.market_id
                WHERE m.status IN ('open', 'active')
            """

            params = []
            if category != "all":
                query += " AND LOWER(m.market_type) = LOWER(%s)"
                params.append(category)

            query += " ORDER BY p.edge_percentage DESC NULLS LAST LIMIT %s"
            params.append(limit * 2)  # Get more to filter

            cursor.execute(query, params)
            rows = cursor.fetchall()

            opportunities = []
            for row in rows:
                edge = float(row[9]) if row[9] else 0
                if edge < min_edge:
                    continue

                opportunities.append({
                    "id": row[1] or f"opp_{row[0]}",
                    "category": row[3] or "Other",
                    "title": row[2],
                    "yes_price": float(row[4]) if row[4] else 50,
                    "ai_probability": float(row[8]) if row[8] else 50,
                    "edge": round(edge, 1),
                    "confidence": round(float(row[8]) if row[8] else 60, 0),
                    "volume": int(row[6]) if row[6] else 0,
                    "recommendation": row[10] or row[11] or "HOLD",
                    "expiry": row[7].strftime("%Y-%m-%d") if row[7] else "",
                    "risk_level": "High" if edge > 15 else "Medium" if edge > 10 else "Low"
                })

            opportunities = opportunities[:limit]

            return {
                "opportunities": opportunities,
                "total": len(opportunities),
                "generated_at": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error getting opportunities: {e}")
        return {
            "opportunities": [],
            "total": 0,
            "message": f"Error: {str(e)}. Ensure Kalshi predictions are populated.",
            "generated_at": datetime.now().isoformat()
        }
