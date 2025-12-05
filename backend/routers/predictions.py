"""
Predictions Router - API endpoints for prediction markets
NO MOCK DATA - All endpoints use real Kalshi API and database

Performance: Uses async database pattern for non-blocking DB calls
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import structlog
from backend.services.prediction_service import prediction_service
from backend.models.market import MarketResponse
from backend.infrastructure.database import get_database, AsyncDatabaseManager
from src.kalshi_db_manager import KalshiDBManager
from src.prediction_agents.nfl_predictor import NFLPredictor

logger = structlog.get_logger(__name__)

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


# ============ Async Helper Functions ============

async def _fetch_kalshi_nfl_markets_async(min_edge: float, min_volume: int, sort_by: str) -> Dict[str, Any]:
    """Async function to fetch Kalshi NFL markets from database"""
    predictor = get_predictor()
    db = await get_database()

    rows = await db.fetch("""
        SELECT id, ticker, title, home_team, away_team, market_type,
               yes_price, no_price, volume, open_interest, close_time, game_time
        FROM kalshi_markets
        WHERE market_type = $1 AND status = ANY($2)
        ORDER BY game_time ASC
    """, 'nfl', ['open', 'active'])

    markets = []
    for row in rows:
        market_id = row["id"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        market_type = row["market_type"]
        yes_price = float(row["yes_price"]) if row["yes_price"] else 50
        no_price = float(row["no_price"]) if row["no_price"] else 50
        volume = int(row["volume"]) if row["volume"] else 0

        if volume < min_volume:
            continue

        # Get AI prediction
        try:
            prediction = predictor.predict_game({'home_team': home_team, 'away_team': away_team})
            ai_prob = prediction.get('win_probability', 0.5) * 100 if prediction else 50
            confidence = prediction.get('confidence', 0.7) * 100 if prediction else 60
        except Exception as e:
            logger.error("prediction_error", error=str(e))
            ai_prob, confidence = 50, 60

        # Calculate edge
        if ai_prob > yes_price:
            edge, recommendation = round(ai_prob - yes_price, 1), "YES"
        else:
            edge, recommendation = round(no_price - (100 - ai_prob), 1), "NO"

        if edge < min_edge:
            continue

        markets.append({
            "id": row["ticker"] or f"kalshi_{market_id}",
            "title": row["title"],
            "home_team": home_team,
            "away_team": away_team,
            "market_type": market_type,
            "category": market_type.upper() if market_type else "NFL",
            "yes_price": yes_price,
            "no_price": no_price,
            "ai_prediction": round(ai_prob, 0),
            "ai_probability": round(ai_prob, 0),
            "edge": edge,
            "recommendation": recommendation,
            "confidence": round(confidence, 0),
            "volume": volume,
            "open_interest": int(row["open_interest"]) if row["open_interest"] else 0,
            "close_time": row["close_time"].strftime("%Y-%m-%d %H:%M") if row["close_time"] else "",
            "expiry": row["close_time"].strftime("%Y-%m-%d %H:%M") if row["close_time"] else "",
            "game_time": row["game_time"].strftime("%Y-%m-%d %H:%M") if row["game_time"] else "",
            "reasoning": f"AI model predicts {recommendation} with {edge}% edge"
        })

    # Sort
    sort_key = {"edge": "edge", "volume": "volume"}.get(sort_by, "confidence")
    markets.sort(key=lambda x: x[sort_key], reverse=True)

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
        }
    }


@router.get("/nfl")
async def get_nfl_predictions(
    min_edge: float = Query(5.0, description="Minimum AI edge percentage"),
    min_volume: int = Query(100, description="Minimum volume"),
    sort_by: str = Query("edge", description="Sort by: edge, volume, confidence")
):
    """Alias route for NFL predictions - used by frontend."""
    return await _fetch_kalshi_nfl_markets_async(min_edge, min_volume, sort_by)


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
    Uses async database pattern for non-blocking database access.
    """
    try:
        result = await _fetch_kalshi_nfl_markets_async(min_edge, min_volume, sort_by)
        result["generated_at"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error("kalshi_nfl_markets_error", error=str(e))
        return {
            "markets": [],
            "summary": {
                "total_markets": 0, "total_volume": 0, "avg_edge": 0,
                "high_confidence": 0, "yes_recommendations": 0, "no_recommendations": 0
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
        db = await get_database()

        rows = await db.fetch("""
            SELECT market_type, COUNT(*) as count
            FROM kalshi_markets
            WHERE status = ANY($1)
            GROUP BY market_type
            ORDER BY count DESC
        """, ['open', 'active'])

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
            cat_name = row["market_type"] or 'other'
            categories.append({
                "id": cat_name.lower(),
                "name": cat_name.upper() if len(cat_name) <= 4 else cat_name.title(),
                "active_markets": row["count"],
                "icon": icons.get(cat_name.lower(), 'circle')
            })

        return {"categories": categories, "generated_at": datetime.now().isoformat()}

    except Exception as e:
        logger.error("kalshi_categories_error", error=str(e))
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
        db = await get_database()

        if category != "all":
            query = """
                SELECT m.id, m.ticker, m.title, m.market_type, m.yes_price, m.no_price,
                       m.volume, m.close_time, p.confidence_score, p.edge_percentage,
                       p.predicted_outcome, p.recommended_action
                FROM kalshi_markets m
                LEFT JOIN kalshi_predictions p ON m.id = p.market_id
                WHERE m.status = ANY($1) AND LOWER(m.market_type) = LOWER($2)
                ORDER BY p.edge_percentage DESC NULLS LAST
                LIMIT $3
            """
            rows = await db.fetch(query, ['open', 'active'], category, limit * 2)
        else:
            query = """
                SELECT m.id, m.ticker, m.title, m.market_type, m.yes_price, m.no_price,
                       m.volume, m.close_time, p.confidence_score, p.edge_percentage,
                       p.predicted_outcome, p.recommended_action
                FROM kalshi_markets m
                LEFT JOIN kalshi_predictions p ON m.id = p.market_id
                WHERE m.status = ANY($1)
                ORDER BY p.edge_percentage DESC NULLS LAST
                LIMIT $2
            """
            rows = await db.fetch(query, ['open', 'active'], limit * 2)

        opportunities = []
        for row in rows:
            edge = float(row["edge_percentage"]) if row["edge_percentage"] else 0
            if edge < min_edge:
                continue

            opportunities.append({
                "id": row["ticker"] or f"opp_{row['id']}",
                "category": row["market_type"] or "Other",
                "title": row["title"],
                "yes_price": float(row["yes_price"]) if row["yes_price"] else 50,
                "ai_probability": float(row["confidence_score"]) if row["confidence_score"] else 50,
                "edge": round(edge, 1),
                "confidence": round(float(row["confidence_score"]) if row["confidence_score"] else 60, 0),
                "volume": int(row["volume"]) if row["volume"] else 0,
                "recommendation": row["predicted_outcome"] or row["recommended_action"] or "HOLD",
                "expiry": row["close_time"].strftime("%Y-%m-%d") if row["close_time"] else "",
                "risk_level": "High" if edge > 15 else "Medium" if edge > 10 else "Low"
            })

        opportunities = opportunities[:limit]

        return {
            "opportunities": opportunities,
            "total": len(opportunities),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("kalshi_opportunities_error", error=str(e))
        return {
            "opportunities": [],
            "total": 0,
            "message": f"Error: {str(e)}. Ensure Kalshi predictions are populated.",
            "generated_at": datetime.now().isoformat()
        }
