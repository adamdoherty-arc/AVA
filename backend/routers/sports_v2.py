"""
Sports Router V2 - Modern Async Implementation
===============================================

Modern sports betting API endpoints using:
- True async database operations (asyncpg)
- Distributed Redis caching
- AI ensemble predictions
- Type-safe data models

Author: AVA Trading Platform
Updated: 2025-11-30
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

from backend.services.sports_data_access import (
    SportsDataAccess,
    Sport,
    GameStatus,
    Game,
    get_sports_data_access,
    sports_data_dependency,
)
from backend.services.ai_sports_predictor import (
    AISportsPredictor,
    Prediction,
    Confidence,
    BetRecommendation,
    get_ai_predictor,
    predict_game,
    get_best_bets as get_best_bets_from_predictor,
)
from backend.infrastructure.cache import get_cache
from backend.infrastructure.errors import safe_internal_error

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/sports/v2",
    tags=["sports-v2"]
)


# =============================================================================
# Response Models
# =============================================================================

class GameResponse(BaseModel):
    """Game response model."""
    game_id: str
    sport: str
    home_team: str
    away_team: str
    home_team_abbr: Optional[str] = None
    away_team_abbr: Optional[str] = None
    game_time: Optional[str] = None
    game_status: str
    venue: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    is_live: bool = False
    period: Optional[int] = None
    time_remaining: Optional[str] = None
    home_rank: Optional[int] = None
    away_rank: Optional[int] = None
    odds: Optional[dict] = None


class PredictionResponse(BaseModel):
    """Prediction response model."""
    game_id: str
    sport: str
    home_team: str
    away_team: str
    winner: str
    win_probability: float
    confidence: str
    confidence_score: float
    predicted_spread: float
    expected_value: float
    edge_vs_market: float
    kelly_fraction: float
    recommendation: str
    model_agreement: float
    models_used: List[str]
    reasoning: str
    key_factors: List[str]


class BestBetsResponse(BaseModel):
    """Best bets response model."""
    opportunities: List[dict]
    total: int
    sports_included: List[str]
    avg_ev: float
    avg_confidence: float
    sport_summary: dict
    generated_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: dict
    cache: dict
    timestamp: str


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of sports betting services.

    Returns database pool stats, cache stats, and overall status.
    """
    try:
        from backend.infrastructure.database import get_database

        db = await get_database()
        db_stats = db.get_stats()

        cache = get_cache()
        cache_stats = cache.get_stats()

        return HealthResponse(
            status="healthy",
            database=db_stats,
            cache=cache_stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database={"error": str(e)},
            cache={"error": "Unknown"},
            timestamp=datetime.now().isoformat()
        )


def _parse_sports(sports_str: str) -> List[Sport]:
    """Parse comma-separated sports string into Sport enums with validation."""
    sport_list = []
    for s in sports_str.split(","):
        s = s.strip().upper()
        if not s:
            continue
        try:
            sport_list.append(Sport(s))
        except ValueError:
            # Skip invalid sports, log warning
            logger.warning(f"Invalid sport ignored: {s}")
    return sport_list if sport_list else [Sport.NFL, Sport.NBA, Sport.NCAAF, Sport.NCAAB]


@router.get("/games/live", response_model=List[GameResponse])
async def get_live_games(
    sports: str = Query("NFL,NBA,NCAAF,NCAAB", description="Comma-separated sports"),
    data_access: SportsDataAccess = Depends(sports_data_dependency)
):
    """
    Get all currently live games across sports.

    Uses UNION query for efficiency and Redis caching (30s TTL).
    """
    try:
        sport_list = _parse_sports(sports)
        games = await data_access.get_live_games(sport_list)

        return [_game_to_response(game) for game in games]
    except Exception as e:
        logger.error(f"Error fetching live games: {e}")
        safe_internal_error(e, "fetch live games")


@router.get("/games/upcoming", response_model=List[GameResponse])
async def get_upcoming_games(
    sports: str = Query("NFL,NBA,NCAAF,NCAAB", description="Comma-separated sports"),
    limit: int = Query(50, ge=1, le=200),
    hours_ahead: int = Query(168, ge=1, le=336),  # Default 1 week, max 2 weeks
    data_access: SportsDataAccess = Depends(sports_data_dependency)
):
    """
    Get upcoming scheduled games.

    Results are cached for 5 minutes.
    """
    try:
        sport_list = _parse_sports(sports)
        games = await data_access.get_upcoming_games(sport_list, limit, hours_ahead)

        return [_game_to_response(game) for game in games]
    except Exception as e:
        logger.error(f"Error fetching upcoming games: {e}")
        safe_internal_error(e, "fetch upcoming games")


@router.get("/games/with-odds", response_model=List[GameResponse])
async def get_games_with_odds(
    sports: str = Query("NFL,NBA,NCAAF,NCAAB", description="Comma-separated sports"),
    data_access: SportsDataAccess = Depends(sports_data_dependency)
):
    """
    Get games that have betting odds available.

    Returns scheduled and live games with odds data.
    """
    try:
        sport_list = _parse_sports(sports)
        games = await data_access.get_games_with_odds(sport_list)

        return [_game_to_response(game) for game in games]
    except Exception as e:
        logger.error(f"Error fetching games with odds: {e}")
        safe_internal_error(e, "fetch games with odds")


@router.get("/predict/{sport}/{game_id}", response_model=PredictionResponse)
async def get_prediction(
    sport: str,
    game_id: str,
    data_access: SportsDataAccess = Depends(sports_data_dependency)
):
    """
    Get AI prediction for a specific game.

    Uses ensemble model combining Elo, momentum, situational, and market models.
    Results are cached for 10 minutes.
    """
    try:
        sport_enum = Sport(sport.upper())
        game = await data_access.get_game_by_id(game_id, sport_enum)

        if not game:
            raise HTTPException(status_code=404, detail=f"Game not found: {game_id}")

        prediction = await predict_game(game)
        return _prediction_to_response(prediction)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        safe_internal_error(e, "get game prediction")


@router.get("/best-bets", response_model=BestBetsResponse)
async def get_best_bets_v2(
    sports: str = Query("NFL,NBA,NCAAF,NCAAB", description="Comma-separated sports"),
    min_ev: float = Query(0.0, ge=-50, le=100, description="Minimum expected value %"),
    min_edge: float = Query(2.0, ge=0, le=50, description="Minimum edge vs market %"),
    min_confidence: str = Query("low", description="Minimum confidence: low, medium, high"),
    sort_by: str = Query("ev", description="Sort by: ev, edge, confidence, combined"),
    limit: int = Query(50, ge=1, le=200),
    data_access: SportsDataAccess = Depends(sports_data_dependency)
):
    """
    Get best betting opportunities across sports.

    Uses AI ensemble predictions with Kelly Criterion sizing.
    Filters by EV, edge, and confidence level.
    """
    try:
        sport_list = _parse_sports(sports)

        # Get all games with odds
        games = await data_access.get_games_with_odds(sport_list)

        if not games:
            return BestBetsResponse(
                opportunities=[],
                total=0,
                sports_included=[s.value for s in sport_list],
                avg_ev=0,
                avg_confidence=0,
                sport_summary={},
                generated_at=datetime.now().isoformat()
            )

        # Get AI predictions for all games
        predictor = get_ai_predictor()

        # Map confidence string to enum
        conf_map = {
            "low": Confidence.LOW,
            "medium": Confidence.MEDIUM,
            "high": Confidence.HIGH
        }
        min_conf = conf_map.get(min_confidence.lower(), Confidence.LOW)

        # Get best bets
        best_bets = await predictor.get_best_bets(
            games,
            min_edge=min_edge,
            min_confidence=min_conf,
            max_results=limit * 2  # Get extra to filter by EV
        )

        # Filter by minimum EV
        filtered_bets = [b for b in best_bets if b.expected_value >= min_ev]

        # Sort
        if sort_by == "ev":
            filtered_bets.sort(key=lambda x: x.expected_value, reverse=True)
        elif sort_by == "edge":
            filtered_bets.sort(key=lambda x: x.edge_vs_market, reverse=True)
        elif sort_by == "confidence":
            filtered_bets.sort(key=lambda x: x.confidence_score, reverse=True)
        else:  # combined
            filtered_bets.sort(
                key=lambda x: x.expected_value * 0.4 + x.confidence_score * 100 * 0.6,
                reverse=True
            )

        # Limit results
        final_bets = filtered_bets[:limit]

        # Convert to response format
        opportunities = []
        for pred in final_bets:
            opportunities.append({
                "id": f"{pred.sport.value}_{pred.game_id}",
                "sport": pred.sport.value,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "event_name": f"{pred.away_team} @ {pred.home_team}",
                "bet_type": "Moneyline" if abs(pred.predicted_spread) < 3 else "Spread",
                "selection": pred.winner,
                "model_probability": pred.win_probability,
                "confidence": pred.confidence.value,
                "confidence_score": pred.confidence_score,
                "ev_percentage": pred.expected_value,
                "ai_edge": pred.edge_vs_market,
                "kelly_fraction": pred.kelly_fraction,
                "recommendation": pred.recommendation.value,
                "reasoning": pred.reasoning,
                "key_factors": pred.key_factors,
                "model_agreement": pred.model_agreement,
                "predicted_spread": pred.predicted_spread,
                "overall_score": pred.expected_value * 0.4 + pred.confidence_score * 100 * 0.6,
            })

        # Calculate stats
        sport_summary = {}
        for opp in opportunities:
            sport = opp["sport"]
            sport_summary[sport] = sport_summary.get(sport, 0) + 1

        avg_ev = sum(o["ev_percentage"] for o in opportunities) / len(opportunities) if opportunities else 0
        avg_conf = sum(o["confidence_score"] for o in opportunities) / len(opportunities) if opportunities else 0

        return BestBetsResponse(
            opportunities=opportunities,
            total=len(opportunities),
            sports_included=[s.value for s in sport_list],
            avg_ev=round(avg_ev, 2),
            avg_confidence=round(avg_conf * 100, 1),
            sport_summary=sport_summary,
            generated_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error getting best bets: {e}")
        safe_internal_error(e, "get best bets")


@router.get("/stats")
async def get_stats(
    data_access: SportsDataAccess = Depends(sports_data_dependency)
):
    """
    Get game statistics by sport and status.
    """
    try:
        counts = await data_access.get_game_counts()

        return {
            "game_counts": counts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        safe_internal_error(e, "get game stats")


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all sports betting cache.

    Useful after data sync or for debugging.
    """
    try:
        cache = get_cache()
        await cache.invalidate_pattern("*games*")
        await cache.invalidate_pattern("*prediction*")
        await cache.invalidate_pattern("live_games*")
        await cache.invalidate_pattern("upcoming_games*")

        return {
            "success": True,
            "message": "Sports cache cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        safe_internal_error(e, "clear sports cache")


# =============================================================================
# Helper Functions
# =============================================================================

def _game_to_response(game: Game) -> GameResponse:
    """Convert Game dataclass to response model."""
    odds_dict = None
    if game.odds:
        odds_dict = {
            "moneyline_home": game.odds.moneyline_home,
            "moneyline_away": game.odds.moneyline_away,
            "spread_home": game.odds.spread_home,
            "over_under": game.odds.over_under,
        }

    return GameResponse(
        game_id=game.game_id,
        sport=game.sport.value,
        home_team=game.home_team,
        away_team=game.away_team,
        home_team_abbr=game.home_team_abbr,
        away_team_abbr=game.away_team_abbr,
        game_time=game.game_time.isoformat() if game.game_time else None,
        game_status=game.game_status.value,
        venue=game.venue,
        home_score=game.home_score,
        away_score=game.away_score,
        is_live=game.is_live,
        period=game.period,
        time_remaining=game.time_remaining,
        home_rank=game.home_rank,
        away_rank=game.away_rank,
        odds=odds_dict,
    )


def _prediction_to_response(prediction: Prediction) -> PredictionResponse:
    """Convert Prediction dataclass to response model."""
    return PredictionResponse(
        game_id=prediction.game_id,
        sport=prediction.sport.value,
        home_team=prediction.home_team,
        away_team=prediction.away_team,
        winner=prediction.winner,
        win_probability=prediction.win_probability,
        confidence=prediction.confidence.value,
        confidence_score=prediction.confidence_score,
        predicted_spread=prediction.predicted_spread,
        expected_value=prediction.expected_value,
        edge_vs_market=prediction.edge_vs_market,
        kelly_fraction=prediction.kelly_fraction,
        recommendation=prediction.recommendation.value,
        model_agreement=prediction.model_agreement,
        models_used=prediction.models_used,
        reasoning=prediction.reasoning,
        key_factors=prediction.key_factors,
    )
