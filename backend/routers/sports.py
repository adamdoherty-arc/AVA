"""
Sports Router - API endpoints for sports betting
NO MOCK DATA - All endpoints use real predictors and database
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging
from datetime import datetime
from pydantic import BaseModel

from backend.services.sports_service import sports_service
from backend.models.market import MarketResponse
from src.prediction_agents.nfl_predictor import NFLPredictor
from src.prediction_agents.nba_predictor import NBAPredictor
from src.prediction_agents.ncaa_predictor import NCAAPredictor
from src.nfl_db_manager import NFLDBManager
from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/sports",
    tags=["sports"]
)

# Initialize predictors (singleton pattern for efficiency)
_nfl_predictor = None
_nba_predictor = None
_ncaa_predictor = None

def get_nfl_predictor():
    global _nfl_predictor
    if _nfl_predictor is None:
        _nfl_predictor = NFLPredictor()
    return _nfl_predictor

def get_nba_predictor():
    global _nba_predictor
    if _nba_predictor is None:
        _nba_predictor = NBAPredictor()
    return _nba_predictor

def get_ncaa_predictor():
    global _ncaa_predictor
    if _ncaa_predictor is None:
        _ncaa_predictor = NCAAPredictor()
    return _ncaa_predictor


@router.get("/markets", response_model=MarketResponse)
async def get_markets(
    market_type: Optional[str] = Query(None, description="Filter by market type (e.g., 'nfl')"),
    limit: int = Query(50, description="Limit number of results")
):
    """
    Get active sports markets with AI predictions from database.
    """
    markets = sports_service.get_markets_with_predictions(market_type, limit)
    return MarketResponse(markets=markets, count=len(markets))


@router.get("/live")
async def get_live_games():
    """
    Get currently live games with real-time scores and odds from database.
    """
    return sports_service.get_live_games()


@router.get("/upcoming")
async def get_upcoming_games(limit: int = 10):
    """
    Get upcoming games with odds from database.
    """
    return sports_service.get_upcoming_games(limit)




@router.get("/games")
async def get_games():
    """
    Get all games (live and upcoming) - combined endpoint.
    """
    live = sports_service.get_live_games()
    upcoming = sports_service.get_upcoming_games(10)
    return {
        "live": live.get("games", []) if isinstance(live, dict) else [],
        "upcoming": upcoming.get("games", []) if isinstance(upcoming, dict) else [],
        "total": len(live.get("games", [])) + len(upcoming.get("games", [])) if isinstance(live, dict) and isinstance(upcoming, dict) else 0
    }

from backend.services.llm_sports_analyzer import LLMSportsAnalyzer

class MatchupRequest(BaseModel):
    home_team: str
    away_team: str
    sport: str
    context_data: dict = {}


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    sport: str


def calculate_expected_value(probability: float, odds: int = -110) -> float:
    """Calculate expected value from win probability and odds."""
    # Convert American odds to decimal
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1

    # EV = (probability * (decimal_odds - 1)) - (1 - probability)
    ev = (probability * (decimal_odds - 1)) - (1 - probability)
    return ev * 100  # Return as percentage


@router.post("/predict")
async def predict_game_endpoint(request: PredictRequest):
    """
    Get AI prediction for a game matchup using prediction agents.
    """
    try:
        sport = request.sport.upper()

        # Get appropriate predictor
        if sport == "NFL":
            predictor = get_nfl_predictor()
        elif sport == "NBA":
            predictor = get_nba_predictor()
        elif sport in ["NCAAF", "NCAAB", "NCAA"]:
            predictor = get_ncaa_predictor()
        else:
            return {"error": f"Unsupported sport: {sport}", "prediction": None}

        # Use predict_winner (the correct method)
        prediction = predictor.predict_winner(
            home_team=request.home_team,
            away_team=request.away_team
        )

        if prediction:
            probability = prediction.get('probability', 0.5)
            confidence_str = prediction.get('confidence', 'medium')
            confidence_map = {'high': 80, 'medium': 65, 'low': 50}
            confidence = confidence_map.get(confidence_str, 60)
            ev = calculate_expected_value(probability, -110)

            return {
                "success": True,
                "prediction": {
                    "pick": prediction.get('winner'),
                    "confidence": confidence,
                    "expected_value": round(ev, 1),
                    "reasoning": prediction.get('explanation', 'AI model prediction'),
                    "bet_type": "Moneyline" if abs(prediction.get('spread', 0)) < 3 else "Spread",
                    "line": f"{prediction.get('spread', 0):+.1f}",
                    "probability": round(probability * 100, 1)
                },
                "matchup": f"{request.away_team} @ {request.home_team}",
                "sport": sport
            }
        else:
            return {
                "success": False,
                "prediction": None,
                "error": "No prediction available",
                "matchup": f"{request.away_team} @ {request.home_team}",
                "sport": sport
            }
    except Exception as e:
        logger.error(f"Error predicting game: {e}")
        return {"success": False, "error": str(e), "prediction": None}


@router.post("/analyze")
async def analyze_matchup(request: MatchupRequest):
    """
    Analyze a sports matchup using Local LLM with real prediction models.
    """
    try:
        analyzer = LLMSportsAnalyzer()
        return analyzer.analyze_matchup(
            home_team=request.home_team,
            away_team=request.away_team,
            sport=request.sport,
            context_data=request.context_data
        )
    except Exception as e:
        logger.error(f"Error analyzing matchup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-bets")
async def get_best_bets(
    sports: str = Query("NFL,NBA,NCAAF,NCAAB", description="Comma-separated sports"),
    min_ev: float = Query(5.0, description="Minimum expected value %"),
    min_confidence: float = Query(60.0, description="Minimum confidence %"),
    sort_by: str = Query("ev", description="Sort by: ev, confidence, or combined"),
    limit: int = Query(50, description="Maximum results")
):
    """Alias route for best-bets - used by SportsBettingHub page"""
    return await get_best_bets_unified(sports, min_ev, min_confidence, sort_by, limit)


@router.get("/best-bets/unified")
async def get_best_bets_unified(
    sports: str = Query("NFL,NBA,NCAAF,NCAAB", description="Comma-separated sports"),
    min_ev: float = Query(0.0, description="Minimum expected value %"),
    min_confidence: float = Query(50.0, description="Minimum confidence %"),
    sort_by: str = Query("ev", description="Sort by: ev, confidence, or combined"),
    limit: int = Query(50, description="Maximum results")
):
    """
    Get unified best bets across all sports using real AI predictors.
    Ranks opportunities by profitability using prediction models.
    """
    try:
        sport_list = [s.strip().upper() for s in sports.split(',')]
        nfl_db = NFLDBManager()

        bets = []

        # Process NFL games
        if "NFL" in sport_list:
            try:
                predictor = get_nfl_predictor()
                upcoming_games = nfl_db.get_upcoming_games(hours_ahead=168)  # Next 7 days

                for game in upcoming_games:
                    try:
                        home_team = game.get('home_team')
                        away_team = game.get('away_team')

                        # Use predict_winner (the correct method)
                        prediction = predictor.predict_winner(
                            home_team=home_team,
                            away_team=away_team
                        )

                        if prediction:
                            # Extract values from prediction
                            winner = prediction.get('winner')
                            probability = prediction.get('probability', 0.5)
                            spread = prediction.get('spread', 0)
                            confidence_str = prediction.get('confidence', 'medium')

                            # Convert confidence string to numeric
                            confidence_map = {'high': 80, 'medium': 65, 'low': 50}
                            confidence = confidence_map.get(confidence_str, 60)

                            # Calculate EV based on probability and standard odds
                            odds = -110
                            ev = calculate_expected_value(probability, odds)

                            if ev >= min_ev and confidence >= min_confidence:
                                bets.append({
                                    "id": f"NFL_{game.get('game_id')}",
                                    "sport": "NFL",
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "matchup": f"{away_team} @ {home_team}",
                                    "bet_type": "Moneyline" if abs(spread) < 3 else "Spread",
                                    "pick": winner,
                                    "line": f"{spread:+.1f}" if spread else "Pick",
                                    "odds": odds,
                                    "ev": round(ev, 1),
                                    "confidence": round(confidence, 1),
                                    "combined_score": round((ev * 0.4 + confidence * 0.6), 1),
                                    "game_time": game.get('game_time').isoformat() if game.get('game_time') else datetime.now().isoformat(),
                                    "reasoning": prediction.get('explanation', 'AI model prediction'),
                                    "source": "NFL AI Predictor"
                                })
                    except Exception as e:
                        logger.error(f"Error predicting NFL game: {e}")
                        continue
            except Exception as e:
                logger.error(f"Error processing NFL games: {e}")

        # Process NBA games
        if "NBA" in sport_list:
            try:
                predictor = get_nba_predictor()
                # Query NBA games from database
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT game_id, home_team, away_team, game_time, spread_home, over_under
                        FROM nba_games
                        WHERE game_time > NOW() AND game_time < NOW() + INTERVAL '7 days'
                        ORDER BY game_time
                        LIMIT 20
                    """)
                    nba_games = cursor.fetchall()

                    for game in nba_games:
                        try:
                            prediction = predictor.predict_game({
                                'home_team': game[1],
                                'away_team': game[2],
                                'spread': game[4],
                                'total': game[5]
                            })

                            if prediction:
                                ev = prediction.get('expected_value', 0)
                                confidence = prediction.get('confidence', 0) * 100

                                if ev >= min_ev and confidence >= min_confidence:
                                    bets.append({
                                        "id": f"NBA_{game[0]}",
                                        "sport": "NBA",
                                        "home_team": game[1],
                                        "away_team": game[2],
                                        "matchup": f"{game[2]} @ {game[1]}",
                                        "bet_type": prediction.get('bet_type', 'Spread'),
                                        "pick": prediction.get('pick'),
                                        "line": prediction.get('line'),
                                        "odds": prediction.get('odds', -110),
                                        "ev": round(ev, 1),
                                        "confidence": round(confidence, 1),
                                        "combined_score": round((ev * 0.4 + confidence * 0.6), 1),
                                        "game_time": game[3].isoformat() if game[3] else datetime.now().isoformat(),
                                        "reasoning": prediction.get('reasoning', 'AI model prediction'),
                                        "source": "NBA AI Predictor"
                                    })
                        except Exception as e:
                            logger.error(f"Error predicting NBA game: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error processing NBA games: {e}")

        # Process NCAAF/NCAAB games
        if "NCAAF" in sport_list or "NCAAB" in sport_list:
            try:
                predictor = get_ncaa_predictor()
                with get_db_connection() as conn:
                    cursor = conn.cursor()

                    for sport in ["NCAAF", "NCAAB"]:
                        if sport not in sport_list:
                            continue

                        table = "ncaa_football_games" if sport == "NCAAF" else "ncaa_basketball_games"
                        try:
                            cursor.execute(f"""
                                SELECT game_id, home_team, away_team, game_time, spread_home, over_under
                                FROM {table}
                                WHERE game_time > NOW() AND game_time < NOW() + INTERVAL '7 days'
                                ORDER BY game_time
                                LIMIT 15
                            """)
                            ncaa_games = cursor.fetchall()

                            for game in ncaa_games:
                                try:
                                    prediction = predictor.predict_game({
                                        'home_team': game[1],
                                        'away_team': game[2],
                                        'spread': game[4],
                                        'total': game[5],
                                        'sport': sport
                                    })

                                    if prediction:
                                        ev = prediction.get('expected_value', 0)
                                        confidence = prediction.get('confidence', 0) * 100

                                        if ev >= min_ev and confidence >= min_confidence:
                                            bets.append({
                                                "id": f"{sport}_{game[0]}",
                                                "sport": sport,
                                                "home_team": game[1],
                                                "away_team": game[2],
                                                "matchup": f"{game[2]} @ {game[1]}",
                                                "bet_type": prediction.get('bet_type', 'Spread'),
                                                "pick": prediction.get('pick'),
                                                "line": prediction.get('line'),
                                                "odds": prediction.get('odds', -110),
                                                "ev": round(ev, 1),
                                                "confidence": round(confidence, 1),
                                                "combined_score": round((ev * 0.4 + confidence * 0.6), 1),
                                                "game_time": game[3].isoformat() if game[3] else datetime.now().isoformat(),
                                                "reasoning": prediction.get('reasoning', 'AI model prediction'),
                                                "source": f"{sport} AI Predictor"
                                            })
                                except Exception as e:
                                    logger.error(f"Error predicting {sport} game: {e}")
                                    continue
                        except Exception as e:
                            logger.error(f"Error querying {sport} games: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error processing NCAA games: {e}")

        # Sort based on preference
        if sort_by == "ev":
            bets.sort(key=lambda x: x["ev"], reverse=True)
        elif sort_by == "confidence":
            bets.sort(key=lambda x: x["confidence"], reverse=True)
        else:  # combined
            bets.sort(key=lambda x: x["combined_score"], reverse=True)

        # Convert to frontend expected format (opportunities)
        opportunities = []
        for bet in bets[:limit]:
            odds_decimal = 1 + (100 / abs(bet["odds"])) if bet["odds"] < 0 else 1 + (bet["odds"] / 100)
            implied_prob = 1 / odds_decimal
            model_prob = (bet["confidence"] / 100) * 0.8 + implied_prob * 0.2  # Blend

            opportunities.append({
                "id": bet["id"],
                "sport": bet["sport"],
                "event_name": bet.get("matchup", f"{bet['away_team']} @ {bet['home_team']}"),
                "bet_type": bet["bet_type"],
                "selection": bet.get("pick", "N/A"),
                "odds": odds_decimal,
                "implied_probability": round(implied_prob, 4),
                "model_probability": round(model_prob, 4),
                "ev_percentage": bet["ev"],
                "confidence": bet["confidence"],
                "overall_score": bet["combined_score"],
                "book": "Best Odds",
                "updated_at": datetime.now().isoformat(),
                "game_time": bet["game_time"],
                "home_team": bet["home_team"],
                "away_team": bet["away_team"]
            })

        # Calculate sport summary
        sport_summary = {}
        for opp in opportunities:
            sport = opp["sport"]
            sport_summary[sport] = sport_summary.get(sport, 0) + 1

        # Calculate averages
        avg_ev = sum(o["ev_percentage"] for o in opportunities) / len(opportunities) if opportunities else 0
        avg_confidence = sum(o["confidence"] for o in opportunities) / len(opportunities) if opportunities else 0

        return {
            "opportunities": opportunities,
            "sport_summary": sport_summary,
            "avg_ev": round(avg_ev, 2),
            "avg_confidence": round(avg_confidence, 1),
            "total": len(opportunities),
            "sports_included": sport_list,
            "message": "Predictions from real AI models" if opportunities else "No games found matching criteria.",
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in best bets unified: {e}")
        return {
            "opportunities": [],
            "sport_summary": {},
            "avg_ev": 0,
            "avg_confidence": 0,
            "total": 0,
            "sports_included": sport_list if 'sport_list' in locals() else [],
            "message": f"Error fetching predictions: {str(e)}. Ensure database is populated with games.",
            "generated_at": datetime.now().isoformat()
        }
