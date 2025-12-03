"""
Sports Router - API endpoints for sports betting
NO MOCK DATA - All endpoints use real predictors and database

OPTIMIZATIONS APPLIED:
1. Async-wrapped sync database calls using asyncio.to_thread()
2. Prevents event loop blocking during DB queries
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging
import asyncio
from datetime import datetime
from pydantic import BaseModel

from backend.services.sports_service import (
    sports_service,
    get_nfl_predictor,  # Reuse singleton predictors from service
    get_nba_predictor,
    get_ncaa_predictor
)
from backend.config import settings
from src.telegram_notifier import TelegramNotifier

# Initialize Telegram notifier singleton
_telegram_notifier = None

def get_telegram_notifier(force_reload: bool = False):
    """Get Telegram notifier, optionally forcing a reload of config."""
    global _telegram_notifier
    if _telegram_notifier is None or force_reload:
        # Always create fresh to pick up latest .env values
        _telegram_notifier = TelegramNotifier()
    return _telegram_notifier
from backend.models.market import MarketResponse
from src.nfl_db_manager import NFLDBManager
from src.database.connection_pool import get_db_connection
from src.espn_live_data import ESPNLiveData
from src.espn_nba_live_data import ESPNNBALiveData
from src.espn_ncaa_live_data import ESPNNCAALiveData
from src.kalshi_public_client import KalshiPublicClient
from src.espn_kalshi_matcher import ESPNKalshiMatcher

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/sports",
    tags=["sports"]
)

# NOTE: Predictor singletons are now imported from sports_service to avoid duplication


# ============ SAFE TABLE MAPPING (prevents SQL injection) ============
# Explicit whitelist of valid sport -> table mappings
SPORT_TABLE_MAP = {
    'NFL': 'nfl_games',
    'NBA': 'nba_games',
    'NCAAF': 'ncaa_football_games',
    'NCAAB': 'ncaa_basketball_games',
}


def _fetch_all_upcoming_games_unified(sport_list: list, limit: int = 50) -> list:
    """
    OPTIMIZED: Fetch all upcoming games for multiple sports using UNION ALL.
    Reduces N+1 queries to a single query.

    Args:
        sport_list: List of sports like ['NFL', 'NBA', 'NCAAF', 'NCAAB']
        limit: Maximum games per sport

    Returns:
        List of game dicts with 'sport' field included
    """
    games = []

    # Build UNION ALL query dynamically using safe table mapping
    union_parts = []

    for sport in sport_list:
        sport_upper = sport.upper()
        table = SPORT_TABLE_MAP.get(sport_upper)
        if not table:
            continue

        # Each sport uses a safe, hardcoded table name from the whitelist
        if sport_upper in ['NFL', 'NBA']:
            union_parts.append(f"""
                SELECT game_id, '{sport_upper}' as sport, home_team, away_team, game_time,
                       spread_home, over_under, moneyline_home, moneyline_away,
                       NULL::integer as home_rank, NULL::integer as away_rank
                FROM {table}
                WHERE game_time > NOW() AND game_time < NOW() + INTERVAL '7 days'
                  AND game_status = 'scheduled'
                ORDER BY game_time
                LIMIT {int(limit)}
            """)
        else:  # NCAA sports have ranking columns
            union_parts.append(f"""
                SELECT game_id, '{sport_upper}' as sport, home_team, away_team, game_time,
                       spread_home, over_under, moneyline_home, moneyline_away,
                       home_rank, away_rank
                FROM {table}
                WHERE game_time > NOW() AND game_time < NOW() + INTERVAL '7 days'
                  AND game_status = 'scheduled'
                ORDER BY game_time
                LIMIT {int(limit)}
            """)

    if not union_parts:
        return []

    # Combine with UNION ALL for single-query execution
    full_query = " UNION ALL ".join(union_parts) + " ORDER BY game_time LIMIT %s"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(full_query, (limit * len(sport_list),))
        rows = cursor.fetchall()

        for row in rows:
            games.append({
                'game_id': row[0],
                'sport': row[1],
                'home_team': row[2],
                'away_team': row[3],
                'game_time': row[4],
                'spread_home': row[5],
                'over_under': row[6],
                'moneyline_home': row[7],
                'moneyline_away': row[8],
                'home_rank': row[9],
                'away_rank': row[10],
            })

    return games


def _batch_predict_all_games(games: list) -> dict:
    """
    OPTIMIZED: Run predictions for all games using batch methods.
    Groups games by sport and uses each predictor's batch method.

    Args:
        games: List of game dicts from _fetch_all_upcoming_games_unified

    Returns:
        Dict mapping game_id to prediction result
    """
    predictions = {}

    # Group games by sport
    games_by_sport = {}
    for game in games:
        sport = game.get('sport', '').upper()
        if sport not in games_by_sport:
            games_by_sport[sport] = []
        games_by_sport[sport].append(game)

    # Process NFL games with batch method
    if 'NFL' in games_by_sport:
        try:
            predictor = get_nfl_predictor()
            for game in games_by_sport['NFL']:
                try:
                    pred = predictor.predict_winner(
                        home_team=game['home_team'],
                        away_team=game['away_team']
                    )
                    if pred:
                        predictions[game['game_id']] = pred
                except Exception as e:
                    logger.warning(f"NFL prediction failed for {game['game_id']}: {e}")
        except Exception as e:
            logger.error(f"NFL predictor error: {e}")

    # Process NBA games with batch method
    if 'NBA' in games_by_sport:
        try:
            predictor = get_nba_predictor()
            # Use batch predict for efficiency
            nba_results = predictor.predict_batch(games_by_sport['NBA'])
            predictions.update(nba_results)
        except Exception as e:
            logger.error(f"NBA batch prediction error: {e}")

    # Process NCAA games
    if 'NCAAF' in games_by_sport or 'NCAAB' in games_by_sport:
        try:
            predictor = get_ncaa_predictor()
            for sport in ['NCAAF', 'NCAAB']:
                if sport in games_by_sport:
                    for game in games_by_sport[sport]:
                        try:
                            pred = predictor.predict_winner(
                                home_team=game['home_team'],
                                away_team=game['away_team']
                            )
                            if pred:
                                predictions[game['game_id']] = pred
                        except Exception as e:
                            logger.warning(f"{sport} prediction failed for {game['game_id']}: {e}")
        except Exception as e:
            logger.error(f"NCAA predictor error: {e}")

    return predictions


def _fetch_best_bets_optimized_sync(
    sport_list: list,
    min_ev: float,
    min_confidence: float,
    limit: int
) -> list:
    """
    OPTIMIZED: Fetch and predict best bets using single query + batch predictions.
    Replaces the N+1 pattern in get_best_bets_unified endpoint.

    Args:
        sport_list: List of sports like ['NFL', 'NBA', 'NCAAF', 'NCAAB']
        min_ev: Minimum expected value threshold
        min_confidence: Minimum confidence threshold
        limit: Maximum games per sport

    Returns:
        List of bet dictionaries with predictions
    """
    bets = []
    confidence_map = {'low': 55, 'medium': 70, 'high': 85}

    # Step 1: Single UNION ALL query to fetch all games
    games = _fetch_all_upcoming_games_unified(sport_list, limit=limit)

    if not games:
        return []

    # Step 2: Batch predictions grouped by sport
    predictions = _batch_predict_all_games(games)

    # Step 3: Process games with predictions into bet format
    for game in games:
        game_id = game.get('game_id')
        sport = game.get('sport', 'UNK')
        prediction = predictions.get(game_id)

        if not prediction:
            continue

        # Extract prediction values
        winner = prediction.get('winner')
        # Ensure probability is 0-1 range (not percentage)
        probability = prediction.get('probability', 0.5)
        if probability > 1:
            probability = probability / 100.0

        spread = prediction.get('spread', 0)
        conf_text = prediction.get('confidence', 'medium')
        confidence = confidence_map.get(conf_text, 70) if isinstance(conf_text, str) else conf_text

        # Get moneyline odds for predicted winner
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        if winner and winner.lower() == home_team.lower():
            odds = game.get('moneyline_home') or -110
        else:
            odds = game.get('moneyline_away') or -110

        odds = int(odds) if odds else -110

        # Calculate EV
        ev = calculate_expected_value(probability, odds)

        # Apply filters
        if ev < min_ev or confidence < min_confidence:
            continue

        # Get spread from database if available
        db_spread = game.get('spread_home')
        display_spread = db_spread if db_spread is not None else spread

        bets.append({
            "id": f"{sport}_{game_id}",
            "sport": sport,
            "home_team": home_team,
            "away_team": away_team,
            "matchup": f"{away_team} @ {home_team}",
            "bet_type": "Moneyline" if abs(display_spread or 0) < 3 else "Spread",
            "pick": winner,
            "line": f"{display_spread:+.1f}" if display_spread else "Pick",
            "odds": odds,
            "odds_home": game.get('moneyline_home'),
            "odds_away": game.get('moneyline_away'),
            "spread": db_spread,
            "over_under": game.get('over_under'),
            "ev": round(ev, 1),
            "confidence": round(confidence, 1),
            "combined_score": round((ev * 0.4 + confidence * 0.6), 1),
            "game_time": game.get('game_time').isoformat() if game.get('game_time') else datetime.now().isoformat(),
            "reasoning": prediction.get('explanation', 'AI model prediction')[:200] if prediction.get('explanation') else 'AI model prediction',
            "source": f"{sport} AI Predictor (Optimized)",
            "home_rank": game.get('home_rank'),
            "away_rank": game.get('away_rank'),
        })

    return bets


# ============ Sync Helper Functions (run via asyncio.to_thread) ============

def _fetch_all_live_games_sync():
    """Sync function to fetch all live games - called via asyncio.to_thread()"""
    live_games = []
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get live NFL games
        cursor.execute("""
            SELECT game_id, 'NFL' as sport, home_team, away_team, home_team_abbr, away_team_abbr,
                   home_score, away_score, quarter as period, time_remaining, game_status, last_synced
            FROM nfl_games WHERE is_live = true
        """)
        for row in cursor.fetchall():
            live_games.append({
                "game_id": row[0], "sport": row[1], "home_team": row[2], "away_team": row[3],
                "home_abbr": row[4], "away_abbr": row[5], "home_score": row[6], "away_score": row[7],
                "period": row[8], "time_remaining": row[9], "status": row[10],
                "last_synced": row[11].isoformat() if row[11] else None
            })

        # Get live NBA games
        cursor.execute("""
            SELECT game_id, 'NBA' as sport, home_team, away_team, home_team_abbr, away_team_abbr,
                   home_score, away_score, quarter as period, time_remaining, game_status, last_synced
            FROM nba_games WHERE is_live = true
        """)
        for row in cursor.fetchall():
            live_games.append({
                "game_id": row[0], "sport": row[1], "home_team": row[2], "away_team": row[3],
                "home_abbr": row[4], "away_abbr": row[5], "home_score": row[6], "away_score": row[7],
                "period": row[8], "time_remaining": row[9], "status": row[10],
                "last_synced": row[11].isoformat() if row[11] else None
            })

        # Get live NCAAF games
        cursor.execute("""
            SELECT game_id, 'NCAAF' as sport, home_team, away_team, home_team_abbr, away_team_abbr,
                   home_score, away_score, quarter as period, time_remaining, game_status, last_synced,
                   home_rank, away_rank
            FROM ncaa_football_games WHERE is_live = true
        """)
        for row in cursor.fetchall():
            live_games.append({
                "game_id": row[0], "sport": row[1], "home_team": row[2], "away_team": row[3],
                "home_abbr": row[4], "away_abbr": row[5], "home_score": row[6], "away_score": row[7],
                "period": row[8], "time_remaining": row[9], "status": row[10],
                "last_synced": row[11].isoformat() if row[11] else None,
                "home_rank": row[12], "away_rank": row[13]
            })

        # Get live NCAAB games
        cursor.execute("""
            SELECT game_id, 'NCAAB' as sport, home_team, away_team, home_team_abbr, away_team_abbr,
                   home_score, away_score, half as period, time_remaining, game_status, last_synced,
                   home_rank, away_rank
            FROM ncaa_basketball_games WHERE is_live = true
        """)
        for row in cursor.fetchall():
            live_games.append({
                "game_id": row[0], "sport": row[1], "home_team": row[2], "away_team": row[3],
                "home_abbr": row[4], "away_abbr": row[5], "home_score": row[6], "away_score": row[7],
                "period": row[8], "time_remaining": row[9], "status": row[10],
                "last_synced": row[11].isoformat() if row[11] else None,
                "home_rank": row[12], "away_rank": row[13]
            })

    return {"live_games": live_games, "count": len(live_games), "timestamp": datetime.now().isoformat()}


def _fetch_games_with_odds_sync(sport: str):
    """Sync function to fetch games with odds - called via asyncio.to_thread()"""
    games = []
    sport = sport.upper()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        if sport in ["NFL", "ALL"]:
            cursor.execute("""
                SELECT g.game_id, 'NFL' as sport, g.home_team, g.away_team,
                       g.home_team_abbr, g.away_team_abbr, g.home_score, g.away_score,
                       g.quarter, g.time_remaining, g.game_status, g.is_live,
                       g.game_time, g.venue, g.moneyline_home, g.moneyline_away,
                       g.spread_home, g.over_under
                FROM nfl_games g WHERE g.game_status IN ('scheduled', 'live', 'final')
                ORDER BY g.game_time LIMIT 20
            """)
            for row in cursor.fetchall():
                games.append({
                    "game_id": row[0], "sport": row[1], "home_team": row[2], "away_team": row[3],
                    "home_abbr": row[4], "away_abbr": row[5], "home_score": row[6], "away_score": row[7],
                    "period": row[8], "time_remaining": row[9], "status": row[10], "is_live": row[11],
                    "game_time": row[12].isoformat() if row[12] else None, "venue": row[13],
                    "odds": {"home_ml": row[14], "away_ml": row[15], "spread": row[16], "total": row[17]}
                })

        if sport in ["NBA", "ALL"]:
            cursor.execute("""
                SELECT g.game_id, 'NBA' as sport, g.home_team, g.away_team,
                       g.home_team_abbr, g.away_team_abbr, g.home_score, g.away_score,
                       g.quarter, g.time_remaining, g.game_status, g.is_live,
                       g.game_time, g.venue, g.moneyline_home, g.moneyline_away,
                       g.spread_home, g.over_under
                FROM nba_games g WHERE g.game_status IN ('scheduled', 'live', 'final')
                ORDER BY g.game_time LIMIT 20
            """)
            for row in cursor.fetchall():
                games.append({
                    "game_id": row[0], "sport": row[1], "home_team": row[2], "away_team": row[3],
                    "home_abbr": row[4], "away_abbr": row[5], "home_score": row[6], "away_score": row[7],
                    "period": row[8], "time_remaining": row[9], "status": row[10], "is_live": row[11],
                    "game_time": row[12].isoformat() if row[12] else None, "venue": row[13],
                    "odds": {"home_ml": row[14], "away_ml": row[15], "spread": row[16], "total": row[17]}
                })

        if sport in ["NCAAF", "ALL"]:
            cursor.execute("""
                SELECT g.game_id, 'NCAAF' as sport, g.home_team, g.away_team,
                       g.home_team_abbr, g.away_team_abbr, g.home_score, g.away_score,
                       g.quarter, g.time_remaining, g.game_status, g.is_live,
                       g.game_time, g.venue, g.spread_home, g.over_under, g.home_rank, g.away_rank
                FROM ncaa_football_games g WHERE g.game_status IN ('scheduled', 'live', 'final')
                ORDER BY g.game_time LIMIT 30
            """)
            for row in cursor.fetchall():
                games.append({
                    "game_id": row[0], "sport": row[1], "home_team": row[2], "away_team": row[3],
                    "home_abbr": row[4], "away_abbr": row[5], "home_score": row[6], "away_score": row[7],
                    "period": row[8], "time_remaining": row[9], "status": row[10], "is_live": row[11],
                    "game_time": row[12].isoformat() if row[12] else None, "venue": row[13],
                    "odds": {"spread": row[14], "total": row[15]}, "home_rank": row[16], "away_rank": row[17]
                })

        if sport in ["NCAAB", "ALL"]:
            cursor.execute("""
                SELECT g.game_id, 'NCAAB' as sport, g.home_team, g.away_team,
                       g.home_team_abbr, g.away_team_abbr, g.home_score, g.away_score,
                       g.half, g.time_remaining, g.game_status, g.is_live,
                       g.game_time, g.venue, g.spread_home, g.over_under, g.home_rank, g.away_rank
                FROM ncaa_basketball_games g WHERE g.game_status IN ('scheduled', 'live', 'final')
                ORDER BY g.game_time LIMIT 20
            """)
            for row in cursor.fetchall():
                games.append({
                    "game_id": row[0], "sport": row[1], "home_team": row[2], "away_team": row[3],
                    "home_abbr": row[4], "away_abbr": row[5], "home_score": row[6], "away_score": row[7],
                    "period": row[8], "time_remaining": row[9], "status": row[10], "is_live": row[11],
                    "game_time": row[12].isoformat() if row[12] else None, "venue": row[13],
                    "odds": {"spread": row[14], "total": row[15]}, "home_rank": row[16], "away_rank": row[17]
                })

    # Sort by live games first, then by game time
    games.sort(key=lambda x: (not x.get('is_live', False), x.get('game_time') or ''))

    return {
        "games": games, "count": len(games),
        "live_count": sum(1 for g in games if g.get('is_live')),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/markets", response_model=MarketResponse)
async def get_markets(
    market_type: Optional[str] = Query(None, description="Filter by market type (e.g., 'nfl')"),
    limit: int = Query(50, description="Limit number of results")
):
    """
    Get active sports markets with AI predictions from database.
    Uses asyncio.to_thread() to prevent blocking event loop.
    """
    markets = await asyncio.to_thread(
        sports_service.get_markets_with_predictions, market_type, limit
    )
    return MarketResponse(markets=markets, count=len(markets))


@router.get("/live")
async def get_live_games():
    """
    Get currently live games with real-time scores and odds from database.
    Uses asyncio.to_thread() to prevent blocking event loop.
    """
    return await asyncio.to_thread(sports_service.get_live_games)


@router.get("/upcoming")
async def get_upcoming_games(limit: int = 10):
    """
    Get upcoming games with odds from database.
    Uses asyncio.to_thread() to prevent blocking event loop.
    """
    return await asyncio.to_thread(sports_service.get_upcoming_games, limit)




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


class BetSlipLeg(BaseModel):
    """Single bet leg for Telegram notification"""
    game_id: str
    sport: str
    home_team: str
    away_team: str
    bet_type: str  # moneyline, spread, total_over, total_under
    selection: str  # home, away, over, under
    odds: int = -110
    line: Optional[float] = None
    game_time: Optional[str] = None
    # AI Analysis
    ai_probability: Optional[float] = None
    ai_edge: Optional[float] = None
    ai_confidence: Optional[str] = None
    ai_reasoning: Optional[str] = None
    # Financials
    ev_percentage: Optional[float] = None
    kelly_fraction: Optional[float] = None
    stake: Optional[float] = None
    potential_payout: Optional[float] = None


class BetSlipNotifyRequest(BaseModel):
    """Request to send Telegram notification for bet slip"""
    legs: list[BetSlipLeg]
    mode: str = "singles"  # singles or parlay
    # Parlay analysis (if mode == parlay)
    combined_probability: Optional[float] = None
    expected_value: Optional[float] = None
    total_odds: Optional[float] = None
    stake: Optional[float] = None
    potential_payout: Optional[float] = None
    kelly_fraction: Optional[float] = None
    correlation_warnings: list[str] = []


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


def calculate_kelly_criterion(
    probability: float,
    odds: int,
    kelly_fraction: float = 0.25  # Quarter Kelly for conservative betting
) -> float:
    """
    Calculate optimal bet size using Kelly Criterion.

    The Kelly Criterion formula: f* = (bp - q) / b
    Where:
        b = decimal odds - 1 (the net profit multiplier)
        p = probability of winning
        q = probability of losing (1 - p)

    Args:
        probability: Model's estimated win probability (0-1)
        odds: American odds (e.g., -110, +150)
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)

    Returns:
        Recommended bet size as fraction of bankroll (0-1)
    """
    if probability <= 0 or probability >= 1:
        return 0.0

    # Convert American odds to decimal
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1

    b = decimal_odds - 1  # Net profit per dollar wagered
    p = probability
    q = 1 - p

    # Kelly formula: f* = (bp - q) / b
    kelly = (b * p - q) / b

    # Apply conservative fraction and cap at reasonable max
    adjusted_kelly = max(0, min(kelly * kelly_fraction, 0.10))  # Max 10% of bankroll

    return round(adjusted_kelly, 4)


def calculate_ai_edge(model_probability: float, implied_probability: float) -> float:
    """
    Calculate AI edge: how much the model thinks the true probability
    differs from what the odds imply.

    Positive edge = model sees value (underpriced)
    Negative edge = model sees overpriced bet

    Args:
        model_probability: AI model's estimated probability
        implied_probability: Probability implied by the betting odds

    Returns:
        Edge as percentage points (e.g., 5.2 means 5.2% edge)
    """
    return round((model_probability - implied_probability) * 100, 2)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def generate_kalshi_url(sport: str, home_team: str, away_team: str, selection: str) -> str:
    """
    Generate Kalshi market URL for a betting opportunity.

    Kalshi URL format: https://kalshi.com/markets/{ticker}
    NFL ticker example: KXNFL-{TEAM_ABBR}-WIN

    Args:
        sport: Sport type (NFL, NBA, NCAAF, NCAAB)
        home_team: Home team name
        away_team: Away team name
        selection: The team selected for the bet

    Returns:
        URL to the Kalshi market page, or search URL if exact market unknown
    """
    # NFL team abbreviations mapping
    NFL_ABBREVS = {
        'arizona cardinals': 'ARI', 'cardinals': 'ARI',
        'atlanta falcons': 'ATL', 'falcons': 'ATL',
        'baltimore ravens': 'BAL', 'ravens': 'BAL',
        'buffalo bills': 'BUF', 'bills': 'BUF',
        'carolina panthers': 'CAR', 'panthers': 'CAR',
        'chicago bears': 'CHI', 'bears': 'CHI',
        'cincinnati bengals': 'CIN', 'bengals': 'CIN',
        'cleveland browns': 'CLE', 'browns': 'CLE',
        'dallas cowboys': 'DAL', 'cowboys': 'DAL',
        'denver broncos': 'DEN', 'broncos': 'DEN',
        'detroit lions': 'DET', 'lions': 'DET',
        'green bay packers': 'GB', 'packers': 'GB',
        'houston texans': 'HOU', 'texans': 'HOU',
        'indianapolis colts': 'IND', 'colts': 'IND',
        'jacksonville jaguars': 'JAX', 'jaguars': 'JAX', 'jags': 'JAX',
        'kansas city chiefs': 'KC', 'chiefs': 'KC',
        'las vegas raiders': 'LV', 'raiders': 'LV',
        'los angeles chargers': 'LAC', 'chargers': 'LAC', 'la chargers': 'LAC',
        'los angeles rams': 'LAR', 'rams': 'LAR', 'la rams': 'LAR',
        'miami dolphins': 'MIA', 'dolphins': 'MIA',
        'minnesota vikings': 'MIN', 'vikings': 'MIN',
        'new england patriots': 'NE', 'patriots': 'NE',
        'new orleans saints': 'NO', 'saints': 'NO',
        'new york giants': 'NYG', 'giants': 'NYG', 'ny giants': 'NYG',
        'new york jets': 'NYJ', 'jets': 'NYJ', 'ny jets': 'NYJ',
        'philadelphia eagles': 'PHI', 'eagles': 'PHI',
        'pittsburgh steelers': 'PIT', 'steelers': 'PIT',
        'san francisco 49ers': 'SF', '49ers': 'SF',
        'seattle seahawks': 'SEA', 'seahawks': 'SEA',
        'tampa bay buccaneers': 'TB', 'buccaneers': 'TB', 'bucs': 'TB',
        'tennessee titans': 'TEN', 'titans': 'TEN',
        'washington commanders': 'WAS', 'commanders': 'WAS'
    }

    # NBA team abbreviations
    NBA_ABBREVS = {
        'atlanta hawks': 'ATL', 'hawks': 'ATL',
        'boston celtics': 'BOS', 'celtics': 'BOS',
        'brooklyn nets': 'BKN', 'nets': 'BKN',
        'charlotte hornets': 'CHA', 'hornets': 'CHA',
        'chicago bulls': 'CHI', 'bulls': 'CHI',
        'cleveland cavaliers': 'CLE', 'cavaliers': 'CLE', 'cavs': 'CLE',
        'dallas mavericks': 'DAL', 'mavericks': 'DAL', 'mavs': 'DAL',
        'denver nuggets': 'DEN', 'nuggets': 'DEN',
        'detroit pistons': 'DET', 'pistons': 'DET',
        'golden state warriors': 'GSW', 'warriors': 'GSW',
        'houston rockets': 'HOU', 'rockets': 'HOU',
        'indiana pacers': 'IND', 'pacers': 'IND',
        'los angeles clippers': 'LAC', 'clippers': 'LAC', 'la clippers': 'LAC',
        'los angeles lakers': 'LAL', 'lakers': 'LAL', 'la lakers': 'LAL',
        'memphis grizzlies': 'MEM', 'grizzlies': 'MEM',
        'miami heat': 'MIA', 'heat': 'MIA',
        'milwaukee bucks': 'MIL', 'bucks': 'MIL',
        'minnesota timberwolves': 'MIN', 'timberwolves': 'MIN', 'wolves': 'MIN',
        'new orleans pelicans': 'NOP', 'pelicans': 'NOP',
        'new york knicks': 'NYK', 'knicks': 'NYK',
        'oklahoma city thunder': 'OKC', 'thunder': 'OKC',
        'orlando magic': 'ORL', 'magic': 'ORL',
        'philadelphia 76ers': 'PHI', '76ers': 'PHI', 'sixers': 'PHI',
        'phoenix suns': 'PHX', 'suns': 'PHX',
        'portland trail blazers': 'POR', 'trail blazers': 'POR', 'blazers': 'POR',
        'sacramento kings': 'SAC', 'kings': 'SAC',
        'san antonio spurs': 'SAS', 'spurs': 'SAS',
        'toronto raptors': 'TOR', 'raptors': 'TOR',
        'utah jazz': 'UTA', 'jazz': 'UTA',
        'washington wizards': 'WAS', 'wizards': 'WAS'
    }

    # Base Kalshi URL
    KALSHI_BASE = "https://kalshi.com"

    # Get team abbreviation based on sport
    team_name = selection.lower() if selection else home_team.lower()
    abbrev = None

    if sport.upper() == "NFL":
        abbrev = NFL_ABBREVS.get(team_name)
        if abbrev:
            # Try exact ticker format for NFL
            return f"{KALSHI_BASE}/markets/kxnfl-24reg-{abbrev.lower()}"
    elif sport.upper() == "NBA":
        abbrev = NBA_ABBREVS.get(team_name)
        if abbrev:
            return f"{KALSHI_BASE}/markets/kxnba-24reg-{abbrev.lower()}"

    # For NCAA or if no match found, generate search URL
    # URL encode the team names for search
    import urllib.parse
    search_term = f"{away_team} {home_team}".replace(" ", "+")
    return f"{KALSHI_BASE}/browse?q={urllib.parse.quote(search_term)}"


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
            confidence_map = settings.prediction_confidence_map
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

        # OPTIMIZED: Single UNION ALL query + batch predictions (replaces N+1 pattern)
        bets = await asyncio.to_thread(
            _fetch_best_bets_optimized_sync,
            sport_list,
            min_ev,
            min_confidence,
            limit
        )

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
            american_odds = bet["odds"]
            odds_decimal = american_to_decimal(american_odds)
            implied_prob = 1 / odds_decimal
            model_prob = (bet["confidence"] / 100) * 0.8 + implied_prob * 0.2  # Blend

            # Calculate AI edge and Kelly Criterion
            ai_edge = calculate_ai_edge(model_prob, implied_prob)
            kelly = calculate_kelly_criterion(model_prob, american_odds)

            # Generate Kalshi URL for this bet
            kalshi_url = generate_kalshi_url(
                sport=bet["sport"],
                home_team=bet["home_team"],
                away_team=bet["away_team"],
                selection=bet.get("pick", "")
            )

            opportunities.append({
                "id": bet["id"],
                "sport": bet["sport"],
                "event_name": bet.get("matchup",
                                      f"{bet['away_team']} @ {bet['home_team']}"),
                "bet_type": bet["bet_type"],
                "selection": bet.get("pick", "N/A"),
                "odds": odds_decimal,
                "odds_american": american_odds,
                "implied_probability": round(implied_prob, 4),
                "model_probability": round(model_prob, 4),
                "ai_edge": ai_edge,
                "kelly_fraction": kelly,
                "ev_percentage": bet["ev"],
                "confidence": bet["confidence"],
                "overall_score": bet["combined_score"],
                "book": bet.get("source", "Best Odds"),
                "reasoning": bet.get("reasoning", "AI model prediction"),
                "spread": bet.get("spread"),
                "over_under": bet.get("over_under"),
                "odds_home": bet.get("odds_home"),
                "odds_away": bet.get("odds_away"),
                "updated_at": datetime.now().isoformat(),
                "game_time": bet["game_time"],
                "home_team": bet["home_team"],
                "away_team": bet["away_team"],
                "kalshi_url": kalshi_url
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


@router.post("/sync")
async def sync_sports_data(sport: str = Query("ALL", description="Sport to sync: NFL, NBA, NCAAF, NCAAB, or ALL")):
    """
    Sync sports data from ESPN API to database.
    Call this endpoint to populate games data for live game cards.
    """
    try:
        sport = sport.upper()
        results = {}

        sports_to_sync = ["NFL", "NBA", "NCAAF", "NCAAB"] if sport == "ALL" else [sport]

        for current_sport in sports_to_sync:
            try:
                if current_sport == "NFL":
                    results["NFL"] = await _sync_nfl_games()
                elif current_sport == "NBA":
                    results["NBA"] = await _sync_nba_games()
                elif current_sport == "NCAAF":
                    results["NCAAF"] = await _sync_ncaa_football_games()
                elif current_sport == "NCAAB":
                    results["NCAAB"] = await _sync_ncaa_basketball_games()
                else:
                    results[current_sport] = {"success": False, "message": f"Unknown sport: {current_sport}", "synced": 0}
            except Exception as e:
                logger.error(f"Error syncing {current_sport}: {e}")
                results[current_sport] = {"success": False, "message": str(e), "synced": 0}

        # Calculate totals
        total_synced = sum(r.get("synced", 0) for r in results.values())
        all_success = all(r.get("success", False) for r in results.values())

        return {
            "success": all_success or total_synced > 0,
            "message": f"Synced {total_synced} total games across {len(sports_to_sync)} sports",
            "total_synced": total_synced,
            "results": results,
            "synced_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error syncing sports data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _sync_nfl_games():
    """Sync NFL games from ESPN"""
    import json
    espn = ESPNLiveData()
    games = espn.get_scoreboard()

    if not games:
        return {"success": False, "message": "No NFL games available from ESPN", "synced": 0}

    synced_count = 0

    # Use direct SQL instead of nfl_db_manager to avoid schema mismatch
    with get_db_connection() as conn:
        cursor = conn.cursor()

        for game in games:
            try:
                # Parse game time
                game_time = game.get('game_time')
                if isinstance(game_time, str):
                    try:
                        game_time = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
                    except:
                        try:
                            game_time = datetime.strptime(game_time, '%Y-%m-%d %H:%M')
                        except:
                            game_time = datetime.now()

                # Map status
                status = 'scheduled'
                if game.get('is_live'):
                    status = 'live'
                elif game.get('is_completed'):
                    status = 'final'

                # Serialize raw data to proper JSON
                raw_data = json.dumps({k: str(v) if hasattr(v, 'isoformat') else v for k, v in game.items()})

                cursor.execute("""
                    INSERT INTO nfl_games (
                        game_id, season, week, home_team, away_team, home_team_abbr, away_team_abbr,
                        game_time, venue, home_score, away_score, quarter, time_remaining,
                        game_status, is_live, raw_game_data, last_synced
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (game_id) DO UPDATE SET
                        home_score = EXCLUDED.home_score,
                        away_score = EXCLUDED.away_score,
                        quarter = EXCLUDED.quarter,
                        time_remaining = EXCLUDED.time_remaining,
                        game_status = EXCLUDED.game_status,
                        is_live = EXCLUDED.is_live,
                        raw_game_data = EXCLUDED.raw_game_data,
                        last_synced = NOW()
                """, (
                    game.get('game_id'),
                    2024,
                    13,  # Current NFL week
                    game.get('home_team'),
                    game.get('away_team'),
                    game.get('home_abbr'),
                    game.get('away_abbr'),
                    game_time,
                    game.get('venue', ''),
                    game.get('home_score', 0),
                    game.get('away_score', 0),
                    game.get('period', 0),
                    game.get('clock', '0:00'),
                    status,
                    game.get('is_live', False),
                    raw_data
                ))
                synced_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert NFL game {game.get('game_id')}: {e}")

        conn.commit()

    return {"success": True, "message": f"Synced {synced_count} NFL games", "synced": synced_count}


async def _sync_nba_games():
    """Sync NBA games from ESPN"""
    import json
    espn = ESPNNBALiveData()
    games = espn.get_scoreboard()

    if not games:
        return {"success": False, "message": "No NBA games available", "synced": 0}

    synced_count = 0

    with get_db_connection() as conn:
        cursor = conn.cursor()

        for game in games:
            try:
                # Parse game time
                game_time = game.get('game_time')
                if isinstance(game_time, str):
                    try:
                        game_time = datetime.strptime(game_time, '%Y-%m-%d %H:%M')
                    except Exception:
                        game_time = datetime.now()

                # Map status
                status = game.get('status', 'scheduled')
                if game.get('is_live'):
                    status = 'live'
                elif game.get('is_completed'):
                    status = 'final'
                elif status == 'STATUS_SCHEDULED':
                    status = 'scheduled'

                # Proper JSON serialization
                raw_data = json.dumps({
                    k: str(v) if hasattr(v, 'isoformat') else v
                    for k, v in game.items()
                })

                cursor.execute("""
                    INSERT INTO nba_games (
                        game_id, season, home_team, away_team,
                        home_team_abbr, away_team_abbr,
                        game_time, venue, home_score, away_score,
                        quarter, time_remaining,
                        game_status, is_live, raw_game_data, last_synced
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (game_id) DO UPDATE SET
                        home_score = EXCLUDED.home_score,
                        away_score = EXCLUDED.away_score,
                        quarter = EXCLUDED.quarter,
                        time_remaining = EXCLUDED.time_remaining,
                        game_status = EXCLUDED.game_status,
                        is_live = EXCLUDED.is_live,
                        raw_game_data = EXCLUDED.raw_game_data,
                        last_synced = NOW()
                """, (
                    game.get('game_id'),
                    '2024-25',
                    game.get('home_team'),
                    game.get('away_team'),
                    game.get('home_abbr'),
                    game.get('away_abbr'),
                    game_time,
                    game.get('venue', ''),
                    game.get('home_score', 0),
                    game.get('away_score', 0),
                    game.get('period', 0),
                    game.get('clock', '0:00'),
                    status,
                    game.get('is_live', False),
                    raw_data
                ))
                synced_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert NBA game: {e}")

        conn.commit()

    return {"success": True, "message": f"Synced {synced_count} NBA games", "synced": synced_count}


async def _sync_ncaa_football_games():
    """Sync NCAA Football games from ESPN"""
    import json
    espn = ESPNNCAALiveData()
    games = espn.get_scoreboard(group='80')  # FBS games

    if not games:
        return {"success": False, "message": "No NCAAF games available", "synced": 0}

    synced_count = 0

    with get_db_connection() as conn:
        cursor = conn.cursor()

        for game in games:
            try:
                # Parse game time - handle datetime objects
                game_time = game.get('game_time')
                if game_time is None:
                    game_time = datetime.now()
                elif isinstance(game_time, str):
                    try:
                        game_time = datetime.fromisoformat(
                            game_time.replace('Z', '+00:00')
                        )
                    except Exception:
                        game_time = datetime.now()

                # Map ESPN status to database constraint values
                # Allowed: 'scheduled', 'live', 'halftime', 'final', 'postponed', 'cancelled'
                raw_status = str(game.get('status', 'scheduled')).upper()
                if game.get('is_live'):
                    status = 'live'
                elif game.get('is_completed'):
                    status = 'final'
                elif 'SCHEDULED' in raw_status or 'PRE' in raw_status:
                    status = 'scheduled'
                elif 'HALF' in raw_status:
                    status = 'halftime'
                elif 'FINAL' in raw_status or 'POST' in raw_status or 'COMPLETE' in raw_status:
                    status = 'final'
                elif 'DELAY' in raw_status or 'SUSPEND' in raw_status:
                    status = 'postponed'
                elif 'CANCEL' in raw_status or 'PPD' in raw_status:
                    status = 'cancelled'
                elif 'IN_PROGRESS' in raw_status or 'LIVE' in raw_status:
                    status = 'live'
                else:
                    # Default to scheduled for any unknown status
                    status = 'scheduled'

                # Proper JSON serialization
                raw_data = json.dumps({
                    k: str(v) if hasattr(v, 'isoformat') else v
                    for k, v in game.items()
                })

                cursor.execute("""
                    INSERT INTO ncaa_football_games (
                        game_id, season, week, home_team, away_team,
                        home_team_abbr, away_team_abbr,
                        home_rank, away_rank, conference, game_time, venue,
                        home_score, away_score, quarter, time_remaining,
                        possession, game_status, is_live,
                        raw_game_data, last_synced
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (game_id) DO UPDATE SET
                        home_score = EXCLUDED.home_score,
                        away_score = EXCLUDED.away_score,
                        quarter = EXCLUDED.quarter,
                        time_remaining = EXCLUDED.time_remaining,
                        possession = EXCLUDED.possession,
                        game_status = EXCLUDED.game_status,
                        is_live = EXCLUDED.is_live,
                        raw_game_data = EXCLUDED.raw_game_data,
                        last_synced = NOW()
                """, (
                    game.get('game_id'),
                    2024,
                    14,
                    game.get('home_team'),
                    game.get('away_team'),
                    game.get('home_abbr'),
                    game.get('away_abbr'),
                    game.get('home_rank'),
                    game.get('away_rank'),
                    game.get('home_conference', ''),
                    game_time,
                    game.get('venue', ''),
                    game.get('home_score', 0),
                    game.get('away_score', 0),
                    game.get('period', 0),
                    game.get('clock', '0:00'),
                    game.get('possession', ''),
                    status,
                    game.get('is_live', False),
                    raw_data
                ))
                synced_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert NCAAF game: {e}")

        conn.commit()

    return {"success": True, "message": f"Synced {synced_count} NCAAF games", "synced": synced_count}


async def _sync_ncaa_basketball_games():
    """Sync NCAA Basketball games from ESPN"""
    import json
    import requests as req

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        response = req.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        response.raise_for_status()
        data = response.json()

        games = []
        for event in data.get('events', []):
            try:
                competitions = event.get('competitions', [])
                if not competitions:
                    continue

                competition = competitions[0]
                competitors = competition.get('competitors', [])

                if len(competitors) < 2:
                    continue

                home_team = next(
                    (c for c in competitors if c.get('homeAway') == 'home'),
                    None
                )
                away_team = next(
                    (c for c in competitors if c.get('homeAway') == 'away'),
                    None
                )

                if not home_team or not away_team:
                    continue

                status = event.get('status', {})
                is_live = status.get('type', {}).get('state', '') == 'in'
                is_completed = status.get('type', {}).get('completed', False)

                games.append({
                    'game_id': event.get('id'),
                    'home_team': home_team.get('team', {}).get('displayName', ''),
                    'away_team': away_team.get('team', {}).get('displayName', ''),
                    'home_abbr': home_team.get('team', {}).get('abbreviation', ''),
                    'away_abbr': away_team.get('team', {}).get('abbreviation', ''),
                    'home_score': int(home_team.get('score', 0) or 0),
                    'away_score': int(away_team.get('score', 0) or 0),
                    'home_rank': home_team.get('curatedRank', {}).get('current'),
                    'away_rank': away_team.get('curatedRank', {}).get('current'),
                    'period': status.get('period', 0),
                    'clock': status.get('displayClock', '0:00'),
                    'is_live': is_live,
                    'is_completed': is_completed,
                    'game_time': event.get('date'),
                    'venue': competition.get('venue', {}).get('fullName', ''),
                    'status': status.get('type', {}).get('name', 'scheduled')
                })
            except Exception as e:
                logger.warning(f"Failed to parse NCAAB game: {e}")
                continue

        if not games:
            return {"success": False, "message": "No NCAAB games available", "synced": 0}

        synced_count = 0

        with get_db_connection() as conn:
            cursor = conn.cursor()

            for game in games:
                try:
                    # Parse game time
                    game_time = game.get('game_time')
                    if isinstance(game_time, str):
                        try:
                            game_time = datetime.fromisoformat(
                                game_time.replace('Z', '+00:00')
                            )
                        except Exception:
                            game_time = datetime.now()

                    # Map status
                    status = 'scheduled'
                    if game.get('is_live'):
                        status = 'live'
                    elif game.get('is_completed'):
                        status = 'final'

                    # Proper JSON serialization
                    raw_data = json.dumps(game)

                    cursor.execute("""
                        INSERT INTO ncaa_basketball_games (
                            game_id, season, home_team, away_team,
                            home_team_abbr, away_team_abbr,
                            home_rank, away_rank, game_time, venue,
                            home_score, away_score, half, time_remaining,
                            game_status, is_live, raw_game_data, last_synced
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, NOW()
                        )
                        ON CONFLICT (game_id) DO UPDATE SET
                            home_score = EXCLUDED.home_score,
                            away_score = EXCLUDED.away_score,
                            half = EXCLUDED.half,
                            time_remaining = EXCLUDED.time_remaining,
                            game_status = EXCLUDED.game_status,
                            is_live = EXCLUDED.is_live,
                            raw_game_data = EXCLUDED.raw_game_data,
                            last_synced = NOW()
                    """, (
                        game.get('game_id'),
                        '2024-25',
                        game.get('home_team'),
                        game.get('away_team'),
                        game.get('home_abbr'),
                        game.get('away_abbr'),
                        game.get('home_rank'),
                        game.get('away_rank'),
                        game_time,
                        game.get('venue', ''),
                        game.get('home_score', 0),
                        game.get('away_score', 0),
                        game.get('period', 0),
                        game.get('clock', '0:00'),
                        status,
                        game.get('is_live', False),
                        raw_data
                    ))
                    synced_count += 1
                except Exception as e:
                    logger.warning(f"Failed to upsert NCAAB game: {e}")

            conn.commit()

        return {
            "success": True,
            "message": f"Synced {synced_count} NCAAB games",
            "synced": synced_count
        }

    except Exception as e:
        logger.error(f"Error fetching NCAAB games: {e}")
        return {"success": False, "message": str(e), "synced": 0}


@router.get("/games/live")
async def get_all_live_games():
    """
    Get all currently live games across all sports with real-time data.
    Uses asyncio.to_thread() for non-blocking DB access.
    """
    try:
        return await asyncio.to_thread(_fetch_all_live_games_sync)
    except Exception as e:
        logger.error(f"Error fetching live games: {e}")
        return {"live_games": [], "count": 0, "error": str(e)}


@router.post("/sync-odds")
async def sync_kalshi_odds():
    """
    Sync live odds from Kalshi prediction markets.
    Matches current games to Kalshi markets and stores odds snapshots.
    Uses live API data directly (no database cache required).
    """
    import json

    try:
        kalshi = KalshiPublicClient()

        # Get all football markets from Kalshi (live API data)
        football_markets = kalshi.get_football_markets()
        nfl_markets = football_markets.get('nfl', [])
        college_markets = football_markets.get('college', [])

        logger.info(f"Fetched {len(nfl_markets)} NFL markets, {len(college_markets)} college markets from Kalshi API")

        synced_count = 0
        odds_data = []

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get all upcoming/live NFL games
            cursor.execute("""
                SELECT game_id, home_team, away_team, game_time, game_status
                FROM nfl_games
                WHERE game_status IN ('scheduled', 'live')
                ORDER BY game_time
            """)
            nfl_games = cursor.fetchall()

            # Match NFL games to live Kalshi markets (direct API matching)
            for game in nfl_games:
                game_id, home_team, away_team, game_time, status = game

                # Try to find matching Kalshi market in live API data
                match = _match_game_to_live_markets(home_team, away_team, nfl_markets)

                if match:
                    home_odds = match.get('home_win_price', 0)
                    away_odds = match.get('away_win_price', 0)
                    ticker = match.get('ticker', '')

                    # Convert Kalshi price (0-100) to implied probability
                    home_prob = home_odds / 100 if home_odds else 0.5
                    away_prob = away_odds / 100 if away_odds else 0.5

                    # Convert to American odds
                    home_american = _prob_to_american(home_prob)
                    away_american = _prob_to_american(away_prob)

                    # Store odds snapshot
                    cursor.execute("""
                        INSERT INTO live_odds_snapshots (
                            game_id, sport, source, moneyline_home, moneyline_away,
                            home_implied_prob, away_implied_prob, raw_odds_data
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        game_id, 'NFL', 'kalshi',
                        home_american, away_american,
                        home_prob, away_prob,
                        json.dumps(match)
                    ))

                    # Update game with current odds
                    cursor.execute("""
                        UPDATE nfl_games
                        SET moneyline_home = %s, moneyline_away = %s
                        WHERE game_id = %s
                    """, (home_american, away_american, game_id))

                    odds_data.append({
                        'game_id': game_id,
                        'matchup': f"{away_team} @ {home_team}",
                        'home_odds': home_american,
                        'away_odds': away_american,
                        'home_prob': round(home_prob * 100, 1),
                        'away_prob': round(away_prob * 100, 1),
                        'source': 'kalshi',
                        'ticker': ticker
                    })
                    synced_count += 1

            # Get NCAAF games and match to college markets
            cursor.execute("""
                SELECT game_id, home_team, away_team, game_time, game_status
                FROM ncaa_football_games
                WHERE game_status IN ('scheduled', 'live')
                ORDER BY game_time
                LIMIT 30
            """)
            ncaa_games = cursor.fetchall()

            for game in ncaa_games:
                game_id, home_team, away_team, game_time, status = game

                # Try to find matching Kalshi market in live API data
                match = _match_game_to_live_markets(home_team, away_team, college_markets)

                if match:
                    home_odds = match.get('home_win_price', 0)
                    away_odds = match.get('away_win_price', 0)

                    home_prob = home_odds / 100 if home_odds else 0.5
                    away_prob = away_odds / 100 if away_odds else 0.5

                    home_american = _prob_to_american(home_prob)
                    away_american = _prob_to_american(away_prob)

                    cursor.execute("""
                        INSERT INTO live_odds_snapshots (
                            game_id, sport, source, moneyline_home, moneyline_away,
                            home_implied_prob, away_implied_prob, raw_odds_data
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        game_id, 'NCAAF', 'kalshi',
                        home_american, away_american,
                        home_prob, away_prob,
                        json.dumps(match)
                    ))

                    odds_data.append({
                        'game_id': game_id,
                        'matchup': f"{away_team} @ {home_team}",
                        'home_odds': home_american,
                        'away_odds': away_american,
                        'source': 'kalshi'
                    })
                    synced_count += 1

            conn.commit()

        return {
            "success": True,
            "message": f"Synced {synced_count} odds from Kalshi",
            "synced": synced_count,
            "kalshi_markets": {
                "nfl": len(nfl_markets),
                "college": len(college_markets)
            },
            "odds": odds_data[:10],  # Return first 10 for preview
            "synced_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error syncing Kalshi odds: {e}")
        return {
            "success": False,
            "message": str(e),
            "synced": 0
        }


def _prob_to_american(probability: float) -> int:
    """Convert probability (0-1) to American odds."""
    if probability <= 0 or probability >= 1:
        return 0

    if probability >= 0.5:
        # Favorite: negative odds
        return int(-100 * probability / (1 - probability))
    else:
        # Underdog: positive odds
        return int(100 * (1 - probability) / probability)


def _match_game_to_live_markets(
    home_team: str,
    away_team: str,
    markets: list,
    team_variations: dict = None
) -> dict | None:
    """
    Match a game to Kalshi markets using live API data (no database needed).

    Args:
        home_team: Home team name
        away_team: Away team name
        markets: List of market dicts from Kalshi API
        team_variations: Optional dict of team name variations

    Returns:
        Match dict with home_win_price, away_win_price, ticker, etc. or None
    """
    if not markets:
        return None

    # NFL team variations for matching
    NFL_VARIATIONS = {
        'Arizona Cardinals': ['arizona', 'cardinals', 'ari'],
        'Atlanta Falcons': ['atlanta', 'falcons', 'atl'],
        'Baltimore Ravens': ['baltimore', 'ravens', 'bal'],
        'Buffalo Bills': ['buffalo', 'bills', 'buf'],
        'Carolina Panthers': ['carolina', 'panthers', 'car'],
        'Chicago Bears': ['chicago', 'bears', 'chi'],
        'Cincinnati Bengals': ['cincinnati', 'bengals', 'cin'],
        'Cleveland Browns': ['cleveland', 'browns', 'cle'],
        'Dallas Cowboys': ['dallas', 'cowboys', 'dal'],
        'Denver Broncos': ['denver', 'broncos', 'den'],
        'Detroit Lions': ['detroit', 'lions', 'det'],
        'Green Bay Packers': ['green bay', 'packers', 'gb'],
        'Houston Texans': ['houston', 'texans', 'hou'],
        'Indianapolis Colts': ['indianapolis', 'colts', 'ind'],
        'Jacksonville Jaguars': ['jacksonville', 'jaguars', 'jax', 'jags'],
        'Kansas City Chiefs': ['kansas city', 'chiefs', 'kc'],
        'Las Vegas Raiders': ['las vegas', 'raiders', 'lv'],
        'Los Angeles Chargers': ['la chargers', 'chargers', 'lac'],
        'Los Angeles Rams': ['la rams', 'rams', 'lar'],
        'Miami Dolphins': ['miami', 'dolphins', 'mia'],
        'Minnesota Vikings': ['minnesota', 'vikings', 'min'],
        'New England Patriots': ['new england', 'patriots', 'ne'],
        'New Orleans Saints': ['new orleans', 'saints', 'no'],
        'New York Giants': ['ny giants', 'giants', 'nyg'],
        'New York Jets': ['ny jets', 'jets', 'nyj'],
        'Philadelphia Eagles': ['philadelphia', 'eagles', 'phi'],
        'Pittsburgh Steelers': ['pittsburgh', 'steelers', 'pit'],
        'San Francisco 49ers': ['san francisco', '49ers', 'sf'],
        'Seattle Seahawks': ['seattle', 'seahawks', 'sea'],
        'Tampa Bay Buccaneers': ['tampa bay', 'buccaneers', 'bucs', 'tb'],
        'Tennessee Titans': ['tennessee', 'titans', 'ten'],
        'Washington Commanders': ['washington', 'commanders', 'was']
    }

    def get_variations(team: str) -> list:
        """Get all name variations for a team."""
        team_lower = team.lower()
        for full_name, vars in NFL_VARIATIONS.items():
            if team_lower in vars or team_lower == full_name.lower():
                return vars + [team_lower]
        # Default: split into parts
        parts = team.lower().split()
        return [team_lower] + parts if len(parts) > 1 else [team_lower]

    home_vars = get_variations(home_team)
    away_vars = get_variations(away_team)

    # Search through markets for a match
    for market in markets:
        title = market.get('title', '').lower()
        ticker = market.get('ticker', '').lower()

        # Check if both teams appear in the market title
        home_found = any(var in title for var in home_vars)
        away_found = any(var in title for var in away_vars)

        if home_found and away_found:
            # Found a match - extract prices
            yes_price = market.get('yes_bid') or market.get('last_price') or 0
            no_price = market.get('no_bid') or (100 - yes_price if yes_price else 0)

            # Determine which team is "yes" based on ticker suffix
            ticker_suffix = ticker.split('-')[-1] if '-' in ticker else ''

            # Check if ticker suffix matches home team
            home_is_yes = any(var in ticker_suffix for var in home_vars)

            if home_is_yes:
                home_price = yes_price
                away_price = no_price
            else:
                home_price = no_price
                away_price = yes_price

            return {
                'home_win_price': home_price,
                'away_win_price': away_price,
                'ticker': market.get('ticker', ''),
                'market_title': market.get('title', ''),
                'volume': market.get('volume', 0)
            }

    return None


@router.get("/all-data")
async def get_all_sports_data(
    limit: int = Query(20, description="Limit games per category"),
    sports: str = Query("NFL,NBA,NCAAF,NCAAB", description="Comma-separated sports")
):
    """
    Combined endpoint that batches live games, upcoming games, and best-bets into one response.
    Reduces API calls from 3 to 1 for massive performance improvement.

    OPTIMIZED: Uses asyncio.gather() for true parallel fetching of all data sources.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        # Create thread pool for sync operations
        executor = ThreadPoolExecutor(max_workers=3)
        loop = asyncio.get_event_loop()

        # OPTIMIZED: Run all data fetches in parallel using asyncio.gather()
        async def fetch_live():
            return await loop.run_in_executor(executor, sports_service.get_live_games)

        async def fetch_upcoming():
            return await loop.run_in_executor(
                executor,
                lambda: sports_service.get_upcoming_games(limit=limit)
            )

        async def fetch_best_bets():
            try:
                return await get_best_bets_unified(
                    sports=sports,
                    min_ev=0.0,
                    min_confidence=50.0,
                    sort_by="ev",
                    limit=limit
                )
            except Exception as e:
                logger.warning(f"Error fetching best bets: {e}")
                return {"opportunities": [], "total": 0, "avg_ev": 0, "avg_confidence": 0}

        # Execute all fetches in parallel
        live_games, upcoming_games, best_bets_response = await asyncio.gather(
            fetch_live(),
            fetch_upcoming(),
            fetch_best_bets(),
            return_exceptions=True
        )

        # Handle any exceptions from parallel execution
        if isinstance(live_games, Exception):
            logger.warning(f"Live games fetch failed: {live_games}")
            live_games = []
        if isinstance(upcoming_games, Exception):
            logger.warning(f"Upcoming games fetch failed: {upcoming_games}")
            upcoming_games = []
        if isinstance(best_bets_response, Exception):
            logger.warning(f"Best bets fetch failed: {best_bets_response}")
            best_bets_response = {"opportunities": [], "total": 0, "avg_ev": 0, "avg_confidence": 0}

        live_games = live_games or []
        upcoming_games = upcoming_games or []

        # Ensure best_bets_response is a dict
        if not best_bets_response or not isinstance(best_bets_response, dict):
            best_bets_response = {"opportunities": [], "total": 0, "avg_ev": 0, "avg_confidence": 0}

        # Calculate summary stats
        total_live = len(live_games)
        total_upcoming = len(upcoming_games)
        total_best_bets = best_bets_response.get('total', 0)

        # Extract high-value opportunities (handle None ai_prediction)
        high_ev_games = [
            g for g in live_games + upcoming_games
            if (g.get('ai_prediction') or {}).get('ev', 0) >= 20
        ]

        return {
            "success": True,
            "live_games": live_games,
            "upcoming_games": upcoming_games,
            "best_bets": best_bets_response.get('opportunities', []),
            "summary": {
                "total_live": total_live,
                "total_upcoming": total_upcoming,
                "total_best_bets": total_best_bets,
                "high_ev_count": len(high_ev_games),
                "sports_included": [s.strip().upper() for s in sports.split(',')],
                "avg_ev": best_bets_response.get('avg_ev', 0),
                "avg_confidence": best_bets_response.get('avg_confidence', 0)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in combined sports data endpoint: {e}")
        return {
            "success": False,
            "live_games": [],
            "upcoming_games": [],
            "best_bets": [],
            "summary": {
                "total_live": 0,
                "total_upcoming": 0,
                "total_best_bets": 0,
                "high_ev_count": 0,
                "sports_included": [],
                "avg_ev": 0,
                "avg_confidence": 0
            },
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/games/with-odds")
async def get_games_with_odds(sport: str = Query("ALL", description="Sport filter: NFL, NBA, NCAAF, NCAAB, or ALL")):
    """
    Get all games with their latest odds for the living game cards.
    Uses asyncio.to_thread() for non-blocking DB access.
    """
    try:
        return await asyncio.to_thread(_fetch_games_with_odds_sync, sport)
    except Exception as e:
        logger.error(f"Error fetching games with odds: {e}")
        return {"games": [], "count": 0, "error": str(e)}


@router.post("/bet-slip/notify")
async def send_bet_slip_notification(request: BetSlipNotifyRequest):
    """
    Send Telegram notification when game(s) are added to bet slip.

    Supports both single bets and parlays with comprehensive AI analysis breakdown.
    """
    try:
        notifier = get_telegram_notifier()

        if not notifier.enabled:
            return {
                "success": False,
                "message": "Telegram notifications not enabled. Set TELEGRAM_ENABLED=true in .env",
                "sent_count": 0
            }

        sent_messages = []
        errors = []

        if request.mode == "parlay" and len(request.legs) > 1:
            # Send parlay notification
            parlay_data = {
                "legs": [leg.model_dump() for leg in request.legs],
                "total_odds": request.total_odds or 1,
                "stake": request.stake or 0,
                "potential_payout": request.potential_payout or 0,
                "combined_probability": request.combined_probability or 0,
                "expected_value": request.expected_value or 0,
                "kelly_fraction": request.kelly_fraction or 0,
                "correlation_warnings": request.correlation_warnings
            }

            message_id = notifier.send_parlay_alert(parlay_data)
            if message_id:
                sent_messages.append({
                    "type": "parlay",
                    "message_id": message_id,
                    "legs": len(request.legs)
                })
            else:
                errors.append("Failed to send parlay notification")

        else:
            # Send individual bet notifications
            for leg in request.legs:
                # Try to get AI prediction if not provided
                ai_prob = leg.ai_probability
                ai_edge = leg.ai_edge
                ai_confidence = leg.ai_confidence
                ai_reasoning = leg.ai_reasoning
                ev_percent = leg.ev_percentage

                # If AI data not provided, fetch it
                if ai_prob is None:
                    try:
                        sport = leg.sport.upper()
                        if sport == "NFL":
                            predictor = get_nfl_predictor()
                        elif sport == "NBA":
                            predictor = get_nba_predictor()
                        else:
                            predictor = get_ncaa_predictor()

                        prediction = predictor.predict_winner(
                            home_team=leg.home_team,
                            away_team=leg.away_team
                        )

                        if prediction:
                            probability = prediction.get('probability', 0.5)
                            ai_prob = probability
                            ai_confidence = prediction.get('confidence', 'medium')
                            ai_reasoning = prediction.get('explanation', '')

                            # Calculate EV
                            ev = calculate_expected_value(probability, leg.odds)
                            ev_percent = ev

                            # Calculate edge vs implied
                            if leg.odds < 0:
                                implied = abs(leg.odds) / (abs(leg.odds) + 100)
                            else:
                                implied = 100 / (leg.odds + 100) if leg.odds > 0 else 0.5
                            ai_edge = probability - implied

                    except Exception as e:
                        logger.warning(f"Could not fetch AI prediction: {e}")

                bet_data = {
                    "game_id": leg.game_id,
                    "sport": leg.sport,
                    "home_team": leg.home_team,
                    "away_team": leg.away_team,
                    "bet_type": leg.bet_type,
                    "selection": leg.selection,
                    "odds": leg.odds,
                    "line": leg.line,
                    "game_time": leg.game_time,
                    "ai_probability": ai_prob,
                    "ai_edge": ai_edge,
                    "ai_confidence": ai_confidence,
                    "ai_reasoning": ai_reasoning,
                    "ev_percentage": ev_percent,
                    "kelly_fraction": leg.kelly_fraction,
                    "stake": leg.stake,
                    "potential_payout": leg.potential_payout
                }

                message_id = notifier.send_bet_slip_alert(bet_data)
                if message_id:
                    sent_messages.append({
                        "type": "single",
                        "message_id": message_id,
                        "game": f"{leg.away_team} @ {leg.home_team}"
                    })
                else:
                    errors.append(f"Failed to send notification for {leg.away_team} @ {leg.home_team}")

        return {
            "success": len(sent_messages) > 0,
            "message": f"Sent {len(sent_messages)} notification(s)" if sent_messages else "No notifications sent",
            "sent_count": len(sent_messages),
            "sent_messages": sent_messages,
            "errors": errors if errors else None
        }

    except Exception as e:
        logger.error(f"Error sending bet slip notification: {e}")
        return {
            "success": False,
            "message": str(e),
            "sent_count": 0
        }


@router.post("/bet-slip/test")
async def test_bet_slip_notification():
    """
    Send a test bet slip notification with sample data.
    Use this to verify Telegram integration is working.
    """
    try:
        # Force reload to pick up latest .env config
        notifier = get_telegram_notifier(force_reload=True)

        if not notifier.enabled:
            return {
                "success": False,
                "message": "Telegram notifications not enabled. Set TELEGRAM_ENABLED=true in .env",
                "telegram_enabled": False
            }

        # Create test bet data
        test_bet = {
            "game_id": "TEST_123",
            "sport": "NFL",
            "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills",
            "bet_type": "spread",
            "selection": "home",
            "odds": -110,
            "line": -3.5,
            "game_time": "Sun 4:25 PM",
            "ai_probability": 0.58,
            "ai_edge": 0.05,
            "ai_confidence": "high",
            "ai_reasoning": "Chiefs have a 58% win probability at home. "
                           "Patrick Mahomes is 8-2 ATS in divisional games. "
                           "Buffalo's defense has allowed 28+ points in 3 of last 5 road games.",
            "ev_percentage": 8.5,
            "kelly_fraction": 0.042,
            "stake": 25.00,
            "potential_payout": 22.73
        }

        message_id = notifier.send_bet_slip_alert(test_bet)

        if message_id:
            return {
                "success": True,
                "message": "Test notification sent successfully!",
                "message_id": message_id,
                "test_data": test_bet
            }
        else:
            return {
                "success": False,
                "message": "Failed to send test notification. Check Telegram credentials.",
                "telegram_enabled": True
            }

    except Exception as e:
        logger.error(f"Error sending test notification: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@router.post("/sync-real-odds")
async def sync_real_odds_from_api(
    sports: str = Query("NFL,NBA", description="Comma-separated sports to sync")
):
    """
    Sync REAL betting odds from The Odds API.

    Uses your API quota (500 requests/month). Each sport = 1 request.

    Recommended usage:
    - Every 6 hours for live odds: ~4 requests/day = 120/month
    - Or call manually before making betting decisions

    The odds are stored in game tables (nfl_games, nba_games, etc.)
    and displayed in the UI with real moneyline, spread, and total data.
    """
    try:
        sport_list = [s.strip().upper() for s in sports.split(',')]
        result = await sports_service.sync_odds_from_api(sport_list)

        return result

    except Exception as e:
        logger.error(f"Error syncing real odds: {e}")
        return {
            "success": False,
            "error": str(e),
            "synced_at": datetime.now().isoformat()
        }


@router.get("/odds-quota")
async def get_odds_api_quota():
    """
    Check The Odds API usage quota.

    Returns remaining requests out of 500/month.
    """
    try:
        return sports_service.get_odds_quota()
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Real-Time Streaming Endpoints (SSE + WebSocket)
# =============================================================================

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from backend.services.sports_infrastructure import (
    get_sports_infra,
    bayesian_probability_blend,
    ensemble_edge_detection
)


@router.get("/stream/odds")
async def stream_odds_sse(
    sports: str = Query("NFL,NBA", description="Comma-separated sports"),
    interval: int = Query(30, description="Update interval seconds")
):
    """
    Server-Sent Events (SSE) endpoint for real-time odds streaming.

    Connect with:
        const eventSource = new EventSource('/api/sports/stream/odds?sports=NFL,NBA');
        eventSource.onmessage = (e) => console.log(JSON.parse(e.data));

    Returns continuous stream of odds updates.
    """
    try:
        infra = await get_sports_infra()
        sport_list = [s.strip().upper() for s in sports.split(',')]

        return StreamingResponse(
            infra.sse_odds_stream(sport_list, interval),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    except Exception as e:
        logger.error(f"SSE odds stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/live-scores")
async def stream_live_scores_sse(
    interval: int = Query(10, description="Update interval seconds")
):
    """
    SSE endpoint for live game scores.

    Provides real-time score updates every 10 seconds (configurable).
    Perfect for live game dashboards.
    """
    try:
        infra = await get_sports_infra()

        return StreamingResponse(
            infra.sse_live_scores_stream(interval),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error(f"SSE live scores error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/live")
async def websocket_live_games(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live game updates.

    Send JSON messages:
        {"action": "subscribe", "rooms": ["live_games", "odds_nfl"]}
        {"action": "unsubscribe", "rooms": ["odds_nfl"]}
        {"action": "ping"}

    Receives:
        {"type": "live_update", "data": {...}, "timestamp": "..."}
        {"type": "odds_update", "sport": "NFL", "data": [...]}
    """
    try:
        infra = await get_sports_infra()
        ws_manager = infra.ws

        # Accept and track connection
        conn_info = await ws_manager.connect(
            websocket,
            user_id=None,
            rooms={"live_games"}  # Default room
        )

        try:
            while True:
                # Receive and process messages
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "ping":
                    await ws_manager.handle_heartbeat(websocket)
                    await ws_manager.send_personal(websocket, {"type": "pong"})

                elif action == "subscribe":
                    rooms = data.get("rooms", [])
                    for room in rooms:
                        await ws_manager.join_room(websocket, room)
                    await ws_manager.send_personal(websocket, {
                        "type": "subscribed",
                        "rooms": rooms
                    })

                elif action == "unsubscribe":
                    rooms = data.get("rooms", [])
                    for room in rooms:
                        await ws_manager.leave_room(websocket, room)
                    await ws_manager.send_personal(websocket, {
                        "type": "unsubscribed",
                        "rooms": rooms
                    })

        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@router.get("/ai/analyze-edge")
async def analyze_betting_edge(
    model_prob: float = Query(..., description="AI model probability (0-1)"),
    implied_prob: float = Query(..., description="Market implied probability (0-1)"),
    model_confidence: float = Query(0.7, description="Model confidence (0-1)")
):
    """
    Analyze betting edge using AI ensemble methods.

    Returns:
    - Blended probability using Bayesian Model Averaging
    - Edge detection with quality assessment
    - Kelly-optimal recommendation

    This is the AI-powered edge detection engine.
    """
    try:
        # Validate inputs
        if not (0 < model_prob < 1):
            raise HTTPException(400, "model_prob must be between 0 and 1")
        if not (0 < implied_prob < 1):
            raise HTTPException(400, "implied_prob must be between 0 and 1")

        # Bayesian probability blending
        blend_result = bayesian_probability_blend(
            model_prob=model_prob,
            implied_prob=implied_prob,
            model_confidence=model_confidence
        )

        # Edge detection
        edge_result = ensemble_edge_detection(
            model_prob=model_prob,
            implied_prob=implied_prob
        )

        # Calculate Kelly criterion for recommended bet size
        if edge_result["recommendation"] == "bet":
            kelly = calculate_kelly_criterion(
                probability=blend_result.final_probability,
                odds=_prob_to_american(implied_prob)
            )
        else:
            kelly = 0.0

        return {
            "success": True,
            "inputs": {
                "model_probability": model_prob,
                "implied_probability": implied_prob,
                "model_confidence": model_confidence
            },
            "probability_blend": blend_result.to_dict(),
            "edge_analysis": edge_result,
            "kelly_fraction": kelly,
            "recommendation": {
                "action": edge_result["recommendation"],
                "bet_strength": edge_result["bet_strength"],
                "suggested_stake_percent": round(kelly * 100, 2) if kelly else 0
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Edge analysis error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/infrastructure/health")
async def sports_infrastructure_health():
    """
    Get health status of all sports betting infrastructure components.

    Returns status of:
    - Async database pool
    - Redis cache
    - WebSocket connections
    - Circuit breakers (ESPN, Kalshi, Odds API)
    """
    try:
        infra = await get_sports_infra()
        return await infra.health_check()
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/telegram/setup")
async def setup_telegram():
    """
    Setup Telegram by retrieving chat ID from bot updates.

    Steps:
    1. Message @ava_n8n_bot on Telegram
    2. Call this endpoint to retrieve your chat ID
    3. Updates .env and sends test message
    """
    import os
    import httpx

    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')

    if not bot_token:
        return {
            "success": False,
            "message": "TELEGRAM_BOT_TOKEN not set in .env",
            "step": "config"
        }

    try:
        # Get bot info
        async with httpx.AsyncClient() as client:
            bot_info = await client.get(f"https://api.telegram.org/bot{bot_token}/getMe")
            bot_data = bot_info.json()

            if not bot_data.get('ok'):
                return {
                    "success": False,
                    "message": "Invalid bot token",
                    "step": "auth"
                }

            bot_username = bot_data['result']['username']

            # Get updates to find chat ID
            updates_resp = await client.get(f"https://api.telegram.org/bot{bot_token}/getUpdates")
            updates = updates_resp.json()

            if not updates.get('ok') or not updates.get('result'):
                return {
                    "success": False,
                    "message": f"No messages found. Please message @{bot_username} on Telegram first, then call this endpoint again.",
                    "bot_username": bot_username,
                    "bot_link": f"https://t.me/{bot_username}",
                    "step": "message_bot"
                }

            # Get the most recent chat ID
            latest_update = updates['result'][-1]
            chat_id = None
            chat_info = {}

            if 'message' in latest_update:
                chat = latest_update['message']['chat']
                chat_id = str(chat['id'])
                chat_info = {
                    'first_name': chat.get('first_name', ''),
                    'username': chat.get('username', ''),
                    'type': chat.get('type', '')
                }

            if not chat_id:
                return {
                    "success": False,
                    "message": "Could not extract chat ID from updates",
                    "step": "parse"
                }

            # Update .env file
            env_path = '/Users/adam/code/AVA/.env'
            with open(env_path, 'r') as f:
                env_content = f.read()

            # Update TELEGRAM_CHAT_ID
            if 'TELEGRAM_CHAT_ID=' in env_content:
                lines = env_content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('TELEGRAM_CHAT_ID='):
                        lines[i] = f'TELEGRAM_CHAT_ID={chat_id}'
                        break
                env_content = '\n'.join(lines)

            # Update TELEGRAM_ENABLED
            if 'TELEGRAM_ENABLED=' in env_content:
                lines = env_content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('TELEGRAM_ENABLED='):
                        lines[i] = 'TELEGRAM_ENABLED=true'
                        break
                env_content = '\n'.join(lines)

            with open(env_path, 'w') as f:
                f.write(env_content)

            # Update environment variables in current process
            os.environ['TELEGRAM_CHAT_ID'] = chat_id
            os.environ['TELEGRAM_ENABLED'] = 'true'

            # Reinitialize notifier with new credentials
            global _telegram_notifier
            _telegram_notifier = TelegramNotifier()

            # Send welcome message
            welcome_msg = (
                "🎉 *AVA Sports Alerts Connected!*\n\n"
                "You'll now receive notifications when:\n"
                "• You add games to your bet slip\n"
                "• AI finds high-value betting opportunities\n"
                "• Live games have significant odds movements\n\n"
                f"Chat ID: `{chat_id}`\n"
                "Status: ✅ Active"
            )

            send_resp = await client.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": welcome_msg,
                    "parse_mode": "Markdown"
                }
            )

            return {
                "success": True,
                "message": "Telegram setup complete! Check your Telegram for a welcome message.",
                "chat_id": chat_id,
                "chat_info": chat_info,
                "telegram_enabled": True,
                "step": "complete"
            }

    except Exception as e:
        logger.error(f"Error setting up Telegram: {e}")
        return {
            "success": False,
            "message": str(e),
            "step": "error"
        }
