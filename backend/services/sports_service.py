from typing import List, Optional
import logging
from psycopg2.extras import RealDictCursor
from backend.database.connection import db_pool
from backend.models.market import Market

logger = logging.getLogger(__name__)

from src.nfl_db_manager import NFLDBManager
from src.database.query_cache import query_cache
from src.prediction_agents.nfl_predictor import NFLPredictor

# Singleton predictor for efficiency
_nfl_predictor = None

def get_nfl_predictor():
    global _nfl_predictor
    if _nfl_predictor is None:
        _nfl_predictor = NFLPredictor()
    return _nfl_predictor

class SportsService:
    def __init__(self) -> None:
        self.nfl_db = NFLDBManager()

    def get_markets_with_predictions(self, market_type: Optional[str] = None, limit: int = 50) -> List[Market]:
        """
        Get markets with AI predictions, ranked by opportunity.
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
                    """
                    
                    params = []
                    if market_type:
                        query += " AND m.market_type = %s"
                        params.append(market_type)
                        
                    query += " ORDER BY p.overall_rank ASC NULLS LAST LIMIT %s"
                    params.append(limit)
                    
                    cur.execute(query, tuple(params))
                    results = cur.fetchall()
                    
                    return [Market(**row) for row in results]
                    
                except Exception as e:
                    logger.error(f"Error fetching markets: {e}")
                    return []

    def get_live_games(self) -> List[dict]:
        """Get all currently live games across sports with AI predictions"""
        try:
            # Check cache first
            cached = query_cache.get('live_games')
            if cached:
                return cached

            nfl_games = self.nfl_db.get_live_games()
            predictor = get_nfl_predictor()

            # Normalize data structure
            normalized_games = []
            for game in nfl_games:
                home_team = game.get('home_team')
                away_team = game.get('away_team')

                # Get AI prediction
                ai_prediction = None
                try:
                    prediction = predictor.predict_winner(
                        home_team=home_team,
                        away_team=away_team
                    )
                    if prediction:
                        confidence_map = {'high': 85, 'medium': 70, 'low': 55}
                        ai_prediction = {
                            'pick': prediction.get('winner'),
                            'confidence': confidence_map.get(prediction.get('confidence', 'medium'), 65),
                            'probability': round(prediction.get('probability', 0.5) * 100, 1),
                            'spread': prediction.get('spread', 0),
                            'reasoning': prediction.get('explanation', 'AI model prediction')[:150]
                        }
                except Exception as e:
                    logger.warning(f"Error getting prediction for {home_team} vs {away_team}: {e}")

                normalized_games.append({
                    'id': game.get('game_id'),
                    'league': 'NFL',
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': game.get('home_score'),
                    'away_score': game.get('away_score'),
                    'status': 'Live',
                    'is_live': True,
                    'game_time': f"Q{game.get('quarter')} {game.get('time_remaining')}",
                    'odds': {
                        'spread_home': game.get('spread_home'),
                        'spread_home_odds': game.get('spread_odds_home'),
                        'total': game.get('over_under'),
                        'moneyline_home': game.get('moneyline_home'),
                        'moneyline_away': game.get('moneyline_away')
                    },
                    'ai_prediction': ai_prediction
                })

            # Cache for 30 seconds
            query_cache.set('live_games', normalized_games, ttl_seconds=30)
            return normalized_games

        except Exception as e:
            logger.error(f"Error getting live games: {e}")
            return []

    def get_upcoming_games(self, limit: int = 10) -> List[dict]:
        """Get upcoming games with odds and AI predictions"""
        try:
            cached = query_cache.get(f'upcoming_games_{limit}')
            if cached:
                return cached

            nfl_games = self.nfl_db.get_upcoming_games(hours_ahead=168)  # 7 days
            predictor = get_nfl_predictor()

            normalized_games = []
            for game in nfl_games[:limit]:
                home_team = game.get('home_team')
                away_team = game.get('away_team')

                # Get AI prediction
                ai_prediction = None
                try:
                    prediction = predictor.predict_winner(
                        home_team=home_team,
                        away_team=away_team
                    )
                    if prediction:
                        confidence_map = {'high': 85, 'medium': 70, 'low': 55}
                        probability = prediction.get('probability', 0.5)

                        # Calculate EV based on probability and odds
                        ml_home = game.get('moneyline_home', -110)
                        if ml_home and ml_home != 0:
                            if ml_home > 0:
                                decimal_odds = (ml_home / 100) + 1
                            else:
                                decimal_odds = (100 / abs(ml_home)) + 1
                            implied_prob = 1 / decimal_odds
                            ev = (probability - implied_prob) * 100
                        else:
                            ev = 0

                        ai_prediction = {
                            'pick': prediction.get('winner'),
                            'confidence': confidence_map.get(prediction.get('confidence', 'medium'), 65),
                            'probability': round(probability * 100, 1),
                            'spread': prediction.get('spread', 0),
                            'ev': round(ev, 1),
                            'reasoning': prediction.get('explanation', 'AI model prediction')[:150]
                        }
                except Exception as e:
                    logger.warning(f"Error getting prediction for {home_team} vs {away_team}: {e}")

                normalized_games.append({
                    'id': game.get('game_id'),
                    'league': 'NFL',
                    'home_team': home_team,
                    'away_team': away_team,
                    'status': 'Scheduled',
                    'is_live': False,
                    'game_time': game.get('game_time').strftime('%a %I:%M %p') if game.get('game_time') else 'TBD',
                    'odds': {
                        'spread_home': game.get('spread_home'),
                        'spread_home_odds': game.get('spread_odds_home'),
                        'total': game.get('over_under'),
                        'moneyline_home': game.get('moneyline_home'),
                        'moneyline_away': game.get('moneyline_away')
                    },
                    'ai_prediction': ai_prediction
                })

            query_cache.set(f'upcoming_games_{limit}', normalized_games, ttl_seconds=300)
            return normalized_games

        except Exception as e:
            logger.error(f"Error getting upcoming games: {e}")
            return []

sports_service = SportsService()
