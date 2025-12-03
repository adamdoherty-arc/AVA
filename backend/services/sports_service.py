"""
Sports Service - Optimized with Batch Operations
=================================================

OPTIMIZATIONS APPLIED:
1. Batch UPDATE for odds sync (single query per sport instead of N queries)
2. UNION queries for live/upcoming games (4x fewer queries)
3. Batch predictions for AI analysis
4. Query caching with proper TTLs
"""

from typing import List, Optional
import logging
from psycopg2.extras import RealDictCursor, execute_values
from backend.database.connection import db_pool
from backend.models.market import Market
from backend.config import settings

logger = logging.getLogger(__name__)

from src.nfl_db_manager import NFLDBManager
from src.database.query_cache import query_cache
from src.prediction_agents.nfl_predictor import NFLPredictor
from src.prediction_agents.nba_predictor import NBAPredictor
from src.prediction_agents.ncaa_predictor import NCAAPredictor
from src.odds_api_client import TheOddsAPIClient
import asyncio
from datetime import datetime

# Singleton predictors for efficiency
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

class SportsService:
    def __init__(self):
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
        """Get all currently live games across ALL sports with AI predictions.

        Optimized:
        - Single UNION query instead of 4 separate queries (4x efficiency)
        - Batch predictions instead of N+1 individual predictor calls
        """
        try:
            # Check cache first
            cached = query_cache.get('live_games_all')
            if cached:
                return cached

            normalized_games = []

            with db_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    try:
                        cur.execute("""
                            SELECT 'NFL' as league, game_id, home_team, away_team,
                                   home_score, away_score, quarter, time_remaining,
                                   moneyline_home, moneyline_away, spread_home, over_under,
                                   NULL::int as home_rank, NULL::int as away_rank
                            FROM nfl_games WHERE is_live = true
                            UNION ALL
                            SELECT 'NBA' as league, game_id, home_team, away_team,
                                   home_score, away_score, quarter, time_remaining,
                                   moneyline_home, moneyline_away, spread_home, over_under,
                                   NULL::int as home_rank, NULL::int as away_rank
                            FROM nba_games WHERE is_live = true
                            UNION ALL
                            SELECT 'NCAAF' as league, game_id, home_team, away_team,
                                   home_score, away_score, quarter, time_remaining,
                                   NULL::int as moneyline_home, NULL::int as moneyline_away,
                                   spread_home, over_under, home_rank, away_rank
                            FROM ncaa_football_games WHERE is_live = true
                            UNION ALL
                            SELECT 'NCAAB' as league, game_id, home_team, away_team,
                                   home_score, away_score, half as quarter, time_remaining,
                                   NULL::int as moneyline_home, NULL::int as moneyline_away,
                                   spread_home, over_under, home_rank, away_rank
                            FROM ncaa_basketball_games WHERE is_live = true
                        """)
                        all_live_games = cur.fetchall()

                        # BATCH PREDICTIONS: Group games by league, then batch predict
                        games_by_league = {'NFL': [], 'NBA': [], 'NCAAF': [], 'NCAAB': []}
                        for game in all_live_games:
                            games_by_league[game['league']].append(dict(game))

                        # Get batch predictions for each league
                        predictions_by_game = {}
                        for league, games in games_by_league.items():
                            if not games:
                                continue
                            if league == 'NFL':
                                predictor = get_nfl_predictor()
                            elif league == 'NBA':
                                predictor = get_nba_predictor()
                            else:
                                predictor = get_ncaa_predictor()

                            batch_preds = predictor.predict_batch(games)
                            predictions_by_game.update(batch_preds)

                        # Normalize games with pre-computed predictions
                        for game in all_live_games:
                            league = game['league']
                            game_id = game['game_id']
                            prediction = predictions_by_game.get(game_id)
                            normalized_games.append(
                                self._normalize_game_with_prediction(game, league, prediction, is_live=True)
                            )
                    except Exception as e:
                        logger.warning(f"Error fetching live games via UNION: {e}")

            # Cache for 30 seconds
            query_cache.set('live_games_all', normalized_games, ttl_seconds=30)
            return normalized_games

        except Exception as e:
            logger.error(f"Error getting live games: {e}")
            return []

    def _normalize_game(self, game: dict, league: str, predictor, is_live: bool = False) -> dict:
        """Normalize game data across all sports"""
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
                confidence_map = settings.prediction_confidence_map
                probability = prediction.get('probability', 0.5)

                # Calculate EV if we have odds using correct formula:
                # EV = (p * (decimal_odds - 1)) - (1 - p)
                # This gives true expected value per $1 wagered
                ml_home = game.get('moneyline_home', -110) or -110
                if ml_home and ml_home != 0:
                    if ml_home > 0:
                        decimal_odds = (ml_home / 100) + 1
                    else:
                        decimal_odds = (100 / abs(ml_home)) + 1
                    # Correct EV formula: (probability * profit) - (1 - probability)
                    ev = (probability * (decimal_odds - 1)) - (1 - probability)
                    ev = ev * 100  # Convert to percentage
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

        # Format game time
        if is_live:
            period_name = 'Q' if league in ['NFL', 'NBA'] else ('Q' if league == 'NCAAF' else 'H')
            game_time = f"{period_name}{game.get('quarter', 1)} {game.get('time_remaining', '0:00')}"
        else:
            gt = game.get('game_time')
            if gt:
                try:
                    game_time = gt.strftime('%a %I:%M %p') if hasattr(gt, 'strftime') else str(gt)
                except (ValueError, AttributeError):
                    game_time = str(gt)
            else:
                game_time = 'TBD'

        # Get live game state if available
        quarter = game.get('quarter') or game.get('period') or game.get('half')
        time_remaining = game.get('time_remaining')
        possession = game.get('possession')
        down_distance = game.get('down_distance')
        yard_line = game.get('yard_line')

        return {
            'id': game.get('game_id'),
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': game.get('home_score'),
            'away_score': game.get('away_score'),
            'status': 'Live' if is_live else 'Scheduled',
            'isLive': is_live,  # Use camelCase for frontend compatibility
            'game_time': game_time,
            'home_rank': game.get('home_rank'),
            'away_rank': game.get('away_rank'),
            # Live game state fields
            'quarter': quarter,
            'time_remaining': time_remaining,
            'possession': possession,
            'down_distance': down_distance,
            'yard_line': yard_line,
            'odds': {
                'spread_home': game.get('spread_home'),
                'spread_home_odds': game.get('spread_odds_home'),
                'total': game.get('over_under'),
                'moneyline_home': game.get('moneyline_home'),
                'moneyline_away': game.get('moneyline_away')
            },
            'ai_prediction': ai_prediction
        }

    def _normalize_game_with_prediction(
        self, game: dict, league: str, prediction: dict, is_live: bool = False
    ) -> dict:
        """Normalize game with pre-computed prediction (batch optimization)."""
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        # Build AI prediction from pre-computed result
        ai_prediction = None
        if prediction:
            probability = prediction.get('probability', 0.5)
            confidence_map = settings.prediction_confidence_map

            # Calculate EV using correct formula
            ml_home = game.get('moneyline_home', -110) or -110
            if ml_home and ml_home != 0:
                if ml_home > 0:
                    decimal_odds = (ml_home / 100) + 1
                else:
                    decimal_odds = (100 / abs(ml_home)) + 1
                ev = (probability * (decimal_odds - 1)) - (1 - probability)
                ev = ev * 100
            else:
                ev = 0

            ai_prediction = {
                'pick': prediction.get('winner'),
                'confidence': confidence_map.get(
                    prediction.get('confidence', 'medium'), 65
                ),
                'probability': round(probability * 100, 1),
                'spread': prediction.get('spread', 0),
                'ev': round(ev, 1),
                'reasoning': prediction.get('explanation', '')[:150]
            }

        # Format game time
        if is_live:
            period = 'Q' if league in ['NFL', 'NBA', 'NCAAF'] else 'H'
            game_time = f"{period}{game.get('quarter', 1)} {game.get('time_remaining', '')}"
        else:
            gt = game.get('game_time')
            if gt:
                try:
                    game_time = gt.strftime('%a %I:%M %p') if hasattr(gt, 'strftime') else str(gt)
                except (ValueError, AttributeError):
                    game_time = str(gt)
            else:
                game_time = 'TBD'

        return {
            'id': game.get('game_id'),
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': game.get('home_score'),
            'away_score': game.get('away_score'),
            'status': 'Live' if is_live else 'Scheduled',
            'isLive': is_live,
            'game_time': game_time,
            'home_rank': game.get('home_rank'),
            'away_rank': game.get('away_rank'),
            'quarter': game.get('quarter') or game.get('half'),
            'time_remaining': game.get('time_remaining'),
            'possession': game.get('possession'),
            'odds': {
                'spread_home': game.get('spread_home'),
                'spread_home_odds': game.get('spread_odds_home'),
                'total': game.get('over_under'),
                'moneyline_home': game.get('moneyline_home'),
                'moneyline_away': game.get('moneyline_away')
            },
            'ai_prediction': ai_prediction
        }

    def get_upcoming_games(self, limit: int = 10) -> List[dict]:
        """Get upcoming games across ALL sports with odds and AI predictions.

        Optimized: Uses single UNION query instead of 4 separate queries for 4x efficiency.
        """
        try:
            cached = query_cache.get(f'upcoming_games_all_{limit}')
            if cached:
                return cached

            normalized_games = []
            # Limit per sport to ensure fair distribution
            per_sport_limit = max(5, limit // 2)

            with db_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # OPTIMIZED: Single UNION query instead of 4 separate queries
                    try:
                        cur.execute("""
                            (SELECT 'NFL' as league, game_id, home_team, away_team, game_time,
                                    moneyline_home, moneyline_away, spread_home, over_under,
                                    NULL::int as home_rank, NULL::int as away_rank
                             FROM nfl_games
                             WHERE game_status = 'scheduled' AND game_time > NOW()
                             ORDER BY game_time LIMIT %s)
                            UNION ALL
                            (SELECT 'NBA' as league, game_id, home_team, away_team, game_time,
                                    moneyline_home, moneyline_away, spread_home, over_under,
                                    NULL::int as home_rank, NULL::int as away_rank
                             FROM nba_games
                             WHERE game_status = 'scheduled' AND game_time > NOW()
                             ORDER BY game_time LIMIT %s)
                            UNION ALL
                            (SELECT 'NCAAF' as league, game_id, home_team, away_team, game_time,
                                    NULL::int as moneyline_home, NULL::int as moneyline_away,
                                    spread_home, over_under, home_rank, away_rank
                             FROM ncaa_football_games
                             WHERE game_status = 'scheduled' AND game_time > NOW()
                             ORDER BY game_time LIMIT %s)
                            UNION ALL
                            (SELECT 'NCAAB' as league, game_id, home_team, away_team, game_time,
                                    NULL::int as moneyline_home, NULL::int as moneyline_away,
                                    spread_home, over_under, home_rank, away_rank
                             FROM ncaa_basketball_games
                             WHERE game_status = 'scheduled' AND game_time > NOW()
                             ORDER BY game_time LIMIT %s)
                            ORDER BY game_time
                        """, (per_sport_limit, per_sport_limit, per_sport_limit, per_sport_limit))
                        all_upcoming_games = cur.fetchall()

                        # BATCH PREDICTIONS: Group by league, then batch predict
                        games_by_league = {'NFL': [], 'NBA': [], 'NCAAF': [], 'NCAAB': []}
                        for game in all_upcoming_games:
                            games_by_league[game['league']].append(dict(game))

                        predictions_by_game = {}
                        for league, games in games_by_league.items():
                            if not games:
                                continue
                            if league == 'NFL':
                                predictor = get_nfl_predictor()
                            elif league == 'NBA':
                                predictor = get_nba_predictor()
                            else:
                                predictor = get_ncaa_predictor()

                            batch_preds = predictor.predict_batch(games)
                            predictions_by_game.update(batch_preds)

                        # Normalize with pre-computed predictions
                        for game in all_upcoming_games:
                            league = game['league']
                            game_id = game['game_id']
                            prediction = predictions_by_game.get(game_id)
                            normalized_games.append(
                                self._normalize_game_with_prediction(
                                    game, league, prediction, is_live=False
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Error fetching upcoming games via UNION: {e}")

            # Already sorted by game_time in SQL
            # Limit total results
            normalized_games = normalized_games[:limit * 4]

            query_cache.set(f'upcoming_games_all_{limit}', normalized_games, ttl_seconds=300)
            return normalized_games

        except Exception as e:
            logger.error(f"Error getting upcoming games: {e}")
            return []

    async def sync_odds_from_api(self, sports: List[str] = None) -> dict:
        """
        Sync real odds from The Odds API to game tables.

        Uses ~4 API requests (1 per sport). With 500/month limit:
        - Daily sync: ~4 req/day = 120/month
        - Hourly sync: ~96 req/day (too many)
        - Every 6 hours: ~16 req/day = 480/month (safe)

        Args:
            sports: List of sports to sync (default: NFL, NBA, NCAAF, NCAAB)

        Returns:
            Sync result dict with counts and quota info
        """
        if sports is None:
            sports = ['NFL', 'NBA', 'NCAAF', 'NCAAB']

        result = {
            'success': True,
            'synced_at': datetime.now().isoformat(),
            'sports': {},
            'totals': {'updated': 0, 'skipped': 0, 'errors': 0},
            'quota': None
        }

        try:
            client = TheOddsAPIClient()

            for sport in sports:
                try:
                    # Fetch normalized odds from API
                    odds_data = await client.get_normalized_odds(sport)
                    logger.info(f"Fetched {len(odds_data)} {sport} games with odds")

                    updated = 0
                    skipped = 0

                    # BATCH UPDATE: Prepare all updates, execute single query
                    table_map = {
                        'NFL': 'nfl_games',
                        'NBA': 'nba_games',
                        'NCAAF': 'ncaa_football_games',
                        'NCAAB': 'ncaa_basketball_games'
                    }
                    table = table_map.get(sport)
                    if not table:
                        continue

                    # Prepare batch update data
                    batch_data = []
                    for game_odds in odds_data:
                        home_team = game_odds.get('home_team', '')
                        away_team = game_odds.get('away_team', '')
                        if not home_team or not away_team:
                            skipped += 1
                            continue

                        # Extract team name patterns for matching
                        home_pattern = home_team.split()[-1].lower()
                        away_pattern = away_team.split()[-1].lower()

                        batch_data.append((
                            game_odds.get('moneyline_home'),
                            game_odds.get('moneyline_away'),
                            game_odds.get('spread_home'),
                            game_odds.get('spread_odds_home'),
                            game_odds.get('spread_odds_away'),
                            game_odds.get('over_under'),
                            game_odds.get('over_odds'),
                            game_odds.get('under_odds'),
                            f'%{home_pattern}%',
                            f'%{away_pattern}%'
                        ))

                    if not batch_data:
                        result['sports'][sport] = {
                            'fetched': len(odds_data),
                            'updated': 0,
                            'skipped': skipped
                        }
                        continue

                    # Execute BATCH UPDATE with CTE
                    with db_pool.get_connection() as conn:
                        with conn.cursor(cursor_factory=RealDictCursor) as cur:
                            try:
                                # Use execute_values for efficient batch insert
                                # into temp table, then UPDATE FROM
                                batch_query = f"""
                                    WITH odds_updates AS (
                                        SELECT * FROM (
                                            VALUES %s
                                        ) AS t(
                                            ml_home, ml_away, spread,
                                            spread_odds_h, spread_odds_a,
                                            total, over_o, under_o,
                                            home_pat, away_pat
                                        )
                                    )
                                    UPDATE {table} g
                                    SET
                                        moneyline_home = COALESCE(
                                            u.ml_home::int, g.moneyline_home
                                        ),
                                        moneyline_away = COALESCE(
                                            u.ml_away::int, g.moneyline_away
                                        ),
                                        spread_home = COALESCE(
                                            u.spread::numeric, g.spread_home
                                        ),
                                        spread_odds_home = COALESCE(
                                            u.spread_odds_h::int, g.spread_odds_home
                                        ),
                                        spread_odds_away = COALESCE(
                                            u.spread_odds_a::int, g.spread_odds_away
                                        ),
                                        over_under = COALESCE(
                                            u.total::numeric, g.over_under
                                        ),
                                        over_odds = COALESCE(
                                            u.over_o::int, g.over_odds
                                        ),
                                        under_odds = COALESCE(
                                            u.under_o::int, g.under_odds
                                        ),
                                        last_synced = NOW()
                                    FROM odds_updates u
                                    WHERE g.game_status = 'scheduled'
                                      AND LOWER(g.home_team) LIKE u.home_pat
                                      AND LOWER(g.away_team) LIKE u.away_pat
                                """
                                execute_values(cur, batch_query, batch_data)
                                updated = cur.rowcount
                                conn.commit()
                                logger.info(
                                    f"Batch updated {updated} {sport} games"
                                )
                            except Exception as e:
                                logger.error(f"Batch update failed: {e}")
                                result['totals']['errors'] += 1
                                conn.rollback()

                    result['sports'][sport] = {
                        'fetched': len(odds_data),
                        'updated': updated,
                        'skipped': skipped
                    }
                    result['totals']['updated'] += updated
                    result['totals']['skipped'] += skipped

                except Exception as e:
                    logger.error(f"Error syncing {sport} odds: {e}")
                    result['sports'][sport] = {'error': str(e)}
                    result['totals']['errors'] += 1

            # Get quota status
            result['quota'] = client.get_quota_status()

            await client.close()

            # Clear relevant caches
            query_cache.invalidate()

            logger.info(f"Odds sync complete: {result['totals']}")
            return result

        except Exception as e:
            logger.error(f"Failed to sync odds: {e}")
            return {
                'success': False,
                'error': str(e),
                'synced_at': datetime.now().isoformat()
            }

    def sync_odds(self, sports: List[str] = None) -> dict:
        """Synchronous wrapper for odds sync."""
        return asyncio.run(self.sync_odds_from_api(sports))

    def get_odds_quota(self) -> dict:
        """Get current API quota status without making a request."""
        try:
            client = TheOddsAPIClient()
            return client.get_quota_status()
        except Exception as e:
            return {'error': str(e)}


sports_service = SportsService()
