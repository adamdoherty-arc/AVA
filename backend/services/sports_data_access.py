"""
Sports Data Access Layer
=========================

Modern async data access for sports betting features.

Uses:
- AsyncDatabaseManager for true async I/O
- RedisCache for distributed caching with stampede protection
- Batch operations for efficiency
- Type-safe query building

Author: AVA Trading Platform
Updated: 2025-11-30
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from backend.infrastructure.database import get_database, AsyncDatabaseManager
from backend.infrastructure.cache import get_cache, cached, RedisCache

logger = logging.getLogger(__name__)


class Sport(str, Enum):
    """Supported sports."""
    NFL = "NFL"
    NBA = "NBA"
    NCAAF = "NCAAF"
    NCAAB = "NCAAB"
    MLB = "MLB"
    NHL = "NHL"


class GameStatus(str, Enum):
    """Game status values."""
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


@dataclass
class GameOdds:
    """Odds data for a game."""
    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None
    spread_home: Optional[float] = None
    spread_odds_home: Optional[int] = None
    spread_odds_away: Optional[int] = None
    over_under: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    bookmaker: Optional[str] = None
    last_updated: Optional[datetime] = None


@dataclass
class Game:
    """Unified game representation across all sports."""
    game_id: str
    sport: Sport
    home_team: str
    away_team: str
    home_team_abbr: Optional[str] = None
    away_team_abbr: Optional[str] = None
    game_time: Optional[datetime] = None
    game_status: GameStatus = GameStatus.SCHEDULED
    venue: Optional[str] = None

    # Scores
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    # Live game state
    is_live: bool = False
    period: Optional[int] = None
    time_remaining: Optional[str] = None

    # Rankings (for NCAA)
    home_rank: Optional[int] = None
    away_rank: Optional[int] = None

    # Odds
    odds: Optional[GameOdds] = None

    # Prediction
    prediction: Optional[Dict[str, Any]] = None


# =============================================================================
# Sport-Specific Query Builders
# =============================================================================

def _get_sport_table(sport: Sport) -> str:
    """Get the database table name for a sport."""
    tables = {
        Sport.NFL: "nfl_games",
        Sport.NBA: "nba_games",
        Sport.NCAAF: "ncaa_football_games",
        Sport.NCAAB: "ncaa_basketball_games",
    }
    return tables.get(sport, "nfl_games")


def _get_period_column(sport: Sport) -> str:
    """Get the period column name for a sport."""
    return "half" if sport == Sport.NCAAB else "quarter"


# =============================================================================
# Async Sports Data Access
# =============================================================================

class SportsDataAccess:
    """
    Modern async data access layer for sports betting.

    Features:
    - True async database operations (no thread pool blocking)
    - Distributed Redis caching with stampede protection
    - Batch operations for efficiency
    - Type-safe game objects
    """

    # Cache TTLs (seconds)
    CACHE_TTL_LIVE = 30        # Live games refresh every 30s
    CACHE_TTL_UPCOMING = 300   # Upcoming games cache 5 min
    CACHE_TTL_ODDS = 120       # Odds cache 2 min
    CACHE_TTL_PREDICTIONS = 600  # Predictions cache 10 min

    def __init__(self, db: Optional[AsyncDatabaseManager] = None):
        self._db = db
        self._cache: Optional[RedisCache] = None

    async def _get_db(self) -> AsyncDatabaseManager:
        """Get database connection (lazy init)."""
        if self._db is None:
            self._db = await get_database()
        return self._db

    def _get_cache(self) -> RedisCache:
        """Get cache instance (lazy init)."""
        if self._cache is None:
            self._cache = get_cache()
        return self._cache

    # =========================================================================
    # Game Queries
    # =========================================================================

    async def get_live_games(self, sports: Optional[List[Sport]] = None) -> List[Game]:
        """
        Get all live games across sports.

        Uses UNION query for efficiency (single round-trip).
        Results are cached for 30 seconds.
        """
        cache_key = f"live_games:{','.join(s.value for s in (sports or []))}"
        cache = self._get_cache()

        # Use get_or_fetch for stampede protection
        async def fetch_live():
            db = await self._get_db()

            # Build UNION query for all requested sports
            target_sports = sports or [Sport.NFL, Sport.NBA, Sport.NCAAF, Sport.NCAAB]
            union_parts = []

            for sport in target_sports:
                table = _get_sport_table(sport)
                period_col = _get_period_column(sport)

                union_parts.append(f"""
                    SELECT
                        '{sport.value}' as sport,
                        game_id, home_team, away_team,
                        home_team_abbr, away_team_abbr,
                        game_time, game_status, venue,
                        home_score, away_score,
                        is_live, {period_col} as period, time_remaining,
                        moneyline_home, moneyline_away,
                        spread_home, over_under,
                        {'home_rank, away_rank' if sport in [Sport.NCAAF, Sport.NCAAB] else 'NULL::int as home_rank, NULL::int as away_rank'}
                    FROM {table}
                    WHERE is_live = true
                """)

            query = " UNION ALL ".join(union_parts) + " ORDER BY game_time"
            rows = await db.fetch(query)

            return [self._row_to_game(dict(row)) for row in rows]

        return await cache.get_or_fetch(cache_key, fetch_live, self.CACHE_TTL_LIVE)

    async def get_upcoming_games(
        self,
        sports: Optional[List[Sport]] = None,
        limit: int = 50,
        hours_ahead: int = 168  # 1 week
    ) -> List[Game]:
        """
        Get upcoming scheduled games.

        Args:
            sports: Filter to specific sports
            limit: Max games to return
            hours_ahead: How far ahead to look
        """
        cache_key = f"upcoming_games:{','.join(s.value for s in (sports or []))}:{limit}:{hours_ahead}"
        cache = self._get_cache()

        async def fetch_upcoming():
            db = await self._get_db()
            target_sports = sports or [Sport.NFL, Sport.NBA, Sport.NCAAF, Sport.NCAAB]

            cutoff_time = datetime.now() + timedelta(hours=hours_ahead)
            union_parts = []

            for sport in target_sports:
                table = _get_sport_table(sport)
                period_col = _get_period_column(sport)

                # Use timestamp literal instead of parameter in UNION
                cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
                union_parts.append(f"""
                    SELECT
                        '{sport.value}' as sport,
                        game_id, home_team, away_team,
                        home_team_abbr, away_team_abbr,
                        game_time, game_status, venue,
                        home_score, away_score,
                        is_live, {period_col} as period, time_remaining,
                        moneyline_home, moneyline_away,
                        spread_home, over_under,
                        {'home_rank, away_rank' if sport in [Sport.NCAAF, Sport.NCAAB] else 'NULL::int as home_rank, NULL::int as away_rank'}
                    FROM {table}
                    WHERE game_status = 'scheduled'
                      AND game_time > NOW()
                      AND game_time < '{cutoff_str}'::timestamp
                """)

            query = f"""
                ({" UNION ALL ".join(union_parts)})
                ORDER BY game_time
                LIMIT {limit}
            """

            rows = await db.fetch(query)
            return [self._row_to_game(dict(row)) for row in rows]

        return await cache.get_or_fetch(cache_key, fetch_upcoming, self.CACHE_TTL_UPCOMING)

    async def get_games_with_odds(
        self,
        sports: Optional[List[Sport]] = None,
        status: Optional[List[GameStatus]] = None,
        min_odds_age_minutes: int = 60
    ) -> List[Game]:
        """
        Get games that have odds data.

        Args:
            sports: Filter to specific sports
            status: Filter to specific statuses
            min_odds_age_minutes: Only include games with odds updated within N minutes
        """
        cache_key = f"games_with_odds:{','.join(s.value for s in (sports or []))}:{min_odds_age_minutes}"
        cache = self._get_cache()

        async def fetch_games():
            db = await self._get_db()
            target_sports = sports or [Sport.NFL, Sport.NBA, Sport.NCAAF, Sport.NCAAB]
            target_status = status or [GameStatus.SCHEDULED, GameStatus.LIVE]

            # Build status list for SQL IN clause
            status_list = ", ".join(f"'{s.value}'" for s in target_status)

            union_parts = []
            for sport in target_sports:
                table = _get_sport_table(sport)
                period_col = _get_period_column(sport)

                union_parts.append(f"""
                    SELECT
                        '{sport.value}' as sport,
                        game_id, home_team, away_team,
                        home_team_abbr, away_team_abbr,
                        game_time, game_status, venue,
                        home_score, away_score,
                        is_live, {period_col} as period, time_remaining,
                        moneyline_home, moneyline_away,
                        spread_home, over_under,
                        {'home_rank, away_rank' if sport in [Sport.NCAAF, Sport.NCAAB] else 'NULL::int as home_rank, NULL::int as away_rank'}
                    FROM {table}
                    WHERE game_status IN ({status_list})
                      AND (moneyline_home IS NOT NULL OR spread_home IS NOT NULL)
                """)

            query = f"""
                ({" UNION ALL ".join(union_parts)})
                ORDER BY is_live DESC, game_time ASC
            """

            rows = await db.fetch(query)
            return [self._row_to_game(dict(row)) for row in rows]

        return await cache.get_or_fetch(cache_key, fetch_games, self.CACHE_TTL_ODDS)

    async def get_game_by_id(self, game_id: str, sport: Sport) -> Optional[Game]:
        """Get a specific game by ID."""
        db = await self._get_db()
        table = _get_sport_table(sport)
        period_col = _get_period_column(sport)

        rank_cols = "home_rank, away_rank" if sport in [Sport.NCAAF, Sport.NCAAB] else "NULL::int as home_rank, NULL::int as away_rank"

        query = f"""
            SELECT
                '{sport.value}' as sport,
                game_id, home_team, away_team,
                home_team_abbr, away_team_abbr,
                game_time, game_status, venue,
                home_score, away_score,
                is_live, {period_col} as period, time_remaining,
                moneyline_home, moneyline_away,
                spread_home, over_under,
                {rank_cols}
            FROM {table}
            WHERE game_id = $1
        """

        row = await db.fetchrow(query, game_id)
        return self._row_to_game(dict(row)) if row else None

    # =========================================================================
    # Odds Updates
    # =========================================================================

    async def update_game_odds(
        self,
        game_id: str,
        sport: Sport,
        odds: GameOdds
    ) -> bool:
        """Update odds for a specific game."""
        db = await self._get_db()
        table = _get_sport_table(sport)

        query = f"""
            UPDATE {table}
            SET
                moneyline_home = $2,
                moneyline_away = $3,
                spread_home = $4,
                over_under = $5,
                last_synced = NOW()
            WHERE game_id = $1
        """

        result = await db.execute(
            query,
            game_id,
            odds.moneyline_home,
            odds.moneyline_away,
            odds.spread_home,
            odds.over_under
        )

        # Invalidate cache
        await self._get_cache().invalidate_pattern(f"*games*")

        return "UPDATE 1" in result

    async def batch_update_odds(
        self,
        updates: List[Tuple[str, Sport, GameOdds]]
    ) -> int:
        """
        Batch update odds for multiple games.

        Uses executemany for efficiency.
        """
        if not updates:
            return 0

        db = await self._get_db()

        # Group by sport for batch updates
        by_sport: Dict[Sport, List[Tuple[str, GameOdds]]] = {}
        for game_id, sport, odds in updates:
            if sport not in by_sport:
                by_sport[sport] = []
            by_sport[sport].append((game_id, odds))

        total_updated = 0

        for sport, sport_updates in by_sport.items():
            table = _get_sport_table(sport)

            # Build batch update values
            values = [
                (
                    game_id,
                    odds.moneyline_home,
                    odds.moneyline_away,
                    odds.spread_home,
                    odds.over_under
                )
                for game_id, odds in sport_updates
            ]

            # Use executemany for batch update
            query = f"""
                UPDATE {table}
                SET
                    moneyline_home = $2,
                    moneyline_away = $3,
                    spread_home = $4,
                    over_under = $5,
                    last_synced = NOW()
                WHERE game_id = $1
            """

            await db.executemany(query, values)
            total_updated += len(values)

        # Invalidate cache
        await self._get_cache().invalidate_pattern(f"*games*")

        logger.info(f"Batch updated odds for {total_updated} games")
        return total_updated

    # =========================================================================
    # Prediction Caching
    # =========================================================================

    async def cache_prediction(
        self,
        game_id: str,
        sport: Sport,
        prediction: Dict[str, Any]
    ) -> None:
        """Cache a prediction for a game."""
        cache_key = f"prediction:{sport.value}:{game_id}"
        cache = self._get_cache()
        await cache.set(cache_key, prediction, self.CACHE_TTL_PREDICTIONS)

    async def get_cached_prediction(
        self,
        game_id: str,
        sport: Sport
    ) -> Optional[Dict[str, Any]]:
        """Get cached prediction for a game."""
        cache_key = f"prediction:{sport.value}:{game_id}"
        cache = self._get_cache()
        return await cache.get(cache_key)

    async def get_cached_predictions_batch(
        self,
        games: List[Tuple[str, Sport]]
    ) -> Dict[str, Dict[str, Any]]:
        """Get cached predictions for multiple games."""
        cache = self._get_cache()
        results = {}

        for game_id, sport in games:
            cache_key = f"prediction:{sport.value}:{game_id}"
            cached = await cache.get(cache_key)
            if cached:
                results[game_id] = cached

        return results

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_game_counts(self) -> Dict[str, Dict[str, int]]:
        """Get game counts by sport and status."""
        db = await self._get_db()

        result = {}
        for sport in [Sport.NFL, Sport.NBA, Sport.NCAAF, Sport.NCAAB]:
            table = _get_sport_table(sport)

            query = f"""
                SELECT
                    game_status,
                    COUNT(*) as count
                FROM {table}
                GROUP BY game_status
            """

            rows = await db.fetch(query)
            result[sport.value] = {
                row['game_status']: row['count']
                for row in rows
            }

        return result

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_game(self, row: Dict[str, Any]) -> Game:
        """Convert database row to Game object."""
        odds = GameOdds(
            moneyline_home=row.get('moneyline_home'),
            moneyline_away=row.get('moneyline_away'),
            spread_home=row.get('spread_home'),
            over_under=row.get('over_under'),
        )

        return Game(
            game_id=row['game_id'],
            sport=Sport(row['sport']),
            home_team=row['home_team'],
            away_team=row['away_team'],
            home_team_abbr=row.get('home_team_abbr'),
            away_team_abbr=row.get('away_team_abbr'),
            game_time=row.get('game_time'),
            game_status=GameStatus(row.get('game_status', 'scheduled')),
            venue=row.get('venue'),
            home_score=row.get('home_score'),
            away_score=row.get('away_score'),
            is_live=row.get('is_live', False),
            period=row.get('period'),
            time_remaining=row.get('time_remaining'),
            home_rank=row.get('home_rank'),
            away_rank=row.get('away_rank'),
            odds=odds,
        )


# =============================================================================
# Singleton Instance
# =============================================================================

_data_access: Optional[SportsDataAccess] = None


async def get_sports_data_access() -> SportsDataAccess:
    """Get or create the global sports data access instance."""
    global _data_access
    if _data_access is None:
        _data_access = SportsDataAccess()
    return _data_access


# =============================================================================
# FastAPI Dependency
# =============================================================================

async def sports_data_dependency() -> SportsDataAccess:
    """FastAPI dependency for sports data access."""
    return await get_sports_data_access()
