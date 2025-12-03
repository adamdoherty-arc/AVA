"""
Sports Betting Infrastructure - Modern Integration Layer
=========================================================

Integrates all modern infrastructure components for sports betting:
- Async database operations with asyncpg
- Circuit breakers for external API resilience
- Redis caching with stampede protection
- WebSocket broadcasting for real-time updates
- Server-Sent Events (SSE) for odds streaming
- AI ensemble probability blending

USAGE:
    from backend.services.sports_infrastructure import sports_infra

    # Async database query
    games = await sports_infra.fetch_live_games()

    # Cached data with stampede protection
    odds = await sports_infra.get_cached_odds("NFL")

    # Broadcast live update
    await sports_infra.broadcast_live_update(game_data)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
import json

# Infrastructure imports
from backend.infrastructure.async_db import get_async_db, AsyncDatabasePool
from backend.infrastructure.circuit_breaker import (
    CircuitBreaker, circuit_protected, CircuitBreakerError
)
from backend.infrastructure.cache import get_cache, RedisCache
from backend.infrastructure.websocket_manager import get_ws_manager, WebSocketManager

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breakers for External APIs
# =============================================================================

# ESPN API circuit breaker (high threshold - reliable API)
espn_breaker = CircuitBreaker(
    name="espn",
    failure_threshold=10,
    recovery_timeout=120,
    half_open_max_calls=3
)

# Kalshi API circuit breaker (prediction markets)
kalshi_breaker = CircuitBreaker(
    name="kalshi",
    failure_threshold=5,
    recovery_timeout=60,
    half_open_max_calls=2
)

# The Odds API circuit breaker (precious quota)
odds_api_breaker = CircuitBreaker(
    name="odds_api",
    failure_threshold=3,
    recovery_timeout=300,  # 5 minutes - protect quota
    half_open_max_calls=1
)


# =============================================================================
# AI Probability Ensemble Methods
# =============================================================================

@dataclass
class ProbabilityBlend:
    """Result of ensemble probability blending."""
    final_probability: float
    model_weight: float
    market_weight: float
    sources: Dict[str, float]
    confidence_adjustment: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_probability": self.final_probability,
            "model_weight": self.model_weight,
            "market_weight": self.market_weight,
            "sources": self.sources,
            "confidence_adjustment": self.confidence_adjustment
        }


def bayesian_probability_blend(
    model_prob: float,
    implied_prob: float,
    model_confidence: float = 0.7,
    market_efficiency: float = 0.85,
    historical_accuracy: float = 0.65
) -> ProbabilityBlend:
    """
    Bayesian Model Averaging for probability estimation.

    Combines AI model prediction with market-implied probability using
    a sophisticated weighting scheme based on:
    - Model confidence (how sure the AI is)
    - Market efficiency (how accurate markets typically are)
    - Historical model accuracy

    Formula:
        w_model = model_confidence * historical_accuracy
        w_market = market_efficiency * (1 - w_model)
        P_final = (w_model * model_prob + w_market * implied_prob) / (w_model + w_market)

    Args:
        model_prob: AI model's predicted probability (0-1)
        implied_prob: Market-implied probability from odds (0-1)
        model_confidence: Model's self-reported confidence (0-1)
        market_efficiency: Assumed market efficiency (0-1)
        historical_accuracy: Model's historical accuracy (0-1)

    Returns:
        ProbabilityBlend with final probability and metadata
    """
    # Calculate dynamic weights
    model_weight = model_confidence * historical_accuracy
    market_weight = market_efficiency * (1 - model_weight * 0.3)  # Reduce market weight when model is confident

    # Normalize weights
    total_weight = model_weight + market_weight
    model_weight_norm = model_weight / total_weight
    market_weight_norm = market_weight / total_weight

    # Blend probabilities
    blended_prob = (model_weight_norm * model_prob + market_weight_norm * implied_prob)

    # Apply confidence adjustment - shrink extreme predictions toward 0.5
    confidence_adjustment = 1.0 - (1.0 - model_confidence) * 0.3
    adjusted_prob = 0.5 + (blended_prob - 0.5) * confidence_adjustment

    # Clamp to valid range
    final_prob = max(0.01, min(0.99, adjusted_prob))

    return ProbabilityBlend(
        final_probability=round(final_prob, 4),
        model_weight=round(model_weight_norm, 3),
        market_weight=round(market_weight_norm, 3),
        sources={
            "model": round(model_prob, 4),
            "market": round(implied_prob, 4)
        },
        confidence_adjustment=round(confidence_adjustment, 3)
    )


def ensemble_edge_detection(
    model_prob: float,
    implied_prob: float,
    min_edge_threshold: float = 0.03,
    max_confidence_level: float = 0.85
) -> Dict[str, Any]:
    """
    Detect profitable betting edges using ensemble analysis.

    Implements multiple edge detection strategies:
    1. Raw edge: model_prob - implied_prob
    2. Confidence-adjusted edge: edge * confidence_factor
    3. Kelly-optimal edge: Only edges that pass Kelly criterion

    Args:
        model_prob: AI model probability
        implied_prob: Market implied probability
        min_edge_threshold: Minimum edge to consider (3% default)
        max_confidence_level: Maximum confidence cap

    Returns:
        Dict with edge analysis and recommendation
    """
    raw_edge = model_prob - implied_prob

    # Confidence factor based on edge magnitude (larger edges are rarer)
    edge_magnitude = abs(raw_edge)
    confidence_factor = max_confidence_level * (1 - edge_magnitude * 0.5)

    adjusted_edge = raw_edge * confidence_factor

    # Kelly criterion check
    kelly_threshold = implied_prob * 0.5  # Half Kelly threshold
    passes_kelly = raw_edge > kelly_threshold

    # Determine edge quality
    if raw_edge > 0.10:
        edge_quality = "exceptional"  # Very rare, check for errors
    elif raw_edge > 0.05:
        edge_quality = "strong"
    elif raw_edge > min_edge_threshold:
        edge_quality = "moderate"
    elif raw_edge > 0:
        edge_quality = "marginal"
    else:
        edge_quality = "negative"

    # Recommendation
    if raw_edge > min_edge_threshold and passes_kelly:
        recommendation = "bet"
        bet_strength = min(1.0, raw_edge / 0.10)  # Scale 0-1
    elif raw_edge > 0.01:
        recommendation = "monitor"
        bet_strength = 0.0
    else:
        recommendation = "skip"
        bet_strength = 0.0

    return {
        "raw_edge_percent": round(raw_edge * 100, 2),
        "adjusted_edge_percent": round(adjusted_edge * 100, 2),
        "edge_quality": edge_quality,
        "confidence_factor": round(confidence_factor, 3),
        "passes_kelly": passes_kelly,
        "recommendation": recommendation,
        "bet_strength": round(bet_strength, 3)
    }


# =============================================================================
# Sports Infrastructure Class
# =============================================================================

class SportsInfrastructure:
    """
    Centralized sports betting infrastructure manager.

    Provides:
    - Async database operations with connection pooling
    - Circuit breaker-protected external API calls
    - Redis caching with stampede protection
    - WebSocket broadcasting for real-time updates
    - SSE streaming for odds updates
    - AI ensemble probability blending
    """

    # Cache TTLs (seconds)
    TTL_LIVE_GAMES = 30       # 30 seconds for live data
    TTL_UPCOMING = 300        # 5 minutes for upcoming games
    TTL_ODDS = 120            # 2 minutes for odds
    TTL_PREDICTIONS = 600     # 10 minutes for AI predictions

    def __init__(self):
        self._db: Optional[AsyncDatabasePool] = None
        self._cache: Optional[RedisCache] = None
        self._ws_manager: Optional[WebSocketManager] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all infrastructure components."""
        if self._initialized:
            return True

        try:
            # Initialize async database
            self._db = await get_async_db()

            # Initialize cache
            self._cache = get_cache()
            await self._cache.connect()

            # Initialize WebSocket manager
            self._ws_manager = get_ws_manager()

            self._initialized = True
            logger.info("Sports infrastructure initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize sports infrastructure: {e}")
            return False

    @property
    def db(self) -> AsyncDatabasePool:
        """Get async database pool."""
        if not self._db:
            raise RuntimeError("Sports infrastructure not initialized")
        return self._db

    @property
    def cache(self) -> RedisCache:
        """Get Redis cache."""
        if not self._cache:
            raise RuntimeError("Sports infrastructure not initialized")
        return self._cache

    @property
    def ws(self) -> WebSocketManager:
        """Get WebSocket manager."""
        if not self._ws_manager:
            raise RuntimeError("Sports infrastructure not initialized")
        return self._ws_manager

    # =========================================================================
    # Async Database Operations
    # =========================================================================

    async def fetch_live_games(self) -> List[Dict[str, Any]]:
        """
        Fetch all live games across sports using true async database.

        Uses UNION query for efficiency - 1 query instead of 4.
        """
        cache_key = "sports:live_games"

        # Use cache with stampede protection
        async def fetch_from_db():
            query = """
                SELECT 'NFL' as league, game_id, home_team, away_team,
                       home_score, away_score, quarter, time_remaining,
                       moneyline_home, moneyline_away, spread_home, over_under
                FROM nfl_games WHERE is_live = true
                UNION ALL
                SELECT 'NBA' as league, game_id, home_team, away_team,
                       home_score, away_score, quarter, time_remaining,
                       moneyline_home, moneyline_away, spread_home, over_under
                FROM nba_games WHERE is_live = true
                UNION ALL
                SELECT 'NCAAF' as league, game_id, home_team, away_team,
                       home_score, away_score, quarter, time_remaining,
                       NULL::int, NULL::int, spread_home, over_under
                FROM ncaa_football_games WHERE is_live = true
                UNION ALL
                SELECT 'NCAAB' as league, game_id, home_team, away_team,
                       home_score, away_score, half, time_remaining,
                       NULL::int, NULL::int, spread_home, over_under
                FROM ncaa_basketball_games WHERE is_live = true
            """
            return await self.db.fetch(query)

        return await self.cache.get_or_fetch(
            cache_key,
            fetch_from_db,
            ttl=self.TTL_LIVE_GAMES
        )

    async def fetch_upcoming_games(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch upcoming games with async database and caching."""
        cache_key = f"sports:upcoming_games:{limit}"
        per_sport = max(5, limit // 4)

        async def fetch_from_db():
            query = """
                (SELECT 'NFL' as league, game_id, home_team, away_team, game_time,
                        moneyline_home, moneyline_away, spread_home, over_under
                 FROM nfl_games
                 WHERE game_status = 'scheduled' AND game_time > NOW()
                 ORDER BY game_time LIMIT $1)
                UNION ALL
                (SELECT 'NBA' as league, game_id, home_team, away_team, game_time,
                        moneyline_home, moneyline_away, spread_home, over_under
                 FROM nba_games
                 WHERE game_status = 'scheduled' AND game_time > NOW()
                 ORDER BY game_time LIMIT $2)
                UNION ALL
                (SELECT 'NCAAF' as league, game_id, home_team, away_team, game_time,
                        NULL::int, NULL::int, spread_home, over_under
                 FROM ncaa_football_games
                 WHERE game_status = 'scheduled' AND game_time > NOW()
                 ORDER BY game_time LIMIT $3)
                UNION ALL
                (SELECT 'NCAAB' as league, game_id, home_team, away_team, game_time,
                        NULL::int, NULL::int, spread_home, over_under
                 FROM ncaa_basketball_games
                 WHERE game_status = 'scheduled' AND game_time > NOW()
                 ORDER BY game_time LIMIT $4)
                ORDER BY game_time
            """
            return await self.db.fetch(query, per_sport, per_sport, per_sport, per_sport)

        return await self.cache.get_or_fetch(
            cache_key,
            fetch_from_db,
            ttl=self.TTL_UPCOMING
        )

    async def fetch_game_by_id(self, game_id: str, sport: str) -> Optional[Dict[str, Any]]:
        """Fetch a single game by ID with caching."""
        cache_key = f"sports:game:{sport}:{game_id}"

        table_map = {
            'NFL': 'nfl_games',
            'NBA': 'nba_games',
            'NCAAF': 'ncaa_football_games',
            'NCAAB': 'ncaa_basketball_games'
        }
        table = table_map.get(sport.upper())
        if not table:
            return None

        async def fetch_from_db():
            query = f"SELECT * FROM {table} WHERE game_id = $1"
            return await self.db.fetchrow(query, game_id)

        return await self.cache.get_or_fetch(
            cache_key,
            fetch_from_db,
            ttl=self.TTL_LIVE_GAMES
        )

    # =========================================================================
    # Circuit Breaker Protected API Calls
    # =========================================================================

    async def fetch_espn_scores(self, sport: str = "NFL") -> List[Dict[str, Any]]:
        """Fetch ESPN scores with circuit breaker protection."""
        try:
            return await espn_breaker.call(
                self._fetch_espn_scores_internal, sport
            )
        except CircuitBreakerError:
            logger.warning(f"ESPN circuit breaker open, returning cached data")
            cached = await self.cache.get(f"espn:scores:{sport}")
            return cached or []

    async def _fetch_espn_scores_internal(self, sport: str) -> List[Dict[str, Any]]:
        """Internal ESPN fetch - called through circuit breaker."""
        from src.espn_live_data import ESPNLiveData
        from src.espn_nba_live_data import ESPNNBALiveData
        from src.espn_ncaa_live_data import ESPNNCAALiveData

        sport = sport.upper()

        if sport == "NFL":
            espn = ESPNLiveData()
            games = await asyncio.to_thread(espn.get_scoreboard)
        elif sport == "NBA":
            espn = ESPNNBALiveData()
            games = await asyncio.to_thread(espn.get_scoreboard)
        elif sport in ["NCAAF", "NCAAB"]:
            espn = ESPNNCAALiveData()
            games = await asyncio.to_thread(espn.get_scoreboard)
        else:
            games = []

        # Cache the result
        await self.cache.set(f"espn:scores:{sport}", games, self.TTL_LIVE_GAMES)
        return games

    async def fetch_kalshi_markets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch Kalshi markets with circuit breaker protection."""
        try:
            return await kalshi_breaker.call(self._fetch_kalshi_internal)
        except CircuitBreakerError:
            logger.warning("Kalshi circuit breaker open, returning cached data")
            cached = await self.cache.get("kalshi:football_markets")
            return cached or {"nfl": [], "college": []}

    async def _fetch_kalshi_internal(self) -> Dict[str, List[Dict[str, Any]]]:
        """Internal Kalshi fetch - called through circuit breaker."""
        from src.kalshi_public_client import KalshiPublicClient

        client = KalshiPublicClient()
        markets = await asyncio.to_thread(client.get_football_markets)

        # Cache result
        await self.cache.set("kalshi:football_markets", markets, self.TTL_ODDS)
        return markets

    async def fetch_real_odds(self, sport: str) -> List[Dict[str, Any]]:
        """Fetch real odds from The Odds API with circuit breaker."""
        try:
            return await odds_api_breaker.call(self._fetch_odds_api_internal, sport)
        except CircuitBreakerError:
            logger.warning("Odds API circuit breaker open, returning cached odds")
            cached = await self.cache.get(f"odds_api:{sport}")
            return cached or []

    async def _fetch_odds_api_internal(self, sport: str) -> List[Dict[str, Any]]:
        """Internal Odds API fetch - called through circuit breaker."""
        from src.odds_api_client import TheOddsAPIClient

        client = TheOddsAPIClient()
        try:
            odds = await client.get_normalized_odds(sport)
            await self.cache.set(f"odds_api:{sport}", odds, self.TTL_ODDS)
            return odds
        finally:
            await client.close()

    # =========================================================================
    # WebSocket Broadcasting
    # =========================================================================

    async def broadcast_live_update(self, game_data: Dict[str, Any]):
        """Broadcast live game update to all WebSocket subscribers."""
        await self.ws.broadcast_to_room("live_games", {
            "type": "live_update",
            "data": game_data,
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_odds_update(self, sport: str, odds_data: List[Dict[str, Any]]):
        """Broadcast odds update to subscribers."""
        await self.ws.broadcast_to_room(f"odds_{sport.lower()}", {
            "type": "odds_update",
            "sport": sport,
            "data": odds_data,
            "timestamp": datetime.now().isoformat()
        })

    async def broadcast_best_bet_alert(self, bet_data: Dict[str, Any]):
        """Broadcast high-value betting opportunity alert."""
        await self.ws.broadcast_to_room("best_bets", {
            "type": "best_bet_alert",
            "data": bet_data,
            "timestamp": datetime.now().isoformat()
        })

    # =========================================================================
    # Server-Sent Events (SSE) Generator
    # =========================================================================

    async def sse_odds_stream(
        self,
        sports: List[str],
        interval: int = 30
    ) -> AsyncGenerator[str, None]:
        """
        Server-Sent Events generator for real-time odds streaming.

        Args:
            sports: List of sports to stream odds for
            interval: Update interval in seconds

        Yields:
            SSE-formatted data strings
        """
        while True:
            try:
                all_odds = {}
                for sport in sports:
                    cached_odds = await self.cache.get(f"odds_api:{sport}")
                    if cached_odds:
                        all_odds[sport] = cached_odds

                # Format as SSE
                data = json.dumps({
                    "type": "odds_update",
                    "odds": all_odds,
                    "timestamp": datetime.now().isoformat()
                })

                yield f"data: {data}\n\n"

            except Exception as e:
                logger.error(f"SSE odds stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            await asyncio.sleep(interval)

    async def sse_live_scores_stream(self, interval: int = 10) -> AsyncGenerator[str, None]:
        """
        SSE generator for live game scores.

        More frequent updates for live games (every 10 seconds).
        """
        while True:
            try:
                live_games = await self.fetch_live_games()

                data = json.dumps({
                    "type": "live_scores",
                    "games": live_games,
                    "count": len(live_games),
                    "timestamp": datetime.now().isoformat()
                })

                yield f"data: {data}\n\n"

            except Exception as e:
                logger.error(f"SSE live scores error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            await asyncio.sleep(interval)

    # =========================================================================
    # AI Probability Analysis
    # =========================================================================

    def blend_probabilities(
        self,
        model_prob: float,
        implied_prob: float,
        model_confidence: float = 0.7
    ) -> ProbabilityBlend:
        """
        Blend AI model probability with market-implied probability.

        Uses Bayesian Model Averaging for sophisticated combination.
        """
        return bayesian_probability_blend(
            model_prob=model_prob,
            implied_prob=implied_prob,
            model_confidence=model_confidence
        )

    def detect_betting_edge(
        self,
        model_prob: float,
        implied_prob: float
    ) -> Dict[str, Any]:
        """Detect profitable betting edges using ensemble analysis."""
        return ensemble_edge_detection(model_prob, implied_prob)

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Get health status of all infrastructure components."""
        db_health = await self.db.health_check() if self._db else {"healthy": False}
        cache_health = await self.cache.health_check() if self._cache else False
        ws_stats = self.ws.get_stats() if self._ws_manager else {}

        return {
            "database": db_health,
            "cache": {
                "healthy": cache_health,
                "stats": self.cache.get_stats() if self._cache else {}
            },
            "websocket": ws_stats,
            "circuit_breakers": {
                "espn": espn_breaker.get_stats(),
                "kalshi": kalshi_breaker.get_stats(),
                "odds_api": odds_api_breaker.get_stats()
            },
            "timestamp": datetime.now().isoformat()
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_sports_infra: Optional[SportsInfrastructure] = None


async def get_sports_infra() -> SportsInfrastructure:
    """Get the sports infrastructure singleton."""
    global _sports_infra
    if _sports_infra is None:
        _sports_infra = SportsInfrastructure()
        await _sports_infra.initialize()
    return _sports_infra


# Convenience function for sync contexts
def get_sports_infra_sync() -> SportsInfrastructure:
    """Get sports infrastructure (must be initialized)."""
    global _sports_infra
    if _sports_infra is None:
        _sports_infra = SportsInfrastructure()
    return _sports_infra
