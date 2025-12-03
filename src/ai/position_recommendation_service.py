"""
Position Recommendation Service - Orchestration Layer

This service coordinates all components of the AI recommendation system:
- Fetches positions from Robinhood
- Enriches with market data
- Generates quantitative recommendations
- Generates LLM recommendations
- Aggregates and stores final recommendations

Author: Claude Code
Date: 2025-11-10
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import os
from dotenv import load_dotenv

# Import our AI components
from src.ai.position_data_aggregator import PositionDataAggregator, EnrichedPosition
from src.ai.position_quantitative_analyzer import PositionQuantitativeAnalyzer
from src.ai.position_llm_analyzer import PositionLLMAnalyzer
from src.ai.position_recommendation_aggregator import PositionRecommendationAggregator
from src.models.position_recommendation import PositionRecommendation, RecommendationAction

# Import database and cache
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import redis

load_dotenv()
logger = logging.getLogger(__name__)


class PositionRecommendationService:
    """
    Main orchestration service for position recommendations

    Usage:
        service = PositionRecommendationService()
        recommendations = await service.generate_all_recommendations()

        # Or get cached
        recommendations = await service.get_recommendations(use_cache=True)
    """

    def __init__(self) -> None:
        """Initialize the recommendation service"""
        # Initialize components
        self.data_aggregator = PositionDataAggregator()
        self.aggregator = PositionRecommendationAggregator()  # Creates its own analyzers internally

        # Database connection
        self.db_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'dbname': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!'),
            'port': os.getenv('DATABASE_PORT', '5432')
        }

        # Redis cache (if available)
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=False  # Store as bytes for pickle
            )
            self.redis_client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Caching disabled.")
            self.redis_client = None

        # Cache TTL
        self.cache_ttl_seconds = 1800  # 30 minutes

    def get_db_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    async def generate_all_recommendations(
        self,
        rh_session=None,
        force_refresh: bool = False
    ) -> List[PositionRecommendation]:
        """
        Generate recommendations for all positions

        Args:
            rh_session: Robinhood session (if None, will create new)
            force_refresh: Bypass cache and regenerate

        Returns:
            List of position recommendations
        """
        try:
            # Step 1: Fetch and enrich positions
            logger.info("Fetching positions from Robinhood...")
            enriched_positions = await self.data_aggregator.fetch_all_positions(rh_session)

            if not enriched_positions:
                logger.warning("No open positions found")
                return []

            logger.info(f"Found {len(enriched_positions)} positions")

            # Step 2: Generate recommendations for each position
            recommendations = []

            for position in enriched_positions:
                try:
                    # Get final recommendation (quant + LLM aggregated)
                    recommendation = await self.aggregator.get_recommendation(position)

                    # Store in database
                    await self._store_recommendation(recommendation)

                    recommendations.append(recommendation)

                    logger.info(
                        f"Generated recommendation for {position.symbol}: "
                        f"{recommendation.action.value} (confidence: {recommendation.confidence}%)"
                    )

                except Exception as e:
                    logger.error(f"Error generating recommendation for {position.symbol}: {e}")
                    continue

            # Step 3: Cache results
            if self.redis_client and not force_refresh:
                await self._cache_recommendations(recommendations)

            logger.info(f"Successfully generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise

    async def get_recommendations(
        self,
        use_cache: bool = True,
        max_age_minutes: int = 30
    ) -> List[PositionRecommendation]:
        """
        Get recommendations (from cache or generate new)

        Args:
            use_cache: Use cached recommendations if available
            max_age_minutes: Maximum age of cached data (minutes)

        Returns:
            List of recommendations
        """
        if use_cache and self.redis_client:
            # Try to get from cache
            cached = await self._get_cached_recommendations(max_age_minutes)
            if cached:
                logger.info(f"Returning {len(cached)} cached recommendations")
                return cached

        # Generate fresh recommendations
        logger.info("Generating fresh recommendations...")
        return await self.generate_all_recommendations()

    async def get_recommendation_by_symbol(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[PositionRecommendation]:
        """
        Get recommendation for a specific symbol

        Args:
            symbol: Stock symbol
            use_cache: Use cached data if available

        Returns:
            Recommendation or None if not found
        """
        recommendations = await self.get_recommendations(use_cache=use_cache)

        for rec in recommendations:
            if rec.position.symbol.upper() == symbol.upper():
                return rec

        return None

    async def _store_recommendation(self, recommendation: PositionRecommendation):
        """Store recommendation in database"""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()

            # Convert to JSON-serializable format
            rec_data = {
                'symbol': recommendation.position.symbol,
                'action': recommendation.action.value,
                'confidence': recommendation.confidence,
                'risk_level': recommendation.risk_level.value,
                'urgency': recommendation.urgency,
                'rationale': recommendation.rationale,
                'key_factors': recommendation.key_factors,
                'action_details': recommendation.action_details,
                'quant_signal': recommendation.quant_signal.value if recommendation.quant_signal else None,
                'llm_signal': recommendation.llm_signal.value if recommendation.llm_signal else None,
                'model_used': recommendation.model_used,
                'expected_outcome': recommendation.expected_outcome,
                'timestamp': datetime.now().isoformat()
            }

            # Insert or update recommendation
            cur.execute("""
                INSERT INTO position_recommendations (
                    symbol, recommendation, created_at, updated_at
                )
                VALUES (%s, %s, NOW(), NOW())
                ON CONFLICT (symbol)
                DO UPDATE SET
                    recommendation = EXCLUDED.recommendation,
                    updated_at = NOW()
            """, (recommendation.position.symbol, Json(rec_data)))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing recommendation: {e}")

    async def _cache_recommendations(self, recommendations: List[PositionRecommendation]):
        """Cache recommendations in Redis"""
        if not self.redis_client:
            return

        try:
            import pickle

            # Serialize recommendations
            cache_data = {
                'recommendations': [rec.to_dict() for rec in recommendations],
                'timestamp': datetime.now().isoformat()
            }

            cache_key = 'position_recommendations:all'
            self.redis_client.setex(
                cache_key,
                self.cache_ttl_seconds,
                pickle.dumps(cache_data)
            )

            logger.info(f"Cached {len(recommendations)} recommendations (TTL: {self.cache_ttl_seconds}s)")

        except Exception as e:
            logger.error(f"Error caching recommendations: {e}")

    async def _get_cached_recommendations(
        self,
        max_age_minutes: int = 30
    ) -> Optional[List[PositionRecommendation]]:
        """Get cached recommendations from Redis"""
        if not self.redis_client:
            return None

        try:
            import pickle

            cache_key = 'position_recommendations:all'
            cached_data = self.redis_client.get(cache_key)

            if not cached_data:
                return None

            data = pickle.loads(cached_data)
            timestamp = datetime.fromisoformat(data['timestamp'])
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60

            if age_minutes > max_age_minutes:
                logger.info(f"Cache expired (age: {age_minutes:.1f} minutes)")
                return None

            # Reconstruct recommendations from dicts
            recommendations = [
                PositionRecommendation.from_dict(rec_dict)
                for rec_dict in data['recommendations']
            ]

            logger.info(f"Cache hit: {len(recommendations)} recommendations (age: {age_minutes:.1f} minutes)")
            return recommendations

        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None

    async def generate_for_symbol(
        self,
        symbol: str,
        rh_session=None
    ) -> Optional[PositionRecommendation]:
        """
        Generate recommendation for a specific symbol

        Args:
            symbol: Stock symbol
            rh_session: Robinhood session

        Returns:
            Recommendation or None if position not found
        """
        try:
            # Fetch all positions and filter
            enriched_positions = await self.data_aggregator.fetch_all_positions(rh_session)

            for position in enriched_positions:
                if position.symbol.upper() == symbol.upper():
                    recommendation = await self.aggregator.get_recommendation(position)
                    await self._store_recommendation(recommendation)
                    return recommendation

            logger.warning(f"Position not found for symbol: {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all cached recommendations"""
        if self.redis_client:
            try:
                self.redis_client.delete('position_recommendations:all')
                logger.info("Cache cleared")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")


# Utility functions for quick access
async def get_all_recommendations(use_cache: bool = True) -> List[PositionRecommendation]:
    """Quick access function to get all recommendations"""
    service = PositionRecommendationService()
    return await service.get_recommendations(use_cache=use_cache)


async def get_recommendation_for(symbol: str, use_cache: bool = True) -> Optional[PositionRecommendation]:
    """Quick access function to get recommendation for a symbol"""
    service = PositionRecommendationService()
    return await service.get_recommendation_by_symbol(symbol, use_cache=use_cache)


async def refresh_all_recommendations() -> List[PositionRecommendation]:
    """Quick access function to force refresh all recommendations"""
    service = PositionRecommendationService()
    return await service.generate_all_recommendations(force_refresh=True)


# CLI test
if __name__ == "__main__":
    import robin_stocks.robinhood as rh

    async def test():
        """Test the recommendation service"""
        print("=== Position Recommendation Service Test ===\n")

        # Login to Robinhood
        username = os.getenv('ROBINHOOD_USERNAME')
        password = os.getenv('ROBINHOOD_PASSWORD')

        if not username or not password:
            print("❌ Robinhood credentials not found in .env")
            return

        print("Logging into Robinhood...")
        rh.login(username, password)
        print("✅ Logged in\n")

        # Initialize service
        service = PositionRecommendationService()

        # Generate recommendations
        print("Generating recommendations...")
        recommendations = await service.generate_all_recommendations(rh_session=rh)

        print(f"\n✅ Generated {len(recommendations)} recommendations\n")

        # Display results
        for rec in recommendations:
            print(f"Symbol: {rec.position.symbol}")
            print(f"Action: {rec.action.value}")
            print(f"Confidence: {rec.confidence}%")
            print(f"Risk: {rec.risk_level.value}")
            print(f"Urgency: {rec.urgency}")
            print(f"Rationale: {rec.rationale}")
            print(f"Key Factors:")
            for factor in rec.key_factors:
                print(f"  • {factor}")
            print(f"Model: {rec.model_used}")
            print("-" * 60 + "\n")

    # Run test
    asyncio.run(test())
