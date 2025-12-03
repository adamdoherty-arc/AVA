"""
Technical Analysis Database Manager
====================================

Manages database caching for technical analysis calculations:
- Fibonacci levels and confluence zones
- Volume Profile data
- Order Flow (CVD) analysis

Implements intelligent caching with TTL to avoid repeated calculations.
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class TechnicalAnalysisDBManager:
    """
    Database manager for caching technical analysis results

    Features:
    - Fibonacci level caching with TTL
    - Volume Profile caching
    - Order Flow (CVD) caching
    - Confluence zone tracking
    - Automatic cache invalidation
    - Cache hit/miss metrics
    """

    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize database manager

        Args:
            db_config: Database configuration dict
                      If None, uses environment variables
        """
        if db_config is None:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'trading'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }
        else:
            self.db_config = db_config

        # Default TTL settings (in seconds)
        self.default_ttl = {
            'fibonacci': 3600,      # 1 hour
            'volume_profile': 3600, # 1 hour
            'order_flow': 1800,     # 30 minutes (more volatile)
            'confluence': 3600      # 1 hour
        }

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    # =============================================================================
    # FIBONACCI LEVELS
    # =============================================================================

    def cache_fibonacci_levels(
        self,
        ticker: str,
        timeframe: str,
        period: str,
        swings: List[Dict],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Cache Fibonacci levels for a stock

        Args:
            ticker: Stock symbol
            timeframe: e.g., '1d', '1wk', '1h'
            period: e.g., '1mo', '3mo', '6mo', '1y'
            swings: List of swing dictionaries from FibonacciCalculator
            ttl_seconds: Time-to-live in seconds (None = use default)

        Returns:
            True if successful
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            ttl = ttl_seconds or self.default_ttl['fibonacci']
            expires_at = datetime.now() + timedelta(seconds=ttl)

            for swing in swings:
                # Insert or update Fibonacci levels
                cur.execute("""
                    INSERT INTO fibonacci_levels (
                        ticker, timeframe, period, swing_type,
                        swing_high, swing_low, high_date, low_date,
                        retracement_levels, extension_levels,
                        golden_zone_top, golden_zone_bottom,
                        price_range, range_pct,
                        calculated_at, expires_at, is_active
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (ticker, timeframe, period, swing_high, swing_low, high_date)
                    DO UPDATE SET
                        retracement_levels = EXCLUDED.retracement_levels,
                        extension_levels = EXCLUDED.extension_levels,
                        calculated_at = EXCLUDED.calculated_at,
                        expires_at = EXCLUDED.expires_at,
                        is_active = TRUE
                """, (
                    ticker, timeframe, period, swing['type'],
                    swing['swing_high'], swing['swing_low'],
                    swing['high_date'], swing['low_date'],
                    Json(swing['retracement_levels']),
                    Json(swing.get('extension_levels', {})),
                    swing['golden_zone']['top'],
                    swing['golden_zone']['bottom'],
                    swing['price_range'], swing['range_pct'],
                    datetime.now(), expires_at, True
                ))

            conn.commit()
            cur.close()
            conn.close()

            # Update cache metadata
            self._update_cache_metadata(
                ticker, 'fibonacci', timeframe, period, ttl
            )

            logger.info(f"Cached {len(swings)} Fibonacci swings for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Error caching Fibonacci levels: {e}")
            return False

    def get_cached_fibonacci_levels(
        self,
        ticker: str,
        timeframe: str,
        period: str
    ) -> Optional[List[Dict]]:
        """
        Get cached Fibonacci levels

        Returns:
            List of swing dictionaries or None if cache miss
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT *
                FROM fibonacci_levels
                WHERE ticker = %s
                  AND timeframe = %s
                  AND period = %s
                  AND is_active = TRUE
                  AND expires_at > CURRENT_TIMESTAMP
                ORDER BY high_date DESC
            """, (ticker, timeframe, period))

            results = cur.fetchall()
            cur.close()
            conn.close()

            if results:
                # Cache HIT
                self._record_cache_hit(ticker, 'fibonacci', timeframe, period)

                # Convert to swing format
                swings = []
                for row in results:
                    swing = {
                        'type': row['swing_type'],
                        'swing_high': float(row['swing_high']),
                        'swing_low': float(row['swing_low']),
                        'high_date': row['high_date'],
                        'low_date': row['low_date'],
                        'retracement_levels': row['retracement_levels'],
                        'golden_zone': {
                            'top': float(row['golden_zone_top']),
                            'bottom': float(row['golden_zone_bottom'])
                        },
                        'price_range': float(row['price_range']),
                        'range_pct': float(row['range_pct'])
                    }
                    swings.append(swing)

                return swings
            else:
                # Cache MISS
                self._record_cache_miss(ticker, 'fibonacci', timeframe, period)
                return None

        except Exception as e:
            logger.error(f"Error fetching cached Fibonacci levels: {e}")
            return None

    # =============================================================================
    # FIBONACCI CONFLUENCE ZONES
    # =============================================================================

    def cache_fibonacci_confluence(
        self,
        ticker: str,
        timeframe: str,
        period: str,
        confluences: List[Dict],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache Fibonacci confluence zones"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            ttl = ttl_seconds or self.default_ttl['confluence']
            expires_at = datetime.now() + timedelta(seconds=ttl)

            for conf in confluences:
                cur.execute("""
                    INSERT INTO fibonacci_confluence_zones (
                        ticker, timeframe, period,
                        price, price_min, price_max,
                        level_count, strength, zone_width_pct,
                        levels, calculated_at, expires_at, is_active
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (ticker, timeframe, period, price)
                    DO UPDATE SET
                        level_count = EXCLUDED.level_count,
                        strength = EXCLUDED.strength,
                        levels = EXCLUDED.levels,
                        calculated_at = EXCLUDED.calculated_at,
                        expires_at = EXCLUDED.expires_at,
                        is_active = TRUE
                """, (
                    ticker, timeframe, period,
                    conf['price'], conf['price_min'], conf['price_max'],
                    conf['level_count'], conf['strength'], conf['zone_width_pct'],
                    Json(conf['levels']), datetime.now(), expires_at, True
                ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Cached {len(confluences)} confluence zones for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Error caching confluence zones: {e}")
            return False

    def get_cached_fibonacci_confluence(
        self,
        ticker: str,
        timeframe: str,
        period: str,
        min_strength: int = 2
    ) -> Optional[List[Dict]]:
        """Get cached Fibonacci confluence zones"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT *
                FROM fibonacci_confluence_zones
                WHERE ticker = %s
                  AND timeframe = %s
                  AND period = %s
                  AND strength >= %s
                  AND is_active = TRUE
                  AND expires_at > CURRENT_TIMESTAMP
                ORDER BY strength DESC
            """, (ticker, timeframe, period, min_strength))

            results = cur.fetchall()
            cur.close()
            conn.close()

            if results:
                return [dict(row) for row in results]
            return None

        except Exception as e:
            logger.error(f"Error fetching cached confluence zones: {e}")
            return None

    # =============================================================================
    # VOLUME PROFILE
    # =============================================================================

    def cache_volume_profile(
        self,
        ticker: str,
        timeframe: str,
        period: str,
        volume_profile: Dict,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache Volume Profile data"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            ttl = ttl_seconds or self.default_ttl['volume_profile']
            expires_at = datetime.now() + timedelta(seconds=ttl)

            cur.execute("""
                INSERT INTO volume_profile_data (
                    ticker, timeframe, period,
                    poc_price, poc_volume, poc_pct_of_total,
                    vah_price, val_price,
                    value_area_volume, value_area_pct,
                    total_volume,
                    price_levels, volume_at_price, volume_pct_at_price,
                    high_volume_nodes, low_volume_nodes,
                    calculated_at, expires_at, is_active
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                ticker, timeframe, period,
                volume_profile['poc']['price'],
                volume_profile['poc']['volume'],
                volume_profile['poc']['pct_of_total'],
                volume_profile['vah'],
                volume_profile['val'],
                volume_profile.get('value_area_volume'),
                volume_profile.get('value_area_pct'),
                volume_profile['total_volume'],
                Json(volume_profile['price_levels'].tolist() if hasattr(volume_profile['price_levels'], 'tolist') else volume_profile['price_levels']),
                Json(volume_profile['volume_at_price'].tolist() if hasattr(volume_profile['volume_at_price'], 'tolist') else volume_profile['volume_at_price']),
                Json(volume_profile['volume_pct_at_price'].tolist() if hasattr(volume_profile['volume_pct_at_price'], 'tolist') else volume_profile['volume_pct_at_price']),
                Json(volume_profile.get('high_volume_nodes', [])),
                Json(volume_profile.get('low_volume_nodes', [])),
                datetime.now(), expires_at, True
            ))

            conn.commit()
            cur.close()
            conn.close()

            self._update_cache_metadata(ticker, 'volume_profile', timeframe, period, ttl)

            logger.info(f"Cached Volume Profile for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Error caching Volume Profile: {e}")
            return False

    def get_cached_volume_profile(
        self,
        ticker: str,
        timeframe: str,
        period: str
    ) -> Optional[Dict]:
        """Get cached Volume Profile data"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT *
                FROM volume_profile_data
                WHERE ticker = %s
                  AND timeframe = %s
                  AND period = %s
                  AND is_active = TRUE
                  AND expires_at > CURRENT_TIMESTAMP
                ORDER BY calculated_at DESC
                LIMIT 1
            """, (ticker, timeframe, period))

            result = cur.fetchone()
            cur.close()
            conn.close()

            if result:
                self._record_cache_hit(ticker, 'volume_profile', timeframe, period)
                return dict(result)
            else:
                self._record_cache_miss(ticker, 'volume_profile', timeframe, period)
                return None

        except Exception as e:
            logger.error(f"Error fetching cached Volume Profile: {e}")
            return None

    # =============================================================================
    # ORDER FLOW (CVD)
    # =============================================================================

    def cache_order_flow(
        self,
        ticker: str,
        timeframe: str,
        period: str,
        order_flow_data: Dict,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache Order Flow (CVD) data"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            ttl = ttl_seconds or self.default_ttl['order_flow']
            expires_at = datetime.now() + timedelta(seconds=ttl)

            cur.execute("""
                INSERT INTO order_flow_data (
                    ticker, timeframe, period,
                    current_cvd, cvd_change_1d, cvd_change_5d, cvd_trend,
                    cvd_series, divergence_count, divergences,
                    buy_pressure_pct, sell_pressure_pct, net_pressure,
                    calculated_at, expires_at, is_active
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                ticker, timeframe, period,
                order_flow_data['current_cvd'],
                order_flow_data.get('cvd_change_1d'),
                order_flow_data.get('cvd_change_5d'),
                order_flow_data.get('cvd_trend'),
                Json(order_flow_data.get('cvd_series', [])),
                order_flow_data.get('divergence_count', 0),
                Json(order_flow_data.get('divergences', [])),
                order_flow_data.get('buy_pressure_pct'),
                order_flow_data.get('sell_pressure_pct'),
                order_flow_data.get('net_pressure'),
                datetime.now(), expires_at, True
            ))

            conn.commit()
            cur.close()
            conn.close()

            self._update_cache_metadata(ticker, 'order_flow', timeframe, period, ttl)

            logger.info(f"Cached Order Flow data for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Error caching Order Flow: {e}")
            return False

    def get_cached_order_flow(
        self,
        ticker: str,
        timeframe: str,
        period: str
    ) -> Optional[Dict]:
        """Get cached Order Flow (CVD) data"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT *
                FROM order_flow_data
                WHERE ticker = %s
                  AND timeframe = %s
                  AND period = %s
                  AND is_active = TRUE
                  AND expires_at > CURRENT_TIMESTAMP
                ORDER BY calculated_at DESC
                LIMIT 1
            """, (ticker, timeframe, period))

            result = cur.fetchone()
            cur.close()
            conn.close()

            if result:
                self._record_cache_hit(ticker, 'order_flow', timeframe, period)
                return dict(result)
            else:
                self._record_cache_miss(ticker, 'order_flow', timeframe, period)
                return None

        except Exception as e:
            logger.error(f"Error fetching cached Order Flow: {e}")
            return None

    # =============================================================================
    # CACHE METADATA & UTILITIES
    # =============================================================================

    def _update_cache_metadata(
        self,
        ticker: str,
        analysis_type: str,
        timeframe: str,
        period: str,
        ttl_seconds: int
    ):
        """Update cache metadata for tracking"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            next_refresh = datetime.now() + timedelta(seconds=ttl_seconds)

            cur.execute("""
                INSERT INTO technical_analysis_cache_metadata (
                    ticker, analysis_type, timeframe, period,
                    last_calculated, ttl_seconds, next_refresh, is_stale
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (ticker, analysis_type, timeframe, period)
                DO UPDATE SET
                    last_calculated = EXCLUDED.last_calculated,
                    ttl_seconds = EXCLUDED.ttl_seconds,
                    next_refresh = EXCLUDED.next_refresh,
                    is_stale = FALSE
            """, (
                ticker, analysis_type, timeframe, period,
                datetime.now(), ttl_seconds, next_refresh, False
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error updating cache metadata: {e}")

    def _record_cache_hit(
        self,
        ticker: str,
        analysis_type: str,
        timeframe: str,
        period: str
    ):
        """Record cache hit for metrics"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE technical_analysis_cache_metadata
                SET cache_hits = cache_hits + 1
                WHERE ticker = %s
                  AND analysis_type = %s
                  AND timeframe = %s
                  AND period = %s
            """, (ticker, analysis_type, timeframe, period))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error recording cache hit: {e}")

    def _record_cache_miss(
        self,
        ticker: str,
        analysis_type: str,
        timeframe: str,
        period: str
    ):
        """Record cache miss for metrics"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE technical_analysis_cache_metadata
                SET cache_misses = cache_misses + 1
                WHERE ticker = %s
                  AND analysis_type = %s
                  AND timeframe = %s
                  AND period = %s
            """, (ticker, analysis_type, timeframe, period))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error recording cache miss: {e}")

    def cleanup_expired_cache(self) -> None:
        """Run cleanup function to mark expired entries as inactive"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("SELECT cleanup_technical_analysis_cache()")

            conn.commit()
            cur.close()
            conn.close()

            logger.info("Cache cleanup completed")
            return True

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return False

    def get_cache_stats(self, ticker: Optional[str] = None) -> List[Dict]:
        """
        Get cache statistics

        Args:
            ticker: Specific ticker or None for all

        Returns:
            List of cache statistics
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            if ticker:
                cur.execute("""
                    SELECT *
                    FROM technical_analysis_cache_metadata
                    WHERE ticker = %s
                    ORDER BY last_calculated DESC
                """, (ticker,))
            else:
                cur.execute("""
                    SELECT *
                    FROM technical_analysis_cache_metadata
                    ORDER BY last_calculated DESC
                    LIMIT 100
                """)

            results = cur.fetchall()
            cur.close()
            conn.close()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error fetching cache stats: {e}")
            return []


if __name__ == "__main__":
    # Test database manager
    print("=" * 80)
    print("TECHNICAL ANALYSIS DATABASE MANAGER TEST")
    print("=" * 80)

    manager = TechnicalAnalysisDBManager()

    print("\n1. Test Cache Stats:")
    print("-" * 80)
    stats = manager.get_cache_stats()
    print(f"Found {len(stats)} cache entries")

    print("\n2. Test Cleanup:")
    print("-" * 80)
    success = manager.cleanup_expired_cache()
    print(f"Cleanup: {'SUCCESS' if success else 'FAILED'}")

    print("\n" + "=" * 80)
    print("✅ Database Manager Test Complete")
    print("=" * 80)
