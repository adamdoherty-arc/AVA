"""
Modern Sync Service for Xtrades
================================

Orchestrates the complete sync workflow:
1. Fetch alerts from profiles
2. AI-analyze trades
3. Store in database
4. Send notifications

Features:
- Async-first design
- Connection pooling with asyncpg
- Concurrent processing
- Structured logging
- Comprehensive error handling
- Health monitoring
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from decimal import Decimal
import structlog

import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import (
    XtradeProfile, XtradeAlert, AIAnalysis,
    SyncResult, SyncBatchResult
)
from .scraper import ModernXtradesScraper
from .analyzer import AITradeAnalyzer

logger = structlog.get_logger(__name__)


class ModernSyncService:
    """
    Modern async sync service for Xtrades alerts.

    Manages the complete data pipeline from scraping to storage.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        enable_ai: bool = True,
        ai_provider: str = "auto",
        max_concurrent_profiles: int = 3,
        max_concurrent_ai: int = 5,
        headless: bool = True,
        notification_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the sync service.

        Args:
            database_url: PostgreSQL connection URL
            enable_ai: Enable AI analysis of alerts
            ai_provider: AI provider (openai, anthropic, ollama, auto)
            max_concurrent_profiles: Max concurrent profile scrapes
            max_concurrent_ai: Max concurrent AI analyses
            headless: Run browser in headless mode
            notification_callback: Callback for notifications
        """
        self.database_url = database_url or self._get_database_url()
        self.enable_ai = enable_ai
        self.ai_provider = ai_provider
        self.max_concurrent_profiles = max_concurrent_profiles
        self.max_concurrent_ai = max_concurrent_ai
        self.headless = headless
        self.notification_callback = notification_callback

        self._pool: Optional[asyncpg.Pool] = None
        self._scraper: Optional[ModernXtradesScraper] = None
        self._analyzer: Optional[AITradeAnalyzer] = None

        self.logger = logger.bind(component="ModernSyncService")

    def _get_database_url(self) -> str:
        """Build database URL from environment variables."""
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        database = os.getenv('POSTGRES_DB', 'trading_dashboard')
        user = os.getenv('POSTGRES_USER', 'postgres')
        password = os.getenv('POSTGRES_PASSWORD', '')

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    async def __aenter__(self) -> 'ModernSyncService':
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Start the sync service."""
        self.logger.info("Starting sync service")

        # Initialize database pool
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )

        # Initialize scraper
        self._scraper = ModernXtradesScraper(headless=self.headless)
        await self._scraper.start()

        # Initialize AI analyzer
        if self.enable_ai:
            self._analyzer = AITradeAnalyzer(provider=self.ai_provider)

        self.logger.info("Sync service started")

    async def stop(self) -> None:
        """Stop the sync service."""
        self.logger.info("Stopping sync service")

        if self._scraper:
            await self._scraper.stop()
            self._scraper = None

        if self._pool:
            await self._pool.close()
            self._pool = None

        self._analyzer = None

        self.logger.info("Sync service stopped")

    async def get_active_profiles(self) -> List[XtradeProfile]:
        """Get all active profiles from database."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, username, display_name, profile_url, avatar_url,
                       bio, is_active, last_sync, created_at, updated_at
                FROM xtrades_profiles
                WHERE is_active = true
                ORDER BY username
            """)

            return [
                XtradeProfile(
                    id=row['id'],
                    username=row['username'],
                    display_name=row['display_name'],
                    profile_url=row['profile_url'],
                    avatar_url=row['avatar_url'],
                    bio=row['bio'],
                    is_active=row['is_active'],
                    last_sync=row['last_sync'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]

    async def add_profile(self, username: str) -> XtradeProfile:
        """Add a new profile to track."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO xtrades_profiles (username, is_active, created_at)
                VALUES ($1, true, NOW())
                ON CONFLICT (username) DO UPDATE SET is_active = true
                RETURNING id, username, display_name, profile_url, avatar_url,
                         bio, is_active, last_sync, created_at, updated_at
            """, username.lower())

            self.logger.info("Added profile", username=username)

            return XtradeProfile(
                id=row['id'],
                username=row['username'],
                display_name=row['display_name'],
                profile_url=row['profile_url'],
                avatar_url=row['avatar_url'],
                bio=row['bio'],
                is_active=row['is_active'],
                last_sync=row['last_sync'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def save_alert(self, alert: XtradeAlert) -> int:
        """Save an alert to the database."""
        async with self._pool.acquire() as conn:
            # Check if alert exists
            existing = await conn.fetchval(
                "SELECT id FROM xtrades_alerts WHERE alert_id = $1",
                alert.alert_id
            )

            if existing:
                # Update existing
                await conn.execute("""
                    UPDATE xtrades_alerts SET
                        alert_text = $1,
                        ticker = $2,
                        strategy = $3,
                        action = $4,
                        strike_price = $5,
                        expiration_date = $6,
                        entry_price = $7,
                        target_price = $8,
                        stop_loss = $9,
                        quantity = $10,
                        sentiment = $11,
                        risk_level = $12,
                        ai_summary = $13,
                        ai_confidence = $14,
                        updated_at = NOW()
                    WHERE alert_id = $15
                """,
                    alert.alert_text,
                    alert.ticker,
                    alert.strategy,
                    alert.action,
                    float(alert.strike_price) if alert.strike_price else None,
                    alert.expiration_date,
                    float(alert.entry_price) if alert.entry_price else None,
                    float(alert.target_price) if alert.target_price else None,
                    float(alert.stop_loss) if alert.stop_loss else None,
                    alert.quantity,
                    alert.sentiment.value if alert.sentiment else None,
                    alert.risk_level.value if alert.risk_level else None,
                    alert.ai_summary,
                    alert.ai_confidence,
                    alert.alert_id
                )
                return existing
            else:
                # Insert new
                return await conn.fetchval("""
                    INSERT INTO xtrades_alerts (
                        profile_id, alert_id, alert_text, alert_type, posted_at,
                        ticker, strategy, action, strike_price, expiration_date,
                        entry_price, target_price, stop_loss, quantity,
                        sentiment, risk_level, ai_summary, ai_confidence,
                        created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, NOW()
                    )
                    RETURNING id
                """,
                    alert.profile_id,
                    alert.alert_id,
                    alert.alert_text,
                    alert.alert_type.value,
                    alert.posted_at,
                    alert.ticker,
                    alert.strategy,
                    alert.action,
                    float(alert.strike_price) if alert.strike_price else None,
                    alert.expiration_date,
                    float(alert.entry_price) if alert.entry_price else None,
                    float(alert.target_price) if alert.target_price else None,
                    float(alert.stop_loss) if alert.stop_loss else None,
                    alert.quantity,
                    alert.sentiment.value if alert.sentiment else None,
                    alert.risk_level.value if alert.risk_level else None,
                    alert.ai_summary,
                    alert.ai_confidence
                )

    async def update_profile_sync_time(self, profile_id: int) -> None:
        """Update the last sync time for a profile."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE xtrades_profiles SET last_sync = NOW() WHERE id = $1",
                profile_id
            )

    async def sync_profile(
        self,
        profile: XtradeProfile,
        since: Optional[datetime] = None
    ) -> SyncResult:
        """
        Sync alerts from a single profile.

        Args:
            profile: Profile to sync
            since: Only sync alerts after this time

        Returns:
            SyncResult with sync statistics
        """
        start_time = datetime.utcnow()
        result = SyncResult(
            success=True,
            profile_username=profile.username,
            start_time=start_time
        )

        self.logger.info("Starting profile sync", username=profile.username)

        try:
            # Fetch alerts
            alerts = await self._scraper.fetch_profile_alerts(profile, since=since)
            result.alerts_found = len(alerts)

            if not alerts:
                self.logger.info("No alerts found", username=profile.username)
                await self.update_profile_sync_time(profile.id)
                result.end_time = datetime.utcnow()
                return result

            # AI analysis if enabled
            if self.enable_ai and self._analyzer:
                trade_alerts = [a for a in alerts if a.has_trade_data()]
                if trade_alerts:
                    analyses = await self._analyzer.batch_analyze(
                        trade_alerts,
                        max_concurrent=self.max_concurrent_ai
                    )
                    result.trades_with_ai = len(analyses)

            # Save alerts to database
            for alert in alerts:
                try:
                    alert_id = await self.save_alert(alert)
                    if alert_id:
                        result.alerts_new += 1
                except Exception as e:
                    self.logger.error(
                        "Failed to save alert",
                        alert_id=alert.alert_id,
                        error=str(e)
                    )
                    result.alerts_failed += 1
                    result.add_error(f"Failed to save alert {alert.alert_id}: {e}")

            # Count trades
            result.trades_extracted = sum(1 for a in alerts if a.has_trade_data())

            # Update sync time
            await self.update_profile_sync_time(profile.id)

            result.end_time = datetime.utcnow()

            self.logger.info(
                "Profile sync complete",
                username=profile.username,
                alerts_found=result.alerts_found,
                alerts_new=result.alerts_new,
                trades_extracted=result.trades_extracted,
                duration=result.duration_seconds
            )

            # Send notification if callback provided
            if self.notification_callback and result.alerts_new > 0:
                self.notification_callback(result.to_summary())

            return result

        except Exception as e:
            self.logger.error(
                "Profile sync failed",
                username=profile.username,
                error=str(e)
            )
            result.add_error(str(e))
            result.end_time = datetime.utcnow()
            return result

    async def sync_all_profiles(
        self,
        since: Optional[datetime] = None
    ) -> SyncBatchResult:
        """
        Sync all active profiles.

        Args:
            since: Only sync alerts after this time

        Returns:
            SyncBatchResult with all results
        """
        start_time = datetime.utcnow()
        batch_result = SyncBatchResult(start_time=start_time)

        profiles = await self.get_active_profiles()
        self.logger.info("Starting batch sync", profile_count=len(profiles))

        if not profiles:
            self.logger.warning("No active profiles to sync")
            batch_result.end_time = datetime.utcnow()
            return batch_result

        # Sync profiles with limited concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_profiles)

        async def sync_with_limit(profile: XtradeProfile) -> SyncResult:
            async with semaphore:
                return await self.sync_profile(profile, since=since)

        tasks = [sync_with_limit(p) for p in profiles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "Profile sync exception",
                    username=profiles[i].username,
                    error=str(result)
                )
                batch_result.profile_results.append(SyncResult(
                    success=False,
                    profile_username=profiles[i].username,
                    start_time=start_time,
                    errors=[str(result)]
                ))
            else:
                batch_result.profile_results.append(result)

        batch_result.end_time = datetime.utcnow()

        self.logger.info(
            "Batch sync complete",
            total_profiles=batch_result.total_profiles,
            successful=batch_result.successful_profiles,
            total_alerts=batch_result.total_alerts,
            total_new=batch_result.total_new_alerts,
            duration=batch_result.duration_seconds
        )

        return batch_result

    async def run_continuous(
        self,
        interval_minutes: int = 5,
        stop_event: Optional[asyncio.Event] = None
    ) -> None:
        """
        Run continuous sync at specified interval.

        Args:
            interval_minutes: Minutes between syncs
            stop_event: Event to signal stop
        """
        self.logger.info(
            "Starting continuous sync",
            interval_minutes=interval_minutes
        )

        while True:
            if stop_event and stop_event.is_set():
                self.logger.info("Stop event received, exiting")
                break

            try:
                # Sync with "since" set to avoid re-processing old alerts
                since = datetime.utcnow() - timedelta(hours=24)
                await self.sync_all_profiles(since=since)

            except Exception as e:
                self.logger.error("Sync cycle failed", error=str(e))

            # Wait for next cycle
            self.logger.info(
                "Waiting for next sync cycle",
                wait_minutes=interval_minutes
            )
            await asyncio.sleep(interval_minutes * 60)

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }

        # Check database
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health['components']['database'] = {'status': 'up'}
        except Exception as e:
            health['status'] = 'degraded'
            health['components']['database'] = {
                'status': 'down',
                'error': str(e)
            }

        # Check scraper
        if self._scraper:
            try:
                logged_in = await self._scraper.check_login_status()
                health['components']['scraper'] = {
                    'status': 'up' if logged_in else 'degraded',
                    'logged_in': logged_in
                }
                if not logged_in:
                    health['status'] = 'degraded'
            except Exception as e:
                health['status'] = 'degraded'
                health['components']['scraper'] = {
                    'status': 'down',
                    'error': str(e)
                }

        # Check AI analyzer
        if self._analyzer:
            health['components']['ai_analyzer'] = {'status': 'up'}

        return health


async def main():
    """CLI entry point for testing."""
    import argparse

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    parser = argparse.ArgumentParser(description="Xtrades Modern Sync Service")
    parser.add_argument('--once', action='store_true', help="Run sync once and exit")
    parser.add_argument('--interval', type=int, default=5, help="Sync interval in minutes")
    parser.add_argument('--headless', action='store_true', help="Run browser headless")
    parser.add_argument('--no-ai', action='store_true', help="Disable AI analysis")
    parser.add_argument('--add-profile', type=str, help="Add a profile to track")
    args = parser.parse_args()

    async with ModernSyncService(
        enable_ai=not args.no_ai,
        headless=args.headless
    ) as service:

        # Add profile if specified
        if args.add_profile:
            profile = await service.add_profile(args.add_profile)
            print(f"Added profile: {profile.username} (ID: {profile.id})")
            return

        # Health check
        health = await service.health_check()
        print(f"Health: {health['status']}")
        for name, status in health['components'].items():
            print(f"  {name}: {status}")

        if args.once:
            # Single sync
            result = await service.sync_all_profiles()
            print(result.to_summary())
        else:
            # Continuous sync
            await service.run_continuous(interval_minutes=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
