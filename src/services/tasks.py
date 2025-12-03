"""
Celery Background Tasks
=======================

Asynchronous background tasks for the Magnus Trading Platform.

Categories:
- Market Data: Sync markets, update prices, fetch options chains
- Predictions: Generate AI predictions, update models
- Notifications: Send alerts, Discord/Telegram messages
- Maintenance: Cleanup old data, warm caches, database optimization

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
from datetime import datetime, timedelta
from celery import shared_task

logger = logging.getLogger(__name__)


# ============================================================================
# Market Data Tasks
# ============================================================================

@shared_task(name='src.services.tasks.sync_kalshi_markets', bind=True, max_retries=3)
def sync_kalshi_markets(self) -> None:
    """
    Sync Kalshi prediction markets

    Runs: Every 5 minutes
    Queue: market_data
    """
    try:
        from src.kalshi_db_manager import KalshiDBManager

        db = KalshiDBManager()
        markets_synced = db.sync_active_markets()

        logger.info(f"✅ Synced {markets_synced} Kalshi markets")
        return {'status': 'success', 'markets_synced': markets_synced}

    except Exception as e:
        logger.error(f"❌ Failed to sync Kalshi markets: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@shared_task(name='src.services.tasks.update_stock_prices', bind=True)
def update_stock_prices(self) -> None:
    """
    Update stock prices for watchlist

    Runs: Every 1 minute (market hours only)
    Queue: market_data
    """
    try:
        from src.yfinance_wrapper import update_watchlist_prices

        tickers_updated = update_watchlist_prices()

        logger.info(f"✅ Updated prices for {tickers_updated} tickers")
        return {'status': 'success', 'tickers_updated': tickers_updated}

    except Exception as e:
        logger.error(f"❌ Failed to update stock prices: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.sync_discord_messages', bind=True)
def sync_discord_messages(self) -> None:
    """
    Sync Discord messages with premium alert prioritization

    Runs: Every 5 minutes
    Queue: market_data

    Features:
    - Prioritizes channel 990331623260180580 (premium alerts)
    - Sends Discord bot notifications for new premium alerts
    - Syncs all other channels
    """
    try:
        from src.discord_premium_alert_sync import sync_premium_alerts

        # Sync with 5-minute lookback window
        result = sync_premium_alerts(minutes_back=6)  # 6 min to ensure overlap

        logger.info(
            f"✅ Discord sync: {result['total_alerts_sent']} premium alerts sent, "
            f"{result['all_channels']['total_messages']} total messages"
        )

        return {
            'status': 'success',
            'premium_alerts_sent': result.get('total_alerts_sent', 0),
            'total_messages': result['all_channels']['total_messages'],
            'channels_synced': result['all_channels']['channels_synced']
        }

    except Exception as e:
        logger.error(f"❌ Failed to sync Discord messages: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.update_earnings_calendar', bind=True)
def update_earnings_calendar(self) -> None:
    """
    Update earnings calendar for next 30 days

    Runs: Daily at 6 AM
    Queue: market_data
    """
    try:
        from src.earnings_manager import EarningsManager

        em = EarningsManager()
        earnings_updated = em.update_calendar(days_ahead=30)

        logger.info(f"✅ Updated {earnings_updated} earnings events")
        return {'status': 'success', 'earnings_updated': earnings_updated}

    except Exception as e:
        logger.error(f"❌ Failed to update earnings calendar: {e}")
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# Prediction Tasks
# ============================================================================

@shared_task(name='src.services.tasks.generate_predictions', bind=True)
def generate_predictions(self) -> None:
    """
    Generate AI predictions for upcoming games

    Runs: Every 15 minutes
    Queue: predictions
    """
    try:
        from src.prediction_agents.nfl_predictor import NFLPredictor
        from src.nfl_db_manager import NFLDBManager

        predictor = NFLPredictor()
        nfl_db = NFLDBManager()

        # Get upcoming games
        upcoming_games = nfl_db.get_upcoming_games(hours_ahead=48)

        predictions_generated = 0
        for game in upcoming_games:
            try:
                prediction = predictor.predict_game(
                    home_team=game['home_team'],
                    away_team=game['away_team'],
                    game_id=game['id']
                )

                # Save prediction to database
                nfl_db.save_prediction(prediction)
                predictions_generated += 1

            except Exception as game_error:
                logger.warning(f"Failed to predict game {game['id']}: {game_error}")
                continue

        logger.info(f"✅ Generated {predictions_generated} predictions")
        return {'status': 'success', 'predictions_generated': predictions_generated}

    except Exception as e:
        logger.error(f"❌ Failed to generate predictions: {e}")
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# Notification Tasks
# ============================================================================

@shared_task(name='src.services.tasks.send_alerts', bind=True)
def send_alerts(self) -> None:
    """
    Send scheduled alerts (high-confidence predictions, opportunities)

    Runs: Every hour
    Queue: notifications
    """
    try:
        from src.alert_manager import AlertManager

        alert_mgr = AlertManager()
        alerts_sent = alert_mgr.process_pending_alerts()

        logger.info(f"✅ Sent {alerts_sent} alerts")
        return {'status': 'success', 'alerts_sent': alerts_sent}

    except Exception as e:
        logger.error(f"❌ Failed to send alerts: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.send_discord_alert')
def send_discord_alert(message: str, channel: str = 'general'):
    """
    Send alert to Discord channel

    Args:
        message: Alert message
        channel: Discord channel name

    Usage:
        send_discord_alert.delay("High confidence prediction: BUF +7.5")
    """
    try:
        import requests
        import os

        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            logger.warning("Discord webhook not configured")
            return {'status': 'skipped', 'reason': 'webhook not configured'}

        payload = {
            'content': message,
            'username': 'Magnus Bot'
        }

        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()

        logger.info(f"✅ Sent Discord alert to {channel}")
        return {'status': 'success', 'channel': channel}

    except Exception as e:
        logger.error(f"❌ Failed to send Discord alert: {e}")
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# Maintenance Tasks
# ============================================================================

@shared_task(name='src.services.tasks.cleanup_old_data', bind=True)
def cleanup_old_data(self, days_to_keep: int = 90):
    """
    Cleanup old data from database

    Runs: Daily at 2 AM
    Queue: maintenance

    Args:
        days_to_keep: Number of days to retain data (default: 90)
    """
    try:
        from src.database.connection_pool import get_connection

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with get_connection() as conn:
            cursor = conn.cursor()

            # Clean old Discord messages
            cursor.execute("""
                DELETE FROM discord_messages
                WHERE timestamp < %s
            """, (cutoff_date,))
            discord_deleted = cursor.rowcount

            # Clean old predictions (keep only settled ones)
            cursor.execute("""
                DELETE FROM prediction_performance
                WHERE created_at < %s
                AND settled_at IS NULL
            """, (cutoff_date,))
            predictions_deleted = cursor.rowcount

            # Clean old cache entries
            cursor.execute("""
                DELETE FROM cache_entries
                WHERE created_at < %s
            """, (cutoff_date,))
            cache_deleted = cursor.rowcount

            conn.commit()

        logger.info(f"✅ Cleaned up: {discord_deleted} messages, {predictions_deleted} predictions, {cache_deleted} cache entries")
        return {
            'status': 'success',
            'discord_deleted': discord_deleted,
            'predictions_deleted': predictions_deleted,
            'cache_deleted': cache_deleted
        }

    except Exception as e:
        logger.error(f"❌ Failed to cleanup old data: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.warm_caches', bind=True)
def warm_caches(self) -> None:
    """
    Warm frequently accessed caches

    Runs: Every 30 minutes
    Queue: maintenance
    """
    try:
        from src.cache.redis_cache_manager import get_cache_manager, CacheNamespaces
        from src.kalshi_db_manager import KalshiDBManager
        from src.nfl_db_manager import NFLDBManager

        cache = get_cache_manager()
        caches_warmed = 0

        # Warm Kalshi markets cache
        kalshi_db = KalshiDBManager()
        active_markets = kalshi_db.get_active_markets()
        cache.set(CacheNamespaces.KALSHI_MARKETS, 'active_markets', active_markets, ttl=300)
        caches_warmed += 1

        # Warm NFL games cache
        nfl_db = NFLDBManager()
        upcoming_games = nfl_db.get_upcoming_games(hours_ahead=72)
        cache.set(CacheNamespaces.GAME_DATA, 'upcoming_nfl_games', upcoming_games, ttl=300)
        caches_warmed += 1

        logger.info(f"✅ Warmed {caches_warmed} caches")
        return {'status': 'success', 'caches_warmed': caches_warmed}

    except Exception as e:
        logger.error(f"❌ Failed to warm caches: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.optimize_database', bind=True)
def optimize_database(self) -> None:
    """
    Run database optimization (VACUUM ANALYZE)

    Runs: Weekly on Sunday at 3 AM
    Queue: maintenance
    """
    try:
        from src.database.connection_pool import get_connection

        with get_connection() as conn:
            conn.set_isolation_level(0)  # Autocommit mode for VACUUM
            cursor = conn.cursor()

            # VACUUM ANALYZE all tables
            cursor.execute("VACUUM ANALYZE")

            logger.info("✅ Database optimization complete")
            return {'status': 'success'}

    except Exception as e:
        logger.error(f"❌ Failed to optimize database: {e}")
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# Custom Task Examples
# ============================================================================

@shared_task(name='src.services.tasks.scan_options_opportunities')
def scan_options_opportunities(strategy: str = 'csp', min_delta: float = -0.30, max_dte: int = 45):
    """
    Scan for options opportunities based on criteria

    Usage:
        scan_options_opportunities.delay(strategy='csp', min_delta=-0.30, max_dte=45)
    """
    try:
        from src.ai_options_agent.scanner import OptionsScanner

        scanner = OptionsScanner()
        opportunities = scanner.scan(
            strategy=strategy,
            min_delta=min_delta,
            max_dte=max_dte
        )

        logger.info(f"✅ Found {len(opportunities)} {strategy} opportunities")
        return {'status': 'success', 'opportunities_found': len(opportunities)}

    except Exception as e:
        logger.error(f"❌ Failed to scan options: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.generate_daily_report')
def generate_daily_report():
    """
    Generate daily performance report and email it

    Runs: Daily at 8 PM
    Queue: notifications
    """
    try:
        from src.reports.daily_report import DailyReportGenerator

        report_gen = DailyReportGenerator()
        report = report_gen.generate()

        # Send via email (if configured)
        # send_email(to='user@example.com', subject='Daily Report', body=report)

        logger.info("✅ Daily report generated")
        return {'status': 'success', 'report_generated': True}

    except Exception as e:
        logger.error(f"❌ Failed to generate daily report: {e}")
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# RAG Knowledge Base Tasks
# ============================================================================

@shared_task(name='src.services.tasks.sync_xtrades_to_rag', bind=True)
def sync_xtrades_to_rag(self) -> None:
    """
    Sync XTrades messages to RAG knowledge base

    Runs: Daily at 1 AM (after message sync at midnight)
    Queue: maintenance
    """
    try:
        from src.rag.document_ingestion_pipeline import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline()

        # Ingest last 24 hours of XTrades messages
        result = pipeline.ingest_xtrades_messages(days_back=1)

        logger.info(f"✅ XTrades RAG sync: {result['success']} messages added")
        return {
            'status': 'success',
            'messages_added': result.get('success', 0),
            'duplicates_skipped': result.get('skipped', 0),
            'stats': pipeline.get_stats()
        }

    except Exception as e:
        logger.error(f"❌ Failed to sync XTrades to RAG: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.sync_discord_to_rag', bind=True)
def sync_discord_to_rag(self) -> None:
    """
    Sync Discord messages to RAG knowledge base

    Runs: Daily at 2 AM
    Queue: maintenance
    """
    try:
        from src.rag.document_ingestion_pipeline import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline()

        # Ingest last 7 days of Discord messages (weekly rolling window)
        result = pipeline.ingest_discord_messages(days_back=7)

        logger.info(f"✅ Discord RAG sync: {result['success']} messages added")
        return {
            'status': 'success',
            'messages_added': result.get('success', 0),
            'stats': pipeline.get_stats()
        }

    except Exception as e:
        logger.error(f"❌ Failed to sync Discord to RAG: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.ingest_documents_batch')
def ingest_documents_batch(
    directory: str,
    category: str = "trading_strategies",
    file_extensions: list = None
):
    """
    Batch ingest documents from directory

    Usage:
        ingest_documents_batch.delay(
            directory="/data/trading_strategies",
            category="trading_strategies"
        )
    """
    try:
        from src.rag.document_ingestion_pipeline import (
            DocumentIngestionPipeline,
            DocumentCategory
        )

        pipeline = DocumentIngestionPipeline()

        # Convert category string to enum
        category_enum = DocumentCategory[category.upper()]

        # Default file extensions
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf', '.docx']

        result = pipeline.ingest_local_directory(
            directory=directory,
            category=category_enum,
            file_extensions=file_extensions,
            recursive=True
        )

        logger.info(f"✅ Batch ingestion: {result['success']} documents added")
        return {
            'status': 'success',
            'documents_added': result.get('success', 0),
            'stats': pipeline.get_stats()
        }

    except Exception as e:
        logger.error(f"❌ Batch ingestion failed: {e}")
        return {'status': 'error', 'error': str(e)}


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test tasks locally
    print("Testing Celery tasks...")

    # This would need Celery worker running
    # result = sync_kalshi_markets.delay()
    # print(f"Task ID: {result.id}")
    # print(f"Status: {result.status}")
    # print(f"Result: {result.get(timeout=60)}")
