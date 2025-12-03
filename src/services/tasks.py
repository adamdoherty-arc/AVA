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
import traceback
from datetime import datetime, timedelta
from functools import wraps
from celery import shared_task

logger = logging.getLogger(__name__)


# ============================================================================
# Controlled Task Decorator
# ============================================================================

def controlled_task(automation_name: str):
    """
    Decorator that adds enable/disable control to a Celery task.

    Features:
    - Checks enabled state before execution (skips if disabled)
    - Registers task start/complete for tracking
    - Captures execution metrics and errors

    Usage:
        @shared_task(name='src.services.tasks.sync_kalshi_markets', bind=True)
        @controlled_task('sync-kalshi-markets')
        def sync_kalshi_markets(self):
            ...

    Args:
        automation_name: The automation identifier matching the database name
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Import here to avoid circular imports
            from src.services.automation_control_service import (
                get_automation_control_service,
                ExecutionStatus
            )

            control = get_automation_control_service()
            task_id = self.request.id if hasattr(self, 'request') else 'unknown'

            # Check if task is enabled
            if not control.is_enabled(automation_name):
                logger.info(f"Task '{automation_name}' is disabled, skipping execution")

                # Record skipped execution
                control.register_task_start(automation_name, task_id, triggered_by='scheduler')
                control.register_task_complete(
                    automation_name,
                    task_id,
                    ExecutionStatus.SKIPPED
                )
                return {'status': 'skipped', 'reason': 'Task is disabled'}

            # Register task start
            control.register_task_start(automation_name, task_id, triggered_by='scheduler')

            try:
                # Execute the actual task
                result = func(self, *args, **kwargs)

                # Extract record count from result if available
                records = None
                if isinstance(result, dict):
                    records = (
                        result.get('markets_synced') or
                        result.get('tickers_updated') or
                        result.get('records_processed') or
                        result.get('messages_added') or
                        result.get('predictions_generated') or
                        result.get('alerts_sent') or
                        result.get('earnings_updated') or
                        result.get('caches_warmed')
                    )

                # Register success
                control.register_task_complete(
                    automation_name,
                    task_id,
                    ExecutionStatus.SUCCESS,
                    result=result,
                    records_processed=records
                )

                return result

            except Exception as e:
                # Register failure
                control.register_task_complete(
                    automation_name,
                    task_id,
                    ExecutionStatus.FAILED,
                    error_message=str(e),
                    error_traceback=traceback.format_exc()
                )
                raise

        return wrapper
    return decorator


# ============================================================================
# Market Data Tasks
# ============================================================================

@shared_task(name='src.services.tasks.sync_kalshi_markets', bind=True, max_retries=3)
@controlled_task('sync-kalshi-markets')
def sync_kalshi_markets(self):
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
@controlled_task('update-stock-prices')
def update_stock_prices(self):
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
@controlled_task('sync-discord-messages')
def sync_discord_messages(self):
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
@controlled_task('update-earnings-calendar')
def update_earnings_calendar(self):
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
@controlled_task('generate-predictions')
def generate_predictions(self):
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
@controlled_task('send-hourly-alerts')
def send_alerts(self):
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
@controlled_task('cleanup-old-data')
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
@controlled_task('warm-caches')
def warm_caches(self):
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
def optimize_database(self):
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
@controlled_task('sync-xtrades-to-rag')
def sync_xtrades_to_rag(self):
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
@controlled_task('sync-discord-to-rag')
def sync_discord_to_rag(self):
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
# Sports Betting Tasks
# ============================================================================

@shared_task(name='src.services.tasks.sync_live_sports_games', bind=True)
@controlled_task('sync-live-sports')
def sync_live_sports_games(self):
    """
    Sync live sports games from ESPN (runs every 30 seconds during game windows).

    This task:
    - Fetches live game data from ESPN for all sports
    - Updates database with current scores/status
    - Broadcasts updates via WebSocket
    - Adjusts AI predictions based on live game state
    """
    try:
        import requests
        from datetime import datetime

        # Import ESPN clients
        from src.espn_nfl_live_data import ESPNNFLLiveData
        from src.espn_nba_live_data import ESPNNBALiveData
        from src.espn_ncaa_live_data import ESPNNCAALiveData

        # Import WebSocket broadcaster
        from src.websocket.sports_broadcaster import push_live_games_sync

        # Import prediction adjuster
        from src.prediction_agents.live_adjuster import get_live_adjuster, GameState

        games_synced = 0
        live_games = []

        # Sync NFL
        try:
            nfl_client = ESPNNFLLiveData()
            nfl_games = nfl_client.get_live_games()
            for game in nfl_games:
                if game.get('is_live'):
                    live_games.append({**game, 'sport': 'NFL'})
                    games_synced += 1
            push_live_games_sync(nfl_games, 'NFL')
        except Exception as e:
            logger.warning(f"NFL sync error: {e}")

        # Sync NBA
        try:
            nba_client = ESPNNBALiveData()
            nba_games = nba_client.get_live_games()
            for game in nba_games:
                if game.get('is_live'):
                    live_games.append({**game, 'sport': 'NBA'})
                    games_synced += 1
            push_live_games_sync(nba_games, 'NBA')
        except Exception as e:
            logger.warning(f"NBA sync error: {e}")

        # Sync NCAA Football
        try:
            ncaa_client = ESPNNCAALiveData()
            ncaa_games = ncaa_client.get_live_games()
            for game in ncaa_games:
                if game.get('is_live'):
                    live_games.append({**game, 'sport': 'NCAAF'})
                    games_synced += 1
            push_live_games_sync(ncaa_games, 'NCAAF')
        except Exception as e:
            logger.warning(f"NCAAF sync error: {e}")

        # Adjust predictions for live games
        adjuster = get_live_adjuster()
        predictions_adjusted = 0

        for game in live_games:
            try:
                # Create GameState from live data
                state = GameState(
                    home_score=game.get('home_score', 0),
                    away_score=game.get('away_score', 0),
                    period=game.get('quarter', game.get('period', 1)),
                    time_remaining_seconds=_parse_clock(game.get('clock', '0:00')),
                    sport=game.get('sport', 'NFL'),
                    possession=game.get('possession'),
                    is_red_zone=game.get('is_red_zone', False)
                )

                # Get pregame probability (would come from database in production)
                pregame_prob = 0.5  # Default, should be fetched from predictions table

                # Adjust prediction
                adjusted = adjuster.adjust_prediction(pregame_prob, state)
                predictions_adjusted += 1

            except Exception as e:
                logger.warning(f"Prediction adjustment error: {e}")

        logger.info(f"✅ Synced {games_synced} live games, adjusted {predictions_adjusted} predictions")

        return {
            'status': 'success',
            'games_synced': games_synced,
            'predictions_adjusted': predictions_adjusted,
            'live_games': len(live_games),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Failed to sync live sports: {e}")
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.sync_sports_odds', bind=True)
@controlled_task('sync-sports-odds')
def sync_sports_odds(self):
    """
    Sync odds from Kalshi prediction markets (runs every 60 seconds during game windows).

    This task:
    - Fetches current odds from Kalshi
    - Matches odds to games in database
    - Records odds history for movement tracking
    - Broadcasts significant odds movements via WebSocket
    """
    try:
        from datetime import datetime
        from src.kalshi_public_client import KalshiPublicClient
        from src.espn_kalshi_matcher import ESPNKalshiMatcher
        from src.services.prediction_tracker import get_prediction_tracker
        from src.websocket.sports_broadcaster import push_odds_sync

        client = KalshiPublicClient()
        tracker = get_prediction_tracker()
        odds_updated = 0
        significant_moves = 0

        # Get football markets from Kalshi
        football_markets = client.get_football_markets()

        # Process NFL markets
        for market in football_markets.get('nfl', []):
            try:
                ticker = market.get('ticker', '')
                yes_price = market.get('yes_price', 50) / 100  # Convert cents to decimal

                # Record odds snapshot
                tracker.record_odds_snapshot(
                    game_id=ticker,
                    sport='NFL',
                    source='kalshi',
                    home_odds=_prob_to_american(yes_price),
                    away_odds=_prob_to_american(1 - yes_price)
                )
                odds_updated += 1

                # Check for significant movement (broadcast if > 5%)
                history = tracker.get_odds_movement(ticker, hours=1)
                if len(history) >= 2:
                    prev_prob = history[-2].get('home_implied_prob', 0.5)
                    curr_prob = yes_price
                    if abs(curr_prob - prev_prob) > 0.05:
                        significant_moves += 1
                        push_odds_sync(ticker, 'NFL', {
                            'home_prob': curr_prob,
                            'away_prob': 1 - curr_prob,
                            'movement': curr_prob - prev_prob
                        })

            except Exception as e:
                logger.warning(f"Error processing market {market.get('ticker')}: {e}")

        logger.info(f"✅ Synced {odds_updated} odds snapshots, {significant_moves} significant moves")

        return {
            'status': 'success',
            'odds_updated': odds_updated,
            'significant_moves': significant_moves,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Failed to sync sports odds: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='src.services.tasks.settle_completed_predictions', bind=True)
@controlled_task('settle-predictions')
def settle_completed_predictions(self):
    """
    Settle predictions for completed games (runs hourly).

    This task:
    - Checks for newly completed games
    - Settles predictions and calculates accuracy
    - Updates model performance metrics
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import os
        from datetime import datetime
        from src.services.prediction_tracker import get_prediction_tracker

        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/magnus")
        tracker = get_prediction_tracker()
        predictions_settled = 0

        # Find completed games with unsettled predictions
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)

        try:
            with conn.cursor() as cur:
                # Get completed games from each sport table
                for table, sport in [
                    ('nfl_games', 'NFL'),
                    ('nba_games', 'NBA'),
                    ('ncaa_football_games', 'NCAAF'),
                    ('ncaa_basketball_games', 'NCAAB')
                ]:
                    try:
                        cur.execute(f"""
                            SELECT g.game_id, g.home_team, g.away_team,
                                   g.home_score, g.away_score, g.status
                            FROM {table} g
                            WHERE g.status IN ('STATUS_FINAL', 'Completed', 'Final')
                            AND g.game_id IN (
                                SELECT pr.game_id FROM prediction_results pr
                                WHERE pr.was_correct IS NULL
                                AND pr.sport = %s
                            )
                        """, (sport,))

                        completed_games = cur.fetchall()

                        for game in completed_games:
                            # Determine winner
                            if game['home_score'] > game['away_score']:
                                winner = game['home_team']
                            elif game['away_score'] > game['home_score']:
                                winner = game['away_team']
                            else:
                                winner = 'TIE'

                            # Settle the prediction
                            result = tracker.settle_prediction(
                                game_id=game['game_id'],
                                actual_winner=winner,
                                home_score=game['home_score'],
                                away_score=game['away_score']
                            )

                            if result.get('prediction_id'):
                                predictions_settled += 1
                                logger.info(
                                    f"Settled {result['prediction_id']}: "
                                    f"{'CORRECT' if result['was_correct'] else 'WRONG'}"
                                )

                    except Exception as e:
                        logger.warning(f"Error settling {sport} predictions: {e}")

        finally:
            conn.close()

        # Log performance metrics
        metrics = tracker.get_accuracy_metrics(days=30)
        logger.info(
            f"✅ Settled {predictions_settled} predictions. "
            f"30-day accuracy: {metrics.get('overall', {}).get('accuracy', 0):.1%}"
        )

        return {
            'status': 'success',
            'predictions_settled': predictions_settled,
            'accuracy_30d': metrics.get('overall', {}).get('accuracy', 0),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Failed to settle predictions: {e}")
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def _parse_clock(clock_str: str) -> int:
    """Parse game clock string to seconds remaining"""
    try:
        if ':' in clock_str:
            parts = clock_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
            return minutes * 60 + seconds
        return 0
    except:
        return 0


def _prob_to_american(prob: float) -> int:
    """Convert probability to American odds"""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)


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
