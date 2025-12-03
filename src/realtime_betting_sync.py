"""
Real-time Betting Sync Service
Runs every 5 minutes to:
1. Fetch live game data from ESPN
2. Sync Kalshi market odds
3. Run AI analysis
4. Find betting opportunities
5. Send Telegram alerts for high-value bets
"""

import logging
import time
import os
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import asyncio

from src.espn_live_data import get_espn_client
from src.espn_ncaa_live_data import get_espn_ncaa_client
from src.kalshi_integration import KalshiIntegration
from src.kalshi_db_manager import KalshiDBManager
from src.kalshi_ai_evaluator import KalshiAIEvaluator
from src.live_betting_analyzer import LiveBettingAnalyzer
from src.price_action_monitor import PriceActionMonitor
from src.game_watchlist_manager import GameWatchlistManager
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent

# Telegram imports
try:
    import telegram
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("Telegram library not available. Install with: pip install python-telegram-bot")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class RealtimeBettingSync:
    """Real-time sync service for betting opportunities"""

    def __init__(self) -> None:
        self.espn_nfl = get_espn_client()
        self.espn_ncaa = get_espn_ncaa_client()
        self.kalshi = KalshiIntegration()
        self.db = KalshiDBManager()
        self.ai_evaluator = KalshiAIEvaluator()
        self.analyzer = LiveBettingAnalyzer()
        self.price_monitor = PriceActionMonitor(self.db)
        self.watchlist_manager = GameWatchlistManager(self.db)
        self.ai_agent = AdvancedBettingAIAgent()

        # Telegram setup
        self.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if self.telegram_enabled and TELEGRAM_AVAILABLE and self.telegram_token:
            self.telegram_bot = Bot(token=self.telegram_token)
            logger.info("Telegram bot initialized")
        else:
            self.telegram_bot = None
            if self.telegram_enabled:
                logger.warning("Telegram enabled but not configured properly")

        # Track sent alerts to avoid duplicates
        self.sent_alerts = set()

    def sync_all_data(self) -> Dict:
        """
        Sync all data sources and analyze opportunities

        Returns:
            Dict with sync results and opportunities found
        """

        logger.info("="*80)
        logger.info("REAL-TIME BETTING SYNC - %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("="*80)

        results = {
            'timestamp': datetime.now().isoformat(),
            'nfl_games': 0,
            'ncaa_games': 0,
            'markets_synced': 0,
            'predictions_generated': 0,
            'opportunities_found': 0,
            'alerts_sent': 0,
            'price_drops_detected': 0,
            'price_action_alerts': 0,
            'watchlist_updates': 0,
            'errors': []
        }

        try:
            # 1. Fetch live game data
            logger.info("[1/9] Fetching live game data...")
            nfl_games = self._fetch_nfl_games()
            ncaa_games = self._fetch_ncaa_games()
            results['nfl_games'] = len(nfl_games)
            results['ncaa_games'] = len(ncaa_games)
            logger.info(f"  Found {len(nfl_games)} NFL games, {len(ncaa_games)} NCAA games")

            # 2. Sync Kalshi markets
            logger.info("[2/9] Syncing Kalshi markets...")
            markets = self._sync_kalshi_markets()
            results['markets_synced'] = len(markets)
            logger.info(f"  Synced {len(markets)} active markets")

            # 3. Record price history for price action monitoring
            logger.info("[3/9] Recording current prices...")
            self.price_monitor.record_current_prices(markets)
            logger.info("  Price history updated")

            # 4. Run AI predictions
            logger.info("[4/9] Running AI predictions...")
            predictions = self._run_ai_predictions(markets)
            results['predictions_generated'] = len(predictions)
            logger.info(f"  Generated {len(predictions)} predictions")

            # 5. Analyze opportunities
            logger.info("[5/9] Analyzing betting opportunities...")
            all_games = nfl_games + ncaa_games
            opportunities = self.analyzer.analyze_all_opportunities(
                all_games,
                {m['ticker']: m for m in markets},
                {p['market_id']: p for p in predictions}
            )
            results['opportunities_found'] = len(opportunities)
            logger.info(f"  Found {len(opportunities)} total opportunities")

            # 6. Store results
            logger.info("[6/9] Storing analysis results...")
            self._store_opportunities(opportunities)

            # 7. Detect price drops (odds changes)
            logger.info("[7/9] Monitoring price action...")
            price_drops = self.price_monitor.detect_price_drops()
            results['price_drops_detected'] = len(price_drops)
            if price_drops:
                logger.info(f"  Detected {len(price_drops)} significant price drops")
                results['price_action_alerts'] = self._send_price_drop_alerts(price_drops)
            else:
                logger.info("  No significant price drops detected")

            # 8. Send alerts for high-value opportunities
            logger.info("[8/9] Checking for alert-worthy opportunities...")
            alerts = self.analyzer.get_alert_opportunities(opportunities, min_score=75)
            if alerts:
                logger.info(f"  Found {len(alerts)} alert-worthy opportunities")
                results['alerts_sent'] = self._send_alerts(alerts)
            else:
                logger.info("  No alert-worthy opportunities at this time")

            # 9. Send updates for watched games
            logger.info("[9/9] Checking watched games for updates...")
            watchlist_updates = self._send_watched_game_updates(all_games)
            results['watchlist_updates'] = watchlist_updates
            if watchlist_updates > 0:
                logger.info(f"  Sent {watchlist_updates} watchlist updates")
            else:
                logger.info("  No watched game updates at this time")

            # Summary
            logger.info("")
            logger.info("="*80)
            logger.info("SYNC COMPLETE")
            logger.info("="*80)
            logger.info(f"Games: {results['nfl_games']} NFL, {results['ncaa_games']} NCAA")
            logger.info(f"Markets: {results['markets_synced']} synced")
            logger.info(f"Predictions: {results['predictions_generated']} generated")
            logger.info(f"Opportunities: {results['opportunities_found']} found")
            logger.info(f"Price Drops: {results['price_drops_detected']} detected")
            logger.info(f"Price Action Alerts: {results['price_action_alerts']} sent")
            logger.info(f"Opportunity Alerts: {results['alerts_sent']} sent")
            logger.info(f"Watchlist Updates: {results['watchlist_updates']} sent")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Error during sync: {e}", exc_info=True)
            results['errors'].append(str(e))

        return results

    def _fetch_nfl_games(self) -> List[Dict]:
        """Fetch live NFL games from ESPN"""
        try:
            games = self.espn_nfl.get_scoreboard()
            return games
        except Exception as e:
            logger.error(f"Error fetching NFL games: {e}")
            return []

    def _fetch_ncaa_games(self) -> List[Dict]:
        """Fetch live NCAA games from ESPN"""
        try:
            games = self.espn_ncaa.get_scoreboard(group='80')  # FBS
            return games
        except Exception as e:
            logger.error(f"Error fetching NCAA games: {e}")
            return []

    def _sync_kalshi_markets(self) -> List[Dict]:
        """Sync active markets from Kalshi"""
        try:
            # Get active football markets
            markets = self.kalshi.get_markets(limit=1000, status='active')

            # Filter for football (NFL and NCAA)
            football_markets = []
            for market in markets:
                ticker = market.get('ticker', '').lower()
                title = market.get('title', '').lower()
                if 'nfl' in ticker or any(team in title for team in ['nfl', 'football']):
                    football_markets.append(market)

            # Store in database
            if football_markets:
                self.db.store_markets(football_markets, 'nfl')

            return football_markets
        except Exception as e:
            logger.error(f"Error syncing Kalshi markets: {e}")
            return []

    def _run_ai_predictions(self, markets: List[Dict]) -> List[Dict]:
        """Run AI predictions on markets"""
        try:
            if not markets:
                return []

            predictions = self.ai_evaluator.evaluate_markets(markets)

            # Store predictions
            if predictions:
                self.db.store_predictions(predictions)

            return predictions
        except Exception as e:
            logger.error(f"Error running AI predictions: {e}")
            return []

    def _store_opportunities(self, opportunities: List[Dict]):
        """Store opportunity analysis results"""
        try:
            # Store in database for historical tracking
            conn = self.db.get_connection()
            cur = conn.cursor()

            # Create opportunities table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS betting_opportunities (
                    id SERIAL PRIMARY KEY,
                    game_id TEXT,
                    away_team TEXT,
                    home_team TEXT,
                    score TEXT,
                    status TEXT,
                    is_live BOOLEAN,
                    opportunity_score NUMERIC,
                    expected_value NUMERIC,
                    recommendation TEXT,
                    reasoning TEXT,
                    ai_confidence NUMERIC,
                    ai_edge NUMERIC,
                    ai_prediction TEXT,
                    alert_worthy BOOLEAN,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Insert opportunities
            for opp in opportunities:
                cur.execute("""
                    INSERT INTO betting_opportunities (
                        game_id, away_team, home_team, score, status, is_live,
                        opportunity_score, expected_value, recommendation,
                        reasoning, ai_confidence, ai_edge, ai_prediction, alert_worthy
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    opp.get('game_id'),
                    opp.get('away_team'),
                    opp.get('home_team'),
                    opp.get('score'),
                    opp.get('status'),
                    opp.get('is_live'),
                    opp.get('opportunity_score'),
                    opp.get('expected_value'),
                    opp.get('recommendation'),
                    ', '.join(opp.get('reasoning', [])),
                    opp.get('ai_confidence'),
                    opp.get('ai_edge'),
                    opp.get('ai_prediction'),
                    opp.get('alert_worthy')
                ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing opportunities: {e}")

    def _send_alerts(self, alerts: List[Dict]) -> int:
        """Send Telegram alerts for high-value opportunities"""

        if not self.telegram_bot or not self.telegram_chat_id:
            logger.warning("Telegram not configured, skipping alerts")
            return 0

        sent_count = 0

        for alert in alerts:
            # Create unique key to avoid duplicate alerts
            alert_key = f"{alert['game_id']}_{alert['opportunity_score']:.0f}"

            # Skip if already sent
            if alert_key in self.sent_alerts:
                continue

            try:
                message = self.analyzer.generate_alert_message(alert)

                # Send to Telegram
                self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message,
                    parse_mode='Markdown'
                )

                self.sent_alerts.add(alert_key)
                sent_count += 1
                logger.info(f"  Sent alert: {alert['away_team']} @ {alert['home_team']}")

                # Rate limit to avoid flooding
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error sending alert: {e}")

        return sent_count

    def _send_price_drop_alerts(self, price_drops: List[Dict]) -> int:
        """Send Telegram alerts for price drops"""

        if not self.telegram_bot or not self.telegram_chat_id:
            logger.warning("Telegram not configured, skipping price drop alerts")
            return 0

        sent_count = 0

        for drop in price_drops:
            # Create unique key to avoid duplicate alerts
            alert_key = f"price_drop_{drop['market_ticker']}"

            # Skip if already sent
            if alert_key in self.sent_alerts:
                continue

            try:
                message = self.price_monitor.generate_price_drop_alert_message(drop)

                # Send to Telegram
                self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message,
                    parse_mode='Markdown'
                )

                self.sent_alerts.add(alert_key)
                sent_count += 1
                logger.info(f"  Sent price drop alert: {drop['title']}")

                # Rate limit to avoid flooding
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error sending price drop alert: {e}")

        return sent_count

    def _send_watched_game_updates(self, all_games: List[Dict]) -> int:
        """
        Send Telegram updates for user's watched games

        Args:
            all_games: All current games (NFL + NCAA)

        Returns:
            Number of updates sent
        """
        if not self.telegram_bot or not self.telegram_chat_id:
            return 0

        sent_count = 0

        try:
            # Get all watched games from database
            # For now, use the authorized user from environment
            user_id = os.getenv('TELEGRAM_AUTHORIZED_USERS', '').split(',')[0]
            if not user_id:
                return 0

            watchlist = self.watchlist_manager.get_user_watchlist(user_id)

            if not watchlist:
                return 0

            logger.info(f"Checking {len(watchlist)} watched games for updates...")

            for watched in watchlist:
                game_id = watched['game_id']
                selected_team = watched.get('selected_team')

                # Find matching game in current games
                matching_game = None
                for game in all_games:
                    if game.get('id') == game_id:
                        matching_game = game
                        break

                if not matching_game:
                    continue

                # Run AI prediction
                try:
                    market_data = matching_game.get('kalshi_odds', {})
                    ai_prediction = self.ai_agent.analyze_betting_opportunity(matching_game, market_data)
                except Exception as e:
                    logger.error(f"Error running AI prediction for watched game {game_id}: {e}")
                    continue

                # Detect changes
                changes = self.watchlist_manager.detect_changes(matching_game, ai_prediction)

                # Check if any significant changes occurred
                if any([
                    changes.get('score_changed'),
                    changes.get('period_changed'),
                    changes.get('ai_changed'),
                    changes.get('odds_changed')
                ]):
                    # Record new state
                    self.watchlist_manager.record_game_state(matching_game, ai_prediction)

                    # Generate and send update message
                    message = self.watchlist_manager.generate_game_update_message(
                        matching_game,
                        ai_prediction,
                        changes,
                        selected_team
                    )

                    # Send to Telegram
                    self.telegram_bot.send_message(
                        chat_id=self.telegram_chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )

                    sent_count += 1
                    logger.info(f"  Sent update for watched game: {matching_game.get('away_team')} @ {matching_game.get('home_team')}")

                    # Rate limit
                    time.sleep(1)
                else:
                    # No changes, but still record state for next check
                    self.watchlist_manager.record_game_state(matching_game, ai_prediction)

        except Exception as e:
            logger.error(f"Error sending watched game updates: {e}", exc_info=True)

        return sent_count

    def run_continuous(self, interval_minutes: int = 5):
        """
        Run sync continuously every N minutes

        Args:
            interval_minutes: Minutes between syncs (default 5)
        """

        logger.info(f"Starting continuous sync every {interval_minutes} minutes")
        logger.info("Press Ctrl+C to stop")

        while True:
            try:
                # Run sync
                results = self.sync_all_data()

                # Wait for next sync
                logger.info(f"Next sync in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Stopping sync service...")
                break
            except Exception as e:
                logger.error(f"Error in continuous sync: {e}", exc_info=True)
                logger.info("Retrying in 1 minute...")
                time.sleep(60)


def main():
    """Main entry point for standalone execution"""
    sync_service = RealtimeBettingSync()

    # Run once or continuously
    import sys
    if '--once' in sys.argv:
        sync_service.sync_all_data()
    else:
        sync_service.run_continuous(interval_minutes=5)


if __name__ == "__main__":
    main()
