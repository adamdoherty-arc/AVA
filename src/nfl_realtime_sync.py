"""
NFL Real-Time Sync Engine
Continuously monitors live games and updates database every 5 seconds
"""

import os
import time
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from nfl_db_manager import NFLDBManager
from nfl_data_fetcher import NFLDataFetcher
from kalshi_client import KalshiClient
from kalshi_db_manager import KalshiDBManager
from telegram_notifier import TelegramNotifier
from espn_rate_limiter import rate_limited

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLRealtimeSync:
    """
    Real-time sync engine for NFL games

    Features:
    - Polls live games every 5 seconds
    - Updates scores, plays, and stats
    - Monitors Kalshi market price movements
    - Sends Telegram alerts for significant events
    - Tracks injuries and weather updates
    """

    def __init__(
        self,
        update_interval_seconds: int = 5,
        enable_notifications: bool = True
    ):
        """
        Initialize the real-time sync engine

        Args:
            update_interval_seconds: How often to poll for updates (default: 5)
            enable_notifications: Enable Telegram notifications
        """
        self.update_interval = update_interval_seconds
        self.enable_notifications = enable_notifications

        # Initialize components
        self.nfl_db = NFLDBManager()
        self.fetcher = NFLDataFetcher()
        self.kalshi_client = KalshiClient()
        self.kalshi_db = KalshiDBManager()
        self.notifier = TelegramNotifier() if enable_notifications else None

        # State tracking
        self.monitored_games: Set[str] = set()  # External game IDs
        self.last_scores: Dict[str, tuple] = {}  # game_id -> (home_score, away_score)
        self.last_kalshi_prices: Dict[str, Decimal] = {}  # ticker -> price
        self.last_update_times: Dict[str, float] = {}  # game_id -> timestamp of last update
        self.game_intervals: Dict[str, int] = {}  # game_id -> current polling interval

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Login to Kalshi
        if not self.kalshi_client.login():
            logger.warning("Kalshi login failed - price monitoring disabled")

        logger.info(f"NFL Real-Time Sync initialized (update interval: {update_interval_seconds}s)")

    # ========================================================================
    # MAIN SYNC LOOP
    # ========================================================================

    def run(self) -> None:
        """
        Main sync loop - runs continuously

        This is the entry point for the background service
        """
        logger.info("Starting NFL real-time sync engine...")

        # Send startup notification
        if self.notifier:
            self.notifier.send_custom_message(
                "🏈 **NFL Real-Time Sync Started**\n\n"
                f"Update Interval: {self.update_interval}s\n"
                f"Monitoring live games and Kalshi markets..."
            )

        try:
            while True:
                cycle_start = time.time()

                # Run sync cycle
                self._sync_cycle()

                # Sleep until next update
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.update_interval - elapsed)

                if sleep_time > 0:
                    logger.debug(f"Cycle completed in {elapsed:.2f}s, sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Sync cycle took {elapsed:.2f}s (longer than {self.update_interval}s interval)")

        except KeyboardInterrupt:
            logger.info("Shutting down NFL real-time sync...")
            self.executor.shutdown(wait=True)

            if self.notifier:
                self.notifier.send_custom_message("🛑 **NFL Real-Time Sync Stopped**")

        except Exception as e:
            logger.error(f"Fatal error in sync loop: {e}", exc_info=True)

            if self.notifier:
                self.notifier.send_sync_error_alert(f"Fatal sync error: {str(e)}")

            raise

    def _sync_cycle(self) -> None:
        """Execute one complete sync cycle"""
        try:
            # 1. Update live game scores
            self._update_live_games()

            # 2. Update Kalshi prices for relevant markets
            self._update_kalshi_prices()

            # 3. Check for injuries (every 5 minutes, not every cycle)
            if int(time.time()) % 300 < self.update_interval:
                self._update_injuries()

            # 4. Process alert triggers
            self._process_alerts()

        except Exception as e:
            logger.error(f"Error in sync cycle: {e}", exc_info=True)

    # ========================================================================
    # ADAPTIVE POLLING LOGIC
    # ========================================================================

    def _get_smart_interval(self, game: Dict) -> int:
        """
        Determine optimal polling interval based on game state

        Args:
            game: Game dictionary with status, quarter, score differential

        Returns:
            Polling interval in seconds

        Intervals:
        - 5s: Live games in 4th quarter with score differential ≤ 7 points
        - 10s: Live games in regulation (1st-3rd quarter or 4th with >7 point diff)
        - 15s: Live games in halftime
        - 60s: Scheduled games (not yet started)
        - 300s: Completed games (minimal updates needed)
        """
        game_status = game.get('game_status', 'unknown')

        # Completed games - check infrequently
        if game_status in ['final', 'completed']:
            return 300

        # Scheduled games - moderate frequency
        if game_status == 'scheduled':
            return 60

        # Live games - adaptive based on game situation
        if game.get('is_live', False):
            quarter = game.get('quarter', 1)
            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            score_diff = abs(home_score - away_score)

            # Halftime - less frequent
            if quarter == 2 and game.get('time_remaining', '').startswith('Halftime'):
                return 15

            # 4th quarter close game - most frequent
            if quarter == 4 and score_diff <= 7:
                return 5

            # Regular live game - moderate frequency
            return 10

        # Default fallback
        return 15

    def _should_update_game(self, game: Dict) -> bool:
        """
        Check if this game should be updated in this cycle based on its interval

        Args:
            game: Game dictionary

        Returns:
            True if game should be updated now
        """
        game_id = game['game_id']
        current_time = time.time()

        # Get last update time
        last_update = self.last_update_times.get(game_id, 0)

        # Get appropriate interval for this game
        interval = self._get_smart_interval(game)

        # Check if interval has elapsed
        time_since_update = current_time - last_update
        should_update = time_since_update >= interval

        # Log interval changes
        old_interval = self.game_intervals.get(game_id)
        if old_interval != interval:
            self.game_intervals[game_id] = interval
            logger.info(
                f"Polling interval changed for {game.get('away_team', '?')} @ {game.get('home_team', '?')}: "
                f"{old_interval}s → {interval}s (Status: {game.get('game_status', '?')}, "
                f"Q{game.get('quarter', '?')}, Diff: {abs(game.get('home_score', 0) - game.get('away_score', 0))})"
            )

        return should_update

    # ========================================================================
    # LIVE GAME UPDATES
    # ========================================================================

    @rate_limited
    def _update_live_games(self) -> None:
        """Update all live games"""
        sync_id = self.nfl_db.start_sync_log('scores', 'live_games')

        try:
            # Fetch today's scoreboard
            scoreboard = self.fetcher.get_scoreboard()
            games = self.fetcher.parse_scoreboard_to_games(scoreboard)

            records_inserted = 0
            records_updated = 0
            skipped_games = 0
            api_calls = 1  # Scoreboard call

            for game in games:
                try:
                    # Check if this game should be updated based on adaptive polling
                    if not self._should_update_game(game):
                        skipped_games += 1
                        continue

                    # Update last update time
                    self.last_update_times[game['game_id']] = time.time()

                    # Check if game is live or upcoming
                    if game['is_live']:
                        # Add to monitored set
                        self.monitored_games.add(game['game_id'])

                        # Fetch weather if outdoor game
                        if game['is_outdoor'] and not game.get('temperature'):
                            weather = self.fetcher.get_weather_for_game(
                                game['venue'],
                                game['game_time']
                            )
                            if weather:
                                game.update(weather)
                                api_calls += 1

                        # Check for score changes
                        old_score = self.last_scores.get(game['game_id'])
                        new_score = (game['home_score'], game['away_score'])

                        # Upsert game
                        existing_game = self.nfl_db.get_game_by_external_id(game['game_id'])
                        game_db_id = self.nfl_db.upsert_game(game)

                        if existing_game:
                            records_updated += 1
                        else:
                            records_inserted += 1

                        # Detect score changes
                        if old_score and old_score != new_score:
                            logger.info(
                                f"Score update: {game['away_team']} {new_score[1]} @ "
                                f"{game['home_team']} {new_score[0]}"
                            )

                            # Fetch detailed play-by-play
                            self._update_play_by_play(game['game_id'], game_db_id)
                            api_calls += 1

                            # Send score alert
                            if self.notifier:
                                self._send_score_alert(game, old_score, new_score)

                        # Update tracking
                        self.last_scores[game['game_id']] = new_score

                    elif game['game_status'] == 'scheduled':
                        # Insert/update scheduled game
                        existing_game = self.nfl_db.get_game_by_external_id(game['game_id'])
                        self.nfl_db.upsert_game(game)

                        if existing_game:
                            records_updated += 1
                        else:
                            records_inserted += 1

                except Exception as e:
                    logger.error(f"Error updating game {game['game_id']}: {e}")

            # Complete sync log
            self.nfl_db.complete_sync_log(
                sync_id,
                records_fetched=len(games),
                records_inserted=records_inserted,
                records_updated=records_updated,
                api_calls=api_calls
            )

            logger.info(
                f"Live games sync: {records_inserted} inserted, {records_updated} updated, "
                f"{skipped_games} skipped (adaptive polling), {api_calls} API calls"
            )

        except Exception as e:
            logger.error(f"Error updating live games: {e}")
            self.nfl_db.complete_sync_log(
                sync_id,
                records_fetched=0,
                records_inserted=0,
                records_updated=0,
                status='failed',
                error_msg=str(e)
            )

    def _update_play_by_play(self, game_id: str, game_db_id: int):
        """Update play-by-play data for a game"""
        try:
            play_data = self.fetcher.get_play_by_play(game_id)
            plays = self.fetcher.parse_plays(play_data, game_db_id)

            for play in plays:
                try:
                    self.nfl_db.insert_play(play)

                    # Check for significant plays
                    if play['is_scoring_play'] or play['is_turnover']:
                        logger.info(
                            f"Significant play: {play['description'][:100]}"
                        )

                        # Track Kalshi correlation
                        self._track_kalshi_correlation(game_db_id, play)

                except Exception as e:
                    logger.error(f"Error inserting play: {e}")

        except Exception as e:
            logger.error(f"Error updating play-by-play for {game_id}: {e}")

    # ========================================================================
    # KALSHI PRICE MONITORING
    # ========================================================================

    def _update_kalshi_prices(self) -> None:
        """Update Kalshi market prices for NFL games"""
        try:
            # Get all NFL markets
            markets = self.kalshi_client.get_football_markets()
            nfl_markets = markets.get('nfl', [])

            if not nfl_markets:
                return

            # Store markets and track price changes
            self.kalshi_db.store_markets(nfl_markets, 'nfl')

            for market in nfl_markets:
                ticker = market['ticker']
                current_price = market.get('last_price', 0) / 100  # Convert cents to decimal

                # Check for significant price movement
                old_price = self.last_kalshi_prices.get(ticker)

                if old_price and abs(current_price - old_price) / old_price > 0.10:  # 10% change
                    price_change_pct = ((current_price - old_price) / old_price) * 100

                    logger.info(
                        f"Kalshi price spike: {ticker} {old_price:.2f} -> {current_price:.2f} "
                        f"({price_change_pct:+.1f}%)"
                    )

                    # Send alert
                    if self.notifier:
                        self._send_kalshi_price_alert(market, old_price, current_price, price_change_pct)

                # Update tracking
                self.last_kalshi_prices[ticker] = current_price

                # Store price snapshot
                self.kalshi_db.store_price_snapshot(
                    ticker,
                    current_price,
                    1 - current_price,
                    market.get('volume', 0),
                    market.get('open_interest', 0)
                )

        except Exception as e:
            logger.error(f"Error updating Kalshi prices: {e}")

    def _track_kalshi_correlation(self, game_db_id: int, play: Dict):
        """
        Track correlation between play and Kalshi price movement

        This is called after significant plays to see if Kalshi markets react
        """
        try:
            # Find Kalshi markets for this game
            game = self.nfl_db.get_game_by_external_id(play['game_id'])
            if not game:
                return

            # Query Kalshi markets
            markets = self.kalshi_db.get_active_markets('nfl')

            for market in markets:
                # Check if market is related to this game
                if game['home_team'] in market.get('title', '') or game['away_team'] in market.get('title', ''):
                    ticker = market['ticker']

                    # Get price before and after
                    old_price = self.last_kalshi_prices.get(ticker)
                    if not old_price:
                        continue

                    # Fetch current price
                    market_details = self.kalshi_client.get_market_details(ticker)
                    if not market_details:
                        continue

                    new_price = market_details.get('last_price', 0) / 100

                    # Calculate change
                    if old_price and abs(new_price - old_price) > 0.01:  # At least 1% change
                        price_change_pct = ((new_price - old_price) / old_price) * 100

                        # Determine impact level
                        impact_level = 'low'
                        if abs(price_change_pct) > 20:
                            impact_level = 'extreme'
                        elif abs(price_change_pct) > 10:
                            impact_level = 'high'
                        elif abs(price_change_pct) > 5:
                            impact_level = 'medium'

                        # Store correlation
                        correlation_data = {
                            'game_id': game_db_id,
                            'play_id': play.get('id'),  # DB ID, not external ID
                            'event_type': 'scoring_play' if play['is_scoring_play'] else 'turnover',
                            'event_timestamp': datetime.now(),
                            'kalshi_market_id': market['id'],
                            'market_ticker': ticker,
                            'price_before': Decimal(str(old_price)),
                            'price_after': Decimal(str(new_price)),
                            'price_change_pct': Decimal(str(price_change_pct)),
                            'volume_before': None,
                            'volume_after': None,
                            'volume_spike_pct': None,
                            'correlation_strength': None,  # Could calculate with more data
                            'impact_level': impact_level
                        }

                        self.nfl_db.insert_kalshi_correlation(correlation_data)

        except Exception as e:
            logger.error(f"Error tracking Kalshi correlation: {e}")

    # ========================================================================
    # INJURY UPDATES
    # ========================================================================

    def _update_injuries(self) -> None:
        """Update injury reports (runs less frequently)"""
        try:
            logger.info("Updating injury reports...")

            injuries = self.fetcher.get_injuries()

            for injury in injuries:
                try:
                    self.nfl_db.insert_injury_report(injury)

                    # Alert on new "Out" status for key positions
                    if injury['injury_status'] == 'Out' and injury['position'] in ['QB', 'RB', 'WR']:
                        if self.notifier:
                            self._send_injury_alert(injury)

                except Exception as e:
                    logger.error(f"Error inserting injury: {e}")

        except Exception as e:
            logger.error(f"Error updating injuries: {e}")

    # ========================================================================
    # ALERT PROCESSING
    # ========================================================================

    def _process_alerts(self) -> None:
        """Process configured alert triggers"""
        try:
            # Get active triggers
            triggers = self.nfl_db.get_active_alert_triggers()

            for trigger in triggers:
                # Check cooldown
                if trigger['last_triggered']:
                    cooldown_end = trigger['last_triggered'] + timedelta(minutes=trigger['cooldown_minutes'])
                    if datetime.now() < cooldown_end:
                        continue

                # Check daily limit
                # (Would need to query alert_history table - simplified for now)

                # Evaluate trigger conditions based on type
                # This is a simplified example - full implementation would be more complex
                alert_type = trigger['alert_type']
                conditions = trigger['trigger_conditions']

                # Example: price_movement trigger
                if alert_type == 'price_movement':
                    min_change = conditions.get('min_price_change', 10)

                    # Check if any tracked markets exceeded threshold
                    for ticker, price in self.last_kalshi_prices.items():
                        # (Would compare to previous price and send alert if threshold met)
                        pass

        except Exception as e:
            logger.error(f"Error processing alerts: {e}")

    # ========================================================================
    # NOTIFICATION HELPERS
    # ========================================================================

    def _send_score_alert(self, game: Dict, old_score: tuple, new_score: tuple):
        """Send Telegram alert for score change"""
        try:
            old_away, old_home = old_score
            new_away, new_home = new_score

            # Determine what changed
            if new_home > old_home:
                scoring_team = game['home_team']
                points = new_home - old_home
            else:
                scoring_team = game['away_team']
                points = new_away - old_away

            message = (
                f"🏈 **SCORE UPDATE**\n\n"
                f"**{game['away_team']}** {new_away} @ **{game['home_team']}** {new_home}\n\n"
                f"🎯 {scoring_team} scores {points} points!\n"
                f"⏱️ Q{game['quarter']} - {game['time_remaining']}\n"
            )

            self.notifier.send_custom_message(message)

        except Exception as e:
            logger.error(f"Error sending score alert: {e}")

    def _send_kalshi_price_alert(self, market: Dict, old_price: float, new_price: float, change_pct: float):
        """Send Telegram alert for Kalshi price spike"""
        try:
            emoji = "📈" if change_pct > 0 else "📉"

            message = (
                f"{emoji} **KALSHI PRICE MOVEMENT**\n\n"
                f"**{market['title']}**\n\n"
                f"Price: `{old_price:.2f}` → `{new_price:.2f}` ({change_pct:+.1f}%)\n"
                f"Volume: ${market.get('volume', 0):,.0f}\n"
                f"Ticker: `{market['ticker']}`\n"
            )

            self.notifier.send_custom_message(message)

        except Exception as e:
            logger.error(f"Error sending Kalshi alert: {e}")

    def _send_injury_alert(self, injury: Dict):
        """Send Telegram alert for injury report"""
        try:
            message = (
                f"🚑 **INJURY UPDATE**\n\n"
                f"**{injury['player_name']}** ({injury['position']})\n"
                f"Team: {injury['team']}\n"
                f"Status: **{injury['injury_status']}**\n"
                f"Injury: {injury['injury_type']}\n"
            )

            if injury['description']:
                message += f"\nDetails: _{injury['description']}_"

            self.notifier.send_custom_message(message)

        except Exception as e:
            logger.error(f"Error sending injury alert: {e}")


# ========================================================================
# STANDALONE RUNNER
# ========================================================================

def main():
    """Main entry point for running as a background service"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       🏈 NFL REAL-TIME SYNC ENGINE 🏈                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Starting NFL real-time data pipeline...

Features:
- Live game scores (updates every 5 seconds)
- Play-by-play tracking
- Kalshi market monitoring
- Telegram notifications
- Injury reports

Press Ctrl+C to stop
""")

    # Create sync engine
    sync_engine = NFLRealtimeSync(
        update_interval_seconds=5,
        enable_notifications=True
    )

    # Run forever
    sync_engine.run()


if __name__ == "__main__":
    main()
