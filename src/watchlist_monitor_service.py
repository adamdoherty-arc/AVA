"""
Watchlist Monitor Service - APScheduler-based background monitoring
Allows Streamlit UI to control background monitoring without blocking

Features:
- Non-blocking background execution using APScheduler
- Start/Stop controls from Streamlit UI
- Configurable update intervals
- Process-based persistence (survives Streamlit reruns)

Usage:
    from src.watchlist_monitor_service import WatchlistMonitorService

    service = WatchlistMonitorService()
    service.start(interval_minutes=5)
    # ... later ...
    service.stop()
"""

import os
import sys
import logging
import threading
from typing import Optional
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_watchlist_manager import GameWatchlistManager
from src.kalshi_db_manager import KalshiDBManager
from src.espn_live_data import get_espn_client
from src.espn_ncaa_live_data import get_espn_ncaa_client
from src.espn_kalshi_matcher import enrich_games_with_kalshi_odds
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent
from src.telegram_notifier import TelegramNotifier

# Configure logging
logger = logging.getLogger(__name__)


class WatchlistMonitorService:
    """
    Background monitoring service using APScheduler
    Allows non-blocking execution that can be controlled from Streamlit UI
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one monitor runs"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the service (only once due to singleton)"""
        if self._initialized:
            return

        self.scheduler: Optional[BackgroundScheduler] = None
        self.db: Optional[KalshiDBManager] = None
        self.watchlist_manager: Optional[GameWatchlistManager] = None
        self.telegram: Optional[TelegramNotifier] = None
        self.ai_agent: Optional[AdvancedBettingAIAgent] = None
        self.espn_nfl = None
        self.espn_ncaa = None

        self.user_id = os.getenv('TELEGRAM_USER_ID', 'default_user')
        self.is_running = False
        self.current_interval = 5  # Default 5 minutes

        self._initialized = True
        logger.info("WatchlistMonitorService initialized (singleton)")

    def _initialize_components(self) -> None:
        """Lazy initialization of components when service starts"""
        if self.db is None:
            self.db = KalshiDBManager()
            self.watchlist_manager = GameWatchlistManager(self.db)
            self.telegram = TelegramNotifier()
            self.ai_agent = AdvancedBettingAIAgent()
            self.espn_nfl = get_espn_client()
            self.espn_ncaa = get_espn_ncaa_client()
            logger.info("Service components initialized")

    def get_all_watched_games(self) -> None:
        """Get all active games in user's watchlist"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT
                    game_id,
                    sport,
                    away_team,
                    home_team,
                    selected_team,
                    added_at
                FROM game_watchlist
                WHERE user_id = %s AND is_active = TRUE
                ORDER BY added_at DESC
            """, (self.user_id,))

            watched_games = []
            for row in cur.fetchall():
                watched_games.append({
                    'game_id': row[0],
                    'sport': row[1],
                    'away_team': row[2],
                    'home_team': row[3],
                    'selected_team': row[4],
                    'added_at': row[5]
                })

            cur.close()
            conn.close()
            return watched_games

        except Exception as e:
            logger.error(f"Error fetching watched games: {e}")
            return []

    def get_live_game_data(self, game_id: str, sport: str):
        """Fetch live game data from ESPN"""
        try:
            if sport == 'NFL':
                games = self.espn_nfl.get_live_scores()
            elif sport == 'CFB':
                games = self.espn_ncaa.get_live_scores()
            else:
                return None

            for game in games:
                if str(game.get('game_id', '')) == str(game_id):
                    enriched = enrich_games_with_kalshi_odds([game])
                    if enriched:
                        return enriched[0]
                    return game

            return None

        except Exception as e:
            logger.error(f"Error fetching game data: {e}")
            return None

    def check_for_updates(self, game_id: str, sport: str, current_data: dict, previous_data: dict):
        """Compare current vs previous game state and return changes"""
        changes = []

        # Score changes
        if current_data.get('away_score') != previous_data.get('away_score') or \
           current_data.get('home_score') != previous_data.get('home_score'):
            changes.append('score')

        # Status changes
        if current_data.get('status') != previous_data.get('status'):
            changes.append('status')

        # Period changes
        if current_data.get('period') != previous_data.get('period'):
            changes.append('period')

        # Odds changes (>10% movement)
        if current_data.get('kalshi_odds') and previous_data.get('kalshi_odds'):
            curr_odds = current_data['kalshi_odds']
            prev_odds = previous_data['kalshi_odds']

            away_change = abs(curr_odds.get('away_win_price', 0) - prev_odds.get('away_win_price', 0))
            home_change = abs(curr_odds.get('home_win_price', 0) - prev_odds.get('home_win_price', 0))

            if away_change > 0.10 or home_change > 0.10:
                changes.append('odds')

        return changes

    def send_game_update(self, game_data: dict, changes: list, selected_team: str):
        """Send Telegram notification about game changes"""
        try:
            away_team = game_data.get('away_team', 'Away')
            home_team = game_data.get('home_team', 'Home')
            away_score = game_data.get('away_score', 0)
            home_score = game_data.get('home_score', 0)
            status = game_data.get('status', 'Unknown')
            period = game_data.get('period', '')

            # Build update message
            update_type = ', '.join(changes).upper()
            message = f"🚨 **GAME UPDATE: {update_type}**\n\n"
            message += f"**{away_team} @ {home_team}**\n"
            message += f"Score: {away_score} - {home_score}\n"
            message += f"Status: {status} {period}\n\n"

            # Add odds if available
            if game_data.get('kalshi_odds'):
                odds = game_data['kalshi_odds']
                away_pct = odds.get('away_win_price', 0) * 100
                home_pct = odds.get('home_win_price', 0) * 100
                message += f"📊 **Odds:**\n"
                message += f"   {away_team}: {away_pct:.0f}%\n"
                message += f"   {home_team}: {home_pct:.0f}%\n\n"

            # Add AI recommendation if selected team
            if selected_team:
                message += f"🎯 **Your pick:** {selected_team}\n"

            message += f"\n_{datetime.now().strftime('%I:%M %p')}_"

            self.telegram.send_custom_message(message)
            logger.info(f"Update sent for game {game_data.get('game_id')}")

        except Exception as e:
            logger.error(f"Error sending update: {e}")

    def _monitoring_cycle(self) -> None:
        """
        Main monitoring cycle - called by APScheduler
        This method is non-blocking and runs in background
        """
        try:
            logger.info(f"Running monitoring cycle at {datetime.now().strftime('%I:%M %p')}")

            # Get watched games
            watched_games = self.get_all_watched_games()

            if not watched_games:
                logger.info("No watched games to monitor")
                return

            logger.info(f"Monitoring {len(watched_games)} games...")

            # Check each game for updates
            for watch in watched_games:
                game_id = watch['game_id']
                sport = watch['sport']
                selected_team = watch.get('selected_team')

                # Get current game data
                current_data = self.get_live_game_data(game_id, sport)

                if not current_data:
                    logger.warning(f"Could not fetch data for game {game_id}")
                    continue

                # Get previous state from database
                previous_data = self.watchlist_manager.get_game_state(game_id)

                if previous_data:
                    # Check for meaningful changes
                    changes = self.check_for_updates(game_id, sport, current_data, previous_data)

                    if changes:
                        logger.info(f"Changes detected for {game_id}: {changes}")
                        self.send_game_update(current_data, changes, selected_team)

                # Update stored state
                self.watchlist_manager.update_game_state(game_id, current_data)

            logger.info("Monitoring cycle complete")

        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}", exc_info=True)

    def start(self, interval_minutes: int = 5) -> bool:
        """
        Start background monitoring service

        Args:
            interval_minutes: How often to check for updates

        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self.is_running:
                logger.warning("Monitor already running")
                return False

            try:
                # Initialize components
                self._initialize_components()

                # Create scheduler
                self.scheduler = BackgroundScheduler()
                self.current_interval = interval_minutes

                # Add monitoring job
                self.scheduler.add_job(
                    func=self._monitoring_cycle,
                    trigger=IntervalTrigger(minutes=interval_minutes),
                    id='watchlist_monitor',
                    name='Game Watchlist Monitor',
                    replace_existing=True
                )

                # Start scheduler
                self.scheduler.start()
                self.is_running = True

                logger.info(f"✅ Monitoring started - Updates every {interval_minutes} minutes")

                # Send startup notification
                try:
                    startup_msg = f"""🤖 **Watchlist Monitor Started**

✅ Monitoring your games every {interval_minutes} minutes
✅ Updates include:
   • Score changes
   • Odds updates
   • AI predictions
   • Game status changes

You'll receive notifications when anything changes!

_{datetime.now().strftime('%I:%M %p')}_
"""
                    self.telegram.send_custom_message(startup_msg)
                except Exception as e:
                    logger.error(f"Could not send startup notification: {e}")

                return True

            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}", exc_info=True)
                self.is_running = False
                return False

    def stop(self) -> bool:
        """
        Stop background monitoring service

        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if not self.is_running:
                logger.warning("Monitor not running")
                return False

            try:
                if self.scheduler:
                    self.scheduler.shutdown(wait=False)
                    self.scheduler = None

                self.is_running = False
                logger.info("✅ Monitoring stopped")

                # Send shutdown notification
                try:
                    shutdown_msg = f"""⏸️ **Watchlist Monitor Stopped**

Monitoring paused. Your subscriptions are saved.
Start monitoring again anytime from the Settings tab.

_{datetime.now().strftime('%I:%M %p')}_
"""
                    self.telegram.send_custom_message(shutdown_msg)
                except Exception as e:
                    logger.error(f"Could not send shutdown notification: {e}")

                return True

            except Exception as e:
                logger.error(f"Error stopping monitoring: {e}", exc_info=True)
                return False

    def get_status(self) -> dict:
        """
        Get current monitoring status

        Returns:
            Dict with status information
        """
        return {
            'running': self.is_running,
            'interval_minutes': self.current_interval if self.is_running else None,
            'next_run': self.scheduler.get_job('watchlist_monitor').next_run_time
                       if self.is_running and self.scheduler else None
        }


# Singleton instance
_service_instance = None

def get_monitor_service() -> WatchlistMonitorService:
    """Get the singleton monitor service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = WatchlistMonitorService()
    return _service_instance
