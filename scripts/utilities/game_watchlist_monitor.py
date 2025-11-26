"""
Game Watchlist Background Monitor
Sends periodic Telegram updates for watched games

Features:
- Configurable update frequency (default: 5 minutes)
- Includes Kalshi odds in updates
- Includes AI predictions in updates
- Detects score changes, status changes, odds changes
- Only sends updates when there are meaningful changes
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.game_watchlist_manager import GameWatchlistManager
from src.kalshi_db_manager import KalshiDBManager
from src.espn_live_data import get_espn_client
from src.espn_ncaa_live_data import get_espn_ncaa_client
from src.espn_kalshi_matcher import enrich_games_with_kalshi_odds
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent
from src.telegram_notifier import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game_watchlist_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GameWatchlistMonitor:
    """
    Background service that monitors watched games and sends Telegram updates
    """

    def __init__(self, update_interval_minutes: int = 5):
        """
        Initialize the monitor

        Args:
            update_interval_minutes: How often to check for updates (default: 5 minutes)
        """
        self.db = KalshiDBManager()
        self.watchlist_manager = GameWatchlistManager(self.db)
        self.telegram = TelegramNotifier()
        self.ai_agent = AdvancedBettingAIAgent()
        self.espn_nfl = get_espn_client()
        self.espn_ncaa = get_espn_ncaa_client()

        # Configuration
        self.update_interval_minutes = update_interval_minutes
        self.update_interval_seconds = update_interval_minutes * 60

        # Get user ID from environment or use default
        self.user_id = os.getenv('TELEGRAM_USER_ID', 'default_user')

        logger.info(f"Watchlist Monitor initialized - Updates every {update_interval_minutes} minutes")

    def get_all_watched_games(self) -> List[Dict]:
        """Get all active games in user's watchlist"""
        conn = None
        cur = None
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

            return watched_games

        except Exception as e:
            logger.error(f"Error fetching watched games: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def get_live_game_data(self, game_id: str, sport: str) -> Optional[Dict]:
        """
        Fetch live data for a specific game from ESPN

        Args:
            game_id: Game identifier
            sport: 'NFL' or 'CFB'

        Returns:
            Game data dict with scores, status, etc.
        """
        try:
            # Get games from appropriate ESPN API
            if sport == 'NFL':
                games = self.espn_nfl.get_live_scores()
            elif sport == 'CFB':
                games = self.espn_ncaa.get_live_scores()
            else:
                logger.warning(f"Unknown sport: {sport}")
                return None

            # Find matching game
            for game in games:
                if str(game.get('game_id', '')) == str(game_id):
                    # Enrich with Kalshi odds
                    enriched = enrich_games_with_kalshi_odds([game])
                    if enriched:
                        return enriched[0]
                    return game

            logger.debug(f"Game {game_id} not found in ESPN data")
            return None

        except Exception as e:
            logger.error(f"Error fetching live game data: {e}")
            return None

    def get_last_known_state(self, game_id: str) -> Optional[Dict]:
        """Get the last recorded state for a game"""
        conn = None
        cur = None
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT
                    away_score,
                    home_score,
                    status,
                    period,
                    clock,
                    ai_confidence,
                    ai_predicted_winner,
                    ai_win_probability,
                    kalshi_away_odds,
                    kalshi_home_odds,
                    timestamp
                FROM game_state_history
                WHERE game_id = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (game_id,))

            row = cur.fetchone()
            if row:
                return {
                    'away_score': row[0],
                    'home_score': row[1],
                    'status': row[2],
                    'period': row[3],
                    'clock': row[4],
                    'ai_confidence': float(row[5]) if row[5] else 0,
                    'ai_predicted_winner': row[6],
                    'ai_win_probability': float(row[7]) if row[7] else 0,
                    'kalshi_away_odds': float(row[8]) if row[8] else 0,
                    'kalshi_home_odds': float(row[9]) if row[9] else 0,
                    'timestamp': row[10]
                }
            return None

        except Exception as e:
            logger.error(f"Error fetching last state: {e}")
            return None
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def save_game_state(self, game_id: str, sport: str, game_data: Dict, ai_prediction: Dict):
        """Save current game state to history"""
        conn = None
        cur = None
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            kalshi_odds = game_data.get('kalshi_odds', {})

            cur.execute("""
                INSERT INTO game_state_history (
                    game_id,
                    sport,
                    away_score,
                    home_score,
                    status,
                    period,
                    clock,
                    ai_confidence,
                    ai_predicted_winner,
                    ai_win_probability,
                    kalshi_away_odds,
                    kalshi_home_odds,
                    timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                game_id,
                sport,
                game_data.get('away_score', 0),
                game_data.get('home_score', 0),
                game_data.get('status_detail', ''),
                game_data.get('period', ''),
                game_data.get('clock', ''),
                ai_prediction.get('confidence_score', 0),
                ai_prediction.get('predicted_winner', ''),
                ai_prediction.get('win_probability', 0),
                kalshi_odds.get('away_win_price', 0),
                kalshi_odds.get('home_win_price', 0)
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error saving game state: {e}")
            if conn:
                conn.rollback()
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def detect_changes(self, game_data: Dict, last_state: Optional[Dict], watched_game: Dict) -> List[str]:
        """
        Detect meaningful changes worth notifying about

        Returns:
            List of change descriptions
        """
        if not last_state:
            return ["Initial update - game added to watchlist"]

        changes = []

        # Score changes
        current_away = game_data.get('away_score', 0)
        current_home = game_data.get('home_score', 0)
        last_away = last_state.get('away_score', 0)
        last_home = last_state.get('home_score', 0)

        if current_away != last_away or current_home != last_home:
            changes.append(f"Score changed: {last_away}-{last_home} → {current_away}-{current_home}")

        # Status changes (quarter, period, final)
        current_period = game_data.get('period', '')
        last_period = last_state.get('period', '')

        if current_period != last_period:
            changes.append(f"Period changed: {last_period} → {current_period}")

        # Game status (pre-game, live, final)
        current_status = game_data.get('status_detail', '')
        last_status = last_state.get('status', '')

        if current_status != last_status and current_status:
            changes.append(f"Status: {current_status}")

        # Check if user's team status changed (winning/losing)
        selected_team = watched_game.get('selected_team', '')
        away_team = watched_game.get('away_team', '')

        if selected_team:
            # Current status
            is_away = selected_team == away_team
            current_winning = (current_away > current_home) if is_away else (current_home > current_away)
            last_winning = (last_away > last_home) if is_away else (last_home > last_away)

            if current_winning != last_winning:
                if current_winning:
                    changes.append(f"🎉 {selected_team} is now WINNING!")
                else:
                    changes.append(f"⚠️ {selected_team} is now LOSING")

        # Significant odds changes (>10 cents)
        kalshi_odds = game_data.get('kalshi_odds', {})
        if kalshi_odds:
            current_away_odds = kalshi_odds.get('away_win_price', 0)
            current_home_odds = kalshi_odds.get('home_win_price', 0)
            last_away_odds = last_state.get('kalshi_away_odds', 0)
            last_home_odds = last_state.get('kalshi_home_odds', 0)

            away_change = abs(current_away_odds - last_away_odds)
            home_change = abs(current_home_odds - last_home_odds)

            if away_change > 0.10:
                changes.append(f"Odds shift: {away_team} {last_away_odds:.0%} → {current_away_odds:.0%}")
            if home_change > 0.10:
                home_team = watched_game.get('home_team', '')
                changes.append(f"Odds shift: {home_team} {last_home_odds:.0%} → {current_home_odds:.0%}")

        return changes

    def build_telegram_update(self, watched_game: Dict, game_data: Dict, ai_prediction: Dict, changes: List[str]) -> str:
        """
        Build formatted Telegram message with game details, odds, and AI prediction

        Args:
            watched_game: Watchlist entry
            game_data: Live game data from ESPN
            ai_prediction: AI analysis
            changes: List of detected changes

        Returns:
            Formatted message string
        """
        away_team = watched_game['away_team']
        home_team = watched_game['home_team']
        selected_team = watched_game.get('selected_team', '')

        away_score = game_data.get('away_score', 0)
        home_score = game_data.get('home_score', 0)
        status = game_data.get('status_detail', 'Scheduled')
        period = game_data.get('period', '')
        clock = game_data.get('clock', '')

        # Build status line
        if clock and period:
            status_line = f"{status} - {period} {clock}"
        else:
            status_line = status

        # Build message
        message = f"""🔔 **GAME UPDATE**

🏈 **{away_team} @ {home_team}**
**{away_score} - {home_score}**
_{status_line}_
"""

        # Add changes section
        if changes:
            message += "\n**📊 What Changed:**\n"
            for change in changes:
                message += f"• {change}\n"

        # Your team status
        if selected_team:
            is_away = selected_team == away_team
            is_winning = (away_score > home_score) if is_away else (home_score > away_score)
            is_tied = away_score == home_score

            if is_tied:
                team_status = "⚖️ TIED"
                status_detail = "Game is tied"
            elif is_winning:
                team_status = "✅ WINNING"
                point_diff = abs(away_score - home_score)
                status_detail = f"By {point_diff} points"
            else:
                team_status = "❌ LOSING"
                point_diff = abs(away_score - home_score)
                status_detail = f"By {point_diff} points"

            message += f"\n🔥 **Your Team ({selected_team}): {team_status}**\n   {status_detail}\n"

        # Kalshi odds
        kalshi_odds = game_data.get('kalshi_odds', {})
        if kalshi_odds:
            away_odds = kalshi_odds.get('away_win_price', 0)
            home_odds = kalshi_odds.get('home_win_price', 0)

            away_cents = int(away_odds * 100) if away_odds else 0
            home_cents = int(home_odds * 100) if home_odds else 0

            message += f"""
💰 **Kalshi Odds:**
   {away_team}: {away_cents}¢
   {home_team}: {home_cents}¢
"""
        else:
            message += "\n💰 **Kalshi Odds:** Not available\n"

        # AI prediction
        predicted_winner_text = away_team if ai_prediction['predicted_winner'] == 'away' else home_team
        win_prob = ai_prediction.get('win_probability', 0) * 100
        confidence = ai_prediction.get('confidence_score', 0) * 100
        ev = ai_prediction.get('expected_value', 0)
        recommendation = ai_prediction.get('recommendation', 'N/A')
        model_used = ai_prediction.get('model_used', 'Local AI')

        ai_emoji = "✅" if predicted_winner_text == selected_team else "❌"

        message += f"""
{ai_emoji} 🤖 **AI Predicts: {predicted_winner_text} wins**
   Model: {model_used}
   Win Probability: {win_prob:.0f}%
   Confidence: {confidence:.0f}%
   Expected Value: {ev:+.1f}%
   Recommendation: **{recommendation}**

_Last updated: {datetime.now().strftime('%I:%M %p')}_
"""

        return message

    def process_watched_game(self, watched_game: Dict):
        """
        Process a single watched game - check for updates and send notifications

        Args:
            watched_game: Watchlist entry dict
        """
        game_id = watched_game['game_id']
        sport = watched_game['sport']

        logger.info(f"Checking game: {watched_game['away_team']} @ {watched_game['home_team']}")

        # Get live data
        game_data = self.get_live_game_data(game_id, sport)
        if not game_data:
            logger.warning(f"Could not fetch data for game {game_id}")
            return

        # Generate AI prediction
        try:
            market_data = game_data.get('kalshi_odds', {})
            ai_prediction = self.ai_agent.analyze_betting_opportunity(game_data, market_data)
        except Exception as e:
            logger.error(f"AI prediction failed: {e}")
            ai_prediction = {
                'predicted_winner': 'away' if game_data.get('away_score', 0) > game_data.get('home_score', 0) else 'home',
                'win_probability': 0.5,
                'confidence_score': 0,
                'expected_value': 0,
                'recommendation': 'PASS',
                'reasoning': ['AI unavailable']
            }

        # Get last known state
        last_state = self.get_last_known_state(game_id)

        # Detect changes
        changes = self.detect_changes(game_data, last_state, watched_game)

        # Send update if there are changes
        if changes:
            logger.info(f"Changes detected: {len(changes)} changes")

            # Build and send message
            message = self.build_telegram_update(watched_game, game_data, ai_prediction, changes)

            try:
                self.telegram.send_custom_message(message)
                logger.info(f"✅ Sent update for {watched_game['away_team']} @ {watched_game['home_team']}")
            except Exception as e:
                logger.error(f"Failed to send Telegram update: {e}")
        else:
            logger.info("No significant changes detected")

        # Save current state
        self.save_game_state(game_id, sport, game_data, ai_prediction)

    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle - check all watched games"""
        logger.info("=" * 80)
        logger.info(f"Starting monitoring cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Get all watched games
        watched_games = self.get_all_watched_games()

        if not watched_games:
            logger.info("No games in watchlist")
            return

        logger.info(f"Monitoring {len(watched_games)} games")

        # Process each game
        for watched_game in watched_games:
            try:
                self.process_watched_game(watched_game)
            except Exception as e:
                logger.error(f"Error processing game: {e}")

            # Small delay between games to avoid rate limiting
            time.sleep(1)

        logger.info(f"Monitoring cycle complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def start(self):
        """Start the background monitoring service"""
        logger.info("=" * 80)
        logger.info("GAME WATCHLIST MONITOR STARTED")
        logger.info(f"Update interval: {self.update_interval_minutes} minutes")
        logger.info(f"User ID: {self.user_id}")
        logger.info("=" * 80)

        # Test Telegram connection
        try:
            test_message = f"""🤖 **Watchlist Monitor Started**

✅ Monitoring your games every {self.update_interval_minutes} minutes
✅ Updates include:
   • Score changes
   • Odds updates
   • AI predictions
   • Game status changes

You'll receive notifications when anything changes!

_{datetime.now().strftime('%I:%M %p')}_
"""
            self.telegram.send_custom_message(test_message)
            logger.info("✅ Telegram connection test successful")
        except Exception as e:
            logger.error(f"❌ Telegram connection failed: {e}")
            logger.warning("Monitor will continue but alerts won't be sent")

        # Main monitoring loop
        try:
            while True:
                try:
                    self.run_monitoring_cycle()
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")

                # Sleep until next check
                logger.info(f"Sleeping for {self.update_interval_minutes} minutes...")
                time.sleep(self.update_interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("Monitor stopped by user")
            logger.info("=" * 80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Game Watchlist Background Monitor')
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Update interval in minutes (default: 5)'
    )

    args = parser.parse_args()

    # Create and start monitor
    monitor = GameWatchlistMonitor(update_interval_minutes=args.interval)
    monitor.start()


if __name__ == "__main__":
    main()
