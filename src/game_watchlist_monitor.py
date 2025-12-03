"""
Background Game Watchlist Monitor
Continuously monitors watched games and sends Telegram updates
"""

import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from src.kalshi_db_manager import KalshiDBManager
from src.game_watchlist_manager import GameWatchlistManager
from src.telegram_notifier import TelegramNotifier
from src.espn_live_data import get_espn_client
from src.espn_ncaa_live_data import get_espn_ncaa_client
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent

logger = logging.getLogger(__name__)


class GameWatchlistMonitor:
    """
    Background service to monitor watched games and send Telegram updates
    
    Features:
    - Polls watched games every 30 seconds
    - Detects score changes, odds changes, AI prediction updates
    - Sends Telegram notifications for significant changes
    - Tracks position P&L and alerts on 20% thresholds
    """

    def __init__(self, check_interval: int = 30):
        """
        Initialize monitor
        
        Args:
            check_interval: Seconds between checks (default: 30)
        """
        self.db = KalshiDBManager()
        self.watchlist_manager = GameWatchlistManager(self.db)
        self.telegram_notifier = TelegramNotifier()
        self.check_interval = check_interval
        self.running = False
        self.ai_agent = AdvancedBettingAIAgent()
        
        logger.info(f"GameWatchlistMonitor initialized (check interval: {check_interval}s)")

    async def start(self) -> None:
        """Start monitoring loop"""
        self.running = True
        logger.info("GameWatchlistMonitor started")
        
        while self.running:
            try:
                await self._check_all_watched_games()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop monitoring"""
        self.running = False
        logger.info("GameWatchlistMonitor stopped")

    async def _check_all_watched_games(self) -> None:
        """Check all watched games for changes"""
        try:
            # Get all active watchlists for all users
            conn = self.db.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT DISTINCT user_id, game_id, sport, selected_team, entry_price, entry_team
                FROM game_watchlist
                WHERE is_active = TRUE
            """)
            
            watchlist_entries = cur.fetchall()
            cur.close()
            conn.close()
            
            if not watchlist_entries:
                return
            
            logger.debug(f"Checking {len(watchlist_entries)} watched games")
            
            # Group by user
            user_games = {}
            for entry in watchlist_entries:
                user_id = entry[0]
                if user_id not in user_games:
                    user_games[user_id] = []
                user_games[user_id].append({
                    'game_id': entry[1],
                    'sport': entry[2],
                    'selected_team': entry[3],
                    'entry_price': entry[4],
                    'entry_team': entry[5]
                })
            
            # Check each user's games
            for user_id, games in user_games.items():
                for game_info in games:
                    await self._check_game(user_id, game_info)
                    
        except Exception as e:
            logger.error(f"Error checking watched games: {e}")

    async def _check_game(self, user_id: str, game_info: Dict):
        """Check a single game for changes"""
        try:
            game_id = game_info['game_id']
            sport = game_info['sport']
            selected_team = game_info.get('selected_team')
            entry_price = game_info.get('entry_price')
            entry_team = game_info.get('entry_team')
            
            # Fetch current game data
            if sport == 'CFB' or sport == 'NCAA':
                espn = get_espn_ncaa_client()
                espn_games = espn.get_scoreboard(group='80')
            else:
                espn = get_espn_client()
                espn_games = espn.get_scoreboard()
            
            # Find matching game
            current_game = None
            for game in espn_games:
                if str(game.get('game_id', '')) == str(game_id):
                    current_game = game
                    break
            
            if not current_game:
                logger.debug(f"Game {game_id} not found in ESPN data")
                return
            
            # Enrich with Kalshi odds
            try:
                from src.espn_kalshi_matcher_optimized import enrich_games_with_kalshi_odds_optimized as enrich_games_with_kalshi_odds
                current_game = enrich_games_with_kalshi_odds([current_game])[0]
            except:
                pass
            
            # Get AI prediction
            try:
                kalshi_odds = current_game.get('kalshi_odds', {})
                ai_prediction = self.ai_agent.analyze_betting_opportunity(
                    current_game, kalshi_odds
                )
            except:
                ai_prediction = {}
            
            # Detect changes
            changes = self.watchlist_manager.detect_changes(
                current_game, ai_prediction
            )
            
            # Record current state
            self.watchlist_manager.record_game_state(current_game, ai_prediction)
            
            # Check for significant changes
            has_significant_change = (
                changes.get('score_changed') or
                changes.get('period_changed') or
                changes.get('ai_changed') or
                changes.get('odds_changed')
            )
            
            if has_significant_change:
                # Generate and send update message
                message = self.watchlist_manager.generate_game_update_message(
                    current_game, ai_prediction, changes, selected_team
                )
                
                try:
                    self.telegram_notifier.send_message(message)
                    logger.info(f"Sent update for game {game_id} to user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send Telegram update: {e}")
            
            # Check position P&L if entry price exists
            if entry_price and entry_price > 0 and entry_team:
                await self._check_position_pnl(
                    user_id, game_id, current_game, entry_price, entry_team
                )
                
        except Exception as e:
            logger.error(f"Error checking game {game_info.get('game_id')}: {e}")

    async def _check_position_pnl(self, user_id: str, game_id: str, game: Dict,
                                   entry_price: float, entry_team: str):
        """Check position P&L and alert on 20% threshold"""
        try:
            kalshi_odds = game.get('kalshi_odds', {})
            if not kalshi_odds:
                return
            
            # Get current odds for entry team
            away_team = game.get('away_team', '')
            home_team = game.get('home_team', '')
            
            if entry_team == away_team:
                current_odds = kalshi_odds.get('away_win_price', 0) * 100
            elif entry_team == home_team:
                current_odds = kalshi_odds.get('home_win_price', 0) * 100
            else:
                return
            
            if current_odds <= 0:
                return
            
            # Calculate P&L
            pnl_percent = ((current_odds - entry_price) / entry_price) * 100
            
            # Check if we've crossed 20% threshold
            # Store last P&L to detect threshold crossing
            conn = self.db.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT last_pnl_percent FROM game_watchlist
                WHERE user_id = %s AND game_id = %s AND is_active = TRUE
            """, (user_id, game_id))
            
            result = cur.fetchone()
            last_pnl = result[0] if result and result[0] else None
            
            # Update last P&L
            cur.execute("""
                UPDATE game_watchlist
                SET last_pnl_percent = %s
                WHERE user_id = %s AND game_id = %s AND is_active = TRUE
            """, (pnl_percent, user_id, game_id))
            
            conn.commit()
            cur.close()
            conn.close()
            
            # Check if we crossed 20% threshold
            if last_pnl is not None:
                last_abs = abs(last_pnl)
                current_abs = abs(pnl_percent)
                
                # Crossed 20% threshold (was below, now above)
                if last_abs < 20 and current_abs >= 20:
                    alert_type = "PROFIT" if pnl_percent > 0 else "LOSS"
                    message = (
                        f"🏈 **{alert_type} ALERT**\n\n"
                        f"**{away_team} @ {home_team}**\n"
                        f"Position: {entry_team}\n"
                        f"Entry: {entry_price:.0f}¢ → Current: {current_odds:.0f}¢\n"
                        f"P&L: {pnl_percent:+.1f}%\n\n"
                        f"⚠️ You've {'gained' if pnl_percent > 0 else 'lost'} more than 20%!\n"
                        f"Consider {'taking profits' if pnl_percent > 0 else 'cutting losses'}."
                    )
                    
                    try:
                        self.telegram_notifier.send_message(message)
                        logger.info(f"Sent P&L alert for game {game_id}")
                    except Exception as e:
                        logger.error(f"Failed to send P&L alert: {e}")
                        
        except Exception as e:
            logger.error(f"Error checking position P&L: {e}")


async def run_monitor():
    """Run the monitor service"""
    monitor = GameWatchlistMonitor(check_interval=30)
    await monitor.start()


if __name__ == "__main__":
    # Run as standalone service
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(run_monitor())

