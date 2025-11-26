"""
Sports Betting Service
Consolidates data from NFL, NCAA, and Kalshi for the betting hub
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from src.nfl_db_manager import NFLDBManager
from src.kalshi_db_manager import KalshiDBManager
from src.database.query_cache import query_cache

logger = logging.getLogger(__name__)

class SportsBettingService:
    """
    Service to aggregate sports data and betting opportunities
    """
    
    def __init__(self):
        self.nfl_db = NFLDBManager()
        self.kalshi_db = KalshiDBManager()
        
    def get_live_games(self) -> List[Dict[str, Any]]:
        """Get all currently live games across sports"""
        # For now, primarily NFL, but structure allows expansion
        try:
            # Check cache first
            cached = query_cache.get('live_games')
            if cached:
                return cached
                
            nfl_games = self.nfl_db.get_live_games()
            
            # Normalize data structure
            normalized_games = []
            for game in nfl_games:
                normalized_games.append({
                    'id': game.get('game_id'),
                    'league': 'NFL',
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'home_score': game.get('home_score'),
                    'away_score': game.get('away_score'),
                    'status': 'Live',
                    'is_live': True,
                    'game_time': f"Q{game.get('quarter')} {game.get('time_remaining')}",
                    'odds': {
                        'spread_home': game.get('spread_home'),
                        'spread_home_odds': game.get('spread_odds_home'),
                        'total': game.get('over_under'),
                        'moneyline_home': game.get('moneyline_home')
                    }
                })
                
            # Cache for 30 seconds
            query_cache.set('live_games', normalized_games, ttl_seconds=30)
            return normalized_games
            
        except Exception as e:
            logger.error(f"Error getting live games: {e}")
            return []

    def get_upcoming_games(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get upcoming games with odds"""
        try:
            cached = query_cache.get(f'upcoming_games_{limit}')
            if cached:
                return cached
                
            nfl_games = self.nfl_db.get_upcoming_games(hours_ahead=48)
            
            normalized_games = []
            for game in nfl_games[:limit]:
                normalized_games.append({
                    'id': game.get('game_id'),
                    'league': 'NFL',
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'status': 'Scheduled',
                    'is_live': False,
                    'game_time': game.get('game_time').strftime('%a %I:%M %p') if game.get('game_time') else 'TBD',
                    'odds': {
                        'spread_home': game.get('spread_home'),
                        'spread_home_odds': game.get('spread_odds_home'),
                        'total': game.get('over_under'),
                        'moneyline_home': game.get('moneyline_home')
                    }
                })
                
            query_cache.set(f'upcoming_games_{limit}', normalized_games, ttl_seconds=300)
            return normalized_games
            
        except Exception as e:
            logger.error(f"Error getting upcoming games: {e}")
            return []

    def get_best_bets(self) -> List[Dict[str, Any]]:
        """Get AI-recommended best bets"""
        # This would integrate with LLMSportsAnalyzer
        # For now, return placeholder high-value opportunities
        return [
            {
                'type': 'Spread',
                'league': 'NFL',
                'matchup': 'KC vs BUF',
                'pick': 'KC -2.5',
                'odds': -110,
                'confidence': 85,
                'reasoning': 'Strong trend analysis favors KC at home'
            },
            {
                'type': 'Total',
                'league': 'NFL',
                'matchup': 'DAL vs PHI',
                'pick': 'Over 48.5',
                'odds': -110,
                'confidence': 78,
                'reasoning': 'Both defenses struggling with injuries'
            }
        ]
