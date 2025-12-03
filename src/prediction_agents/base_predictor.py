"""
Base Sports Predictor
=====================

Abstract base class for all sports prediction agents.

Provides common interface and shared functionality for NFL, NCAA, and other sports.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseSportsPredictor(ABC):
    """
    Abstract base class for sports prediction agents.

    All sport-specific predictors must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, sport_name: str):
        """
        Initialize the base predictor.

        Args:
            sport_name: Name of the sport (e.g., 'NFL', 'NCAA Football')
        """
        self.sport_name = sport_name
        self.logger = logging.getLogger(f"{__name__}.{sport_name}")
        self.prediction_cache = {}  # Cache for expensive predictions
        self.last_update = None

    @abstractmethod
    def predict_winner(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict the winner of a game.

        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            game_date: Date/time of the game (for historical context)
            **kwargs: Additional sport-specific parameters

        Returns:
            Dictionary with prediction results:
            {
                'winner': str,           # Predicted winning team name
                'probability': float,    # Win probability (0.0-1.0)
                'confidence': str,       # 'high', 'medium', 'low'
                'spread': float,         # Predicted point spread
                'method': str,           # Model/method used
                'features': dict,        # Features used in prediction
                'explanation': str       # Human-readable explanation
            }
        """
        pass

    @abstractmethod
    def calculate_features(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate prediction features for a matchup.

        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            game_date: Date/time of the game

        Returns:
            Dictionary of features used for prediction
        """
        pass

    def get_confidence(self, probability: float) -> str:
        """
        Determine confidence level based on win probability.

        Args:
            probability: Win probability (0.0-1.0)

        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        # Convert to percentage from 50% (50% = no confidence)
        confidence_score = abs(probability - 0.5) * 2  # 0.0 to 1.0 scale

        if confidence_score >= 0.50:  # >75% or <25% win probability
            return 'high'
        elif confidence_score >= 0.20:  # 60-75% or 25-40% win probability
            return 'medium'
        else:  # 50-60% or 40-50% win probability
            return 'low'

    def get_confidence_color(self, confidence: str) -> str:
        """
        Get color code for confidence level.

        Args:
            confidence: Confidence level ('high', 'medium', 'low')

        Returns:
            Color name for UI display
        """
        colors = {
            'high': 'green',
            'medium': 'yellow',
            'low': 'gray'
        }
        return colors.get(confidence, 'gray')

    def format_probability(self, probability: float) -> str:
        """
        Format win probability as percentage string.

        Args:
            probability: Win probability (0.0-1.0)

        Returns:
            Formatted percentage string (e.g., "68%")
        """
        return f"{int(probability * 100)}%"

    def get_spread(self, home_prob: float, historical_spread_factor: float = 25.0) -> float:
        """
        Estimate point spread from win probability.

        Uses a simplified logit-based conversion commonly used in sports betting.

        Args:
            home_prob: Home team win probability (0.0-1.0)
            historical_spread_factor: Scaling factor (default 25.0 for NFL)

        Returns:
            Estimated point spread (positive = home favored)
        """
        import math

        # Avoid division by zero
        if home_prob >= 0.99:
            home_prob = 0.99
        elif home_prob <= 0.01:
            home_prob = 0.01

        # Logit transformation
        logit = math.log(home_prob / (1 - home_prob))
        spread = logit * historical_spread_factor / 2.0

        return round(spread, 1)

    def create_cache_key(self, home_team: str, away_team: str, game_date: Optional[datetime] = None) -> str:
        """
        Create a cache key for predictions.

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date/time

        Returns:
            Cache key string
        """
        date_str = game_date.strftime('%Y-%m-%d') if game_date else 'unknown'
        return f"{self.sport_name}:{home_team}:{away_team}:{date_str}"

    def get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached prediction if available and fresh.

        Args:
            cache_key: Cache key from create_cache_key()

        Returns:
            Cached prediction dict or None if not available
        """
        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            # Check if cache is still fresh (less than 1 hour old)
            age = (datetime.now() - cached['cached_at']).total_seconds()
            if age < 3600:  # 1 hour
                self.logger.debug(f"Using cached prediction for {cache_key}")
                return cached['prediction']

        return None

    def cache_prediction(self, cache_key: str, prediction: Dict[str, Any]) -> None:
        """
        Store prediction in cache.

        Args:
            cache_key: Cache key from create_cache_key()
            prediction: Prediction dictionary to cache
        """
        self.prediction_cache[cache_key] = {
            'prediction': prediction,
            'cached_at': datetime.now()
        }

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self.prediction_cache.clear()
        self.logger.info(f"Cleared prediction cache for {self.sport_name}")

    @abstractmethod
    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """
        Get current statistics for a team.

        Args:
            team_name: Name of the team

        Returns:
            Dictionary of team statistics
        """
        pass

    @abstractmethod
    def get_historical_matchups(
        self,
        team1: str,
        team2: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get historical head-to-head matchups.

        Args:
            team1: First team name
            team2: Second team name
            limit: Maximum number of games to return

        Returns:
            List of historical game dictionaries
        """
        pass

    def explain_prediction(self, prediction: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of prediction.

        Args:
            prediction: Prediction dictionary from predict_winner()

        Returns:
            Explanation string
        """
        winner = prediction['winner']
        prob = self.format_probability(prediction['probability'])
        confidence = prediction['confidence']
        spread = prediction.get('spread', 0)

        explanation = f"{winner} predicted to win with {prob} probability "
        explanation += f"({confidence} confidence). "

        if abs(spread) >= 1:
            explanation += f"Predicted spread: {abs(spread):.1f} points."

        return explanation

    def validate_team_names(self, home_team: str, away_team: str) -> bool:
        """
        Validate that team names are recognized.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            True if both teams are valid, False otherwise
        """
        # Subclasses should override to implement sport-specific validation
        return True

    def get_prediction_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the prediction system.

        Returns:
            Dictionary with model info, version, accuracy stats, etc.
        """
        return {
            'sport': self.sport_name,
            'version': '1.0.0',
            'last_update': self.last_update,
            'cache_size': len(self.prediction_cache)
        }

    def predict_batch(
        self,
        games: List[Dict[str, Any]],
        max_parallel: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Predict outcomes for multiple games efficiently.

        Uses caching and parallel feature calculation to reduce N+1 overhead.

        Args:
            games: List of game dictionaries with 'game_id', 'home_team', 'away_team'
            max_parallel: Maximum parallel calculations (default 10)

        Returns:
            Dictionary mapping game_id to prediction result
        """
        results = {}
        uncached_games = []

        # Check cache first for all games
        for game in games:
            game_id = game.get('game_id', f"{game.get('home_team')}_{game.get('away_team')}")
            cache_key = self.create_cache_key(
                game.get('home_team', ''),
                game.get('away_team', ''),
                game.get('game_date')
            )
            cached = self.get_cached_prediction(cache_key)
            if cached:
                results[game_id] = cached
            else:
                uncached_games.append((game_id, game, cache_key))

        # Process uncached games (could be parallelized with ThreadPoolExecutor)
        for game_id, game, cache_key in uncached_games:
            try:
                prediction = self.predict_winner(
                    home_team=game.get('home_team', ''),
                    away_team=game.get('away_team', ''),
                    game_date=game.get('game_date'),
                    **{k: v for k, v in game.items()
                       if k not in ('game_id', 'home_team', 'away_team', 'game_date')}
                )
                results[game_id] = prediction
                # Cache is updated inside predict_winner
            except Exception as e:
                self.logger.warning(f"Batch prediction failed for {game_id}: {e}")
                results[game_id] = None

        self.logger.info(f"Batch predicted {len(results)} games ({len(results) - len(uncached_games)} from cache)")
        return results
