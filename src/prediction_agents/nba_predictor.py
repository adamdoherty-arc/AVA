"""
NBA Predictor Agent
Advanced NBA game predictions using Elo ratings and statistical models
"""

import json
import logging
import math
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from datetime import datetime, timedelta
import os
from .llm_explanation_enhancer import get_llm_enhancer

logger = logging.getLogger(__name__)


class NBAPredictor:
    """NBA game prediction engine using Elo ratings and advanced stats"""
    
    # Elo rating constants
    ELO_BASE = 1500
    ELO_K_FACTOR = 20
    HOME_COURT_ADVANTAGE = 100  # ~3 points in NBA
    
    # File paths for persistence
    ELO_FILE = 'data/nba_elo_ratings.json'
    
    def __init__(self) -> None:
        """Initialize NBA predictor"""
        self.elo_ratings = {}
        self.logger = logging.getLogger(__name__)
        self._load_elo_ratings()
    
    def _load_elo_ratings(self) -> None:
        """Load Elo ratings from file or initialize defaults"""
        try:
            if os.path.exists(self.ELO_FILE):
                with open(self.ELO_FILE, 'r') as f:
                    self.elo_ratings = json.load(f)
                self.logger.info(f"Loaded Elo ratings for {len(self.elo_ratings)} NBA teams")
            else:
                # Initialize with base ratings
                from src.nba_team_database import NBA_TEAMS
                for abbr in NBA_TEAMS.keys():
                    self.elo_ratings[abbr] = self.ELO_BASE
                self._save_elo_ratings()
                self.logger.info("Initialized default Elo ratings")
        except Exception as e:
            self.logger.error(f"Error loading Elo ratings: {e}")
            self.elo_ratings = {}
    
    def _save_elo_ratings(self) -> None:
        """Save Elo ratings to file"""
        try:
            os.makedirs(os.path.dirname(self.ELO_FILE), exist_ok=True)
            with open(self.ELO_FILE, 'w') as f:
                json.dump(self.elo_ratings, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving Elo ratings: {e}")
    
    def _calculate_elo_win_prob(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate win probability using Elo ratings
        
        Args:
            rating_a: Team A's Elo rating
            rating_b: Team B's Elo rating
        
        Returns:
            Probability of Team A winning (0-1)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        home_record: str = "",
        away_record: str = "",
        rest_days_home: int = 1,
        rest_days_away: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict NBA game outcome
        
        Args:
            home_team: Home team name or abbreviation
            away_team: Away team name or abbreviation
            home_record: Home team record (e.g., "24-18")
            away_record: Away team record
            rest_days_home: Days of rest for home team
            rest_days_away: Days of rest for away team
        
        Returns:
            Dictionary with prediction details
        """
        try:
            # Get Elo ratings
            home_elo = self.elo_ratings.get(home_team, self.ELO_BASE)
            away_elo = self.elo_ratings.get(away_team, self.ELO_BASE)
            
            # Apply home court advantage
            home_elo_adjusted = home_elo + self.HOME_COURT_ADVANTAGE
            
            # Rest days adjustment (back-to-back games hurt performance)
            rest_adjustment = 0
            if rest_days_home == 0:  # Back-to-back for home
                rest_adjustment -= 20
            if rest_days_away == 0:  # Back-to-back for away
                rest_adjustment += 20
            
            home_elo_adjusted += rest_adjustment
            
            # Calculate win probability
            home_win_prob = self._calculate_elo_win_prob(home_elo_adjusted, away_elo)
            
            # Determine winner and confidence
            if home_win_prob > 0.5:
                winner = home_team
                win_prob = home_win_prob
            else:
                winner = away_team
                win_prob = 1 - home_win_prob
            
            # Confidence levels
            if win_prob >= 0.70:
                confidence = 'high'
            elif win_prob >= 0.60:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            # Calculate predicted spread (1 Elo point ≈ 0.03 points)
            elo_diff = home_elo_adjusted - away_elo
            predicted_spread = elo_diff * 0.03
            
            # Generate explanation
            explanation = self._generate_explanation(
                winner=winner,
                probability=win_prob,
                home_team=home_team,
                away_team=away_team,
                home_elo=home_elo,
                away_elo=away_elo,
                rest_days_home=rest_days_home,
                rest_days_away=rest_days_away
            )
            
            return {
                'winner': winner,
                'probability': win_prob,
                'confidence': confidence,
                'spread': predicted_spread,
                'explanation': explanation,
                'features': {
                    'home_elo': home_elo,
                    'away_elo': away_elo,
                    'home_court_advantage': self.HOME_COURT_ADVANTAGE,
                    'rest_days_home': rest_days_home,
                    'rest_days_away': rest_days_away,
                },
                'adjustments': {
                    'home_court': self.HOME_COURT_ADVANTAGE,
                    'rest_impact': rest_adjustment
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error predicting NBA game {away_team} @ {home_team}: {e}")
            return None
    
    def _generate_explanation(
        self,
        winner: str,
        probability: float,
        home_team: str,
        away_team: str,
        home_elo: float,
        away_elo: float,
        rest_days_home: int,
        rest_days_away: int
    ) -> str:
        """Generate human-readable explanation of prediction"""
        
        explanation_parts = []
        
        # Winner and probability
        explanation_parts.append(
            f"**Prediction: {winner} ({probability*100:.1f}% win probability)**\n"
        )
        
        # Elo ratings
        explanation_parts.append(f"**Elo Ratings:**")
        explanation_parts.append(f"- {home_team}: {home_elo:.0f}")
        explanation_parts.append(f"- {away_team}: {away_elo:.0f}")
        
        # Home court advantage
        if winner == home_team:
            explanation_parts.append(f"\n**Home Court Advantage:** +{self.HOME_COURT_ADVANTAGE} Elo (~3 points)")
        
        # Rest days
        if rest_days_home == 0 or rest_days_away == 0:
            explanation_parts.append(f"\n**Rest Analysis:**")
            if rest_days_home == 0:
                explanation_parts.append(f"- {home_team} playing back-to-back (fatigue factor)")
            if rest_days_away == 0:
                explanation_parts.append(f"- {away_team} playing back-to-back (fatigue factor)")
        
        # Key factors
        elo_diff = abs(home_elo - away_elo)
        if elo_diff > 100:
            explanation_parts.append(f"\n**Key Factor:** Significant Elo rating gap ({elo_diff:.0f} points)")
        
        # Generate base template explanation
        template_explanation = "\n".join(explanation_parts)
        
        # Enhance with LLM if available
        try:
            enhancer = get_llm_enhancer()
            loser = away_team if winner == home_team else home_team
            
            # Prepare context for LLM
            context = {
                'home_team': home_team,
                'away_team': away_team,
                'home_elo': home_elo,
                'away_elo': away_elo,
                'rest_days_home': rest_days_home,
                'rest_days_away': rest_days_away,
                'home_court_advantage': self.HOME_COURT_ADVANTAGE
            }
            
            return enhancer.enhance_explanation(
                sport="NBA",
                winner=winner,
                loser=loser,
                probability=probability,
                features={
                    'home_elo': home_elo,
                    'away_elo': away_elo,
                    'elo_diff': abs(home_elo - away_elo)
                },
                context=context,
                template_explanation=template_explanation
            )
        except Exception as e:
            self.logger.warning(f"Failed to enhance explanation with LLM: {e}")
            return template_explanation
    
    def update_elo_ratings(
        self,
        winner: str,
        loser: str,
        winner_score: int,
        loser_score: int,
        home_team: str
    ):
        """
        Update Elo ratings after a game result
        
        Args:
            winner: Winning team
            loser: Losing team
            winner_score: Winner's score
            loser_score: Loser's score
            home_team: Which team was home
        """
        # Get current ratings
        winner_elo = self.elo_ratings.get(winner, self.ELO_BASE)
        loser_elo = self.elo_ratings.get(loser, self.ELO_BASE)
        
        # Calculate expected win probability
        if winner == home_team:
            winner_expected = self._calculate_elo_win_prob(
                winner_elo + self.HOME_COURT_ADVANTAGE,
                loser_elo
            )
        else:
            winner_expected = self._calculate_elo_win_prob(
                winner_elo,
                loser_elo + self.HOME_COURT_ADVANTAGE
            )
        
        # Calculate margin of victory multiplier
        point_diff = abs(winner_score - loser_score)
        mov_multiplier = math.log(point_diff + 1) * (2.2 / ((winner_elo - loser_elo) * 0.001 + 2.2))
        
        # Update ratings
        rating_change = self.ELO_K_FACTOR * mov_multiplier * (1 - winner_expected)
        
        self.elo_ratings[winner] = winner_elo + rating_change
        self.elo_ratings[loser] = loser_elo - rating_change
        
        self.logger.info(f"Updated Elo: {winner} +{rating_change:.1f}, {loser} -{rating_change:.1f}")
        
        # Save updated ratings
        self._save_elo_ratings()
    
    def get_team_rating(self, team: str) -> float:
        """Get current Elo rating for a team"""
        return self.elo_ratings.get(team, self.ELO_BASE)
    
    def get_top_teams(self, n: int = 10) -> list:
        """
        Get top N teams by Elo rating
        
        Args:
            n: Number of teams to return
        
        Returns:
            List of (team, rating) tuples sorted by rating
        """
        sorted_teams = sorted(
            self.elo_ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_teams[:n]


# Singleton instance
_nba_predictor = None


def get_nba_predictor() -> NBAPredictor:
    """Get or create NBA predictor singleton"""
    global _nba_predictor
    if _nba_predictor is None:
        _nba_predictor = NBAPredictor()
    return _nba_predictor


# Testing
if __name__ == "__main__":
    print("Testing NBA Predictor")
    print("=" * 60)
    
    predictor = NBAPredictor()
    
    # Test 1: Basic prediction
    print("\nTest 1: Predict Lakers vs Celtics")
    prediction = predictor.predict_game(
        home_team='LAL',
        away_team='BOS',
        home_record='24-18',
        away_record='32-10',
        rest_days_home=2,
        rest_days_away=1
    )
    
    if prediction:
        print(f"Winner: {prediction['winner']}")
        print(f"Probability: {prediction['probability']*100:.1f}%")
        print(f"Confidence: {prediction['confidence']}")
        print(f"Spread: {abs(prediction['spread']):.1f} points")
        print(f"\nExplanation:\n{prediction['explanation']}")
    
    # Test 2: Top teams
    print("\n\nTest 2: Top 10 NBA Teams by Elo")
    top_teams = predictor.get_top_teams(10)
    for i, (team, rating) in enumerate(top_teams, 1):
        print(f"{i}. {team}: {rating:.0f}")
    
    # Test 3: Back-to-back effect
    print("\n\nTest 3: Back-to-back Game Effect")
    normal = predictor.predict_game('GSW', 'LAC', rest_days_home=2, rest_days_away=2)
    b2b = predictor.predict_game('GSW', 'LAC', rest_days_home=0, rest_days_away=2)
    
    if normal and b2b:
        print(f"Normal rest: GSW {normal['probability']*100:.1f}% to win")
        print(f"Back-to-back: GSW {b2b['probability']*100:.1f}% to win")
        print(f"Impact: {(normal['probability'] - b2b['probability'])*100:.1f}% decrease")
    
    print("\n" + "=" * 60)
    print("NBA Predictor test complete!")

