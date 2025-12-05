"""
Live Prediction Adjuster
Adjusts pre-game win probabilities in real-time using Bayesian updating
based on live game state, score differential, and momentum.
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MomentumSignal(str, Enum):
    """Momentum signals detected in game flow"""
    STRONG_HOME = "strong_home"
    MODERATE_HOME = "moderate_home"
    NEUTRAL = "neutral"
    MODERATE_AWAY = "moderate_away"
    STRONG_AWAY = "strong_away"


@dataclass
class GameState:
    """Current state of a live game"""
    home_score: int
    away_score: int
    period: int  # Quarter/Half
    time_remaining_seconds: int
    total_periods: int = 4  # 4 for NFL/NBA, 2 for soccer
    possession: Optional[str] = None  # 'home' or 'away'
    is_red_zone: bool = False
    home_timeouts: int = 3
    away_timeouts: int = 3
    sport: str = "NFL"

    @property
    def score_differential(self) -> int:
        """Home team's score margin (positive = home leading)"""
        return self.home_score - self.away_score

    @property
    def total_points(self) -> int:
        """Total points scored"""
        return self.home_score + self.away_score

    @property
    def game_progress(self) -> float:
        """
        Game completion percentage (0.0 to 1.0)
        Based on period and time remaining
        """
        if self.sport in ["NFL", "NBA", "NCAAF", "NCAAB"]:
            total_seconds = self.total_periods * 900  # 15 min quarters (approx)
            elapsed_periods = (self.period - 1) * 900
            elapsed_in_period = 900 - self.time_remaining_seconds
            elapsed = elapsed_periods + elapsed_in_period
            return min(1.0, max(0.0, elapsed / total_seconds))
        else:
            # Default to 50% if sport unknown
            return 0.5


class LivePredictionAdjuster:
    """
    Adjusts pre-game predictions in real-time using Bayesian updating.

    The core idea: P(Win | Score) = P(Score | Win) * P(Win) / P(Score)

    We approximate this using empirical win probability curves based on
    score differential and time remaining.
    """

    # Empirical win probability parameters by sport
    # These are based on historical data analysis
    SPORT_PARAMS = {
        "NFL": {
            "points_per_possession": 2.8,  # NFL average ~2.8 points per drive
            "scoring_volatility": 7.0,     # Standard deviation of final margin
            "lead_safety_threshold": 17,   # Points needed for "safe" lead
            "comeback_rate": 0.05,         # Base rate of comebacks per minute
        },
        "NBA": {
            "points_per_possession": 1.1,
            "scoring_volatility": 12.0,
            "lead_safety_threshold": 15,
            "comeback_rate": 0.08,
        },
        "NCAAF": {
            "points_per_possession": 2.5,
            "scoring_volatility": 8.0,
            "lead_safety_threshold": 21,
            "comeback_rate": 0.04,
        },
        "NCAAB": {
            "points_per_possession": 1.0,
            "scoring_volatility": 10.0,
            "lead_safety_threshold": 12,
            "comeback_rate": 0.10,
        }
    }

    def __init__(self) -> None:
        self._momentum_history: Dict[str, List[Tuple[datetime, int]]] = {}

    def adjust_prediction(
        self,
        pregame_home_prob: float,
        game_state: GameState,
        momentum_window_minutes: int = 5
    ) -> Dict:
        """
        Adjust pre-game probability based on live game state.

        Args:
            pregame_home_prob: Pre-game win probability for home team (0-1)
            game_state: Current game state
            momentum_window_minutes: Window for momentum calculation

        Returns:
            Dictionary with adjusted probabilities and analysis
        """
        sport = game_state.sport.upper()
        params = self.SPORT_PARAMS.get(sport, self.SPORT_PARAMS["NFL"])

        # Calculate base adjustment from score differential
        score_adjustment = self._calculate_score_adjustment(
            game_state.score_differential,
            game_state.game_progress,
            params
        )

        # Calculate momentum adjustment
        momentum = self._detect_momentum(game_state)
        momentum_adjustment = self._calculate_momentum_adjustment(momentum)

        # Bayesian update: combine prior with likelihood
        # Prior: pregame probability
        # Likelihood: score-based probability adjustment

        # Calculate the likelihood ratio based on score
        score_likelihood = self._score_to_likelihood(
            game_state.score_differential,
            game_state.game_progress,
            params
        )

        # Apply Bayesian update
        adjusted_prob = self._bayesian_update(
            prior=pregame_home_prob,
            likelihood_ratio=score_likelihood,
            momentum_factor=momentum_adjustment
        )

        # Ensure probabilities are valid
        adjusted_home_prob = max(0.01, min(0.99, adjusted_prob))
        adjusted_away_prob = 1.0 - adjusted_home_prob

        # Calculate confidence in the adjustment
        adjustment_confidence = self._calculate_confidence(
            game_state.game_progress,
            abs(game_state.score_differential),
            params
        )

        # Generate explanation
        explanation = self._generate_explanation(
            pregame_home_prob,
            adjusted_home_prob,
            game_state,
            momentum
        )

        return {
            "pregame_home_prob": pregame_home_prob,
            "pregame_away_prob": 1.0 - pregame_home_prob,
            "live_home_prob": round(adjusted_home_prob, 4),
            "live_away_prob": round(adjusted_away_prob, 4),
            "probability_change": round(adjusted_home_prob - pregame_home_prob, 4),
            "score_differential": game_state.score_differential,
            "game_progress": round(game_state.game_progress, 3),
            "momentum": momentum.value,
            "momentum_adjustment": round(momentum_adjustment, 4),
            "adjustment_confidence": round(adjustment_confidence, 3),
            "explanation": explanation,
            "factors": {
                "score_impact": round(score_adjustment, 4),
                "time_impact": round(1.0 - (1.0 - game_state.game_progress) ** 2, 4),
                "momentum_impact": round(momentum_adjustment, 4)
            }
        }

    def _calculate_score_adjustment(
        self,
        score_diff: int,
        game_progress: float,
        params: Dict
    ) -> float:
        """
        Calculate probability adjustment based on score differential.

        Uses a sigmoid function scaled by game progress.
        Early in game: scores matter less
        Late in game: scores matter more
        """
        # Time-weighted score importance
        # Early game (0-25%): Low weight
        # Mid game (25-75%): Medium weight
        # Late game (75-100%): High weight
        time_weight = game_progress ** 1.5

        # Normalize score by sport's volatility
        volatility = params["scoring_volatility"]
        normalized_diff = score_diff / volatility

        # Sigmoid transformation
        # Maps score diff to probability adjustment
        adjustment = 2 / (1 + math.exp(-normalized_diff * time_weight)) - 1

        return adjustment

    def _score_to_likelihood(
        self,
        score_diff: int,
        game_progress: float,
        params: Dict
    ) -> float:
        """
        Convert score differential to likelihood ratio for Bayesian update.

        Likelihood ratio: P(this score | home wins) / P(this score | away wins)
        """
        # Time remaining affects how much the score matters
        time_remaining = 1.0 - game_progress

        # Expected points remaining (rough estimate)
        volatility = params["scoring_volatility"]

        if time_remaining < 0.05:
            # Game nearly over - score is nearly deterministic
            if score_diff > 0:
                return 50.0  # Strong evidence for home
            elif score_diff < 0:
                return 0.02  # Strong evidence for away
            else:
                return 1.0  # Tie - neutral

        # Calculate standard deviation of remaining score change
        remaining_sd = volatility * math.sqrt(time_remaining)

        # Probability that home team catches up/maintains lead
        # Using normal distribution approximation
        z_score = score_diff / remaining_sd

        # Convert to likelihood ratio
        # Positive z = home more likely to win
        likelihood_ratio = math.exp(z_score * 0.5)

        # Clamp to reasonable bounds
        return max(0.02, min(50.0, likelihood_ratio))

    def _bayesian_update(
        self,
        prior: float,
        likelihood_ratio: float,
        momentum_factor: float = 0.0
    ) -> float:
        """
        Perform Bayesian probability update.

        P(Win | Evidence) = P(Evidence | Win) * P(Win) / P(Evidence)

        Using odds form: posterior_odds = likelihood_ratio * prior_odds
        """
        # Convert prior to odds
        prior_odds = prior / (1 - prior) if prior < 1 else float('inf')

        # Apply likelihood ratio
        posterior_odds = prior_odds * likelihood_ratio

        # Apply momentum factor (additive adjustment to odds)
        posterior_odds *= (1 + momentum_factor)

        # Convert back to probability
        posterior = posterior_odds / (1 + posterior_odds)

        return posterior

    def _detect_momentum(self, game_state: GameState) -> MomentumSignal:
        """
        Detect momentum based on recent scoring and game situation.

        Returns momentum signal indicating which team has momentum.
        """
        score_diff = game_state.score_differential

        # Red zone possession is strong momentum indicator
        if game_state.is_red_zone:
            if game_state.possession == "home":
                return MomentumSignal.STRONG_HOME
            elif game_state.possession == "away":
                return MomentumSignal.STRONG_AWAY

        # Large lead late in game
        if game_state.game_progress > 0.75:
            if score_diff > 10:
                return MomentumSignal.STRONG_HOME
            elif score_diff < -10:
                return MomentumSignal.STRONG_AWAY

        # Moderate lead
        if score_diff > 7:
            return MomentumSignal.MODERATE_HOME
        elif score_diff < -7:
            return MomentumSignal.MODERATE_AWAY

        return MomentumSignal.NEUTRAL

    def _calculate_momentum_adjustment(self, momentum: MomentumSignal) -> float:
        """Convert momentum signal to probability adjustment factor"""
        momentum_values = {
            MomentumSignal.STRONG_HOME: 0.15,
            MomentumSignal.MODERATE_HOME: 0.05,
            MomentumSignal.NEUTRAL: 0.0,
            MomentumSignal.MODERATE_AWAY: -0.05,
            MomentumSignal.STRONG_AWAY: -0.15,
        }
        return momentum_values.get(momentum, 0.0)

    def _calculate_confidence(
        self,
        game_progress: float,
        score_margin: int,
        params: Dict
    ) -> float:
        """
        Calculate confidence in the live probability adjustment.

        Higher confidence when:
        - Later in game
        - Larger score margin
        - Less volatility remaining
        """
        # Time-based confidence (more confident later)
        time_confidence = game_progress ** 0.5

        # Score-based confidence (more confident with larger margins)
        safety_threshold = params["lead_safety_threshold"]
        score_confidence = min(1.0, score_margin / safety_threshold)

        # Combined confidence
        confidence = 0.4 * time_confidence + 0.6 * score_confidence

        return min(1.0, max(0.0, confidence))

    def _generate_explanation(
        self,
        pregame_prob: float,
        live_prob: float,
        game_state: GameState,
        momentum: MomentumSignal
    ) -> str:
        """Generate human-readable explanation of the adjustment"""
        change = live_prob - pregame_prob
        change_pct = abs(change) * 100

        explanations = []

        # Score impact
        if game_state.score_differential > 0:
            explanations.append(
                f"Home team leads by {game_state.score_differential}"
            )
        elif game_state.score_differential < 0:
            explanations.append(
                f"Away team leads by {abs(game_state.score_differential)}"
            )
        else:
            explanations.append("Game is tied")

        # Game progress
        progress_pct = game_state.game_progress * 100
        if progress_pct > 75:
            explanations.append(f"Late in Q{game_state.period} ({progress_pct:.0f}% complete)")
        elif progress_pct > 50:
            explanations.append(f"Second half ({progress_pct:.0f}% complete)")
        else:
            explanations.append(f"Early in game ({progress_pct:.0f}% complete)")

        # Momentum
        if momentum in [MomentumSignal.STRONG_HOME, MomentumSignal.STRONG_AWAY]:
            team = "Home" if "home" in momentum.value else "Away"
            explanations.append(f"{team} has strong momentum")
        elif momentum in [MomentumSignal.MODERATE_HOME, MomentumSignal.MODERATE_AWAY]:
            team = "Home" if "home" in momentum.value else "Away"
            explanations.append(f"{team} showing momentum")

        # Probability change summary
        if change > 0.1:
            summary = f"Win probability increased {change_pct:.1f}% for home team"
        elif change < -0.1:
            summary = f"Win probability decreased {change_pct:.1f}% for home team"
        elif abs(change) > 0.02:
            direction = "up" if change > 0 else "down"
            summary = f"Win probability slightly {direction} ({change_pct:.1f}%)"
        else:
            summary = "Win probability largely unchanged"

        return f"{summary}. {'. '.join(explanations)}."

    def create_live_snapshot(
        self,
        game_id: str,
        game_state: GameState,
        pregame_home_prob: float
    ) -> Dict:
        """
        Create a snapshot of live predictions for database storage.

        Returns data suitable for live_prediction_snapshots table.
        """
        adjustment = self.adjust_prediction(pregame_home_prob, game_state)

        return {
            "game_id": game_id,
            "sport": game_state.sport,
            "pregame_home_prob": pregame_home_prob,
            "pregame_away_prob": 1.0 - pregame_home_prob,
            "live_home_prob": adjustment["live_home_prob"],
            "live_away_prob": adjustment["live_away_prob"],
            "home_score": game_state.home_score,
            "away_score": game_state.away_score,
            "quarter_period": game_state.period,
            "time_remaining": f"{game_state.time_remaining_seconds // 60}:{game_state.time_remaining_seconds % 60:02d}",
            "possession": game_state.possession,
            "momentum_score": self._momentum_to_score(adjustment["momentum"]),
            "scoring_run": None,  # Could be enhanced with scoring run detection
            "snapshot_at": datetime.now().isoformat()
        }

    def _momentum_to_score(self, momentum_str: str) -> float:
        """Convert momentum string to numeric score (-1 to 1)"""
        momentum_scores = {
            "strong_home": 1.0,
            "moderate_home": 0.5,
            "neutral": 0.0,
            "moderate_away": -0.5,
            "strong_away": -1.0
        }
        return momentum_scores.get(momentum_str, 0.0)


# Singleton instance
_adjuster: Optional[LivePredictionAdjuster] = None


def get_live_adjuster() -> LivePredictionAdjuster:
    """Get the singleton live adjuster instance"""
    global _adjuster
    if _adjuster is None:
        _adjuster = LivePredictionAdjuster()
    return _adjuster


# Convenience function for quick adjustments
def adjust_live_probability(
    pregame_prob: float,
    home_score: int,
    away_score: int,
    period: int,
    time_remaining_seconds: int,
    sport: str = "NFL"
) -> Dict:
    """
    Quick function to adjust probability based on live game state.

    Args:
        pregame_prob: Pre-game win probability for home team
        home_score: Current home team score
        away_score: Current away team score
        period: Current period/quarter
        time_remaining_seconds: Seconds left in current period
        sport: Sport type (NFL, NBA, NCAAF, NCAAB)

    Returns:
        Dictionary with adjusted probabilities
    """
    adjuster = get_live_adjuster()

    game_state = GameState(
        home_score=home_score,
        away_score=away_score,
        period=period,
        time_remaining_seconds=time_remaining_seconds,
        sport=sport
    )

    return adjuster.adjust_prediction(pregame_prob, game_state)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    adjuster = LivePredictionAdjuster()

    # Test 1: Early game, home team slight favorite
    print("=" * 60)
    print("TEST 1: Early game, home leading by 7")
    print("=" * 60)

    state = GameState(
        home_score=14,
        away_score=7,
        period=1,
        time_remaining_seconds=420,  # 7 minutes left in Q1
        sport="NFL"
    )

    result = adjuster.adjust_prediction(0.55, state)
    print(f"Pre-game home prob: {result['pregame_home_prob']:.1%}")
    print(f"Live home prob: {result['live_home_prob']:.1%}")
    print(f"Change: {result['probability_change']:+.1%}")
    print(f"Momentum: {result['momentum']}")
    print(f"Explanation: {result['explanation']}")

    # Test 2: Late game, away team comeback
    print("\n" + "=" * 60)
    print("TEST 2: Late game, close score")
    print("=" * 60)

    state = GameState(
        home_score=21,
        away_score=24,
        period=4,
        time_remaining_seconds=180,  # 3 minutes left in Q4
        sport="NFL"
    )

    result = adjuster.adjust_prediction(0.55, state)
    print(f"Pre-game home prob: {result['pregame_home_prob']:.1%}")
    print(f"Live home prob: {result['live_home_prob']:.1%}")
    print(f"Change: {result['probability_change']:+.1%}")
    print(f"Momentum: {result['momentum']}")
    print(f"Confidence: {result['adjustment_confidence']:.1%}")
    print(f"Explanation: {result['explanation']}")

    # Test 3: Blowout scenario
    print("\n" + "=" * 60)
    print("TEST 3: Blowout - home dominating")
    print("=" * 60)

    state = GameState(
        home_score=35,
        away_score=7,
        period=3,
        time_remaining_seconds=600,
        sport="NFL"
    )

    result = adjuster.adjust_prediction(0.50, state)
    print(f"Pre-game home prob: {result['pregame_home_prob']:.1%}")
    print(f"Live home prob: {result['live_home_prob']:.1%}")
    print(f"Change: {result['probability_change']:+.1%}")
    print(f"Momentum: {result['momentum']}")
    print(f"Confidence: {result['adjustment_confidence']:.1%}")
    print(f"Explanation: {result['explanation']}")

    # Test 4: NBA game
    print("\n" + "=" * 60)
    print("TEST 4: NBA close game")
    print("=" * 60)

    state = GameState(
        home_score=95,
        away_score=92,
        period=4,
        time_remaining_seconds=300,
        sport="NBA"
    )

    result = adjuster.adjust_prediction(0.60, state)
    print(f"Pre-game home prob: {result['pregame_home_prob']:.1%}")
    print(f"Live home prob: {result['live_home_prob']:.1%}")
    print(f"Change: {result['probability_change']:+.1%}")
    print(f"Momentum: {result['momentum']}")
    print(f"Explanation: {result['explanation']}")
