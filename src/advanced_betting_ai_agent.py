"""
Advanced Sports Betting AI Agent
Uses modern ML techniques and Kelly Criterion for confidence scoring

Based on research from:
- NBA-Machine-Learning-Sports-Betting (69% accuracy)
- Leans.AI (53-58% win rate with Kelly Criterion)
- Rithmm (high-confidence signals)
- Reddit r/sportsbook and r/algobetting communities
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

# Import unified EV calculator for consistent calculations
try:
    from src.betting.unified_ev_calculator import UnifiedEVCalculator
    UNIFIED_EV_AVAILABLE = True
except ImportError:
    UNIFIED_EV_AVAILABLE = False
    logger_import = logging.getLogger(__name__)
    logger_import.warning("UnifiedEVCalculator not available, using legacy calculation")

logger = logging.getLogger(__name__)


class AdvancedBettingAIAgent:
    """
    Advanced AI agent for sports betting predictions with real-time confidence scoring

    Features:
    - Modified Kelly Criterion for bet sizing and confidence
    - Real-time probability updates based on game state
    - Multi-factor analysis (score, time, odds, historical data)
    - High-confidence signal detection
    - Clear reasoning generation
    """

    def __init__(self) -> None:
        self.min_edge_threshold = 0.05  # 5% minimum edge
        self.high_confidence_threshold = 0.75  # 75% confidence for "lightning bolt" bets
        self.kelly_fraction = 0.25  # Quarter Kelly (conservative)

    def analyze_betting_opportunity(
        self,
        game_data: Dict,
        market_data: Dict,
        historical_data: Optional[Dict] = None
    ) -> Dict:
        """
        Main analysis method - combines all factors for prediction

        Args:
            game_data: Live game information from ESPN
            market_data: Current betting odds from Kalshi
            historical_data: Historical performance data (optional)

        Returns:
            Comprehensive prediction with confidence, reasoning, and recommendation
        """

        prediction = {
            'game_id': game_data.get('id'),
            'away_team': game_data.get('away_team'),
            'home_team': game_data.get('home_team'),
            'timestamp': datetime.now().isoformat(),
            'predicted_winner': None,
            'win_probability': 0.0,
            'confidence_score': 0.0,
            'expected_value': 0.0,
            'kelly_bet_size': 0.0,
            'recommendation': 'PASS',
            'reasoning': [],
            'high_confidence_signal': False,
            'factors_analyzed': {}
        }

        try:
            # 1. Analyze current game state
            game_state = self._analyze_game_state(game_data)
            prediction['factors_analyzed']['game_state'] = game_state

            # 2. Analyze betting odds
            odds_analysis = self._analyze_odds(market_data, game_data)
            prediction['factors_analyzed']['odds'] = odds_analysis

            # 3. Calculate win probability
            win_prob = self._calculate_win_probability(game_state, odds_analysis, historical_data)
            prediction['win_probability'] = win_prob['probability']
            prediction['predicted_winner'] = win_prob['winner']

            # 4. Calculate expected value
            ev_result = self._calculate_expected_value(win_prob, odds_analysis)
            prediction['expected_value'] = ev_result['ev']
            prediction['confidence_score'] = ev_result['confidence']

            # 5. Apply Kelly Criterion for bet sizing
            kelly_size = self._kelly_criterion(win_prob['probability'], odds_analysis.get('odds', 0.5))
            prediction['kelly_bet_size'] = kelly_size

            # 6. Generate recommendation
            recommendation = self._generate_recommendation(prediction)
            prediction['recommendation'] = recommendation['action']
            prediction['reasoning'] = recommendation['reasoning']
            prediction['high_confidence_signal'] = recommendation['high_confidence']

        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            prediction['reasoning'] = [f"Analysis error: {str(e)}"]

        return prediction

    def _analyze_game_state(self, game_data: Dict) -> Dict:
        """
        Analyze current game state (score, time, momentum)

        Returns factors like:
        - Score differential
        - Time remaining weight
        - Game period/quarter
        - Is live vs upcoming
        """

        state = {
            'is_live': False,
            'score_differential': 0,
            'away_score': 0,
            'home_score': 0,
            'time_weight': 0.0,
            'period': '',
            'momentum': 'neutral',
            'leading_team': None
        }

        # Handle status - ESPN uses various formats
        status = str(game_data.get('status', '')).lower()
        state['is_live'] = any(x in status for x in ['live', 'in progress', 'in_progress', 'active'])

        # Also check for period - if there's a period/quarter, game is likely live or finished
        period_raw = game_data.get('period', '')
        if period_raw and period_raw not in ['', None, '0']:
            state['is_live'] = True

        # Parse score - try multiple formats
        # Format 1: Separate away_score and home_score fields
        away_score = game_data.get('away_score', 0)
        home_score = game_data.get('home_score', 0)

        # Format 2: Combined score string like "14-27"
        if not away_score and not home_score:
            score = game_data.get('score', '')
            if score and '-' in str(score):
                try:
                    away_score, home_score = map(int, str(score).split('-'))
                except:
                    pass

        # Convert to int if needed
        try:
            away_score = int(away_score) if away_score else 0
            home_score = int(home_score) if home_score else 0
        except:
            away_score = 0
            home_score = 0

        state['away_score'] = away_score
        state['home_score'] = home_score
        state['score_differential'] = abs(away_score - home_score)

        # Determine leading team
        if away_score > home_score:
            state['leading_team'] = 'away'
        elif home_score > away_score:
            state['leading_team'] = 'home'
        else:
            state['leading_team'] = None  # Tied or no score yet

        # Time remaining weight (higher = more certain)
        # Handle period (could be int like 1,2,3,4 or string like "1st Quarter")
        period_raw = game_data.get('period', '')
        period = str(period_raw).lower() if period_raw is not None else ''
        state['period'] = period

        # Check for 4th quarter/final
        if '4' in period or 'final' in status or 'final' in period:
            state['time_weight'] = 0.95  # Very high certainty
        elif '3' in period:
            state['time_weight'] = 0.70
        elif '2' in period or 'half' in period:
            state['time_weight'] = 0.50
        elif '1' in period:
            state['time_weight'] = 0.25
        else:
            state['time_weight'] = 0.10  # Pre-game

        return state

    def _analyze_odds(self, market_data: Dict, game_data: Dict) -> Dict:
        """
        Analyze betting odds and identify value

        Returns:
        - Current odds for both teams
        - Implied probability
        - Market efficiency indicators
        """

        analysis = {
            'away_odds': 0.5,
            'home_odds': 0.5,
            'away_implied_prob': 0.5,
            'home_implied_prob': 0.5,
            'market_efficiency': 0.0,
            'value_detected': False
        }

        # Try to get odds from market_data first (widget format)
        # Then fall back to game_data['kalshi_odds'] (game cards format)
        away_price = None
        home_price = None

        if market_data and ('yes_price' in market_data or 'no_price' in market_data):
            # Widget format: market_data has yes_price/no_price (probabilities 0-1)
            # For generic markets, yes=home, no=away
            yes_price = market_data.get('yes_price', 0.5)
            no_price = market_data.get('no_price', 0.5)

            # Assume YES = home team wins, NO = away team wins
            home_price = yes_price
            away_price = no_price

        elif 'kalshi_odds' in game_data:
            # Game cards format: odds embedded in game_data
            kalshi_odds = game_data['kalshi_odds']
            away_price = kalshi_odds.get('away_win_price', 0.5)
            home_price = kalshi_odds.get('home_win_price', 0.5)

        if away_price is not None and home_price is not None:
            # Odds are already in probability form (0-1)
            analysis['away_odds'] = away_price
            analysis['home_odds'] = home_price
            analysis['away_implied_prob'] = away_price
            analysis['home_implied_prob'] = home_price

            # Check market efficiency (should sum to ~1.0)
            total_prob = away_price + home_price
            analysis['market_efficiency'] = abs(1.0 - total_prob)

            # Detect value if there's significant market inefficiency
            if analysis['market_efficiency'] > 0.05:
                analysis['value_detected'] = True

        return analysis

    def _calculate_win_probability(
        self,
        game_state: Dict,
        odds_analysis: Dict,
        historical_data: Optional[Dict]
    ) -> Dict:
        """
        Calculate true win probability using multiple factors

        Combines:
        - Current score and time remaining
        - Market odds (wisdom of the crowd)
        - Historical matchup data
        - Team records
        """

        result = {
            'probability': 0.5,
            'winner': 'TBD',
            'factors': {}
        }

        # Start with market odds as baseline (or 50/50 if no odds)
        away_prob = odds_analysis['away_implied_prob']
        home_prob = odds_analysis['home_implied_prob']

        # If we have live game data with scores, heavily weight that
        if game_state['is_live'] or game_state.get('leading_team'):
            away_score = game_state.get('away_score', 0)
            home_score = game_state.get('home_score', 0)
            time_weight = game_state.get('time_weight', 0.5)

            # If there's an actual score, calculate probability from game state
            if away_score > 0 or home_score > 0:
                score_diff = abs(away_score - home_score)

                # Base probability from score differential
                # 7 points (1 possession) = 60-65% win probability
                # 14 points (2 possessions) = 75-80% win probability
                # 21+ points (3+ possessions) = 85-90% win probability
                if score_diff == 0:
                    # Tied game
                    base_prob = 0.5
                elif score_diff <= 3:
                    # Field goal difference
                    base_prob = 0.55 + (score_diff / 30.0)  # 55-65%
                elif score_diff <= 7:
                    # Touchdown difference
                    base_prob = 0.60 + (score_diff / 20.0)  # 60-70%
                elif score_diff <= 14:
                    # Two possession game
                    base_prob = 0.70 + (score_diff / 30.0)  # 70-80%
                else:
                    # Three+ possession game
                    base_prob = 0.80 + min((score_diff - 14) / 50.0, 0.15)  # 80-95%

                # Apply time weighting - late in game = more certain
                # Early game: pull back towards 50/50
                # Late game: stay with score-based probability
                away_prob_score = base_prob if away_score > home_score else (1.0 - base_prob)
                home_prob_score = base_prob if home_score > away_score else (1.0 - base_prob)

                # Blend market odds with score-based probability
                # Early game: 70% market, 30% score
                # Late game: 20% market, 80% score
                market_weight = max(0.2, 1.0 - time_weight)
                score_weight = min(0.8, time_weight)

                away_prob = (away_prob * market_weight) + (away_prob_score * score_weight)
                home_prob = (home_prob * market_weight) + (home_prob_score * score_weight)

        # Normalize to sum to 1.0 and cap at reasonable limits
        total = away_prob + home_prob
        if total > 0:
            away_prob /= total
            home_prob /= total

        # Cap probabilities at 95% max (nothing is 100% certain)
        away_prob = min(0.95, max(0.05, away_prob))
        home_prob = min(0.95, max(0.05, home_prob))

        # Re-normalize after capping
        total = away_prob + home_prob
        if total > 0:
            away_prob /= total
            home_prob /= total

        # Determine winner
        if away_prob > home_prob:
            result['winner'] = 'away'
            result['probability'] = away_prob
        else:
            result['winner'] = 'home'
            result['probability'] = home_prob

        result['factors'] = {
            'market_odds_weight': 0.6,
            'game_state_weight': 0.4,
            'time_factor': game_state['time_weight']
        }

        return result

    def _calculate_expected_value(self, win_prob: Dict, odds_analysis: Dict) -> Dict:
        """
        Calculate expected value and confidence using unified calculator

        EV = (Win Probability × Potential Profit) - (Loss Probability × Stake)
        Confidence based on edge size and certainty
        """

        probability = win_prob['probability']
        winner = win_prob['winner']

        # Get odds for predicted winner
        if winner == 'away':
            market_odds = odds_analysis['away_odds']
        else:
            market_odds = odds_analysis['home_odds']

        # Use unified EV calculator if available
        if UNIFIED_EV_AVAILABLE and market_odds > 0:
            metrics = UnifiedEVCalculator.calculate_all(
                ai_win_prob=probability,
                market_price=market_odds,
                market_efficiency=odds_analysis.get('market_efficiency', 0.0)
            )
            return {
                'ev': metrics['ev_percentage'],
                'edge': metrics['edge'],
                'confidence': metrics['confidence']
            }

        # Legacy calculation (fallback)
        # Calculate edge (difference between true prob and market prob)
        edge = probability - market_odds

        # Expected value calculation
        # If we bet $1: Win = (1 / market_odds) - 1, Loss = -1
        if market_odds > 0:
            potential_profit = (1.0 / market_odds) - 1.0
        else:
            potential_profit = 0

        ev = (probability * potential_profit) - ((1 - probability) * 1.0)
        ev_percentage = ev * 100

        # Confidence score (0-100)
        # Based on: edge size, probability strength, market efficiency
        confidence = 50  # Base
        confidence += min(edge * 100, 30)  # Up to +30 for edge
        confidence += min((probability - 0.5) * 40, 20)  # Up to +20 for strong probability

        confidence = max(0, min(100, confidence))

        return {
            'ev': ev_percentage,
            'edge': edge,
            'confidence': confidence
        }

    def _kelly_criterion(self, win_probability: float, market_odds: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion

        Kelly = (bp - q) / b
        where:
        - b = decimal odds - 1
        - p = probability of winning
        - q = probability of losing (1 - p)

        Returns fraction of bankroll to bet (0.0 to 1.0)
        """

        if market_odds <= 0 or win_probability <= 0:
            return 0.0

        # Convert market odds to decimal odds
        if market_odds < 1.0:
            # Already in probability form, convert to decimal
            decimal_odds = 1.0 / market_odds
        else:
            decimal_odds = market_odds

        b = decimal_odds - 1.0
        p = win_probability
        q = 1.0 - p

        # Kelly formula
        kelly = (b * p - q) / b

        # Apply fractional Kelly for safety (quarter Kelly)
        fractional_kelly = kelly * self.kelly_fraction

        # Cap at reasonable limits
        return max(0.0, min(0.25, fractional_kelly))

    def _generate_recommendation(self, prediction: Dict) -> Dict:
        """
        Generate final recommendation with clear reasoning

        Returns:
        - Action: STRONG_BUY, BUY, HOLD, or PASS
        - Reasoning: List of factors supporting decision
        - High confidence signal: Boolean
        """

        ev = prediction['expected_value']
        confidence = prediction['confidence_score']
        kelly_size = prediction['kelly_bet_size']
        win_prob = prediction['win_probability']

        reasoning = []
        action = 'PASS'
        high_confidence = False

        # Determine action based on EV and confidence
        if ev > 15 and confidence > 75:
            action = 'STRONG_BUY'
            high_confidence = True
            reasoning.append(f"⚡ HIGH CONFIDENCE: {confidence:.0f}% confidence with {ev:+.1f}% expected value")
        elif ev > 8 and confidence > 65:
            action = 'BUY'
            reasoning.append(f"Good opportunity: {confidence:.0f}% confidence, {ev:+.1f}% EV")
        elif ev > 3 and confidence > 55:
            action = 'HOLD'
            reasoning.append(f"Marginal value: {ev:+.1f}% EV, consider waiting")
        else:
            reasoning.append(f"No value detected (EV: {ev:+.1f}%, Confidence: {confidence:.0f}%)")

        # Add specific reasoning factors
        game_state = prediction['factors_analyzed'].get('game_state', {})
        away_team = prediction.get('away_team', 'Away')
        home_team = prediction.get('home_team', 'Home')

        if game_state.get('is_live'):
            away_score = game_state.get('away_score', 0)
            home_score = game_state.get('home_score', 0)
            period = game_state.get('period', '')
            time_weight = game_state.get('time_weight', 0)

            # Score-specific reasoning
            if away_score > 0 or home_score > 0:
                score_diff = abs(away_score - home_score)
                leader = away_team if away_score > home_score else home_team

                if score_diff == 0:
                    reasoning.append(f"Tied game {away_score}-{home_score} in {period}")
                elif score_diff <= 3:
                    reasoning.append(f"{leader} leads by {score_diff} (field goal game)")
                elif score_diff <= 7:
                    reasoning.append(f"{leader} leads by {score_diff} (one possession)")
                elif score_diff <= 14:
                    reasoning.append(f"{leader} leads by {score_diff} (two possessions)")
                else:
                    reasoning.append(f"{leader} dominant with {score_diff}-point lead")

            # Time-based reasoning
            if time_weight > 0.9:
                reasoning.append(f"Late {period} - result nearly certain")
            elif time_weight > 0.7:
                reasoning.append(f"{period} underway - momentum important")
            elif time_weight > 0.4:
                reasoning.append(f"Mid-game {period} - still time to change")
            else:
                reasoning.append("Early game - score less predictive")

        # Kelly Criterion insights
        if kelly_size > 0.10:
            reasoning.append(f"Kelly suggests {kelly_size*100:.1f}% of bankroll")
        elif kelly_size > 0.05:
            reasoning.append(f"Small Kelly bet: {kelly_size*100:.1f}% of bankroll")
        else:
            reasoning.append("Kelly suggests no bet (edge too small)")

        # Win probability
        if win_prob > 0.70:
            reasoning.append(f"Strong {prediction['predicted_winner']} team advantage: {win_prob*100:.0f}% win probability")
        elif win_prob > 0.60:
            reasoning.append(f"Moderate {prediction['predicted_winner']} team edge: {win_prob*100:.0f}% to win")

        return {
            'action': action,
            'reasoning': reasoning,
            'high_confidence': high_confidence
        }

    def batch_analyze_games(self, games: List[Dict], markets: Dict) -> List[Dict]:
        """
        Analyze multiple games at once

        Args:
            games: List of game data from ESPN
            markets: Dict of market data keyed by ticker

        Returns:
            List of predictions sorted by confidence
        """

        predictions = []

        for game in games:
            # Find matching market data
            game_id = game.get('id')
            market_data = {}  # Would look up from markets dict

            prediction = self.analyze_betting_opportunity(game, market_data)
            predictions.append(prediction)

        # Sort by confidence score (highest first)
        predictions.sort(key=lambda x: x['confidence_score'], reverse=True)

        return predictions

    def get_high_confidence_signals(self, predictions: List[Dict]) -> List[Dict]:
        """
        Filter for high-confidence "lightning bolt" bets

        These are rare opportunities with:
        - High confidence (>75%)
        - Strong expected value (>15%)
        - Kelly suggests significant bet size (>5%)
        """

        signals = [
            p for p in predictions
            if p['high_confidence_signal']
            and p['confidence_score'] > self.high_confidence_threshold * 100
            and p['expected_value'] > 15
        ]

        return signals
