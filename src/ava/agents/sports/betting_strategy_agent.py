"""
Betting Strategy Agent - Advanced betting strategy recommendations
with Kelly Criterion, momentum detection, and situational analysis
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ...core.agent_base import BaseAgent, AgentState
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class BetType(str, Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PROP = "prop"
    LIVE = "live"


class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class BetRecommendation:
    """A single bet recommendation"""
    bet_type: BetType
    side: str  # 'home', 'away', 'over', 'under'
    odds: int  # American odds
    probability: float  # Model's estimated probability
    edge: float  # Expected edge (prob - implied prob)
    kelly_fraction: float  # Optimal Kelly bet size (0-1)
    recommended_size: float  # Actual recommended bet as fraction of bankroll
    confidence: str  # 'high', 'medium', 'low'
    reasoning: str


@tool
def calculate_kelly_criterion(probability: float, odds: int) -> str:
    """
    Calculate optimal bet size using Kelly Criterion.

    Args:
        probability: Your estimated true probability of winning (0-1)
        odds: American odds (e.g., -110, +150)

    Returns:
        Kelly fraction and recommended bet size
    """
    try:
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        # Kelly formula: f* = (bp - q) / b
        # Where b = decimal odds - 1, p = probability, q = 1 - p
        b = decimal_odds - 1
        p = probability
        q = 1 - probability

        kelly = (b * p - q) / b

        # Use fractional Kelly for safety (25% of full Kelly)
        fractional_kelly = max(0, kelly * 0.25)

        return f"Kelly: {kelly:.4f}, Recommended (25%): {fractional_kelly:.4f}"
    except Exception as e:
        return f"Error calculating Kelly: {str(e)}"


@tool
def analyze_betting_edge(model_prob: float, market_odds: int) -> str:
    """
    Analyze the edge between model probability and market odds.

    Args:
        model_prob: Model's probability estimate (0-1)
        market_odds: Market American odds

    Returns:
        Edge analysis and recommendation
    """
    try:
        # Convert American odds to implied probability
        if market_odds > 0:
            implied_prob = 100 / (market_odds + 100)
        else:
            implied_prob = abs(market_odds) / (abs(market_odds) + 100)

        edge = model_prob - implied_prob
        edge_pct = edge * 100

        if edge > 0.05:
            rating = "STRONG BET"
        elif edge > 0.02:
            rating = "MODERATE EDGE"
        elif edge > 0:
            rating = "SLIGHT EDGE"
        else:
            rating = "NO EDGE - PASS"

        return (
            f"Model Prob: {model_prob:.1%}, "
            f"Implied Prob: {implied_prob:.1%}, "
            f"Edge: {edge_pct:+.1f}%, "
            f"Rating: {rating}"
        )
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def evaluate_situational_factors(
    home_team: str,
    away_team: str,
    is_primetime: bool,
    is_rivalry: bool,
    rest_days_home: int,
    rest_days_away: int,
    travel_miles_away: int
) -> str:
    """
    Evaluate situational factors that affect game outcome.

    Args:
        home_team: Home team name
        away_team: Away team name
        is_primetime: Is this a primetime game
        is_rivalry: Is this a rivalry game
        rest_days_home: Days of rest for home team
        rest_days_away: Days of rest for away team
        travel_miles_away: Miles traveled by away team

    Returns:
        Situational analysis and adjustments
    """
    try:
        adjustments = []
        total_adjustment = 0.0

        # Home field advantage base
        adjustments.append("Home field: +3%")
        total_adjustment += 0.03

        # Rest advantage
        rest_diff = rest_days_home - rest_days_away
        if rest_diff >= 3:
            adjustments.append(f"Rest advantage ({rest_diff} days): +2%")
            total_adjustment += 0.02
        elif rest_diff <= -3:
            adjustments.append(f"Rest disadvantage ({rest_diff} days): -2%")
            total_adjustment -= 0.02

        # Travel fatigue
        if travel_miles_away > 1500:
            adjustments.append(f"Away travel ({travel_miles_away}mi): +1.5%")
            total_adjustment += 0.015
        elif travel_miles_away > 500:
            adjustments.append(f"Away travel ({travel_miles_away}mi): +0.5%")
            total_adjustment += 0.005

        # Primetime boost (teams play differently under lights)
        if is_primetime:
            adjustments.append("Primetime game: volatility +10%")

        # Rivalry factor (more unpredictable)
        if is_rivalry:
            adjustments.append("Rivalry game: upset potential +5%")

        result = f"Situational Analysis for {away_team} @ {home_team}:\n"
        result += "\n".join(f"  - {adj}" for adj in adjustments)
        result += f"\nTotal home team adjustment: {total_adjustment:+.1%}"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


class BettingStrategyAgent(BaseAgent):
    """
    Betting Strategy Agent - Advanced betting strategy recommendations

    Capabilities:
    - Kelly Criterion bet sizing with fractional Kelly for safety
    - Edge calculation between model and market
    - Situational factor analysis (rest, travel, primetime, rivalry)
    - Momentum detection for live betting
    - Multi-leg parlay analysis
    - Bankroll management recommendations
    """

    # Default fractional Kelly (25% of full Kelly)
    DEFAULT_KELLY_FRACTION = 0.25

    # Maximum single bet size (as fraction of bankroll)
    MAX_BET_SIZE = 0.05

    # Minimum edge to recommend a bet
    MIN_EDGE_THRESHOLD = 0.02

    def __init__(self, use_huggingface: bool = False):
        """Initialize Betting Strategy Agent"""
        tools = [
            calculate_kelly_criterion,
            analyze_betting_edge,
            evaluate_situational_factors
        ]

        super().__init__(
            name="betting_strategy_agent",
            description="Recommends betting strategies with Kelly Criterion and bankroll management",
            tools=tools,
            use_huggingface=use_huggingface
        )

        self.metadata['capabilities'] = [
            'recommend_strategies',
            'kelly_criterion',
            'bankroll_management',
            'strategy_optimization',
            'bet_sizing',
            'edge_calculation',
            'situational_analysis',
            'momentum_detection',
            'live_betting',
            'parlay_analysis'
        ]

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute betting strategy agent.

        Expected context:
        - game_data: Dict with game information
        - market_data: Dict with odds and lines
        - model_prediction: Dict with model's probability estimates
        - bankroll: Optional bankroll amount
        - risk_level: Optional risk tolerance
        """
        try:
            input_text = state.get('input', '')
            context = state.get('context', {})

            game_data = context.get('game_data', {})
            market_data = context.get('market_data', {})
            model_prediction = context.get('model_prediction', {})
            bankroll = context.get('bankroll', 1000.0)
            risk_level = context.get('risk_level', RiskLevel.MODERATE)

            # Generate comprehensive recommendation
            recommendation = self._generate_recommendation(
                game_data=game_data,
                market_data=market_data,
                model_prediction=model_prediction,
                bankroll=bankroll,
                risk_level=risk_level
            )

            state['result'] = recommendation
            return state

        except Exception as e:
            logger.error(f"BettingStrategyAgent error: {e}")
            state['error'] = str(e)
            return state

    def _generate_recommendation(
        self,
        game_data: Dict,
        market_data: Dict,
        model_prediction: Dict,
        bankroll: float,
        risk_level: RiskLevel
    ) -> Dict:
        """Generate comprehensive betting recommendation"""

        recommendations = []

        # Get model probability and market odds
        model_prob = model_prediction.get('home_win_prob', 0.5)
        market_odds = market_data.get('home_odds', -110)

        # Calculate edge
        edge_info = self._calculate_edge(model_prob, market_odds)

        # Analyze situational factors
        situational = self._analyze_situational_factors(game_data)

        # Adjust probability based on situational factors
        adjusted_prob = min(0.95, max(0.05, model_prob + situational['adjustment']))

        # Calculate Kelly sizing
        kelly_info = self._calculate_kelly(adjusted_prob, market_odds, risk_level)

        # Generate recommendation if edge exists
        if edge_info['edge'] >= self.MIN_EDGE_THRESHOLD:
            bet_rec = BetRecommendation(
                bet_type=BetType.MONEYLINE,
                side='home',
                odds=market_odds,
                probability=adjusted_prob,
                edge=edge_info['edge'],
                kelly_fraction=kelly_info['full_kelly'],
                recommended_size=kelly_info['recommended_size'],
                confidence=self._determine_confidence(edge_info['edge']),
                reasoning=self._generate_reasoning(
                    edge_info, situational, game_data
                )
            )
            recommendations.append(bet_rec)

        # Check away side
        away_prob = 1 - model_prob
        away_odds = market_data.get('away_odds', -110)
        away_edge_info = self._calculate_edge(away_prob, away_odds)

        if away_edge_info['edge'] >= self.MIN_EDGE_THRESHOLD:
            away_kelly = self._calculate_kelly(
                1 - adjusted_prob, away_odds, risk_level
            )
            bet_rec = BetRecommendation(
                bet_type=BetType.MONEYLINE,
                side='away',
                odds=away_odds,
                probability=1 - adjusted_prob,
                edge=away_edge_info['edge'],
                kelly_fraction=away_kelly['full_kelly'],
                recommended_size=away_kelly['recommended_size'],
                confidence=self._determine_confidence(away_edge_info['edge']),
                reasoning=self._generate_reasoning(
                    away_edge_info, situational, game_data, is_away=True
                )
            )
            recommendations.append(bet_rec)

        # Calculate optimal bet amounts
        bet_amounts = []
        for rec in recommendations:
            amount = bankroll * rec.recommended_size
            bet_amounts.append({
                'side': rec.side,
                'bet_type': rec.bet_type.value,
                'odds': rec.odds,
                'probability': rec.probability,
                'edge': rec.edge,
                'recommended_amount': round(amount, 2),
                'confidence': rec.confidence,
                'reasoning': rec.reasoning
            })

        return {
            'game_id': game_data.get('game_id', ''),
            'home_team': game_data.get('home_team', ''),
            'away_team': game_data.get('away_team', ''),
            'model_home_prob': model_prob,
            'adjusted_home_prob': adjusted_prob,
            'situational_adjustment': situational['adjustment'],
            'situational_factors': situational['factors'],
            'recommendations': bet_amounts,
            'bankroll_analysis': {
                'current_bankroll': bankroll,
                'risk_level': risk_level.value if isinstance(risk_level, RiskLevel) else risk_level,
                'max_single_bet': bankroll * self.MAX_BET_SIZE,
                'kelly_fraction_used': self.DEFAULT_KELLY_FRACTION
            },
            'pass_recommendation': len(recommendations) == 0,
            'pass_reason': 'No sufficient edge found' if len(recommendations) == 0 else None,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_edge(self, model_prob: float, market_odds: int) -> Dict:
        """Calculate edge between model probability and market odds"""
        # Convert American odds to implied probability
        if market_odds > 0:
            implied_prob = 100 / (market_odds + 100)
        else:
            implied_prob = abs(market_odds) / (abs(market_odds) + 100)

        edge = model_prob - implied_prob

        return {
            'model_prob': model_prob,
            'implied_prob': implied_prob,
            'edge': edge,
            'edge_pct': edge * 100
        }

    def _calculate_kelly(
        self,
        probability: float,
        odds: int,
        risk_level: RiskLevel
    ) -> Dict:
        """Calculate Kelly Criterion bet size"""
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        # Kelly formula: f* = (bp - q) / b
        b = decimal_odds - 1
        p = probability
        q = 1 - probability

        if b <= 0:
            full_kelly = 0
        else:
            full_kelly = (b * p - q) / b

        # Adjust Kelly fraction based on risk level
        risk_multipliers = {
            RiskLevel.CONSERVATIVE: 0.15,
            RiskLevel.MODERATE: 0.25,
            RiskLevel.AGGRESSIVE: 0.40
        }

        kelly_fraction = risk_multipliers.get(risk_level, 0.25)
        recommended_size = max(0, min(self.MAX_BET_SIZE, full_kelly * kelly_fraction))

        return {
            'full_kelly': full_kelly,
            'kelly_fraction': kelly_fraction,
            'recommended_size': recommended_size,
            'expected_growth': full_kelly * (decimal_odds - 1) if full_kelly > 0 else 0
        }

    def _analyze_situational_factors(self, game_data: Dict) -> Dict:
        """Analyze situational factors affecting the game"""
        adjustment = 0.0
        factors = []

        # Home field advantage (baseline ~3%)
        adjustment += 0.03
        factors.append({'factor': 'home_field', 'value': '+3%'})

        # Rest days (if available)
        rest_home = game_data.get('rest_days_home', 0)
        rest_away = game_data.get('rest_days_away', 0)

        if rest_home > 0 and rest_away > 0:
            rest_diff = rest_home - rest_away
            if rest_diff >= 3:
                adjustment += 0.02
                factors.append({'factor': 'rest_advantage', 'value': '+2%'})
            elif rest_diff <= -3:
                adjustment -= 0.02
                factors.append({'factor': 'rest_disadvantage', 'value': '-2%'})

        # Primetime game (higher variance)
        if game_data.get('is_primetime', False):
            factors.append({'factor': 'primetime', 'value': 'increased_variance'})

        # Weather (for outdoor sports)
        weather = game_data.get('weather', {})
        if weather.get('is_bad_weather'):
            factors.append({'factor': 'weather', 'value': 'favor_running_game'})

        # Injuries (if significant)
        injuries = game_data.get('injuries', {})
        if injuries.get('home_key_out'):
            adjustment -= 0.03
            factors.append({'factor': 'home_injuries', 'value': '-3%'})
        if injuries.get('away_key_out'):
            adjustment += 0.03
            factors.append({'factor': 'away_injuries', 'value': '+3%'})

        return {
            'adjustment': adjustment,
            'factors': factors
        }

    def _determine_confidence(self, edge: float) -> str:
        """Determine confidence level based on edge"""
        if edge >= 0.08:
            return 'high'
        elif edge >= 0.05:
            return 'medium'
        else:
            return 'low'

    def _generate_reasoning(
        self,
        edge_info: Dict,
        situational: Dict,
        game_data: Dict,
        is_away: bool = False
    ) -> str:
        """Generate human-readable reasoning for the bet"""
        team = game_data.get('away_team' if is_away else 'home_team', 'Team')

        parts = []

        # Edge explanation
        edge_pct = edge_info['edge'] * 100
        parts.append(f"Model shows {edge_pct:.1f}% edge on {team}")

        # Situational factors
        if situational['factors']:
            factor_names = [f['factor'] for f in situational['factors'][:3]]
            parts.append(f"Key factors: {', '.join(factor_names)}")

        # Confidence
        confidence = self._determine_confidence(edge_info['edge'])
        parts.append(f"Confidence: {confidence}")

        return ". ".join(parts) + "."

    def analyze_live_bet_opportunity(
        self,
        pregame_prob: float,
        live_prob: float,
        live_odds: int,
        game_progress: float
    ) -> Dict:
        """
        Analyze live betting opportunity based on probability shift.

        Returns recommendation for live bet if opportunity exists.
        """
        # Calculate pregame edge vs current odds
        edge_info = self._calculate_edge(live_prob, live_odds)

        # Live bets have lower Kelly due to higher variance
        live_kelly_adjustment = 0.5  # Use 50% of normal Kelly for live

        # Check if probability shift creates opportunity
        prob_shift = live_prob - pregame_prob

        opportunity = {
            'has_opportunity': False,
            'type': None,
            'edge': edge_info['edge'],
            'recommended_size': 0,
            'reasoning': ''
        }

        # Overreaction opportunity (market moved too much)
        if abs(prob_shift) > 0.15 and edge_info['edge'] > 0.03:
            opportunity['has_opportunity'] = True
            opportunity['type'] = 'overreaction'
            opportunity['reasoning'] = (
                f"Market appears to have overreacted to in-game events. "
                f"Probability shifted {prob_shift:+.1%} but edge of {edge_info['edge']:.1%} remains."
            )

        # Value opportunity (good line hasn't adjusted)
        elif edge_info['edge'] > 0.06:
            opportunity['has_opportunity'] = True
            opportunity['type'] = 'value_opportunity'
            opportunity['reasoning'] = (
                f"Significant edge of {edge_info['edge']:.1%} available. "
                f"Market line slow to adjust to game flow."
            )

        if opportunity['has_opportunity']:
            kelly = self._calculate_kelly(live_prob, live_odds, RiskLevel.CONSERVATIVE)
            opportunity['recommended_size'] = kelly['recommended_size'] * live_kelly_adjustment

        return opportunity

    def calculate_parlay_value(
        self,
        legs: List[Dict]
    ) -> Dict:
        """
        Analyze parlay bet for expected value.

        Args:
            legs: List of dicts with 'probability' and 'odds' for each leg

        Returns:
            Analysis of parlay expected value
        """
        if not legs:
            return {'error': 'No legs provided'}

        # Calculate combined probability
        combined_prob = 1.0
        for leg in legs:
            combined_prob *= leg.get('probability', 0.5)

        # Calculate parlay payout multiplier
        parlay_multiplier = 1.0
        for leg in legs:
            odds = leg.get('odds', -110)
            if odds > 0:
                leg_mult = (odds / 100) + 1
            else:
                leg_mult = (100 / abs(odds)) + 1
            parlay_multiplier *= leg_mult

        # Expected value
        ev = (combined_prob * (parlay_multiplier - 1)) - (1 - combined_prob)

        return {
            'num_legs': len(legs),
            'combined_probability': combined_prob,
            'parlay_odds_decimal': parlay_multiplier,
            'parlay_odds_american': self._decimal_to_american(parlay_multiplier),
            'expected_value': ev,
            'is_positive_ev': ev > 0,
            'recommendation': 'BET' if ev > 0.05 else ('CONSIDER' if ev > 0 else 'PASS'),
            'reasoning': (
                f"Combined probability: {combined_prob:.2%}, "
                f"Payout: {parlay_multiplier:.2f}x, "
                f"EV: {ev:+.2%}"
            )
        }

    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
