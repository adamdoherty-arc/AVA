"""
Prediction Market Analyzer - Scores opportunities without needing OpenAI API
Uses mathematical models to rate markets based on multiple factors
"""

from datetime import datetime
from typing import Dict, Optional
import math

class PredictionMarketAnalyzer:
    """
    Analyzes prediction markets and generates opportunity scores (0-100)
    Uses quantitative factors: mispricing, liquidity, time value, risk-reward
    """

    def __init__(self) -> None:
        self.weights = {
            'liquidity': 0.30,      # 30% - Can you enter/exit easily?
            'time_value': 0.25,     # 25% - Time until resolution
            'risk_reward': 0.25,    # 25% - Expected value calculation
            'spread': 0.20          # 20% - Bid-ask spread tightness
        }

    def analyze_market(self, market: Dict) -> Dict:
        """
        Analyze a prediction market and return scoring details

        Args:
            market: Dictionary with market data from Kalshi

        Returns:
            Dictionary with score, reasoning, recommendation
        """
        ticker = market.get('ticker', 'Unknown')

        # Extract key metrics
        yes_price = market.get('yes_price')
        no_price = market.get('no_price')
        yes_bid = market.get('yes_bid')
        yes_ask = market.get('yes_ask')
        volume_24h = market.get('volume_24h', 0)
        open_interest = market.get('open_interest', 0)
        days_to_close = market.get('days_to_close', 0)
        bid_ask_spread = market.get('bid_ask_spread', 0)

        # If critical data is missing, skip analysis
        if not yes_price or not yes_bid or not yes_ask:
            return {
                'ai_score': 0,
                'ai_reasoning': 'Insufficient pricing data',
                'recommended_position': 'Skip',
                'risk_level': 'Unknown',
                'expected_value': 0
            }

        # Calculate component scores
        liquidity_score = self._calculate_liquidity_score(volume_24h, open_interest)
        time_score = self._calculate_time_score(days_to_close)
        risk_reward_score = self._calculate_risk_reward_score(yes_price, no_price)
        spread_score = self._calculate_spread_score(bid_ask_spread)

        # Weighted total score
        total_score = (
            liquidity_score * self.weights['liquidity'] +
            time_score * self.weights['time_value'] +
            risk_reward_score * self.weights['risk_reward'] +
            spread_score * self.weights['spread']
        )

        # Round to 2 decimal places
        total_score = round(total_score, 1)

        # Determine recommendation and risk level
        recommendation, risk_level = self._get_recommendation(
            total_score, yes_price, liquidity_score
        )

        # Calculate expected value
        expected_value = self._calculate_expected_value(yes_price)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            liquidity_score, time_score, risk_reward_score, spread_score,
            yes_price, volume_24h, days_to_close
        )

        return {
            'ai_score': total_score,
            'ai_reasoning': reasoning,
            'recommended_position': recommendation,
            'risk_level': risk_level,
            'expected_value': expected_value
        }

    def _calculate_liquidity_score(self, volume_24h: int, open_interest: int) -> float:
        """
        Score based on trading volume and open interest
        Higher volume = better liquidity = higher score
        """
        # Combine volume and open interest
        liquidity_metric = (volume_24h * 2) + open_interest

        # Score on logarithmic scale (0-100)
        if liquidity_metric <= 0:
            return 0
        elif liquidity_metric < 100:
            return 20
        elif liquidity_metric < 1000:
            return 40
        elif liquidity_metric < 10000:
            return 60
        elif liquidity_metric < 100000:
            return 80
        else:
            return 100

    def _calculate_time_score(self, days_to_close: int) -> float:
        """
        Score based on time until market closes
        Sweet spot: 7-30 days (not too soon, not too far)
        """
        if days_to_close <= 0:
            return 0
        elif days_to_close < 3:
            return 40  # Too soon - risky
        elif days_to_close <= 7:
            return 80  # Good - near term
        elif days_to_close <= 30:
            return 100  # Optimal - medium term
        elif days_to_close <= 90:
            return 70  # Okay - longer term
        else:
            return 40  # Too far out - uncertain

    def _calculate_risk_reward_score(self, yes_price: float, no_price: float) -> float:
        """
        Score based on risk-reward profile
        Look for asymmetric payoffs (small risk, big reward)
        """
        if not yes_price or not no_price:
            return 0

        # Calculate potential return for yes and no
        yes_potential_return = (1 - yes_price) / yes_price if yes_price > 0 else 0
        no_potential_return = (1 - no_price) / no_price if no_price > 0 else 0

        # Find the better side
        max_return = max(yes_potential_return, no_potential_return)

        # Score based on potential return
        if max_return < 0.2:  # <20% return
            return 30
        elif max_return < 0.5:  # 20-50% return
            return 50
        elif max_return < 1.0:  # 50-100% return
            return 70
        elif max_return < 2.0:  # 100-200% return
            return 85
        else:  # >200% return
            return 100

    def _calculate_spread_score(self, bid_ask_spread: float) -> float:
        """
        Score based on bid-ask spread tightness
        Tighter spread = easier to trade = higher score
        """
        if bid_ask_spread is None:
            return 50

        # Spread is in decimal (0.01 = 1%)
        if bid_ask_spread <= 0.01:  # 1% or less
            return 100
        elif bid_ask_spread <= 0.02:  # 1-2%
            return 85
        elif bid_ask_spread <= 0.05:  # 2-5%
            return 70
        elif bid_ask_spread <= 0.10:  # 5-10%
            return 50
        else:  # >10%
            return 30

    def _calculate_expected_value(self, yes_price: float) -> float:
        """
        Simple expected value calculation
        Assumes market is efficient (yes_price = true probability)
        """
        if not yes_price:
            return 0

        # EV for buying Yes at current price
        ev_yes = (1 * yes_price) - (yes_price)
        # EV for buying No
        ev_no = (1 * (1 - yes_price)) - (1 - yes_price)

        # Return the better EV as percentage
        return round(max(ev_yes, ev_no) * 100, 2)

    def _get_recommendation(self, score: float, yes_price: float,
                          liquidity_score: float) -> tuple:
        """
        Generate recommendation and risk level based on score and metrics
        """
        # Skip if liquidity is too low
        if liquidity_score < 40:
            return ('Skip', 'High')

        # Recommend based on score
        if score >= 75:
            # Determine Yes or No
            if yes_price and yes_price < 0.50:
                position = 'Yes'
            else:
                position = 'No'

            # Risk level
            if score >= 85:
                risk = 'Low'
            else:
                risk = 'Medium'

            return (position, risk)

        elif score >= 60:
            return ('Maybe', 'Medium')
        else:
            return ('Skip', 'High')

    def _generate_reasoning(self, liquidity_score: float, time_score: float,
                           risk_reward_score: float, spread_score: float,
                           yes_price: float, volume_24h: int,
                           days_to_close: int) -> str:
        """
        Generate human-readable reasoning for the score
        """
        reasons = []

        # Liquidity analysis
        if liquidity_score >= 80:
            reasons.append("Excellent liquidity with high volume")
        elif liquidity_score >= 60:
            reasons.append("Good liquidity")
        elif liquidity_score >= 40:
            reasons.append("Moderate liquidity")
        else:
            reasons.append("Low liquidity - difficult to trade")

        # Time analysis
        if time_score >= 80:
            reasons.append(f"Optimal {days_to_close}-day timeframe")
        elif time_score >= 60:
            reasons.append(f"Reasonable {days_to_close}-day window")
        else:
            reasons.append(f"Suboptimal {days_to_close}-day timeframe")

        # Risk-reward
        if risk_reward_score >= 80:
            reasons.append("Strong risk-reward ratio")
        elif risk_reward_score >= 60:
            reasons.append("Acceptable risk-reward")
        else:
            reasons.append("Limited upside potential")

        # Spread
        if spread_score >= 80:
            reasons.append("Tight bid-ask spread")
        elif spread_score < 50:
            reasons.append("Wide spread increases costs")

        # Price analysis
        if yes_price:
            if yes_price <= 0.30:
                reasons.append(f"Yes priced low at {yes_price:.0%} (high upside if correct)")
            elif yes_price >= 0.70:
                reasons.append(f"Yes priced high at {yes_price:.0%} (safer but lower return)")

        return ". ".join(reasons) + "."
