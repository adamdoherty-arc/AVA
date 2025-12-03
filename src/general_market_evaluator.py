"""
General Market AI Evaluator
Analyzes prediction markets across all sectors with AI-powered ratings
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralMarketEvaluator:
    """AI-powered evaluation system for all Kalshi prediction markets"""

    def __init__(self) -> None:
        """Initialize General Market Evaluator"""

        # Scoring weights (total = 1.0)
        self.weights = {
            'value': 0.30,      # Price inefficiency / value opportunity
            'liquidity': 0.25,  # Volume and market activity
            'timing': 0.20,     # Time until close (sweet spot)
            'clarity': 0.15,    # Market clarity and resolvability
            'momentum': 0.10    # Price momentum and trend
        }

    def evaluate_markets(self, markets: List[Dict]) -> List[Dict]:
        """
        Evaluate all markets and generate AI ratings

        Args:
            markets: List of market dictionaries from Kalshi API

        Returns:
            List of evaluated markets with AI ratings
        """
        if not markets:
            return []

        evaluated = []

        for market in markets:
            try:
                evaluation = self._evaluate_market(market)
                if evaluation:
                    evaluated.append(evaluation)
            except Exception as e:
                logger.error(f"Error evaluating market {market.get('ticker')}: {e}")
                continue

        # Rank evaluations
        evaluated = self._rank_markets(evaluated)

        return evaluated

    def _evaluate_market(self, market: Dict) -> Optional[Dict]:
        """Evaluate a single market and generate AI rating"""
        ticker = market.get('ticker')
        title = market.get('title', '')

        # Get prices
        yes_price_raw = market.get('yes_bid') or market.get('last_price')
        no_price_raw = market.get('no_bid') or (100 - yes_price_raw if yes_price_raw else None)

        # Skip markets without valid prices
        if yes_price_raw is None:
            return None

        # Convert to percentage (Kalshi uses cents, 0-100)
        try:
            yes_price = float(yes_price_raw) / 100.0
            no_price = float(no_price_raw) / 100.0 if no_price_raw else (1.0 - yes_price)
        except (TypeError, ValueError):
            return None

        # Calculate component scores
        value_score = self._calculate_value_score(yes_price, no_price, market)
        liquidity_score = self._calculate_liquidity_score(market)
        timing_score = self._calculate_timing_score(market)
        clarity_score = self._calculate_clarity_score(market)
        momentum_score = self._calculate_momentum_score(yes_price, market)

        # Calculate overall score
        overall_score = (
            value_score * self.weights['value'] +
            liquidity_score * self.weights['liquidity'] +
            timing_score * self.weights['timing'] +
            clarity_score * self.weights['clarity'] +
            momentum_score * self.weights['momentum']
        ) * 100

        # Determine recommended action
        recommended_action = self._determine_action(overall_score, value_score, yes_price)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            value_score, liquidity_score, timing_score,
            clarity_score, momentum_score, yes_price, market
        )

        return {
            'ticker': ticker,
            'title': title,
            'sector': market.get('category', 'Other'),
            'yes_price': yes_price,
            'no_price': no_price,
            'overall_score': round(overall_score, 1),
            'value_score': round(value_score * 100, 1),
            'liquidity_score': round(liquidity_score * 100, 1),
            'timing_score': round(timing_score * 100, 1),
            'clarity_score': round(clarity_score * 100, 1),
            'momentum_score': round(momentum_score * 100, 1),
            'recommended_action': recommended_action,
            'reasoning': reasoning,
            'volume': market.get('volume', 0),
            'close_time': market.get('close_time'),
            'raw_market': market
        }

    def _calculate_value_score(self, yes_price: float, no_price: float, market: Dict) -> float:
        """
        Calculate value opportunity score based on price inefficiency

        Higher scores for:
        - Prices away from 50% (more conviction)
        - Clear YES or NO opportunities
        - Mispriced markets
        """
        # Distance from 50% (indicates market conviction)
        distance_from_50 = abs(yes_price - 0.5)
        conviction_score = min(distance_from_50 * 2, 1.0)  # Max at 100% conviction

        # Check for obvious mispricing (prices that don't add to ~1.0)
        price_sum = yes_price + no_price
        if abs(price_sum - 1.0) > 0.05:  # >5% discrepancy
            mispricing_bonus = 0.2
        else:
            mispricing_bonus = 0

        # Favor markets with clear direction (not 50/50)
        if 0.3 <= yes_price <= 0.7:
            # Coin flip territory - lower value
            value_score = conviction_score * 0.5
        else:
            # Clear market direction - higher value
            value_score = conviction_score * 1.2

        return min(value_score + mispricing_bonus, 1.0)

    def _calculate_liquidity_score(self, market: Dict) -> float:
        """
        Calculate liquidity score based on volume and activity

        Higher scores for active, liquid markets
        """
        volume = float(market.get('volume', 0))
        open_interest = float(market.get('open_interest', 0))

        # Logarithmic scaling for volume (handles wide range)
        if volume > 0:
            volume_score = min(math.log10(volume + 1) / 5.0, 1.0)  # Cap at 100k volume
        else:
            volume_score = 0

        # Open interest bonus (indicates active participation)
        if open_interest > 100:
            oi_bonus = 0.2
        elif open_interest > 50:
            oi_bonus = 0.1
        else:
            oi_bonus = 0

        return min(volume_score + oi_bonus, 1.0)

    def _calculate_timing_score(self, market: Dict) -> float:
        """
        Calculate timing score based on days until close

        Sweet spot: 7-60 days
        - Too soon: Not enough time to capitalize
        - Too far: Too much uncertainty
        """
        close_time = market.get('close_time')
        if not close_time:
            return 0.5  # Unknown = medium score

        try:
            if isinstance(close_time, str):
                close_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
            else:
                close_dt = close_time

            days_until_close = (close_dt - datetime.now(close_dt.tzinfo)).days

            if days_until_close < 0:
                return 0  # Already closed
            elif days_until_close < 3:
                return 0.4  # Too soon
            elif days_until_close < 7:
                return 0.7  # Decent
            elif 7 <= days_until_close <= 60:
                return 1.0  # Sweet spot
            elif days_until_close <= 120:
                return 0.8  # Good
            elif days_until_close <= 180:
                return 0.6  # Okay
            else:
                return 0.3  # Too far out

        except Exception as e:
            logger.warning(f"Error parsing close_time: {e}")
            return 0.5

    def _calculate_clarity_score(self, market: Dict) -> float:
        """
        Calculate clarity score based on how clear/resolvable the market is

        Higher scores for:
        - Binary yes/no outcomes
        - Clear resolution criteria
        - Well-defined events
        """
        title = market.get('title', '').lower()
        subtitle = market.get('subtitle', '').lower()

        # Check for clarity indicators
        clarity_indicators = {
            'high': ['will', 'by', 'before', 'above', 'below', 'on', 'win', 'lose'],
            'medium': ['reach', 'exceed', 'announce', 'report'],
            'low': ['might', 'could', 'may', 'possibly']
        }

        score = 0.5  # Default medium clarity

        if any(word in title or word in subtitle for word in clarity_indicators['high']):
            score = 0.9
        elif any(word in title or word in subtitle for word in clarity_indicators['medium']):
            score = 0.7
        elif any(word in title or word in subtitle for word in clarity_indicators['low']):
            score = 0.4

        return score

    def _calculate_momentum_score(self, yes_price: float, market: Dict) -> float:
        """
        Calculate momentum score based on price trends

        For now, basic implementation using price position
        Future: Track historical prices for true momentum
        """
        # Higher momentum for prices showing strong direction
        if yes_price > 0.7 or yes_price < 0.3:
            return 0.8  # Strong momentum
        elif yes_price > 0.6 or yes_price < 0.4:
            return 0.6  # Moderate momentum
        else:
            return 0.3  # Weak momentum (indecisive)

    def _determine_action(self, overall_score: float, value_score: float, yes_price: float) -> str:
        """Determine recommended trading action"""
        if overall_score >= 75 and value_score >= 0.6:
            if yes_price > 0.6:
                return 'BUY_YES'
            elif yes_price < 0.4:
                return 'BUY_NO'
            else:
                return 'WATCH'
        elif overall_score >= 60:
            return 'WATCH'
        else:
            return 'PASS'

    def _generate_reasoning(self, value_score: float, liquidity_score: float,
                           timing_score: float, clarity_score: float,
                           momentum_score: float, yes_price: float, market: Dict) -> str:
        """Generate human-readable reasoning for the rating"""
        reasons = []

        # Value analysis
        if value_score > 0.7:
            if yes_price > 0.6:
                reasons.append(f"Strong YES at {yes_price*100:.0f}% - market shows conviction")
            elif yes_price < 0.4:
                reasons.append(f"Strong NO at {(1-yes_price)*100:.0f}% - clear direction")
            else:
                reasons.append("Good value opportunity identified")
        elif value_score < 0.3:
            reasons.append("Market is 50/50 - no clear edge")

        # Liquidity
        if liquidity_score > 0.7:
            reasons.append("High liquidity - easy entry/exit")
        elif liquidity_score < 0.3:
            reasons.append("Low liquidity - may be hard to trade")

        # Timing
        if timing_score > 0.8:
            reasons.append("Optimal timeframe (7-60 days)")
        elif timing_score < 0.4:
            reasons.append("Suboptimal timing")

        # Clarity
        if clarity_score > 0.8:
            reasons.append("Clear resolution criteria")
        elif clarity_score < 0.5:
            reasons.append("Ambiguous outcome - higher risk")

        # Momentum
        if momentum_score > 0.7:
            reasons.append("Strong price momentum")

        if not reasons:
            reasons.append("Mixed signals - neutral opportunity")

        return " • ".join(reasons)

    def _rank_markets(self, markets: List[Dict]) -> List[Dict]:
        """Rank markets by overall score"""
        markets.sort(key=lambda x: x.get('overall_score', 0), reverse=True)

        # Add rank field
        for i, market in enumerate(markets):
            market['rank'] = i + 1

        return markets


def main():
    """Test the evaluator"""
    evaluator = GeneralMarketEvaluator()

    # Test market
    test_market = {
        'ticker': 'TEST-001',
        'title': 'Will Bitcoin reach $100,000 by end of 2025?',
        'category': 'Crypto',
        'yes_bid': 65,  # 65 cents = 65%
        'volume': 10000,
        'open_interest': 500,
        'close_time': '2025-12-31T23:59:59Z'
    }

    result = evaluator.evaluate_markets([test_market])

    if result:
        print("\n=== Market Evaluation Test ===")
        r = result[0]
        print(f"Market: {r['title']}")
        print(f"Overall Score: {r['overall_score']}/100")
        print(f"Recommended: {r['recommended_action']}")
        print(f"Reasoning: {r['reasoning']}")
        print(f"\nComponent Scores:")
        print(f"  Value:     {r['value_score']}/100")
        print(f"  Liquidity: {r['liquidity_score']}/100")
        print(f"  Timing:    {r['timing_score']}/100")
        print(f"  Clarity:   {r['clarity_score']}/100")
        print(f"  Momentum:  {r['momentum_score']}/100")


if __name__ == '__main__':
    main()
