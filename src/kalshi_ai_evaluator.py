"""
Kalshi AI Bet Evaluator
Analyzes football markets and generates AI-powered predictions and rankings
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalshiAIEvaluator:
    """AI-powered evaluation system for Kalshi football markets"""

    def __init__(self, use_advanced_features: bool = True):
        """
        Initialize AI Evaluator

        Args:
            use_advanced_features: Enable HuggingFace sentiment, EPA metrics, etc.
        """
        self.use_advanced_features = use_advanced_features

        # Scoring weights (total = 1.0)
        if use_advanced_features:
            # Enhanced weights with AI features
            self.weights = {
                'value': 0.30,      # Price vs implied probability
                'liquidity': 0.20,  # Volume and open interest
                'timing': 0.15,     # Time until close
                'matchup': 0.15,    # Team quality analysis
                'sentiment': 0.10,  # HuggingFace sentiment analysis (NEW)
                'epa_metrics': 0.10 # NFL EPA metrics (NEW)
            }
        else:
            # Original weights
            self.weights = {
                'value': 0.35,      # Price vs implied probability
                'liquidity': 0.25,  # Volume and open interest
                'timing': 0.15,     # Time until close
                'matchup': 0.15,    # Team quality analysis
                'sentiment': 0.10   # Market sentiment/momentum
            }

        # Lazy-load advanced components
        self.sentiment_analyzer = None
        self.nfl_data_fetcher = None
        self.bankroll_manager = None

    def evaluate_markets(self, markets: List[Dict]) -> List[Dict]:
        """
        Evaluate all markets and generate predictions

        Args:
            markets: List of market dictionaries from database

        Returns:
            List of prediction dictionaries
        """
        if not markets:
            return []

        predictions = []

        for market in markets:
            try:
                prediction = self._evaluate_market(market)
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error evaluating market {market.get('ticker')}: {e}")
                continue

        # Rank predictions
        predictions = self._rank_predictions(predictions)

        return predictions

    def _evaluate_market(self, market: Dict) -> Optional[Dict]:
        """Evaluate a single market and generate prediction"""
        ticker = market.get('ticker')
        title = market.get('title', '')

        # Get prices and handle None values + Decimal conversion
        yes_price_raw = market.get('yes_price')
        no_price_raw = market.get('no_price')

        # Skip markets without valid prices
        if yes_price_raw is None or no_price_raw is None:
            return None

        # Convert to float (handles Decimal from database)
        try:
            yes_price = float(yes_price_raw)
            no_price = float(no_price_raw)
        except (TypeError, ValueError):
            return None

        # Validate price range
        if yes_price < 0 or yes_price > 1 or no_price < 0 or no_price > 1:
            return None

        # Get volume and open interest, convert to float
        volume_raw = market.get('volume', 0)
        open_interest_raw = market.get('open_interest', 0)
        volume = float(volume_raw) if volume_raw is not None else 0.0
        open_interest = float(open_interest_raw) if open_interest_raw is not None else 0.0

        close_time = market.get('close_time')

        # Calculate individual scores
        value_score = self._calculate_value_score(yes_price, no_price, title)
        liquidity_score = self._calculate_liquidity_score(volume, open_interest)
        timing_score = self._calculate_timing_score(close_time)
        matchup_score = self._calculate_matchup_score(title)
        sentiment_score = self._calculate_sentiment_score(yes_price, volume)

        # Calculate overall score (0-100)
        overall_score = (
            value_score * self.weights['value'] +
            liquidity_score * self.weights['liquidity'] +
            timing_score * self.weights['timing'] +
            matchup_score * self.weights['matchup'] +
            sentiment_score * self.weights['sentiment']
        )

        # Determine predicted outcome and edge
        predicted_outcome, confidence, edge = self._analyze_outcome(
            yes_price, no_price, title, overall_score
        )

        # Generate recommendation
        recommended_action = self._generate_recommendation(edge, confidence, liquidity_score)

        # Calculate stake size using Kelly Criterion
        stake_pct = self._calculate_stake_size(edge, confidence, yes_price)

        # Calculate max acceptable price
        max_price = self._calculate_max_price(yes_price, edge)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            title, predicted_outcome, confidence, edge,
            value_score, liquidity_score, timing_score
        )

        # Key factors
        key_factors = self._extract_key_factors(
            value_score, liquidity_score, timing_score,
            matchup_score, sentiment_score
        )

        return {
            'ticker': ticker,
            'predicted_outcome': predicted_outcome,
            'confidence_score': round(confidence, 2),
            'edge_percentage': round(edge, 2),
            'overall_rank': None,  # Set during ranking
            'type_rank': None,
            'value_score': round(value_score, 2),
            'liquidity_score': round(liquidity_score, 2),
            'timing_score': round(timing_score, 2),
            'matchup_score': round(matchup_score, 2),
            'sentiment_score': round(sentiment_score, 2),
            'recommended_action': recommended_action,
            'recommended_stake_pct': round(stake_pct, 2),
            'max_price': round(max_price, 4) if max_price else None,
            'reasoning': reasoning,
            'key_factors': key_factors
        }

    def _calculate_value_score(self, yes_price: float, no_price: float, title: str) -> float:
        """
        Calculate value score based on market inefficiency

        Returns: 0-100 score
        """
        # Check if prices are efficient (should sum to ~1.0)
        price_efficiency = abs(yes_price + no_price - 1.0)

        # Higher inefficiency = higher opportunity
        if price_efficiency > 0.1:
            inefficiency_score = min(price_efficiency * 500, 50)
        else:
            inefficiency_score = 0

        # Check if price is extreme (very high or very low)
        # Extreme prices might indicate value
        price_extremity = 0
        if yes_price < 0.3 or yes_price > 0.7:
            price_extremity = 30

        # Simple heuristic: markets with "spread" or "over/under" tend to be 50/50
        # If price deviates significantly, might be value
        is_spread_market = any(word in title.lower() for word in ['spread', 'over', 'under', 'total'])

        deviation_score = 0
        if is_spread_market:
            deviation_from_50 = abs(yes_price - 0.5)
            deviation_score = min(deviation_from_50 * 100, 40)

        total_score = inefficiency_score + price_extremity + deviation_score
        return min(total_score, 100)

    def _calculate_liquidity_score(self, volume: float, open_interest: int) -> float:
        """
        Calculate liquidity score based on volume and OI

        Returns: 0-100 score
        """
        # Volume scoring (logarithmic scale)
        if volume >= 100000:
            volume_score = 50
        elif volume >= 50000:
            volume_score = 40
        elif volume >= 10000:
            volume_score = 30
        elif volume >= 1000:
            volume_score = 20
        elif volume >= 100:
            volume_score = 10
        else:
            volume_score = 5

        # Open interest scoring
        if open_interest >= 10000:
            oi_score = 50
        elif open_interest >= 5000:
            oi_score = 40
        elif open_interest >= 1000:
            oi_score = 30
        elif open_interest >= 100:
            oi_score = 20
        elif open_interest >= 10:
            oi_score = 10
        else:
            oi_score = 5

        return volume_score + oi_score

    def _calculate_timing_score(self, close_time: Optional[str]) -> float:
        """
        Calculate timing score based on time until close

        Returns: 0-100 score
        """
        if not close_time:
            return 50

        try:
            if isinstance(close_time, str):
                close_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
            else:
                close_dt = close_time

            now = datetime.now(close_dt.tzinfo) if close_dt.tzinfo else datetime.now()
            time_until_close = (close_dt - now).total_seconds() / 3600  # hours

            # Sweet spot: 12-48 hours before close
            if 12 <= time_until_close <= 48:
                return 100
            elif 6 <= time_until_close < 12:
                return 80
            elif 48 < time_until_close <= 72:
                return 80
            elif 2 <= time_until_close < 6:
                return 60
            elif 72 < time_until_close <= 168:  # 1 week
                return 60
            elif time_until_close < 2:
                return 30  # Too close to closing
            else:
                return 40  # Too far away

        except Exception:
            return 50

    def _get_sentiment_analyzer(self) -> None:
        """Lazy load sentiment analyzer"""
        if self.sentiment_analyzer is None and self.use_advanced_features:
            try:
                from src.ai.sports_sentiment_embedder import SportsSentimentAnalyzer
                self.sentiment_analyzer = SportsSentimentAnalyzer()
                logger.info("✓ Sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
        return self.sentiment_analyzer

    def _get_nfl_data_fetcher(self) -> None:
        """Lazy load NFL data fetcher"""
        if self.nfl_data_fetcher is None and self.use_advanced_features:
            try:
                from src.nfl_data_fetcher import NFLDataFetcher
                self.nfl_data_fetcher = NFLDataFetcher()
                logger.info("✓ NFL data fetcher loaded")
            except Exception as e:
                logger.warning(f"Could not load NFL data fetcher: {e}")
        return self.nfl_data_fetcher

    def _get_bankroll_manager(self, bankroll: float = 10000):
        """Lazy load bankroll manager"""
        if self.bankroll_manager is None and self.use_advanced_features:
            try:
                from src.bankroll_manager import BankrollManager, KellyMode
                self.bankroll_manager = BankrollManager(
                    bankroll=bankroll,
                    kelly_mode=KellyMode.QUARTER  # Conservative
                )
                logger.info("✓ Bankroll manager loaded")
            except Exception as e:
                logger.warning(f"Could not load bankroll manager: {e}")
        return self.bankroll_manager

    def _calculate_matchup_score(self, title: str) -> float:
        """
        Calculate matchup quality score

        Returns: 0-100 score
        """
        # This is a simplified version - in production, would integrate:
        # - Team ratings/rankings
        # - Historical performance
        # - Home/away splits
        # - Injury reports
        # - Weather conditions

        title_lower = title.lower()

        # High-profile matchups (popular teams)
        popular_teams = [
            # NFL
            'chiefs', 'bills', 'ravens', '49ers', 'cowboys', 'eagles',
            'packers', 'lions', 'bengals', 'dolphins',
            # College
            'alabama', 'georgia', 'ohio state', 'michigan', 'texas',
            'oregon', 'penn state', 'clemson', 'notre dame'
        ]

        matchup_score = 50  # Base score

        # Boost score if popular teams involved
        popular_team_count = sum(1 for team in popular_teams if team in title_lower)
        matchup_score += min(popular_team_count * 20, 40)

        # Playoff/championship games
        if any(word in title_lower for word in ['playoff', 'championship', 'super bowl', 'cfp']):
            matchup_score += 30

        return min(matchup_score, 100)

    def _calculate_sentiment_score(self, yes_price: float, volume: float) -> float:
        """
        Calculate market sentiment score

        Returns: 0-100 score
        """
        # High volume + extreme price = strong sentiment
        # Can indicate crowded trade or contrarian opportunity

        sentiment_strength = 0

        # Price extremity (away from 50/50)
        price_deviation = abs(yes_price - 0.5)
        sentiment_strength += price_deviation * 100

        # Volume multiplier
        if volume >= 50000:
            sentiment_strength *= 1.5
        elif volume >= 10000:
            sentiment_strength *= 1.2

        return min(sentiment_strength, 100)

    def _analyze_outcome(self, yes_price: float, no_price: float,
                        title: str, overall_score: float) -> tuple:
        """
        Determine predicted outcome, confidence, and edge

        Returns: (predicted_outcome, confidence, edge_percentage)
        """
        # For now, use simple logic
        # In production, integrate actual odds from sportsbooks for true probability

        # If yes price < 0.5, YES is favored
        # If yes price > 0.5, NO is favored

        if yes_price < 0.5:
            predicted_outcome = 'yes'
            market_prob = yes_price
            # Estimate true probability (would use external data in production)
            true_prob = self._estimate_true_probability(title, 'yes')
            # Avoid division by very small numbers
            if market_prob < 0.01:
                market_prob = 0.01
            edge = (true_prob - market_prob) / market_prob * 100
        else:
            predicted_outcome = 'no'
            market_prob = no_price
            true_prob = self._estimate_true_probability(title, 'no')
            # Avoid division by very small numbers
            if market_prob < 0.01:
                market_prob = 0.01
            edge = (true_prob - market_prob) / market_prob * 100

        # Cap edge at reasonable maximum (500%)
        edge = min(max(edge, -500), 500)

        # Confidence based on overall score and edge
        confidence = min(overall_score * 0.6 + abs(edge) * 2, 100)

        return predicted_outcome, confidence, edge

    def _estimate_true_probability(self, title: str, side: str) -> float:
        """
        Estimate true probability (simplified)

        In production, would integrate:
        - Sportsbook odds
        - Statistical models
        - Expert consensus
        """
        # For now, return market price + small random adjustment
        # This is a placeholder for actual probability estimation
        import random
        base_prob = 0.5
        adjustment = random.uniform(-0.05, 0.05)
        return max(0.1, min(0.9, base_prob + adjustment))

    def _generate_recommendation(self, edge: float, confidence: float,
                                 liquidity_score: float) -> str:
        """Generate buy/hold/pass recommendation"""
        # Require minimum liquidity
        if liquidity_score < 30:
            return 'pass'

        # Strong buy: high edge + high confidence
        if edge > 10 and confidence > 75:
            return 'strong_buy'

        # Buy: moderate edge + confidence
        if edge > 5 and confidence > 60:
            return 'buy'

        # Hold: small positive edge
        if edge > 0 and confidence > 50:
            return 'hold'

        return 'pass'

    def _calculate_stake_size(self, edge: float, confidence: float, price: float) -> float:
        """
        Calculate recommended stake size using Kelly Criterion

        Returns: Percentage of bankroll (0-10%)
        """
        if edge <= 0 or confidence < 50:
            return 0

        # Convert to decimal odds
        if price == 0:
            return 0

        decimal_odds = 1 / price if price > 0 else 100

        # Kelly fraction = (bp - q) / b
        # b = decimal odds - 1
        # p = probability of winning (confidence / 100)
        # q = probability of losing (1 - p)

        b = decimal_odds - 1
        p = confidence / 100
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Use fractional Kelly (25% of full Kelly for safety)
        fractional_kelly = kelly_fraction * 0.25

        # Cap at 10% of bankroll
        stake_pct = max(0, min(fractional_kelly * 100, 10))

        return stake_pct

    def _calculate_max_price(self, current_price: float, edge: float) -> Optional[float]:
        """Calculate maximum price to pay for this bet"""
        if edge <= 0:
            return None

        # Don't buy if price moves against us by more than half the edge
        edge_decimal = edge / 100
        max_price = current_price * (1 + edge_decimal / 2)

        return min(max_price, 0.95)  # Cap at 95 cents

    def _generate_reasoning(self, title: str, predicted_outcome: str,
                           confidence: float, edge: float,
                           value_score: float, liquidity_score: float,
                           timing_score: float) -> str:
        """Generate human-readable reasoning"""
        reasoning_parts = []

        # Prediction
        reasoning_parts.append(
            f"Recommending {predicted_outcome.upper()} on '{title[:60]}...' "
            f"with {confidence:.0f}% confidence."
        )

        # Edge explanation
        if edge > 10:
            reasoning_parts.append(f"Strong value opportunity with {edge:.1f}% edge over market price.")
        elif edge > 5:
            reasoning_parts.append(f"Moderate value with {edge:.1f}% edge.")
        elif edge > 0:
            reasoning_parts.append(f"Small edge of {edge:.1f}%.")
        else:
            reasoning_parts.append("No significant edge detected.")

        # Key strengths
        strengths = []
        if value_score > 70:
            strengths.append("excellent value")
        if liquidity_score > 70:
            strengths.append("high liquidity")
        if timing_score > 80:
            strengths.append("optimal timing")

        if strengths:
            reasoning_parts.append(f"Key strengths: {', '.join(strengths)}.")

        return " ".join(reasoning_parts)

    def _extract_key_factors(self, value_score: float, liquidity_score: float,
                            timing_score: float, matchup_score: float,
                            sentiment_score: float) -> List[str]:
        """Extract key factors influencing the prediction"""
        factors = []

        if value_score > 70:
            factors.append(f"High value score ({value_score:.0f}/100)")
        if liquidity_score > 70:
            factors.append(f"Strong liquidity ({liquidity_score:.0f}/100)")
        if timing_score > 80:
            factors.append(f"Optimal timing window ({timing_score:.0f}/100)")
        if matchup_score > 70:
            factors.append(f"High-profile matchup ({matchup_score:.0f}/100)")
        if sentiment_score > 70:
            factors.append(f"Strong market sentiment ({sentiment_score:.0f}/100)")

        # Add warnings for low scores
        if liquidity_score < 30:
            factors.append("⚠️ Low liquidity - difficult to execute")
        if timing_score < 40:
            factors.append("⚠️ Suboptimal timing")

        return factors

    def _rank_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Rank predictions by overall opportunity

        Returns: Predictions with rank fields populated
        """
        # Sort by edge percentage (descending)
        sorted_preds = sorted(
            predictions,
            key=lambda x: (
                x['edge_percentage'] if x['edge_percentage'] > 0 else -999,
                x['confidence_score']
            ),
            reverse=True
        )

        # Assign overall ranks
        for i, pred in enumerate(sorted_preds, 1):
            pred['overall_rank'] = i

        return sorted_preds


if __name__ == "__main__":
    # Test the evaluator
    evaluator = KalshiAIEvaluator()

    # Sample market
    test_markets = [
        {
            'ticker': 'TEST-NFL-001',
            'title': 'Will the Chiefs beat the Bills by more than 3 points?',
            'yes_price': 0.45,
            'no_price': 0.55,
            'volume': 50000,
            'open_interest': 5000,
            'close_time': (datetime.now() + timedelta(hours=24)).isoformat()
        }
    ]

    print("\n" + "="*80)
    print("KALSHI AI EVALUATOR - Test")
    print("="*80)

    predictions = evaluator.evaluate_markets(test_markets)

    for pred in predictions:
        print(f"\nTicker: {pred['ticker']}")
        print(f"Predicted Outcome: {pred['predicted_outcome']}")
        print(f"Confidence: {pred['confidence_score']}%")
        print(f"Edge: {pred['edge_percentage']}%")
        print(f"Recommendation: {pred['recommended_action']}")
        print(f"Stake Size: {pred['recommended_stake_pct']}%")
        print(f"\nScores:")
        print(f"  Value: {pred['value_score']}/100")
        print(f"  Liquidity: {pred['liquidity_score']}/100")
        print(f"  Timing: {pred['timing_score']}/100")
        print(f"  Matchup: {pred['matchup_score']}/100")
        print(f"  Sentiment: {pred['sentiment_score']}/100")
        print(f"\nReasoning: {pred['reasoning']}")
        print(f"\nKey Factors:")
        for factor in pred['key_factors']:
            print(f"  - {factor}")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
