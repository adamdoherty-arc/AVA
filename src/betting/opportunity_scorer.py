"""
Opportunity Scorer - Unified Ranking System for All Betting Opportunities

Scores and ranks betting opportunities across ALL sports and markets on a 0-100 scale.
Combines multiple factors to identify the most profitable bets.

Factors considered:
- Expected Value (EV): How profitable the bet is
- Confidence: How reliable the AI prediction is
- Edge Size: How much AI disagrees with market
- Liquidity: Market volume (higher = easier to place bet)
- Recency: How fresh the odds are (newer = better)
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .unified_ev_calculator import UnifiedEVCalculator

logger = logging.getLogger(__name__)


class OpportunityScorer:
    """
    Universal scoring system to rank betting opportunities across all sports

    Returns a score from 0-100 where:
    - 90-100: Exceptional opportunity (top tier)
    - 75-89: Great opportunity (highly recommended)
    - 60-74: Good opportunity (recommended)
    - 40-59: Decent opportunity (consider)
    - 0-39: Poor opportunity (avoid)
    """

    # Weighting factors (must sum to 1.0)
    WEIGHTS = {
        'ev': 0.40,          # 40% - Most important: profitability
        'confidence': 0.25,  # 25% - Second: reliability
        'edge': 0.15,        # 15% - Third: size of disagreement
        'liquidity': 0.10,   # 10% - Fourth: ease of placing bet
        'recency': 0.10      # 10% - Fifth: freshness of odds
    }

    def __init__(self) -> None:
        self.ev_calculator = UnifiedEVCalculator()

    def score_opportunity(
        self,
        ai_win_prob: float,
        market_price: float,
        confidence: Optional[float] = None,
        market_volume: Optional[float] = None,
        last_updated: Optional[datetime] = None,
        sport: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Score a single betting opportunity

        Args:
            ai_win_prob: AI predicted win probability (0-1)
            market_price: Market price/probability (0-1)
            confidence: Optional confidence score (0-100). If not provided, will be calculated
            market_volume: Optional market volume in dollars (higher = more liquid)
            last_updated: Optional datetime when odds were last updated
            sport: Optional sport identifier (for logging/debugging)

        Returns:
            Dictionary with:
            {
                'total_score': Overall score (0-100),
                'ev_score': EV component score (0-100),
                'confidence_score': Confidence component score (0-100),
                'edge_score': Edge component score (0-100),
                'liquidity_score': Liquidity component score (0-100),
                'recency_score': Recency component score (0-100),
                'rating': Text rating (Exceptional/Great/Good/Decent/Poor),
                'recommendation': Recommendation text
            }
        """
        # Calculate EV and edge
        metrics = self.ev_calculator.calculate_all(ai_win_prob, market_price)

        # Use provided confidence or calculate it
        if confidence is None:
            confidence = metrics['confidence']

        # Score each component (0-100)
        ev_score = self._score_ev(metrics['ev'])
        confidence_score = confidence  # Already 0-100
        edge_score = self._score_edge(metrics['edge'])
        liquidity_score = self._score_liquidity(market_volume)
        recency_score = self._score_recency(last_updated)

        # Calculate weighted total score
        total_score = (
            ev_score * self.WEIGHTS['ev'] +
            confidence_score * self.WEIGHTS['confidence'] +
            edge_score * self.WEIGHTS['edge'] +
            liquidity_score * self.WEIGHTS['liquidity'] +
            recency_score * self.WEIGHTS['recency']
        )

        # Determine rating and recommendation
        rating, recommendation = self._get_rating(total_score)

        result = {
            'total_score': round(total_score, 2),
            'ev_score': round(ev_score, 2),
            'confidence_score': round(confidence_score, 2),
            'edge_score': round(edge_score, 2),
            'liquidity_score': round(liquidity_score, 2),
            'recency_score': round(recency_score, 2),
            'rating': rating,
            'recommendation': recommendation,
            # Include underlying metrics for reference
            'ev_percentage': metrics['ev_percentage'],
            'edge_percentage': metrics['edge_percentage']
        }

        if sport:
            result['sport'] = sport

        logger.debug(f"Scored {sport or 'unknown'} opportunity: {total_score:.1f}/100 ({rating})")

        return result

    def _score_ev(self, ev: float) -> float:
        """
        Convert EV to 0-100 score

        EV ranges:
        - 0.30+ (30%+): Perfect score (100)
        - 0.20 (20%): Excellent (90)
        - 0.10 (10%): Good (70)
        - 0.05 (5%): Decent (50)
        - 0.00 (0%): Breakeven (25)
        - Negative: Poor (0-25)
        """
        if ev >= 0.30:
            return 100.0
        elif ev >= 0.20:
            # Scale 20-30% EV to 90-100 points
            return 90 + ((ev - 0.20) / 0.10) * 10
        elif ev >= 0.10:
            # Scale 10-20% EV to 70-90 points
            return 70 + ((ev - 0.10) / 0.10) * 20
        elif ev >= 0.05:
            # Scale 5-10% EV to 50-70 points
            return 50 + ((ev - 0.05) / 0.05) * 20
        elif ev >= 0.00:
            # Scale 0-5% EV to 25-50 points
            return 25 + (ev / 0.05) * 25
        else:
            # Negative EV: scale -10% to 0% as 0-25 points
            return max(0, 25 + (ev / 0.10) * 25)

    def _score_edge(self, edge: float) -> float:
        """
        Convert edge (AI vs market disagreement) to 0-100 score

        Edge ranges:
        - 0.30+ (30%+): Maximum disagreement (100)
        - 0.20 (20%): Large disagreement (85)
        - 0.10 (10%): Moderate disagreement (65)
        - 0.05 (5%): Small disagreement (45)
        - 0.00 (0%): No disagreement (25)
        """
        edge_abs = abs(edge)

        if edge_abs >= 0.30:
            return 100.0
        elif edge_abs >= 0.20:
            return 85 + ((edge_abs - 0.20) / 0.10) * 15
        elif edge_abs >= 0.10:
            return 65 + ((edge_abs - 0.10) / 0.10) * 20
        elif edge_abs >= 0.05:
            return 45 + ((edge_abs - 0.05) / 0.05) * 20
        else:
            return 25 + (edge_abs / 0.05) * 20

    def _score_liquidity(self, market_volume: Optional[float]) -> float:
        """
        Convert market volume to 0-100 score

        If volume not available, return neutral score (50)

        Volume ranges (in dollars):
        - $1M+: Highly liquid (100)
        - $500K: Very liquid (85)
        - $100K: Liquid (70)
        - $50K: Moderate (55)
        - $10K: Low (40)
        - <$10K: Very low (25)
        """
        if market_volume is None:
            return 50.0  # Neutral if unknown

        if market_volume >= 1_000_000:
            return 100.0
        elif market_volume >= 500_000:
            return 85 + ((market_volume - 500_000) / 500_000) * 15
        elif market_volume >= 100_000:
            return 70 + ((market_volume - 100_000) / 400_000) * 15
        elif market_volume >= 50_000:
            return 55 + ((market_volume - 50_000) / 50_000) * 15
        elif market_volume >= 10_000:
            return 40 + ((market_volume - 10_000) / 40_000) * 15
        else:
            return max(25, 25 + (market_volume / 10_000) * 15)

    def _score_recency(self, last_updated: Optional[datetime]) -> float:
        """
        Convert odds freshness to 0-100 score

        If last_updated not available, return neutral score (50)

        Age ranges:
        - <5 min: Very fresh (100)
        - 5-15 min: Fresh (85)
        - 15-30 min: Recent (70)
        - 30-60 min: Moderate (55)
        - 1-3 hours: Stale (40)
        - >3 hours: Very stale (25)
        """
        if last_updated is None:
            return 50.0  # Neutral if unknown

        age = datetime.now() - last_updated
        age_minutes = age.total_seconds() / 60

        if age_minutes < 5:
            return 100.0
        elif age_minutes < 15:
            return 85 + ((15 - age_minutes) / 10) * 15
        elif age_minutes < 30:
            return 70 + ((30 - age_minutes) / 15) * 15
        elif age_minutes < 60:
            return 55 + ((60 - age_minutes) / 30) * 15
        elif age_minutes < 180:
            return 40 + ((180 - age_minutes) / 120) * 15
        else:
            return max(25, 40 - ((age_minutes - 180) / 180) * 15)

    def _get_rating(self, score: float) -> tuple[str, str]:
        """
        Convert numeric score to rating and recommendation

        Returns:
            (rating, recommendation)
        """
        if score >= 90:
            return ("Exceptional", "Strong BUY - Top tier opportunity")
        elif score >= 75:
            return ("Great", "BUY - Highly recommended")
        elif score >= 60:
            return ("Good", "BUY - Recommended")
        elif score >= 40:
            return ("Decent", "CONSIDER - Evaluate carefully")
        else:
            return ("Poor", "AVOID - Not recommended")

    def rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Score and rank multiple betting opportunities

        Args:
            opportunities: List of dicts, each containing:
                - ai_win_prob: AI predicted probability (required)
                - market_price: Market price (required)
                - confidence: Optional confidence score
                - market_volume: Optional volume
                - last_updated: Optional datetime
                - sport: Optional sport name
                - game_info: Optional game information (team names, etc.)

        Returns:
            List of opportunities sorted by total_score (highest first),
            with 'score_details' added to each opportunity
        """
        scored_opportunities = []

        for opp in opportunities:
            # Skip if missing required fields
            if 'ai_win_prob' not in opp or 'market_price' not in opp:
                logger.warning(f"Skipping opportunity missing required fields: {opp}")
                continue

            # Score the opportunity
            score_details = self.score_opportunity(
                ai_win_prob=opp['ai_win_prob'],
                market_price=opp['market_price'],
                confidence=opp.get('confidence'),
                market_volume=opp.get('market_volume'),
                last_updated=opp.get('last_updated'),
                sport=opp.get('sport')
            )

            # Add score details to opportunity
            opp_with_score = opp.copy()
            opp_with_score['score_details'] = score_details
            opp_with_score['total_score'] = score_details['total_score']

            scored_opportunities.append(opp_with_score)

        # Sort by total score (highest first)
        scored_opportunities.sort(key=lambda x: x['total_score'], reverse=True)

        logger.info(f"Ranked {len(scored_opportunities)} opportunities")

        return scored_opportunities
