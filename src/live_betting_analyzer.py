"""
Live Betting Analyzer - Real-time analysis of betting opportunities
Compares live game data with odds and AI predictions
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

class LiveBettingAnalyzer:
    """Analyzes live games vs betting odds to find opportunities"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze_game_opportunity(self, game_data: Dict, market_data: Dict, ai_prediction: Optional[Dict] = None) -> Dict:
        """
        Analyze a single game for betting opportunities

        Args:
            game_data: Live game data from ESPN (scores, status, etc)
            market_data: Betting market data from Kalshi (odds, prices)
            ai_prediction: AI prediction data if available

        Returns:
            Dict with opportunity score, reasoning, and recommendation
        """

        opportunity = {
            'game_id': game_data.get('game_id'),
            'away_team': game_data.get('away_team'),
            'home_team': game_data.get('home_team'),
            'score': f"{game_data.get('away_score', 0)}-{game_data.get('home_score', 0)}",
            'status': game_data.get('status_detail', 'Unknown'),
            'is_live': game_data.get('is_live', False),
            'opportunity_score': 0,
            'expected_value': 0,
            'recommendation': 'PASS',
            'reasoning': [],
            'alert_worthy': False,
            'timestamp': datetime.now().isoformat()
        }

        # Only analyze live or upcoming games
        if not game_data.get('is_live') and game_data.get('status_type') != 'pre':
            opportunity['reasoning'].append("Game not live or upcoming")
            return opportunity

        # Extract market odds
        yes_price = market_data.get('yes_price', 0.5)
        no_price = market_data.get('no_price', 0.5)

        # Calculate implied probabilities
        yes_implied_prob = yes_price
        no_implied_prob = no_price

        # Analyze based on game state
        if game_data.get('is_live'):
            opportunity = self._analyze_live_game(game_data, market_data, opportunity)
        else:
            opportunity = self._analyze_pregame(game_data, market_data, ai_prediction, opportunity)

        # Add AI prediction analysis if available
        if ai_prediction:
            opportunity = self._incorporate_ai_prediction(opportunity, ai_prediction, market_data)

        # Calculate final opportunity score (0-100)
        opportunity['opportunity_score'] = self._calculate_opportunity_score(opportunity)

        # Determine if alert-worthy (score > 75)
        opportunity['alert_worthy'] = opportunity['opportunity_score'] > 75

        return opportunity

    def _analyze_live_game(self, game_data: Dict, market_data: Dict, opportunity: Dict) -> Dict:
        """Analyze a live game for in-game betting opportunities"""

        away_score = game_data.get('away_score', 0)
        home_score = game_data.get('home_score', 0)
        period = game_data.get('period', 1)
        clock = game_data.get('clock', '')

        yes_price = market_data.get('yes_price', 0.5)
        no_price = market_data.get('no_price', 0.5)

        # Analyze score differential vs odds
        score_diff = home_score - away_score

        # If home team is winning but odds are cheap (value opportunity)
        if score_diff > 7 and yes_price < 0.6:
            opportunity['reasoning'].append(f"Home team leading by {score_diff} but odds only {yes_price:.0%}")
            opportunity['expected_value'] += 15
            opportunity['recommendation'] = 'BUY'

        # If away team is winning but underdog odds
        elif score_diff < -7 and no_price < 0.6:
            opportunity['reasoning'].append(f"Away team leading by {abs(score_diff)} but odds only {no_price:.0%}")
            opportunity['expected_value'] += 15
            opportunity['recommendation'] = 'BUY'

        # Close game analysis (within 1 score)
        elif abs(score_diff) <= 7:
            # Look for value in the odds
            if yes_price < 0.45 or no_price < 0.45:
                opportunity['reasoning'].append(f"Close game but odds heavily favor one side - potential value")
                opportunity['expected_value'] += 10

        # Time remaining analysis
        if period >= 4:  # 4th quarter/final period
            opportunity['reasoning'].append("4th quarter - higher certainty in outcome")
            opportunity['expected_value'] += 5
        elif period <= 2:  # Early game
            opportunity['reasoning'].append("Early game - more volatility, riskier")
            opportunity['expected_value'] -= 5

        return opportunity

    def _analyze_pregame(self, game_data: Dict, market_data: Dict, ai_prediction: Optional[Dict], opportunity: Dict) -> Dict:
        """Analyze pre-game betting opportunities"""

        yes_price = market_data.get('yes_price', 0.5)
        no_price = market_data.get('no_price', 0.5)

        # Check for value in the odds
        if yes_price < 0.4:
            opportunity['reasoning'].append(f"Cheap odds on favorite ({yes_price:.0%})")
            opportunity['expected_value'] += 5
        elif no_price < 0.4:
            opportunity['reasoning'].append(f"Cheap odds on underdog ({no_price:.0%})")
            opportunity['expected_value'] += 5

        # Check time until game
        if game_data.get('minutes_until'):
            mins = game_data['minutes_until']
            if mins < 60:
                opportunity['reasoning'].append("Game starting soon - less time for odds to change")
                opportunity['expected_value'] += 5
            elif mins > 1440:  # More than 24 hours
                opportunity['reasoning'].append("Game far away - more uncertainty")
                opportunity['expected_value'] -= 3

        return opportunity

    def _incorporate_ai_prediction(self, opportunity: Dict, ai_prediction: Dict, market_data: Dict) -> Dict:
        """Incorporate AI prediction into opportunity analysis"""

        predicted_outcome = ai_prediction.get('predicted_outcome', 'YES')
        confidence = ai_prediction.get('confidence_score', 50)
        edge = ai_prediction.get('edge_percentage', 0)

        # Get corresponding price
        price = market_data.get('yes_price' if predicted_outcome == 'YES' else 'no_price', 0.5)

        # High confidence + good odds = strong opportunity
        if confidence > 70 and price < 0.6:
            opportunity['reasoning'].append(f"AI {confidence:.0f}% confident, odds at {price:.0%} - strong value")
            opportunity['expected_value'] += 20
            opportunity['recommendation'] = 'STRONG_BUY'
        elif confidence > 60 and price < 0.5:
            opportunity['reasoning'].append(f"AI {confidence:.0f}% confident with cheap odds")
            opportunity['expected_value'] += 15
            opportunity['recommendation'] = 'BUY'

        # Add edge to expected value
        opportunity['expected_value'] += edge * 0.1  # Scale edge contribution

        # Add AI details
        opportunity['ai_confidence'] = confidence
        opportunity['ai_edge'] = edge
        opportunity['ai_prediction'] = predicted_outcome

        return opportunity

    def _calculate_opportunity_score(self, opportunity: Dict) -> float:
        """Calculate overall opportunity score (0-100)"""

        score = 50  # Base score

        # Add expected value contribution (max 30 points)
        ev_contribution = min(opportunity['expected_value'], 30)
        score += ev_contribution

        # Add AI confidence contribution if available (max 20 points)
        if 'ai_confidence' in opportunity:
            ai_contribution = (opportunity['ai_confidence'] - 50) * 0.4  # Scale 50-100% to 0-20 points
            score += max(0, ai_contribution)

        # Penalize if not live (reduce by 10 points)
        if not opportunity.get('is_live'):
            score -= 10

        # Bonus for strong recommendation
        if opportunity['recommendation'] == 'STRONG_BUY':
            score += 10

        # Ensure score is within 0-100
        return max(0, min(100, score))

    def analyze_all_opportunities(self, games: List[Dict], markets: Dict, predictions: Dict) -> List[Dict]:
        """
        Analyze all games for opportunities

        Args:
            games: List of live game data from ESPN
            markets: Dict of market data keyed by game/team
            predictions: Dict of AI predictions keyed by market_id

        Returns:
            List of opportunities sorted by opportunity_score
        """

        opportunities = []

        for game in games:
            # Try to find matching market
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')

            # Find market for this game
            market = self._find_market_for_game(home_team, away_team, markets)

            if market:
                # Find prediction for this market
                market_id = market.get('ticker')
                prediction = predictions.get(market_id)

                # Analyze opportunity
                opp = self.analyze_game_opportunity(game, market, prediction)
                opportunities.append(opp)
            else:
                # No market data, but still track the game
                opportunities.append({
                    'game_id': game.get('game_id'),
                    'away_team': away_team,
                    'home_team': home_team,
                    'score': f"{game.get('away_score', 0)}-{game.get('home_score', 0)}",
                    'status': game.get('status_detail', 'Unknown'),
                    'is_live': game.get('is_live', False),
                    'opportunity_score': 0,
                    'recommendation': 'NO_MARKET',
                    'reasoning': ['No betting market available'],
                    'alert_worthy': False,
                    'timestamp': datetime.now().isoformat()
                })

        # Sort by opportunity score (highest first)
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)

        return opportunities

    def _find_market_for_game(self, home_team: str, away_team: str, markets: Dict) -> Optional[Dict]:
        """Find betting market that matches the game"""

        # Simple team name matching
        # In production, would need more sophisticated matching
        for market_id, market in markets.items():
            title = market.get('title', '').lower()

            # Check if both teams are mentioned in market title
            if home_team.lower() in title and away_team.lower() in title:
                return market

            # Check if team names match ticker
            ticker = market.get('ticker', '').lower()
            if home_team.lower() in ticker or away_team.lower() in ticker:
                return market

        return None

    def get_alert_opportunities(self, opportunities: List[Dict], min_score: float = 75) -> List[Dict]:
        """
        Filter opportunities that are alert-worthy

        Args:
            opportunities: List of analyzed opportunities
            min_score: Minimum opportunity score for alerts (default 75)

        Returns:
            List of high-value opportunities worthy of alerts
        """

        alerts = [
            opp for opp in opportunities
            if opp.get('opportunity_score', 0) >= min_score
            and opp.get('recommendation') in ['BUY', 'STRONG_BUY']
        ]

        return alerts

    def generate_alert_message(self, opportunity: Dict) -> str:
        """Generate formatted alert message for Telegram"""

        msg_lines = []
        msg_lines.append("🚨 *HIGH-VALUE BETTING OPPORTUNITY* 🚨")
        msg_lines.append("")
        msg_lines.append(f"*Game:* {opportunity['away_team']} @ {opportunity['home_team']}")
        msg_lines.append(f"*Score:* {opportunity['score']}")
        msg_lines.append(f"*Status:* {opportunity['status']}")
        msg_lines.append("")
        msg_lines.append(f"*Opportunity Score:* {opportunity['opportunity_score']:.1f}/100")
        msg_lines.append(f"*Recommendation:* {opportunity['recommendation']}")
        msg_lines.append(f"*Expected Value:* {opportunity['expected_value']:+.1f}%")
        msg_lines.append("")

        # Add AI prediction if available
        if 'ai_confidence' in opportunity:
            msg_lines.append(f"*AI Prediction:* {opportunity['ai_prediction']} ({opportunity['ai_confidence']:.0f}% confidence)")
            msg_lines.append(f"*AI Edge:* {opportunity['ai_edge']:+.1f}%")
            msg_lines.append("")

        # Add reasoning
        msg_lines.append("*Why this is a good bet:*")
        for reason in opportunity['reasoning']:
            msg_lines.append(f"  • {reason}")

        msg_lines.append("")
        msg_lines.append(f"_Analysis time: {opportunity['timestamp']}_")

        return "\n".join(msg_lines)
