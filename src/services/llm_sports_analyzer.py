"""
LLM Sports Analyzer Service
Provides AI-powered analysis for sports betting using Local LLM
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.magnus_local_llm import get_magnus_llm, TaskComplexity

logger = logging.getLogger(__name__)

class LLMSportsAnalyzer:
    """
    Analyzes sports matchups and betting opportunities using Local LLM
    """

    def __init__(self):
        self.llm = get_magnus_llm()

    def analyze_matchup(self, 
                       home_team: str, 
                       away_team: str, 
                       sport: str,
                       context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a specific matchup with provided context data
        
        Args:
            home_team: Home team name
            away_team: Away team name
            sport: Sport name (NFL, NBA, etc.)
            context_data: Dictionary containing stats, odds, injuries, etc.
            
        Returns:
            Dictionary with analysis, prediction, and reasoning
        """
        try:
            # Format context for prompt
            stats_summary = self._format_stats(context_data.get('stats', {}))
            odds_summary = self._format_odds(context_data.get('odds', {}))
            injury_summary = self._format_injuries(context_data.get('injuries', []))
            
            prompt = f"""Analyze this {sport} matchup: {away_team} at {home_team}

CONTEXT:
{stats_summary}

ODDS:
{odds_summary}

INJURIES:
{injury_summary}

Please provide a comprehensive analysis including:
1. Winner prediction with confidence score (0-100%)
2. Key factors influencing the outcome
3. Analysis of the spread and total (over/under)
4. Value betting opportunities (if any)
5. A "Best Bet" recommendation

Format the response as JSON with keys: 'prediction', 'confidence', 'reasoning', 'best_bet', 'analysis'.
"""
            
            response = self.llm.query(
                prompt=prompt,
                complexity=TaskComplexity.ANALYTICAL,
                use_trading_context=False, # It's sports, not trading context per se, but analytical
                max_tokens=1000
            )
            
            # Parse JSON response if possible, otherwise wrap text
            try:
                # Try to find JSON block in response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                elif "{" in response and "}" in response:
                    # Naive extraction
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    json_str = response[start:end]
                    return json.loads(json_str)
                else:
                    return {
                        'prediction': 'Unknown',
                        'confidence': 0,
                        'reasoning': response,
                        'best_bet': 'N/A',
                        'analysis': response
                    }
            except json.JSONDecodeError:
                return {
                    'prediction': 'Error parsing',
                    'confidence': 0,
                    'reasoning': response,
                    'best_bet': 'N/A',
                    'analysis': response
                }
                
        except Exception as e:
            logger.error(f"Error analyzing matchup {away_team} vs {home_team}: {e}")
            return {'error': str(e)}

    def _format_stats(self, stats: Dict) -> str:
        """Format stats dictionary into readable string"""
        if not stats:
            return "No detailed stats available."
        
        output = []
        for team, team_stats in stats.items():
            output.append(f"{team}:")
            for k, v in team_stats.items():
                output.append(f"  - {k}: {v}")
        return "\n".join(output)

    def _format_odds(self, odds: Dict) -> str:
        """Format odds dictionary into readable string"""
        if not odds:
            return "No odds available."
            
        return f"""
Spread: {odds.get('spread', 'N/A')}
Moneyline: {odds.get('moneyline', 'N/A')}
Over/Under: {odds.get('total', 'N/A')}
"""

    def _format_injuries(self, injuries: List[Dict]) -> str:
        """Format injury list into readable string"""
        if not injuries:
            return "No significant injuries reported."
            
        output = []
        for inj in injuries:
            status = inj.get('status', 'Unknown')
            player = inj.get('player', 'Unknown Player')
            team = inj.get('team', '')
            output.append(f"- {player} ({team}): {status}")
        return "\n".join(output)

    def generate_parlay_ideas(self, games: List[Dict]) -> Dict[str, Any]:
        """
        Generate AI-powered parlay ideas from a list of games.

        Args:
            games: List of game dictionaries with team info and odds

        Returns:
            Dictionary with parlay suggestions and analysis
        """
        if not games or len(games) < 2:
            return {
                "error": "Need at least 2 games for parlay generation",
                "parlays": []
            }

        try:
            # Format games for the prompt
            games_summary = []
            for i, game in enumerate(games, 1):
                home = game.get('home_team', 'Home')
                away = game.get('away_team', 'Away')
                spread = game.get('spread', 'N/A')
                total = game.get('total', 'N/A')
                home_ml = game.get('home_odds', 'N/A')
                away_ml = game.get('away_odds', 'N/A')

                games_summary.append(f"""
Game {i}: {away} @ {home}
  Spread: {home} {spread}
  Total: {total}
  Moneyline: {home} ({home_ml}) / {away} ({away_ml})
""")

            prompt = f"""You are a sports betting expert. Analyze these games and generate 3 parlay ideas of varying risk levels.

GAMES:
{''.join(games_summary)}

Generate 3 parlay ideas:
1. CONSERVATIVE (2-3 legs, high probability picks)
2. MODERATE (3-4 legs, balanced risk/reward)
3. AGGRESSIVE (4-5 legs, high payout potential)

For each parlay, provide:
- The specific legs (team/side to bet)
- Brief reasoning for each leg
- Estimated combined probability
- Suggested bet sizing (as % of bankroll)
- Correlation notes (do legs have dependencies?)

IMPORTANT: Consider correlations - avoid combining legs that move together (e.g., two road favorites in same conference).

Format as JSON with structure:
{{
  "parlays": [
    {{
      "type": "conservative/moderate/aggressive",
      "legs": [{{"game": 1, "pick": "Team -3", "reasoning": "..."}}],
      "combined_probability": 0.XX,
      "payout_multiplier": X.XX,
      "correlation_risk": "low/medium/high",
      "bet_sizing": "X% of bankroll",
      "overall_edge": "+X.X%"
    }}
  ],
  "best_parlay": "conservative/moderate/aggressive",
  "avoid_combinations": ["description of bad combos to avoid"]
}}
"""

            response = self.llm.query(
                prompt=prompt,
                complexity=TaskComplexity.ANALYTICAL,
                use_trading_context=False,
                max_tokens=1500
            )

            # Parse JSON response
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                elif "{" in response and "}" in response:
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    json_str = response[start:end]
                    return json.loads(json_str)
                else:
                    # Fallback: generate structured response from text
                    return self._generate_fallback_parlays(games)
            except json.JSONDecodeError:
                return self._generate_fallback_parlays(games)

        except Exception as e:
            logger.error(f"Error generating parlay ideas: {e}")
            return self._generate_fallback_parlays(games)

    def _generate_fallback_parlays(self, games: List[Dict]) -> Dict[str, Any]:
        """Generate basic parlay suggestions when LLM fails"""
        from src.prediction_agents.ensemble_predictor import get_ensemble_predictor

        predictor = get_ensemble_predictor()
        parlays = []

        # Get predictions for all games
        predictions = []
        for game in games:
            try:
                pred = predictor.predict(
                    home_team=game.get('home_team', ''),
                    away_team=game.get('away_team', ''),
                    sport=game.get('sport', 'NFL'),
                    game_data=game,
                    market_odds=game.get('odds')
                )
                predictions.append({
                    "game": game,
                    "prediction": pred
                })
            except:
                continue

        if len(predictions) < 2:
            return {"parlays": [], "error": "Not enough valid predictions"}

        # Sort by confidence
        predictions.sort(key=lambda x: x['prediction'].confidence_score, reverse=True)

        # Conservative: Top 2 highest confidence
        if len(predictions) >= 2:
            conservative_legs = []
            combined_prob = 1.0
            for p in predictions[:2]:
                pick = p['game'].get('home_team') if p['prediction'].home_win_prob > 0.5 else p['game'].get('away_team')
                prob = max(p['prediction'].home_win_prob, 1 - p['prediction'].home_win_prob)
                combined_prob *= prob
                conservative_legs.append({
                    "game": f"{p['game'].get('away_team')} @ {p['game'].get('home_team')}",
                    "pick": pick,
                    "probability": round(prob, 3),
                    "reasoning": f"{p['prediction'].confidence} confidence pick"
                })

            parlays.append({
                "type": "conservative",
                "legs": conservative_legs,
                "combined_probability": round(combined_prob, 4),
                "payout_multiplier": round(1 / combined_prob, 2) if combined_prob > 0 else 0,
                "correlation_risk": "low",
                "bet_sizing": "2% of bankroll"
            })

        # Moderate: Top 3-4 picks
        if len(predictions) >= 3:
            moderate_legs = []
            combined_prob = 1.0
            for p in predictions[:min(4, len(predictions))]:
                pick = p['game'].get('home_team') if p['prediction'].home_win_prob > 0.5 else p['game'].get('away_team')
                prob = max(p['prediction'].home_win_prob, 1 - p['prediction'].home_win_prob)
                combined_prob *= prob
                moderate_legs.append({
                    "game": f"{p['game'].get('away_team')} @ {p['game'].get('home_team')}",
                    "pick": pick,
                    "probability": round(prob, 3)
                })

            parlays.append({
                "type": "moderate",
                "legs": moderate_legs,
                "combined_probability": round(combined_prob, 4),
                "payout_multiplier": round(1 / combined_prob, 2) if combined_prob > 0 else 0,
                "correlation_risk": "medium",
                "bet_sizing": "1% of bankroll"
            })

        # Aggressive: All games with some underdogs
        if len(predictions) >= 4:
            aggressive_legs = []
            combined_prob = 1.0
            for i, p in enumerate(predictions[:min(5, len(predictions))]):
                # Mix in some underdogs for higher payout
                if i >= 3 and p['prediction'].home_win_prob < 0.55:
                    # Take underdog
                    pick = p['game'].get('away_team') if p['prediction'].home_win_prob > 0.5 else p['game'].get('home_team')
                    prob = 1 - max(p['prediction'].home_win_prob, 1 - p['prediction'].home_win_prob)
                else:
                    pick = p['game'].get('home_team') if p['prediction'].home_win_prob > 0.5 else p['game'].get('away_team')
                    prob = max(p['prediction'].home_win_prob, 1 - p['prediction'].home_win_prob)

                combined_prob *= prob
                aggressive_legs.append({
                    "game": f"{p['game'].get('away_team')} @ {p['game'].get('home_team')}",
                    "pick": pick,
                    "probability": round(prob, 3)
                })

            parlays.append({
                "type": "aggressive",
                "legs": aggressive_legs,
                "combined_probability": round(combined_prob, 4),
                "payout_multiplier": round(1 / combined_prob, 2) if combined_prob > 0 else 0,
                "correlation_risk": "high",
                "bet_sizing": "0.5% of bankroll"
            })

        return {
            "parlays": parlays,
            "best_parlay": "conservative" if parlays else None,
            "generated_by": "ensemble_predictor_fallback"
        }

    def analyze_value_bet(
        self,
        game: Dict,
        model_probability: float,
        market_odds: int
    ) -> Dict[str, Any]:
        """
        Analyze a potential value betting opportunity.

        Args:
            game: Game information
            model_probability: Model's estimated probability
            market_odds: Current market odds (American format)

        Returns:
            Dictionary with value analysis and recommendation
        """
        # Convert American odds to implied probability
        if market_odds < 0:
            implied_prob = abs(market_odds) / (abs(market_odds) + 100)
        else:
            implied_prob = 100 / (market_odds + 100)

        edge = model_probability - implied_prob

        # Generate LLM analysis
        prompt = f"""Analyze this potential value bet:

GAME: {game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}
Model Probability: {model_probability:.1%}
Market Implied Probability: {implied_prob:.1%}
Edge: {edge:+.1%}
Market Odds: {market_odds:+d if market_odds > 0 else market_odds}

Provide a brief analysis (2-3 sentences) on:
1. Is this a true value bet or noise?
2. Key risks that could make the model wrong
3. Recommended action and bet sizing

Keep response under 100 words. Be direct and actionable.
"""

        try:
            response = self.llm.query(
                prompt=prompt,
                complexity=TaskComplexity.SIMPLE,
                max_tokens=200
            )
            analysis = response.strip()
        except:
            analysis = f"Edge of {edge:.1%} detected. Model favors this side."

        # Determine recommendation
        if edge >= 0.08:
            recommendation = "STRONG VALUE"
            bet_size = "2-3% of bankroll"
        elif edge >= 0.05:
            recommendation = "VALUE BET"
            bet_size = "1-2% of bankroll"
        elif edge >= 0.02:
            recommendation = "SLIGHT EDGE"
            bet_size = "0.5-1% of bankroll"
        else:
            recommendation = "NO VALUE"
            bet_size = "Pass"

        return {
            "game": f"{game.get('away_team')} @ {game.get('home_team')}",
            "model_probability": round(model_probability, 4),
            "implied_probability": round(implied_prob, 4),
            "edge": round(edge, 4),
            "market_odds": market_odds,
            "recommendation": recommendation,
            "suggested_bet_size": bet_size,
            "analysis": analysis,
            "kelly_fraction": round(max(0, edge / (1 - implied_prob)), 4) if edge > 0 else 0
        }
