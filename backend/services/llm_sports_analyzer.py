"""
LLM Sports Analyzer Service
Provides AI-powered analysis for sports betting using Local LLM
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from backend.utils.magnus_local_llm import get_magnus_llm, TaskComplexity

logger = logging.getLogger(__name__)

class LLMSportsAnalyzer:
    """
    Analyzes sports matchups and betting opportunities using Local LLM
    """

    def __init__(self) -> None:
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

    def generate_parlay_ideas(self, games: List[Dict]) -> str:
        """Generate parlay ideas from a list of games"""
        # Implementation for parlay generation
        pass
