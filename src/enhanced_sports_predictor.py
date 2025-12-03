"""
Enhanced Sports Predictor
Combines research, odds, statistics, and AI to generate comprehensive predictions

Updated: 2025-11-29 - Now uses FREE LLM providers (Groq/Ollama) by default
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os
from dotenv import load_dotenv

from src.sports_prediction_research import SportsResearchAggregator
from src.ava.core.llm_engine import LLMClient, LLMProvider
from src.ava.core.config import get_config

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedSportsPredictor:
    """
    Advanced sports prediction engine that combines:
    - Research from GitHub, Medium, Reddit
    - Statistical models (Elo, power rankings)
    - Betting odds and line movements
    - AI-powered analysis (FREE via Groq/Ollama)
    """

    def __init__(self) -> None:
        self.research_aggregator = SportsResearchAggregator()

        # Use centralized config for FREE LLM provider
        config = get_config()
        provider_map = {
            "ollama": LLMProvider.OLLAMA,
            "groq": LLMProvider.GROQ,
            "huggingface": LLMProvider.HUGGINGFACE,
            "anthropic": LLMProvider.ANTHROPIC,
            "openai": LLMProvider.OPENAI,
        }
        provider = provider_map.get(config.ai.provider.lower(), LLMProvider.GROQ)

        try:
            self.llm = LLMClient(
                provider=provider,
                model=config.ai.default_model,
                cache_enabled=True,
                cache_ttl=300
            )
            logger.info(f"LLM initialized with FREE provider: {provider.value}")
        except Exception as e:
            self.llm = None
            logger.warning(f"Failed to initialize LLM: {e} - AI analysis will be limited")

    # ========================================================================
    # MAIN PREDICTION ENGINE
    # ========================================================================

    def predict_game(self, game_data: Dict) -> Dict:
        """
        Generate comprehensive prediction for a single game

        Args:
            game_data: Dictionary containing:
                - home_team: Home team name
                - away_team: Away team name
                - home_odds: Betting odds for home team (decimal or american)
                - away_odds: Betting odds for away team
                - spread: Point spread
                - over_under: Total points over/under
                - sport: 'nfl' or 'college-football'
                - game_time: datetime of game

        Returns:
            Comprehensive prediction dictionary
        """
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')
        sport = game_data.get('sport', 'nfl')

        logger.info(f"Predicting {sport.upper()} game: {away_team} @ {home_team}")

        # Step 1: Gather research from all sources
        research = self.research_aggregator.get_comprehensive_research(
            sport=sport,
            team1=home_team,
            team2=away_team
        )

        # Step 2: Analyze betting odds for value
        odds_analysis = self._analyze_odds(game_data)

        # Step 3: Calculate statistical factors
        stat_factors = self._calculate_statistical_factors(game_data)

        # Step 4: Get AI analysis (if available)
        ai_analysis = self._get_ai_analysis(game_data, research, odds_analysis, stat_factors)

        # Step 5: Synthesize everything into final prediction
        prediction = self._synthesize_prediction(
            game_data, research, odds_analysis, stat_factors, ai_analysis
        )

        return prediction

    def predict_all_games(self, games: List[Dict]) -> List[Dict]:
        """
        Predict all games and rank by confidence/value

        Args:
            games: List of game dictionaries

        Returns:
            List of predictions ranked by opportunity score
        """
        predictions = []

        for game in games:
            try:
                prediction = self.predict_game(game)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting game {game.get('home_team')} vs {game.get('away_team')}: {e}")
                continue

        # Rank by opportunity score (confidence × value)
        predictions.sort(key=lambda x: x.get('opportunity_score', 0), reverse=True)

        return predictions

    # ========================================================================
    # ODDS ANALYSIS
    # ========================================================================

    def _analyze_odds(self, game_data: Dict) -> Dict:
        """Analyze betting odds for value opportunities"""
        home_odds = game_data.get('home_odds')
        away_odds = game_data.get('away_odds')
        spread = game_data.get('spread', 0)

        # Convert american odds to implied probability
        home_prob = self._odds_to_probability(home_odds)
        away_prob = self._odds_to_probability(away_odds)

        # Calculate vig-adjusted probabilities
        total_prob = home_prob + away_prob
        home_prob_adj = home_prob / total_prob if total_prob > 0 else 0.5
        away_prob_adj = away_prob / total_prob if total_prob > 0 else 0.5

        # Analyze spread value
        spread_value = abs(spread) if spread else 0

        return {
            'home_implied_prob': home_prob_adj,
            'away_implied_prob': away_prob_adj,
            'spread': spread,
            'spread_value': spread_value,
            'value_rating': self._calculate_value_rating(home_prob_adj, away_prob_adj, spread_value)
        }

    def _odds_to_probability(self, odds: Optional[float]) -> float:
        """Convert odds to implied probability"""
        if odds is None:
            return 0.5

        # Assume american odds if > 100 or < -100, otherwise decimal
        if abs(odds) > 50:
            # American odds
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        else:
            # Decimal odds
            if odds > 1:
                return 1 / odds
            else:
                return 0.5

    def _calculate_value_rating(self, home_prob: float, away_prob: float, spread: float) -> str:
        """Calculate value rating for the odds"""
        # Close game (spread < 3) = higher variance = better value potential
        if spread < 3:
            if abs(home_prob - 0.5) < 0.1:  # Very close to 50/50
                return "HIGH VALUE"
            else:
                return "MODERATE VALUE"
        elif spread < 7:
            return "MODERATE VALUE"
        else:
            return "LOW VALUE"  # Blowouts harder to predict profitably

    # ========================================================================
    # STATISTICAL FACTORS
    # ========================================================================

    def _calculate_statistical_factors(self, game_data: Dict) -> List[Dict]:
        """Calculate statistical factors for prediction"""
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')

        # These would ideally pull from database, for now use defaults
        factors = [
            {
                'name': 'Home Field Advantage',
                'weight': 0.15,
                'home_advantage': 3.0,  # Average points
                'description': f"{home_team} playing at home (+3 points average)"
            },
            {
                'name': 'Recent Form',
                'weight': 0.20,
                'description': 'Last 3 games performance analysis'
            },
            {
                'name': 'Head-to-Head History',
                'weight': 0.10,
                'description': 'Historical matchup results'
            },
            {
                'name': 'Strength of Schedule',
                'weight': 0.15,
                'description': 'Quality of opponents faced'
            },
            {
                'name': 'Injury Report',
                'weight': 0.10,
                'description': 'Key player availability'
            },
            {
                'name': 'Weather Conditions',
                'weight': 0.05,
                'description': 'Game-day weather impact (if outdoor)'
            },
            {
                'name': 'Rest Days',
                'weight': 0.05,
                'description': 'Days since last game'
            }
        ]

        return factors

    # ========================================================================
    # AI ANALYSIS
    # ========================================================================

    def _get_ai_analysis(self, game_data: Dict, research: Dict, odds: Dict, factors: List[Dict]) -> Dict:
        """Get AI-powered analysis using FREE LLM (Groq/Ollama)"""
        if not self.llm:
            return {
                'analysis': 'AI analysis unavailable',
                'confidence': 0.5,
                'recommended_pick': None
            }

        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')
        sport = game_data.get('sport', 'nfl')

        # Construct prompt
        prompt = f"""Analyze this {sport.upper()} game and provide a prediction:

**Matchup:** {away_team} @ {home_team}

**Betting Odds:**
- Home implied win probability: {odds['home_implied_prob']:.1%}
- Away implied win probability: {odds['away_implied_prob']:.1%}
- Spread: {odds.get('spread', 'N/A')}
- Value Rating: {odds['value_rating']}

**Research Quality Score:** {research.get('quality_score', 0):.1f}/100

**Key Factors:**
{self._format_factors_for_ai(factors)}

**Research Insights:**
- GitHub models analyzed: {len(research.get('github_models', []))}
- Expert articles reviewed: {len(research.get('medium_articles', []))}
- Community sentiment: {research.get('reddit_sentiment', {}).get('confidence_score', 0)}/100

Based on all available data, provide:
1. Your recommended pick (HOME WIN or AWAY WIN)
2. Confidence level (0-100)
3. Brief 2-3 sentence reasoning
4. Key factor that most influenced your decision

Format your response as:
PICK: [HOME WIN/AWAY WIN]
CONFIDENCE: [0-100]
REASONING: [Your analysis]
KEY_FACTOR: [Most important factor]
"""

        try:
            # Use async LLM call - run in event loop
            async def get_response():
                return await self.llm.generate(
                    system="You are an expert sports analyst specializing in NFL and college football predictions.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )

            # Run async call
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, get_response())
                    response = future.result(timeout=30)
            else:
                response = asyncio.run(get_response())

            analysis_text = response.content
            return self._parse_ai_response(analysis_text)

        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return {
                'analysis': f'AI analysis error: {str(e)}',
                'confidence': 0.5,
                'recommended_pick': None
            }

    def _format_factors_for_ai(self, factors: List[Dict]) -> str:
        """Format factors for AI prompt"""
        lines = []
        for factor in factors:
            weight = factor.get('weight', 0)
            name = factor.get('name')
            desc = factor.get('description', '')
            lines.append(f"- {name} ({weight:.0%} weight): {desc}")
        return "\n".join(lines)

    def _parse_ai_response(self, response_text: str) -> Dict:
        """Parse Claude's response into structured data"""
        lines = response_text.strip().split('\n')
        result = {
            'analysis': response_text,
            'confidence': 50,
            'recommended_pick': None,
            'key_factor': 'Multiple factors'
        }

        for line in lines:
            line = line.strip()
            if line.startswith('PICK:'):
                pick = line.split(':', 1)[1].strip()
                result['recommended_pick'] = 'HOME' if 'HOME' in pick.upper() else 'AWAY'
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf = line.split(':', 1)[1].strip()
                    result['confidence'] = int(''.join(filter(str.isdigit, conf)))
                except:
                    pass
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
            elif line.startswith('KEY_FACTOR:'):
                result['key_factor'] = line.split(':', 1)[1].strip()

        return result

    # ========================================================================
    # PREDICTION SYNTHESIS
    # ========================================================================

    def _synthesize_prediction(self, game_data: Dict, research: Dict,
                               odds: Dict, factors: List[Dict], ai: Dict) -> Dict:
        """Synthesize all analysis into final prediction"""
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')

        # Calculate confidence score (0-100)
        confidence = self._calculate_confidence(research, odds, ai)

        # Determine final pick
        final_pick = ai.get('recommended_pick', 'HOME')

        # Calculate opportunity score (confidence × value)
        value_multiplier = {'HIGH VALUE': 1.5, 'MODERATE VALUE': 1.2, 'LOW VALUE': 0.8}
        value_factor = value_multiplier.get(odds['value_rating'], 1.0)
        opportunity_score = confidence * value_factor

        return {
            'game_id': f"{away_team}_{home_team}_{datetime.now().strftime('%Y%m%d')}",
            'home_team': home_team,
            'away_team': away_team,
            'sport': game_data.get('sport', 'nfl'),
            'game_time': game_data.get('game_time'),

            # Prediction
            'predicted_winner': home_team if final_pick == 'HOME' else away_team,
            'confidence': confidence,
            'opportunity_score': opportunity_score,

            # Odds Analysis
            'home_odds': game_data.get('home_odds'),
            'away_odds': game_data.get('away_odds'),
            'spread': odds.get('spread'),
            'value_rating': odds['value_rating'],
            'home_win_probability': odds['home_implied_prob'],
            'away_win_probability': odds['away_implied_prob'],

            # Research & Factors
            'research_quality': research.get('quality_score', 0),
            'prediction_factors': factors,
            'research_sources': {
                'github_models': len(research.get('github_models', [])),
                'medium_articles': len(research.get('medium_articles', [])),
                'reddit_sentiment': research.get('reddit_sentiment', {}).get('confidence_score', 0)
            },

            # AI Analysis
            'ai_analysis': ai.get('reasoning', ai.get('analysis', '')),
            'ai_confidence': ai.get('confidence', 50),
            'key_factor': ai.get('key_factor', 'Multiple factors'),

            # Metadata
            'timestamp': datetime.now().isoformat(),
            'prediction_version': '2.0'
        }

    def _calculate_confidence(self, research: Dict, odds: Dict, ai: Dict) -> int:
        """Calculate overall confidence score"""
        # Base confidence from AI
        ai_confidence = ai.get('confidence', 50)

        # Research quality bonus (0-20 points)
        research_bonus = min(research.get('quality_score', 0) / 5, 20)

        # Odds clarity bonus (0-10 points)
        prob_diff = abs(odds['home_implied_prob'] - odds['away_implied_prob'])
        odds_bonus = min(prob_diff * 20, 10)

        # Calculate total (cap at 95 to show humility)
        total = min(ai_confidence + research_bonus + odds_bonus, 95)

        return int(total)


# ============================================================================
# QUICK ACCESS FUNCTIONS
# ============================================================================

def predict_nfl_games(games: List[Dict]) -> List[Dict]:
    """Quick function to predict NFL games"""
    predictor = EnhancedSportsPredictor()
    return predictor.predict_all_games(games)


def predict_college_games(games: List[Dict]) -> List[Dict]:
    """Quick function to predict college football games"""
    predictor = EnhancedSportsPredictor()
    return predictor.predict_all_games(games)
