"""
LLM Explanation Enhancer for Sports Predictions
================================================

Uses local LLMs for generating natural language explanations:
- Qwen 2.5 32B: Fast explanations (~5s)
- DeepSeek R1 32B: Deep reasoning analysis (~30s) for high-stakes predictions

The R1-0528 update provides chain-of-thought reasoning approaching O3/Gemini 2.5 Pro.
"""

import logging
from typing import Dict, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class LLMExplanationEnhancer:
    """
    Enhances sports prediction explanations using local LLMs.

    Models:
    - Qwen 2.5 32B: Default for quick explanations
    - DeepSeek R1 32B: Deep reasoning for high-value predictions (chain-of-thought)

    Falls back to template-based explanations if LLM is unavailable.
    """

    def __init__(self, model_name: str = "qwen2.5:32b-instruct-q4_K_M", use_llm: bool = True):
        """
        Initialize LLM explanation enhancer.

        Args:
            model_name: Ollama model to use for explanations
            use_llm: Whether to use LLM (if False, always use template fallback)
        """
        self.model_name = model_name
        self.use_llm = use_llm
        self.llm_service = None

        if self.use_llm:
            try:
                from src.services import llm_service
                self.llm_service = llm_service.LLMService()
                logger.info(f"LLM Explanation Enhancer initialized with model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize LLM service: {e}. Will use template fallback.")
                self.use_llm = False

    @lru_cache(maxsize=128)
    def _format_prediction_context(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        winner: str,
        probability: float,
        features_str: str,
        adjustments_str: str
    ) -> str:
        """
        Format prediction data into prompt context (cached for performance).

        Args:
            sport: Sport type (NFL, NCAA Football, etc.)
            home_team: Home team name
            away_team: Away team name
            winner: Predicted winner
            probability: Win probability (0-1)
            features_str: String representation of features dict
            adjustments_str: String representation of adjustments dict

        Returns:
            Formatted context string for LLM
        """
        return f"""You are an expert {sport} analyst providing betting insights.

PREDICTION DATA:
- Matchup: {away_team} @ {home_team}
- Predicted Winner: {winner}
- Win Probability: {probability:.1%}

ANALYSIS FACTORS:
{features_str}

ADJUSTMENTS:
{adjustments_str}

Generate a concise, professional 2-3 sentence explanation of why {winner} is favored. Focus on:
1. The key statistical or situational factors driving the prediction
2. Specific strengths/weaknesses that matter for this matchup
3. Confidence level based on the probability

Write in a confident, analytical tone suitable for serious bettors. Be specific and actionable."""

    def enhance_explanation(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        winner: str,
        probability: float,
        features: Dict[str, Any],
        adjustments: Dict[str, Any],
        template_fallback: str
    ) -> str:
        """
        Generate LLM-enhanced explanation or fall back to template.

        Args:
            sport: Sport type (NFL, NCAA Football, etc.)
            home_team: Home team name
            away_team: Away team name
            winner: Predicted winner
            probability: Win probability (0-1)
            features: Features dictionary
            adjustments: Adjustments dictionary
            template_fallback: Template-based explanation to use if LLM fails

        Returns:
            Enhanced explanation string
        """
        # If LLM disabled or unavailable, use template
        if not self.use_llm or self.llm_service is None:
            return template_fallback

        try:
            # Format features for LLM
            features_str = self._format_features(features)
            adjustments_str = self._format_adjustments(adjustments)

            # Create prompt (cached)
            prompt = self._format_prediction_context(
                sport=sport,
                home_team=home_team,
                away_team=away_team,
                winner=winner,
                probability=probability,
                features_str=features_str,
                adjustments_str=adjustments_str
            )

            # Generate explanation with LLM
            response = self.llm_service.generate(
                prompt=prompt,
                model=self.model_name,
                max_tokens=200,
                temperature=0.7,
                timeout=10  # Fast timeout for real-time betting
            )

            if response and response.strip():
                logger.debug(f"LLM explanation generated for {home_team} vs {away_team}")
                return response.strip()
            else:
                logger.warning("LLM returned empty response, using template fallback")
                return template_fallback

        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}. Using template fallback.")
            return template_fallback

    def _format_features(self, features: Dict[str, Any]) -> str:
        """Format features dict into readable bullet points."""
        lines = []

        if 'elo_diff' in features:
            lines.append(f"- Elo Rating Difference: {features['elo_diff']:.1f} points")

        if 'home_elo' in features and 'away_elo' in features:
            lines.append(f"- Home Elo: {features['home_elo']:.0f}, Away Elo: {features['away_elo']:.0f}")

        if 'home_offense_rank' in features:
            lines.append(f"- Home Offense Rank: #{features['home_offense_rank']}")

        if 'home_defense_rank' in features:
            lines.append(f"- Home Defense Rank: #{features['home_defense_rank']}")

        if 'away_offense_rank' in features:
            lines.append(f"- Away Offense Rank: #{features['away_offense_rank']}")

        if 'away_defense_rank' in features:
            lines.append(f"- Away Defense Rank: #{features['away_defense_rank']}")

        if 'home_form' in features:
            lines.append(f"- Home Recent Form: {features['home_form']}/5 wins")

        if 'away_form' in features:
            lines.append(f"- Away Recent Form: {features['away_form']}/5 wins")

        return "\n".join(lines) if lines else "- No detailed features available"

    def _format_adjustments(self, adjustments: Dict[str, Any]) -> str:
        """Format adjustments dict into readable bullet points."""
        lines = []

        if adjustments.get('home_field'):
            lines.append("- Home field advantage applied")

        if adjustments.get('divisional'):
            lines.append("- Divisional rivalry adjustment")

        if 'momentum' in adjustments and abs(adjustments['momentum']) >= 0.03:
            direction = "positive" if adjustments['momentum'] > 0 else "negative"
            lines.append(f"- Momentum adjustment: {direction} ({adjustments['momentum']:.2%})")

        if 'matchup' in adjustments and abs(adjustments['matchup']) >= 0.03:
            direction = "favors home" if adjustments['matchup'] > 0 else "favors away"
            lines.append(f"- Matchup adjustment: {direction} ({adjustments['matchup']:.2%})")

        if 'injury' in adjustments and abs(adjustments['injury']) >= 0.03:
            lines.append(f"- Injury impact: {adjustments['injury']:.2%}")

        if 'rest' in adjustments and abs(adjustments['rest']) >= 0.02:
            lines.append(f"- Rest days adjustment: {adjustments['rest']:.2%}")

        if 'weather' in adjustments and abs(adjustments['weather']) >= 0.02:
            lines.append(f"- Weather conditions: {adjustments['weather']:.2%}")

        return "\n".join(lines) if lines else "- No significant adjustments"

    def generate_quick_summary(
        self,
        winner: str,
        probability: float,
        confidence_level: str = "medium"
    ) -> str:
        """
        Generate a quick one-line summary without LLM.

        Args:
            winner: Predicted winner
            probability: Win probability
            confidence_level: Confidence level (high/medium/low)

        Returns:
            Quick summary string
        """
        confidence_desc = {
            'high': 'strongly favored',
            'medium': 'moderately favored',
            'low': 'slightly favored'
        }

        return f"{winner} {confidence_desc.get(confidence_level, 'favored')} to win ({probability:.1%} probability)"

    def enhance_with_deep_reasoning(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        winner: str,
        probability: float,
        features: Dict[str, Any],
        adjustments: Dict[str, Any],
        betting_line: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate deep reasoning analysis using DeepSeek R1 32B.

        This method uses chain-of-thought reasoning for:
        - High-stakes predictions (probability edge > 10%)
        - Complex multi-factor analysis
        - Value bet identification
        - Risk assessment

        Args:
            sport: Sport type
            home_team: Home team name
            away_team: Away team name
            winner: Predicted winner
            probability: Win probability (0-1)
            features: Features dictionary
            adjustments: Adjustments dictionary
            betting_line: Optional betting line data (spread, total, odds)

        Returns:
            Dict with:
            - thinking: Chain-of-thought reasoning process
            - analysis: Final analysis
            - confidence: Confidence level (High/Medium/Low)
            - value_assessment: Whether this is a value bet
            - key_factors: Top factors driving prediction
        """
        try:
            from src.magnus_local_llm import get_magnus_llm, TaskComplexity

            llm = get_magnus_llm()

            # Format all data
            features_str = self._format_features(features)
            adjustments_str = self._format_adjustments(adjustments)

            betting_str = ""
            if betting_line:
                betting_str = f"""
BETTING LINE DATA:
- Spread: {betting_line.get('spread', 'N/A')}
- Over/Under: {betting_line.get('total', 'N/A')}
- Moneyline: {betting_line.get('moneyline', 'N/A')}
- Implied Probability: {betting_line.get('implied_prob', 'N/A')}
"""

            problem = f"""Analyze this {sport} prediction with deep reasoning:

MATCHUP: {away_team} @ {home_team}
PREDICTED WINNER: {winner}
MODEL PROBABILITY: {probability:.1%}

ANALYSIS FACTORS:
{features_str}

ADJUSTMENTS:
{adjustments_str}
{betting_str}

Provide a deep analysis covering:
1. Why is {winner} favored? What are the key statistical drivers?
2. What could go wrong with this prediction? Identify risks.
3. Is this a value bet? Compare model probability to implied odds.
4. What specific game factors should we watch?
5. Final confidence level (High/Medium/Low) with justification."""

            context = {
                "sport": sport,
                "matchup": f"{away_team} @ {home_team}",
                "model_probability": f"{probability:.1%}",
                "predicted_winner": winner
            }

            # Use DeepSeek R1 for deep reasoning
            result = llm.deep_reasoning(
                problem=problem,
                context=context,
                reasoning_depth="deep",
                extract_thinking=True
            )

            if result.get("status") == "success":
                logger.info(f"Deep reasoning generated for {home_team} vs {away_team}")
                return {
                    "status": "success",
                    "model": "DeepSeek R1 32B (R1-0528)",
                    "thinking": result.get("thinking", ""),
                    "analysis": result.get("answer", result.get("response", "")),
                    "confidence": result.get("confidence", "Medium"),
                    "value_assessment": self._extract_value_assessment(result.get("answer", "")),
                    "key_factors": self._extract_key_factors(result.get("answer", "")),
                    "raw_response": result.get("response", "")
                }
            else:
                logger.warning(f"Deep reasoning failed: {result.get('error')}")
                return {
                    "status": "error",
                    "error": result.get("error", "Unknown error"),
                    "fallback_available": True
                }

        except Exception as e:
            logger.error(f"Deep reasoning error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "fallback_available": True
            }

    def _extract_value_assessment(self, analysis: str) -> Optional[str]:
        """Extract value bet assessment from analysis."""
        import re
        analysis_lower = analysis.lower()

        if "strong value" in analysis_lower or "excellent value" in analysis_lower:
            return "Strong Value"
        elif "value bet" in analysis_lower or "good value" in analysis_lower:
            return "Good Value"
        elif "fair value" in analysis_lower or "correctly priced" in analysis_lower:
            return "Fair Value"
        elif "no value" in analysis_lower or "overpriced" in analysis_lower:
            return "No Value"

        return None

    def _extract_key_factors(self, analysis: str) -> list:
        """Extract key factors from analysis."""
        import re
        factors = []

        # Look for numbered points or bullet points
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            # Match "1." or "- " or "•" patterns
            if re.match(r'^(\d+\.|-|•|\*)\s*', line):
                clean = re.sub(r'^(\d+\.|-|•|\*)\s*', '', line).strip()
                if clean and len(clean) > 10 and len(clean) < 200:
                    factors.append(clean)

        return factors[:5]  # Return top 5 factors


# Singleton instance
_enhancer_instance: Optional[LLMExplanationEnhancer] = None


def get_llm_enhancer(use_llm: bool = True) -> LLMExplanationEnhancer:
    """
    Get or create singleton LLM explanation enhancer.

    Args:
        use_llm: Whether to use LLM (False for template-only mode)

    Returns:
        LLMExplanationEnhancer instance
    """
    global _enhancer_instance

    if _enhancer_instance is None:
        _enhancer_instance = LLMExplanationEnhancer(use_llm=use_llm)

    return _enhancer_instance
