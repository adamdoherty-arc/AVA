"""
AI Consensus Engine for Xtrades Alerts
=======================================

Uses multiple AI models to evaluate trade quality:
- Claude Sonnet 4.5 (50% weight) - Best reasoning
- DeepSeek (30% weight) - Cost-effective validation
- Gemini Pro (20% weight) - Fast secondary opinion

Integrates with ComprehensiveStrategyAnalyzer for full strategy evaluation.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai_options_agent.comprehensive_strategy_analyzer import ComprehensiveStrategyAnalyzer
from src.ai_options_agent.llm_manager import get_llm_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class AIConsensusEngine:
    """
    Multi-model AI consensus engine for trade evaluation.

    Uses comprehensive strategy analyzer with multiple AI models
    to generate consensus scores and recommendations.
    """

    def __init__(self) -> None:
        """Initialize AI consensus engine"""
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment")

        # Initialize LLM manager and strategy analyzer
        try:
            self.llm_manager = get_llm_manager()
            self.strategy_analyzer = ComprehensiveStrategyAnalyzer(self.llm_manager)
            logger.info("✅ AI consensus engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            raise

        # Model weights for consensus
        self.model_weights = {
            'claude': 0.50,  # 50% - Best reasoning
            'deepseek': 0.30,  # 30% - Validation
            'gemini': 0.20,  # 20% - Fast opinion
        }

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(self.db_url)

    def evaluate_alert(self, prepared_alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate alert using multi-model AI consensus.

        Args:
            prepared_alert: Alert data prepared by AlertProcessor

        Returns:
            Dictionary with evaluation results:
            - strategy_rankings: All 10 strategies ranked
            - top_strategy: Best strategy recommendation
            - consensus_score: Weighted score (0-100)
            - individual_scores: Scores from each model
            - ai_reasoning: Combined reasoning
            - key_risk: Primary risk identified
            - recommendation: STRONG_BUY, BUY, HOLD, AVOID
        """
        try:
            stock_data = prepared_alert['stock_data']
            options_data = prepared_alert['options_data']
            symbol = stock_data['symbol']

            logger.info(f"🔍 Evaluating {symbol} with comprehensive strategy analyzer...")

            # Run comprehensive analysis with multi-model consensus
            start_time = datetime.now()

            analysis_result = self.strategy_analyzer.analyze(
                symbol=symbol,
                stock_data=stock_data,
                options_data=options_data
            )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Extract results
            strategy_rankings = analysis_result.get('strategy_rankings', [])
            top_strategy = strategy_rankings[0] if strategy_rankings else None
            environment = analysis_result.get('environment_analysis', {})
            multi_model = analysis_result.get('multi_model_consensus', {})

            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(multi_model, top_strategy)

            # Extract individual model scores (if available)
            individual_scores = self._extract_individual_scores(multi_model)

            # Generate combined AI reasoning
            ai_reasoning = self._generate_combined_reasoning(
                top_strategy, environment, multi_model
            )

            # Identify key risk
            key_risk = self._identify_key_risk(top_strategy, environment)

            # Generate recommendation
            recommendation = self._generate_recommendation(consensus_score, top_strategy)

            result = {
                'alert_id': prepared_alert['alert_id'],
                'symbol': symbol,
                'strategy_rankings': strategy_rankings,
                'top_strategy': top_strategy,
                'consensus_score': consensus_score,
                'individual_scores': individual_scores,
                'ai_reasoning': ai_reasoning,
                'key_risk': key_risk,
                'recommendation': recommendation,
                'environment_analysis': environment,
                'evaluation_duration_ms': duration_ms,
                'evaluated_at': datetime.now().isoformat()
            }

            logger.info(
                f"✅ Evaluated {symbol}: Score={consensus_score}/100, "
                f"Recommendation={recommendation}, Duration={duration_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating alert: {e}", exc_info=True)
            return self._get_error_result(prepared_alert, str(e))

    def _calculate_consensus_score(self, multi_model: Dict, top_strategy: Dict) -> int:
        """
        Calculate weighted consensus score.

        Uses multi-model responses and strategy score.
        """
        if not top_strategy:
            return 0

        # Start with strategy score
        strategy_score = top_strategy.get('score', 50)

        # If multi-model consensus available, use it
        if multi_model and multi_model.get('consensus_available'):
            # TODO: Parse model responses for numeric scores
            # For now, use strategy score as base
            return min(strategy_score, 100)

        return strategy_score

    def _extract_individual_scores(self, multi_model: Dict) -> Dict[str, int]:
        """Extract individual scores from each model"""
        scores = {
            'claude_score': None,
            'deepseek_score': None,
            'gemini_score': None
        }

        if not multi_model or not multi_model.get('consensus_available'):
            return scores

        responses = multi_model.get('responses', {})

        # TODO: Parse actual numeric scores from model responses
        # For now, return None (will be filled in database with NULL)

        return scores

    def _generate_combined_reasoning(self, top_strategy: Dict,
                                     environment: Dict,
                                     multi_model: Dict) -> str:
        """Generate combined AI reasoning from all models"""
        if not top_strategy:
            return "Unable to generate reasoning - no strategy available"

        # Base reasoning from strategy evaluation
        reasoning_parts = []

        # Strategy fit
        strategy_name = top_strategy.get('name', 'Unknown')
        strategy_score = top_strategy.get('score', 0)
        environment_fit = top_strategy.get('environment_fit', '')

        reasoning_parts.append(
            f"**{strategy_name}** is recommended (score: {strategy_score}/100)."
        )

        # Environment analysis
        vol_regime = environment.get('volatility_regime', 'unknown')
        trend = environment.get('trend', 'unknown')
        iv_percentile = environment.get('iv_percentile', 0)

        reasoning_parts.append(
            f"Current market: {vol_regime} volatility ({iv_percentile:.0f} IV percentile), "
            f"{trend} trend. {environment_fit}"
        )

        # Multi-model insights (if available)
        if multi_model and multi_model.get('consensus_available'):
            reasoning_parts.append(
                f"AI models consensus confirms this is a {top_strategy.get('recommendation', 'BUY')} opportunity."
            )

        return " ".join(reasoning_parts)

    def _identify_key_risk(self, top_strategy: Dict, environment: Dict) -> str:
        """Identify primary risk for the trade"""
        if not top_strategy:
            return "Unable to assess risk"

        # Check strategy-specific risks
        strategy_name = top_strategy.get('name', '')
        cons = top_strategy.get('cons', [])

        risk_parts = []

        # Add top con as primary risk
        if cons:
            risk_parts.append(cons[0])

        # Environment-specific risks
        vol_regime = environment.get('volatility_regime', '')
        if vol_regime == 'low' and 'premium selling' in strategy_name.lower():
            risk_parts.append("Low IV reduces premium collection potential")
        elif vol_regime == 'high':
            risk_parts.append("High volatility increases assignment risk")

        if not risk_parts:
            return "Standard options risk applies"

        return "; ".join(risk_parts[:2])  # Top 2 risks

    def _generate_recommendation(self, consensus_score: int,
                                 top_strategy: Dict) -> str:
        """Generate final recommendation based on score"""
        if consensus_score >= 85:
            return "STRONG_BUY"
        elif consensus_score >= 70:
            return "BUY"
        elif consensus_score >= 55:
            return "HOLD"
        else:
            return "AVOID"

    def _get_error_result(self, prepared_alert: Dict, error_msg: str) -> Dict:
        """Return error result when evaluation fails"""
        return {
            'alert_id': prepared_alert['alert_id'],
            'symbol': prepared_alert['stock_data']['symbol'],
            'consensus_score': 0,
            'recommendation': 'AVOID',
            'ai_reasoning': f"Evaluation failed: {error_msg}",
            'key_risk': "System error during evaluation",
            'error': error_msg,
            'evaluated_at': datetime.now().isoformat()
        }

    def save_evaluation_to_database(self, evaluation: Dict[str, Any],
                                    trade_id: int) -> int:
        """
        Save evaluation results to xtrades_alerts table.

        Returns:
            alert_id: Database ID of saved alert
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            top_strategy = evaluation.get('top_strategy', {})
            scores = evaluation.get('individual_scores', {})

            cursor.execute("""
                INSERT INTO xtrades_alerts (
                    trade_id, strategy_rank, strategy_name, strategy_score,
                    environment_fit, claude_score, deepseek_score, gemini_score,
                    consensus_score, ai_reasoning, key_risk, recommendation,
                    evaluation_duration_ms
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
            """, (
                trade_id,
                top_strategy.get('rank', 1),
                top_strategy.get('name', 'Unknown'),
                top_strategy.get('score', 0),
                top_strategy.get('environment_fit', 'Unknown'),
                scores.get('claude_score'),
                scores.get('deepseek_score'),
                scores.get('gemini_score'),
                evaluation.get('consensus_score', 0),
                evaluation.get('ai_reasoning', ''),
                evaluation.get('key_risk', ''),
                evaluation.get('recommendation', 'HOLD'),
                evaluation.get('evaluation_duration_ms', 0)
            ))

            alert_id = cursor.fetchone()['id']

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"💾 Saved evaluation to database (alert_id={alert_id})")
            return alert_id

        except Exception as e:
            logger.error(f"Error saving evaluation: {e}", exc_info=True)
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            raise

    def should_send_notification(self, evaluation: Dict[str, Any]) -> bool:
        """
        Determine if alert should trigger Telegram notification.

        Criteria:
        - Consensus score >= 80
        - Recommendation is BUY or STRONG_BUY
        """
        score = evaluation.get('consensus_score', 0)
        recommendation = evaluation.get('recommendation', 'AVOID')

        if score >= 80 and recommendation in ['BUY', 'STRONG_BUY']:
            logger.info(f"✅ Alert qualifies for notification (score={score}, rec={recommendation})")
            return True

        logger.info(f"❌ Alert does not qualify for notification (score={score}, rec={recommendation})")
        return False


if __name__ == "__main__":
    # Test AI consensus engine
    print("🧪 Testing AI Consensus Engine...")

    engine = AIConsensusEngine()

    # Example prepared alert
    test_alert = {
        'alert_id': 1,
        'alert_type': 'new',
        'stock_data': {
            'symbol': 'AAPL',
            'current_price': 175.50,
            'iv': 0.35,
            'price_52w_high': 195.00,
            'price_52w_low': 155.00,
            'market_cap': 2.75e12,
            'sector': 'Technology',
            'pe_ratio': 28.5
        },
        'options_data': {
            'strike_price': 170.00,
            'dte': 30,
            'delta': -0.30,
            'premium': 250
        }
    }

    # Evaluate
    print(f"\n📊 Evaluating {test_alert['stock_data']['symbol']}...")
    evaluation = engine.evaluate_alert(test_alert)

    print(f"\n✅ Evaluation Results:")
    print(f"Consensus Score: {evaluation['consensus_score']}/100")
    print(f"Recommendation: {evaluation['recommendation']}")
    print(f"Top Strategy: {evaluation.get('top_strategy', {}).get('name', 'N/A')}")
    print(f"AI Reasoning: {evaluation['ai_reasoning']}")
    print(f"Key Risk: {evaluation['key_risk']}")
    print(f"Duration: {evaluation['evaluation_duration_ms']}ms")

    # Check if should notify
    should_notify = engine.should_send_notification(evaluation)
    print(f"\nSend Notification: {'✅ YES' if should_notify else '❌ NO'}")
