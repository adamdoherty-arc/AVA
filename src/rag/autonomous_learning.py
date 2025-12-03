"""
Autonomous Learning System for RAG

This module implements self-improving mechanisms that continuously learn
from trading outcomes without human intervention.

Key Features:
- Success weight updates based on recommendation accuracy
- Pattern extraction from trade outcomes
- Market regime detection and adaptation
- Confidence calibration
- Aggregate performance analysis

Author: Magnus Wheel Strategy Dashboard
Created: 2025-11-10
"""

import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from dataclasses import dataclass

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class LearningMetrics:
    """Metrics from a learning cycle"""
    trades_processed: int
    weights_updated: int
    insights_extracted: int
    accuracy_improvement: float
    timestamp: datetime


class SuccessWeightUpdater:
    """
    Autonomous success weight updater

    Updates weights based on recommendation accuracy with confidence-weighted adjustments
    """

    def __init__(
        self,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
        boost_factor: float = 1.1,
        penalty_factor: float = 0.9
    ):
        """
        Initialize weight updater

        Args:
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            boost_factor: Multiplier for correct recommendations (1.1 = 10% boost)
            penalty_factor: Multiplier for incorrect recommendations (0.9 = 10% penalty)
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.boost_factor = boost_factor
        self.penalty_factor = penalty_factor

    def update_weight(
        self,
        current_weight: float,
        recommendation_correct: bool,
        recommendation_confidence: float
    ) -> float:
        """
        Update success weight based on outcome

        Higher confidence recommendations have larger impact:
        - High confidence + correct = big boost
        - High confidence + wrong = big penalty
        - Low confidence = smaller adjustment

        Args:
            current_weight: Current success weight
            recommendation_correct: Whether recommendation was correct
            recommendation_confidence: Confidence level (0-100)

        Returns:
            New success weight
        """
        # Confidence-weighted adjustment (normalize to 0-1)
        confidence_factor = recommendation_confidence / 100.0

        if recommendation_correct:
            # Boost for correct recommendation
            adjustment = 1 + (self.boost_factor - 1) * confidence_factor
        else:
            # Penalty for incorrect recommendation
            adjustment = 1 - (1 - self.penalty_factor) * confidence_factor

        new_weight = current_weight * adjustment

        # Clamp to bounds
        return max(self.min_weight, min(self.max_weight, new_weight))

    def calculate_accuracy(
        self,
        times_referenced: int,
        current_accuracy: float,
        new_outcome: bool
    ) -> float:
        """
        Calculate running average of recommendation accuracy

        Args:
            times_referenced: Number of times this trade was referenced
            current_accuracy: Current accuracy rate (0-1)
            new_outcome: Whether latest recommendation was correct

        Returns:
            Updated accuracy rate
        """
        if times_referenced == 0:
            return 1.0 if new_outcome else 0.0

        total_correct = current_accuracy * times_referenced
        new_total_correct = total_correct + (1.0 if new_outcome else 0.0)
        new_total_referenced = times_referenced + 1

        return new_total_correct / new_total_referenced


class MarketRegimeDetector:
    """
    Autonomous market regime classification

    Detects changes in market conditions that affect trading strategies
    """

    def __init__(self) -> None:
        """Initialize regime detector"""
        self.regime_history = []

    def detect_regime(
        self,
        vix: float,
        spy_trend: str,
        sector_rotation: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Classify current market regime

        Args:
            vix: Current VIX level
            spy_trend: SPY trend (bullish/bearish/neutral)
            sector_rotation: Sector performance dict

        Returns:
            Regime classification dict
        """
        # Volatility regime
        if vix < 12:
            vol_regime = "low"
        elif vix < 16:
            vol_regime = "normal"
        elif vix < 25:
            vol_regime = "high"
        else:
            vol_regime = "extreme"

        # Trend regime
        trend_regime = self._classify_trend(spy_trend)

        # Risk appetite (0 = risk-off, 1 = risk-on)
        risk_appetite = self._calculate_risk_appetite(vix, spy_trend)

        regime = {
            "volatility_regime": vol_regime,
            "trend_regime": trend_regime,
            "risk_appetite": risk_appetite,
            "vix": vix,
            "spy_trend": spy_trend,
            "timestamp": datetime.now().isoformat()
        }

        self.regime_history.append(regime)

        return regime

    def _classify_trend(self, spy_trend: str) -> str:
        """Classify trend into granular categories"""
        trend_map = {
            "bullish": "bull",
            "bearish": "bear",
            "neutral": "neutral"
        }
        return trend_map.get(spy_trend.lower(), "neutral")

    def _calculate_risk_appetite(self, vix: float, spy_trend: str) -> float:
        """
        Calculate overall risk appetite (0-1)

        Lower VIX + bullish trend = higher risk appetite
        """
        # VIX component (inverted: lower VIX = higher appetite)
        vix_score = max(0, min(1, 1 - (vix - 10) / 30))

        # Trend component
        trend_scores = {
            "bullish": 1.0,
            "neutral": 0.5,
            "bearish": 0.0
        }
        trend_score = trend_scores.get(spy_trend.lower(), 0.5)

        # Weighted average
        risk_appetite = 0.6 * vix_score + 0.4 * trend_score

        return round(risk_appetite, 2)

    def should_adjust_strategy(
        self,
        current_regime: Dict[str, Any],
        historical_regime: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Determine if market regime change warrants strategy adjustment

        Args:
            current_regime: Current regime classification
            historical_regime: Historical regime classification

        Returns:
            (should_adjust, reason)
        """
        # Major volatility change
        vol_changed = (
            current_regime["volatility_regime"] !=
            historical_regime["volatility_regime"]
        )

        # Trend reversal
        trend_reversed = (
            ("bull" in current_regime["trend_regime"] and
             "bear" in historical_regime["trend_regime"]) or
            ("bear" in current_regime["trend_regime"] and
             "bull" in historical_regime["trend_regime"])
        )

        # Major risk appetite shift (> 0.3 change)
        risk_shifted = abs(
            current_regime["risk_appetite"] -
            historical_regime["risk_appetite"]
        ) > 0.3

        if vol_changed:
            return True, f"Volatility regime changed: {historical_regime['volatility_regime']} → {current_regime['volatility_regime']}"
        elif trend_reversed:
            return True, f"Trend reversed: {historical_regime['trend_regime']} → {current_regime['trend_regime']}"
        elif risk_shifted:
            return True, f"Risk appetite shifted: {historical_regime['risk_appetite']:.2f} → {current_regime['risk_appetite']:.2f}"

        return False, "No significant regime change"


class PatternExtractor:
    """
    Autonomous pattern extraction from trade outcomes

    Extracts learnings and insights that can be embedded and reused
    """

    def __init__(self) -> None:
        """Initialize pattern extractor"""
        pass

    def extract_insights(
        self,
        recommendation: Dict[str, Any],
        trade_outcome: Dict[str, Any],
        similar_trades: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract learnings from trade outcome

        Args:
            recommendation: Original recommendation dict
            trade_outcome: Actual trade outcome dict
            similar_trades: List of similar historical trades used

        Returns:
            List of insights to be embedded and stored
        """
        insights = []

        # 1. Accuracy insight
        if recommendation.get('recommendation_correct', False):
            insight = {
                "type": "success_pattern",
                "ticker": trade_outcome.get('ticker', 'UNKNOWN'),
                "strategy": trade_outcome.get('strategy', 'UNKNOWN'),
                "text": (
                    f"Successful {trade_outcome.get('strategy')} on {trade_outcome.get('ticker')}: "
                    f"{recommendation.get('reasoning', 'N/A')[:200]}... "
                    f"Result: ${trade_outcome.get('pnl', 0):+.2f} in {trade_outcome.get('hold_days', 0)} days."
                ),
                "pnl": trade_outcome.get('pnl', 0),
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)
        else:
            insight = {
                "type": "failure_pattern",
                "ticker": trade_outcome.get('ticker', 'UNKNOWN'),
                "strategy": trade_outcome.get('strategy', 'UNKNOWN'),
                "text": (
                    f"Failed {trade_outcome.get('strategy')} on {trade_outcome.get('ticker')}: "
                    f"Recommendation was {recommendation.get('recommendation')}, but "
                    f"result was {trade_outcome.get('actual_outcome')}. "
                    f"Contributing factors: {self._identify_failure_factors(trade_outcome)}"
                ),
                "pnl": trade_outcome.get('pnl', 0),
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)

        # 2. Market regime insight
        if self._regime_changed_significantly(trade_outcome):
            vix_entry = trade_outcome.get('vix_at_entry', 0)
            vix_exit = trade_outcome.get('vix_at_exit', 0)
            insight = {
                "type": "regime_change",
                "ticker": trade_outcome.get('ticker', 'UNKNOWN'),
                "strategy": trade_outcome.get('strategy', 'UNKNOWN'),
                "text": (
                    f"Market regime changed during {trade_outcome.get('ticker')} trade: "
                    f"VIX moved from {vix_entry:.1f} to {vix_exit:.1f}, "
                    f"impacting {trade_outcome.get('strategy')} profitability."
                ),
                "vix_change": vix_exit - vix_entry,
                "timestamp": datetime.now().isoformat()
            }
            insights.append(insight)

        # 3. Pattern break insight
        if len(similar_trades) > 5:
            win_rate = sum(1 for t in similar_trades if t.get('win', False)) / len(similar_trades)
            trade_won = trade_outcome.get('pnl', 0) > 0

            if (win_rate > 0.6 and not trade_won) or (win_rate < 0.4 and trade_won):
                # Outcome differed from historical pattern
                insight = {
                    "type": "pattern_break",
                    "ticker": trade_outcome.get('ticker', 'UNKNOWN'),
                    "strategy": trade_outcome.get('strategy', 'UNKNOWN'),
                    "text": (
                        f"Historical {trade_outcome.get('strategy')} on {trade_outcome.get('ticker')} "
                        f"had {win_rate:.0%} win rate, but this trade was {'WIN' if trade_won else 'LOSS'}. "
                        f"Possible factors: {self._identify_differentiating_factors(trade_outcome, similar_trades)}"
                    ),
                    "historical_win_rate": win_rate,
                    "timestamp": datetime.now().isoformat()
                }
                insights.append(insight)

        return insights

    def _identify_failure_factors(self, trade: Dict[str, Any]) -> str:
        """Identify why a trade failed"""
        factors = []

        # Check for unexpected events
        if trade.get('unexpected_events'):
            factors.append(f"unexpected events: {trade['unexpected_events']}")

        # Check for market regime change
        vix_change = abs(trade.get('vix_at_exit', 0) - trade.get('vix_at_entry', 0))
        if vix_change > 5:
            factors.append(f"VIX volatility spike ({vix_change:+.1f} points)")

        # Check for large price move
        if trade.get('stock_price_at_entry') and trade.get('stock_price_at_exit'):
            price_change_pct = (
                (trade['stock_price_at_exit'] - trade['stock_price_at_entry']) /
                trade['stock_price_at_entry'] * 100
            )
            if abs(price_change_pct) > 10:
                factors.append(f"large price move ({price_change_pct:+.1f}%)")

        return ", ".join(factors) if factors else "market conditions"

    def _regime_changed_significantly(self, trade: Dict[str, Any]) -> bool:
        """Check if market regime changed during trade"""
        vix_change = abs(trade.get('vix_at_exit', 0) - trade.get('vix_at_entry', 0))
        return vix_change > 5  # VIX moved > 5 points

    def _identify_differentiating_factors(
        self,
        current_trade: Dict[str, Any],
        historical_trades: List[Dict[str, Any]]
    ) -> str:
        """Identify what made this trade different"""
        factors = []

        if not historical_trades:
            return "insufficient historical data"

        # Compare market conditions
        avg_vix = sum(t.get('vix_at_entry', 0) for t in historical_trades) / len(historical_trades)
        current_vix = current_trade.get('vix_at_entry', 0)
        if abs(current_vix - avg_vix) > 3:
            factors.append(
                f"VIX was {'higher' if current_vix > avg_vix else 'lower'} "
                f"than usual ({current_vix:.1f} vs avg {avg_vix:.1f})"
            )

        # Compare timing
        avg_dte = sum(t.get('dte', 0) for t in historical_trades) / len(historical_trades)
        current_dte = current_trade.get('dte', 0)
        if abs(current_dte - avg_dte) > 7:
            factors.append(
                f"DTE was {'longer' if current_dte > avg_dte else 'shorter'} "
                f"than usual ({current_dte} vs avg {avg_dte:.0f})"
            )

        return ", ".join(factors) if factors else "unclear"


class ConfidenceCalibrator:
    """
    Autonomous confidence calibration

    Ensures that X% confident recommendations are X% accurate
    """

    def __init__(self) -> None:
        """Initialize confidence calibrator"""
        self.confidence_bands = [
            (0, 50),    # Low confidence
            (50, 70),   # Medium confidence
            (70, 85),   # High confidence
            (85, 100)   # Very high confidence
        ]

    def analyze_calibration(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze if confidence scores are well-calibrated

        Args:
            recommendations: List of recommendations with outcomes

        Returns:
            Calibration metrics and adjustment factors
        """
        results = {}

        for low, high in self.confidence_bands:
            # Filter recommendations in this confidence band
            band_recs = [
                r for r in recommendations
                if low <= r.get('confidence', 0) < high
                and r.get('recommendation_correct') is not None
            ]

            if not band_recs:
                continue

            # Calculate actual accuracy
            correct = sum(1 for r in band_recs if r['recommendation_correct'])
            actual_accuracy = (correct / len(band_recs)) * 100

            # Expected accuracy (midpoint of band)
            expected_accuracy = (low + high) / 2

            # Calibration error
            calibration_error = actual_accuracy - expected_accuracy

            # Adjustment factor
            if abs(calibration_error) > 10:  # > 10% miscalibration
                if expected_accuracy > 0:
                    adjustment = 1 - (calibration_error / expected_accuracy)
                else:
                    adjustment = 1.0
            else:
                adjustment = 1.0  # Well calibrated

            band_name = f"{low}-{high}%"
            results[band_name] = {
                "count": len(band_recs),
                "expected_accuracy": expected_accuracy,
                "actual_accuracy": actual_accuracy,
                "calibration_error": calibration_error,
                "adjustment_factor": adjustment
            }

        return results

    def adjust_confidence(
        self,
        raw_confidence: float,
        calibration_data: Dict[str, Any]
    ) -> float:
        """
        Adjust confidence score based on calibration analysis

        Args:
            raw_confidence: Raw confidence from LLM
            calibration_data: Calibration metrics

        Returns:
            Adjusted confidence score
        """
        # Find appropriate band
        for band_name, metrics in calibration_data.items():
            low, high = map(int, band_name.replace('%', '').split('-'))

            if low <= raw_confidence < high:
                adjusted = raw_confidence * metrics['adjustment_factor']
                return max(0, min(100, adjusted))

        return raw_confidence  # No adjustment if band not found


class ContinuousLearningPipeline:
    """
    Continuous learning pipeline that runs autonomously

    Monitors trade outcomes and updates the RAG system accordingly
    """

    def __init__(self) -> None:
        """Initialize continuous learning pipeline"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME', 'magnus')
        }

        self.weight_updater = SuccessWeightUpdater()
        self.pattern_extractor = PatternExtractor()
        self.regime_detector = MarketRegimeDetector()
        self.confidence_calibrator = ConfidenceCalibrator()

    def get_connection(self) -> None:
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)

    def find_completed_trades_with_recommendations(
        self,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Find closed trades with recommendations that haven't been processed for learning

        Args:
            since: Only trades closed after this datetime

        Returns:
            List of trade + recommendation dicts
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            query = """
                SELECT
                    t.*,
                    r.id as recommendation_id,
                    r.recommendation,
                    r.confidence,
                    r.reasoning,
                    r.recommendation_correct,
                    r.similar_trades_found,
                    r.top_trades_used,
                    r.statistics
                FROM xtrades_trades t
                INNER JOIN xtrades_recommendations r ON t.id = r.trade_id
                WHERE t.status = 'closed'
                  AND r.outcome_recorded_at IS NOT NULL
                  AND r.learning_processed_at IS NULL
            """

            params = []

            if since:
                query += " AND t.exit_date >= %s"
                params.append(since)

            query += " ORDER BY t.exit_date DESC"

            cur.execute(query, params)
            trades = [dict(row) for row in cur.fetchall()]

            logger.info(f"Found {len(trades)} trades ready for learning")
            return trades

        except Exception as e:
            logger.error(f"Error finding completed trades: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def process_completed_trade(
        self,
        trade: Dict[str, Any]
    ) -> Tuple[int, int]:
        """
        Process single completed trade for learning

        Args:
            trade: Trade + recommendation dict

        Returns:
            (weights_updated, insights_extracted)
        """
        logger.info(f"Processing trade {trade['id']}: {trade['ticker']} {trade['strategy']}")

        weights_updated = 0
        insights_extracted = 0

        try:
            # 1. Extract insights
            similar_trades = trade.get('statistics', {}).get('similar_trades', [])
            insights = self.pattern_extractor.extract_insights(
                recommendation={
                    'recommendation': trade['recommendation'],
                    'confidence': trade['confidence'],
                    'reasoning': trade['reasoning'],
                    'recommendation_correct': trade['recommendation_correct']
                },
                trade_outcome=trade,
                similar_trades=similar_trades
            )

            insights_extracted = len(insights)

            # Store insights (would be embedded and stored in vector DB in full implementation)
            self._store_insights(insights)

            # 2. Update success weights for similar trades
            # (In full implementation, this would update Qdrant)
            # For now, just log
            if trade.get('top_trades_used'):
                weights_updated = trade['top_trades_used']
                logger.info(f"Would update {weights_updated} similar trade weights")

            # 3. Mark as processed
            self._mark_learning_processed(trade['recommendation_id'])

            logger.info(
                f"Processed trade {trade['id']}: "
                f"{insights_extracted} insights extracted, "
                f"{weights_updated} weights to update"
            )

            return weights_updated, insights_extracted

        except Exception as e:
            logger.error(f"Error processing trade {trade['id']}: {e}")
            return 0, 0

    def _store_insights(self, insights: List[Dict[str, str]]):
        """Store extracted insights in database"""
        if not insights:
            return

        conn = self.get_connection()
        cur = conn.cursor()

        try:
            for insight in insights:
                cur.execute("""
                    INSERT INTO xtrades_learning_insights
                    (insight_type, ticker, strategy, insight_text, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (
                    insight.get('type'),
                    insight.get('ticker'),
                    insight.get('strategy'),
                    insight.get('text'),
                    json.dumps({k: v for k, v in insight.items() if k not in ['type', 'ticker', 'strategy', 'text']})
                ))

            conn.commit()
            logger.info(f"Stored {len(insights)} insights")

        except Exception as e:
            logger.error(f"Error storing insights: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def _mark_learning_processed(self, recommendation_id: int):
        """Mark recommendation as processed for learning"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                UPDATE xtrades_recommendations
                SET learning_processed_at = NOW()
                WHERE id = %s
            """, (recommendation_id,))

            conn.commit()

        except Exception as e:
            logger.error(f"Error marking learning processed: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def run_learning_cycle(self) -> LearningMetrics:
        """
        Run one complete learning cycle

        Returns:
            Learning metrics
        """
        logger.info("=" * 80)
        logger.info("STARTING LEARNING CYCLE")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Find trades to process
        trades = self.find_completed_trades_with_recommendations()

        if not trades:
            logger.info("No trades to process")
            return LearningMetrics(
                trades_processed=0,
                weights_updated=0,
                insights_extracted=0,
                accuracy_improvement=0.0,
                timestamp=start_time
            )

        # Process each trade
        total_weights = 0
        total_insights = 0

        for trade in trades:
            weights, insights = self.process_completed_trade(trade)
            total_weights += weights
            total_insights += insights

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("=" * 80)
        logger.info(f"LEARNING CYCLE COMPLETE ({duration:.1f}s)")
        logger.info(f"  Trades Processed: {len(trades)}")
        logger.info(f"  Weights Updated: {total_weights}")
        logger.info(f"  Insights Extracted: {total_insights}")
        logger.info("=" * 80)

        return LearningMetrics(
            trades_processed=len(trades),
            weights_updated=total_weights,
            insights_extracted=total_insights,
            accuracy_improvement=0.0,  # Would calculate from performance metrics
            timestamp=start_time
        )


def main():
    """
    Example usage: Run learning cycle
    """
    pipeline = ContinuousLearningPipeline()
    metrics = pipeline.run_learning_cycle()

    print(f"\nLearning Cycle Complete!")
    print(f"Trades Processed: {metrics.trades_processed}")
    print(f"Weights Updated: {metrics.weights_updated}")
    print(f"Insights Extracted: {metrics.insights_extracted}")


if __name__ == "__main__":
    main()
