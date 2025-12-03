"""
Recommendation Tracker - Store and analyze RAG recommendations

This module handles:
1. Storing recommendations in PostgreSQL
2. Tracking trade outcomes
3. Updating learning weights
4. Generating performance metrics

Author: Magnus Wheel Strategy Dashboard
Created: 2025-11-06
"""

import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RecommendationTracker:
    """
    Tracks RAG recommendations and learns from outcomes
    """

    def __init__(self) -> None:
        """Initialize recommendation tracker"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME', 'magnus')
        }

    def get_connection(self) -> None:
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)

    def store_recommendation(
        self,
        trade_id: int,
        recommendation: Dict[str, Any],
        query_latency_ms: Optional[int] = None
    ) -> int:
        """
        Store RAG recommendation in database

        Args:
            trade_id: Trade ID this recommendation is for
            recommendation: Recommendation dictionary from RAG engine
            query_latency_ms: How long RAG query took

        Returns:
            recommendation_id
        """
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO xtrades_recommendations (
                    trade_id,
                    recommendation,
                    confidence,
                    reasoning,
                    historical_evidence,
                    risk_factors,
                    suggested_adjustments,
                    similar_trades_found,
                    top_trades_used,
                    statistics,
                    query_latency_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                trade_id,
                recommendation.get('recommendation'),
                recommendation.get('confidence'),
                recommendation.get('reasoning'),
                Json(recommendation.get('historical_evidence', [])),
                Json(recommendation.get('risk_factors', [])),
                recommendation.get('suggested_adjustments'),
                recommendation.get('similar_trades_found', 0),
                recommendation.get('top_trades_used', 0),
                Json(recommendation.get('statistics', {})),
                query_latency_ms
            ))

            rec_id = cur.fetchone()[0]
            conn.commit()

            logger.info(f"Stored recommendation {rec_id} for trade {trade_id}")
            return rec_id

        except Exception as e:
            logger.error(f"Error storing recommendation: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def update_outcome(
        self,
        recommendation_id: int,
        trade: Dict[str, Any]
    ) -> bool:
        """
        Update recommendation with actual trade outcome

        Args:
            recommendation_id: Recommendation ID
            trade: Trade dictionary with outcome data

        Returns:
            True if updated successfully
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get original recommendation
            cur.execute("""
                SELECT recommendation, confidence
                FROM xtrades_recommendations
                WHERE id = %s
            """, (recommendation_id,))

            rec = cur.fetchone()
            if not rec:
                logger.error(f"Recommendation {recommendation_id} not found")
                return False

            # Determine actual outcome
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                actual_outcome = 'WIN'
            elif pnl < 0:
                actual_outcome = 'LOSS'
            else:
                actual_outcome = 'BREAK_EVEN'

            # Check if recommendation was correct
            rec_correct = self._is_recommendation_correct(
                rec['recommendation'],
                actual_outcome,
                pnl
            )

            # Calculate hold days
            entry_date = trade.get('entry_date')
            exit_date = trade.get('exit_date')
            hold_days = None
            if entry_date and exit_date:
                hold_days = (exit_date - entry_date).days

            # Update recommendation
            cur.execute("""
                UPDATE xtrades_recommendations
                SET actual_outcome = %s,
                    actual_pnl = %s,
                    actual_pnl_percent = %s,
                    actual_hold_days = %s,
                    recommendation_correct = %s,
                    outcome_recorded_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
            """, (
                actual_outcome,
                pnl,
                trade.get('pnl_percent'),
                hold_days,
                rec_correct,
                recommendation_id
            ))

            conn.commit()

            logger.info(f"Updated outcome for recommendation {recommendation_id}: {actual_outcome} (Correct: {rec_correct})")

            # Update learning weights
            self._update_learning_weights(recommendation_id, conn)

            return True

        except Exception as e:
            logger.error(f"Error updating outcome: {e}")
            conn.rollback()
            return False
        finally:
            cur.close()
            conn.close()

    def _is_recommendation_correct(
        self,
        recommendation: str,
        actual_outcome: str,
        pnl: float
    ) -> bool:
        """
        Determine if recommendation was correct

        Args:
            recommendation: 'TAKE', 'PASS', or 'MONITOR'
            actual_outcome: 'WIN', 'LOSS', or 'BREAK_EVEN'
            pnl: Actual P&L

        Returns:
            True if recommendation was correct
        """
        if recommendation == 'TAKE':
            # Correct if trade was profitable
            return pnl > 0

        elif recommendation == 'PASS':
            # Correct if trade would have lost money
            return pnl <= 0

        elif recommendation == 'MONITOR':
            # Monitor is neutral - always correct
            return True

        return False

    def _update_learning_weights(
        self,
        recommendation_id: int,
        conn
    ) -> None:
        """
        Update learning weights for trades used in recommendation

        Args:
            recommendation_id: Recommendation ID
            conn: Database connection
        """
        try:
            # Call PostgreSQL function
            cur = conn.cursor()
            cur.execute("""
                SELECT update_rag_learning_weights(%s)
            """, (recommendation_id,))
            conn.commit()
            cur.close()

            logger.info(f"Updated learning weights for recommendation {recommendation_id}")

        except Exception as e:
            logger.error(f"Error updating learning weights: {e}")

    def get_recommendation_by_trade_id(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Get recommendation for a specific trade

        Args:
            trade_id: Trade ID

        Returns:
            Recommendation dictionary or None
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM xtrades_recommendations
                WHERE trade_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (trade_id,))

            rec = cur.fetchone()
            return dict(rec) if rec else None

        except Exception as e:
            logger.error(f"Error fetching recommendation: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get RAG system performance metrics

        Args:
            days: Number of days to analyze

        Returns:
            Performance metrics dictionary
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Calculate and store performance
            cur.execute("""
                SELECT calculate_rag_performance(%s)
            """, (days,))
            performance_id = cur.fetchone()[0]
            conn.commit()

            # Fetch calculated performance
            cur.execute("""
                SELECT * FROM xtrades_rag_performance
                WHERE id = %s
            """, (performance_id,))

            performance = dict(cur.fetchone())

            logger.info(f"Calculated performance metrics for last {days} days")
            return performance

        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
        finally:
            cur.close()
            conn.close()

    def get_accuracy_by_recommendation(self) -> List[Dict[str, Any]]:
        """
        Get accuracy breakdown by recommendation type

        Returns:
            List of accuracy metrics
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM v_rag_accuracy_by_recommendation
            """)

            results = [dict(row) for row in cur.fetchall()]
            return results

        except Exception as e:
            logger.error(f"Error fetching accuracy metrics: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def get_confidence_calibration(self) -> List[Dict[str, Any]]:
        """
        Get confidence calibration analysis

        Returns:
            List of calibration metrics
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM v_rag_confidence_calibration
            """)

            results = [dict(row) for row in cur.fetchall()]
            return results

        except Exception as e:
            logger.error(f"Error fetching calibration metrics: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def get_top_learning_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get top performing trades for learning

        Args:
            limit: Number of trades to return

        Returns:
            List of top trades
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute(f"""
                SELECT * FROM v_rag_top_learning_trades
                LIMIT {limit}
            """)

            results = [dict(row) for row in cur.fetchall()]
            return results

        except Exception as e:
            logger.error(f"Error fetching top learning trades: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def print_performance_report(self, days: int = 30) -> None:
        """
        Print comprehensive performance report

        Args:
            days: Number of days to analyze
        """
        print("\n" + "="*80)
        print(f"RAG SYSTEM PERFORMANCE REPORT (Last {days} Days)")
        print("="*80)

        # Overall metrics
        metrics = self.get_performance_metrics(days)

        print("\nOVERALL METRICS")
        print("-"*80)
        print(f"Total Recommendations: {metrics.get('total_recommendations', 0)}")
        print(f"Recommendations with Outcomes: {metrics.get('recommendations_with_outcomes', 0)}")
        print(f"Overall Accuracy: {metrics.get('overall_accuracy', 0):.1f}%")
        print(f"Total P&L Impact: ${metrics.get('total_pnl_impact', 0):,.2f}")

        # By recommendation type
        print("\nBY RECOMMENDATION TYPE")
        print("-"*80)
        accuracy = self.get_accuracy_by_recommendation()
        for rec in accuracy:
            print(f"{rec['recommendation']:8s}: {rec['total']:3d} recs, "
                  f"{rec['accuracy_pct']:.1f}% accurate, "
                  f"avg confidence: {rec['avg_confidence']:.0f}%, "
                  f"avg P&L: ${rec['avg_pnl']:+,.2f}")

        # Confidence calibration
        print("\nCONFIDENCE CALIBRATION")
        print("-"*80)
        calibration = self.get_confidence_calibration()
        for band in calibration:
            print(f"{band['confidence_band']:20s}: {band['total']:3d} recs, "
                  f"{band['actual_accuracy']:.1f}% accurate, "
                  f"avg conf: {band['avg_confidence']:.0f}%")

        # Top learning trades
        print("\nTOP 10 LEARNING TRADES")
        print("-"*80)
        top_trades = self.get_top_learning_trades(limit=10)
        for i, trade in enumerate(top_trades, 1):
            print(f"{i:2d}. {trade['ticker']:5s} {trade['strategy']:15s} - "
                  f"P&L: ${trade['pnl']:+7.2f} ({trade['pnl_percent']:+5.1f}%) - "
                  f"Weight: {trade['success_weight']:.2f}, "
                  f"Accuracy: {trade['accuracy_rate']:.0f}%, "
                  f"Used: {trade['times_referenced']}x")

        # False positives/negatives
        print("\nERROR ANALYSIS")
        print("-"*80)
        print(f"False Positives (TAKE but lost): {metrics.get('false_positives', 0)}")
        print(f"False Negatives (PASS but won): {metrics.get('false_negatives', 0)}")

        print("\n" + "="*80)


def main():
    """
    Example usage: Print performance report
    """
    tracker = RecommendationTracker()
    tracker.print_performance_report(days=30)


if __name__ == "__main__":
    main()
