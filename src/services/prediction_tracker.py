"""
Prediction Accuracy Tracking Service
Tracks AI prediction outcomes and calculates performance metrics
"""

import logging
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import os

logger = logging.getLogger(__name__)


class PredictionTracker:
    """
    Tracks prediction accuracy and calculates performance metrics.
    Enables model calibration and ROI analysis.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/magnus"
        )

    def _get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    def record_prediction(
        self,
        game_id: str,
        sport: str,
        predicted_winner: str,
        predicted_probability: float,
        predicted_spread: Optional[float] = None,
        predicted_total: Optional[float] = None,
        model_version: str = "v1.0"
    ) -> str:
        """
        Record a new prediction before a game starts.

        Returns:
            prediction_id for tracking
        """
        prediction_id = f"pred_{sport.lower()}_{game_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Determine confidence tier
        if predicted_probability >= 0.70:
            confidence_tier = "high"
        elif predicted_probability >= 0.55:
            confidence_tier = "medium"
        else:
            confidence_tier = "low"

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO prediction_results (
                        prediction_id, game_id, sport,
                        predicted_winner, predicted_probability,
                        predicted_spread, predicted_total,
                        confidence_tier, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (prediction_id) DO UPDATE SET
                        predicted_winner = EXCLUDED.predicted_winner,
                        predicted_probability = EXCLUDED.predicted_probability,
                        confidence_tier = EXCLUDED.confidence_tier
                    RETURNING prediction_id
                """, (
                    prediction_id, game_id, sport,
                    predicted_winner, predicted_probability,
                    predicted_spread, predicted_total,
                    confidence_tier, model_version
                ))
                conn.commit()
                logger.info(f"Recorded prediction {prediction_id} for {game_id}")
                return prediction_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record prediction: {e}")
            raise
        finally:
            conn.close()

    def settle_prediction(
        self,
        game_id: str,
        actual_winner: str,
        home_score: int,
        away_score: int
    ) -> Dict:
        """
        Settle a prediction after a game completes.

        Returns:
            Settlement result with accuracy info
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Get the prediction for this game
                cur.execute("""
                    SELECT * FROM prediction_results
                    WHERE game_id = %s AND was_correct IS NULL
                    ORDER BY prediction_timestamp DESC
                    LIMIT 1
                """, (game_id,))
                prediction = cur.fetchone()

                if not prediction:
                    logger.warning(f"No unsettled prediction found for game {game_id}")
                    return {"status": "no_prediction"}

                # Determine if prediction was correct
                was_correct = (
                    prediction['predicted_winner'].lower() == actual_winner.lower()
                )

                # Update the prediction
                cur.execute("""
                    UPDATE prediction_results SET
                        actual_winner = %s,
                        actual_home_score = %s,
                        actual_away_score = %s,
                        was_correct = %s,
                        game_completed_at = NOW()
                    WHERE prediction_id = %s
                """, (
                    actual_winner, home_score, away_score,
                    was_correct, prediction['prediction_id']
                ))

                conn.commit()

                result = {
                    "prediction_id": prediction['prediction_id'],
                    "predicted_winner": prediction['predicted_winner'],
                    "predicted_probability": float(prediction['predicted_probability']),
                    "actual_winner": actual_winner,
                    "was_correct": was_correct,
                    "confidence_tier": prediction['confidence_tier']
                }

                logger.info(
                    f"Settled prediction {prediction['prediction_id']}: "
                    f"{'CORRECT' if was_correct else 'WRONG'}"
                )

                return result

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to settle prediction: {e}")
            raise
        finally:
            conn.close()

    def get_accuracy_metrics(
        self,
        sport: Optional[str] = None,
        days: int = 30,
        model_version: Optional[str] = None
    ) -> Dict:
        """
        Get accuracy metrics for predictions.

        Returns:
            Dictionary with accuracy statistics
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Build query with filters
                where_clauses = ["was_correct IS NOT NULL"]
                params = []

                if sport:
                    where_clauses.append("sport = %s")
                    params.append(sport)

                if model_version:
                    where_clauses.append("model_version = %s")
                    params.append(model_version)

                where_clauses.append("game_completed_at >= NOW() - INTERVAL '%s days'")
                params.append(days)

                where_sql = " AND ".join(where_clauses)

                # Overall accuracy
                cur.execute(f"""
                    SELECT
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                        ROUND(AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END), 4) as accuracy,
                        ROUND(AVG(predicted_probability), 4) as avg_confidence
                    FROM prediction_results
                    WHERE {where_sql}
                """, params)
                overall = cur.fetchone()

                # By confidence tier
                cur.execute(f"""
                    SELECT
                        confidence_tier,
                        COUNT(*) as total,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                        ROUND(AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END), 4) as accuracy
                    FROM prediction_results
                    WHERE {where_sql}
                    GROUP BY confidence_tier
                    ORDER BY confidence_tier
                """, params)
                by_tier = {row['confidence_tier']: dict(row) for row in cur.fetchall()}

                # Calculate Brier score (calibration metric)
                cur.execute(f"""
                    SELECT
                        ROUND(AVG(POWER(
                            predicted_probability - CASE WHEN was_correct THEN 1.0 ELSE 0.0 END,
                            2
                        )), 5) as brier_score
                    FROM prediction_results
                    WHERE {where_sql}
                """, params)
                brier = cur.fetchone()

                # Recent trend (last 7 days vs previous 7 days)
                cur.execute("""
                    SELECT
                        AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END)
                            FILTER (WHERE game_completed_at >= NOW() - INTERVAL '7 days') as recent_accuracy,
                        AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END)
                            FILTER (WHERE game_completed_at >= NOW() - INTERVAL '14 days'
                                    AND game_completed_at < NOW() - INTERVAL '7 days') as prev_accuracy
                    FROM prediction_results
                    WHERE was_correct IS NOT NULL
                """)
                trend = cur.fetchone()

                return {
                    "period_days": days,
                    "sport": sport or "all",
                    "overall": {
                        "total": overall['total_predictions'] or 0,
                        "correct": overall['correct'] or 0,
                        "accuracy": float(overall['accuracy'] or 0),
                        "avg_confidence": float(overall['avg_confidence'] or 0)
                    },
                    "by_confidence_tier": by_tier,
                    "calibration": {
                        "brier_score": float(brier['brier_score'] or 0),
                        "interpretation": self._interpret_brier(float(brier['brier_score'] or 0))
                    },
                    "trend": {
                        "last_7_days": float(trend['recent_accuracy'] or 0),
                        "previous_7_days": float(trend['prev_accuracy'] or 0),
                        "direction": "improving" if (trend['recent_accuracy'] or 0) > (trend['prev_accuracy'] or 0) else "declining"
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get accuracy metrics: {e}")
            return {"error": str(e)}
        finally:
            conn.close()

    def _interpret_brier(self, brier_score: float) -> str:
        """Interpret Brier score (0 = perfect, 0.25 = random)"""
        if brier_score < 0.1:
            return "Excellent calibration"
        elif brier_score < 0.15:
            return "Good calibration"
        elif brier_score < 0.2:
            return "Fair calibration"
        elif brier_score < 0.25:
            return "Needs improvement"
        else:
            return "Poor - worse than random"

    def calculate_theoretical_roi(
        self,
        sport: Optional[str] = None,
        days: int = 30,
        bet_amount: float = 100.0
    ) -> Dict:
        """
        Calculate theoretical ROI if betting on all predictions.

        Assumes standard -110 odds for simplicity.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                where_clauses = ["was_correct IS NOT NULL"]
                params = []

                if sport:
                    where_clauses.append("sport = %s")
                    params.append(sport)

                where_clauses.append("game_completed_at >= NOW() - INTERVAL '%s days'")
                params.append(days)

                where_sql = " AND ".join(where_clauses)

                cur.execute(f"""
                    SELECT
                        COUNT(*) as total_bets,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN NOT was_correct THEN 1 ELSE 0 END) as losses
                    FROM prediction_results
                    WHERE {where_sql}
                """, params)
                results = cur.fetchone()

                if not results or results['total_bets'] == 0:
                    return {"error": "No settled predictions found"}

                # Standard -110 odds: win $90.91 on $100, lose $100
                wins = results['wins'] or 0
                losses = results['losses'] or 0
                total = results['total_bets']

                win_payout = wins * (bet_amount * 0.9091)  # Win amount at -110
                loss_amount = losses * bet_amount
                total_wagered = total * bet_amount

                profit = win_payout - loss_amount
                roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0

                # High confidence picks only
                cur.execute(f"""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as wins
                    FROM prediction_results
                    WHERE {where_sql} AND confidence_tier = 'high'
                """, params)
                high_conf = cur.fetchone()

                hc_wins = high_conf['wins'] or 0
                hc_total = high_conf['total'] or 0
                hc_losses = hc_total - hc_wins
                hc_profit = (hc_wins * bet_amount * 0.9091) - (hc_losses * bet_amount)
                hc_wagered = hc_total * bet_amount
                hc_roi = (hc_profit / hc_wagered) * 100 if hc_wagered > 0 else 0

                return {
                    "period_days": days,
                    "bet_amount": bet_amount,
                    "all_picks": {
                        "total_bets": total,
                        "wins": wins,
                        "losses": losses,
                        "win_rate": wins / total if total > 0 else 0,
                        "profit": round(profit, 2),
                        "roi_percent": round(roi, 2)
                    },
                    "high_confidence_only": {
                        "total_bets": hc_total,
                        "wins": hc_wins,
                        "losses": hc_losses,
                        "win_rate": hc_wins / hc_total if hc_total > 0 else 0,
                        "profit": round(hc_profit, 2),
                        "roi_percent": round(hc_roi, 2)
                    }
                }

        except Exception as e:
            logger.error(f"Failed to calculate ROI: {e}")
            return {"error": str(e)}
        finally:
            conn.close()

    def record_odds_snapshot(
        self,
        game_id: str,
        sport: str,
        source: str,
        home_odds: int,
        away_odds: int,
        spread: Optional[float] = None,
        total: Optional[float] = None
    ) -> bool:
        """Record a point-in-time odds snapshot for movement tracking"""
        conn = self._get_connection()
        try:
            # Calculate implied probabilities from American odds
            home_prob = self._american_to_implied(home_odds)
            away_prob = self._american_to_implied(away_odds)

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO odds_history (
                        game_id, sport, source,
                        home_odds, away_odds,
                        spread, total,
                        home_implied_prob, away_implied_prob
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    game_id, sport, source,
                    home_odds, away_odds,
                    spread, total,
                    home_prob, away_prob
                ))
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record odds snapshot: {e}")
            return False
        finally:
            conn.close()

    def _american_to_implied(self, american_odds: int) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def get_odds_movement(
        self,
        game_id: str,
        hours: int = 24
    ) -> List[Dict]:
        """Get odds movement history for a game"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        home_odds, away_odds,
                        spread, total,
                        home_implied_prob, away_implied_prob,
                        source, recorded_at
                    FROM odds_history
                    WHERE game_id = %s
                      AND recorded_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY recorded_at ASC
                """, (game_id, hours))

                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get odds movement: {e}")
            return []
        finally:
            conn.close()

    def get_calibration_chart_data(
        self,
        sport: Optional[str] = None,
        days: int = 90
    ) -> List[Dict]:
        """
        Get data for calibration chart visualization.
        Groups predictions by confidence bucket and shows actual win rate.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                where_clause = "was_correct IS NOT NULL"
                params = []

                if sport:
                    where_clause += " AND sport = %s"
                    params.append(sport)

                where_clause += " AND game_completed_at >= NOW() - INTERVAL '%s days'"
                params.append(days)

                cur.execute(f"""
                    SELECT
                        FLOOR(predicted_probability * 10) / 10 as confidence_bucket,
                        COUNT(*) as total,
                        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                        ROUND(AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END), 4) as actual_rate
                    FROM prediction_results
                    WHERE {where_clause}
                    GROUP BY FLOOR(predicted_probability * 10)
                    ORDER BY confidence_bucket
                """, params)

                return [
                    {
                        "predicted_range": f"{int(row['confidence_bucket']*100)}-{int((row['confidence_bucket']+0.1)*100)}%",
                        "predicted_midpoint": float(row['confidence_bucket']) + 0.05,
                        "actual_win_rate": float(row['actual_rate']),
                        "total_predictions": row['total'],
                        "correct_predictions": row['correct']
                    }
                    for row in cur.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to get calibration data: {e}")
            return []
        finally:
            conn.close()


# Singleton instance
_tracker = None


def get_prediction_tracker() -> PredictionTracker:
    """Get prediction tracker singleton"""
    global _tracker
    if _tracker is None:
        _tracker = PredictionTracker()
    return _tracker


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tracker = get_prediction_tracker()

    # Test recording a prediction
    pred_id = tracker.record_prediction(
        game_id="test_game_123",
        sport="NFL",
        predicted_winner="Kansas City Chiefs",
        predicted_probability=0.72,
        predicted_spread=-3.5,
        model_version="v1.0"
    )
    print(f"Recorded prediction: {pred_id}")

    # Test settling the prediction
    result = tracker.settle_prediction(
        game_id="test_game_123",
        actual_winner="Kansas City Chiefs",
        home_score=31,
        away_score=27
    )
    print(f"Settlement result: {result}")

    # Test getting metrics
    metrics = tracker.get_accuracy_metrics(sport="NFL", days=30)
    print(f"Accuracy metrics: {metrics}")

    # Test ROI calculation
    roi = tracker.calculate_theoretical_roi(sport="NFL", days=30)
    print(f"ROI analysis: {roi}")

    # Test calibration data
    calibration = tracker.get_calibration_chart_data(sport="NFL", days=90)
    print(f"Calibration data: {calibration}")
