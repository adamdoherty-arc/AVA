"""
Outcome Tracking & Learning System
===================================

Track every recommendation AVA makes and learn from outcomes:
- Win rate tracking
- P&L attribution
- Strategy effectiveness
- Continuous improvement
- Performance dashboard

AVA gets smarter over time!

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OutcomeTracker:
    """Track and learn from every recommendation"""

    def __init__(self) -> None:
        """Initialize outcome tracker"""
        self.db_file = Path.home() / '.ava_outcomes.json'
        self.recommendations = self._load_database()

    def _load_database(self) -> Dict:
        """Load outcomes database"""
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r') as f:
                    return json.load(f)
            except:
                return {'recommendations': [], 'stats': {}}
        return {'recommendations': [], 'stats': {}}

    def _save_database(self) -> None:
        """Save outcomes database"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.recommendations, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving outcomes database: {e}")

    def log_recommendation(
        self,
        ticker: str,
        action: str,
        details: Dict,
        recommendation: str,
        confidence: float
    ) -> int:
        """
        Log a new recommendation.

        Args:
            ticker: Stock ticker
            action: 'BUY', 'SELL', 'HOLD', 'SELL_CSP', etc.
            details: Trade details (strike, expiration, premium, etc.)
            recommendation: AVA's recommendation text
            confidence: Confidence level (0-1)

        Returns:
            Recommendation ID
        """
        rec_id = len(self.recommendations['recommendations']) + 1

        recommendation_record = {
            'id': rec_id,
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'action': action,
            'details': details,
            'recommendation': recommendation,
            'confidence': confidence,
            'outcome': None,  # Filled in later
            'outcome_tracked': False
        }

        self.recommendations['recommendations'].append(recommendation_record)
        self._save_database()

        logger.info(f"✅ Logged recommendation #{rec_id} for {ticker}")
        return rec_id

    def track_outcome(self, rec_id: int, outcome: Dict):
        """
        Track the outcome of a recommendation.

        Args:
            rec_id: Recommendation ID
            outcome: Outcome dict with:
                - success: bool
                - profit: float
                - return_pct: float
                - notes: str
        """
        for rec in self.recommendations['recommendations']:
            if rec['id'] == rec_id:
                rec['outcome'] = outcome
                rec['outcome_tracked'] = True
                rec['outcome_timestamp'] = datetime.now().isoformat()

                self._save_database()
                self._update_statistics()

                logger.info(f"✅ Tracked outcome for recommendation #{rec_id}")
                return

        logger.warning(f"Recommendation #{rec_id} not found")

    def _update_statistics(self) -> None:
        """Update overall statistics"""
        tracked = [r for r in self.recommendations['recommendations'] if r['outcome_tracked']]

        if not tracked:
            return

        # Calculate stats
        total_tracked = len(tracked)
        wins = sum(1 for r in tracked if r['outcome'].get('success', False))
        win_rate = wins / total_tracked if total_tracked > 0 else 0

        total_profit = sum(r['outcome'].get('profit', 0) for r in tracked)
        avg_profit = total_profit / total_tracked if total_tracked > 0 else 0

        self.recommendations['stats'] = {
            'total_recommendations': len(self.recommendations['recommendations']),
            'total_tracked': total_tracked,
            'wins': wins,
            'losses': total_tracked - wins,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'last_updated': datetime.now().isoformat()
        }

        self._save_database()

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        return self.recommendations.get('stats', {})

    def get_recent_outcomes(self, limit: int = 10) -> List[Dict]:
        """Get recent tracked outcomes"""
        tracked = [r for r in self.recommendations['recommendations'] if r['outcome_tracked']]
        tracked.sort(key=lambda x: x['outcome_timestamp'], reverse=True)
        return tracked[:limit]

    def get_ytd_performance(self) -> Optional[Dict]:
        """Get year-to-date performance"""
        year_start = datetime(datetime.now().year, 1, 1)

        ytd_recs = [
            r for r in self.recommendations['recommendations']
            if r['outcome_tracked'] and datetime.fromisoformat(r['timestamp']) >= year_start
        ]

        if not ytd_recs:
            return None

        total_profit = sum(r['outcome'].get('profit', 0) for r in ytd_recs)
        wins = sum(1 for r in ytd_recs if r['outcome'].get('success', False))

        return {
            'total_trades': len(ytd_recs),
            'wins': wins,
            'win_rate': wins / len(ytd_recs) if ytd_recs else 0,
            'total_profit': total_profit,
            'avg_profit_per_trade': total_profit / len(ytd_recs) if ytd_recs else 0,
            'return_pct': 0,  # Would calculate from portfolio value
            'sp500_return': 0  # Would fetch from market data
        }
