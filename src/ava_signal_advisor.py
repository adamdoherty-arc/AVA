"""
AVA Signal Advisor
High-level interface for AVA to query the RAG system and get trade recommendations
"""
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from src.signal_performance_tracker import SignalPerformanceTracker
from src.signal_vector_search import SignalVectorSearch


class AVASignalAdvisor:
    """AVA's interface to the trading signal RAG system"""

    def __init__(self) -> None:
        self.db_password = os.getenv('DB_PASSWORD')
        self.performance_tracker = SignalPerformanceTracker()
        self.vector_search = SignalVectorSearch()

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(
            host='localhost',
            port='5432',
            database='magnus',
            user='postgres',
            password=self.db_password
        )

    def get_top_recommendations(
        self,
        hours_back: int = 168,
        limit: int = 5,
        min_composite_score: float = 60.0
    ) -> List[Dict]:
        """
        Get AVA's top trade recommendations based on multi-factor analysis

        Returns signals ranked by composite quality score considering:
        - Author historical win rate (40% weight)
        - Setup success rate for ticker (30% weight)
        - Signal completeness (20% weight)
        - Similar past winning trades (10% weight)
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT
                s.id,
                s.primary_ticker,
                s.setup_type,
                s.sentiment,
                s.entry,
                s.target,
                s.stop_loss,
                s.option_strike,
                s.option_type,
                s.option_expiration,
                s.author,
                s.timestamp,
                s.content,
                s.confidence as base_confidence,
                sq.composite_score,
                sq.rank,
                sq.recommendation,
                sq.reasoning,
                sq.author_credibility,
                sq.setup_success_rate,
                a.win_rate as author_win_rate,
                a.total_pnl_dollars as author_total_pnl,
                a.wins as author_wins,
                a.losses as author_losses,
                sp.win_rate as setup_win_rate,
                sp.avg_pnl_percent as setup_avg_pnl
            FROM discord_trading_signals s
            JOIN signal_quality_scores sq ON s.id = sq.signal_id
            LEFT JOIN author_performance a ON s.author = a.author
            LEFT JOIN setup_performance sp
                ON s.primary_ticker = sp.ticker
                AND s.setup_type = sp.setup_type
            WHERE s.timestamp >= NOW() - INTERVAL '%s hours'
            AND sq.composite_score >= %s
            AND sq.recommendation IN ('strong_buy', 'buy')
            ORDER BY sq.composite_score DESC
            LIMIT %s
        """, (hours_back, min_composite_score, limit))

        signals = cur.fetchall()
        cur.close()
        conn.close()

        # Enrich with similar trade analysis
        enriched = []
        for signal in signals:
            signal_dict = dict(signal)

            # Find similar winning trades
            similar_wins = self.vector_search.find_similar_signals(
                signal['id'],
                n_results=5,
                only_winners=True
            )

            signal_dict['similar_winning_trades'] = similar_wins

            # Calculate risk/reward ratio if prices available
            if signal['entry'] and signal['target'] and signal['stop_loss']:
                entry = float(signal['entry'])
                target = float(signal['target'])
                stop = float(signal['stop_loss'])

                reward = target - entry
                risk = entry - stop

                if risk > 0:
                    signal_dict['risk_reward_ratio'] = reward / risk
                else:
                    signal_dict['risk_reward_ratio'] = None
            else:
                signal_dict['risk_reward_ratio'] = None

            enriched.append(signal_dict)

        return enriched

    def analyze_signal_with_context(self, signal_id: int) -> Dict:
        """
        Perform deep analysis on a specific signal with full historical context

        Returns comprehensive analysis including:
        - Signal details
        - Author track record
        - Setup performance history
        - Similar past trades and their outcomes
        - Risk assessment
        - Final recommendation with reasoning
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get signal with all related data
        cur.execute("""
            SELECT
                s.*,
                sq.composite_score,
                sq.recommendation,
                sq.reasoning,
                sq.author_credibility,
                sq.setup_success_rate,
                a.win_rate as author_win_rate,
                a.wins as author_wins,
                a.losses as author_losses,
                a.total_pnl_dollars as author_total_pnl,
                sp.win_rate as setup_win_rate,
                sp.wins as setup_wins,
                sp.losses as setup_losses,
                sp.avg_pnl_percent as setup_avg_pnl
            FROM discord_trading_signals s
            LEFT JOIN signal_quality_scores sq ON s.id = sq.signal_id
            LEFT JOIN author_performance a ON s.author = a.author
            LEFT JOIN setup_performance sp
                ON s.primary_ticker = sp.ticker
                AND s.setup_type = sp.setup_type
            WHERE s.id = %s
        """, (signal_id,))

        signal = cur.fetchone()
        cur.close()
        conn.close()

        if not signal:
            return {'error': 'Signal not found'}

        analysis = dict(signal)

        # Similar trades
        similar_all = self.vector_search.find_similar_signals(signal_id, n_results=10)
        similar_wins = [s for s in similar_all if s.get('outcome') == 'win']
        similar_losses = [s for s in similar_all if s.get('outcome') == 'loss']

        analysis['similar_trades'] = {
            'total': len(similar_all),
            'wins': len(similar_wins),
            'losses': len(similar_losses),
            'win_rate': (len(similar_wins) / len(similar_all) * 100) if similar_all else 0,
            'top_similar': similar_all[:5]
        }

        # Risk/Reward
        if signal['entry'] and signal['target'] and signal['stop_loss']:
            entry = float(signal['entry'])
            target = float(signal['target'])
            stop = float(signal['stop_loss'])

            analysis['risk_reward'] = {
                'entry': entry,
                'target': target,
                'stop': stop,
                'potential_gain_dollars': target - entry,
                'potential_loss_dollars': entry - stop,
                'ratio': (target - entry) / (entry - stop) if (entry - stop) > 0 else None,
                'potential_gain_percent': ((target - entry) / entry * 100) if entry > 0 else None,
                'potential_loss_percent': ((entry - stop) / entry * 100) if entry > 0 else None
            }

        # Comprehensive recommendation
        composite = signal.get('composite_score', 50)
        author_wr = signal.get('author_win_rate', 50)
        setup_wr = signal.get('setup_win_rate', 50)

        if composite >= 75:
            final_rec = "STRONG BUY"
            confidence_level = "Very High"
        elif composite >= 65:
            final_rec = "BUY"
            confidence_level = "High"
        elif composite >= 55:
            final_rec = "MODERATE BUY"
            confidence_level = "Moderate"
        elif composite >= 45:
            final_rec = "HOLD / MONITOR"
            confidence_level = "Low"
        else:
            final_rec = "PASS"
            confidence_level = "Very Low"

        analysis['final_recommendation'] = {
            'action': final_rec,
            'confidence_level': confidence_level,
            'composite_score': composite,
            'key_factors': [
                f"Author win rate: {author_wr:.1f}%" if author_wr else "Author: No track record",
                f"Setup success rate: {setup_wr:.1f}%" if setup_wr else "Setup: No historical data",
                f"Similar trades: {len(similar_wins)}/{len(similar_all)} wins" if similar_all else "No similar trades found"
            ],
            'reasoning': signal.get('reasoning', 'Insufficient data for analysis')
        }

        return analysis

    def get_best_setups_by_ticker(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Find the best historical setups for a specific ticker"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT
                setup_type,
                total_signals,
                trades_taken,
                wins,
                losses,
                win_rate,
                avg_pnl_percent,
                success_score
            FROM setup_performance
            WHERE ticker = %s
            AND trades_taken >= 3
            ORDER BY success_score DESC
            LIMIT %s
        """, (ticker.upper(), limit))

        return cur.fetchall()

    def get_top_authors(self, min_trades: int = 5, limit: int = 10) -> List[Dict]:
        """Get most credible authors based on track record"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT
                author,
                total_signals,
                trades_taken,
                wins,
                losses,
                win_rate,
                avg_pnl_percent,
                total_pnl_dollars,
                credibility_score
            FROM author_performance
            WHERE trades_taken >= %s
            ORDER BY credibility_score DESC
            LIMIT %s
        """, (min_trades, limit))

        return cur.fetchall()

    def search_similar_to_description(
        self,
        description: str,
        only_winners: bool = True,
        n_results: int = 10
    ) -> List[Dict]:
        """
        Natural language search for signals

        Example: "SPY breakout above 550 with high volume"
        """
        return self.vector_search.search_by_criteria(
            query=description,
            outcome='win' if only_winners else None,
            n_results=n_results
        )

    def get_win_rate_by_time_of_day(self, ticker: Optional[str] = None) -> Dict:
        """Analyze if certain times of day have better win rates"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        where_clause = ""
        params = []
        if ticker:
            where_clause = "AND s.primary_ticker = %s"
            params.append(ticker.upper())

        cur.execute(f"""
            SELECT
                EXTRACT(HOUR FROM s.timestamp) as hour_of_day,
                COUNT(o.id) as total_trades,
                COUNT(o.id) FILTER (WHERE o.outcome = 'win') as wins,
                COUNT(o.id) FILTER (WHERE o.outcome = 'loss') as losses,
                CASE
                    WHEN COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss')) > 0
                    THEN (COUNT(o.id) FILTER (WHERE o.outcome = 'win')::DECIMAL /
                          COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss'))) * 100
                    ELSE NULL
                END as win_rate
            FROM discord_trading_signals s
            JOIN signal_outcomes o ON s.id = o.signal_id
            WHERE o.trade_taken = TRUE
            {where_clause}
            GROUP BY EXTRACT(HOUR FROM s.timestamp)
            HAVING COUNT(o.id) >= 3
            ORDER BY win_rate DESC NULLS LAST
        """, params)

        results = cur.fetchall()
        cur.close()
        conn.close()

        return {
            'ticker': ticker or 'All tickers',
            'hourly_performance': results,
            'best_hour': results[0] if results else None
        }


if __name__ == "__main__":
    advisor = AVASignalAdvisor()
    print("AVA Signal Advisor initialized!")
    print("\nExample queries:")
    print("  advisor.get_top_recommendations(limit=5)")
    print("  advisor.analyze_signal_with_context(signal_id=1)")
    print("  advisor.get_best_setups_by_ticker('SPY')")
    print("  advisor.get_top_authors()")
