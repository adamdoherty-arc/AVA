"""
Signal Performance Tracker
Tracks trade outcomes and calculates win rates, author credibility, setup performance
"""
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import Dict, List, Optional
import os


class SignalPerformanceTracker:
    """Track and analyze trading signal performance"""

    def __init__(self) -> None:
        self.db_password = os.getenv('DB_PASSWORD')

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(
            host='localhost',
            port='5432',
            database='magnus',
            user='postgres',
            password=self.db_password
        )

    def create_performance_tables(self) -> None:
        """Create tables for tracking signal performance"""
        conn = self.get_connection()
        cur = conn.cursor()

        # Signal outcomes table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id SERIAL PRIMARY KEY,
                signal_id INTEGER REFERENCES discord_trading_signals(id) ON DELETE CASCADE,
                trade_taken BOOLEAN DEFAULT FALSE,
                entry_price DECIMAL,
                exit_price DECIMAL,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                outcome TEXT,  -- 'win', 'loss', 'breakeven', 'pending'
                pnl_dollars DECIMAL,
                pnl_percent DECIMAL,
                position_size INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(signal_id)
            )
        """)

        # Author performance table (aggregated stats)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS author_performance (
                id SERIAL PRIMARY KEY,
                author TEXT UNIQUE NOT NULL,
                total_signals INTEGER DEFAULT 0,
                trades_taken INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                breakeven INTEGER DEFAULT 0,
                win_rate DECIMAL,
                avg_pnl_percent DECIMAL,
                total_pnl_dollars DECIMAL,
                credibility_score DECIMAL,  -- 0-100 score
                last_updated TIMESTAMP DEFAULT NOW()
            )
        """)

        # Setup type performance (aggregated by setup + ticker)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS setup_performance (
                id SERIAL PRIMARY KEY,
                ticker TEXT,
                setup_type TEXT,
                total_signals INTEGER DEFAULT 0,
                trades_taken INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                win_rate DECIMAL,
                avg_pnl_percent DECIMAL,
                success_score DECIMAL,  -- 0-100 composite score
                last_updated TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, setup_type)
            )
        """)

        # Signal quality scores (enhanced scoring with multiple factors)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS signal_quality_scores (
                id SERIAL PRIMARY KEY,
                signal_id INTEGER REFERENCES discord_trading_signals(id) ON DELETE CASCADE,
                base_confidence INTEGER,  -- Original 0-100 confidence
                author_credibility DECIMAL,  -- 0-100 from author_performance
                setup_success_rate DECIMAL,  -- 0-100 from setup_performance
                similarity_score DECIMAL,  -- 0-100 from similar past trades
                market_alignment DECIMAL,  -- 0-100 based on market conditions
                composite_score DECIMAL,  -- Weighted average of all factors
                rank INTEGER,  -- Rank among all signals (1 = best)
                recommendation TEXT,  -- 'strong_buy', 'buy', 'hold', 'pass'
                reasoning TEXT,  -- Explanation of score
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(signal_id)
            )
        """)

        # Indexes for fast queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_outcomes_signal_id
            ON signal_outcomes(signal_id)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_outcomes_outcome
            ON signal_outcomes(outcome)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_author_performance_credibility
            ON author_performance(credibility_score DESC)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_setup_performance_ticker
            ON setup_performance(ticker, success_score DESC)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_quality_scores_composite
            ON signal_quality_scores(composite_score DESC)
        """)

        conn.commit()
        cur.close()
        conn.close()

    def record_trade_outcome(self, signal_id: int, outcome_data: Dict):
        """Record the outcome of a trade based on a signal"""
        conn = self.get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO signal_outcomes (
                signal_id, trade_taken, entry_price, exit_price,
                entry_time, exit_time, outcome, pnl_dollars, pnl_percent,
                position_size, notes, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (signal_id) DO UPDATE SET
                trade_taken = EXCLUDED.trade_taken,
                exit_price = EXCLUDED.exit_price,
                exit_time = EXCLUDED.exit_time,
                outcome = EXCLUDED.outcome,
                pnl_dollars = EXCLUDED.pnl_dollars,
                pnl_percent = EXCLUDED.pnl_percent,
                notes = EXCLUDED.notes,
                updated_at = NOW()
        """, (
            signal_id,
            outcome_data.get('trade_taken', False),
            outcome_data.get('entry_price'),
            outcome_data.get('exit_price'),
            outcome_data.get('entry_time'),
            outcome_data.get('exit_time'),
            outcome_data.get('outcome', 'pending'),
            outcome_data.get('pnl_dollars'),
            outcome_data.get('pnl_percent'),
            outcome_data.get('position_size'),
            outcome_data.get('notes')
        ))

        conn.commit()
        cur.close()
        conn.close()

        # Update aggregated stats
        self.update_author_performance()
        self.update_setup_performance()
        self.calculate_quality_scores()

    def update_author_performance(self) -> None:
        """Recalculate author performance stats"""
        conn = self.get_connection()
        cur = conn.cursor()

        # Calculate stats for each author
        cur.execute("""
            INSERT INTO author_performance (
                author, total_signals, trades_taken, wins, losses, breakeven,
                win_rate, avg_pnl_percent, total_pnl_dollars, credibility_score, last_updated
            )
            SELECT
                s.author,
                COUNT(s.id) as total_signals,
                COUNT(o.id) FILTER (WHERE o.trade_taken = TRUE) as trades_taken,
                COUNT(o.id) FILTER (WHERE o.outcome = 'win') as wins,
                COUNT(o.id) FILTER (WHERE o.outcome = 'loss') as losses,
                COUNT(o.id) FILTER (WHERE o.outcome = 'breakeven') as breakeven,
                CASE
                    WHEN COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss')) > 0
                    THEN (COUNT(o.id) FILTER (WHERE o.outcome = 'win')::DECIMAL /
                          COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss'))) * 100
                    ELSE NULL
                END as win_rate,
                AVG(o.pnl_percent) FILTER (WHERE o.outcome IN ('win', 'loss')) as avg_pnl_percent,
                SUM(o.pnl_dollars) FILTER (WHERE o.outcome IN ('win', 'loss', 'breakeven')) as total_pnl_dollars,
                CASE
                    -- Credibility score: win_rate * 0.6 + avg_pnl_percent * 0.3 + trade_volume_factor * 0.1
                    WHEN COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss')) >= 10
                    THEN LEAST(100, (
                        (COUNT(o.id) FILTER (WHERE o.outcome = 'win')::DECIMAL /
                         COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss'))) * 60 +
                        COALESCE(AVG(o.pnl_percent) FILTER (WHERE o.outcome = 'win'), 0) * 0.3 +
                        LEAST(10, COUNT(o.id) FILTER (WHERE o.trade_taken = TRUE) / 10.0 * 10)
                    ))
                    WHEN COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss')) >= 5
                    THEN LEAST(100, (
                        (COUNT(o.id) FILTER (WHERE o.outcome = 'win')::DECIMAL /
                         COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss'))) * 50
                    ))
                    ELSE 50  -- Default neutral score for new authors
                END as credibility_score,
                NOW() as last_updated
            FROM discord_trading_signals s
            LEFT JOIN signal_outcomes o ON s.id = o.signal_id
            GROUP BY s.author
            ON CONFLICT (author) DO UPDATE SET
                total_signals = EXCLUDED.total_signals,
                trades_taken = EXCLUDED.trades_taken,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                breakeven = EXCLUDED.breakeven,
                win_rate = EXCLUDED.win_rate,
                avg_pnl_percent = EXCLUDED.avg_pnl_percent,
                total_pnl_dollars = EXCLUDED.total_pnl_dollars,
                credibility_score = EXCLUDED.credibility_score,
                last_updated = NOW()
        """)

        conn.commit()
        cur.close()
        conn.close()

    def update_setup_performance(self) -> None:
        """Recalculate setup type performance by ticker"""
        conn = self.get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO setup_performance (
                ticker, setup_type, total_signals, trades_taken, wins, losses,
                win_rate, avg_pnl_percent, success_score, last_updated
            )
            SELECT
                s.primary_ticker as ticker,
                s.setup_type,
                COUNT(s.id) as total_signals,
                COUNT(o.id) FILTER (WHERE o.trade_taken = TRUE) as trades_taken,
                COUNT(o.id) FILTER (WHERE o.outcome = 'win') as wins,
                COUNT(o.id) FILTER (WHERE o.outcome = 'loss') as losses,
                CASE
                    WHEN COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss')) > 0
                    THEN (COUNT(o.id) FILTER (WHERE o.outcome = 'win')::DECIMAL /
                          COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss'))) * 100
                    ELSE NULL
                END as win_rate,
                AVG(o.pnl_percent) FILTER (WHERE o.outcome IN ('win', 'loss')) as avg_pnl_percent,
                CASE
                    -- Success score: win_rate * 0.7 + avg_pnl_percent * 0.3
                    WHEN COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss')) >= 5
                    THEN LEAST(100, (
                        (COUNT(o.id) FILTER (WHERE o.outcome = 'win')::DECIMAL /
                         COUNT(o.id) FILTER (WHERE o.outcome IN ('win', 'loss'))) * 70 +
                        COALESCE(AVG(o.pnl_percent) FILTER (WHERE o.outcome = 'win'), 0) * 0.3
                    ))
                    ELSE 50  -- Default neutral score
                END as success_score,
                NOW() as last_updated
            FROM discord_trading_signals s
            LEFT JOIN signal_outcomes o ON s.id = o.signal_id
            WHERE s.primary_ticker IS NOT NULL
            GROUP BY s.primary_ticker, s.setup_type
            ON CONFLICT (ticker, setup_type) DO UPDATE SET
                total_signals = EXCLUDED.total_signals,
                trades_taken = EXCLUDED.trades_taken,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                win_rate = EXCLUDED.win_rate,
                avg_pnl_percent = EXCLUDED.avg_pnl_percent,
                success_score = EXCLUDED.success_score,
                last_updated = NOW()
        """)

        conn.commit()
        cur.close()
        conn.close()

    def calculate_quality_scores(self) -> None:
        """Calculate composite quality scores for all signals"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get all signals
        cur.execute("""
            SELECT
                s.id as signal_id,
                s.confidence as base_confidence,
                s.author,
                s.primary_ticker,
                s.setup_type,
                a.credibility_score as author_credibility,
                sp.success_score as setup_success_rate
            FROM discord_trading_signals s
            LEFT JOIN author_performance a ON s.author = a.author
            LEFT JOIN setup_performance sp ON s.primary_ticker = sp.ticker
                AND s.setup_type = sp.setup_type
        """)

        signals = cur.fetchall()

        for signal in signals:
            # Multi-factor composite score
            base_conf = float(signal['base_confidence']) if signal['base_confidence'] else 40.0
            author_cred = float(signal['author_credibility']) if signal['author_credibility'] else 50.0  # Default neutral
            setup_success = float(signal['setup_success_rate']) if signal['setup_success_rate'] else 50.0  # Default neutral

            # Weighted average:
            # - Author credibility: 40% weight
            # - Setup success rate: 30% weight
            # - Base confidence: 20% weight
            # - Market alignment: 10% weight (placeholder for now)

            market_align = 50.0  # TODO: Implement market condition analysis

            composite = (
                author_cred * 0.40 +
                setup_success * 0.30 +
                base_conf * 0.20 +
                market_align * 0.10
            )

            # Determine recommendation
            if composite >= 75:
                recommendation = 'strong_buy'
                reasoning = f"High confidence: {author_cred:.0f}% author credibility, {setup_success:.0f}% setup success rate"
            elif composite >= 60:
                recommendation = 'buy'
                reasoning = f"Good opportunity: {author_cred:.0f}% author credibility, {setup_success:.0f}% setup success rate"
            elif composite >= 45:
                recommendation = 'hold'
                reasoning = f"Monitor: {author_cred:.0f}% author credibility, {setup_success:.0f}% setup success rate"
            else:
                recommendation = 'pass'
                reasoning = f"Low confidence: {author_cred:.0f}% author credibility, {setup_success:.0f}% setup success rate"

            # Insert/update quality score
            cur.execute("""
                INSERT INTO signal_quality_scores (
                    signal_id, base_confidence, author_credibility, setup_success_rate,
                    similarity_score, market_alignment, composite_score, recommendation, reasoning, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (signal_id) DO UPDATE SET
                    base_confidence = EXCLUDED.base_confidence,
                    author_credibility = EXCLUDED.author_credibility,
                    setup_success_rate = EXCLUDED.setup_success_rate,
                    similarity_score = EXCLUDED.similarity_score,
                    market_alignment = EXCLUDED.market_alignment,
                    composite_score = EXCLUDED.composite_score,
                    recommendation = EXCLUDED.recommendation,
                    reasoning = EXCLUDED.reasoning,
                    updated_at = NOW()
            """, (
                signal['signal_id'],
                base_conf,
                author_cred,
                setup_success,
                50,  # Similarity score placeholder
                market_align,
                composite,
                recommendation,
                reasoning
            ))

        conn.commit()

        # Update ranks
        cur.execute("""
            UPDATE signal_quality_scores sq
            SET rank = ranked.rank
            FROM (
                SELECT signal_id, ROW_NUMBER() OVER (ORDER BY composite_score DESC) as rank
                FROM signal_quality_scores
            ) ranked
            WHERE sq.signal_id = ranked.signal_id
        """)

        conn.commit()
        cur.close()
        conn.close()

    def get_top_signals(self, limit: int = 10) -> List[Dict]:
        """Get top-ranked signals for AVA to recommend"""
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
                s.author,
                s.timestamp,
                s.content,
                sq.composite_score,
                sq.rank,
                sq.recommendation,
                sq.reasoning,
                sq.author_credibility,
                sq.setup_success_rate,
                a.win_rate as author_win_rate,
                sp.win_rate as setup_win_rate
            FROM discord_trading_signals s
            JOIN signal_quality_scores sq ON s.id = sq.signal_id
            LEFT JOIN author_performance a ON s.author = a.author
            LEFT JOIN setup_performance sp ON s.primary_ticker = sp.ticker
                AND s.setup_type = sp.setup_type
            WHERE sq.recommendation IN ('strong_buy', 'buy')
            AND s.timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY sq.composite_score DESC
            LIMIT %s
        """, (limit,))

        return cur.fetchall()


if __name__ == "__main__":
    tracker = SignalPerformanceTracker()
    print("Creating performance tracking tables...")
    tracker.create_performance_tables()
    print("Performance tracking system initialized!")
