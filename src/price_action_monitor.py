"""
Price Action Monitor for Sports Betting
Monitors odds changes and alerts on significant drops when AI still predicts team will win

Key Features:
- Tracks odds history for all active markets
- Detects significant price movements (>10% drop)
- Compares price action vs AI predictions
- Sends Telegram alerts for value opportunities
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import psycopg2
import psycopg2.extras
from src.kalshi_db_manager import KalshiDBManager
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent

logger = logging.getLogger(__name__)


class PriceActionMonitor:
    """
    Monitors betting odds for significant price movements

    Alert triggers:
    - Odds drop > 10% in last 30 minutes
    - AI still predicts team will win (>60% confidence)
    - Expected value increases due to price drop
    """

    def __init__(self, db_manager: KalshiDBManager):
        self.db = db_manager
        self.ai_agent = AdvancedBettingAIAgent()

        # Alert thresholds
        self.min_price_drop_pct = 10  # 10% drop
        self.min_ai_confidence = 60  # 60% confidence
        self.min_time_window = 30  # 30 minutes
        self.alert_cooldown = 60  # 60 minutes between same alerts

        # Create price history table if not exists
        self._create_price_history_table()

    def _create_price_history_table(self) -> None:
        """Create table to track price history"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS kalshi_price_history (
                    id SERIAL PRIMARY KEY,
                    market_ticker TEXT NOT NULL,
                    yes_price NUMERIC NOT NULL,
                    no_price NUMERIC NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    volume NUMERIC,
                    UNIQUE(market_ticker, timestamp)
                )
            """)

            # Create index for faster queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_history_ticker_time
                ON kalshi_price_history(market_ticker, timestamp DESC)
            """)

            # Create price drop alerts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS price_drop_alerts (
                    id SERIAL PRIMARY KEY,
                    market_ticker TEXT NOT NULL,
                    team_name TEXT,
                    price_before NUMERIC,
                    price_after NUMERIC,
                    price_drop_pct NUMERIC,
                    ai_confidence NUMERIC,
                    ai_prediction TEXT,
                    expected_value NUMERIC,
                    reasoning TEXT[],
                    alert_sent BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            conn.commit()
            cur.close()
            conn.close()

            logger.info("Price history tables created successfully")

        except Exception as e:
            logger.error(f"Error creating price history table: {e}")

    def record_current_prices(self, markets: List[Dict]):
        """
        Record current prices for all active markets

        Args:
            markets: List of market data from Kalshi
        """
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            for market in markets:
                ticker = market.get('ticker')
                yes_price = market.get('yes_price', 0)
                no_price = market.get('no_price', 0)
                volume = market.get('volume', 0)

                if not ticker or yes_price == 0:
                    continue

                # Insert price snapshot
                cur.execute("""
                    INSERT INTO kalshi_price_history
                        (market_ticker, yes_price, no_price, volume, timestamp)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (market_ticker, timestamp) DO NOTHING
                """, (ticker, yes_price, no_price, volume))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Recorded prices for {len(markets)} markets")

        except Exception as e:
            logger.error(f"Error recording prices: {e}")

    def detect_price_drops(self) -> List[Dict]:
        """
        Detect significant price drops in the last time window

        Returns:
            List of price drop opportunities with AI analysis
        """
        opportunities = []

        try:
            conn = self.db.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Find markets with significant price drops
            cur.execute("""
                WITH recent_prices AS (
                    SELECT
                        market_ticker,
                        yes_price,
                        no_price,
                        timestamp,
                        LAG(yes_price) OVER (PARTITION BY market_ticker ORDER BY timestamp) as prev_yes_price,
                        LAG(no_price) OVER (PARTITION BY market_ticker ORDER BY timestamp) as prev_no_price,
                        LAG(timestamp) OVER (PARTITION BY market_ticker ORDER BY timestamp) as prev_timestamp
                    FROM kalshi_price_history
                    WHERE timestamp > NOW() - INTERVAL '%s minutes'
                ),
                price_changes AS (
                    SELECT
                        market_ticker,
                        yes_price as current_yes_price,
                        no_price as current_no_price,
                        prev_yes_price,
                        prev_no_price,
                        ((prev_yes_price - yes_price) / NULLIF(prev_yes_price, 0)) * 100 as yes_drop_pct,
                        ((prev_no_price - no_price) / NULLIF(prev_no_price, 0)) * 100 as no_drop_pct,
                        timestamp,
                        prev_timestamp
                    FROM recent_prices
                    WHERE prev_yes_price IS NOT NULL
                )
                SELECT
                    pc.*,
                    m.title,
                    m.close_time,
                    m.market_type
                FROM price_changes pc
                JOIN kalshi_markets m ON pc.market_ticker = m.ticker
                WHERE
                    (pc.yes_drop_pct > %s OR pc.no_drop_pct > %s)
                    AND m.status = 'active'
                ORDER BY GREATEST(pc.yes_drop_pct, pc.no_drop_pct) DESC
                LIMIT 20
            """, (self.min_time_window, self.min_price_drop_pct, self.min_price_drop_pct))

            price_drops = cur.fetchall()
            cur.close()
            conn.close()

            # Analyze each price drop with AI
            for drop in price_drops:
                # Check if already alerted recently
                if self._was_recently_alerted(drop['market_ticker']):
                    continue

                # Get AI prediction for current state
                game_data = {
                    'id': drop['market_ticker'],
                    'title': drop['title'],
                    'away_team': 'Team A',  # Extract from title
                    'home_team': 'Team B',  # Extract from title
                    'is_live': True,
                    'kalshi_odds': {
                        'away_win_price': drop['current_yes_price'],
                        'home_win_price': drop['current_no_price']
                    }
                }

                market_data = {
                    'yes_price': drop['current_yes_price'],
                    'no_price': drop['current_no_price']
                }

                ai_prediction = self.ai_agent.analyze_betting_opportunity(game_data, market_data)

                # Check if AI still predicts win despite price drop
                if ai_prediction['confidence_score'] >= self.min_ai_confidence:
                    # This is an opportunity - price dropped but AI still confident
                    opportunity = {
                        'market_ticker': drop['market_ticker'],
                        'title': drop['title'],
                        'price_before_yes': drop['prev_yes_price'],
                        'price_after_yes': drop['current_yes_price'],
                        'price_before_no': drop['prev_no_price'],
                        'price_after_no': drop['current_no_price'],
                        'yes_drop_pct': drop['yes_drop_pct'],
                        'no_drop_pct': drop['no_drop_pct'],
                        'ai_confidence': ai_prediction['confidence_score'],
                        'ai_prediction': ai_prediction['predicted_winner'],
                        'expected_value': ai_prediction['expected_value'],
                        'reasoning': ai_prediction['reasoning'],
                        'recommendation': ai_prediction['recommendation'],
                        'timestamp': datetime.now().isoformat()
                    }

                    opportunities.append(opportunity)

                    # Store alert
                    self._store_price_drop_alert(opportunity)

            logger.info(f"Found {len(opportunities)} price drop opportunities")

        except Exception as e:
            logger.error(f"Error detecting price drops: {e}")

        return opportunities

    def _was_recently_alerted(self, market_ticker: str) -> bool:
        """Check if we already sent an alert for this market recently"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT COUNT(*)
                FROM price_drop_alerts
                WHERE market_ticker = %s
                AND created_at > NOW() - INTERVAL '%s minutes'
            """, (market_ticker, self.alert_cooldown))

            count = cur.fetchone()[0]
            cur.close()
            conn.close()

            return count > 0

        except Exception as e:
            logger.error(f"Error checking alert history: {e}")
            return False

    def _store_price_drop_alert(self, opportunity: Dict):
        """Store price drop alert in database"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO price_drop_alerts (
                    market_ticker,
                    team_name,
                    price_before,
                    price_after,
                    price_drop_pct,
                    ai_confidence,
                    ai_prediction,
                    expected_value,
                    reasoning,
                    alert_sent
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            """, (
                opportunity['market_ticker'],
                opportunity.get('title'),
                opportunity.get('price_before_yes', 0),
                opportunity.get('price_after_yes', 0),
                opportunity.get('yes_drop_pct', 0),
                opportunity.get('ai_confidence', 0),
                opportunity.get('ai_prediction', ''),
                opportunity.get('expected_value', 0),
                opportunity.get('reasoning', [])
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing alert: {e}")

    def generate_price_drop_alert_message(self, opportunity: Dict) -> str:
        """
        Generate Telegram alert message for price drop

        Args:
            opportunity: Price drop opportunity data

        Returns:
            Formatted alert message
        """
        yes_drop = opportunity.get('yes_drop_pct', 0)
        no_drop = opportunity.get('no_drop_pct', 0)
        max_drop = max(yes_drop, no_drop)

        msg_lines = [
            "💰 PRICE DROP ALERT 💰",
            "",
            f"Market: {opportunity['title']}",
            "",
            f"📉 Price Drop: {max_drop:.1f}%",
            f"   Before: {opportunity.get('price_before_yes', 0):.2f}",
            f"   After: {opportunity.get('price_after_yes', 0):.2f}",
            "",
            f"🤖 AI Still Predicts: {opportunity['ai_prediction']}",
            f"   Confidence: {opportunity['ai_confidence']:.0f}%",
            f"   Expected Value: {opportunity['expected_value']:+.1f}%",
            "",
            "💡 Why this is an opportunity:",
        ]

        for reason in opportunity.get('reasoning', []):
            msg_lines.append(f"  • {reason}")

        msg_lines.extend([
            "",
            f"⏰ Detected: {opportunity['timestamp']}",
            "",
            "✅ This is a value betting opportunity - price dropped but AI still confident!"
        ])

        return "\n".join(msg_lines)

    def cleanup_old_price_history(self, days_to_keep: int = 7):
        """
        Clean up old price history to save space

        Args:
            days_to_keep: Number of days of history to keep
        """
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                DELETE FROM kalshi_price_history
                WHERE timestamp < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))

            deleted_count = cur.rowcount
            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Cleaned up {deleted_count} old price records")

        except Exception as e:
            logger.error(f"Error cleaning up price history: {e}")
