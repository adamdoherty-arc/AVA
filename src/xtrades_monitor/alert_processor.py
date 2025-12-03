"""
Xtrades Alert Processor
=======================

Processes Xtrades alerts in real-time:
- Detects new alerts (NEW)
- Detects updated alerts (UPDATE)
- Detects closed alerts (CLOSE)
- Enriches with market data
- Triggers AI evaluation pipeline
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import os
from dotenv import load_dotenv

# Import connection pool
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from db_connection_pool import get_db_pool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class AlertType(Enum):
    """Type of alert event"""
    NEW = "new"  # New trade alert
    UPDATE = "update"  # Existing trade updated
    CLOSE = "close"  # Trade closed
    UNKNOWN = "unknown"


class AlertProcessor:
    """
    Processes Xtrades trade alerts and detects events.

    Responsibilities:
    - Compare current scrape with previous state
    - Identify new, updated, and closed trades
    - Enrich alerts with market data
    - Prepare for AI evaluation
    """

    def __init__(self) -> None:
        """Initialize alert processor with database connection pool"""
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment")

        # Get connection pool instance
        self.pool = get_db_pool()
        logger.info("Alert processor initialized with connection pooling")

    def process_scrape_results(self, profile_username: str,
                               scraped_trades: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Process newly scraped trades and detect events with ACID transaction.

        Args:
            profile_username: Username of the profile scraped
            scraped_trades: List of trades from latest scrape

        Returns:
            Dict with keys: new_alerts, updated_alerts, closed_alerts
        """
        # Use connection pool with context manager for automatic cleanup
        with self.pool.get_connection() as conn:
            try:
                # Set serializable isolation level to prevent race conditions
                conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE)

                # Use transaction context - all or nothing
                with conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        # Get profile ID with row lock to prevent concurrent modifications
                        cursor.execute(
                            "SELECT id FROM xtrades_profiles WHERE username = %s FOR UPDATE",
                            (profile_username,)
                        )
                        profile_row = cursor.fetchone()
                        if not profile_row:
                            logger.error(f"Profile not found: {profile_username}")
                            return {"new_alerts": [], "updated_alerts": [], "closed_alerts": []}

                        profile_id = profile_row['id']

                        # Get existing open trades with row locks (SKIP LOCKED prevents deadlocks)
                        cursor.execute("""
                            SELECT id, ticker, xtrades_alert_id, alert_text, entry_price,
                                   strike_price, expiration_date, quantity, status, updated_at
                            FROM xtrades_trades
                            WHERE profile_id = %s AND status = %s
                            FOR UPDATE SKIP LOCKED
                        """, (profile_id, 'open'))

                        existing_trades = {
                            row['xtrades_alert_id']: dict(row)
                            for row in cursor.fetchall()
                            if row['xtrades_alert_id']
                        }

                        # Track scraped alert IDs
                        scraped_alert_ids = set()

                        # Results
                        new_alerts = []
                        updated_alerts = []
                        closed_alerts = []

                        # Process each scraped trade
                        for trade in scraped_trades:
                            alert_id = trade.get('xtrades_alert_id')
                            if not alert_id:
                                # Generate alert ID from trade details
                                alert_id = self._generate_alert_id(trade)
                                trade['xtrades_alert_id'] = alert_id

                            scraped_alert_ids.add(alert_id)

                            if alert_id in existing_trades:
                                # Existing trade - check if updated
                                existing = existing_trades[alert_id]

                                if self._is_trade_updated(existing, trade):
                                    # Trade has been updated
                                    updated_trade = self._update_trade(cursor, existing['id'], trade)
                                    updated_alerts.append({
                                        'type': AlertType.UPDATE,
                                        'trade_id': existing['id'],
                                        'trade_data': updated_trade,
                                        'changes': self._get_trade_changes(existing, trade)
                                    })
                            else:
                                # New trade
                                new_trade = self._insert_new_trade(cursor, profile_id, trade)
                                new_alerts.append({
                                    'type': AlertType.NEW,
                                    'trade_id': new_trade['id'],
                                    'trade_data': new_trade
                                })

                        # Check for closed trades (no longer in scrape results)
                        for alert_id, existing_trade in existing_trades.items():
                            if alert_id not in scraped_alert_ids:
                                # Trade is no longer in active alerts - mark as closed
                                closed_trade = self._close_trade(cursor, existing_trade['id'])
                                closed_alerts.append({
                                    'type': AlertType.CLOSE,
                                    'trade_id': existing_trade['id'],
                                    'trade_data': closed_trade
                                })

                        # Transaction commits here automatically (with context)

                logger.info(
                    f"Processed scrape for {profile_username}: "
                    f"{len(new_alerts)} new, {len(updated_alerts)} updated, "
                    f"{len(closed_alerts)} closed"
                )

                return {
                    'new_alerts': new_alerts,
                    'updated_alerts': updated_alerts,
                    'closed_alerts': closed_alerts
                }

            except psycopg2.OperationalError as e:
                logger.error(f"Database connection error: {e}", exc_info=True)
                return {"new_alerts": [], "updated_alerts": [], "closed_alerts": []}

            except Exception as e:
                logger.error(f"Error processing scrape results: {e}", exc_info=True)
                # Transaction automatically rolled back by context manager
                return {"new_alerts": [], "updated_alerts": [], "closed_alerts": []}

    def _generate_alert_id(self, trade: Dict[str, Any]) -> str:
        """Generate unique alert ID from trade details"""
        ticker = trade.get('ticker', 'UNKNOWN')
        action = trade.get('action', 'UNKNOWN')
        strike = trade.get('strike_price', 0)
        expiry = trade.get('expiration_date', 'UNKNOWN')
        timestamp = trade.get('alert_timestamp', datetime.now())

        # Create unique ID
        return f"{ticker}_{action}_{strike}_{expiry}_{timestamp}".replace(" ", "_")

    def _is_trade_updated(self, existing: Dict, new: Dict) -> bool:
        """Check if trade has been updated"""
        # Compare key fields
        check_fields = ['entry_price', 'quantity', 'strike_price',
                       'expiration_date', 'alert_text', 'status']

        for field in check_fields:
            if field in new and existing.get(field) != new.get(field):
                return True

        return False

    def _get_trade_changes(self, existing: Dict, new: Dict) -> Dict[str, tuple]:
        """Get what changed in the trade"""
        changes = {}
        check_fields = ['entry_price', 'quantity', 'strike_price',
                       'expiration_date', 'alert_text', 'status']

        for field in check_fields:
            if field in new and existing.get(field) != new.get(field):
                changes[field] = (existing.get(field), new.get(field))

        return changes

    def _insert_new_trade(self, cursor, profile_id: int,
                         trade: Dict[str, Any]) -> Dict[str, Any]:
        """Insert new trade into database - uses passed cursor (part of transaction)"""
        cursor.execute("""
            INSERT INTO xtrades_trades (
                profile_id, ticker, strategy, action, entry_price,
                entry_date, quantity, status, strike_price, expiration_date,
                alert_text, alert_timestamp, xtrades_alert_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING *
        """, (
            profile_id,
            trade.get('ticker'),
            trade.get('strategy'),
            trade.get('action'),
            trade.get('entry_price'),
            trade.get('entry_date', datetime.now()),
            trade.get('quantity', 1),
            'open',
            trade.get('strike_price'),
            trade.get('expiration_date'),
            trade.get('alert_text'),
            trade.get('alert_timestamp', datetime.now()),
            trade.get('xtrades_alert_id')
        ))

        new_trade = dict(cursor.fetchone())

        logger.info(f"Inserted new trade: {new_trade['ticker']} (ID: {new_trade['id']})")
        return new_trade

    def _update_trade(self, cursor, trade_id: int,
                     trade: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing trade - uses passed cursor (part of transaction)"""
        cursor.execute("""
            UPDATE xtrades_trades
            SET entry_price = COALESCE(%s, entry_price),
                quantity = COALESCE(%s, quantity),
                strike_price = COALESCE(%s, strike_price),
                expiration_date = COALESCE(%s, expiration_date),
                alert_text = COALESCE(%s, alert_text),
                updated_at = NOW()
            WHERE id = %s
            RETURNING *
        """, (
            trade.get('entry_price'),
            trade.get('quantity'),
            trade.get('strike_price'),
            trade.get('expiration_date'),
            trade.get('alert_text'),
            trade_id
        ))

        updated_trade = dict(cursor.fetchone())

        logger.info(f"Updated trade: {updated_trade['ticker']} (ID: {trade_id})")
        return updated_trade

    def _close_trade(self, cursor, trade_id: int) -> Dict[str, Any]:
        """Mark trade as closed - uses passed cursor (part of transaction)"""
        cursor.execute("""
            UPDATE xtrades_trades
            SET status = %s,
                exit_date = NOW(),
                updated_at = NOW()
            WHERE id = %s
            RETURNING *
        """, ('closed', trade_id))

        closed_trade = dict(cursor.fetchone())

        logger.info(f"Closed trade: {closed_trade['ticker']} (ID: {trade_id})")
        return closed_trade

    def enrich_alert_with_market_data(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich alert with additional market data.

        This will fetch:
        - Current stock price
        - IV (implied volatility)
        - 52-week high/low
        - Market cap
        - Other relevant data for strategy evaluation

        For now, returns a placeholder. Will integrate with market data provider.
        """
        trade_data = alert['trade_data']
        ticker = trade_data.get('ticker')

        if not ticker:
            logger.warning("No ticker found in trade data")
            return alert

        # TODO: Integrate with Yahoo Finance / Polygon / etc
        # For now, return basic enrichment
        enriched_data = {
            'ticker': ticker,
            'current_price': trade_data.get('entry_price', 0),  # Placeholder
            'iv': 0.30,  # Placeholder - 30% IV
            'price_52w_high': 0,  # TODO: Fetch
            'price_52w_low': 0,  # TODO: Fetch
            'market_cap': 0,  # TODO: Fetch
            'sector': 'Unknown',  # TODO: Fetch
            'pe_ratio': 0,  # TODO: Fetch
        }

        alert['market_data'] = enriched_data
        return alert

    def prepare_for_evaluation(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare alert data for AI evaluation.

        Converts trade data into format expected by ComprehensiveStrategyAnalyzer.
        """
        trade_data = alert['trade_data']
        market_data = alert.get('market_data', {})

        # Stock data for analyzer
        stock_data = {
            'symbol': trade_data.get('ticker'),
            'current_price': market_data.get('current_price', trade_data.get('entry_price', 0)),
            'iv': market_data.get('iv', 0.30),
            'price_52w_high': market_data.get('price_52w_high', 0),
            'price_52w_low': market_data.get('price_52w_low', 0),
            'market_cap': market_data.get('market_cap', 0),
            'sector': market_data.get('sector', 'Unknown'),
            'pe_ratio': market_data.get('pe_ratio', 0),
        }

        # Options data for analyzer
        options_data = {
            'strike_price': trade_data.get('strike_price', 0),
            'dte': self._calculate_dte(trade_data.get('expiration_date')),
            'delta': -0.30,  # TODO: Calculate or fetch actual delta
            'premium': float(trade_data.get('entry_price', 0)) * 100,  # Convert to cents
        }

        return {
            'alert_id': alert['trade_id'],
            'alert_type': alert['type'].value,
            'stock_data': stock_data,
            'options_data': options_data,
            'trade_data': trade_data
        }

    def _calculate_dte(self, expiration_date) -> int:
        """Calculate days to expiration"""
        if not expiration_date:
            return 30  # Default

        if isinstance(expiration_date, str):
            try:
                expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
            except:
                return 30

        if isinstance(expiration_date, datetime):
            expiration_date = expiration_date.date()

        today = datetime.now().date()
        dte = (expiration_date - today).days

        return max(dte, 0)


if __name__ == "__main__":
    # Test alert processor
    processor = AlertProcessor()

    # Example scraped trades
    test_trades = [
        {
            'ticker': 'AAPL',
            'strategy': 'CSP',
            'action': 'BTO',
            'entry_price': 2.50,
            'strike_price': 170.0,
            'expiration_date': '2025-12-06',
            'alert_text': 'BTO AAPL $170 PUT @ $2.50 - 30 DTE',
            'alert_timestamp': datetime.now(),
            'xtrades_alert_id': 'test_aapl_csp_001'
        }
    ]

    # Process
    results = processor.process_scrape_results('behappy', test_trades)

    print(f"\n✅ Alert Processing Test Results:")
    print(f"New alerts: {len(results['new_alerts'])}")
    print(f"Updated alerts: {len(results['updated_alerts'])}")
    print(f"Closed alerts: {len(results['closed_alerts'])}")

    if results['new_alerts']:
        alert = results['new_alerts'][0]
        enriched = processor.enrich_alert_with_market_data(alert)
        prepared = processor.prepare_for_evaluation(enriched)

        print(f"\n📊 Prepared for evaluation:")
        print(f"Stock: {prepared['stock_data']['symbol']}")
        print(f"Strike: ${prepared['options_data']['strike_price']}")
        print(f"DTE: {prepared['options_data']['dte']}")
        print(f"Premium: {prepared['options_data']['premium']} cents")
