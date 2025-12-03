"""
Trade History Manager - Manage trade lifecycle and P&L calculations
"""
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any
from src.tradingview_db_manager import TradingViewDBManager
import logging

logger = logging.getLogger(__name__)


class TradeHistoryManager:
    """Manages trade history for cash-secured puts and other options strategies"""

    def __init__(self) -> None:
        self.db = TradingViewDBManager()

    def add_trade(
        self,
        symbol: str,
        strike_price: float,
        expiration_date: str,
        premium_collected: float,
        contracts: int = 1,
        open_date: Optional[str] = None,
        strategy_type: str = 'cash_secured_put',
        notes: Optional[str] = None
    ) -> int:
        """
        Open a new trade (sell option)

        Args:
            symbol: Stock ticker symbol
            strike_price: Strike price of option
            expiration_date: Expiration date (YYYY-MM-DD)
            premium_collected: Premium collected per contract ($)
            contracts: Number of contracts (default 1)
            open_date: Trade open date (default today)
            strategy_type: Type of strategy (default 'cash_secured_put')
            notes: Optional notes about the trade

        Returns:
            trade_id: ID of created trade
        """
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            # Calculate DTE at open
            if open_date:
                open_dt = datetime.strptime(open_date, '%Y-%m-%d')
            else:
                open_dt = datetime.now()
                open_date = open_dt.strftime('%Y-%m-%d')

            exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
            dte_at_open = (exp_dt - open_dt).days

            cur.execute("""
                INSERT INTO trade_history (
                    symbol, strike_price, expiration_date, premium_collected,
                    contracts, open_date, dte_at_open, strategy_type, notes, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'open')
                RETURNING id
            """, (
                symbol.upper(), strike_price, expiration_date, premium_collected,
                contracts, open_date, dte_at_open, strategy_type, notes
            ))

            trade_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Trade opened: {symbol} ${strike_price} Put, Premium ${premium_collected}, ID {trade_id}")
            return trade_id

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            if conn:
                conn.rollback()
                cur.close()
                conn.close()
            raise

    def close_trade(
        self,
        trade_id: int,
        close_price: float,
        close_reason: str = 'early_close',
        close_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Close an existing trade (buy back option or let expire)

        Args:
            trade_id: ID of trade to close
            close_price: Price paid to buy back option (0 if expired worthless)
            close_reason: Reason for close ('early_close', 'expiration', 'assignment')
            close_date: Close date (default today)

        Returns:
            dict with P&L details
        """
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            # Get trade details
            cur.execute("""
                SELECT symbol, strike_price, expiration_date, premium_collected,
                       contracts, open_date, dte_at_open
                FROM trade_history
                WHERE id = %s AND status = 'open'
            """, (trade_id,))

            row = cur.fetchone()
            if not row:
                raise ValueError(f"Trade {trade_id} not found or already closed")

            symbol, strike, exp_date, premium, contracts, open_date, dte = row

            # Calculate close date and days held
            if close_date:
                close_dt = datetime.strptime(close_date, '%Y-%m-%d')
            else:
                close_dt = datetime.now()
                close_date = close_dt.strftime('%Y-%m-%d')

            if isinstance(open_date, str):
                open_dt = datetime.strptime(open_date, '%Y-%m-%d')
            else:
                # Convert timezone-aware datetime to naive datetime
                open_dt = open_date.replace(tzinfo=None) if hasattr(open_date, 'tzinfo') and open_date.tzinfo else open_date

            days_held = (close_dt - open_dt).days
            if days_held == 0:
                days_held = 1  # Minimum 1 day for annualized calculation

            # Calculate P&L
            profit_loss = float(premium) - float(close_price)
            profit_loss_pct = (profit_loss / float(premium)) * 100 if premium > 0 else 0

            # Calculate annualized return
            annualized_return = (profit_loss_pct / days_held) * 365 if days_held > 0 else 0

            # Determine status
            if close_reason == 'assignment':
                status = 'assigned'
            else:
                status = 'closed'

            # Update trade
            cur.execute("""
                UPDATE trade_history
                SET close_date = %s,
                    close_price = %s,
                    close_reason = %s,
                    days_held = %s,
                    profit_loss = %s,
                    profit_loss_percent = %s,
                    annualized_return = %s,
                    status = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                close_date, close_price, close_reason, days_held,
                profit_loss, profit_loss_pct, annualized_return,
                status, trade_id
            ))

            conn.commit()
            cur.close()
            conn.close()

            result = {
                'trade_id': trade_id,
                'symbol': symbol,
                'profit_loss': profit_loss,
                'profit_loss_percent': profit_loss_pct,
                'days_held': days_held,
                'annualized_return': annualized_return
            }

            logger.info(f"Trade closed: ID {trade_id}, P/L ${profit_loss:.2f} ({profit_loss_pct:.1f}%), {days_held} days")
            return result

        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            if conn:
                conn.rollback()
                cur.close()
                conn.close()
            raise

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT id, symbol, strike_price, expiration_date, premium_collected,
                       contracts, open_date, dte_at_open, strategy_type, notes
                FROM trade_history
                WHERE status = 'open'
                ORDER BY expiration_date ASC
            """)

            rows = cur.fetchall()
            cur.close()
            conn.close()

            trades = []
            for row in rows:
                trades.append({
                    'id': row[0],
                    'symbol': row[1],
                    'strike_price': float(row[2]),
                    'expiration_date': row[3],
                    'premium_collected': float(row[4]),
                    'contracts': row[5],
                    'open_date': row[6],
                    'dte_at_open': row[7],
                    'strategy_type': row[8],
                    'notes': row[9]
                })

            return trades

        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
            if conn:
                cur.close()
                conn.close()
            return []

    def get_closed_trades(
        self,
        limit: int = 100,
        symbol: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get closed trade history with optional filters"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            query = """
                SELECT id, symbol, strike_price, expiration_date, premium_collected,
                       contracts, open_date, close_date, close_price, close_reason,
                       days_held, profit_loss, profit_loss_percent, annualized_return,
                       strategy_type, status
                FROM trade_history
                WHERE status IN ('closed', 'assigned')
            """
            params = []

            if symbol:
                query += " AND symbol = %s"
                params.append(symbol.upper())

            if date_from:
                query += " AND close_date >= %s"
                params.append(date_from)

            if date_to:
                query += " AND close_date <= %s"
                params.append(date_to)

            query += " ORDER BY close_date DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            cur.close()
            conn.close()

            trades = []
            for row in rows:
                trades.append({
                    'id': row[0],
                    'symbol': row[1],
                    'strike_price': float(row[2]),
                    'expiration_date': row[3],
                    'premium_collected': float(row[4]),
                    'contracts': row[5],
                    'open_date': row[6],
                    'close_date': row[7],
                    'close_price': float(row[8]) if row[8] else 0,
                    'close_reason': row[9],
                    'days_held': row[10],
                    'profit_loss': float(row[11]) if row[11] else 0,
                    'profit_loss_percent': float(row[12]) if row[12] else 0,
                    'annualized_return': float(row[13]) if row[13] else 0,
                    'strategy_type': row[14],
                    'status': row[15]
                })

            return trades

        except Exception as e:
            logger.error(f"Error getting closed trades: {e}")
            if conn:
                cur.close()
                conn.close()
            return []

    def get_trade_stats(self) -> Dict[str, Any]:
        """Calculate aggregate statistics for all closed trades"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            # Get aggregate stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    COALESCE(SUM(profit_loss), 0) as total_pl,
                    COALESCE(AVG(days_held), 0) as avg_days_held,
                    COALESCE(MAX(profit_loss), 0) as best_trade,
                    COALESCE(MIN(profit_loss), 0) as worst_trade,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as winning_trades
                FROM trade_history
                WHERE status IN ('closed', 'assigned')
            """)

            row = cur.fetchone()

            total_trades = row[0]
            total_pl = float(row[1])
            avg_days = int(row[2])
            best = float(row[3])
            worst = float(row[4])
            wins = row[5]

            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            # Get best trade details
            cur.execute("""
                SELECT symbol, profit_loss, profit_loss_percent
                FROM trade_history
                WHERE status IN ('closed', 'assigned')
                ORDER BY profit_loss DESC
                LIMIT 1
            """)
            best_trade_row = cur.fetchone()
            best_trade = {
                'symbol': best_trade_row[0] if best_trade_row else '',
                'profit_loss': float(best_trade_row[1]) if best_trade_row else 0,
                'profit_loss_percent': float(best_trade_row[2]) if best_trade_row else 0
            }

            # Get worst trade details
            cur.execute("""
                SELECT symbol, profit_loss, profit_loss_percent
                FROM trade_history
                WHERE status IN ('closed', 'assigned')
                ORDER BY profit_loss ASC
                LIMIT 1
            """)
            worst_trade_row = cur.fetchone()
            worst_trade = {
                'symbol': worst_trade_row[0] if worst_trade_row else '',
                'profit_loss': float(worst_trade_row[1]) if worst_trade_row else 0,
                'profit_loss_percent': float(worst_trade_row[2]) if worst_trade_row else 0
            }

            cur.close()
            conn.close()

            return {
                'total_trades': total_trades,
                'total_pl': total_pl,
                'win_rate': win_rate,
                'avg_days_held': avg_days,
                'best_trade': best_trade,
                'worst_trade': worst_trade
            }

        except Exception as e:
            logger.error(f"Error getting trade stats: {e}")
            if conn:
                cur.close()
                conn.close()
            return {
                'total_trades': 0,
                'total_pl': 0.0,
                'win_rate': 0.0,
                'avg_days_held': 0,
                'best_trade': {'symbol': '', 'profit_loss': 0, 'profit_loss_percent': 0},
                'worst_trade': {'symbol': '', 'profit_loss': 0, 'profit_loss_percent': 0}
            }

    def get_cumulative_pl(self) -> List[Dict[str, Any]]:
        """Get cumulative P/L over time for charting"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT close_date, profit_loss
                FROM trade_history
                WHERE status IN ('closed', 'assigned')
                ORDER BY close_date ASC
            """)

            rows = cur.fetchall()
            cur.close()
            conn.close()

            cumulative = []
            running_total = 0

            for row in rows:
                close_date = row[0]
                pl = float(row[1]) if row[1] else 0
                running_total += pl

                cumulative.append({
                    'date': close_date.strftime('%Y-%m-%d') if isinstance(close_date, (datetime, date)) else close_date,
                    'cumulative_pl': running_total
                })

            return cumulative

        except Exception as e:
            logger.error(f"Error getting cumulative P/L: {e}")
            if conn:
                cur.close()
                conn.close()
            return []

    def get_trade_by_id(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific trade by ID"""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT id, symbol, strike_price, expiration_date, premium_collected,
                       contracts, open_date, close_date, close_price, close_reason,
                       days_held, profit_loss, profit_loss_percent, annualized_return,
                       strategy_type, status, notes
                FROM trade_history
                WHERE id = %s
            """, (trade_id,))

            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return None

            return {
                'id': row[0],
                'symbol': row[1],
                'strike_price': float(row[2]),
                'expiration_date': row[3],
                'premium_collected': float(row[4]),
                'contracts': row[5],
                'open_date': row[6],
                'close_date': row[7],
                'close_price': float(row[8]) if row[8] else None,
                'close_reason': row[9],
                'days_held': row[10],
                'profit_loss': float(row[11]) if row[11] else None,
                'profit_loss_percent': float(row[12]) if row[12] else None,
                'annualized_return': float(row[13]) if row[13] else None,
                'strategy_type': row[14],
                'status': row[15],
                'notes': row[16]
            }

        except Exception as e:
            logger.error(f"Error getting trade by ID: {e}")
            if conn:
                cur.close()
                conn.close()
            return None
