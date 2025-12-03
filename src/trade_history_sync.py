"""
Trade History Sync Service
Syncs closed trades from Robinhood to database for fast retrieval
Eliminates slow API calls on every page load
"""

import os
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import robin_stocks.robinhood as rh

load_dotenv()

class TradeHistorySyncService:
    def __init__(self) -> None:
        self.db_params = {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }

    def get_db_connection(self) -> None:
        """Create database connection"""
        return psycopg2.connect(**self.db_params)

    def sync_trades_from_robinhood(self, rh_session):
        """
        Sync closed trades from Robinhood to database
        Only syncs trades that aren't already in the database
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            all_orders = rh_session.get_all_option_orders()

            # Track trades we've seen (to avoid duplicates)
            open_orders = {}  # key: (symbol, strike, exp, type) -> order
            synced_count = 0

            for order in all_orders:
                if order.get('state') != 'filled':
                    continue

                legs = order.get('legs', [])
                if not legs:
                    continue

                leg = legs[0]
                side = leg.get('side')
                position_effect = leg.get('position_effect')

                opt_url = leg.get('option')
                if not opt_url:
                    continue

                # Extract option ID
                if isinstance(opt_url, str) and 'options/instruments/' in opt_url:
                    opt_id = opt_url.split('/')[-2]
                else:
                    opt_id = opt_url

                # Get option details
                opt_data = rh_session.get_option_instrument_data_by_id(opt_id)
                if not opt_data:
                    continue

                symbol = opt_data.get('chain_symbol', 'Unknown')
                strike = float(opt_data.get('strike_price', 0))
                exp_date = opt_data.get('expiration_date', 'Unknown')
                opt_type = opt_data.get('type', 'unknown')

                # Trade details
                quantity = float(order.get('quantity', 0))
                processed_premium = float(order.get('processed_premium', 0))
                premium_direction = order.get('processed_premium_direction', 'debit')

                # Credit = you received money (positive), Debit = you paid money (negative)
                if premium_direction == 'credit':
                    premium_value = processed_premium
                else:  # debit
                    premium_value = -processed_premium

                # Parse trade date
                created_at = order.get('created_at', '')
                if created_at:
                    trade_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    trade_date = datetime.now()

                # Create unique trade key
                trade_key = (symbol, strike, exp_date, opt_type)

                # Opening trade (you sold the option = collected premium)
                if position_effect == 'open':
                    if side == 'sell':  # Cash-secured put
                        open_orders[trade_key] = {
                            'symbol': symbol,
                            'strike': strike,
                            'expiration': exp_date,
                            'option_type': opt_type,
                            'open_date': trade_date,
                            'open_premium': premium_value,
                            'contracts': int(quantity)
                        }

                # Closing trade (you bought back the option)
                elif position_effect == 'close':
                    if side == 'buy' and trade_key in open_orders:
                        open_trade = open_orders[trade_key]

                        # Calculate P/L
                        open_premium = open_trade['open_premium']
                        close_premium = premium_value  # This will be negative (you paid to close)
                        total_pl = open_premium + close_premium  # Net premium

                        # Calculate days held
                        days_held = (trade_date - open_trade['open_date']).days

                        # Check if already in database
                        cursor.execute("""
                            SELECT id FROM trade_history
                            WHERE symbol = %s
                            AND strike_price = %s
                            AND expiration_date = %s
                            AND open_date = %s
                            AND close_date = %s
                        """, (symbol, strike, exp_date, open_trade['open_date'], trade_date))

                        if cursor.fetchone():
                            # Already exists, skip
                            continue

                        # Insert into database
                        cursor.execute("""
                            INSERT INTO trade_history (
                                symbol, strategy_type, open_date, strike_price,
                                expiration_date, premium_collected, contracts,
                                close_date, close_price, close_reason,
                                days_held, profit_loss, status
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                        """, (
                            symbol,
                            'cash_secured_put' if opt_type == 'put' else 'covered_call',
                            open_trade['open_date'],
                            strike,
                            exp_date,
                            open_premium,
                            open_trade['contracts'],
                            trade_date,
                            abs(close_premium),
                            'early_close',
                            days_held,
                            total_pl,
                            'closed'
                        ))

                        synced_count += 1

                        # Remove from open orders
                        del open_orders[trade_key]

            conn.commit()
            print(f"✅ Synced {synced_count} closed trades to database")
            return synced_count

        except Exception as e:
            conn.rollback()
            print(f"❌ Error syncing trades: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def get_closed_trades_from_db(self, days_back=365):
        """
        Retrieve closed trades from database
        Much faster than fetching from Robinhood API
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            # Get trades from last X days
            cutoff_date = datetime.now() - timedelta(days=days_back)

            cursor.execute("""
                SELECT
                    symbol, strategy_type, open_date, strike_price,
                    expiration_date, premium_collected, contracts,
                    close_date, close_price, close_reason,
                    days_held, profit_loss, status
                FROM trade_history
                WHERE status = 'closed'
                AND close_date >= %s
                ORDER BY close_date DESC
            """, (cutoff_date,))

            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'symbol': row[0],
                    'strategy_type': row[1],
                    'open_date': row[2],
                    'strike': row[3],
                    'expiration': row[4],
                    'premium_collected': float(row[5]),
                    'contracts': row[6],
                    'close_date': row[7],
                    'close_price': float(row[8]),
                    'close_reason': row[9],
                    'days_held': row[10],
                    'profit_loss': float(row[11]),
                    'status': row[12]
                })

            return trades

        except Exception as e:
            print(f"❌ Error fetching trades from database: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_last_sync_time(self) -> None:
        """Check when trades were last synced"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT MAX(updated_at) FROM trade_history
            """)
            result = cursor.fetchone()
            return result[0] if result[0] else None
        except:
            return None
        finally:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    # Test the sync service
    print("🔄 Starting trade history sync...")

    # Login to Robinhood
    rh_user = os.getenv('RH_USERNAME')
    rh_pass = os.getenv('RH_PASSWORD')

    if rh_user and rh_pass:
        rh.login(rh_user, rh_pass)

        service = TradeHistorySyncService()
        count = service.sync_trades_from_robinhood(rh)

        print(f"\n✅ Sync complete: {count} new trades added to database")

        # Test retrieval
        trades = service.get_closed_trades_from_db(days_back=90)
        print(f"📊 Retrieved {len(trades)} closed trades from last 90 days")

        rh.logout()
    else:
        print("❌ Robinhood credentials not found in .env file")
