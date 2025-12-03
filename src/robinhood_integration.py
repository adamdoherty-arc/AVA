"""Robinhood Integration for pulling positions and account data"""

import robin_stocks.robinhood as rh
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import json
from pathlib import Path
import pyotp

class RobinhoodClient:
    """Client for interacting with Robinhood API"""

    def __init__(self, username: str = None, password: str = None, mfa_code: str = None):
        self.username = username or os.getenv('ROBINHOOD_USERNAME')
        self.password = password or os.getenv('ROBINHOOD_PASSWORD')
        self.mfa_code = mfa_code or os.getenv('ROBINHOOD_MFA_CODE')
        self.logged_in = False

    def login(self, store_session: bool = True) -> bool:
        """Login to Robinhood"""
        try:
            if store_session:
                # Try to use stored session first
                pickle_path = Path.home() / '.robinhood_session.pickle'
                if pickle_path.exists():
                    try:
                        login = rh.login(username=self.username, password=self.password, pickle_path=str(pickle_path))
                        self.logged_in = True
                        logger.info("Logged in using stored session")
                        return True
                    except:
                        pass

            # Login with MFA if provided
            if self.mfa_code:
                totp = pyotp.TOTP(self.mfa_code).now()
                login = rh.login(
                    username=self.username,
                    password=self.password,
                    mfa_code=totp,
                    store_session=store_session
                )
            else:
                # Try regular login
                login = rh.login(
                    username=self.username,
                    password=self.password,
                    store_session=store_session
                )

            self.logged_in = True
            logger.info("Successfully logged in to Robinhood")
            return True

        except Exception as e:
            logger.error(f"Failed to login to Robinhood: {e}")
            self.logged_in = False
            return False

    def logout(self) -> None:
        """Logout from Robinhood"""
        rh.logout()
        self.logged_in = False
        logger.info("Logged out from Robinhood")

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.logged_in:
            logger.error("Not logged in to Robinhood")
            return {}

        try:
            profile = rh.profiles.load_account_profile()
            positions = rh.profiles.load_portfolio_profile()

            return {
                'account_number': profile.get('account_number'),
                'buying_power': float(profile.get('buying_power', 0)),
                'total_value': float(positions.get('market_value', 0)),
                'cash': float(profile.get('cash', 0)),
                'portfolio_value': float(positions.get('total_return_today', 0)),
                'day_trade_count': profile.get('day_trade_count', 0),
                'updated_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def get_stock_positions(self) -> List[Dict[str, Any]]:
        """Get all stock positions"""
        if not self.logged_in:
            logger.error("Not logged in to Robinhood")
            return []

        try:
            positions = rh.account.get_open_stock_positions()

            formatted_positions = []
            for position in positions:
                symbol = rh.stocks.get_symbol_by_url(position['instrument'])

                if float(position['quantity']) > 0:
                    pos_data = {
                        'symbol': symbol,
                        'quantity': float(position['quantity']),
                        'avg_cost': float(position['average_buy_price']),
                        'current_price': 0,  # Will be updated separately
                        'market_value': 0,
                        'total_return': 0,
                        'total_return_pct': 0,
                        'position_type': 'stock',
                        'created_at': position.get('created_at'),
                        'updated_at': position.get('updated_at')
                    }

                    # Get current price
                    try:
                        quote = rh.stocks.get_latest_price(symbol)[0]
                        pos_data['current_price'] = float(quote)
                        pos_data['market_value'] = pos_data['current_price'] * pos_data['quantity']
                        pos_data['total_return'] = (pos_data['current_price'] - pos_data['avg_cost']) * pos_data['quantity']
                        pos_data['total_return_pct'] = ((pos_data['current_price'] - pos_data['avg_cost']) / pos_data['avg_cost']) * 100
                    except:
                        pass

                    formatted_positions.append(pos_data)

            return formatted_positions

        except Exception as e:
            logger.error(f"Error getting stock positions: {e}")
            return []

    def get_option_positions(self) -> List[Dict[str, Any]]:
        """Get all option positions"""
        if not self.logged_in:
            logger.error("Not logged in to Robinhood")
            return []

        try:
            options = rh.options.get_open_option_positions()

            formatted_options = []
            for option in options:
                if float(option['quantity']) > 0:
                    # Get option details
                    option_data = rh.options.get_option_market_data_by_id(option['option_id'])

                    opt_data = {
                        'symbol': option['symbol'],
                        'option_type': option['type'],  # 'call' or 'put'
                        'position_type': 'short' if float(option['quantity']) < 0 else 'long',
                        'quantity': abs(float(option['quantity'])),
                        'strike_price': float(option_data['strike_price']),
                        'expiration_date': option_data['expiration_date'],
                        'avg_cost': float(option['average_price']) / 100,  # Convert cents to dollars
                        'current_price': 0,
                        'market_value': 0,
                        'total_return': 0,
                        'total_return_pct': 0,
                        'created_at': option.get('created_at'),
                        'updated_at': option.get('updated_at')
                    }

                    # Get current price
                    try:
                        current = rh.options.get_option_market_data_by_id(option['option_id'])
                        mark_price = float(current['mark_price'])
                        opt_data['current_price'] = mark_price
                        opt_data['market_value'] = mark_price * opt_data['quantity'] * 100
                        opt_data['total_return'] = (mark_price - opt_data['avg_cost']) * opt_data['quantity'] * 100

                        if opt_data['avg_cost'] != 0:
                            opt_data['total_return_pct'] = ((mark_price - opt_data['avg_cost']) / opt_data['avg_cost']) * 100
                    except:
                        pass

                    formatted_options.append(opt_data)

            return formatted_options

        except Exception as e:
            logger.error(f"Error getting option positions: {e}")
            return []

    def get_all_positions(self) -> Dict[str, Any]:
        """Get all positions (stocks and options)"""
        if not self.logged_in:
            logger.error("Not logged in to Robinhood")
            return {'stocks': [], 'options': []}

        return {
            'stocks': self.get_stock_positions(),
            'options': self.get_option_positions(),
            'account': self.get_account_info(),
            'updated_at': datetime.now().isoformat()
        }

    def get_wheel_positions(self) -> List[Dict[str, Any]]:
        """Get positions relevant to wheel strategy"""
        if not self.logged_in:
            return []

        wheel_positions = []

        # Get all positions
        all_positions = self.get_all_positions()

        # Process stock positions (potential covered call candidates)
        for stock in all_positions['stocks']:
            if stock['quantity'] >= 100:  # Need at least 100 shares for covered calls
                wheel_positions.append({
                    'symbol': stock['symbol'],
                    'strategy': 'potential_cc',
                    'position_type': 'stock',
                    'shares': stock['quantity'],
                    'cost_basis': stock['avg_cost'],
                    'current_price': stock['current_price'],
                    'market_value': stock['market_value'],
                    'contracts_available': int(stock['quantity'] / 100),
                    'unrealized_pnl': stock['total_return'],
                    'unrealized_pnl_pct': stock['total_return_pct']
                })

        # Process option positions
        for option in all_positions['options']:
            if option['option_type'] == 'put' and option['position_type'] == 'short':
                # Cash-secured put
                wheel_positions.append({
                    'symbol': option['symbol'],
                    'strategy': 'csp',
                    'position_type': 'put',
                    'contracts': option['quantity'],
                    'strike': option['strike_price'],
                    'expiration': option['expiration_date'],
                    'premium_collected': option['avg_cost'] * option['quantity'] * 100,
                    'current_value': option['market_value'],
                    'unrealized_pnl': -option['total_return'],  # Negative because we sold
                    'days_to_expiry': self._days_to_expiry(option['expiration_date'])
                })

            elif option['option_type'] == 'call' and option['position_type'] == 'short':
                # Covered call
                wheel_positions.append({
                    'symbol': option['symbol'],
                    'strategy': 'cc',
                    'position_type': 'call',
                    'contracts': option['quantity'],
                    'strike': option['strike_price'],
                    'expiration': option['expiration_date'],
                    'premium_collected': option['avg_cost'] * option['quantity'] * 100,
                    'current_value': option['market_value'],
                    'unrealized_pnl': -option['total_return'],  # Negative because we sold
                    'days_to_expiry': self._days_to_expiry(option['expiration_date'])
                })

        return wheel_positions

    def _days_to_expiry(self, expiration_date: str) -> int:
        """Calculate days until expiration"""
        try:
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            return (exp_date - datetime.now()).days
        except:
            return 0

    def get_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent orders"""
        if not self.logged_in:
            return []

        try:
            orders = rh.orders.get_all_stock_orders()[:limit]

            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'symbol': order['symbol'],
                    'side': order['side'],  # 'buy' or 'sell'
                    'quantity': float(order['quantity']),
                    'price': float(order.get('executed_price', 0) or order.get('price', 0)),
                    'type': order['type'],
                    'state': order['state'],
                    'created_at': order['created_at'],
                    'executed_at': order.get('executed_at')
                })

            return formatted_orders

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []


# Singleton instance
_robinhood_client = None

def get_robinhood_client() -> RobinhoodClient:
    """Get or create Robinhood client instance"""
    global _robinhood_client
    if _robinhood_client is None:
        _robinhood_client = RobinhoodClient()
    return _robinhood_client


# CLI for testing
if __name__ == "__main__":
    import sys

    print("Robinhood Integration Test")
    print("-" * 40)

    # Get credentials
    username = input("Username/Email: ")
    password = input("Password: ")
    mfa = input("MFA Secret Key (optional, press Enter to skip): ")

    # Create client
    client = RobinhoodClient(username, password, mfa)

    # Login
    if client.login():
        print("\nSuccessfully logged in!")

        # Get account info
        account = client.get_account_info()
        print(f"\nAccount Info:")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"  Total Value: ${account.get('total_value', 0):,.2f}")

        # Get positions
        positions = client.get_all_positions()
        print(f"\nPositions:")
        print(f"  Stocks: {len(positions['stocks'])}")
        print(f"  Options: {len(positions['options'])}")

        # Get wheel positions
        wheel = client.get_wheel_positions()
        print(f"\nWheel Strategy Positions: {len(wheel)}")
        for pos in wheel:
            print(f"  - {pos['symbol']}: {pos['strategy']}")

        # Logout
        client.logout()
    else:
        print("Failed to login. Please check credentials.")