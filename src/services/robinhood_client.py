"""
Robinhood Client Service
Thread-safe singleton Robinhood client with rate limiting, retry logic, and automatic token refresh
"""

import robin_stocks.robinhood as rh
import os
import threading
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pickle
import pyotp
from loguru import logger

from src.services.config import get_service_config
from src.services.rate_limiter import rate_limit


class RobinhoodClient:
    """
    Singleton Robinhood client with automatic session management

    Features:
    - Thread-safe singleton pattern
    - Automatic token refresh
    - Rate limiting (60 requests/minute)
    - Exponential backoff retry logic
    - Session caching
    - Comprehensive error handling
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize Robinhood client (singleton pattern)"""
        if not hasattr(self, '_initialized'):
            self.username = os.getenv('ROBINHOOD_USERNAME')
            self.password = os.getenv('ROBINHOOD_PASSWORD')
            self.mfa_code = os.getenv('ROBINHOOD_MFA_CODE')
            self.logged_in = False
            self.session_file = Path.home() / '.robinhood_session.pickle'
            self._login_lock = threading.Lock()
            self.config = get_service_config("robinhood")
            self._initialized = True
            logger.info("Robinhood client initialized")

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry logic

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result of function execution

        Raises:
            Last exception if all retries fail
        """
        retry_policy = self.config.retry_policy
        last_exception = None

        for attempt in range(retry_policy.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if we should retry
                if attempt < retry_policy.max_retries:
                    delay = retry_policy.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{retry_policy.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    raise last_exception

        raise last_exception

    def _ensure_logged_in(self) -> None:
        """Ensure client is logged in, login if needed"""
        if not self.logged_in:
            with self._login_lock:
                if not self.logged_in:  # Double-check after acquiring lock
                    self.login()

    def login(self, force_fresh: bool = False) -> bool:
        """
        Login to Robinhood with session caching

        Args:
            force_fresh: Force fresh login even if cached session exists

        Returns:
            True if login successful, False otherwise
        """
        if not self.username or not self.password:
            logger.error("Missing Robinhood credentials in environment variables")
            return False

        try:
            # Try cached session first (unless forcing fresh login)
            if not force_fresh and self.session_file.exists():
                try:
                    with open(self.session_file, 'rb') as f:
                        pickle.load(f)
                        rh.authentication.set_login_state(True)

                    # Verify session is still valid
                    profile = rh.profiles.load_account_profile()
                    if profile:
                        self.logged_in = True
                        logger.info("Restored cached Robinhood session")
                        return True
                    else:
                        logger.info("Cached session expired, performing fresh login")
                        self.session_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to restore cached session: {e}")
                    if self.session_file.exists():
                        self.session_file.unlink()

            # Fresh login
            logger.info("Performing fresh Robinhood login...")

            if self.mfa_code:
                # Use TOTP for MFA
                totp = pyotp.TOTP(self.mfa_code).now()
                login_result = rh.login(
                    username=self.username,
                    password=self.password,
                    mfa_code=totp,
                    store_session=True
                )
            else:
                # Regular login (may prompt for MFA)
                login_result = rh.login(
                    username=self.username,
                    password=self.password,
                    store_session=True
                )

            if login_result:
                # Save session
                with open(self.session_file, 'wb') as f:
                    pickle.dump(login_result, f)

                self.logged_in = True
                logger.info("Successfully logged in to Robinhood")
                return True
            else:
                logger.error("Robinhood login failed")
                return False

        except Exception as e:
            logger.error(f"Error during Robinhood login: {e}")
            self.logged_in = False
            return False

    def logout(self) -> None:
        """Logout from Robinhood"""
        try:
            rh.logout()
            self.logged_in = False

            # Clean up session file
            if self.session_file.exists():
                self.session_file.unlink()

            logger.info("Logged out from Robinhood")
        except Exception as e:
            logger.error(f"Error during logout: {e}")

    @rate_limit("robinhood", tokens=1)
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information

        Returns:
            Dict with account details including buying power, portfolio value, etc.
        """
        self._ensure_logged_in()

        def _fetch():
            try:
                profile = rh.profiles.load_account_profile()
                portfolio = rh.profiles.load_portfolio_profile()

                return {
                    'account_number': profile.get('account_number'),
                    'buying_power': float(profile.get('buying_power', 0)),
                    'cash': float(profile.get('cash', 0)),
                    'portfolio_value': float(portfolio.get('market_value', 0) if portfolio else 0),
                    'total_return': float(portfolio.get('equity', 0) if portfolio else 0),
                    'day_trade_count': profile.get('day_trade_count', 0),
                    'updated_at': datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error fetching account info: {e}")
                return {}

        return self._retry_with_backoff(_fetch)

    @rate_limit("robinhood", tokens=1)
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all positions (stocks and options combined)

        Returns:
            List of position dicts
        """
        self._ensure_logged_in()

        positions = []
        positions.extend(self.get_stock_positions())
        positions.extend(self.get_options_positions())

        return positions

    @rate_limit("robinhood", tokens=1)
    def get_stock_positions(self) -> List[Dict[str, Any]]:
        """
        Get all stock positions

        Returns:
            List of stock position dicts
        """
        self._ensure_logged_in()

        def _fetch():
            try:
                positions = rh.account.get_open_stock_positions()
                formatted_positions = []

                for position in positions:
                    if float(position['quantity']) <= 0:
                        continue

                    symbol = rh.stocks.get_symbol_by_url(position['instrument'])

                    pos_data = {
                        'type': 'stock',
                        'symbol': symbol,
                        'quantity': float(position['quantity']),
                        'avg_cost': float(position['average_buy_price']),
                        'current_price': 0.0,
                        'market_value': 0.0,
                        'total_return': 0.0,
                        'total_return_pct': 0.0,
                        'created_at': position.get('created_at'),
                        'updated_at': position.get('updated_at')
                    }

                    # Get current price
                    try:
                        quote = rh.stocks.get_latest_price(symbol)
                        if quote and len(quote) > 0:
                            current_price = float(quote[0])
                            pos_data['current_price'] = current_price
                            pos_data['market_value'] = current_price * pos_data['quantity']
                            pos_data['total_return'] = (current_price - pos_data['avg_cost']) * pos_data['quantity']

                            if pos_data['avg_cost'] != 0:
                                pos_data['total_return_pct'] = (
                                    (current_price - pos_data['avg_cost']) / pos_data['avg_cost']
                                ) * 100
                    except Exception as e:
                        logger.warning(f"Failed to get price for {symbol}: {e}")

                    formatted_positions.append(pos_data)

                return formatted_positions

            except Exception as e:
                logger.error(f"Error fetching stock positions: {e}")
                return []

        return self._retry_with_backoff(_fetch)

    @rate_limit("robinhood", tokens=1)
    def get_options_positions(self) -> List[Dict[str, Any]]:
        """
        Get all option positions

        Returns:
            List of option position dicts
        """
        self._ensure_logged_in()

        def _fetch():
            try:
                options = rh.options.get_open_option_positions()
                formatted_options = []

                for option in options:
                    quantity = float(option.get('quantity', 0))
                    if quantity == 0:
                        continue

                    # Get option details
                    option_id = option.get('option_id') or option.get('option')
                    if not option_id:
                        continue

                    try:
                        option_data = rh.options.get_option_market_data_by_id(option_id)
                    except:
                        continue

                    opt_data = {
                        'type': 'option',
                        'symbol': option.get('chain_symbol'),
                        'option_type': option.get('type', 'unknown'),  # 'call' or 'put'
                        'position_type': 'short' if quantity < 0 else 'long',
                        'quantity': abs(quantity),
                        'strike_price': float(option_data.get('strike_price', 0)),
                        'expiration_date': option_data.get('expiration_date'),
                        'avg_cost': float(option.get('average_price', 0)) / 100,  # Convert cents to dollars
                        'current_price': 0.0,
                        'market_value': 0.0,
                        'total_return': 0.0,
                        'total_return_pct': 0.0,
                        'created_at': option.get('created_at'),
                        'updated_at': option.get('updated_at')
                    }

                    # Get current price
                    try:
                        mark_price = float(option_data.get('mark_price', 0))
                        opt_data['current_price'] = mark_price
                        opt_data['market_value'] = mark_price * opt_data['quantity'] * 100

                        # For short positions, profit is negative when price goes up
                        multiplier = -1 if opt_data['position_type'] == 'short' else 1
                        opt_data['total_return'] = (
                            (mark_price - opt_data['avg_cost']) * opt_data['quantity'] * 100 * multiplier
                        )

                        if opt_data['avg_cost'] != 0:
                            opt_data['total_return_pct'] = (
                                (mark_price - opt_data['avg_cost']) / opt_data['avg_cost']
                            ) * 100 * multiplier
                    except Exception as e:
                        logger.warning(f"Failed to get price for option: {e}")

                    formatted_options.append(opt_data)

                return formatted_options

            except Exception as e:
                logger.error(f"Error fetching options positions: {e}")
                return []

        return self._retry_with_backoff(_fetch)

    @rate_limit("robinhood", tokens=1)
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dict with market data
        """
        self._ensure_logged_in()

        def _fetch():
            try:
                quote = rh.stocks.get_quotes(symbol)
                if not quote or len(quote) == 0:
                    return {}

                data = quote[0]
                return {
                    'symbol': symbol,
                    'price': float(data.get('last_trade_price', 0)),
                    'bid': float(data.get('bid_price', 0)),
                    'ask': float(data.get('ask_price', 0)),
                    'volume': int(data.get('volume', 0)),
                    'previous_close': float(data.get('previous_close', 0)),
                    'updated_at': data.get('updated_at')
                }
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")
                return {}

        return self._retry_with_backoff(_fetch)

    @rate_limit("robinhood", tokens=2)
    def get_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get options chain for a symbol

        Args:
            symbol: Stock symbol
            expiration_date: Expiration date (YYYY-MM-DD), None for all expirations

        Returns:
            List of option contracts
        """
        self._ensure_logged_in()

        def _fetch():
            try:
                # Get option chain
                if expiration_date:
                    chains = rh.options.find_options_by_expiration(
                        symbol,
                        expirationDate=expiration_date,
                        optionType='both'
                    )
                else:
                    chains = rh.options.find_options_for_stock_info(symbol)

                return chains if chains else []

            except Exception as e:
                logger.error(f"Error fetching options chain for {symbol}: {e}")
                return []

        return self._retry_with_backoff(_fetch)

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status and health check

        Returns:
            Dict with connection status info
        """
        return {
            'logged_in': self.logged_in,
            'username': self.username[:3] + '***' if self.username else None,
            'session_file_exists': self.session_file.exists(),
            'config': {
                'rate_limit': f"{self.config.rate_limit.max_calls}/{self.config.rate_limit.time_window}s",
                'timeout': self.config.timeout
            }
        }


# =============================================================================
# Singleton Access
# =============================================================================

_robinhood_client = None
_client_lock = threading.Lock()


def get_robinhood_client() -> RobinhoodClient:
    """
    Get the global Robinhood client instance

    Returns:
        Singleton RobinhoodClient instance
    """
    global _robinhood_client

    if _robinhood_client is None:
        with _client_lock:
            if _robinhood_client is None:
                _robinhood_client = RobinhoodClient()

    return _robinhood_client


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing Robinhood Client")
    print("=" * 60)

    client = get_robinhood_client()

    # Test 1: Connection status
    print("\nTest 1: Connection Status")
    status = client.get_connection_status()
    print(f"  Logged in: {status['logged_in']}")
    print(f"  Username: {status['username']}")
    print(f"  Rate limit: {status['config']['rate_limit']}")

    # Test 2: Login
    print("\nTest 2: Login")
    if client.login():
        print("  ✓ Login successful")

        # Test 3: Account info
        print("\nTest 3: Account Info")
        account = client.get_account_info()
        if account:
            print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
            print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        else:
            print("  ✗ Failed to get account info")

        # Test 4: Positions
        print("\nTest 4: Positions")
        positions = client.get_positions()
        print(f"  Total positions: {len(positions)}")
        for pos in positions[:5]:  # Show first 5
            print(f"    {pos['symbol']}: {pos['type']}")

        # Test 5: Market data
        print("\nTest 5: Market Data")
        data = client.get_market_data('AAPL')
        if data:
            print(f"  AAPL: ${data.get('price', 0):.2f}")
        else:
            print("  ✗ Failed to get market data")

        # Logout
        client.logout()
        print("\n✓ Logged out")
    else:
        print("  ✗ Login failed")

    print("\n" + "=" * 60)
    print("Robinhood client tests complete!")
