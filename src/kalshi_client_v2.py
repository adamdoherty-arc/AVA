"""
Kalshi API Client v2 - Supports both Email/Password and API Key authentication
"""

import os
import requests
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalshiClientV2:
    """Kalshi API client with support for API key + RSA signing"""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, api_key: Optional[str] = None, private_key_path: Optional[str] = None,
                 email: Optional[str] = None, password: Optional[str] = None,
                 session_token: Optional[str] = None):
        """
        Initialize Kalshi client with API key, email/password, or session token

        Args:
            api_key: Kalshi API key (or set KALSHI_API_KEY env var)
            private_key_path: Path to RSA private key PEM file (or set KALSHI_PRIVATE_KEY_PATH env var)
            email: Kalshi account email (for legacy auth)
            password: Kalshi account password (for legacy auth)
            session_token: Session token from browser (or set KALSHI_SESSION_TOKEN env var)
        """
        # Session token authentication (easiest - from browser login)
        self.session_token = session_token or os.getenv('KALSHI_SESSION_TOKEN')

        # API Key authentication (preferred if no session token)
        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        self.private_key_path = private_key_path or os.getenv('KALSHI_PRIVATE_KEY_PATH')

        # Email/password authentication (fallback)
        self.email = email or os.getenv('KALSHI_EMAIL')
        self.password = password or os.getenv('KALSHI_PASSWORD')

        self.bearer_token = None
        self.token_expires_at = None
        self.private_key = None

        # If session token provided, use it directly
        if self.session_token:
            self.bearer_token = f"Bearer {self.session_token}" if not self.session_token.startswith('Bearer ') else self.session_token
            # Session tokens from browser typically last 24 hours
            self.token_expires_at = datetime.now() + timedelta(hours=24)
            logger.info("Using session token from browser login")

        # Load private key if provided
        if self.private_key_path and os.path.exists(self.private_key_path):
            self._load_private_key()

    def _load_private_key(self) -> None:
        """Load RSA private key from file"""
        try:
            with open(self.private_key_path, 'rb') as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
            logger.info("RSA private key loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            self.private_key = None

    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Sign a request using RSA private key

        Args:
            timestamp: Unix timestamp in milliseconds
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            body: Request body (empty string for GET requests)

        Returns:
            Base64-encoded signature
        """
        if not self.private_key:
            raise ValueError("Private key not loaded")

        # Create signing string
        signing_string = f"{timestamp}{method}{path}{body}"

        # Sign with SHA-256 using PSS padding (Kalshi API requirement)
        signature = self.private_key.sign(
            signing_string.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # Base64 encode
        return base64.b64encode(signature).decode('utf-8')

    def _needs_token_refresh(self) -> bool:
        """Check if token needs refresh"""
        if not self.bearer_token or not self.token_expires_at:
            return True

        # Refresh 5 minutes before expiration
        return datetime.now() >= (self.token_expires_at - timedelta(minutes=5))

    def login_with_api_key(self) -> bool:
        """
        Login using API key and RSA signature

        Returns:
            True if login successful
        """
        if not self.api_key:
            logger.error("API key not provided")
            return False

        if not self.private_key:
            logger.error("Private key not loaded")
            return False

        try:
            # Prepare request
            timestamp = str(int(time.time() * 1000))
            method = "POST"
            path = "/login"
            body = ""

            # Sign request
            signature = self._sign_request(timestamp, method, path, body)

            url = f"{self.BASE_URL}{path}"
            headers = {
                "Content-Type": "application/json",
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-SIGNATURE": signature,
                "KALSHI-ACCESS-TIMESTAMP": timestamp
            }

            response = requests.post(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            token = data.get('token')

            if not token:
                logger.error("No token in API key login response")
                return False

            self.bearer_token = f"Bearer {token}"
            self.token_expires_at = datetime.now() + timedelta(minutes=30)
            logger.info("Successfully logged in with API key")
            return True

        except Exception as e:
            logger.error(f"API key login failed: {e}")
            return False

    def login_with_password(self) -> bool:
        """
        Login using email and password (legacy method)

        Returns:
            True if login successful
        """
        if not self.email or not self.password:
            logger.error("Email/password not provided")
            return False

        try:
            url = f"{self.BASE_URL}/login"
            headers = {
                "accept": "application/json",
                "content-type": "application/json"
            }
            body = {
                "email": self.email,
                "password": self.password
            }

            response = requests.post(url, headers=headers, json=body, timeout=10)
            response.raise_for_status()

            token = response.json().get('token')
            if not token:
                logger.error("No token in password login response")
                return False

            self.bearer_token = f"Bearer {token}"
            self.token_expires_at = datetime.now() + timedelta(minutes=30)
            logger.info("Successfully logged in with email/password")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Password login failed: {e}")
            return False

    def login(self) -> bool:
        """
        Login using available authentication method

        Tries session token first, then API key, then email/password

        Returns:
            True if login successful
        """
        # If session token exists, we're already authenticated
        if self.session_token and self.bearer_token:
            logger.info("Already authenticated with session token")
            return True

        # Try API key authentication first
        if self.api_key and self.private_key:
            logger.info("Attempting API key authentication...")
            if self.login_with_api_key():
                return True
            logger.warning("API key authentication failed, trying password...")

        # Fall back to email/password
        if self.email and self.password:
            logger.info("Attempting email/password authentication...")
            return self.login_with_password()

        logger.error("No valid authentication credentials provided")
        logger.error("Options: 1) Set KALSHI_SESSION_TOKEN (from browser)")
        logger.error("         2) Set KALSHI_API_KEY + private key (requires Premier account)")
        logger.error("         3) Set KALSHI_EMAIL + KALSHI_PASSWORD (may require SMS)")
        return False

    def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid token, refresh if needed"""
        # If using session token from browser, check if it's still valid
        if self.session_token:
            if self._needs_token_refresh():
                logger.warning("Session token expired. Please extract a new token from browser.")
                logger.warning("Run: python extract_kalshi_session.py")
                return False
            return True

        # Otherwise use standard authentication flow
        if self._needs_token_refresh():
            return self.login()
        return True

    def get_all_markets(self, status: str = "open", limit: int = 1000) -> List[Dict]:
        """
        Get all markets from Kalshi

        Args:
            status: Market status filter ('open', 'closed', 'settled')
            limit: Results per page (max 1000)

        Returns:
            List of market dictionaries
        """
        # Use API key authentication if available
        if not self.api_key or not self.private_key:
            # Fallback to bearer token authentication
            if not self._ensure_authenticated():
                return []

        all_markets = []
        cursor = None
        page = 0
        max_pages = 20  # Safety limit: 20 pages * 1000 = 20,000 markets max

        try:
            while True:
                page += 1
                logger.info(f"Fetching page {page} (limit={limit})...")

                path = "/markets"
                url = f"{self.BASE_URL}{path}"
                params = {
                    "limit": limit,
                    "status": status
                }

                if cursor:
                    params['cursor'] = cursor

                # Use signed headers for API key auth
                if self.api_key and self.private_key:
                    timestamp = str(int(time.time() * 1000))
                    method = "GET"
                    signature = self._sign_request(timestamp, method, path, "")

                    headers = {
                        "accept": "application/json",
                        "KALSHI-ACCESS-KEY": self.api_key,
                        "KALSHI-ACCESS-SIGNATURE": signature,
                        "KALSHI-ACCESS-TIMESTAMP": timestamp
                    }
                else:
                    headers = {
                        "accept": "application/json",
                        "Authorization": self.bearer_token
                    }

                response = requests.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                markets = data.get('markets', [])
                all_markets.extend(markets)

                logger.info(f"  Page {page}: Got {len(markets)} markets (total so far: {len(all_markets)})")

                # Check for next page
                cursor = data.get('cursor')
                if not cursor:
                    logger.info("No more pages")
                    break

                # Safety limit
                if page >= max_pages:
                    logger.warning(f"Reached max pages limit ({max_pages}), stopping pagination")
                    break

                # Small delay to avoid rate limits
                time.sleep(0.5)

            logger.info(f"Retrieved {len(all_markets)} total markets from Kalshi")
            return all_markets

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def get_markets_by_series(self, series_ticker: str, limit: int = 1000) -> List[Dict]:
        """
        Get markets for a specific series ticker

        Args:
            series_ticker: Series ticker to filter by (e.g., 'KXNFLGAME')
            limit: Maximum number of markets to fetch

        Returns:
            List of market dictionaries
        """
        try:
            path = "/markets"
            url = f"{self.BASE_URL}{path}"
            params = {
                "series_ticker": series_ticker,
                "limit": limit,
                "status": "open"  # Use 'open' status for active markets
            }

            # Use signed headers for API key auth
            if self.api_key and self.private_key:
                timestamp = str(int(time.time() * 1000))
                method = "GET"
                signature = self._sign_request(timestamp, method, path, "")

                headers = {
                    "accept": "application/json",
                    "KALSHI-ACCESS-KEY": self.api_key,
                    "KALSHI-ACCESS-SIGNATURE": signature,
                    "KALSHI-ACCESS-TIMESTAMP": timestamp
                }
            else:
                headers = {
                    "accept": "application/json",
                    "Authorization": self.bearer_token
                }

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            markets = data.get('markets', [])

            logger.info(f"Retrieved {len(markets)} markets from series {series_ticker}")
            return markets

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching markets for series {series_ticker}: {e}")
            return []

    def get_football_markets(self) -> Dict[str, List[Dict]]:
        """
        Get all NFL and college football game markets using official series tickers

        Returns:
            Dictionary with 'nfl' and 'college' keys
        """
        logger.info("Fetching NFL game markets...")
        nfl_markets = self.get_markets_by_series('KXNFLGAME')

        logger.info("Fetching college football game markets...")
        college_markets = self.get_markets_by_series('KXNCAAFGAME')

        logger.info(f"Found {len(nfl_markets)} NFL markets and {len(college_markets)} college football markets")

        return {
            'nfl': nfl_markets,
            'college': college_markets
        }

    def get_market_details(self, market_ticker: str) -> Optional[Dict]:
        """
        Get detailed information for a specific market

        Args:
            market_ticker: Market ticker symbol

        Returns:
            Market details dictionary or None
        """
        if not self._ensure_authenticated():
            return None

        try:
            url = f"{self.BASE_URL}/markets/{market_ticker}"
            headers = {
                "accept": "application/json",
                "Authorization": self.bearer_token
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching market details for {market_ticker}: {e}")
            return None

    def get_market_orderbook(self, market_ticker: str) -> Optional[Dict]:
        """
        Get current orderbook (bids/asks) for a market

        Args:
            market_ticker: Market ticker symbol

        Returns:
            Orderbook data or None
        """
        if not self._ensure_authenticated():
            return None

        try:
            url = f"{self.BASE_URL}/markets/{market_ticker}/orderbook"
            headers = {
                "accept": "application/json",
                "Authorization": self.bearer_token
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching orderbook for {market_ticker}: {e}")
            return None


if __name__ == "__main__":
    # Test the new client
    from dotenv import load_dotenv
    load_dotenv()

    client = KalshiClientV2()

    print("\n" + "="*80)
    print("KALSHI API CLIENT V2 - Football Markets Test")
    print("="*80)

    # Login
    if not client.login():
        print("\n❌ Login failed. Check credentials.")
        exit(1)

    print("\n✅ Login successful!")

    # Get football markets
    print("\n📊 Fetching all football markets...")
    football_markets = client.get_football_markets()

    # Display NFL markets
    print(f"\n{'='*80}")
    print(f"NFL MARKETS ({len(football_markets['nfl'])} found)")
    print(f"{'='*80}")

    for market in football_markets['nfl'][:10]:  # Show first 10
        print(f"\n{market.get('title', 'N/A')}")
        print(f"  Ticker: {market.get('ticker', 'N/A')}")
        print(f"  Close Time: {market.get('close_time', 'N/A')}")
        print(f"  Volume: ${market.get('volume', 0):,.0f}")

    # Display College markets
    print(f"\n{'='*80}")
    print(f"COLLEGE FOOTBALL MARKETS ({len(football_markets['college'])} found)")
    print(f"{'='*80}")

    for market in football_markets['college'][:10]:  # Show first 10
        print(f"\n{market.get('title', 'N/A')}")
        print(f"  Ticker: {market.get('ticker', 'N/A')}")
        print(f"  Close Time: {market.get('close_time', 'N/A')}")
        print(f"  Volume: ${market.get('volume', 0):,.0f}")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
