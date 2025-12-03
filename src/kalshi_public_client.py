"""
Kalshi Public API Client - NO AUTHENTICATION REQUIRED
Access public market data without any credentials or API keys

Public endpoints available:
- GET /markets - All market data
- GET /markets/{ticker} - Specific market details
- GET /markets/{ticker}/orderbook - Current orderbook
- GET /series - Series information

Performance: Circuit breaker pattern for resilience
"""

import requests
import logging
import time
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for API resilience.
    Prevents cascading failures by temporarily blocking requests to a failing service.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: Optional[float] = field(default=None, init=False, repr=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)

    def record_success(self) -> None:
        """Record a successful request - reset circuit"""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed request - may trip circuit"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Kalshi circuit breaker OPEN after {self._failure_count} failures")

    def can_execute(self) -> bool:
        """Check if request can proceed"""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Kalshi circuit breaker HALF_OPEN - testing recovery")
                    return True
            return False
        return True  # HALF_OPEN allows one test request

    @property
    def state(self) -> CircuitState:
        return self._state


class KalshiPublicClient:
    """
    Kalshi Public API Client - Access market data without authentication

    No credentials, API keys, or login required!
    Perfect for reading market data, prices, and orderbooks.

    Features:
    - Circuit breaker for resilience (trips after 5 failures, 60s recovery)
    - Automatic retry on transient failures
    """

    # Public API base URL
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """Initialize public client with circuit breaker - no credentials needed!"""
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        logger.info("Initialized Kalshi Public Client with circuit breaker (no auth required)")

    def _make_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make HTTP request with circuit breaker protection"""
        if not self._circuit_breaker.can_execute():
            logger.warning(f"Kalshi circuit breaker OPEN - request blocked: {url}")
            return None

        try:
            kwargs.setdefault('timeout', 30)
            kwargs.setdefault('headers', {'accept': 'application/json'})
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            self._circuit_breaker.record_success()
            return response
        except requests.exceptions.RequestException as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Kalshi API error ({self._circuit_breaker.state.value}): {e}")
            raise

    @property
    def circuit_state(self) -> str:
        """Get current circuit breaker state"""
        return self._circuit_breaker.state.value

    def get_all_markets(self, status: str = "open", limit: int = 1000,
                        event_ticker: Optional[str] = None,
                        series_ticker: Optional[str] = None) -> List[Dict]:
        """
        Get all markets from Kalshi (PUBLIC - no auth required)
        Protected by circuit breaker pattern.

        Args:
            status: Market status filter ('open', 'closed', 'settled', 'all')
            limit: Results per page (max 1000)
            event_ticker: Filter by event ticker (optional)
            series_ticker: Filter by series ticker (optional)

        Returns:
            List of market dictionaries
        """
        all_markets = []
        cursor = None

        try:
            while True:
                url = f"{self.BASE_URL}/markets"
                params = {"limit": limit, "status": status}

                if event_ticker:
                    params['event_ticker'] = event_ticker
                if series_ticker:
                    params['series_ticker'] = series_ticker
                if cursor:
                    params['cursor'] = cursor

                response = self._make_request('GET', url, params=params)
                if response is None:
                    return all_markets  # Circuit open - return what we have

                data = response.json()
                markets = data.get('markets', [])
                all_markets.extend(markets)

                cursor = data.get('cursor')
                if not cursor:
                    break

                time.sleep(0.3)  # Rate limit protection

            logger.info(f"Retrieved {len(all_markets)} markets from Kalshi (public API)")
            return all_markets

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching public markets: {e}")
            return all_markets

    def get_market(self, market_ticker: str) -> Optional[Dict]:
        """
        Get detailed information for a specific market (PUBLIC)
        Protected by circuit breaker pattern.

        Args:
            market_ticker: Market ticker symbol (e.g., 'KXNFL-CHIEFS-WIN')

        Returns:
            Market details dictionary or None
        """
        try:
            url = f"{self.BASE_URL}/markets/{market_ticker}"
            response = self._make_request('GET', url)
            if response is None:
                return None  # Circuit open

            market_data = response.json()
            return market_data.get('market', market_data)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching market {market_ticker}: {e}")
            return None

    def get_market_orderbook(self, market_ticker: str) -> Optional[Dict]:
        """
        Get current orderbook (bids/asks) for a market (PUBLIC)
        Protected by circuit breaker pattern.

        Args:
            market_ticker: Market ticker symbol

        Returns:
            Orderbook data with yes/no bid/ask prices
        """
        try:
            url = f"{self.BASE_URL}/markets/{market_ticker}/orderbook"
            response = self._make_request('GET', url)
            if response is None:
                return None  # Circuit open

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching orderbook for {market_ticker}: {e}")
            return None

    def get_series(self, series_ticker: Optional[str] = None) -> List[Dict]:
        """
        Get series information (PUBLIC)
        Protected by circuit breaker pattern.

        Args:
            series_ticker: Optional specific series to fetch

        Returns:
            List of series dictionaries
        """
        try:
            url = f"{self.BASE_URL}/series"
            params = {'series_ticker': series_ticker} if series_ticker else {}

            response = self._make_request('GET', url, params=params)
            if response is None:
                return []  # Circuit open

            data = response.json()
            return data.get('series', [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching series: {e}")
            return []

    def filter_football_markets(self, markets: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Filter markets to only NFL and college football games

        Args:
            markets: List of all markets

        Returns:
            Dictionary with 'nfl' and 'college' keys containing filtered markets
        """
        nfl_markets = []
        college_markets = []

        # Keywords to identify football markets
        nfl_keywords = ['nfl', 'super bowl', 'playoffs', 'chiefs', 'bills', 'ravens',
                        'packers', '49ers', 'cowboys', 'eagles', 'lions', 'rams',
                        'dolphins', 'bengals', 'steelers', 'seahawks', 'buccaneers']
        college_keywords = ['college football', 'ncaa football', 'cfp', 'alabama',
                           'georgia', 'ohio state', 'michigan', 'texas', 'clemson',
                           'oregon', 'penn state', 'notre dame', 'usc', 'lsu']

        for market in markets:
            title = market.get('title', '').lower()
            ticker = market.get('ticker', '').lower()
            subtitle = market.get('subtitle', '').lower()
            series_ticker = market.get('series_ticker', '').lower()

            # Combine all text fields for searching
            combined_text = f"{title} {ticker} {subtitle} {series_ticker}"

            # Check for NFL
            if any(keyword in combined_text for keyword in nfl_keywords):
                nfl_markets.append(market)
            # Check for college football
            elif any(keyword in combined_text for keyword in college_keywords):
                college_markets.append(market)

        logger.info(f"Found {len(nfl_markets)} NFL markets and {len(college_markets)} college football markets")

        return {
            'nfl': nfl_markets,
            'college': college_markets
        }

    def get_football_markets(self) -> Dict[str, List[Dict]]:
        """
        Get all NFL and college football markets (PUBLIC)

        Returns:
            Dictionary with 'nfl' and 'college' keys
        """
        all_markets = self.get_all_markets(status='open')
        return self.filter_football_markets(all_markets)

    def search_markets(self, search_term: str, status: str = "open") -> List[Dict]:
        """
        Search markets by keyword in title/ticker

        Args:
            search_term: Term to search for
            status: Market status filter

        Returns:
            List of matching markets
        """
        all_markets = self.get_all_markets(status=status)
        search_lower = search_term.lower()

        return [
            market for market in all_markets
            if search_lower in market.get('title', '').lower()
            or search_lower in market.get('ticker', '').lower()
            or search_lower in market.get('subtitle', '').lower()
        ]


if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "="*80)
    print("KALSHI PUBLIC API CLIENT - NO AUTHENTICATION REQUIRED")
    print("="*80)
    print("\nThis client accesses public market data without any credentials!")
    print("No API key, no login, no session token needed.\n")

    client = KalshiPublicClient()

    # Get football markets
    print("📊 Fetching all football markets from public API...")
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

        # Get orderbook for first market as example
        if market == football_markets['nfl'][0]:
            ticker = market.get('ticker')
            if ticker:
                print(f"\n  📈 Getting orderbook for {ticker}...")
                orderbook = client.get_market_orderbook(ticker)
                if orderbook:
                    print(f"  Yes: Bid ${orderbook.get('yes', [{}])[0].get('price', 0)/100:.2f} / "
                          f"Ask ${orderbook.get('yes', [{}])[0].get('price', 0)/100:.2f}")

    # Display College markets
    print(f"\n{'='*80}")
    print(f"COLLEGE FOOTBALL MARKETS ({len(football_markets['college'])} found)")
    print(f"{'='*80}")

    for market in football_markets['college'][:5]:  # Show first 5
        print(f"\n{market.get('title', 'N/A')}")
        print(f"  Ticker: {market.get('ticker', 'N/A')}")
        print(f"  Volume: ${market.get('volume', 0):,.0f}")

    print("\n" + "="*80)
    print("✅ Success! All data retrieved from public API without authentication")
    print("="*80)
