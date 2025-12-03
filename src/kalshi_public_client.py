"""
Kalshi Public API Client - NO AUTHENTICATION REQUIRED
Access public market data without any credentials or API keys

Public endpoints available:
- GET /markets - All market data
- GET /markets/{ticker} - Specific market details
- GET /markets/{ticker}/orderbook - Current orderbook
- GET /series - Series information
"""

import requests
import logging
import time
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalshiPublicClient:
    """
    Kalshi Public API Client - Access market data without authentication

    No credentials, API keys, or login required!
    Perfect for reading market data, prices, and orderbooks.
    """

    # Public API base URL
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self) -> None:
        """Initialize public client - no credentials needed!"""
        logger.info("Initialized Kalshi Public Client (no auth required)")

    def get_all_markets(self, status: str = "open", limit: int = 1000,
                        event_ticker: Optional[str] = None,
                        series_ticker: Optional[str] = None) -> List[Dict]:
        """
        Get all markets from Kalshi (PUBLIC - no auth required)

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
                params = {
                    "limit": limit,
                    "status": status
                }

                if event_ticker:
                    params['event_ticker'] = event_ticker
                if series_ticker:
                    params['series_ticker'] = series_ticker
                if cursor:
                    params['cursor'] = cursor

                headers = {
                    "accept": "application/json"
                }

                response = requests.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                markets = data.get('markets', [])
                all_markets.extend(markets)

                # Check for next page
                cursor = data.get('cursor')
                if not cursor:
                    break

                # Small delay to avoid rate limits
                time.sleep(0.3)

            logger.info(f"Retrieved {len(all_markets)} markets from Kalshi (public API)")
            return all_markets

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching public markets: {e}")
            return []

    def get_market(self, market_ticker: str) -> Optional[Dict]:
        """
        Get detailed information for a specific market (PUBLIC)

        Args:
            market_ticker: Market ticker symbol (e.g., 'KXNFL-CHIEFS-WIN')

        Returns:
            Market details dictionary or None
        """
        try:
            url = f"{self.BASE_URL}/markets/{market_ticker}"
            headers = {
                "accept": "application/json"
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            market_data = response.json()

            # Extract market from response if nested
            if 'market' in market_data:
                return market_data['market']
            return market_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching market {market_ticker}: {e}")
            return None

    def get_market_orderbook(self, market_ticker: str) -> Optional[Dict]:
        """
        Get current orderbook (bids/asks) for a market (PUBLIC)

        Args:
            market_ticker: Market ticker symbol

        Returns:
            Orderbook data with yes/no bid/ask prices
        """
        try:
            url = f"{self.BASE_URL}/markets/{market_ticker}/orderbook"
            headers = {
                "accept": "application/json"
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching orderbook for {market_ticker}: {e}")
            return None

    def get_series(self, series_ticker: Optional[str] = None) -> List[Dict]:
        """
        Get series information (PUBLIC)

        Args:
            series_ticker: Optional specific series to fetch

        Returns:
            List of series dictionaries
        """
        try:
            url = f"{self.BASE_URL}/series"
            params = {}
            if series_ticker:
                params['series_ticker'] = series_ticker

            headers = {
                "accept": "application/json"
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

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
