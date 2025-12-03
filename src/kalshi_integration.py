"""
Kalshi API Integration for Prediction Markets
Fetches event contract data from Kalshi exchange
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

class KalshiIntegration:
    """
    Wrapper for Kalshi API to fetch prediction market data
    Note: Market data endpoints are PUBLIC (no authentication required)
    """

    def __init__(self) -> None:
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def get_markets(self, limit: int = 200, status: str = 'active') -> List[Dict]:
        """
        Fetch all active markets from Kalshi

        Args:
            limit: Maximum markets to return (default 200, max 1000)
            status: 'active', 'closed', or 'settled' (Note: API may not support this filter)

        Returns:
            List of market dictionaries
        """
        try:
            url = f"{self.base_url}/markets"
            params = {
                'limit': min(limit, 1000)
            }
            # Note: The elections API doesn't accept status parameter
            # We'll filter for active markets after fetching

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            markets = data.get('markets', [])

            # Filter by status if requested
            if status:
                markets = [m for m in markets if m.get('status', '').lower() == status.lower()]

            print(f"Fetched {len(markets)} markets from Kalshi")
            return markets

        except requests.exceptions.RequestException as e:
            print(f"Error fetching markets: {e}")
            return []

    def get_market_details(self, ticker: str) -> Optional[Dict]:
        """
        Get detailed information about a specific market

        Args:
            ticker: Market ticker (e.g., 'PRES-2024-DEM')

        Returns:
            Market details dictionary or None
        """
        try:
            url = f"{self.base_url}/markets/{ticker}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('market', {})

        except requests.exceptions.RequestException as e:
            print(f"Error fetching market {ticker}: {e}")
            return None

    def get_orderbook(self, ticker: str, depth: int = 5) -> Optional[Dict]:
        """
        Get current orderbook for a market

        Args:
            ticker: Market ticker
            depth: Order book depth (1-100)

        Returns:
            Orderbook with bids/asks or None
        """
        try:
            url = f"{self.base_url}/markets/{ticker}/orderbook"
            params = {'depth': min(depth, 100)}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('orderbook', {})

        except requests.exceptions.RequestException as e:
            print(f"Error fetching orderbook for {ticker}: {e}")
            return None

    def get_enriched_markets(self, limit: int = 100) -> List[Dict]:
        """
        Fetch markets with enriched data (prices, volume, orderbook)

        Args:
            limit: Number of markets to enrich

        Returns:
            List of enriched market dictionaries
        """
        markets = self.get_markets(limit=limit)
        enriched = []

        for idx, market in enumerate(markets):
            ticker = market.get('ticker')
            if not ticker:
                continue

            # Rate limiting: 100 req/min = ~1.67/sec, so sleep 0.6s between calls
            if idx > 0 and idx % 10 == 0:
                print(f"Processed {idx}/{len(markets)} markets...")
                time.sleep(0.6)

            # Get orderbook for current prices
            orderbook = self.get_orderbook(ticker)

            # Extract yes/no prices from orderbook
            yes_bid = None
            yes_ask = None
            no_bid = None
            no_ask = None

            if orderbook:
                yes_bids = orderbook.get('yes', [])
                no_bids = orderbook.get('no', [])

                if yes_bids and len(yes_bids) > 0:
                    yes_bid = yes_bids[0][0] / 100  # Convert cents to dollars
                if yes_bids and len(yes_bids) > 1:
                    yes_ask = yes_bids[-1][0] / 100

                if no_bids and len(no_bids) > 0:
                    no_bid = no_bids[0][0] / 100
                if no_bids and len(no_bids) > 1:
                    no_ask = no_bids[-1][0] / 100

            # Calculate prices (yes + no should = 1)
            yes_price = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else None
            no_price = 1 - yes_price if yes_price else None

            # Enrich market data
            enriched_market = {
                'ticker': ticker,
                'title': market.get('title', ''),
                'category': market.get('category', 'Other'),
                'subcategory': market.get('subcategory', ''),
                'yes_price': yes_price,
                'no_price': no_price,
                'yes_bid': yes_bid,
                'yes_ask': yes_ask,
                'no_bid': no_bid,
                'no_ask': no_ask,
                'volume_24h': market.get('volume_24h', 0),
                'open_interest': market.get('open_interest', 0),
                'bid_ask_spread': abs(yes_ask - yes_bid) if yes_ask and yes_bid else None,
                'open_date': market.get('open_time'),
                'close_date': market.get('close_time'),
                'description': market.get('subtitle', ''),
                'market_status': market.get('status', 'active'),
                'last_updated': datetime.now().isoformat()
            }

            # Calculate days to close
            if enriched_market['close_date']:
                try:
                    close_dt = datetime.fromisoformat(enriched_market['close_date'].replace('Z', '+00:00'))
                    days_to_close = (close_dt - datetime.now()).days
                    enriched_market['days_to_close'] = max(0, days_to_close)
                except:
                    enriched_market['days_to_close'] = None

            enriched.append(enriched_market)

        print(f"Enriched {len(enriched)} markets with pricing data")
        return enriched

    def get_markets_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """
        Get markets filtered by category

        Args:
            category: Category name (Politics, Sports, Economics, etc.)
            limit: Maximum markets to return

        Returns:
            Filtered list of markets
        """
        all_markets = self.get_markets(limit=limit * 2)  # Get more to filter
        filtered = [m for m in all_markets if m.get('category', '').lower() == category.lower()]
        return filtered[:limit]
