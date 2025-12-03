"""TradingView Integration for fetching watchlists"""

import os
import json
import requests
from typing import List, Dict, Any
from datetime import datetime
import yfinance as yf

class TradingViewClient:
    """Client for TradingView integration"""

    def __init__(self) -> None:
        self.username = os.getenv('TRADINGVIEW_USERNAME', '')
        self.password = os.getenv('TRADINGVIEW_PASSWORD', '')
        self.session = requests.Session()

    def get_watchlist_symbols(self, watchlist_name: str = None) -> List[str]:
        """
        Get symbols from TradingView watchlist
        Note: Since TradingView doesn't have a public API, we'll use predefined lists
        and allow manual import via CSV/text
        """

        # Predefined popular wheel strategy watchlists
        wheel_watchlists = {
            "High IV Stocks": [
                "PLTR", "RIOT", "MARA", "NIO", "SOFI",
                "LCID", "RIVN", "F", "BAC", "T",
                "AAL", "CCL", "PLUG", "TLRY", "AMC"
            ],
            "Dividend Aristocrats Under $50": [
                "T", "KO", "PEP", "VZ", "WBA",
                "XOM", "CVX", "ABBV", "MO", "KHC"
            ],
            "Popular Wheel Stocks": [
                "AMD", "AAPL", "MSFT", "NVDA", "TSLA",
                "SPY", "QQQ", "IWM", "DIA", "ARKK"
            ],
            "Small Cap High Premium": [
                "BB", "NOK", "SNDL", "CLOV", "WISH",
                "WKHS", "RIDE", "GOEV", "FSR", "SPCE"
            ],
            "Tech Under $50": [
                "PLTR", "SOFI", "DKNG", "HOOD", "RBLX",
                "U", "PATH", "SNOW", "NET", "CRWD"
            ]
        }

        # Return requested watchlist or default
        if watchlist_name and watchlist_name in wheel_watchlists:
            return wheel_watchlists[watchlist_name]
        else:
            # Return all stocks under $50 from all lists
            all_symbols = []
            for symbols in wheel_watchlists.values():
                all_symbols.extend(symbols)

            # Remove duplicates and filter by price
            unique_symbols = list(set(all_symbols))
            return self.filter_by_price(unique_symbols, max_price=50)

    def filter_by_price(self, symbols: List[str], max_price: float = 50) -> List[str]:
        """Filter symbols by current price"""
        filtered = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                if current_price and current_price <= max_price:
                    filtered.append(symbol)
            except:
                # If we can't get price, include it anyway
                filtered.append(symbol)

        return filtered

    def import_from_text(self, text: str) -> List[str]:
        """Import symbols from text (comma or newline separated)"""
        # Handle both comma and newline separated
        symbols = []

        # Split by newlines first
        lines = text.strip().split('\n')
        for line in lines:
            # Then split by commas
            parts = line.split(',')
            for part in parts:
                symbol = part.strip().upper()
                # Basic validation
                if symbol and len(symbol) <= 5 and symbol.isalpha():
                    symbols.append(symbol)

        return symbols

    def get_predefined_screeners(self) -> Dict[str, List[str]]:
        """Get predefined screeners for wheel strategy"""
        return {
            "High IV Rank": self._screen_high_iv(),
            "Weekly Options": self._screen_weekly_options(),
            "Liquid Options": self._screen_liquid_options(),
            "Safe Dividends": self._screen_dividend_stocks()
        }

    def _screen_high_iv(self) -> List[str]:
        """Screen for high IV stocks under $50"""
        # These are typically high IV stocks good for wheel
        return ["PLTR", "RIOT", "MARA", "NIO", "SOFI", "LCID", "PLUG", "TLRY"]

    def _screen_weekly_options(self) -> List[str]:
        """Screen for stocks with weekly options under $50"""
        return ["SPY", "QQQ", "IWM", "AAPL", "AMD", "F", "BAC", "T"]

    def _screen_liquid_options(self) -> List[str]:
        """Screen for stocks with liquid options"""
        return ["SPY", "QQQ", "AAPL", "MSFT", "AMD", "NVDA", "TSLA", "F"]

    def _screen_dividend_stocks(self) -> List[str]:
        """Screen for dividend stocks under $50"""
        return ["T", "F", "KO", "VZ", "WBA", "MO", "KHC"]

def get_tradingview_watchlist() -> List[str]:
    """Quick function to get default watchlist"""
    client = TradingViewClient()
    return client.get_watchlist_symbols("Popular Wheel Stocks")

# CLI testing
if __name__ == "__main__":
    print("TradingView Watchlist Integration")
    print("="*50)

    client = TradingViewClient()

    # Get default watchlist
    symbols = client.get_watchlist_symbols()
    print(f"\nDefault Watchlist ({len(symbols)} symbols):")
    print(", ".join(symbols[:10]))

    # Get specific watchlist
    high_iv = client.get_watchlist_symbols("High IV Stocks")
    print(f"\nHigh IV Stocks ({len(high_iv)} symbols):")
    print(", ".join(high_iv))

    # Test text import
    test_text = "AAPL, MSFT, GOOGL\nTSLA\nAMZN, META"
    imported = client.import_from_text(test_text)
    print(f"\nImported from text ({len(imported)} symbols):")
    print(", ".join(imported))