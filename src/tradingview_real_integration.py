"""Real TradingView Integration with Robinhood for Options"""

import os
import json
from typing import List, Dict, Optional, Any
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from datetime import datetime, timedelta
import robin_stocks.robinhood as rh
from pathlib import Path

class TradingViewRealIntegration:
    """Properly integrate TradingView watchlists with real option data"""

    def __init__(self) -> None:
        self.watchlist_file = Path("tradingview_watchlist.json")
        self.load_saved_watchlist()
        # Don't auto-sync on init - only when button pressed

    def load_saved_watchlist(self) -> Dict[str, List[str]]:
        """Load saved watchlist from file"""
        if self.watchlist_file.exists():
            with open(self.watchlist_file, 'r') as f:
                self.watchlists = json.load(f)
        else:
            # No default watchlists - start with empty
            self.watchlists = {}
        return self.watchlists

    def auto_sync_watchlists(self) -> None:
        """Automatically sync watchlists from TradingView"""
        try:
            from src.tradingview_scraper import TradingViewScraper

            scraper = TradingViewScraper()
            scraped_watchlists = scraper.run()

            if scraped_watchlists:
                # Merge with existing watchlists
                for name, symbols in scraped_watchlists.items():
                    if symbols:  # Only add non-empty watchlists
                        self.watchlists[f"TV: {name}"] = symbols

                # Save updated watchlists
                with open(self.watchlist_file, 'w') as f:
                    json.dump(self.watchlists, f, indent=2)

                print(f"Successfully synced {len(scraped_watchlists)} watchlists from TradingView")
        except Exception as e:
            print(f"Auto-sync failed, using cached data: {e}")

    def save_watchlist(self, name: str, symbols: List[str]):
        """Save a watchlist"""
        self.watchlists[name] = symbols
        with open(self.watchlist_file, 'w') as f:
            json.dump(self.watchlists, f, indent=2)

    def import_from_text(self, text: str, watchlist_name: str = "My Watchlist") -> List[str]:
        """Import symbols from text (comma or line separated)"""
        symbols = []

        # Clean and parse text
        text = text.strip().upper()

        # Handle both comma and newline separated
        text = text.replace('\n', ',').replace(' ', ',')
        parts = text.split(',')

        for part in parts:
            symbol = part.strip()
            # Basic validation - alphanumeric, 1-5 chars
            if symbol and len(symbol) <= 5 and symbol.replace('.', '').replace('-', '').isalnum():
                symbols.append(symbol)

        # Remove duplicates
        symbols = list(set(symbols))

        # Save the watchlist
        if symbols:
            self.save_watchlist(watchlist_name, symbols)

        return symbols

    def get_stock_data_with_changes(self, symbols: List[str]) -> List[Dict]:
        """Get stock data with price changes and percentages"""
        stock_data = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Get price data
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                previous_close = info.get('previousClose', current_price)

                # Calculate change
                price_change = current_price - previous_close
                pct_change = (price_change / previous_close * 100) if previous_close > 0 else 0

                # Get additional info
                volume = info.get('volume', 0)
                market_cap = info.get('marketCap', 0)

                stock_data.append({
                    'symbol': symbol,
                    'price': current_price,
                    'change': price_change,
                    'pct_change': pct_change,
                    'volume': volume,
                    'market_cap': market_cap,
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A')
                })

            except Exception as e:
                stock_data.append({
                    'symbol': symbol,
                    'price': 0,
                    'change': 0,
                    'pct_change': 0,
                    'volume': 0,
                    'market_cap': 0,
                    'sector': 'Error',
                    'industry': str(e)[:50]
                })

        return stock_data

    def get_options_from_robinhood(self, symbol: str) -> Dict[str, Any]:
        """Get option chain from Robinhood"""
        try:
            # Get option chains from Robinhood
            chains = rh.options.find_options_by_expiration(
                symbol,
                expirationDate=None,  # Get all dates
                optionType='put'
            )

            if not chains:
                return {}

            options_data = {}

            for option in chains[:10]:  # Limit to first 10 expirations
                exp_date = option.get('expiration_date', '')
                if not exp_date:
                    continue

                # Parse expiration
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_dt - datetime.now()).days

                if days_to_exp > 45 or days_to_exp < 7:
                    continue

                # Get option details
                strike = float(option.get('strike_price', 0))
                mark_price = float(option.get('mark_price', 0))
                bid = float(option.get('bid_price', 0))
                ask = float(option.get('ask_price', 0))
                volume = int(option.get('volume', 0))
                oi = int(option.get('open_interest', 0))
                iv = float(option.get('implied_volatility', 0))

                # Calculate premium
                premium = mark_price if mark_price > 0 else (bid + ask) / 2

                if premium > 0:
                    options_data[exp_date] = {
                        'strike': strike,
                        'premium': premium,
                        'bid': bid,
                        'ask': ask,
                        'volume': volume,
                        'open_interest': oi,
                        'iv': iv,
                        'days_to_expiry': days_to_exp
                    }

            return options_data

        except Exception as e:
            return {'error': str(e)}

    def get_comprehensive_options_table(self, symbols: List[str]) -> List[Dict]:
        """Get comprehensive options data for all symbols"""
        table_data = []

        # First get stock data
        stock_data = self.get_stock_data_with_changes(symbols)

        for stock in stock_data:
            symbol = stock['symbol']
            row = {
                'Symbol': symbol,
                'Price': stock['price'],
                'Change': stock['change'],
                '% Change': stock['pct_change'],
                'Volume': stock['volume']
            }

            if stock['price'] > 0:
                try:
                    # Try Robinhood first if connected
                    if 'rh_connected' in globals() and rh_connected:
                        options = self.get_options_from_robinhood(symbol)
                    else:
                        # Fallback to yfinance
                        options = self.get_options_from_yfinance(symbol, stock['price'])

                    # Add option data to row
                    for exp_date, opt_data in options.items():
                        if 'error' not in opt_data:
                            days = opt_data.get('days_to_expiry', 0)
                            strike = opt_data.get('strike', 0)
                            premium = opt_data.get('premium', 0)

                            if premium > 0 and strike > 0:
                                capital = strike * 100
                                premium_total = premium * 100
                                return_pct = (premium_total / capital * 100)

                                # Add columns for this expiration
                                row[f'{days}d Strike'] = strike
                                row[f'{days}d Premium'] = premium_total
                                row[f'{days}d Capital'] = capital
                                row[f'{days}d Return%'] = return_pct

                except Exception:
                    pass

            table_data.append(row)

        return table_data

    def get_options_from_yfinance(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Fallback to yfinance for options"""
        options_data = {}

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options[:5]  # Get first 5 expirations

            for exp_date in expirations:
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_dt - datetime.now()).days

                if days_to_exp > 45 or days_to_exp < 7:
                    continue

                # Get option chain
                opt_chain = ticker.option_chain(exp_date)
                puts = opt_chain.puts

                if not puts.empty:
                    # Find 5% OTM put
                    target_strike = current_price * 0.95
                    otm_puts = puts[puts['strike'] <= target_strike]

                    if not otm_puts.empty:
                        # Get closest to target
                        best_put = otm_puts.iloc[-1]  # Highest strike below target

                        strike = best_put['strike']
                        bid = best_put['bid']
                        ask = best_put['ask']
                        premium = (bid + ask) / 2 if bid > 0 and ask > 0 else bid

                        if premium > 0:
                            options_data[exp_date] = {
                                'strike': strike,
                                'premium': premium,
                                'bid': bid,
                                'ask': ask,
                                'volume': best_put.get('volume', 0),
                                'open_interest': best_put.get('openInterest', 0),
                                'iv': best_put.get('impliedVolatility', 0),
                                'days_to_expiry': days_to_exp
                            }

        except Exception as e:
            options_data['error'] = str(e)

        return options_data