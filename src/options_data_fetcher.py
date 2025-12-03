"""Multi-source Options Data Fetcher
Tries multiple APIs to get options data: Polygon, Tradier, Yahoo Finance
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionsDataFetcher:
    """Fetches options data from multiple sources"""

    def __init__(self) -> None:
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.tradier_key = os.getenv('TRADIER_API_KEY')  # If you have one

    def fetch_options_polygon(self, symbol: str, target_dte: int = 45) -> Optional[Dict]:
        """Fetch options from Polygon.io"""
        if not self.polygon_key:
            return None

        try:
            # Get current price first
            price_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            price_resp = requests.get(price_url, params={'apiKey': self.polygon_key}, timeout=5)

            if price_resp.status_code != 200:
                logger.debug(f"Polygon price failed for {symbol}: {price_resp.status_code}")
                return None

            price_data = price_resp.json()
            if not price_data.get('results'):
                return None

            current_price = price_data['results'][0]['c']

            # Calculate target expiration date
            target_date = datetime.now() + timedelta(days=target_dte)
            exp_date_str = target_date.strftime('%Y-%m-%d')

            # Get options chain snapshot
            # Note: Options data might require paid plan - let's try
            options_url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
            options_resp = requests.get(
                options_url,
                params={
                    'apiKey': self.polygon_key,
                    'expiration_date.gte': exp_date_str,
                    'expiration_date.lte': (target_date + timedelta(days=14)).strftime('%Y-%m-%d'),
                    'contract_type': 'put',
                    'limit': 100
                },
                timeout=10
            )

            if options_resp.status_code == 200:
                data = options_resp.json()
                if data.get('results'):
                    # Find closest strike to 5% OTM
                    target_strike = current_price * 0.95
                    closest_option = min(
                        data['results'],
                        key=lambda x: abs(x.get('details', {}).get('strike_price', 0) - target_strike)
                    )

                    details = closest_option.get('details', {})
                    greeks = closest_option.get('greeks', {})
                    last_quote = closest_option.get('last_quote', {})

                    strike_price = details.get('strike_price', 0)
                    bid = last_quote.get('bid', 0)
                    ask = last_quote.get('ask', 0)
                    mid = (bid + ask) / 2
                    premium = mid * 100

                    exp_date = datetime.strptime(details.get('expiration_date', exp_date_str), '%Y-%m-%d')
                    dte = (exp_date - datetime.now()).days

                    capital = strike_price * 100
                    premium_pct = (premium / capital * 100) if capital > 0 else 0
                    monthly_return = (premium_pct / dte * 30) if dte > 0 else 0

                    return {
                        'symbol': symbol,
                        'expiration_date': details.get('expiration_date'),
                        'dte': dte,
                        'strike_price': strike_price,
                        'bid': bid,
                        'ask': ask,
                        'premium': premium,
                        'premium_pct': premium_pct,
                        'monthly_return': monthly_return,
                        'annual_return': monthly_return * 12,
                        'iv': greeks.get('implied_volatility', 0) * 100 if greeks.get('implied_volatility') else 0,
                        'volume': closest_option.get('day', {}).get('volume', 0),
                        'open_interest': closest_option.get('open_interest', 0),
                        'source': 'polygon'
                    }
            else:
                logger.debug(f"Polygon options failed for {symbol}: {options_resp.status_code} - {options_resp.text[:100]}")

        except Exception as e:
            logger.debug(f"Polygon options error for {symbol}: {e}")

        return None

    def fetch_options_tradier(self, symbol: str, target_dte: int = 45) -> Optional[Dict]:
        """Fetch options from Tradier (sandbox - 15min delayed)"""
        if not self.tradier_key:
            return None

        try:
            # Get current price
            quote_url = "https://sandbox.tradier.com/v1/markets/quotes"
            headers = {
                'Authorization': f'Bearer {self.tradier_key}',
                'Accept': 'application/json'
            }

            quote_resp = requests.get(
                quote_url,
                params={'symbols': symbol},
                headers=headers,
                timeout=5
            )

            if quote_resp.status_code != 200:
                return None

            quote_data = quote_resp.json()
            if not quote_data.get('quotes', {}).get('quote'):
                return None

            current_price = quote_data['quotes']['quote'].get('last', 0)
            if not current_price:
                return None

            # Get expirations
            exp_url = f"https://sandbox.tradier.com/v1/markets/options/expirations"
            exp_resp = requests.get(
                exp_url,
                params={'symbol': symbol},
                headers=headers,
                timeout=5
            )

            if exp_resp.status_code != 200:
                return None

            expirations = exp_resp.json().get('expirations', {}).get('date', [])
            if not expirations:
                return None

            # Find expiration closest to target DTE
            target_date = datetime.now() + timedelta(days=target_dte)
            closest_exp = min(
                expirations,
                key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days)
            )

            # Get options chain for that expiration
            chain_url = "https://sandbox.tradier.com/v1/markets/options/chains"
            chain_resp = requests.get(
                chain_url,
                params={
                    'symbol': symbol,
                    'expiration': closest_exp,
                    'greeks': 'true'
                },
                headers=headers,
                timeout=10
            )

            if chain_resp.status_code != 200:
                return None

            chain_data = chain_resp.json()
            options = chain_data.get('options', {}).get('option', [])

            if not options:
                return None

            # Filter for puts only
            puts = [opt for opt in options if opt.get('option_type') == 'put']

            if not puts:
                return None

            # Find closest strike to 5% OTM
            target_strike = current_price * 0.95
            closest_put = min(puts, key=lambda x: abs(x.get('strike', 0) - target_strike))

            strike_price = closest_put.get('strike', 0)
            bid = closest_put.get('bid', 0)
            ask = closest_put.get('ask', 0)
            mid = (bid + ask) / 2
            premium = mid * 100

            exp_date = datetime.strptime(closest_exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days

            capital = strike_price * 100
            premium_pct = (premium / capital * 100) if capital > 0 else 0
            monthly_return = (premium_pct / dte * 30) if dte > 0 else 0

            greeks = closest_put.get('greeks', {})

            return {
                'symbol': symbol,
                'expiration_date': closest_exp,
                'dte': dte,
                'strike_price': strike_price,
                'bid': bid,
                'ask': ask,
                'premium': premium,
                'premium_pct': premium_pct,
                'monthly_return': monthly_return,
                'annual_return': monthly_return * 12,
                'iv': greeks.get('mid_iv', 0) * 100 if greeks.get('mid_iv') else 0,
                'volume': closest_put.get('volume', 0),
                'open_interest': closest_put.get('open_interest', 0),
                'source': 'tradier'
            }

        except Exception as e:
            logger.debug(f"Tradier options error for {symbol}: {e}")

        return None

    def fetch_options_robinhood(self, symbol: str, target_dte: int = 45) -> Optional[Dict]:
        """Fetch options from Robinhood (if logged in)"""
        try:
            import robin_stocks.robinhood as rh

            # Check if already logged in
            if not rh.account.load_account_profile():
                logger.debug("Not logged into Robinhood")
                return None

            # Get current price
            quote = rh.stocks.get_stock_quote_by_symbol(symbol)
            if not quote:
                return None

            current_price = float(quote.get('last_trade_price', 0))
            if not current_price:
                return None

            # Get option chains
            chains = rh.options.get_chains(symbol)
            if not chains:
                return None

            # chains is a dict, not a list
            exp_dates = chains.get('expiration_dates', [])
            if not exp_dates:
                return None

            # Find expiration closest to target DTE
            target_date = datetime.now() + timedelta(days=target_dte)
            closest_exp = min(
                exp_dates,
                key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days)
            )

            # Get put options for that expiration
            target_strike = current_price * 0.95
            puts = rh.options.find_options_by_expiration(
                symbol,
                closest_exp,
                optionType='put'
            )

            if not puts:
                return None

            # Find closest strike
            closest_put = min(puts, key=lambda x: abs(float(x.get('strike_price', 0)) - target_strike))

            # Get market data for this option
            option_data = rh.options.get_option_market_data_by_id(closest_put['id'])

            if not option_data:
                return None

            strike_price = float(closest_put.get('strike_price', 0))
            bid = float(option_data[0].get('bid_price', 0))
            ask = float(option_data[0].get('ask_price', 0))
            mid = (bid + ask) / 2
            premium = mid * 100

            exp_date = datetime.strptime(closest_exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days

            capital = strike_price * 100
            premium_pct = (premium / capital * 100) if capital > 0 else 0
            monthly_return = (premium_pct / dte * 30) if dte > 0 else 0

            return {
                'symbol': symbol,
                'expiration_date': closest_exp,
                'dte': dte,
                'strike_price': strike_price,
                'bid': bid,
                'ask': ask,
                'premium': premium,
                'premium_pct': premium_pct,
                'monthly_return': monthly_return,
                'annual_return': monthly_return * 12,
                'iv': float(option_data[0].get('implied_volatility', 0)) * 100,
                'volume': int(option_data[0].get('volume', 0)),
                'open_interest': int(option_data[0].get('open_interest', 0)),
                'source': 'robinhood'
            }

        except Exception as e:
            logger.debug(f"Robinhood options error for {symbol}: {e}")

        return None

    def get_best_options_data(self, symbol: str, target_dte: int = 45) -> Optional[Dict]:
        """Try multiple sources and return the first successful one"""

        # Try Robinhood first (most reliable if logged in)
        logger.debug(f"Trying Robinhood for {symbol} options...")
        data = self.fetch_options_robinhood(symbol, target_dte)
        if data:
            logger.info(f"  → Got options from Robinhood")
            return data

        # Try Polygon
        logger.debug(f"Trying Polygon for {symbol} options...")
        data = self.fetch_options_polygon(symbol, target_dte)
        if data:
            logger.info(f"  → Got options from Polygon")
            return data

        # Try Tradier
        logger.debug(f"Trying Tradier for {symbol} options...")
        data = self.fetch_options_tradier(symbol, target_dte)
        if data:
            logger.info(f"  → Got options from Tradier")
            return data

        logger.debug(f"No options data available for {symbol}")
        return None


if __name__ == "__main__":
    # Test the fetcher
    fetcher = OptionsDataFetcher()

    test_symbols = ['AAPL', 'NVDA', 'AMD']

    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print(f"{'='*60}")

        options_data = fetcher.get_best_options_data(symbol)

        if options_data:
            print(f"[OK] Source: {options_data['source']}")
            print(f"  Expiration: {options_data['expiration_date']} ({options_data['dte']} DTE)")
            print(f"  Strike: ${options_data['strike_price']:.2f}")
            print(f"  Premium: ${options_data['premium']:.2f}")
            print(f"  Monthly Return: {options_data['monthly_return']:.2f}%")
        else:
            print(f"[FAIL] No options data available")
