"""Background Watchlist Sync Service
Continuously syncs watchlist data to database without blocking UI
Uses multiple data sources: Polygon, Alpaca, Robinhood for prices/premiums
"""

import psycopg2
import os
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Dict, Optional
import requests
from options_data_fetcher import OptionsDataFetcher
from enhanced_options_fetcher import EnhancedOptionsFetcher
import robin_stocks.robinhood as rh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class WatchlistSyncService:
    """Background service to sync watchlist data to database"""

    def __init__(self) -> None:
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', 5432),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres123!')
        )
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.alpaca_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        self.options_fetcher = OptionsDataFetcher()
        self.enhanced_fetcher = EnhancedOptionsFetcher()
        self.robinhood_logged_in = False

    def get_watchlist_symbols(self, watchlist_name: str = None) -> List[str]:
        """Get symbols from TradingView watchlists"""
        cur = self.conn.cursor()

        if watchlist_name:
            cur.execute("""
                SELECT DISTINCT s.symbol
                FROM tv_symbols_api s
                JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
                WHERE w.name = %s
                ORDER BY s.symbol
            """, (watchlist_name,))
        else:
            cur.execute("SELECT DISTINCT symbol FROM tv_symbols_api ORDER BY symbol")

        symbols = [row[0] for row in cur.fetchall()]
        cur.close()

        # Filter out crypto symbols - only keep stock symbols
        stock_symbols = [s for s in symbols if not any(x in s for x in ['USDT', 'USD', 'BTC', '.D', 'WETH'])]

        return stock_symbols

    def fetch_price_polygon(self, symbol: str) -> Optional[Dict]:
        """Fetch price from Polygon.io"""
        if not self.polygon_key:
            return None

        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            params = {'apiKey': self.polygon_key}
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    return {
                        'symbol': symbol,
                        'price': result['c'],
                        'change': result['c'] - result['o'],
                        'change_pct': ((result['c'] - result['o']) / result['o'] * 100) if result['o'] > 0 else 0,
                        'volume': result['v'],
                        'source': 'polygon'
                    }
        except Exception as e:
            logger.debug(f"Polygon error for {symbol}: {e}")

        return None

    def fetch_price_alpaca(self, symbol: str) -> Optional[Dict]:
        """Fetch price from Alpaca"""
        if not self.alpaca_key:
            return None

        try:
            url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars/latest"
            headers = {
                'APCA-API-KEY-ID': self.alpaca_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret
            }
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get('bar'):
                    bar = data['bar']
                    return {
                        'symbol': symbol,
                        'price': bar['c'],
                        'change': bar['c'] - bar['o'],
                        'change_pct': ((bar['c'] - bar['o']) / bar['o'] * 100) if bar['o'] > 0 else 0,
                        'volume': bar['v'],
                        'source': 'alpaca'
                    }
        except Exception as e:
            logger.debug(f"Alpaca error for {symbol}: {e}")

        return None

    def fetch_price_yfinance(self, symbol: str) -> Optional[Dict]:
        """Fetch price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close * 100) if prev_close > 0 else 0

                return {
                    'symbol': symbol,
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': int(hist['Volume'].iloc[-1]),
                    'source': 'yfinance'
                }
        except Exception as e:
            logger.debug(f"YFinance error for {symbol}: {e}")

        return None

    def fetch_price_with_fallback(self, symbol: str) -> Optional[Dict]:
        """Try multiple sources in order of preference"""
        # Try Polygon first (fastest, most reliable)
        data = self.fetch_price_polygon(symbol)
        if data:
            return data

        # Try Alpaca
        data = self.fetch_price_alpaca(symbol)
        if data:
            return data

        # Fallback to Yahoo Finance
        data = self.fetch_price_yfinance(symbol)
        if data:
            return data

        return None

    def login_robinhood_once(self) -> None:
        """Login to Robinhood once for the session"""
        if self.robinhood_logged_in:
            return True

        try:
            username = os.getenv('ROBINHOOD_USERNAME')
            password = os.getenv('ROBINHOOD_PASSWORD')

            if not username or not password:
                logger.warning("No Robinhood credentials found")
                return False

            rh.authentication.login(
                username=username,
                password=password,
                expiresIn=86400,
                store_session=True
            )
            self.robinhood_logged_in = True
            logger.info("✓ Logged into Robinhood for options data")
            return True

        except Exception as e:
            logger.warning(f"Robinhood login failed: {e}")
            return False

    def fetch_options_yfinance(self, symbol: str, target_dte: int = 45) -> Optional[Dict]:
        """Fetch 45-day options data"""
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]

            expirations = ticker.options
            if not expirations:
                return None

            # Find expiration closest to target DTE
            target_date = datetime.now() + timedelta(days=target_dte)
            closest_exp = min(expirations, key=lambda x: abs(
                (datetime.strptime(x, '%Y-%m-%d') - target_date).days
            ))

            exp_date = datetime.strptime(closest_exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days

            options = ticker.option_chain(closest_exp)
            puts = options.puts

            if puts.empty:
                return None

            # Find 5% OTM strike
            target_strike = current_price * 0.95
            closest_put = puts.iloc[(puts['strike'] - target_strike).abs().argsort()[0]]

            bid = closest_put['bid']
            ask = closest_put['ask']
            mid = (bid + ask) / 2
            premium = mid * 100
            capital = closest_put['strike'] * 100
            premium_pct = (premium / capital * 100) if capital > 0 else 0
            monthly_return = (premium_pct / dte * 30) if dte > 0 else 0
            annual_return = (premium_pct / dte * 365) if dte > 0 else 0

            return {
                'symbol': symbol,
                'expiration_date': closest_exp,
                'dte': dte,
                'strike_price': closest_put['strike'],
                'bid': bid,
                'ask': ask,
                'premium': premium,
                'premium_pct': premium_pct,
                'monthly_return': monthly_return,
                'annual_return': annual_return,
                'iv': closest_put.get('impliedVolatility', 0) * 100 if closest_put.get('impliedVolatility') else 0,
                'volume': int(closest_put['volume']) if closest_put['volume'] else 0,
                'open_interest': int(closest_put['openInterest']) if closest_put['openInterest'] else 0
            }
        except Exception as e:
            logger.debug(f"Options error for {symbol}: {e}")
            return None

    def upsert_price_data(self, data: Dict):
        """Update price data in database"""
        cur = self.conn.cursor()

        try:
            cur.execute("""
                INSERT INTO stock_data (
                    symbol, current_price, price_change, price_change_pct,
                    volume, last_updated
                ) VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (symbol) DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    price_change = EXCLUDED.price_change,
                    price_change_pct = EXCLUDED.price_change_pct,
                    volume = EXCLUDED.volume,
                    last_updated = NOW()
            """, (
                data['symbol'],
                data['price'],
                data['change'],
                data['change_pct'],
                data.get('volume', 0)
            ))

            self.conn.commit()
            logger.info(f"✓ {data['symbol']}: ${data['price']:.2f} ({data['change_pct']:+.2f}%) [{data['source']}]")
        except Exception as e:
            logger.error(f"DB error for {data['symbol']}: {e}")
            self.conn.rollback()
        finally:
            cur.close()

    def upsert_options_data(self, data: Dict):
        """Update options data in database"""
        cur = self.conn.cursor()

        try:
            cur.execute("""
                INSERT INTO stock_premiums (
                    symbol, expiration_date, dte, strike_type, strike_price,
                    bid, ask, mid, premium, premium_pct, monthly_return, annual_return,
                    implied_volatility, volume, open_interest, delta, prob_profit, last_updated
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (symbol, expiration_date, strike_price) DO UPDATE SET
                    dte = EXCLUDED.dte,
                    strike_type = EXCLUDED.strike_type,
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    mid = EXCLUDED.mid,
                    premium = EXCLUDED.premium,
                    premium_pct = EXCLUDED.premium_pct,
                    monthly_return = EXCLUDED.monthly_return,
                    annual_return = EXCLUDED.annual_return,
                    implied_volatility = EXCLUDED.implied_volatility,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest,
                    delta = EXCLUDED.delta,
                    prob_profit = EXCLUDED.prob_profit,
                    last_updated = NOW()
            """, (
                data['symbol'],
                data['expiration_date'],
                data.get('actual_dte', data.get('dte', 0)),
                '30_delta',
                data['strike_price'],
                data['bid'],
                data['ask'],
                data.get('mid', (data['bid'] + data['ask']) / 2),
                data['premium'],
                data['premium_pct'],
                data['monthly_return'],
                data.get('annual_return', data['monthly_return'] * 12),
                data.get('iv', 0),
                data.get('volume', 0),
                data.get('open_interest', 0),
                data.get('delta', None),
                data.get('prob_profit', None)
            ))

            self.conn.commit()
            delta_str = f", Δ={data['delta']:.3f}" if data.get('delta') is not None else ""
            logger.info(f"  → Options: ${data['premium']:.0f} premium ({data['monthly_return']:.2f}%/mo){delta_str}")
        except Exception as e:
            logger.error(f"Options DB error for {data['symbol']}: {e}")
            self.conn.rollback()
        finally:
            cur.close()

    def sync_watchlist(self, watchlist_name: str, delay: float = 0.5):
        """Sync a specific watchlist"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Syncing watchlist: {watchlist_name}")
        logger.info(f"{'='*60}\n")

        # Login to Robinhood once for options data
        self.login_robinhood_once()

        symbols = self.get_watchlist_symbols(watchlist_name)
        logger.info(f"Found {len(symbols)} stock symbols in {watchlist_name}")

        success_count = 0
        options_count = 0
        total_expirations = 0

        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"[{idx}/{len(symbols)}] {symbol}")

            # Fetch and update price
            price_data = self.fetch_price_with_fallback(symbol)
            if price_data:
                self.upsert_price_data(price_data)
                success_count += 1

                # Fetch multiple expirations with delta using enhanced fetcher
                all_options = self.enhanced_fetcher.get_all_expirations_data(symbol)
                if all_options:
                    logger.info(f"  → Found {len(all_options)} expiration dates")
                    for opt_data in all_options:
                        # Add symbol to opt_data
                        opt_data['symbol'] = symbol
                        self.upsert_options_data(opt_data)
                        total_expirations += 1
                    options_count += 1

            # Rate limiting
            time.sleep(delay)

        logger.info(f"\n{'='*60}")
        logger.info(f"Sync Complete: {success_count}/{len(symbols)} prices, {options_count} symbols with {total_expirations} total expirations")
        logger.info(f"{'='*60}\n")

    def close(self) -> None:
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    import sys

    service = WatchlistSyncService()

    # Get watchlist name from command line or default to NVDA
    watchlist_name = sys.argv[1] if len(sys.argv) > 1 else "NVDA"

    try:
        service.sync_watchlist(watchlist_name, delay=0.3)
    finally:
        service.close()
