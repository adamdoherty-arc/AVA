"""
Populate ALL US Stocks - Comprehensive Universe
Fetches all stocks from NASDAQ, NYSE, and AMEX exchanges
"""

import os
import psycopg2
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Set
import logging
from dotenv import load_dotenv
import time
import requests
import json

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_value(val):
    """Convert numpy types to Python native types"""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    return val


class AllUSStocksPopulator:
    """Populate ALL US stocks universe"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'database': os.getenv('DB_NAME', 'magnus')
        }
        self.conn = psycopg2.connect(**self.db_config)
        self.processed_stocks: Set[str] = set()
        self.failed_symbols: Set[str] = set()

    def get_existing_symbols(self) -> None:
        """Get already processed symbols from database"""
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT symbol FROM stocks_universe")
            self.processed_stocks = {row[0] for row in cur.fetchall()}
            logger.info(f"Already have {len(self.processed_stocks)} stocks in DB")
        except Exception:
            pass
        finally:
            cur.close()

    def get_all_nasdaq_stocks(self) -> List[Dict]:
        """Get ALL stocks from NASDAQ screener API"""
        logger.info("Fetching all stocks from NASDAQ API...")

        url = "https://api.nasdaq.com/api/screener/stocks"
        params = {
            "tableonly": "true",
            "limit": 10000,
            "offset": 0
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }

        all_stocks = []

        try:
            response = requests.get(url, params=params, headers=headers)
            data = response.json()

            if data.get("data") and data["data"].get("rows"):
                rows = data["data"]["rows"]
                all_stocks = rows
                logger.info(f"Found {len(all_stocks)} stocks from NASDAQ API")
        except Exception as e:
            logger.error(f"Error fetching NASDAQ stocks: {e}")

        return all_stocks

    def get_symbols_from_nasdaq_csv(self) -> List[str]:
        """Alternative: Get symbols from NASDAQ FTP - more reliable"""
        symbols = []

        # NASDAQ listed
        try:
            url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
            response = requests.get(url)
            lines = response.text.strip().split('\n')
            for line in lines[1:-1]:  # Skip header and footer
                parts = line.split('|')
                if len(parts) > 0 and parts[0]:
                    symbol = parts[0].strip()
                    if symbol and not symbol.endswith('$') and '.' not in symbol:
                        symbols.append(symbol)
            logger.info(f"Found {len(symbols)} NASDAQ symbols")
        except Exception as e:
            logger.error(f"Error fetching NASDAQ list: {e}")

        # NYSE/AMEX listed (otherlisted.txt)
        try:
            url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
            response = requests.get(url)
            lines = response.text.strip().split('\n')
            nyse_count = 0
            for line in lines[1:-1]:  # Skip header and footer
                parts = line.split('|')
                if len(parts) > 0 and parts[0]:
                    symbol = parts[0].strip()
                    # Skip preferred shares and warrants
                    if symbol and not any(c in symbol for c in ['$', '-', '.']):
                        if symbol not in symbols:
                            symbols.append(symbol)
                            nyse_count += 1
            logger.info(f"Found {nyse_count} NYSE/AMEX symbols")
        except Exception as e:
            logger.error(f"Error fetching NYSE/AMEX list: {e}")

        return symbols

    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch stock data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Skip ETFs
            if info.get('quoteType') == 'ETF':
                return None

            # Skip if no meaningful data
            if not info.get('regularMarketPrice') and not info.get('currentPrice'):
                return None

            hist = ticker.history(period="1y")
            sma_50, sma_200, rsi_14 = None, None, None

            if not hist.empty and len(hist) > 0:
                closes = hist['Close']
                if len(closes) >= 50:
                    sma_50 = float(closes.tail(50).mean())
                if len(closes) >= 200:
                    sma_200 = float(closes.tail(200).mean())
                if len(closes) >= 15:
                    delta = closes.diff()
                    gain = (delta.where(delta > 0, 0)).tail(14).mean()
                    loss = (-delta.where(delta < 0, 0)).tail(14).mean()
                    if loss != 0:
                        rs = gain / loss
                        rsi_14 = float(100 - (100 / (1 + rs)))

            ex_div_date = None
            if info.get('exDividendDate'):
                try:
                    ex_div_date = datetime.fromtimestamp(
                        info['exDividendDate']
                    ).date()
                except Exception:
                    pass

            has_options = False
            try:
                has_options = len(ticker.options) > 0
            except Exception:
                pass

            return {
                'symbol': symbol,
                'company_name': info.get('longName') or info.get('shortName'),
                'exchange': info.get('exchange'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'current_price': convert_value(
                    info.get('currentPrice') or info.get('regularMarketPrice')
                ),
                'previous_close': convert_value(info.get('previousClose')),
                'open_price': convert_value(
                    info.get('open') or info.get('regularMarketOpen')
                ),
                'day_high': convert_value(
                    info.get('dayHigh') or info.get('regularMarketDayHigh')
                ),
                'day_low': convert_value(
                    info.get('dayLow') or info.get('regularMarketDayLow')
                ),
                'week_52_high': convert_value(info.get('fiftyTwoWeekHigh')),
                'week_52_low': convert_value(info.get('fiftyTwoWeekLow')),
                'volume': convert_value(
                    info.get('volume') or info.get('regularMarketVolume')
                ),
                'avg_volume_10d': convert_value(info.get('averageVolume10days')),
                'avg_volume_3m': convert_value(info.get('averageVolume')),
                'market_cap': convert_value(info.get('marketCap')),
                'shares_outstanding': convert_value(info.get('sharesOutstanding')),
                'float_shares': convert_value(info.get('floatShares')),
                'pe_ratio': convert_value(info.get('trailingPE')),
                'forward_pe': convert_value(info.get('forwardPE')),
                'peg_ratio': convert_value(info.get('pegRatio')),
                'price_to_book': convert_value(info.get('priceToBook')),
                'price_to_sales': convert_value(
                    info.get('priceToSalesTrailing12Months')
                ),
                'enterprise_value': convert_value(info.get('enterpriseValue')),
                'ev_to_ebitda': convert_value(info.get('enterpriseToEbitda')),
                'ev_to_revenue': convert_value(info.get('enterpriseToRevenue')),
                'profit_margin': convert_value(info.get('profitMargins')),
                'operating_margin': convert_value(info.get('operatingMargins')),
                'gross_margin': convert_value(info.get('grossMargins')),
                'roe': convert_value(info.get('returnOnEquity')),
                'roa': convert_value(info.get('returnOnAssets')),
                'revenue_growth': convert_value(info.get('revenueGrowth')),
                'earnings_growth': convert_value(info.get('earningsGrowth')),
                'dividend_yield': convert_value(info.get('dividendYield')),
                'dividend_rate': convert_value(info.get('dividendRate')),
                'payout_ratio': convert_value(info.get('payoutRatio')),
                'ex_dividend_date': ex_div_date,
                'total_cash': convert_value(info.get('totalCash')),
                'total_debt': convert_value(info.get('totalDebt')),
                'debt_to_equity': convert_value(info.get('debtToEquity')),
                'current_ratio': convert_value(info.get('currentRatio')),
                'free_cash_flow': convert_value(info.get('freeCashflow')),
                'beta': convert_value(info.get('beta')),
                'sma_50': sma_50,
                'sma_200': sma_200,
                'rsi_14': rsi_14,
                'target_high_price': convert_value(info.get('targetHighPrice')),
                'target_low_price': convert_value(info.get('targetLowPrice')),
                'target_mean_price': convert_value(info.get('targetMeanPrice')),
                'recommendation_key': info.get('recommendationKey'),
                'number_of_analysts': convert_value(
                    info.get('numberOfAnalystOpinions')
                ),
                'has_options': has_options,
            }
        except Exception as e:
            logger.debug(f"Error {symbol}: {e}")
            return None

    def insert_stock(self, data: Dict):
        """Insert stock data"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO stocks_universe (
                    symbol, company_name, exchange, sector, industry,
                    current_price, previous_close, open_price, day_high,
                    day_low, week_52_high, week_52_low, volume,
                    avg_volume_10d, avg_volume_3m, market_cap,
                    shares_outstanding, float_shares, pe_ratio, forward_pe,
                    peg_ratio, price_to_book, price_to_sales,
                    enterprise_value, ev_to_ebitda, ev_to_revenue,
                    profit_margin, operating_margin, gross_margin, roe, roa,
                    revenue_growth, earnings_growth,
                    dividend_yield, dividend_rate, payout_ratio,
                    ex_dividend_date, total_cash, total_debt,
                    debt_to_equity, current_ratio, free_cash_flow,
                    beta, sma_50, sma_200, rsi_14,
                    target_high_price, target_low_price, target_mean_price,
                    recommendation_key, number_of_analysts, has_options,
                    last_updated
                ) VALUES (
                    %(symbol)s, %(company_name)s, %(exchange)s, %(sector)s,
                    %(industry)s, %(current_price)s, %(previous_close)s,
                    %(open_price)s, %(day_high)s, %(day_low)s,
                    %(week_52_high)s, %(week_52_low)s, %(volume)s,
                    %(avg_volume_10d)s, %(avg_volume_3m)s, %(market_cap)s,
                    %(shares_outstanding)s, %(float_shares)s, %(pe_ratio)s,
                    %(forward_pe)s, %(peg_ratio)s, %(price_to_book)s,
                    %(price_to_sales)s, %(enterprise_value)s,
                    %(ev_to_ebitda)s, %(ev_to_revenue)s, %(profit_margin)s,
                    %(operating_margin)s, %(gross_margin)s, %(roe)s, %(roa)s,
                    %(revenue_growth)s, %(earnings_growth)s,
                    %(dividend_yield)s, %(dividend_rate)s, %(payout_ratio)s,
                    %(ex_dividend_date)s, %(total_cash)s, %(total_debt)s,
                    %(debt_to_equity)s, %(current_ratio)s, %(free_cash_flow)s,
                    %(beta)s, %(sma_50)s, %(sma_200)s, %(rsi_14)s,
                    %(target_high_price)s, %(target_low_price)s,
                    %(target_mean_price)s, %(recommendation_key)s,
                    %(number_of_analysts)s, %(has_options)s,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (symbol) DO UPDATE SET
                    company_name = COALESCE(EXCLUDED.company_name, stocks_universe.company_name),
                    exchange = COALESCE(EXCLUDED.exchange, stocks_universe.exchange),
                    sector = COALESCE(EXCLUDED.sector, stocks_universe.sector),
                    industry = COALESCE(EXCLUDED.industry, stocks_universe.industry),
                    current_price = EXCLUDED.current_price,
                    volume = EXCLUDED.volume,
                    market_cap = EXCLUDED.market_cap,
                    sma_50 = EXCLUDED.sma_50,
                    sma_200 = EXCLUDED.sma_200,
                    rsi_14 = EXCLUDED.rsi_14,
                    has_options = EXCLUDED.has_options,
                    last_updated = CURRENT_TIMESTAMP
            """, data)
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Insert error {data.get('symbol')}: {e}")
            self.conn.rollback()
            return False
        finally:
            cur.close()

    def populate_all_stocks(self, symbols: List[str], batch_name: str):
        """Populate stocks from a list"""
        new_symbols = [s for s in symbols if s not in self.processed_stocks]
        logger.info(f"Processing {len(new_symbols)} new stocks from {batch_name}")

        added = 0
        skipped = 0
        errors = 0

        for i, symbol in enumerate(new_symbols):
            try:
                data = self.get_stock_data(symbol)
                if data:
                    if self.insert_stock(data):
                        self.processed_stocks.add(symbol)
                        added += 1
                else:
                    skipped += 1

                # Progress logging every 100 stocks
                if (i + 1) % 100 == 0:
                    logger.info(
                        f"[{batch_name}] Progress: {i+1}/{len(new_symbols)} "
                        f"(added: {added}, skipped: {skipped}, errors: {errors})"
                    )

                # Rate limiting - be nice to yfinance
                if (i + 1) % 10 == 0:
                    time.sleep(1)  # 1 second pause every 10 stocks

            except Exception as e:
                errors += 1
                logger.debug(f"Error {symbol}: {e}")

        logger.info(
            f"[{batch_name}] Complete: added {added}, "
            f"skipped {skipped}, errors {errors}"
        )
        return added

    def run(self) -> None:
        """Run the full population process"""
        logger.info("=" * 60)
        logger.info("POPULATING ALL US STOCKS")
        logger.info("=" * 60)

        self.get_existing_symbols()

        # Get all symbols from NASDAQ trader files
        all_symbols = self.get_symbols_from_nasdaq_csv()
        logger.info(f"Total unique symbols to process: {len(all_symbols)}")

        # Process in batches
        added = self.populate_all_stocks(all_symbols, "ALL_US_STOCKS")

        # Final count
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM stocks_universe")
        total_count = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM stocks_universe WHERE has_options = true"
        )
        options_count = cur.fetchone()[0]
        cur.close()

        logger.info("=" * 60)
        logger.info(f"COMPLETE: {total_count} total stocks, {options_count} with options")
        logger.info("=" * 60)

    def close(self) -> None:
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    populator = AllUSStocksPopulator()
    try:
        populator.run()
    finally:
        populator.close()
