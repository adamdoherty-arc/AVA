"""
Populate Stocks and ETFs Universe Tables
Creates comprehensive database tables for stocks and ETFs with market data
"""

import os
import psycopg2
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import logging
from dotenv import load_dotenv
import time

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


class StockETFPopulator:
    """Populate stocks and ETFs universe tables with market data"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'database': os.getenv('DB_NAME', 'magnus')
        }
        self.conn = psycopg2.connect(**self.db_config)

    def create_tables(self) -> None:
        """Create stocks_universe and etfs_universe tables"""
        cur = self.conn.cursor()

        try:
            # Stocks Universe Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stocks_universe (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) UNIQUE NOT NULL,
                    company_name VARCHAR(255),
                    exchange VARCHAR(50),
                    sector VARCHAR(100),
                    industry VARCHAR(150),
                    current_price NUMERIC(12, 4),
                    previous_close NUMERIC(12, 4),
                    open_price NUMERIC(12, 4),
                    day_high NUMERIC(12, 4),
                    day_low NUMERIC(12, 4),
                    week_52_high NUMERIC(12, 4),
                    week_52_low NUMERIC(12, 4),
                    volume BIGINT,
                    avg_volume_10d BIGINT,
                    avg_volume_3m BIGINT,
                    market_cap BIGINT,
                    shares_outstanding BIGINT,
                    float_shares BIGINT,
                    pe_ratio NUMERIC(12, 4),
                    forward_pe NUMERIC(12, 4),
                    peg_ratio NUMERIC(12, 4),
                    price_to_book NUMERIC(12, 4),
                    price_to_sales NUMERIC(12, 4),
                    enterprise_value BIGINT,
                    ev_to_ebitda NUMERIC(12, 4),
                    ev_to_revenue NUMERIC(12, 4),
                    profit_margin NUMERIC(12, 6),
                    operating_margin NUMERIC(12, 6),
                    gross_margin NUMERIC(12, 6),
                    roe NUMERIC(12, 6),
                    roa NUMERIC(12, 6),
                    revenue_growth NUMERIC(12, 6),
                    earnings_growth NUMERIC(12, 6),
                    dividend_yield NUMERIC(12, 6),
                    dividend_rate NUMERIC(12, 4),
                    payout_ratio NUMERIC(12, 6),
                    ex_dividend_date DATE,
                    total_cash BIGINT,
                    total_debt BIGINT,
                    debt_to_equity NUMERIC(12, 4),
                    current_ratio NUMERIC(12, 4),
                    free_cash_flow BIGINT,
                    beta NUMERIC(12, 6),
                    sma_50 NUMERIC(12, 4),
                    sma_200 NUMERIC(12, 4),
                    rsi_14 NUMERIC(12, 4),
                    target_high_price NUMERIC(12, 4),
                    target_low_price NUMERIC(12, 4),
                    target_mean_price NUMERIC(12, 4),
                    recommendation_key VARCHAR(50),
                    number_of_analysts INTEGER,
                    has_options BOOLEAN DEFAULT FALSE,
                    data_source VARCHAR(50) DEFAULT 'yfinance',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # ETFs Universe Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS etfs_universe (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) UNIQUE NOT NULL,
                    fund_name VARCHAR(255),
                    exchange VARCHAR(50),
                    category VARCHAR(100),
                    fund_family VARCHAR(150),
                    current_price NUMERIC(12, 4),
                    previous_close NUMERIC(12, 4),
                    open_price NUMERIC(12, 4),
                    day_high NUMERIC(12, 4),
                    day_low NUMERIC(12, 4),
                    week_52_high NUMERIC(12, 4),
                    week_52_low NUMERIC(12, 4),
                    nav_price NUMERIC(12, 4),
                    volume BIGINT,
                    avg_volume_10d BIGINT,
                    avg_volume_3m BIGINT,
                    total_assets BIGINT,
                    expense_ratio NUMERIC(12, 6),
                    yield_ttm NUMERIC(12, 6),
                    ytd_return NUMERIC(12, 6),
                    three_year_return NUMERIC(12, 6),
                    five_year_return NUMERIC(12, 6),
                    holdings_count INTEGER,
                    beta NUMERIC(12, 6),
                    sma_50 NUMERIC(12, 4),
                    sma_200 NUMERIC(12, 4),
                    rsi_14 NUMERIC(12, 4),
                    dividend_yield NUMERIC(12, 6),
                    dividend_rate NUMERIC(12, 4),
                    ex_dividend_date DATE,
                    has_options BOOLEAN DEFAULT FALSE,
                    data_source VARCHAR(50) DEFAULT 'yfinance',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stocks_symbol
                ON stocks_universe(symbol)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stocks_sector
                ON stocks_universe(sector)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stocks_market_cap
                ON stocks_universe(market_cap DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_etfs_symbol
                ON etfs_universe(symbol)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_etfs_category
                ON etfs_universe(category)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_etfs_assets
                ON etfs_universe(total_assets DESC)
            """)

            self.conn.commit()
            logger.info("Tables created successfully")

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch comprehensive stock data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            quote_type = info.get('quoteType', '')
            if quote_type == 'ETF':
                return None

            # Get historical data for technical indicators
            hist = ticker.history(period="1y")

            sma_50 = None
            sma_200 = None
            rsi_14 = None

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
                    else:
                        rsi_14 = 50.0

            # Parse ex-dividend date
            ex_div_date = None
            if info.get('exDividendDate'):
                try:
                    ex_div_date = datetime.fromtimestamp(
                        info['exDividendDate']
                    ).date()
                except Exception:
                    pass

            # Check for options
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
            logger.warning(f"Error fetching data for {symbol}: {e}")
            return None

    def get_etf_data(self, symbol: str) -> Optional[Dict]:
        """Fetch comprehensive ETF data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info.get('quoteType') != 'ETF':
                return None

            # Get historical data for technical indicators
            hist = ticker.history(period="1y")

            sma_50 = None
            sma_200 = None
            rsi_14 = None

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
                    else:
                        rsi_14 = 50.0

            # Parse ex-dividend date
            ex_div_date = None
            if info.get('exDividendDate'):
                try:
                    ex_div_date = datetime.fromtimestamp(
                        info['exDividendDate']
                    ).date()
                except Exception:
                    pass

            # Check for options
            has_options = False
            try:
                has_options = len(ticker.options) > 0
            except Exception:
                pass

            return {
                'symbol': symbol,
                'fund_name': info.get('longName') or info.get('shortName'),
                'exchange': info.get('exchange'),
                'category': info.get('category'),
                'fund_family': info.get('fundFamily'),
                'current_price': convert_value(
                    info.get('regularMarketPrice') or info.get('previousClose')
                ),
                'previous_close': convert_value(info.get('previousClose')),
                'open_price': convert_value(info.get('regularMarketOpen')),
                'day_high': convert_value(info.get('regularMarketDayHigh')),
                'day_low': convert_value(info.get('regularMarketDayLow')),
                'week_52_high': convert_value(info.get('fiftyTwoWeekHigh')),
                'week_52_low': convert_value(info.get('fiftyTwoWeekLow')),
                'nav_price': convert_value(info.get('navPrice')),
                'volume': convert_value(info.get('regularMarketVolume')),
                'avg_volume_10d': convert_value(info.get('averageVolume10days')),
                'avg_volume_3m': convert_value(info.get('averageVolume')),
                'total_assets': convert_value(info.get('totalAssets')),
                'expense_ratio': convert_value(
                    info.get('annualReportExpenseRatio')
                ),
                'yield_ttm': convert_value(info.get('yield')),
                'ytd_return': convert_value(info.get('ytdReturn')),
                'three_year_return': convert_value(
                    info.get('threeYearAverageReturn')
                ),
                'five_year_return': convert_value(
                    info.get('fiveYearAverageReturn')
                ),
                'holdings_count': convert_value(info.get('holdingsCount')),
                'beta': convert_value(
                    info.get('beta3Year') or info.get('beta')
                ),
                'sma_50': sma_50,
                'sma_200': sma_200,
                'rsi_14': rsi_14,
                'dividend_yield': convert_value(info.get('yield')),
                'dividend_rate': convert_value(info.get('dividendRate')),
                'ex_dividend_date': ex_div_date,
                'has_options': has_options,
            }

        except Exception as e:
            logger.warning(f"Error fetching ETF data for {symbol}: {e}")
            return None

    def insert_stock(self, data: Dict):
        """Insert or update stock data"""
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
                    company_name = EXCLUDED.company_name,
                    current_price = EXCLUDED.current_price,
                    previous_close = EXCLUDED.previous_close,
                    volume = EXCLUDED.volume,
                    market_cap = EXCLUDED.market_cap,
                    pe_ratio = EXCLUDED.pe_ratio,
                    sma_50 = EXCLUDED.sma_50,
                    sma_200 = EXCLUDED.sma_200,
                    rsi_14 = EXCLUDED.rsi_14,
                    last_updated = CURRENT_TIMESTAMP
            """, data)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error inserting stock {data.get('symbol')}: {e}")
            self.conn.rollback()
        finally:
            cur.close()

    def insert_etf(self, data: Dict):
        """Insert or update ETF data"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO etfs_universe (
                    symbol, fund_name, exchange, category, fund_family,
                    current_price, previous_close, open_price, day_high,
                    day_low, week_52_high, week_52_low, nav_price,
                    volume, avg_volume_10d, avg_volume_3m, total_assets,
                    expense_ratio, yield_ttm, ytd_return,
                    three_year_return, five_year_return, holdings_count,
                    beta, sma_50, sma_200, rsi_14,
                    dividend_yield, dividend_rate, ex_dividend_date,
                    has_options, last_updated
                ) VALUES (
                    %(symbol)s, %(fund_name)s, %(exchange)s, %(category)s,
                    %(fund_family)s, %(current_price)s, %(previous_close)s,
                    %(open_price)s, %(day_high)s, %(day_low)s,
                    %(week_52_high)s, %(week_52_low)s, %(nav_price)s,
                    %(volume)s, %(avg_volume_10d)s, %(avg_volume_3m)s,
                    %(total_assets)s, %(expense_ratio)s, %(yield_ttm)s,
                    %(ytd_return)s, %(three_year_return)s,
                    %(five_year_return)s, %(holdings_count)s,
                    %(beta)s, %(sma_50)s, %(sma_200)s, %(rsi_14)s,
                    %(dividend_yield)s, %(dividend_rate)s,
                    %(ex_dividend_date)s, %(has_options)s, CURRENT_TIMESTAMP
                )
                ON CONFLICT (symbol) DO UPDATE SET
                    fund_name = EXCLUDED.fund_name,
                    current_price = EXCLUDED.current_price,
                    previous_close = EXCLUDED.previous_close,
                    volume = EXCLUDED.volume,
                    total_assets = EXCLUDED.total_assets,
                    sma_50 = EXCLUDED.sma_50,
                    sma_200 = EXCLUDED.sma_200,
                    rsi_14 = EXCLUDED.rsi_14,
                    last_updated = CURRENT_TIMESTAMP
            """, data)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error inserting ETF {data.get('symbol')}: {e}")
            self.conn.rollback()
        finally:
            cur.close()

    def get_symbols_from_watchlists(self) -> None:
        """Get unique stock symbols from TradingView watchlists"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                SELECT DISTINCT symbol, exchange
                FROM tv_symbols_api
                WHERE exchange IN (
                    'NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS', 'NMS', 'NGM'
                )
                ORDER BY symbol
            """)
            rows = cur.fetchall()
            symbols = [row[0] for row in rows]
            logger.info(f"Found {len(symbols)} symbols from watchlists")
            return symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []
        finally:
            cur.close()

    def populate_from_watchlists(self) -> None:
        """Populate stocks and ETFs tables from TradingView watchlists"""
        symbols = self.get_symbols_from_watchlists()
        if not symbols:
            logger.warning("No symbols found in watchlists")
            return

        stocks_count = 0
        etfs_count = 0
        failed = []

        logger.info(f"Processing {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols):
            try:
                # Try as stock first
                stock_data = self.get_stock_data(symbol)
                if stock_data:
                    self.insert_stock(stock_data)
                    stocks_count += 1
                    logger.info(f"[{i+1}/{len(symbols)}] Stock: {symbol}")
                else:
                    # Try as ETF
                    etf_data = self.get_etf_data(symbol)
                    if etf_data:
                        self.insert_etf(etf_data)
                        etfs_count += 1
                        logger.info(f"[{i+1}/{len(symbols)}] ETF: {symbol}")
                    else:
                        failed.append(symbol)

                # Rate limiting
                if (i + 1) % 10 == 0:
                    time.sleep(1)

            except Exception as e:
                failed.append(symbol)
                logger.error(f"Error processing {symbol}: {e}")

        logger.info(f"\nPopulation complete!")
        logger.info(f"Stocks: {stocks_count}, ETFs: {etfs_count}")
        logger.info(f"Failed: {len(failed)}")

    def populate_common_etfs(self) -> None:
        """Populate common ETFs that traders use"""
        common_etfs = [
            # Major Index ETFs
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO',
            # Sector ETFs
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC',
            'XLY', 'XLP', 'XLB', 'XLU', 'XLRE',
            # Bond ETFs
            'TLT', 'IEF', 'SHY', 'BND', 'AGG', 'LQD', 'HYG', 'JNK',
            # Commodity ETFs
            'GLD', 'SLV', 'USO', 'UNG', 'GDX', 'GDXJ',
            # International ETFs
            'EFA', 'EEM', 'VEA', 'VWO', 'FXI', 'EWJ', 'EWZ',
            # Leveraged/Inverse ETFs
            'TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'UVXY', 'VXX',
            # Thematic ETFs
            'ARKK', 'ARKG', 'ARKF', 'ARKW', 'XBI', 'IBB', 'SOXX', 'SMH',
            # Dividend ETFs
            'VIG', 'VYM', 'SCHD', 'DVY', 'SDY',
            # Real Estate ETFs
            'VNQ', 'IYR', 'SCHH',
            # Other Popular
            'URA', 'JETS', 'KWEB', 'TAN', 'ICLN', 'LIT', 'BLOK'
        ]

        logger.info(f"Populating {len(common_etfs)} common ETFs...")

        for i, symbol in enumerate(common_etfs):
            try:
                etf_data = self.get_etf_data(symbol)
                if etf_data:
                    self.insert_etf(etf_data)
                    logger.info(f"[{i+1}/{len(common_etfs)}] ETF: {symbol}")

                if (i + 1) % 10 == 0:
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error processing ETF {symbol}: {e}")

        logger.info("Common ETFs population complete!")

    def populate_sp500_stocks(self) -> None:
        """Populate S&P 500 stocks"""
        try:
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(sp500_url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].str.replace(
                '.', '-', regex=False
            ).tolist()

            logger.info(f"Populating {len(symbols)} S&P 500 stocks...")

            for i, symbol in enumerate(symbols):
                try:
                    stock_data = self.get_stock_data(symbol)
                    if stock_data:
                        self.insert_stock(stock_data)
                        logger.info(f"[{i+1}/{len(symbols)}] {symbol}")

                    if (i + 1) % 10 == 0:
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Error: {symbol}: {e}")

            logger.info("S&P 500 population complete!")

        except Exception as e:
            logger.error(f"Error fetching S&P 500 list: {e}")

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    populator = StockETFPopulator()

    try:
        print("Creating tables...")
        populator.create_tables()

        print("\nPopulating from TradingView watchlists...")
        populator.populate_from_watchlists()

        print("\nPopulating common ETFs...")
        populator.populate_common_etfs()

        print("\nDone!")

    finally:
        populator.close()
