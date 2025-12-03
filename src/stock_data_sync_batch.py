"""Stock Data Sync - Batch Mode (syncs small batches to avoid rate limiting)"""

import psycopg2
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class StockDataSyncBatch:
    """Syncs stock market data and premiums in small batches"""

    def __init__(self, batch_size=20):
        self.batch_size = batch_size
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', 5432),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres123!')
        )
        self.create_tables()

    def create_tables(self) -> None:
        """Create stock data tables"""
        cur = self.conn.cursor()

        # Stock market data table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                symbol VARCHAR(20) PRIMARY KEY,
                company_name VARCHAR(255),
                current_price DECIMAL(10,2),
                price_change DECIMAL(10,2),
                price_change_pct DECIMAL(10,2),
                day_high DECIMAL(10,2),
                day_low DECIMAL(10,2),
                volume BIGINT,
                avg_volume BIGINT,
                market_cap BIGINT,
                pe_ratio DECIMAL(10,2),
                dividend_yield DECIMAL(10,2),
                beta DECIMAL(10,2),
                week_52_high DECIMAL(10,2),
                week_52_low DECIMAL(10,2),
                sector VARCHAR(100),
                industry VARCHAR(100),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Stock premiums table (for put options)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_premiums (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) REFERENCES stock_data(symbol) ON DELETE CASCADE,
                expiration_date DATE,
                dte INTEGER,
                strike_type VARCHAR(20),
                strike_price DECIMAL(10,2),
                bid DECIMAL(10,2),
                ask DECIMAL(10,2),
                mid DECIMAL(10,2),
                premium DECIMAL(10,2),
                premium_pct DECIMAL(10,2),
                monthly_return DECIMAL(10,2),
                annual_return DECIMAL(10,2),
                implied_volatility DECIMAL(10,2),
                volume INTEGER,
                open_interest INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, expiration_date, strike_type)
            )
        """)

        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_premiums_symbol ON stock_premiums(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_premiums_monthly_return ON stock_premiums(monthly_return DESC)")

        self.conn.commit()
        cur.close()
        logger.info("Database tables created successfully")

    def sync_stock_data(self, symbol: str) -> bool:
        """Sync market data for a single stock"""
        try:
            # Add random user agent to avoid detection
            yf_session = yf.Ticker(symbol)
            info = yf_session.info
            hist = yf_session.history(period="5d")

            if hist.empty:
                logger.warning(f"No data for {symbol}")
                return False

            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close > 0 else 0

            cur = self.conn.cursor()

            # Upsert stock data
            cur.execute("""
                INSERT INTO stock_data (
                    symbol, company_name, current_price, price_change, price_change_pct,
                    day_high, day_low, volume, avg_volume, market_cap, pe_ratio,
                    dividend_yield, beta, week_52_high, week_52_low, sector, industry, last_updated
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (symbol) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    current_price = EXCLUDED.current_price,
                    price_change = EXCLUDED.price_change,
                    price_change_pct = EXCLUDED.price_change_pct,
                    day_high = EXCLUDED.day_high,
                    day_low = EXCLUDED.day_low,
                    volume = EXCLUDED.volume,
                    avg_volume = EXCLUDED.avg_volume,
                    market_cap = EXCLUDED.market_cap,
                    pe_ratio = EXCLUDED.pe_ratio,
                    dividend_yield = EXCLUDED.dividend_yield,
                    beta = EXCLUDED.beta,
                    week_52_high = EXCLUDED.week_52_high,
                    week_52_low = EXCLUDED.week_52_low,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    last_updated = NOW()
            """, (
                symbol,
                info.get('longName', symbol),
                current_price,
                change,
                change_pct,
                hist['High'].iloc[-1],
                hist['Low'].iloc[-1],
                int(hist['Volume'].iloc[-1]),
                int(hist['Volume'].mean()),
                info.get('marketCap', 0),
                info.get('trailingPE', 0),
                (info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0,
                info.get('beta', 0),
                info.get('fiftyTwoWeekHigh', 0),
                info.get('fiftyTwoWeekLow', 0),
                info.get('sector', 'N/A'),
                info.get('industry', 'N/A')
            ))

            self.conn.commit()
            cur.close()
            return True

        except Exception as e:
            logger.error(f"Error syncing {symbol}: {e}")
            return False

    def sync_premiums(self, symbol: str, target_dte: int = 45) -> bool:
        """Sync premium data for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]

            expirations = ticker.options
            if not expirations:
                return False

            # Find expiration closest to target DTE
            target_date = datetime.now() + timedelta(days=target_dte)
            closest_exp = min(expirations, key=lambda x: abs(
                (datetime.strptime(x, '%Y-%m-%d') - target_date).days
            ))

            options = ticker.option_chain(closest_exp)
            puts = options.puts

            if puts.empty:
                return False

            exp_date = datetime.strptime(closest_exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days

            # Delete old premiums for this symbol and expiration
            cur = self.conn.cursor()
            cur.execute("DELETE FROM stock_premiums WHERE symbol = %s AND expiration_date = %s", (symbol, closest_exp))

            # Calculate premiums for different strikes
            strikes_config = {
                'ATM': 1.00,
                '5%_OTM': 0.95,
                '10%_OTM': 0.90,
                '15%_OTM': 0.85
            }

            for strike_type, multiplier in strikes_config.items():
                target_strike = round(current_price * multiplier, 0)
                closest_strike = puts.iloc[(puts['strike'] - target_strike).abs().argsort()[0]]

                bid = closest_strike['bid']
                ask = closest_strike['ask']
                mid = (bid + ask) / 2
                premium = mid * 100
                capital = closest_strike['strike'] * 100
                premium_pct = (premium / capital * 100) if capital > 0 else 0
                monthly_return = (premium_pct / dte * 30) if dte > 0 else 0
                annual_return = (premium_pct / dte * 365) if dte > 0 else 0

                cur.execute("""
                    INSERT INTO stock_premiums (
                        symbol, expiration_date, dte, strike_type, strike_price,
                        bid, ask, mid, premium, premium_pct, monthly_return, annual_return,
                        implied_volatility, volume, open_interest, last_updated
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol, expiration_date, strike_type) DO UPDATE SET
                        dte = EXCLUDED.dte,
                        strike_price = EXCLUDED.strike_price,
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
                        last_updated = NOW()
                """, (
                    symbol, closest_exp, dte, strike_type, closest_strike['strike'],
                    bid, ask, mid, premium, premium_pct, monthly_return, annual_return,
                    closest_strike.get('impliedVolatility', 0) * 100,
                    int(closest_strike['volume']) if closest_strike['volume'] else 0,
                    int(closest_strike['openInterest']) if closest_strike['openInterest'] else 0
                ))

            self.conn.commit()
            cur.close()
            return True

        except Exception as e:
            logger.error(f"Error syncing premiums for {symbol}: {e}")
            return False

    def sync_batch(self) -> None:
        """Sync a batch of symbols that haven't been updated recently"""
        cur = self.conn.cursor()

        # Get symbols that haven't been synced yet or are oldest
        cur.execute("""
            SELECT s.symbol, sd.last_updated
            FROM tv_symbols_api s
            LEFT JOIN stock_data sd ON s.symbol = sd.symbol
            WHERE sd.symbol IS NULL OR sd.last_updated < NOW() - INTERVAL '1 day'
            GROUP BY s.symbol, sd.last_updated
            ORDER BY sd.last_updated NULLS FIRST
            LIMIT %s
        """, (self.batch_size,))

        symbols = [row[0] for row in cur.fetchall()]
        cur.close()

        if not symbols:
            logger.info("All symbols are up to date!")
            return

        logger.info(f"Syncing batch of {len(symbols)} symbols...")
        logger.info("Using aggressive rate limiting to avoid IP ban...")

        success_count = 0
        premium_count = 0

        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"[{idx}/{len(symbols)}] Processing {symbol}...")

            if self.sync_stock_data(symbol):
                success_count += 1
                logger.info(f"  [OK] {symbol} - market data synced")

                # Wait between stock data and premium fetch
                time.sleep(random.uniform(2, 4))

                if self.sync_premiums(symbol):
                    premium_count += 1
                    logger.info(f"  [OK] {symbol} - premiums synced")

            # Aggressive rate limiting: 5-8 seconds between each symbol
            if idx < len(symbols):  # Don't wait after last symbol
                wait_time = random.uniform(5, 8)
                logger.info(f"  Waiting {wait_time:.1f}s before next symbol...")
                time.sleep(wait_time)

        logger.info(f"Batch complete: {success_count} stocks, {premium_count} with premiums")

    def get_sync_status(self) -> None:
        """Get synchronization status"""
        cur = self.conn.cursor()

        # Total symbols
        cur.execute("SELECT COUNT(DISTINCT symbol) FROM tv_symbols_api")
        total = cur.fetchone()[0]

        # Synced today
        cur.execute("""
            SELECT COUNT(*) FROM stock_data
            WHERE last_updated >= CURRENT_DATE
        """)
        synced_today = cur.fetchone()[0]

        # Synced ever
        cur.execute("SELECT COUNT(*) FROM stock_data")
        synced_total = cur.fetchone()[0]

        # Never synced
        never_synced = total - synced_total

        cur.close()

        return {
            'total': total,
            'synced_today': synced_today,
            'synced_total': synced_total,
            'never_synced': never_synced,
            'percent_complete': (synced_total / total * 100) if total > 0 else 0
        }

    def close(self) -> None:
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    print("="*60)
    print("Stock Data Sync - Magnus (Batch Mode)")
    print("="*60)

    sync = StockDataSyncBatch(batch_size=20)

    # Show status
    status = sync.get_sync_status()
    print(f"\nSync Status:")
    print(f"  Total symbols: {status['total']}")
    print(f"  Synced (ever): {status['synced_total']} ({status['percent_complete']:.1f}%)")
    print(f"  Synced today: {status['synced_today']}")
    print(f"  Never synced: {status['never_synced']}")

    print(f"\nSyncing next batch of {sync.batch_size} symbols...")
    print("This uses aggressive rate limiting to avoid Yahoo Finance bans.")
    print("Run this script multiple times throughout the day to sync all symbols.\n")

    sync.sync_batch()

    # Show updated status
    status = sync.get_sync_status()
    print(f"\n" + "="*60)
    print("Batch Sync Complete!")
    print("="*60)
    print(f"Progress: {status['synced_total']}/{status['total']} symbols ({status['percent_complete']:.1f}%)")
    print(f"Remaining: {status['never_synced']} symbols")
    print("\nRun this script again later to sync more symbols.")
    print("Refresh your Magnus dashboard to see updated information.")

    sync.close()
