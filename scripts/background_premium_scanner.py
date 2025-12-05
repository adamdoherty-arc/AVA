"""
Background Premium Scanner - Pre-computes premiums for all watchlists
Runs as a standalone process to populate the premium_opportunities table.

Usage:
    python scripts/background_premium_scanner.py          # Full scan
    python scripts/background_premium_scanner.py --quick  # Quick scan (top 50 symbols)
    python scripts/background_premium_scanner.py --dte 7  # Single DTE scan

Features:
- Efficient batching with rate limiting
- Concurrent symbol processing
- Progress tracking with ETA
- Stores results in premium_opportunities table
- Can be scheduled via cron/Task Scheduler
"""

import os
import sys
import time
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackgroundPremiumScanner:
    """
    Efficient background scanner for premium opportunities.

    Design:
    - Uses ThreadPoolExecutor for parallel yfinance calls
    - Batches database inserts for efficiency
    - Rate limits to avoid API throttling
    - Tracks progress with ETA
    """

    # DTE targets to scan
    DTE_TARGETS = [7, 14, 30, 45]

    # Batch sizes - Aggressive parallelism for full scans
    SYMBOL_BATCH_SIZE = 50  # Process 50 symbols concurrently
    DB_BATCH_SIZE = 200     # Insert 200 records at a time

    # Rate limiting - Minimal delay for speed
    DELAY_BETWEEN_BATCHES = 0.2  # seconds

    def __init__(self) -> None:
        self.db_pool: Optional[asyncpg.Pool] = None
        self.stats = {
            'symbols_scanned': 0,
            'options_found': 0,
            'errors': 0,
            'start_time': None
        }

    async def connect_db(self) -> None:
        """Create database connection pool."""
        self.db_pool = await asyncpg.create_pool(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD'),
            min_size=2,
            max_size=10
        )
        logger.info("Database connection established")

    async def close_db(self) -> None:
        """Close database pool."""
        if self.db_pool:
            await self.db_pool.close()

    async def ensure_table_exists(self) -> None:
        """Create premium_opportunities table if not exists."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS premium_opportunities (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    company_name VARCHAR(200),
                    option_type VARCHAR(10) NOT NULL,
                    strike NUMERIC(10,2) NOT NULL,
                    expiration DATE NOT NULL,
                    dte INTEGER NOT NULL,
                    stock_price NUMERIC(10,2),
                    bid NUMERIC(10,4),
                    ask NUMERIC(10,4),
                    mid NUMERIC(10,4),
                    premium NUMERIC(10,4),
                    premium_pct NUMERIC(10,4),
                    annualized_return NUMERIC(10,2),
                    monthly_return NUMERIC(10,4),
                    delta NUMERIC(10,4),
                    gamma NUMERIC(10,6),
                    theta NUMERIC(10,4),
                    vega NUMERIC(10,4),
                    implied_volatility NUMERIC(10,4),
                    volume INTEGER,
                    open_interest INTEGER,
                    break_even NUMERIC(10,2),
                    pop NUMERIC(10,2),
                    last_updated TIMESTAMP DEFAULT NOW(),
                    UNIQUE(symbol, option_type, strike, expiration)
                )
            """)

            # Create indexes for fast queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_premium_opp_symbol ON premium_opportunities(symbol);
                CREATE INDEX IF NOT EXISTS idx_premium_opp_dte ON premium_opportunities(dte);
                CREATE INDEX IF NOT EXISTS idx_premium_opp_annual ON premium_opportunities(annualized_return DESC);
                CREATE INDEX IF NOT EXISTS idx_premium_opp_updated ON premium_opportunities(last_updated);
            """)

        logger.info("Table premium_opportunities ready")

    async def get_watchlist_symbols(self) -> List[str]:
        """Get all unique symbols from scanner_watchlists."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT unnest(symbols) as symbol
                FROM scanner_watchlists
                WHERE symbols IS NOT NULL
            """)
            symbols = [row['symbol'] for row in rows if row['symbol']]

        # Add popular stocks that might not be in watchlists
        popular = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'AMZN', 'GOOG', 'META',
                   'PLTR', 'SOFI', 'COIN', 'HOOD', 'SPY', 'QQQ', 'IWM', 'DIA']

        all_symbols = list(set(symbols + popular))
        logger.info(f"Found {len(all_symbols)} unique symbols to scan")
        return all_symbols

    def fetch_options_for_symbol(self, symbol: str, target_dte: int) -> List[Dict]:
        """
        Fetch put options for a symbol near target DTE.
        Returns list of option opportunities.
        """
        try:
            ticker = yf.Ticker(symbol)

            # Get stock price
            info = ticker.info
            stock_price = info.get('regularMarketPrice') or info.get('currentPrice')
            company_name = info.get('shortName', symbol)

            if not stock_price:
                return []

            # Get expiration dates
            expirations = ticker.options
            if not expirations:
                return []

            # Find expiration closest to target DTE
            today = datetime.now().date()
            best_exp = None
            best_diff = float('inf')

            for exp in expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                dte = (exp_date - today).days
                diff = abs(dte - target_dte)
                if diff < best_diff and dte > 0:
                    best_diff = diff
                    best_exp = exp

            if not best_exp:
                return []

            # Get options chain
            chain = ticker.option_chain(best_exp)
            puts = chain.puts

            opportunities = []
            exp_date = datetime.strptime(best_exp, '%Y-%m-%d').date()
            dte = (exp_date - today).days

            # Filter for OTM puts with good premium
            for _, row in puts.iterrows():
                strike = float(row['strike'])

                # Only OTM puts (strike below stock price)
                if strike >= stock_price:
                    continue

                bid = float(row.get('bid', 0) or 0)
                ask = float(row.get('ask', 0) or 0)

                if bid <= 0:
                    continue

                mid = (bid + ask) / 2
                premium = bid  # Use bid for conservative estimate
                premium_pct = (premium / strike) * 100

                # Calculate returns
                if dte > 0:
                    monthly_return = premium_pct * (30 / dte)
                    annualized_return = premium_pct * (365 / dte)
                else:
                    monthly_return = premium_pct
                    annualized_return = premium_pct * 12

                # Only include decent opportunities
                if annualized_return < 10:
                    continue

                iv = float(row.get('impliedVolatility', 0) or 0) * 100
                delta = float(row.get('delta', 0) or 0)
                gamma = float(row.get('gamma', 0) or 0)
                theta = float(row.get('theta', 0) or 0)
                vega = float(row.get('vega', 0) or 0)
                volume = int(row.get('volume', 0) or 0)
                oi = int(row.get('openInterest', 0) or 0)

                # Calculate break-even and probability of profit
                break_even = strike - premium
                otm_pct = ((stock_price - strike) / stock_price) * 100
                pop = 50 + (otm_pct * 2)  # Rough estimate
                pop = min(95, max(20, pop))

                opportunities.append({
                    'symbol': symbol,
                    'company_name': company_name[:200] if company_name else symbol,
                    'option_type': 'PUT',
                    'strike': strike,
                    'expiration': exp_date,
                    'dte': dte,
                    'stock_price': stock_price,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'premium': premium,
                    'premium_pct': premium_pct,
                    'annualized_return': annualized_return,
                    'monthly_return': monthly_return,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'implied_volatility': iv,
                    'volume': volume,
                    'open_interest': oi,
                    'break_even': break_even,
                    'pop': pop
                })

            return opportunities

        except Exception as e:
            logger.debug(f"Error fetching options for {symbol}: {e}")
            return []

    async def save_opportunities(self, opportunities: List[Dict]):
        """Batch insert opportunities to database."""
        if not opportunities:
            return

        async with self.db_pool.acquire() as conn:
            # Use UPSERT to update existing records
            await conn.executemany("""
                INSERT INTO premium_opportunities (
                    symbol, company_name, option_type, strike, expiration, dte,
                    stock_price, bid, ask, mid, premium, premium_pct,
                    annualized_return, monthly_return,
                    delta, gamma, theta, vega, implied_volatility,
                    volume, open_interest, break_even, pop, last_updated
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10, $11, $12,
                    $13, $14,
                    $15, $16, $17, $18, $19,
                    $20, $21, $22, $23, NOW()
                )
                ON CONFLICT (symbol, option_type, strike, expiration) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    dte = EXCLUDED.dte,
                    stock_price = EXCLUDED.stock_price,
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    mid = EXCLUDED.mid,
                    premium = EXCLUDED.premium,
                    premium_pct = EXCLUDED.premium_pct,
                    annualized_return = EXCLUDED.annualized_return,
                    monthly_return = EXCLUDED.monthly_return,
                    delta = EXCLUDED.delta,
                    gamma = EXCLUDED.gamma,
                    theta = EXCLUDED.theta,
                    vega = EXCLUDED.vega,
                    implied_volatility = EXCLUDED.implied_volatility,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest,
                    break_even = EXCLUDED.break_even,
                    pop = EXCLUDED.pop,
                    last_updated = NOW()
            """, [
                (
                    o['symbol'], o['company_name'], o['option_type'], o['strike'], o['expiration'], o['dte'],
                    o['stock_price'], o['bid'], o['ask'], o['mid'], o['premium'], o['premium_pct'],
                    o['annualized_return'], o['monthly_return'],
                    o['delta'], o['gamma'], o['theta'], o['vega'], o['implied_volatility'],
                    o['volume'], o['open_interest'], o['break_even'], o['pop']
                ) for o in opportunities
            ])

        self.stats['options_found'] += len(opportunities)

    async def scan_batch(self, symbols: List[str], dte: int) -> List[Dict]:
        """Scan a batch of symbols concurrently."""
        all_opportunities = []

        with ThreadPoolExecutor(max_workers=self.SYMBOL_BATCH_SIZE) as executor:
            futures = {
                executor.submit(self.fetch_options_for_symbol, symbol, dte): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    opportunities = future.result()
                    all_opportunities.extend(opportunities)
                    self.stats['symbols_scanned'] += 1
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")
                    self.stats['errors'] += 1

        return all_opportunities

    async def run_scan(self, symbols: List[str], dte_targets: List[int] = None):
        """Run the full scan for all symbols and DTE targets."""
        if dte_targets is None:
            dte_targets = self.DTE_TARGETS

        self.stats['start_time'] = datetime.now()
        total_symbols = len(symbols)

        logger.info(f"Starting scan: {total_symbols} symbols x {len(dte_targets)} DTE targets")

        for dte in dte_targets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Scanning DTE {dte}")
            logger.info(f"{'='*50}")

            # Process in batches
            for i in range(0, total_symbols, self.SYMBOL_BATCH_SIZE):
                batch = symbols[i:i + self.SYMBOL_BATCH_SIZE]
                batch_num = (i // self.SYMBOL_BATCH_SIZE) + 1
                total_batches = (total_symbols + self.SYMBOL_BATCH_SIZE - 1) // self.SYMBOL_BATCH_SIZE

                # Scan batch
                opportunities = await self.scan_batch(batch, dte)

                # Save to database
                if opportunities:
                    await self.save_opportunities(opportunities)

                # Progress update
                elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                progress = self.stats['symbols_scanned'] / (total_symbols * len(dte_targets)) * 100

                if elapsed > 0:
                    rate = self.stats['symbols_scanned'] / elapsed
                    remaining = (total_symbols * len(dte_targets) - self.stats['symbols_scanned']) / rate if rate > 0 else 0
                    eta = str(timedelta(seconds=int(remaining)))
                else:
                    eta = "calculating..."

                logger.info(
                    f"DTE {dte} | Batch {batch_num}/{total_batches} | "
                    f"Progress: {progress:.1f}% | "
                    f"Found: {self.stats['options_found']} | "
                    f"ETA: {eta}"
                )

                # Rate limiting
                await asyncio.sleep(self.DELAY_BETWEEN_BATCHES)

        # Final summary
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        logger.info(f"\n{'='*50}")
        logger.info("SCAN COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Time: {timedelta(seconds=int(elapsed))}")
        logger.info(f"Symbols scanned: {self.stats['symbols_scanned']}")
        logger.info(f"Opportunities found: {self.stats['options_found']}")
        logger.info(f"Errors: {self.stats['errors']}")

    async def cleanup_old_data(self, days: int = 7):
        """Remove opportunities older than specified days."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM premium_opportunities
                WHERE last_updated < NOW() - INTERVAL '%s days'
            """ % days)
            logger.info(f"Cleaned up old data: {result}")


async def main():
    parser = argparse.ArgumentParser(description='Background Premium Scanner')
    parser.add_argument('--quick', action='store_true', help='Quick scan (top 50 symbols)')
    parser.add_argument('--dte', type=int, help='Scan specific DTE only')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to scan')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old data only')
    args = parser.parse_args()

    scanner = BackgroundPremiumScanner()

    try:
        await scanner.connect_db()
        await scanner.ensure_table_exists()

        if args.cleanup:
            await scanner.cleanup_old_data()
            return

        # Get symbols
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        else:
            symbols = await scanner.get_watchlist_symbols()

        # Quick mode uses fewer symbols
        if args.quick:
            symbols = symbols[:50]

        # DTE targets
        dte_targets = [args.dte] if args.dte else None

        # Run scan
        await scanner.run_scan(symbols, dte_targets)

    finally:
        await scanner.close_db()


if __name__ == '__main__':
    asyncio.run(main())
