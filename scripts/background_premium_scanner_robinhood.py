"""
Background Premium Scanner - Robinhood Edition
Uses Robinhood API instead of Yahoo Finance to avoid rate limiting.

Usage:
    python scripts/background_premium_scanner_robinhood.py          # Full scan
    python scripts/background_premium_scanner_robinhood.py --quick  # Quick scan (top 50 symbols)
    python scripts/background_premium_scanner_robinhood.py --dte 7  # Single DTE scan
    python scripts/background_premium_scanner_robinhood.py --symbols AAPL,TSLA,NVDA

Features:
- Uses Robinhood API (no yfinance rate limits)
- Built-in rate limiting (60 requests/minute)
- Stores results in premium_opportunities table
"""

import os
import sys
import time
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import asyncpg
import robin_stocks.robinhood as rh

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobinhoodPremiumScanner:
    """
    Premium scanner using Robinhood API.

    Advantages over yfinance:
    - No rate limiting issues
    - More accurate real-time data
    - Direct access to Greeks
    """

    # DTE targets to scan
    DTE_TARGETS = [7, 14, 30, 45]

    # Rate limiting - Robinhood is generous but we still pace ourselves
    DELAY_BETWEEN_SYMBOLS = 0.5  # 0.5 seconds between symbols

    def __init__(self) -> None:
        self.db_pool: Optional[asyncpg.Pool] = None
        self.logged_in = False
        self.stats = {
            'symbols_scanned': 0,
            'options_found': 0,
            'errors': 0,
            'start_time': None
        }

    def login_robinhood(self) -> None:
        """Login to Robinhood."""
        username = os.getenv('ROBINHOOD_USERNAME') or os.getenv('RH_USERNAME')
        password = os.getenv('ROBINHOOD_PASSWORD') or os.getenv('RH_PASSWORD')
        totp = os.getenv('ROBINHOOD_TOTP') or os.getenv('RH_TOTP')

        if not username or not password:
            raise ValueError("ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD must be set")

        try:
            if totp:
                import pyotp
                totp_code = pyotp.TOTP(totp).now()
                rh.login(username, password, mfa_code=totp_code)
            else:
                rh.login(username, password)

            self.logged_in = True
            logger.info("Successfully logged into Robinhood")
        except Exception as e:
            logger.error(f"Failed to login to Robinhood: {e}")
            raise

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

    async def get_tradingview_symbols(self) -> List[str]:
        """Get symbols from TradingView watchlists specifically."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT unnest(symbols) as symbol
                FROM scanner_watchlists
                WHERE source = 'tradingview' AND symbols IS NOT NULL
            """)
            symbols = [row['symbol'] for row in rows if row['symbol']]

        logger.info(f"Found {len(symbols)} TradingView watchlist symbols")
        return symbols

    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock price from Robinhood."""
        try:
            quote = rh.stocks.get_latest_price(symbol)
            if quote and quote[0]:
                return float(quote[0])
        except Exception as e:
            logger.debug(f"Error getting stock price for {symbol}: {e}")
        return None

    def get_stock_info(self, symbol: str) -> Dict:
        """Get stock info including name from Robinhood."""
        try:
            info = rh.stocks.get_fundamentals(symbol)
            if info and len(info) > 0:
                return info[0] if isinstance(info, list) else info
        except Exception as e:
            logger.debug(f"Error getting stock info for {symbol}: {e}")
        return {}

    def get_options_for_dte(self, symbol: str, target_dte: int) -> List[Dict]:
        """
        Get put options for a symbol near target DTE.
        Returns list of option opportunities.
        """
        try:
            # Get stock price first
            stock_price = self.get_stock_price(symbol)
            if not stock_price:
                return []

            # Get stock name
            stock_info = self.get_stock_info(symbol)
            company_name = stock_info.get('description', symbol) if stock_info else symbol

            # Get expiration dates
            today = datetime.now().date()
            target_date = today + timedelta(days=target_dte)

            # Find the closest expiration
            try:
                chains = rh.options.get_chains(symbol)
                if not chains or 'expiration_dates' not in chains:
                    return []

                expirations = chains['expiration_dates']
                if not expirations:
                    return []

                # Find closest expiration to target DTE
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

                exp_date = datetime.strptime(best_exp, '%Y-%m-%d').date()
                dte = (exp_date - today).days

                # Only accept if within 10 days of target
                if abs(dte - target_dte) > 10:
                    return []

            except Exception as e:
                logger.debug(f"Error getting expiration dates for {symbol}: {e}")
                return []

            # Get puts for this expiration
            try:
                puts = rh.options.find_options_by_expiration(
                    symbol,
                    expirationDate=best_exp,
                    optionType='put'
                )

                if not puts:
                    return []

            except Exception as e:
                logger.debug(f"Error getting puts for {symbol}: {e}")
                return []

            opportunities = []

            for option in puts:
                try:
                    strike = float(option.get('strike_price', 0))

                    # Only OTM puts (strike below stock price)
                    if strike >= stock_price:
                        continue

                    # Get market data
                    bid = float(option.get('bid_price', 0) or 0)
                    ask = float(option.get('ask_price', 0) or 0)

                    if bid <= 0:
                        continue

                    mid = (bid + ask) / 2
                    premium = bid  # Conservative estimate
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

                    # Greeks
                    delta = float(option.get('delta', 0) or 0)
                    gamma = float(option.get('gamma', 0) or 0)
                    theta = float(option.get('theta', 0) or 0)
                    vega = float(option.get('vega', 0) or 0)
                    iv = float(option.get('implied_volatility', 0) or 0) * 100

                    # Volume and OI
                    volume = int(option.get('volume', 0) or 0)
                    oi = int(option.get('open_interest', 0) or 0)

                    # Calculate break-even and POP
                    break_even = strike - premium
                    otm_pct = ((stock_price - strike) / stock_price) * 100
                    pop = 50 + (otm_pct * 2)
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

                except Exception as e:
                    logger.debug(f"Error processing option for {symbol}: {e}")
                    continue

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

    async def run_scan(self, symbols: List[str], dte_targets: List[int] = None):
        """Run the scan for all symbols and DTE targets."""
        if dte_targets is None:
            dte_targets = self.DTE_TARGETS

        self.stats['start_time'] = datetime.now()
        total_symbols = len(symbols)
        total_iterations = total_symbols * len(dte_targets)

        logger.info(f"Starting Robinhood scan: {total_symbols} symbols x {len(dte_targets)} DTE targets")

        iteration = 0
        for dte in dte_targets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Scanning DTE {dte}")
            logger.info(f"{'='*50}")

            batch_opportunities = []

            for i, symbol in enumerate(symbols):
                iteration += 1

                # Fetch options for this symbol
                opportunities = self.get_options_for_dte(symbol, dte)

                if opportunities:
                    batch_opportunities.extend(opportunities)
                    logger.info(f"  {symbol}: Found {len(opportunities)} opportunities")

                self.stats['symbols_scanned'] += 1

                # Save in batches of 100
                if len(batch_opportunities) >= 100:
                    await self.save_opportunities(batch_opportunities)
                    batch_opportunities = []

                # Progress update every 10 symbols
                if (i + 1) % 10 == 0:
                    elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                    progress = iteration / total_iterations * 100

                    if elapsed > 0:
                        rate = iteration / elapsed
                        remaining = (total_iterations - iteration) / rate if rate > 0 else 0
                        eta = str(timedelta(seconds=int(remaining)))
                    else:
                        eta = "calculating..."

                    logger.info(
                        f"DTE {dte} | {i+1}/{total_symbols} | "
                        f"Progress: {progress:.1f}% | "
                        f"Found: {self.stats['options_found']} | "
                        f"ETA: {eta}"
                    )

                # Rate limiting
                await asyncio.sleep(self.DELAY_BETWEEN_SYMBOLS)

            # Save remaining opportunities
            if batch_opportunities:
                await self.save_opportunities(batch_opportunities)

        # Final summary
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        logger.info(f"\n{'='*50}")
        logger.info("ROBINHOOD SCAN COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Time: {timedelta(seconds=int(elapsed))}")
        logger.info(f"Symbols scanned: {self.stats['symbols_scanned']}")
        logger.info(f"Opportunities found: {self.stats['options_found']}")
        logger.info(f"Errors: {self.stats['errors']}")


async def main():
    parser = argparse.ArgumentParser(description='Background Premium Scanner (Robinhood)')
    parser.add_argument('--quick', action='store_true', help='Quick scan (top 50 symbols)')
    parser.add_argument('--dte', type=int, help='Scan specific DTE only')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to scan')
    parser.add_argument('--tradingview', action='store_true', help='Scan only TradingView watchlist symbols')
    args = parser.parse_args()

    scanner = RobinhoodPremiumScanner()

    try:
        # Login to Robinhood first
        scanner.login_robinhood()

        await scanner.connect_db()
        await scanner.ensure_table_exists()

        # Get symbols
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        elif args.tradingview:
            symbols = await scanner.get_tradingview_symbols()
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
        # Logout
        try:
            rh.logout()
        except:
            pass


if __name__ == '__main__':
    asyncio.run(main())
