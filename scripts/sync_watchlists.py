"""
Sync Scanner Watchlists to Database
Collects watchlists from all sources and caches them for fast API responses.
Run periodically (every 15-30 minutes) or after adding new watchlists.
"""

import os
import sys
import time
import psycopg2
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres'),
        database=os.getenv('DB_NAME', 'magnus')
    )


def upsert_watchlist(conn, watchlist: Dict) -> bool:
    """Insert or update a single watchlist."""
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO scanner_watchlists
                (watchlist_id, source, name, symbols, category, sort_order, is_active, last_synced)
            VALUES (%s, %s, %s, %s, %s, %s, true, CURRENT_TIMESTAMP)
            ON CONFLICT (watchlist_id) DO UPDATE SET
                source = EXCLUDED.source,
                name = EXCLUDED.name,
                symbols = EXCLUDED.symbols,
                category = EXCLUDED.category,
                sort_order = EXCLUDED.sort_order,
                is_active = true,
                last_synced = CURRENT_TIMESTAMP
        """, (
            watchlist['watchlist_id'],
            watchlist['source'],
            watchlist['name'],
            watchlist['symbols'],
            watchlist.get('category'),
            watchlist.get('sort_order', 1000)
        ))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error upserting watchlist {watchlist.get('watchlist_id')}: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()


def get_predefined_watchlists() -> List[Dict]:
    """Get predefined static watchlists."""
    return [
        {
            "watchlist_id": "predefined_popular",
            "source": "predefined",
            "name": "Popular Stocks",
            "symbols": ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOG", "META", "PLTR", "SOFI"],
            "category": "popular",
            "sort_order": 10
        },
        {
            "watchlist_id": "predefined_tech",
            "source": "predefined",
            "name": "Tech Leaders",
            "symbols": ["AAPL", "MSFT", "NVDA", "AMD", "GOOG", "META", "INTC", "AVGO", "CRM", "ORCL"],
            "category": "popular",
            "sort_order": 20
        },
        {
            "watchlist_id": "predefined_high_iv",
            "source": "predefined",
            "name": "High IV Favorites",
            "symbols": ["TSLA", "MARA", "COIN", "RIOT", "GME", "AMC", "RIVN", "LCID", "NIO", "SNAP"],
            "category": "popular",
            "sort_order": 30
        },
        {
            "watchlist_id": "predefined_wheel_friendly",
            "source": "predefined",
            "name": "Wheel Friendly (<$50)",
            "symbols": ["SOFI", "PLTR", "F", "BAC", "SNAP", "HOOD", "NIO", "INTC", "AAL", "T"],
            "category": "popular",
            "sort_order": 40
        },
        {
            "watchlist_id": "predefined_etf_premium",
            "source": "predefined",
            "name": "ETF Premium Plays",
            "symbols": ["SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "GLD", "SLV", "TLT", "HYG"],
            "category": "popular",
            "sort_order": 50
        }
    ]


def get_database_watchlists(conn) -> List[Dict]:
    """Get watchlists from stocks_universe, etfs_universe, and sectors."""
    watchlists = []
    cur = conn.cursor()

    try:
        # Optionable stocks from stocks_universe
        cur.execute("""
            SELECT symbol
            FROM stocks_universe
            WHERE is_active = true AND has_options = true
            ORDER BY market_cap DESC NULLS LAST, symbol
        """)
        symbols = [row[0] for row in cur.fetchall()]
        if symbols:
            watchlists.append({
                "watchlist_id": "database_optionable_stocks",
                "source": "database",
                "name": f"Optionable Stocks ({len(symbols)})",
                "symbols": symbols,
                "category": "universe",
                "sort_order": 100
            })

        # All ETFs
        cur.execute("""
            SELECT symbol
            FROM etfs_universe
            WHERE is_active = true
            ORDER BY total_assets DESC NULLS LAST, symbol
        """)
        etf_symbols = [row[0] for row in cur.fetchall()]
        if etf_symbols:
            watchlists.append({
                "watchlist_id": "database_all_etfs",
                "source": "database",
                "name": f"All ETFs ({len(etf_symbols)})",
                "symbols": etf_symbols,
                "category": "universe",
                "sort_order": 110
            })

        # Sectors from stocks_universe
        cur.execute("""
            SELECT sector, array_agg(symbol ORDER BY market_cap DESC NULLS LAST) as symbols
            FROM stocks_universe
            WHERE is_active = true AND has_options = true AND sector IS NOT NULL AND sector != ''
            GROUP BY sector
            HAVING COUNT(*) >= 5
            ORDER BY sector
        """)
        rows = cur.fetchall()
        for i, (sector, sector_symbols) in enumerate(rows):
            sector_id = sector.lower().replace(' ', '_').replace('-', '_')
            watchlists.append({
                "watchlist_id": f"database_sector_{sector_id}",
                "source": "database",
                "name": f"Sector: {sector}",
                "symbols": sector_symbols[:100],  # Limit to top 100
                "category": "sector",
                "sort_order": 200 + i
            })

    except Exception as e:
        logger.error(f"Error fetching database watchlists: {e}")
    finally:
        cur.close()

    return watchlists


def get_tradingview_watchlists(conn) -> List[Dict]:
    """Get TradingView watchlists from tv_watchlists_api table."""
    watchlists = []
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT watchlist_id, name, symbols, symbol_count
            FROM tv_watchlists_api
            WHERE symbol_count > 0
            ORDER BY symbol_count DESC, name
        """)
        rows = cur.fetchall()

        stock_exchanges = ('NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS')
        for i, (tv_id, name, full_symbols, count) in enumerate(rows):
            if full_symbols:
                # Extract symbol from 'EXCHANGE:SYMBOL' format
                stock_symbols = []
                for fs in full_symbols:
                    if ':' in fs:
                        exchange, symbol = fs.split(':', 1)
                        if exchange.upper() in stock_exchanges:
                            stock_symbols.append(symbol)

                if stock_symbols:
                    watchlists.append({
                        "watchlist_id": f"tradingview_{tv_id}",
                        "source": "tradingview",
                        "name": f"TV: {name}",
                        "symbols": stock_symbols,
                        "category": "tradingview",
                        "sort_order": 300 + i
                    })

    except Exception as e:
        logger.error(f"Error fetching TradingView watchlists: {e}")
    finally:
        cur.close()

    return watchlists


def get_robinhood_portfolio() -> Optional[Dict]:
    """Get Robinhood portfolio as a watchlist (with timeout)."""
    try:
        import asyncio
        from backend.services.portfolio_service import get_portfolio_service

        async def fetch_portfolio():
            service = get_portfolio_service()
            return await asyncio.wait_for(service.get_positions(), timeout=10.0)

        positions = asyncio.run(fetch_portfolio())
        stock_symbols = [s.get("symbol") for s in positions.get("stocks", []) if s.get("symbol")]

        if stock_symbols:
            return {
                "watchlist_id": "robinhood_portfolio",
                "source": "robinhood",
                "name": "RH: My Portfolio",
                "symbols": stock_symbols,
                "category": "portfolio",
                "sort_order": 5  # Show portfolio first
            }
    except asyncio.TimeoutError:
        logger.warning("Robinhood portfolio fetch timed out after 10 seconds")
    except Exception as e:
        logger.warning(f"Error fetching Robinhood portfolio: {e}")

    return None


def log_sync(conn, sync_type: str, source: str, watchlists_count: int,
             total_symbols: int, duration: float, status: str, error: str = None):
    """Log sync operation to history table."""
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO scanner_watchlists_sync_log
                (sync_type, source, watchlists_synced, total_symbols,
                 duration_seconds, status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (sync_type, source, watchlists_count, total_symbols, duration, status, error))
        conn.commit()
    except Exception as e:
        logger.error(f"Error logging sync: {e}")
        conn.rollback()
    finally:
        cur.close()


def run_full_sync(include_robinhood: bool = True):
    """Run full sync of all watchlist sources."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("WATCHLIST SYNC - FULL")
    logger.info("=" * 60)

    conn = get_db_connection()

    try:
        all_watchlists = []
        total_symbols = 0

        # 1. Predefined watchlists
        logger.info("Syncing predefined watchlists...")
        predefined = get_predefined_watchlists()
        all_watchlists.extend(predefined)
        logger.info(f"  Found {len(predefined)} predefined watchlists")

        # 2. Database watchlists (stocks, ETFs, sectors)
        logger.info("Syncing database watchlists...")
        database = get_database_watchlists(conn)
        all_watchlists.extend(database)
        logger.info(f"  Found {len(database)} database watchlists")

        # 3. TradingView watchlists
        logger.info("Syncing TradingView watchlists...")
        tradingview = get_tradingview_watchlists(conn)
        all_watchlists.extend(tradingview)
        logger.info(f"  Found {len(tradingview)} TradingView watchlists")

        # 4. Robinhood portfolio (optional, may timeout)
        if include_robinhood:
            logger.info("Syncing Robinhood portfolio...")
            rh_portfolio = get_robinhood_portfolio()
            if rh_portfolio:
                all_watchlists.append(rh_portfolio)
                logger.info(f"  Found Robinhood portfolio with {len(rh_portfolio['symbols'])} symbols")
            else:
                logger.info("  Robinhood portfolio not available")

        # Upsert all watchlists
        logger.info(f"\nUpserting {len(all_watchlists)} watchlists to database...")
        success_count = 0
        for wl in all_watchlists:
            if upsert_watchlist(conn, wl):
                success_count += 1
                total_symbols += len(wl['symbols'])

        # Mark any missing watchlists as inactive
        cur = conn.cursor()
        active_ids = [wl['watchlist_id'] for wl in all_watchlists]
        cur.execute("""
            UPDATE scanner_watchlists
            SET is_active = false
            WHERE watchlist_id != ALL(%s)
        """, (active_ids,))
        deactivated = cur.rowcount
        conn.commit()
        cur.close()

        if deactivated > 0:
            logger.info(f"  Deactivated {deactivated} stale watchlists")

        duration = time.time() - start_time

        # Log the sync
        log_sync(conn, 'full', None, success_count, total_symbols, duration, 'success')

        # Final stats
        logger.info("\n" + "=" * 60)
        logger.info(f"SYNC COMPLETE in {duration:.2f}s")
        logger.info(f"  Watchlists synced: {success_count}/{len(all_watchlists)}")
        logger.info(f"  Total symbols: {total_symbols}")
        logger.info("=" * 60)

    except Exception as e:
        duration = time.time() - start_time
        log_sync(conn, 'full', None, 0, 0, duration, 'failed', str(e))
        logger.error(f"Sync failed: {e}")
        raise
    finally:
        conn.close()


def get_sync_stats():
    """Get current sync statistics."""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get watchlist counts
        cur.execute("""
            SELECT source, COUNT(*), SUM(symbol_count)
            FROM scanner_watchlists
            WHERE is_active = true
            GROUP BY source
            ORDER BY source
        """)
        by_source = cur.fetchall()

        # Get last sync time
        cur.execute("""
            SELECT MAX(last_synced), MIN(last_synced)
            FROM scanner_watchlists
            WHERE is_active = true
        """)
        newest, oldest = cur.fetchone()

        # Get total
        cur.execute("""
            SELECT COUNT(*), SUM(symbol_count)
            FROM scanner_watchlists
            WHERE is_active = true
        """)
        total_wl, total_sym = cur.fetchone()

        print("\n" + "=" * 50)
        print("SCANNER WATCHLISTS STATISTICS")
        print("=" * 50)
        print(f"\n{'Source':<15} {'Watchlists':>12} {'Symbols':>12}")
        print("-" * 40)
        for source, count, symbols in by_source:
            print(f"{source:<15} {count:>12} {symbols or 0:>12}")
        print("-" * 40)
        print(f"{'TOTAL':<15} {total_wl or 0:>12} {total_sym or 0:>12}")
        print(f"\nNewest sync: {newest}")
        print(f"Oldest sync: {oldest}")
        print("=" * 50)

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sync scanner watchlists to database')
    parser.add_argument('--stats', action='store_true', help='Show current statistics only')
    parser.add_argument('--no-robinhood', action='store_true', help='Skip Robinhood portfolio sync')
    args = parser.parse_args()

    if args.stats:
        get_sync_stats()
    else:
        run_full_sync(include_robinhood=not args.no_robinhood)
        get_sync_stats()
