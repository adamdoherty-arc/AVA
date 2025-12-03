"""
Update Premium Opportunities in Database
Scans popular stocks and stores premium data for quick access.
Run periodically (every 15-30 minutes during market hours).
"""

import os
import sys
import psycopg2
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import uuid

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


def get_optionable_stocks(conn, limit: int = 500) -> List[str]:
    """Get top optionable stocks by market cap."""
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol FROM stocks_universe
        WHERE is_active = true AND has_options = true
        ORDER BY market_cap DESC NULLS LAST
        LIMIT %s
    """, (limit,))
    symbols = [row[0] for row in cur.fetchall()]
    cur.close()
    return symbols


def scan_symbol_premiums(symbol: str, dte: int = 30, max_price: float = 500) -> List[Dict]:
    """Scan a single symbol for premium opportunities."""
    try:
        from src.premium_scanner_v2 import PremiumScannerV2
        scanner = PremiumScannerV2()
        results = scanner.scan_symbols(
            symbols=[symbol],
            dte=dte,
            max_price=max_price,
            min_premium_pct=0.5  # Lower threshold to capture more data
        )
        return results
    except Exception as e:
        logger.debug(f"Error scanning {symbol}: {e}")
        return []


def upsert_premium_opportunity(conn, data: Dict, scan_id: str):
    """Insert or update a premium opportunity."""
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO premium_opportunities (
                symbol, company_name, option_type, strike, expiration, dte,
                stock_price, bid, ask, mid, premium, premium_pct,
                annualized_return, monthly_return,
                delta, gamma, theta, vega, rho,
                implied_volatility, volume, open_interest,
                break_even, max_profit, max_loss, pop,
                last_updated, scan_id
            ) VALUES (
                %(symbol)s, %(company_name)s, %(option_type)s, %(strike)s,
                %(expiration)s, %(dte)s, %(stock_price)s, %(bid)s, %(ask)s,
                %(mid)s, %(premium)s, %(premium_pct)s,
                %(annualized_return)s, %(monthly_return)s,
                %(delta)s, %(gamma)s, %(theta)s, %(vega)s, %(rho)s,
                %(implied_volatility)s, %(volume)s, %(open_interest)s,
                %(break_even)s, %(max_profit)s, %(max_loss)s, %(pop)s,
                CURRENT_TIMESTAMP, %(scan_id)s
            )
            ON CONFLICT (symbol, option_type, strike, expiration)
            DO UPDATE SET
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
                rho = EXCLUDED.rho,
                implied_volatility = EXCLUDED.implied_volatility,
                volume = EXCLUDED.volume,
                open_interest = EXCLUDED.open_interest,
                break_even = EXCLUDED.break_even,
                max_profit = EXCLUDED.max_profit,
                max_loss = EXCLUDED.max_loss,
                pop = EXCLUDED.pop,
                last_updated = CURRENT_TIMESTAMP,
                scan_id = EXCLUDED.scan_id
        """, {
            'symbol': data.get('symbol'),
            'company_name': data.get('company_name'),
            'option_type': data.get('option_type', 'PUT'),
            'strike': data.get('strike'),
            'expiration': data.get('expiration'),
            'dte': data.get('dte'),
            'stock_price': data.get('stock_price'),
            'bid': data.get('bid'),
            'ask': data.get('ask'),
            'mid': data.get('mid'),
            'premium': data.get('premium'),
            'premium_pct': data.get('premium_pct'),
            'annualized_return': data.get('annualized_return'),
            'monthly_return': data.get('monthly_return'),
            'delta': data.get('delta'),
            'gamma': data.get('gamma'),
            'theta': data.get('theta'),
            'vega': data.get('vega'),
            'rho': data.get('rho'),
            'implied_volatility': data.get('implied_volatility'),
            'volume': data.get('volume'),
            'open_interest': data.get('open_interest'),
            'break_even': data.get('break_even'),
            'max_profit': data.get('max_profit'),
            'max_loss': data.get('max_loss'),
            'pop': data.get('pop'),
            'scan_id': scan_id
        })
        conn.commit()
        return True
    except Exception as e:
        logger.debug(f"Error upserting premium: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()


def save_scan_to_history(conn, scan_id: str, symbols: List[str], dte: int,
                         max_price: float, min_premium_pct: float,
                         results: List[Dict]):
    """Save scan metadata to history."""
    cur = conn.cursor()
    try:
        import json
        cur.execute("""
            INSERT INTO premium_scan_history
            (scan_id, symbols, symbol_count, dte, max_price, min_premium_pct, result_count, results)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (scan_id) DO UPDATE SET
                result_count = EXCLUDED.result_count,
                results = EXCLUDED.results
        """, (scan_id, symbols, len(symbols), dte, max_price, min_premium_pct,
              len(results), json.dumps(results)))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving scan history: {e}")
        conn.rollback()
    finally:
        cur.close()


def cleanup_old_data(conn, days: int = 7):
    """Remove premium data older than specified days."""
    cur = conn.cursor()
    try:
        # Remove old opportunities where expiration has passed
        cur.execute("""
            DELETE FROM premium_opportunities
            WHERE expiration < CURRENT_DATE
        """)
        expired = cur.rowcount

        # Remove stale data not updated recently
        cur.execute("""
            DELETE FROM premium_opportunities
            WHERE last_updated < NOW() - INTERVAL '%s days'
        """, (days,))
        stale = cur.rowcount

        # Remove old scan history
        cur.execute("""
            DELETE FROM premium_scan_history
            WHERE created_at < NOW() - INTERVAL '30 days'
        """)
        history = cur.rowcount

        conn.commit()
        if expired or stale or history:
            logger.info(f"Cleanup: removed {expired} expired, {stale} stale, {history} history entries")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        cur.close()


def run_premium_update(stock_limit: int = 100, dte_targets: List[int] = None):
    """Main function to update premium data for multiple DTEs."""
    if dte_targets is None:
        dte_targets = [7, 14, 30, 45]  # Default: scan weekly, 2-week, monthly, and 45-day

    logger.info("=" * 60)
    logger.info("PREMIUM DATA UPDATE - MULTI-DTE")
    logger.info(f"DTE Targets: {dte_targets}")
    logger.info("=" * 60)

    conn = get_db_connection()

    try:
        # Get symbols to scan
        symbols = get_optionable_stocks(conn, limit=stock_limit)
        logger.info(f"Scanning {len(symbols)} symbols across {len(dte_targets)} DTEs...")

        total_success = 0
        total_errors = 0

        # Scan each DTE target
        for dte in dte_targets:
            scan_id = f"auto_{dte}dte_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"\n--- Scanning DTE {dte} ---")

            dte_results = []
            success_count = 0
            error_count = 0

            for i, symbol in enumerate(symbols):
                try:
                    results = scan_symbol_premiums(symbol, dte=dte)
                    for result in results:
                        if upsert_premium_opportunity(conn, result, scan_id):
                            success_count += 1
                            dte_results.append(result)
                        else:
                            error_count += 1

                    # Progress every 25 symbols
                    if (i + 1) % 25 == 0:
                        logger.info(f"  DTE {dte}: {i+1}/{len(symbols)} symbols, {success_count} premiums")

                except Exception as e:
                    error_count += 1
                    logger.debug(f"Error scanning {symbol} DTE {dte}: {e}")

            # Save scan to history
            save_scan_to_history(
                conn, scan_id, symbols, dte,
                500.0, 0.5, dte_results
            )

            total_success += success_count
            total_errors += error_count
            logger.info(f"  DTE {dte} complete: {success_count} premiums stored")

        # Cleanup old data
        cleanup_old_data(conn)

        # Final stats
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM premium_opportunities")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT dte) FROM premium_opportunities")
        unique_dtes = cur.fetchone()[0]
        cur.close()

        logger.info("\n" + "=" * 60)
        logger.info(f"COMPLETE: {total_success} premiums stored, {total_errors} errors")
        logger.info(f"Total premiums in database: {total} across {unique_dtes} DTEs")
        logger.info("=" * 60)

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update premium opportunities for multiple DTEs')
    parser.add_argument('--limit', type=int, default=100, help='Number of stocks to scan')
    parser.add_argument('--dte', type=int, nargs='+', default=[7, 14, 30, 45],
                        help='Days to expiration targets (default: 7 14 30 45)')
    args = parser.parse_args()

    run_premium_update(stock_limit=args.limit, dte_targets=args.dte)
