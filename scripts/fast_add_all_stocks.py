"""
FAST: Add ALL US Stock symbols to database
Just adds symbols - no yfinance data fetch (that's slow)
Data can be updated later in batches
"""

import os
import psycopg2
import requests
from typing import List, Set
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres'),
        database=os.getenv('DB_NAME', 'magnus')
    )


def get_all_us_symbols() -> List[str]:
    """Get ALL US stock symbols from NASDAQ trader files"""
    symbols = []

    # NASDAQ listed stocks
    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        response = requests.get(url, timeout=30)
        lines = response.text.strip().split('\n')
        for line in lines[1:-1]:
            parts = line.split('|')
            if len(parts) > 0 and parts[0]:
                symbol = parts[0].strip()
                # Skip test symbols and special chars
                if symbol and len(symbol) <= 5 and symbol.isalpha():
                    symbols.append(symbol)
        logger.info(f"NASDAQ: {len(symbols)} symbols")
    except Exception as e:
        logger.error(f"Error fetching NASDAQ: {e}")

    # NYSE/AMEX listed stocks
    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        response = requests.get(url, timeout=30)
        lines = response.text.strip().split('\n')
        nyse_count = 0
        for line in lines[1:-1]:
            parts = line.split('|')
            if len(parts) > 0 and parts[0]:
                symbol = parts[0].strip()
                # Skip special symbols
                if symbol and len(symbol) <= 5 and symbol.isalpha():
                    if symbol not in symbols:
                        symbols.append(symbol)
                        nyse_count += 1
        logger.info(f"NYSE/AMEX: {nyse_count} symbols")
    except Exception as e:
        logger.error(f"Error fetching NYSE/AMEX: {e}")

    return symbols


def get_existing_symbols(conn) -> Set[str]:
    """Get symbols already in database"""
    cur = conn.cursor()
    cur.execute("SELECT symbol FROM stocks_universe")
    existing = {row[0] for row in cur.fetchall()}
    cur.close()
    return existing


def bulk_insert_symbols(conn, symbols: List[str]):
    """Bulk insert symbols - FAST"""
    cur = conn.cursor()

    inserted = 0
    for symbol in symbols:
        try:
            cur.execute("""
                INSERT INTO stocks_universe (symbol, is_active, has_options)
                VALUES (%s, true, true)
                ON CONFLICT (symbol) DO NOTHING
            """, (symbol,))
            if cur.rowcount > 0:
                inserted += 1
        except Exception as e:
            logger.debug(f"Skip {symbol}: {e}")

    conn.commit()
    cur.close()
    return inserted


def main():
    logger.info("=" * 50)
    logger.info("FAST ADD ALL US STOCKS")
    logger.info("=" * 50)

    conn = get_db_connection()

    # Get existing
    existing = get_existing_symbols(conn)
    logger.info(f"Already have: {len(existing)} stocks")

    # Get all symbols
    all_symbols = get_all_us_symbols()
    logger.info(f"Total symbols found: {len(all_symbols)}")

    # Filter new ones
    new_symbols = [s for s in all_symbols if s not in existing]
    logger.info(f"New symbols to add: {len(new_symbols)}")

    # Bulk insert
    if new_symbols:
        inserted = bulk_insert_symbols(conn, new_symbols)
        logger.info(f"Inserted: {inserted} new stocks")

    # Final count
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM stocks_universe")
    total = cur.fetchone()[0]
    cur.close()

    logger.info("=" * 50)
    logger.info(f"DONE! Total stocks in database: {total}")
    logger.info("=" * 50)

    conn.close()


if __name__ == "__main__":
    main()
