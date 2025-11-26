"""
Add performance indexes for multi-expiration options display
"""

import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def add_indexes():
    """Add composite index for efficient option lookups"""

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    cur = conn.cursor()

    print("Adding performance indexes...")

    # Composite index for multi-DTE option lookups
    print("1. Creating composite index on (symbol, dte, delta, monthly_return)...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_premiums_multi_dte_lookup
        ON stock_premiums(symbol, dte, delta, monthly_return DESC)
    """)

    # Index for delta range queries
    print("2. Creating index on delta for range queries...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_premiums_delta
        ON stock_premiums(delta)
        WHERE delta IS NOT NULL
    """)

    # Index for DTE range queries
    print("3. Creating index on dte...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_premiums_dte
        ON stock_premiums(dte)
    """)

    conn.commit()

    print("\n✓ All indexes created successfully!")

    # Analyze table to update statistics
    print("\nAnalyzing table to update statistics...")
    cur.execute("ANALYZE stock_premiums")
    conn.commit()

    print("✓ Analysis complete!")

    # Show index information
    print("\nCurrent indexes on stock_premiums:")
    cur.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'stock_premiums'
        ORDER BY indexname
    """)

    for row in cur.fetchall():
        print(f"  - {row[0]}")

    cur.close()
    conn.close()

    print("\n✓ Database optimization complete!")

if __name__ == '__main__':
    add_indexes()
