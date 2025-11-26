"""
Diagnostic script to identify Premium Scanner bug
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def test_delta_queries():
    """Test different delta query variations"""

    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        database='magnus',
        user='postgres',
        password=os.getenv('DB_PASSWORD')
    )

    cur = conn.cursor()

    print("=" * 80)
    print("PREMIUM SCANNER BUG DIAGNOSIS")
    print("=" * 80)

    # First, check what delta values actually exist
    print("\n1. ACTUAL DELTA VALUES IN DATABASE (7-day DTE)")
    print("-" * 80)
    cur.execute("""
        SELECT
            MIN(delta) as min_delta,
            MAX(delta) as max_delta,
            AVG(delta) as avg_delta,
            COUNT(*) as total_count,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM stock_premiums
        WHERE dte BETWEEN 5 AND 9
    """)
    result = cur.fetchone()
    print(f"Min Delta: {result[0]}")
    print(f"Max Delta: {result[1]}")
    print(f"Avg Delta: {result[2]}")
    print(f"Total Records: {result[3]}")
    print(f"Unique Symbols: {result[4]}")

    # Show sample delta values
    print("\n2. SAMPLE DELTA VALUES (First 10 records)")
    print("-" * 80)
    cur.execute("""
        SELECT symbol, delta, dte, premium
        FROM stock_premiums
        WHERE dte BETWEEN 5 AND 9
        ORDER BY delta DESC
        LIMIT 10
    """)
    for row in cur.fetchall():
        print(f"Symbol: {row[0]:6s} | Delta: {row[1]:8.4f} | DTE: {row[2]:2d} | Premium: ${row[3]:6.2f}")

    # Test the CURRENT BROKEN query (with default filters)
    print("\n3. CURRENT QUERY (BROKEN) - Default filters (-1.0 to 0.0)")
    print("-" * 80)
    delta_min, delta_max = -1.0, 0.0
    print(f"Filter: delta BETWEEN {delta_min} AND {delta_max}")
    print(f"SQL: WHERE (sp.delta BETWEEN {delta_min} AND {delta_max} OR ABS(sp.delta) BETWEEN ABS({delta_min}) AND ABS({delta_max}))")
    print(f"Simplified: WHERE (sp.delta BETWEEN -1.0 AND 0.0 OR ABS(sp.delta) BETWEEN 1.0 AND 0.0)")
    print(f"\nPROBLEM: 'BETWEEN 1.0 AND 0.0' is BACKWARDS! Should be 'BETWEEN 0.0 AND 1.0'")

    cur.execute("""
        SELECT COUNT(*), COUNT(DISTINCT symbol)
        FROM stock_premiums sp
        WHERE sp.dte BETWEEN 5 AND 9
          AND (sp.delta BETWEEN %s AND %s OR ABS(sp.delta) BETWEEN ABS(%s) AND ABS(%s))
    """, (delta_min, delta_max, delta_min, delta_max))
    result = cur.fetchone()
    print(f"\nResults with BROKEN query: {result[0]} records, {result[1]} symbols")

    # Test FIXED query (swap parameters for ABS)
    print("\n4. FIXED QUERY - Swap parameters for ABS clause")
    print("-" * 80)
    print(f"SQL: WHERE (sp.delta BETWEEN {delta_min} AND {delta_max} OR ABS(sp.delta) BETWEEN ABS({delta_max}) AND ABS({delta_min}))")
    print(f"Simplified: WHERE (sp.delta BETWEEN -1.0 AND 0.0 OR ABS(sp.delta) BETWEEN 0.0 AND 1.0)")

    cur.execute("""
        SELECT COUNT(*), COUNT(DISTINCT symbol)
        FROM stock_premiums sp
        WHERE sp.dte BETWEEN 5 AND 9
          AND (sp.delta BETWEEN %s AND %s OR ABS(sp.delta) BETWEEN ABS(%s) AND ABS(%s))
    """, (delta_min, delta_max, delta_max, delta_min))  # SWAPPED: delta_max, delta_min
    result = cur.fetchone()
    print(f"\nResults with FIXED query: {result[0]} records, {result[1]} symbols")

    # Test just the first part (without ABS)
    print("\n5. QUERY WITHOUT ABS CLAUSE (simpler approach)")
    print("-" * 80)
    cur.execute("""
        SELECT COUNT(*), COUNT(DISTINCT symbol)
        FROM stock_premiums sp
        WHERE sp.dte BETWEEN 5 AND 9
          AND sp.delta BETWEEN %s AND %s
    """, (delta_min, delta_max))
    result = cur.fetchone()
    print(f"Results with simple BETWEEN (no ABS): {result[0]} records, {result[1]} symbols")

    # Test what happens with old default (-0.4 to -0.2)
    print("\n6. OLD DEFAULT FILTERS (-0.4 to -0.2)")
    print("-" * 80)
    delta_min_old, delta_max_old = -0.4, -0.2
    cur.execute("""
        SELECT COUNT(*), COUNT(DISTINCT symbol)
        FROM stock_premiums sp
        WHERE sp.dte BETWEEN 5 AND 9
          AND sp.delta BETWEEN %s AND %s
    """, (delta_min_old, delta_max_old))
    result = cur.fetchone()
    print(f"Results with old filters: {result[0]} records, {result[1]} symbols")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("The bug is in the ABS() clause parameter order.")
    print("Current: ABS(sp.delta) BETWEEN ABS(delta_min) AND ABS(delta_max)")
    print("         ABS(sp.delta) BETWEEN 1.0 AND 0.0  ← WRONG (backwards)")
    print("\nFix: Swap parameters for ABS clause")
    print("     ABS(sp.delta) BETWEEN ABS(delta_max) AND ABS(delta_min)")
    print("     ABS(sp.delta) BETWEEN 0.0 AND 1.0  ← CORRECT")
    print("=" * 80)

    cur.close()
    conn.close()

if __name__ == "__main__":
    test_delta_queries()
