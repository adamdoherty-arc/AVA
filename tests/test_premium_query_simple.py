"""
Simple test of Premium Scanner query
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor()

# Default parameters
dte_min, dte_max = 5, 9
delta_min, delta_max = -1.0, 0.0
min_premium = 0.0
min_stock_price = 0.0
max_stock_price = 10000.0

print("=" * 80)
print("PREMIUM SCANNER QUERY TEST")
print("=" * 80)
print(f"\nParameters: DTE {dte_min}-{dte_max}, Delta {delta_min} to {delta_max}")

# Test the exact query
query = '''
    SELECT DISTINCT ON (sp.symbol)
        sp.symbol,
        sd.current_price as stock_price,
        sp.strike_price,
        sp.premium,
        sp.delta
    FROM stock_premiums sp
    LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
    LEFT JOIN stocks s ON sp.symbol = s.symbol
    WHERE sp.dte BETWEEN %s AND %s
      AND sp.premium >= %s
      AND (sp.delta BETWEEN %s AND %s OR ABS(sp.delta) BETWEEN ABS(%s) AND ABS(%s))
      AND sp.strike_price > 0
      AND (sd.current_price BETWEEN %s AND %s OR sd.current_price IS NULL)
    ORDER BY sp.symbol, (sp.premium / sp.dte) DESC
'''

try:
    cur.execute(query, (dte_min, dte_max, min_premium, delta_min, delta_max,
                        delta_min, delta_max, min_stock_price, max_stock_price))
    results = cur.fetchall()
    print(f"\nQUERY SUCCESS: {len(results)} rows returned")

    if len(results) > 0:
        print("\nFirst 10 results:")
        for row in results[:10]:
            print(f"  {row[0]:6s} | Stock: ${row[1] or 0:7.2f} | Strike: ${row[2]:7.2f} | Premium: ${row[3]:6.2f} | Delta: {row[4]:6.3f}")
    else:
        print("\nNO RESULTS! Debugging...")

        # Test without joins
        print("\n1. Test without joins:")
        cur.execute("""
            SELECT COUNT(*) FROM stock_premiums sp
            WHERE sp.dte BETWEEN %s AND %s
              AND sp.premium >= %s
              AND (sp.delta BETWEEN %s AND %s OR ABS(sp.delta) BETWEEN ABS(%s) AND ABS(%s))
              AND sp.strike_price > 0
        """, (dte_min, dte_max, min_premium, delta_min, delta_max, delta_min, delta_max))
        print(f"   Without joins: {cur.fetchone()[0]} records")

        # Test with joins but without stock price filter
        print("\n2. Test without stock price filter:")
        cur.execute("""
            SELECT COUNT(*) FROM stock_premiums sp
            LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
            WHERE sp.dte BETWEEN %s AND %s
              AND sp.premium >= %s
              AND (sp.delta BETWEEN %s AND %s OR ABS(sp.delta) BETWEEN ABS(%s) AND ABS(%s))
              AND sp.strike_price > 0
        """, (dte_min, dte_max, min_premium, delta_min, delta_max, delta_min, delta_max))
        print(f"   Without stock price filter: {cur.fetchone()[0]} records")

        # Test with stock price filter
        print("\n3. Test WITH stock price filter:")
        cur.execute("""
            SELECT COUNT(*) FROM stock_premiums sp
            LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
            WHERE sp.dte BETWEEN %s AND %s
              AND sp.premium >= %s
              AND (sp.delta BETWEEN %s AND %s OR ABS(sp.delta) BETWEEN ABS(%s) AND ABS(%s))
              AND sp.strike_price > 0
              AND (sd.current_price BETWEEN %s AND %s OR sd.current_price IS NULL)
        """, (dte_min, dte_max, min_premium, delta_min, delta_max, delta_min, delta_max,
              min_stock_price, max_stock_price))
        print(f"   With stock price filter: {cur.fetchone()[0]} records")

except Exception as e:
    print(f"\nQUERY FAILED: {e}")
    import traceback
    traceback.print_exc()

cur.close()
conn.close()

print("\n" + "=" * 80)
