"""
Test the EXACT query from Premium Scanner with all joins and filters
"""
import psycopg2
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor()

print("Testing EXACT Premium Scanner query...")
print("=" * 80)

# Default parameters from Premium Scanner
dte_min, dte_max = 5, 9
delta_min, delta_max = -1.0, 0.0
min_premium = 0.0
min_stock_price = 0.0
max_stock_price = 10000.0

print(f"\nParameters:")
print(f"  DTE: {dte_min} to {dte_max}")
print(f"  Delta: {delta_min} to {delta_max}")
print(f"  Min Premium: ${min_premium}")
print(f"  Stock Price: ${min_stock_price} to ${max_stock_price}")

# EXACT query from premium_scanner_page.py lines 55-84
query = '''
    SELECT DISTINCT ON (sp.symbol)
        sp.symbol,
        sd.current_price as stock_price,
        sp.strike_price,
        sp.premium,
        sp.dte,
        sp.premium_pct,
        sp.annual_return,
        sp.delta,
        sp.prob_profit,
        sp.implied_volatility,
        sp.volume,
        sp.open_interest,
        sp.strike_type,
        sp.expiration_date,
        sp.bid,
        sp.ask,
        s.company_name,
        s.sector
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

print("\n" + "-" * 80)
print("EXECUTING QUERY...")
print("-" * 80)

try:
    # EXACT parameters from line 86-87
    cur.execute(query, (dte_min, dte_max, min_premium, delta_min, delta_max,
                        delta_min, delta_max, min_stock_price, max_stock_price))

    columns = [desc[0] for desc in cur.description]
    results = cur.fetchall()

    print(f"\n✓ Query executed successfully!")
    print(f"✓ Columns: {len(columns)}")
    print(f"✓ Rows returned: {len(results)}")

    if len(results) > 0:
        df = pd.DataFrame(results, columns=columns)
        print(f"\n✓ DataFrame created: {len(df)} rows")
        print("\nFirst 5 results:")
        print("-" * 80)
        for idx, row in df.head(5).iterrows():
            print(f"{row['symbol']:6s} | Stock: ${row['stock_price']:7.2f if pd.notna(row['stock_price']) else 0:7.2f} | "
                  f"Strike: ${row['strike_price']:7.2f} | Premium: ${row['premium']:6.2f} | Delta: {row['delta']:6.3f}")
    else:
        print("\n✗ NO RESULTS RETURNED!")
        print("\nDebugging: Let's check each filter condition separately...")

        # Check without delta filter
        print("\n1. Without delta filter:")
        cur.execute("""
            SELECT COUNT(DISTINCT sp.symbol), COUNT(*)
            FROM stock_premiums sp
            LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
            WHERE sp.dte BETWEEN %s AND %s
              AND sp.premium >= %s
              AND sp.strike_price > 0
              AND (sd.current_price BETWEEN %s AND %s OR sd.current_price IS NULL)
        """, (dte_min, dte_max, min_premium, min_stock_price, max_stock_price))
        result = cur.fetchone()
        print(f"   Result: {result[0]} symbols, {result[1]} records")

        # Check with delta filter
        print("\n2. With delta filter:")
        cur.execute("""
            SELECT COUNT(DISTINCT sp.symbol), COUNT(*)
            FROM stock_premiums sp
            LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
            WHERE sp.dte BETWEEN %s AND %s
              AND sp.premium >= %s
              AND (sp.delta BETWEEN %s AND %s OR ABS(sp.delta) BETWEEN ABS(%s) AND ABS(%s))
              AND sp.strike_price > 0
              AND (sd.current_price BETWEEN %s AND %s OR sd.current_price IS NULL)
        """, (dte_min, dte_max, min_premium, delta_min, delta_max,
              delta_min, delta_max, min_stock_price, max_stock_price))
        result = cur.fetchone()
        print(f"   Result: {result[0]} symbols, {result[1]} records")

        # Check if DISTINCT ON is the issue
        print("\n3. Without DISTINCT ON:")
        cur.execute(query.replace("SELECT DISTINCT ON (sp.symbol)", "SELECT"),
                    (dte_min, dte_max, min_premium, delta_min, delta_max,
                     delta_min, delta_max, min_stock_price, max_stock_price))
        results_no_distinct = cur.fetchall()
        print(f"   Result: {len(results_no_distinct)} records")

except Exception as e:
    print(f"\n✗ QUERY FAILED!")
    print(f"✗ Error: {e}")
    print(f"✗ Error type: {type(e).__name__}")

    import traceback
    print("\nFull traceback:")
    print("-" * 80)
    traceback.print_exc()

cur.close()
conn.close()

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("If query returned results: Premium Scanner bug is likely caching or UI issue")
print("If query failed: There's a SQL syntax or parameter issue")
print("=" * 80)
