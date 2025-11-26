"""
Test if ABS(%s) in SQL query is the problem
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

print("Testing ABS(%s) in SQL query...")
print("=" * 80)

# Test 1: Using ABS(%s) - WRONG
print("\n1. CURRENT APPROACH - ABS(%s) in SQL")
print("-" * 80)
delta_min, delta_max = -1.0, 0.0
try:
    cur.execute("""
        SELECT COUNT(*)
        FROM stock_premiums
        WHERE dte BETWEEN 5 AND 9
          AND (delta BETWEEN %s AND %s OR ABS(delta) BETWEEN ABS(%s) AND ABS(%s))
    """, (delta_min, delta_max, delta_min, delta_max))
    result = cur.fetchone()
    print(f"Result: {result[0]} records")
    print("ERROR: This query worked but ABS(%s) is not standard SQL!")
    print("PostgreSQL might be interpreting this differently than expected")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Calculate ABS in Python, pass as value - CORRECT
print("\n2. CORRECT APPROACH - Calculate ABS in Python")
print("-" * 80)
abs_delta_min = abs(delta_min)  # 1.0
abs_delta_max = abs(delta_max)  # 0.0
# For BETWEEN, need smaller value first
abs_min = min(abs_delta_min, abs_delta_max)  # 0.0
abs_max = max(abs_delta_min, abs_delta_max)  # 1.0

try:
    cur.execute("""
        SELECT COUNT(*)
        FROM stock_premiums
        WHERE dte BETWEEN 5 AND 9
          AND (delta BETWEEN %s AND %s OR ABS(delta) BETWEEN %s AND %s)
    """, (delta_min, delta_max, abs_min, abs_max))
    result = cur.fetchone()
    print(f"Parameters: delta_min={delta_min}, delta_max={delta_max}")
    print(f"            abs_min={abs_min}, abs_max={abs_max}")
    print(f"Result: {result[0]} records")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: Simpler approach - just use negative delta BETWEEN
print("\n3. SIMPLEST APPROACH - Just use BETWEEN (since all deltas are negative)")
print("-" * 80)
try:
    cur.execute("""
        SELECT COUNT(*)
        FROM stock_premiums
        WHERE dte BETWEEN 5 AND 9
          AND delta BETWEEN %s AND %s
    """, (delta_min, delta_max))
    result = cur.fetchone()
    print(f"Result: {result[0]} records")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("The ABS(%s) syntax in SQL is non-standard and problematic.")
print("PostgreSQL may accept it, but it's not doing what we expect.")
print("\nBetter approaches:")
print("1. Calculate ABS values in Python before passing to SQL")
print("2. Or simplify to just: delta BETWEEN %s AND %s")
print("   (This works fine if all deltas have the same sign)")
print("=" * 80)

cur.close()
conn.close()
