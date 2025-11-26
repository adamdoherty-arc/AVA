"""
Check if Streamlit cache is causing issues
"""
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("CHECKING PREMIUM_PCT DATABASE VALUES")
print("=" * 80)

conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

# Check if ANY premium_pct values exist
query = """
    SELECT
        COUNT(*) as total_rows,
        COUNT(premium_pct) as non_null_pct,
        SUM(CASE WHEN premium_pct IS NULL OR premium_pct = 0 THEN 1 ELSE 0 END) as null_or_zero_pct,
        MIN(premium_pct) as min_pct,
        MAX(premium_pct) as max_pct,
        AVG(premium_pct) as avg_pct
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
      AND delta BETWEEN -0.4 AND -0.2
"""

cur.execute(query)
result = cur.fetchone()

print(f"\nDelta -0.4 to -0.2, DTE 5-9:")
print(f"  Total rows: {result[0]}")
print(f"  Non-null premium_pct: {result[1]}")
print(f"  NULL or 0 premium_pct: {result[2]}")
print(f"  Min premium_pct: {result[3]}")
print(f"  Max premium_pct: {result[4]}")
print(f"  Avg premium_pct: {result[5]}")

# Get sample rows with and without premium_pct
print("\n[1] Sample rows WITH premium_pct:")
query = """
    SELECT symbol, strike_price, premium, premium_pct, annual_return
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
      AND delta BETWEEN -0.4 AND -0.2
      AND premium_pct IS NOT NULL
      AND premium_pct > 0
    LIMIT 5
"""
cur.execute(query)
results = cur.fetchall()
for row in results:
    print(f"  {row[0]:>6}: strike=${row[1]:>8.2f}, premium=${row[2]:>8.2f}, pct={row[3]}, annual={row[4]}%")

print("\n[2] Sample rows WITHOUT premium_pct:")
query = """
    SELECT symbol, strike_price, premium, premium_pct, annual_return
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
      AND delta BETWEEN -0.4 AND -0.2
      AND (premium_pct IS NULL OR premium_pct = 0)
    LIMIT 5
"""
cur.execute(query)
results = cur.fetchall()
for row in results:
    pct_val = row[3] if row[3] is not None else "NULL"
    print(f"  {row[0]:>6}: strike=${row[1]:>8.2f}, premium=${row[2]:>8.2f}, pct={pct_val}, annual={row[4]}%")

# Check if calculation would be correct
print("\n[3] Manual calculation check:")
query = """
    SELECT
        symbol,
        strike_price,
        premium,
        premium_pct as db_pct,
        (premium / strike_price * 100) as calc_pct,
        annual_return as db_annual,
        ((premium / strike_price * 100) * 365.0 / 7) as calc_annual
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
      AND delta BETWEEN -0.4 AND -0.2
    ORDER BY premium DESC
    LIMIT 5
"""
cur.execute(query)
results = cur.fetchall()
print(f"  {'Symbol':>6} {'Strike':>10} {'Premium':>10} {'DB_Pct':>8} {'Calc_Pct':>8} {'DB_Ann':>8} {'Calc_Ann':>9}")
for row in results:
    db_pct = row[3] if row[3] is not None else 0
    db_ann = row[5] if row[5] is not None else 0
    print(f"  {row[0]:>6} ${row[1]:>9.2f} ${row[2]:>9.2f} {db_pct:>8.2f} {row[4]:>8.2f} {db_ann:>8.1f} {row[6]:>9.1f}")

cur.close()
conn.close()

print("\n" + "=" * 80)
