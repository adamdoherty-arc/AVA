"""
Diagnose 7-Day Scanner Data Issues
Check what's actually in the stock_premiums table
"""
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("7-DAY SCANNER DATA DIAGNOSIS")
print("=" * 80)

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

# Check overall stats
print("\n[1] Overall Database Stats")
print("-" * 80)
query = """
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT symbol) as unique_symbols,
        MIN(dte) as min_dte,
        MAX(dte) as max_dte,
        MIN(delta) as min_delta,
        MAX(delta) as max_delta,
        AVG(delta) as avg_delta,
        MIN(annual_return) as min_annual_return,
        MAX(annual_return) as max_annual_return,
        AVG(annual_return) as avg_annual_return
    FROM stock_premiums
"""
cur.execute(query)
result = cur.fetchone()
print(f"Total rows: {result[0]}")
print(f"Unique symbols: {result[1]}")
print(f"DTE range: {result[2]} to {result[3]}")
print(f"Delta range: {result[4]:.3f} to {result[5]:.3f}")
print(f"Avg delta: {result[6]:.3f}")
print(f"Annual return range: {result[7]:.1f}% to {result[8]:.1f}%")
print(f"Avg annual return: {result[9]:.1f}%")

# Check 7-day DTE range (5-9 days)
print("\n[2] 7-Day DTE Range (5-9 days)")
print("-" * 80)
query = """
    SELECT
        COUNT(*) as total_opportunities,
        COUNT(DISTINCT symbol) as unique_symbols,
        MIN(delta) as min_delta,
        MAX(delta) as max_delta,
        AVG(delta) as avg_delta,
        MIN(annual_return) as min_annual_return,
        MAX(annual_return) as max_annual_return,
        AVG(annual_return) as avg_annual_return,
        MIN(premium) as min_premium,
        MAX(premium) as max_premium,
        AVG(premium) as avg_premium
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
"""
cur.execute(query)
result = cur.fetchone()
print(f"Total opportunities: {result[0]}")
print(f"Unique symbols: {result[1]}")
print(f"Delta range: {result[2]:.3f} to {result[3]:.3f}")
print(f"Avg delta: {result[4]:.3f}")
print(f"Annual return range: {result[5]:.1f}% to {result[6]:.1f}%")
print(f"Avg annual return: {result[7]:.1f}%")
print(f"Premium range: ${result[8]:.2f} to ${result[9]:.2f}")
print(f"Avg premium: ${result[10]:.2f}")

# Check delta distribution for 7-day DTE
print("\n[3] Delta Distribution for 7-Day DTE")
print("-" * 80)
query = """
    SELECT
        CASE
            WHEN delta < -0.5 THEN 'Below -0.5'
            WHEN delta BETWEEN -0.5 AND -0.4 THEN '-0.5 to -0.4'
            WHEN delta BETWEEN -0.4 AND -0.3 THEN '-0.4 to -0.3'
            WHEN delta BETWEEN -0.3 AND -0.2 THEN '-0.3 to -0.2'
            WHEN delta BETWEEN -0.2 AND -0.1 THEN '-0.2 to -0.1'
            WHEN delta BETWEEN -0.1 AND 0 THEN '-0.1 to 0'
            ELSE 'Above 0'
        END as delta_range,
        COUNT(*) as count
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
    GROUP BY delta_range
    ORDER BY delta_range
"""
cur.execute(query)
results = cur.fetchall()
for row in results:
    print(f"  {row[0]:>15}: {row[1]:>5} opportunities")

# Check with current filters (delta -0.4 to -0.2, min annual 30%)
print("\n[4] With Current Default Filters")
print("-" * 80)
print("Filters: Delta -0.4 to -0.2, Min Premium $0, Min Annual Return 30%")
query = """
    SELECT COUNT(*) as filtered_count
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
      AND delta BETWEEN -0.4 AND -0.2
      AND premium > 0
      AND (premium / (dte / 365.0)) / (SELECT AVG(strike_price) FROM stock_premiums WHERE dte BETWEEN 5 AND 9) * 100 >= 30
"""
cur.execute(query)
result = cur.fetchone()
print(f"Opportunities matching filters: {result[0]}")

# Simpler check - just delta filter
print("\n[5] With Just Delta Filter (-0.4 to -0.2)")
print("-" * 80)
query = """
    SELECT COUNT(*) as count
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
      AND delta BETWEEN -0.4 AND -0.2
"""
cur.execute(query)
result = cur.fetchone()
print(f"Opportunities with delta -0.4 to -0.2: {result[0]}")

# Sample some actual rows
print("\n[6] Sample Rows from 7-Day DTE")
print("-" * 80)
query = """
    SELECT symbol, strike_price, premium, dte, delta, annual_return, volume
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
    ORDER BY premium DESC
    LIMIT 10
"""
cur.execute(query)
results = cur.fetchall()
print(f"{'Symbol':<10} {'Strike':<10} {'Premium':<10} {'DTE':<5} {'Delta':<8} {'Annual%':<10} {'Volume':<10}")
print("-" * 80)
for row in results:
    print(f"{row[0]:<10} ${row[1]:<9.2f} ${row[2]:<9.2f} {row[3]:<5} {row[4]:<8.3f} {row[5]:<10.1f} {row[6] or 0:<10}")

# Check if annualized_52wk column exists
print("\n[7] Schema Check")
print("-" * 80)
query = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'stock_premiums'
    ORDER BY ordinal_position
"""
cur.execute(query)
results = cur.fetchall()
columns = [row[0] for row in results]
print(f"Columns in stock_premiums: {', '.join(columns)}")

if 'annualized_52wk' not in columns:
    print("\n⚠️  WARNING: 'annualized_52wk' column does NOT exist in table!")
    print("   This column is calculated in the code, not stored in DB")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
