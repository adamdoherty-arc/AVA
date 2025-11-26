"""
Check premium_pct calculations
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

print("=" * 80)
print("PREMIUM_PCT ANALYSIS")
print("=" * 80)

query = """
    SELECT
        symbol,
        strike_price,
        premium,
        dte,
        premium_pct,
        annual_return,
        (premium / strike_price * 100) as calculated_pct,
        (premium_pct * (365.0 / dte)) as calculated_annual
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
    ORDER BY premium DESC
    LIMIT 10
"""

cur.execute(query)
results = cur.fetchall()

print(f"\n{'Symbol':<8} {'Strike':>10} {'Premium':>10} {'DTE':>4} {'DB_Pct%':>8} {'DB_Ann%':>8} {'Calc_Pct%':>10} {'Calc_Ann%':>10}")
print("-" * 80)
for row in results:
    db_pct = row[4] if row[4] is not None else 0
    db_ann = row[5] if row[5] is not None else 0
    calc_pct = row[6] if row[6] is not None else 0
    calc_ann = row[7] if row[7] is not None else 0
    print(f"{row[0]:<8} ${row[1]:>9.2f} ${row[2]:>9.2f} {row[3]:>4} {db_pct:>8.2f} {db_ann:>8.1f} {calc_pct:>10.2f} {calc_ann:>10.1f}")

print("\nIssue Identified:")
print("If premium_pct is stored as a fraction (0-1) instead of percentage (0-100),")
print("then multiplying by 365/dte would give tiny values.")
print("\nCheck: Is premium_pct stored as 0.02 (fraction) or 2.0 (percentage)?")

cur.close()
conn.close()
