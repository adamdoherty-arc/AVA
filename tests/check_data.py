import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor()

# Check total data
cur.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM stock_premiums WHERE strike_type IN ('30_delta', '5%_OTM')")
total, symbols = cur.fetchone()
print(f"Total options: {total}")
print(f"Unique symbols: {symbols}\n")

# Show sample data
cur.execute("""
    SELECT symbol, strike_price, premium, monthly_return, dte, delta, expiration_date
    FROM stock_premiums
    WHERE strike_type IN ('30_delta', '5%_OTM')
    ORDER BY monthly_return DESC
    LIMIT 20
""")

print("Top 20 Premium Opportunities:")
print("="*100)
print(f"{'Symbol':<8} {'Strike':<10} {'Premium':<10} {'Monthly%':<10} {'DTE':<6} {'Delta':<8} {'Expiration'}")
print("="*100)

for row in cur.fetchall():
    symbol, strike, premium, monthly, dte, delta, exp_date = row
    print(f"{symbol:<8} ${strike:<9.2f} ${premium:<9.0f} {monthly:<9.2f}% {dte:<6} {delta:<8.3f} {exp_date}")

cur.close()
conn.close()
