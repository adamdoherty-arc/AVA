import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host='localhost',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor()

# Check stock_premiums
cur.execute("""
    SELECT symbol, strike_price, premium, monthly_return, dte, strike_type
    FROM stock_premiums
    WHERE symbol IN ('AAPL', 'COIN', 'URA')
    ORDER BY symbol, dte
""")

print("Current options data in database:")
print("="*80)
rows = cur.fetchall()
if rows:
    for row in rows:
        print(f"{row[0]}: Strike=${row[1]:.2f}, Premium=${row[2]:.2f}, Monthly={row[3]:.2f}%, DTE={row[4]}, Type={row[5]}")
else:
    print("NO DATA FOUND - Database is empty!")
    print("\nYou need to run: python src/watchlist_sync_service.py Stocks")

cur.close()
conn.close()
