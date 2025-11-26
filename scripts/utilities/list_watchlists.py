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

# Get all watchlists with counts
cur.execute("""
    SELECT w.name, COUNT(DISTINCT s.symbol) as symbol_count
    FROM tv_watchlists_api w
    JOIN tv_symbols_api s ON w.watchlist_id = s.watchlist_id
    WHERE s.symbol NOT LIKE '%USD%'
      AND s.symbol NOT LIKE '%BTC%'
      AND s.symbol NOT LIKE '%.D'
      AND s.symbol NOT LIKE '%WETH%'
    GROUP BY w.name
    ORDER BY w.name
""")

print("Available Watchlists (Stock Symbols Only):")
print("="*60)
for row in cur.fetchall():
    print(f"{row[0]}: {row[1]} symbols")

cur.close()
conn.close()
