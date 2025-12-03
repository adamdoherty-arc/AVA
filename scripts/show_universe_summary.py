"""Show stocks and ETF universe summary"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv(override=True)

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'postgres'),
    database=os.getenv('DB_NAME', 'magnus')
)
cur = conn.cursor()

# Count stocks
cur.execute('SELECT COUNT(*) FROM stocks_universe')
stocks_count = cur.fetchone()[0]

# Count ETFs
cur.execute('SELECT COUNT(*) FROM etfs_universe')
etfs_count = cur.fetchone()[0]

# Get sector breakdown for stocks
cur.execute("""
    SELECT sector, COUNT(*) as count
    FROM stocks_universe
    WHERE sector IS NOT NULL AND sector <> ''
    GROUP BY sector
    ORDER BY count DESC
""")
sectors = cur.fetchall()

# Get sample stocks by market cap
cur.execute("""
    SELECT symbol, company_name, market_cap, sector
    FROM stocks_universe
    WHERE market_cap IS NOT NULL
    ORDER BY market_cap DESC
    LIMIT 15
""")
top_stocks = cur.fetchall()

# Get ETF categories
cur.execute("""
    SELECT category, COUNT(*) as count
    FROM etfs_universe
    WHERE category IS NOT NULL AND category <> ''
    GROUP BY category
    ORDER BY count DESC
    LIMIT 15
""")
categories = cur.fetchall()

# Get sample ETFs by AUM
cur.execute("""
    SELECT symbol, fund_name, total_assets
    FROM etfs_universe
    WHERE total_assets IS NOT NULL
    ORDER BY total_assets DESC
    LIMIT 15
""")
top_etfs = cur.fetchall()

print('='*70)
print(f'TOTAL UNIVERSE: {stocks_count} STOCKS | {etfs_count} ETFs')
print('='*70)

print('\nSTOCKS BY SECTOR:')
for sector, count in sectors[:12]:
    print(f'  {sector}: {count}')

print('\nTOP 15 STOCKS BY MARKET CAP:')
for symbol, name, mcap, sector in top_stocks:
    mcap_b = mcap / 1e9 if mcap else 0
    name_short = name[:30] if name else symbol
    print(f'  {symbol:6} ${mcap_b:>7.0f}B  {name_short}')

print('\nETF CATEGORIES:')
for cat, count in categories:
    print(f'  {cat}: {count}')

print('\nTOP 15 ETFs BY AUM:')
for symbol, name, aum in top_etfs:
    aum_b = aum / 1e9 if aum else 0
    name_short = name[:35] if name else symbol
    print(f'  {symbol:6} ${aum_b:>7.0f}B  {name_short}')

conn.close()
