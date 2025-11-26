from src.tradingview_db_manager import TradingViewDBManager

tv = TradingViewDBManager()
conn = tv.get_connection()
cur = conn.cursor()

print('\n' + '=' * 80)
print('STOCKS TABLE ANALYSIS')
print('=' * 80)

# Total count
cur.execute('SELECT COUNT(*) FROM stocks')
total = cur.fetchone()[0]
print(f'\nTotal stocks: {total:,}')

# Count by asset type
cur.execute("""
    SELECT asset_type, COUNT(*) as count
    FROM stocks
    GROUP BY asset_type
    ORDER BY count DESC
""")
print('\nBreakdown by asset type:')
for row in cur.fetchall():
    print(f'  {row[0]:15} {row[1]:>6,} stocks')

# Count by sector
cur.execute("""
    SELECT sector, COUNT(*) as count
    FROM stocks
    WHERE sector IS NOT NULL
    GROUP BY sector
    ORDER BY count DESC
    LIMIT 15
""")
print('\nTop 15 sectors:')
for row in cur.fetchall():
    print(f'  {row[0]:40} {row[1]:>6,} stocks')

# Check how many have options data
cur.execute("""
    SELECT COUNT(DISTINCT s.ticker)
    FROM stocks s
    WHERE EXISTS (
        SELECT 1 FROM stock_premiums sp
        WHERE sp.symbol = s.ticker
    )
""")
with_options = cur.fetchone()[0]
print(f'\n{with_options:,} of {total:,} stocks have options data in stock_premiums table')
print(f'That means {total - with_options:,} stocks could potentially have options synced!')

# Show sample tickers
cur.execute("""
    SELECT ticker, name, sector, asset_type
    FROM stocks
    WHERE asset_type = 'STOCK'
    ORDER BY ticker
    LIMIT 30
""")
print('\nSample stock tickers (first 30):')
print(f"{'TICKER':<8} {'NAME':<50} {'SECTOR':<30}")
print('-' * 90)
for row in cur.fetchall():
    print(f'{row[0]:<8} {row[1][:48]:<50} {(row[2] or "N/A")[:28]:<30}')

cur.close()
conn.close()
