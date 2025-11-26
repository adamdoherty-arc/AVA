from src.tradingview_db_manager import TradingViewDBManager

tv = TradingViewDBManager()
conn = tv.get_connection()
cur = conn.cursor()

# Check total unique symbols
cur.execute("SELECT COUNT(DISTINCT symbol) FROM stock_premiums")
total_symbols = cur.fetchone()[0]
print(f'Total unique symbols in stock_premiums: {total_symbols}')

# Check symbols per DTE
cur.execute("""
    SELECT dte, COUNT(DISTINCT symbol) as symbol_count
    FROM stock_premiums
    GROUP BY dte
    ORDER BY dte
""")
print('\nSymbols per DTE:')
for row in cur.fetchall():
    print(f'  DTE {row[0]:3}: {row[1]:3} symbols')

# Check which symbols from NVDA watchlist are missing 30-day data
cur.execute("""
    SELECT s.symbol
    FROM tv_watchlist_symbols ws
    JOIN tv_symbols_api s ON ws.symbol_id = s.id
    WHERE ws.watchlist_id = (SELECT id FROM tv_watchlists WHERE name = 'NVDA')
    AND s.symbol NOT IN (
        SELECT DISTINCT symbol
        FROM stock_premiums
        WHERE dte BETWEEN 28 AND 32
        AND ABS(delta) BETWEEN 0.28 AND 0.32
    )
    ORDER BY s.symbol
""")
missing_symbols = [row[0] for row in cur.fetchall()]
print(f'\n{len(missing_symbols)} symbols from NVDA watchlist missing 30-day options:')
print(', '.join(missing_symbols[:20]))
if len(missing_symbols) > 20:
    print(f'... and {len(missing_symbols) - 20} more')

cur.close()
conn.close()
