from src.tradingview_db_manager import TradingViewDBManager

tv = TradingViewDBManager()
conn = tv.get_connection()
cur = conn.cursor()

# Check all tables
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
print('Available tables:')
for row in cur.fetchall():
    print(f'  - {row[0]}')

# Check stock_premiums data
cur.execute('SELECT COUNT(*) FROM stock_premiums')
count = cur.fetchone()[0]
print(f'\nTotal rows in stock_premiums: {count}')

# Get sample data
cur.execute("""
    SELECT sp.symbol, sd.current_price, sp.dte, sp.premium, sp.delta, sp.monthly_return, sp.strike_price, sp.bid, sp.ask
    FROM stock_premiums sp
    LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
    WHERE sp.dte BETWEEN 28 AND 32
    AND ABS(sp.delta) BETWEEN 0.28 AND 0.32
    AND sp.premium IS NOT NULL
    ORDER BY sp.monthly_return DESC
    LIMIT 10
""")
print('\nSample 30-day options data (~0.3 delta):')
print('Symbol | Stock$  | DTE | Premium | Delta  | Monthly% | Strike  | Bid    | Ask')
for row in cur.fetchall():
    stock_price = row[1] if row[1] else 0
    bid = row[7] if row[7] else 0
    ask = row[8] if row[8] else 0
    print(f'{row[0]:6} | ${stock_price:6.2f} | {row[2]:3} | ${row[3]:6.2f} | {row[4]:6.3f} | {row[5]:6.2f}% | ${row[6]:6.2f} | ${bid:5.2f} | ${ask:5.2f}')

cur.close()
conn.close()
