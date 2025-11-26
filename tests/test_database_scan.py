from src.tradingview_db_manager import TradingViewDBManager

tv = TradingViewDBManager()
conn = tv.get_connection()
cur = conn.cursor()

query = """
    SELECT DISTINCT ON (sp.symbol)
        sp.symbol,
        sd.current_price as stock_price,
        sp.strike_price,
        sp.dte,
        sp.premium,
        sp.delta,
        sp.monthly_return,
        sp.implied_volatility as iv,
        sp.bid,
        sp.ask,
        sp.volume,
        sp.open_interest as oi,
        s.company_name,
        s.sector
    FROM stock_premiums sp
    LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
    LEFT JOIN stocks s ON sp.symbol = s.symbol
    WHERE sp.dte BETWEEN 29 AND 33
        AND ABS(sp.delta) BETWEEN 0.25 AND 0.40
        AND sp.premium >= 0
        AND (sd.current_price BETWEEN 0 AND 10000 OR sd.current_price IS NULL)
    ORDER BY sp.symbol, sp.monthly_return DESC
"""

cur.execute(query)
rows = cur.fetchall()

print(f'\nDatabase Scan Test Results:')
print(f'Found {len(rows)} stocks with 30-day options\n')
print(f"{'SYMBOL':<8} {'STOCK $':>9} {'STRIKE':>9} {'PREMIUM':>9} {'MONTHLY':>9} {'NAME':<35} {'SECTOR':<25}")
print('=' * 115)

for row in rows[:20]:
    symbol = row[0]
    stock_price = row[1] if row[1] else 0
    strike = row[2]
    premium = row[4]
    monthly = row[6]
    name = (row[12][:32] if row[12] else "N/A") + "..."
    sector = (row[13][:22] if row[13] else "N/A") + "..."

    print(f'{symbol:<8} ${stock_price:>8.2f} ${strike:>8.2f} ${premium:>8.2f} {monthly:>8.2f}% {name:<35} {sector:<25}')

print(f'\n... and {len(rows) - 20} more stocks')

cur.close()
conn.close()
