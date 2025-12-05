#!/usr/bin/env python3
"""Quick check of premium status"""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv
load_dotenv()

async def check():
    conn = await asyncpg.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD')
    )

    # Check NVDA premiums
    nvda = await conn.fetch(
        "SELECT symbol, strike, dte, premium_pct, monthly_return, last_updated "
        "FROM premium_opportunities WHERE symbol = $1 ORDER BY monthly_return DESC",
        'NVDA'
    )
    print('=== NVDA Premiums ===')
    if nvda:
        for r in nvda:
            print(f"Strike ${r['strike']} | DTE {r['dte']} | Prem {r['premium_pct']:.1f}% | Monthly {r['monthly_return']:.1f}%")
    else:
        print('No NVDA premiums found!')

    # Check TradingView watchlist
    wl = await conn.fetchrow(
        "SELECT name, symbols FROM scanner_watchlists WHERE source = $1 LIMIT 1",
        'tradingview'
    )
    print('\n=== TradingView Watchlist ===')
    if wl:
        s = wl['symbols'] or []
        print(f"Name: {wl['name']}")
        print(f"Total symbols: {len(s)}")
        print(f"First 15: {s[:15]}")
        print(f"NVDA in list: {'NVDA' in s}")
    else:
        print('No TradingView watchlist found')

    # Check scheduled tasks status
    print('\n=== Scan Stats ===')
    total = await conn.fetchval('SELECT COUNT(*) FROM premium_opportunities')
    distinct_symbols = await conn.fetchval('SELECT COUNT(DISTINCT symbol) FROM premium_opportunities')
    newest = await conn.fetchval('SELECT MAX(last_updated) FROM premium_opportunities')
    print(f"Total opportunities: {total}")
    print(f"Distinct symbols: {distinct_symbols}")
    print(f"Last updated: {newest}")

    await conn.close()

if __name__ == '__main__':
    asyncio.run(check())
