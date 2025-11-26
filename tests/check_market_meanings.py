"""Check what YES/NO actually means in Kalshi markets"""
from src.kalshi_db_manager import KalshiDBManager

kalshi_db = KalshiDBManager()
markets = kalshi_db.get_active_markets(market_type='nfl')

print("Checking first 10 NFL markets to see what YES/NO means:\n")
for i, market in enumerate(markets[:10]):
    title = market.get('title', '')
    ticker = market.get('ticker', '')
    away_team = market.get('away_team', '')
    home_team = market.get('home_team', '')
    yes_price = market.get('yes_price', 0)
    no_price = market.get('no_price', 0)

    print(f"{i+1}. {ticker}")
    print(f"   Title: {title}")
    print(f"   Teams: {away_team} @ {home_team}")
    print(f"   YES price: {yes_price:.2f} ({yes_price*100:.0f}%)")
    print(f"   NO price: {no_price:.2f} ({no_price*100:.0f}%)")
    print()

    if i == 4:  # Show 5 examples
        break
