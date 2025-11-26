"""Find markets that are about team winners, not spreads"""
from src.kalshi_db_manager import KalshiDBManager

kalshi_db = KalshiDBManager()
markets = kalshi_db.get_active_markets(market_type='nfl')

print(f"Total markets: {len(markets)}\n")

# Find markets with team names
markets_with_teams = [m for m in markets if m.get('away_team') and m.get('home_team')]
print(f"Markets with away_team and home_team: {len(markets_with_teams)}\n")

if markets_with_teams:
    print("First 5 markets with team names:")
    for i, market in enumerate(markets_with_teams[:5]):
        title = market.get('title', '')
        ticker = market.get('ticker', '')
        away_team = market.get('away_team', '')
        home_team = market.get('home_team', '')
        yes_price = market.get('yes_price', 0)
        no_price = market.get('no_price', 0)

        print(f"\n{i+1}. {ticker}")
        print(f"   Title: {title}")
        print(f"   Teams: {away_team} @ {home_team}")
        print(f"   YES: {yes_price*100:.0f}% | NO: {no_price*100:.0f}%")
else:
    # Check market titles for patterns
    print("\nLooking for 'to win' patterns in titles:")
    win_markets = [m for m in markets if 'to win' in m.get('title', '').lower() or 'wins' in m.get('title', '').lower()]
    print(f"Found {len(win_markets)} markets with 'win' in title")

    if win_markets:
        for i, market in enumerate(win_markets[:5]):
            print(f"\n{i+1}. {market.get('title')}")
            print(f"   YES: {market.get('yes_price', 0)*100:.0f}% | NO: {market.get('no_price', 0)*100:.0f}%")
