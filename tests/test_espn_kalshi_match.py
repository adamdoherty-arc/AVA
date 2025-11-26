"""Test ESPN to Kalshi matching"""
from src.espn_live_data import get_espn_client
from src.espn_kalshi_matcher import ESPNKalshiMatcher

# Get ESPN games
print("Fetching ESPN scoreboard...")
espn_client = get_espn_client()
espn_games = espn_client.get_scoreboard()
print(f"Found {len(espn_games)} ESPN games\n")

# Show first few games
print("ESPN Games:")
for i, game in enumerate(espn_games[:5], 1):
    print(f"{i}. {game['away_team']} @ {game['home_team']}")
    print(f"   Status: {game['status']}, Score: {game['away_score']}-{game['home_score']}")
    print(f"   Game time: {game.get('game_time')}")

print("\n" + "="*60)
print("Attempting to match with Kalshi markets...")
print("="*60 + "\n")

# Try matching
matcher = ESPNKalshiMatcher()
enriched_games = matcher.enrich_espn_games_with_kalshi(espn_games)

# Show results
matched = [g for g in enriched_games if g.get('kalshi_odds')]
unmatched = [g for g in enriched_games if not g.get('kalshi_odds')]

print(f"\nMatched: {len(matched)} games")
print(f"Unmatched: {len(unmatched)} games\n")

if matched:
    print("MATCHED GAMES:")
    for game in matched:
        odds = game['kalshi_odds']
        print(f"✅ {game['away_team']} @ {game['home_team']}")
        print(f"   Kalshi: {odds['ticker']}")
        print(f"   Away win: {odds['away_win_price']:.1%}, Home win: {odds['home_win_price']:.1%}")
        print()
else:
    print("❌ NO MATCHES FOUND!\n")
    print("First unmatched game details:")
    if unmatched:
        game = unmatched[0]
        print(f"  Away: {game['away_team']} ({game.get('away_abbr')})")
        print(f"  Home: {game['home_team']} ({game.get('home_abbr')})")
        print(f"  Game time: {game.get('game_time')}")

        # Try to manually find a matching market
        print("\n  Searching for matching Kalshi markets...")
        from src.kalshi_db_manager import KalshiDBManager
        db = KalshiDBManager()
        conn = db.get_connection()
        cur = conn.cursor()

        # Search for markets with these teams
        away_team = game['away_team']
        home_team = game['home_team']

        cur.execute("""
            SELECT ticker, title, close_time, yes_price, no_price
            FROM kalshi_markets
            WHERE (title ILIKE %s OR title ILIKE %s)
            AND market_type = 'nfl'
            ORDER BY close_time DESC
            LIMIT 5
        """, (f'%{away_team}%', f'%{home_team}%'))

        results = cur.fetchall()
        if results:
            print(f"\n  Found {len(results)} potential Kalshi markets:")
            for ticker, title, close_time, yes_price, no_price in results:
                print(f"    {ticker}")
                print(f"    {title}")
                print(f"    Close: {close_time}, Prices: {yes_price}/{no_price}")
        else:
            print(f"  No Kalshi markets found for {away_team} or {home_team}")

        cur.close()
        db.release_connection(conn)
