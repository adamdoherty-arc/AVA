"""
Check which games have Kalshi odds and which don't
"""
import psycopg2
from datetime import datetime

# Connect to database
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="magnus",
    user="postgres",
    password="postgres123"
)

cursor = conn.cursor()

print("=" * 80)
print("GAME CARDS - KALSHI ODDS STATUS CHECK")
print("=" * 80)

# Check ESPN games by sport
print("\n1. ESPN GAMES BY SPORT (Today and Future):")
print("-" * 80)
cursor.execute("""
    SELECT
        sport,
        COUNT(*) as total_games,
        COUNT(CASE WHEN game_date::date = CURRENT_DATE THEN 1 END) as today_games,
        COUNT(CASE WHEN game_date::date > CURRENT_DATE THEN 1 END) as future_games
    FROM espn_live_games
    WHERE game_date >= CURRENT_DATE
    GROUP BY sport
    ORDER BY sport
""")

for row in cursor.fetchall():
    sport, total, today, future = row
    print(f"{sport:10} | Total: {total:3} | Today: {today:3} | Future: {future:3}")

# Check games with Kalshi markets
print("\n\n2. GAMES WITH KALSHI MARKETS:")
print("-" * 80)
cursor.execute("""
    SELECT
        e.sport,
        COUNT(DISTINCT e.game_id) as games_with_kalshi
    FROM espn_live_games e
    INNER JOIN kalshi_markets k ON (
        k.sport = e.sport
        AND k.is_active = true
    )
    WHERE e.game_date >= CURRENT_DATE
    GROUP BY e.sport
    ORDER BY e.sport
""")

kalshi_by_sport = {}
for row in cursor.fetchall():
    sport, count = row
    kalshi_by_sport[sport] = count
    print(f"{sport:10} | Games with Kalshi: {count}")

# Check games WITHOUT Kalshi markets
print("\n\n3. GAMES WITHOUT KALSHI MARKETS:")
print("-" * 80)
cursor.execute("""
    SELECT
        sport,
        COUNT(*) as games_without_kalshi
    FROM espn_live_games e
    WHERE e.game_date >= CURRENT_DATE
    AND NOT EXISTS (
        SELECT 1 FROM kalshi_markets k
        WHERE k.sport = e.sport
        AND k.is_active = true
        AND (
            k.home_team = e.home_team
            OR k.away_team = e.away_team
            OR k.home_team ILIKE '%' || e.home_team || '%'
            OR k.away_team ILIKE '%' || e.away_team || '%'
        )
    )
    GROUP BY sport
    ORDER BY sport
""")

print("\nSport      | Games Missing Kalshi")
for row in cursor.fetchall():
    sport, count = row
    print(f"{sport:10} | {count:3} games")

# List specific NCAA games without Kalshi
print("\n\n4. NCAA GAMES WITHOUT KALSHI ODDS (Sample - First 20):")
print("-" * 80)
cursor.execute("""
    SELECT
        away_team || ' @ ' || home_team as matchup,
        TO_CHAR(game_date, 'Mon DD HH24:MI') as game_time,
        status
    FROM espn_live_games
    WHERE sport = 'CFB'
    AND game_date >= CURRENT_DATE
    AND NOT EXISTS (
        SELECT 1 FROM kalshi_markets k
        WHERE k.sport = 'CFB'
        AND k.is_active = true
        AND (
            k.home_team ILIKE '%' || espn_live_games.home_team || '%'
            OR k.away_team ILIKE '%' || espn_live_games.away_team || '%'
        )
    )
    ORDER BY game_date
    LIMIT 20
""")

for i, row in enumerate(cursor.fetchall(), 1):
    matchup, game_time, status = row
    print(f"{i:2}. {matchup:50} | {game_time} | {status}")

# List specific NFL games without Kalshi
print("\n\n5. NFL GAMES WITHOUT KALSHI ODDS:")
print("-" * 80)
cursor.execute("""
    SELECT
        away_team || ' @ ' || home_team as matchup,
        TO_CHAR(game_date, 'Mon DD HH24:MI') as game_time,
        status
    FROM espn_live_games
    WHERE sport = 'NFL'
    AND game_date >= CURRENT_DATE
    AND NOT EXISTS (
        SELECT 1 FROM kalshi_markets k
        WHERE k.sport = 'NFL'
        AND k.is_active = true
        AND (
            k.home_team ILIKE '%' || espn_live_games.home_team || '%'
            OR k.away_team ILIKE '%' || espn_live_games.away_team || '%'
        )
    )
    ORDER BY game_date
""")

nfl_missing = cursor.fetchall()
if nfl_missing:
    for i, row in enumerate(nfl_missing, 1):
        matchup, game_time, status = row
        print(f"{i:2}. {matchup:50} | {game_time} | {status}")
else:
    print("All NFL games have Kalshi odds!")

# Check Kalshi markets in database
print("\n\n6. KALSHI MARKETS IN DATABASE:")
print("-" * 80)
cursor.execute("""
    SELECT
        sport,
        COUNT(*) as total_markets,
        COUNT(CASE WHEN is_active THEN 1 END) as active_markets,
        COUNT(CASE WHEN close_time >= NOW() THEN 1 END) as open_markets
    FROM kalshi_markets
    GROUP BY sport
    ORDER BY sport
""")

print("\nSport      | Total | Active | Open")
for row in cursor.fetchall():
    sport, total, active, open_count = row
    print(f"{sport:10} | {total:5} | {active:6} | {open_count:4}")

# Show sample of Kalshi markets
print("\n\n7. SAMPLE KALSHI MARKETS (First 10 active):")
print("-" * 80)
cursor.execute("""
    SELECT
        sport,
        ticker,
        home_team || ' vs ' || away_team as matchup,
        yes_price,
        no_price
    FROM kalshi_markets
    WHERE is_active = true
    AND close_time >= NOW()
    ORDER BY sport, close_time
    LIMIT 10
""")

for row in cursor.fetchall():
    sport, ticker, matchup, yes_price, no_price = row
    print(f"{sport:4} | {ticker:20} | {matchup:40} | Yes: {yes_price}¢ No: {no_price}¢")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Calculate overall stats
cursor.execute("""
    SELECT
        COUNT(*) as total_upcoming_games,
        COUNT(DISTINCT CASE WHEN sport = 'NFL' THEN game_id END) as nfl_games,
        COUNT(DISTINCT CASE WHEN sport = 'CFB' THEN game_id END) as ncaa_games,
        COUNT(DISTINCT CASE WHEN sport = 'NBA' THEN game_id END) as nba_games
    FROM espn_live_games
    WHERE game_date >= CURRENT_DATE
""")

total, nfl, ncaa, nba = cursor.fetchone()

print(f"""
Total Upcoming Games: {total}
  - NFL:  {nfl}
  - NCAA: {ncaa}
  - NBA:  {nba}

Kalshi Coverage:
  - NFL:  {kalshi_by_sport.get('NFL', 0)} games with odds
  - NCAA: {kalshi_by_sport.get('CFB', 0)} games with odds
  - NBA:  {kalshi_by_sport.get('NBA', 0)} games with odds

NOTE: Many NCAA games don't have Kalshi markets because Kalshi focuses on
      major conference games and top 25 matchups.
""")

cursor.close()
conn.close()

print("=" * 80)
