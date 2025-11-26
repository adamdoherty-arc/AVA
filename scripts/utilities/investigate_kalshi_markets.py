"""
Investigate actual Kalshi markets to understand matching issues
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

print("=" * 100)
print("KALSHI MARKETS INVESTIGATION")
print("=" * 100)

# Check sports-related markets
print("\n1. FOOTBALL MARKETS:")
print("-" * 100)
cursor.execute("""
    SELECT ticker, title, home_team, away_team, yes_price, no_price
    FROM kalshi_markets
    WHERE title ILIKE '%football%' OR title ILIKE '%nfl%' OR title ILIKE '%ncaa%' OR title ILIKE '%college%'
    LIMIT 20
""")

rows = cursor.fetchall()
if rows:
    for i, row in enumerate(rows, 1):
        ticker, title, home, away, yes_p, no_p = row
        print(f"{i}. {ticker}")
        print(f"   Title: {title}")
        print(f"   Home: {home} | Away: {away}")
        print(f"   Prices: Yes={yes_p}c No={no_p}c")
        print()
else:
    print("No football-related markets found")

# Check for team names
print("\n2. SAMPLE TEAM NAMES IN KALSHI:")
print("-" * 100)
cursor.execute("""
    SELECT DISTINCT home_team
    FROM kalshi_markets
    WHERE home_team IS NOT NULL
    AND home_team != ''
    LIMIT 30
""")

teams = [row[0] for row in cursor.fetchall() if row[0]]
if teams:
    print("Home teams found:")
    for team in teams[:30]:
        print(f"  - {team}")
else:
    print("No home_team values found!")

# Check market structure
print("\n3. MARKET DATA STRUCTURE:")
print("-" * 100)
cursor.execute("""
    SELECT
        ticker,
        title,
        market_type,
        sector,
        status,
        close_time
    FROM kalshi_markets
    LIMIT 5
""")

print("Sample market structure:")
for row in cursor.fetchall():
    ticker, title, mtype, sector, status, close_time = row
    print(f"\nTicker: {ticker}")
    print(f"Title: {title}")
    print(f"Type: {mtype}")
    print(f"Sector: {sector}")
    print(f"Status: {status}")
    print(f"Close: {close_time}")

# Search for specific teams from ESPN
print("\n4. SEARCHING FOR SPECIFIC ESPN TEAMS:")
print("-" * 100)

test_teams = [
    "Florida State",
    "NC State",
    "Buffalo Bills",
    "Houston Texans",
    "Georgia",
    "Ohio State"
]

for team in test_teams:
    cursor.execute("""
        SELECT COUNT(*),
               string_agg(DISTINCT ticker, ', ') as tickers
        FROM kalshi_markets
        WHERE title ILIKE %s OR home_team ILIKE %s OR away_team ILIKE %s
    """, (f'%{team}%', f'%{team}%', f'%{team}%'))

    count, tickers = cursor.fetchone()
    if count > 0:
        print(f"'{team}': {count} markets - {tickers[:100] if tickers else 'N/A'}")
    else:
        print(f"'{team}': NOT FOUND")

# Check if markets use different format
print("\n5. ALL MARKET TITLES (Sample):")
print("-" * 100)
cursor.execute("""
    SELECT title
    FROM kalshi_markets
    WHERE close_time >= NOW()
    ORDER BY RANDOM()
    LIMIT 30
""")

print("Random sample of active market titles:")
for i, row in enumerate(cursor.fetchall(), 1):
    print(f"{i}. {row[0]}")

# Check raw_data field
print("\n6. CHECKING raw_data FIELD:")
print("-" * 100)
cursor.execute("""
    SELECT ticker, raw_data::text
    FROM kalshi_markets
    WHERE raw_data IS NOT NULL
    LIMIT 3
""")

print("Sample raw_data contents:")
for ticker, raw in cursor.fetchall():
    print(f"\nTicker: {ticker}")
    print(f"Raw data (first 500 chars): {raw[:500] if raw else 'NULL'}")

cursor.close()
conn.close()

print("\n" + "=" * 100)
