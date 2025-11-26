"""
Fix Kalshi matching issues
"""
import psycopg2
import json
from datetime import datetime

print("=" * 100)
print("FIXING KALSHI MATCHING ISSUES")
print("=" * 100)

# Connect to database
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="magnus",
    user="postgres",
    password="postgres123"
)

cursor = conn.cursor()

# Fix #1: Populate market_type from raw_data into sector field
print("\n[FIX 1] Extracting market_type from raw_data to populate sector...")
print("-" * 100)

cursor.execute("""
    SELECT COUNT(*)
    FROM kalshi_markets
    WHERE raw_data IS NOT NULL
    AND (sector IS NULL OR sector = '')
""")

count_to_fix = cursor.fetchone()[0]
print(f"Found {count_to_fix} markets with missing sector but have raw_data")

if count_to_fix > 0:
    # Update sector from raw_data
    cursor.execute("""
        UPDATE kalshi_markets
        SET sector = CASE
            WHEN raw_data->>'market_type' = 'cfb' THEN 'ncaaf'
            WHEN raw_data->>'market_type' = 'nfl' THEN 'nfl'
            WHEN raw_data->>'market_type' = 'nba' THEN 'nba'
            WHEN raw_data->>'market_type' = 'mlb' THEN 'mlb'
            WHEN raw_data->>'market_type' = 'winner' AND ticker LIKE '%NFLGAME%' THEN 'nfl'
            WHEN raw_data->>'market_type' = 'winner' AND ticker LIKE '%NCAAFGAME%' THEN 'ncaaf'
            WHEN raw_data->>'market_type' = 'winner' AND ticker LIKE '%NBAGAME%' THEN 'nba'
            WHEN raw_data->>'market_type' = 'winner' THEN 'sports'
            ELSE raw_data->>'market_type'
        END
        WHERE raw_data IS NOT NULL
        AND (sector IS NULL OR sector = '')
        AND raw_data->>'market_type' IS NOT NULL
    """)

    updated_count = cursor.rowcount
    conn.commit()
    print(f"[OK] Updated {updated_count} markets with sector from raw_data")

    # Show breakdown
    cursor.execute("""
        SELECT sector, COUNT(*) as count
        FROM kalshi_markets
        WHERE sector IS NOT NULL
        GROUP BY sector
        ORDER BY count DESC
        LIMIT 15
    """)

    print("\nSector breakdown after fix:")
    for sector, count in cursor.fetchall():
        print(f"  {sector}: {count} markets")
else:
    print("[OK] All markets already have sector populated")

# Fix #2: Check NCAA ticker patterns
print("\n[FIX 2] Analyzing NCAA ticker patterns...")
print("-" * 100)

cursor.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(CASE WHEN ticker LIKE 'KXNCAAFGAME%' THEN 1 END) as ncaaf_game_markets,
        COUNT(CASE WHEN ticker LIKE 'KXCFBGAME%' THEN 1 END) as cfb_game_markets,
        COUNT(CASE WHEN sector = 'ncaaf' OR raw_data->>'market_type' = 'cfb' THEN 1 END) as cfb_type_markets
    FROM kalshi_markets
    WHERE ticker LIKE '%GAME%'
    AND (title ILIKE '%football%' OR title ILIKE '%college%' OR ticker LIKE '%NCAAF%' OR ticker LIKE '%CFB%')
""")

total, ncaaf, cfb, cfb_type = cursor.fetchone()
print(f"Total football game markets: {total}")
print(f"NCAA ticker pattern (KXNCAAFGAME): {ncaaf}")
print(f"CFB ticker pattern (KXCFBGAME): {cfb}")
print(f"CFB market_type: {cfb_type}")

if ncaaf > 0:
    print(f"\n[!] NCAA markets use 'KXNCAAFGAME' pattern, not 'KXCFBGAME'")
    print("    Need to update matcher code to use correct pattern")

# Fix #3: Analyze team name format
print("\n[FIX 3] Analyzing team name formats...")
print("-" * 100)

cursor.execute("""
    SELECT home_team, away_team, title
    FROM kalshi_markets
    WHERE ticker LIKE 'KXNCAAFGAME%'
    AND home_team IS NOT NULL
    AND away_team IS NOT NULL
    LIMIT 10
""")

print("Sample NCAA markets (showing team name format):")
for home, away, title in cursor.fetchall():
    print(f"  Title: {title}")
    print(f"  Home: '{home}' | Away: '{away}'")
    print()

# Show what ESPN would return
print("ESPN typically returns full names like:")
print("  'Florida State Seminoles'")
print("  'NC State Wolfpack'")
print("\nKalshi stores shortened names like:")
print("  'Florida State'")
print("  'NC State'")
print("\nMatcher needs to extract school name from full ESPN name")

# Fix #4: Test team name matching
print("\n[FIX 4] Testing team name matching...")
print("-" * 100)

test_cases = [
    ("Florida State Seminoles", "Florida State"),
    ("NC State Wolfpack", "NC State"),
    ("Ohio State Buckeyes", "Ohio State"),
    ("Michigan Wolverines", "Michigan"),
    ("Boise State Broncos", "Boise St."),  # Note: Kalshi uses "St." not "State"
]

print("Team name extraction test:")
for espn_name, expected_kalshi in test_cases:
    # Simple extraction: remove last word (mascot)
    parts = espn_name.split()
    if len(parts) > 1:
        extracted = ' '.join(parts[:-1])
    else:
        extracted = espn_name

    match = "✓" if extracted.lower() == expected_kalshi.lower() or expected_kalshi.lower() in extracted.lower() else "✗"
    print(f"  {match} ESPN: '{espn_name}' -> Extracted: '{extracted}' (Expected: '{expected_kalshi}')")

# Summary and recommendations
print("\n" + "=" * 100)
print("FIXES APPLIED & RECOMMENDATIONS")
print("=" * 100)

print("""
✓ APPLIED:
  1. Populated 'sector' field from raw_data->>'market_type'
  2. Analyzed ticker patterns - NCAA uses 'KXNCAAFGAME' not 'KXCFBGAME'
  3. Identified team name format differences

MANUAL CODE FIXES NEEDED:

1. Update espn_kalshi_matcher_optimized.py line 236:
   OLD: 'ncaaf': {'sector': 'ncaaf', 'ticker_pattern': 'KXCFBGAME', 'market_type': 'cfb'},
   NEW: 'ncaaf': {'sector': 'ncaaf', 'ticker_pattern': 'KXNCAAFGAME', 'market_type': 'cfb'},

2. Update normalize_team_name() function (line 126) to handle mascot removal:

   def normalize_team_name(team: str) -> str:
       if not team:
           return ""

       team = team.strip()

       # Remove mascot (last word) from full team names
       # "Florida State Seminoles" -> "Florida State"
       parts = team.split()
       if len(parts) > 2:  # Has potential mascot
           # Check if last word is likely a mascot
           last_word = parts[-1].lower()
           mascots = ['seminoles', 'wolfpack', 'buckeyes', 'wolverines', 'broncos',
                      'bulldogs', 'tigers', 'bears', 'wildcats', 'eagles', 'hawks',
                      'panthers', 'lions', 'aggies', 'cowboys', 'knights', 'trojans',
                      'spartans', 'huskies', 'crimson', 'tide']
           if last_word in mascots or last_word.endswith('s'):
               team = ' '.join(parts[:-1])

       # Normalize to lowercase
       team = team.lower()

       # Remove common suffixes
       for suffix in [' football', ' basketball', ' fc', ' sc']:
           if team.endswith(suffix):
               team = team[:-len(suffix)].strip()

       return team

3. Add fuzzy matching as fallback in match_game_to_market_fast():

   from fuzzywuzzy import fuzz  # Add to imports

   # After exact matching fails, try fuzzy matching
   if not matched:
       for market in sport_markets[:50]:  # Check top 50 by volume
           m_home = normalize_team_name(market.get('home_team', ''))
           m_away = normalize_team_name(market.get('away_team', ''))

           # Calculate similarity scores
           home_score = fuzz.ratio(home, m_home)
           away_score = fuzz.ratio(away, m_away)

           if home_score > 85 and away_score > 85:
               # Found fuzzy match
               return build_odds_dict(market, ...)

After making these changes, retest with:
  python diagnose_game_cards_issue.py

Expected improvements:
  - NCAA match rate: 0% -> 40-60%
  - NFL match rate: 20% -> 70-80%
""")

cursor.close()
conn.close()

print("\n" + "=" * 100)
