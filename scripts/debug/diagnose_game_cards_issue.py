"""
Comprehensive diagnostic for Game Cards - ESPN + Kalshi sync status
"""
import sys
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("GAME CARDS DIAGNOSTIC - ESPN & KALSHI SYNC STATUS")
print("=" * 80)
print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Test 1: ESPN NFL API
print("\n[TEST 1] ESPN NFL API")
print("-" * 80)
try:
    from src.espn_live_data import get_espn_client
    espn_nfl = get_espn_client()

    # Fetch current week
    nfl_games = espn_nfl.get_scoreboard()
    print(f"[OK] NFL API: Fetched {len(nfl_games)} games")

    if nfl_games:
        # Show sample
        game = nfl_games[0]
        print(f"   Sample: {game.get('away_team')} @ {game.get('home_team')}")
        print(f"   Status: {game.get('status')}")
        print(f"   Time: {game.get('game_time')}")
    else:
        print("   [WARN] No NFL games found (might be off-season or between weeks)")

except Exception as e:
    print(f"[ERROR] NFL API failed: {e}")

# Test 2: ESPN NCAA API
print("\n[TEST 2] ESPN NCAA API")
print("-" * 80)
try:
    from src.espn_ncaa_live_data import get_espn_ncaa_client
    espn_ncaa = get_espn_ncaa_client()

    # Fetch FBS games
    ncaa_games = espn_ncaa.get_scoreboard(group='80')  # 80 = FBS
    print(f"[OK] NCAA API: Fetched {len(ncaa_games)} FBS games")

    if ncaa_games:
        # Show sample
        game = ncaa_games[0]
        print(f"   Sample: {game.get('away_team')} @ {game.get('home_team')}")
        print(f"   Status: {game.get('status')}")
        print(f"   Time: {game.get('game_time')}")

        # Count by status
        upcoming = sum(1 for g in ncaa_games if not g.get('is_completed') and not g.get('is_live'))
        live = sum(1 for g in ncaa_games if g.get('is_live'))
        final = sum(1 for g in ncaa_games if g.get('is_completed'))

        print(f"   Status Breakdown:")
        print(f"     Upcoming: {upcoming}")
        print(f"     Live:     {live}")
        print(f"     Final:    {final}")
    else:
        print("   [WARN] No NCAA games found")

except Exception as e:
    print(f"[ERROR] NCAA API failed: {e}")

# Test 3: ESPN NBA API
print("\n[TEST 3] ESPN NBA API")
print("-" * 80)
try:
    from src.espn_nba_live_data import ESPNNBALiveData
    espn_nba = ESPNNBALiveData()

    # Fetch today's games
    today_str = datetime.now().strftime('%Y%m%d')
    nba_games = espn_nba.get_scoreboard(date=today_str)
    print(f"[OK] NBA API: Fetched {len(nba_games)} games for today")

    if nba_games:
        # Show sample
        game = nba_games[0]
        print(f"   Sample: {game.get('away_team')} @ {game.get('home_team')}")
        print(f"   Status: {game.get('status')}")
    else:
        print("   [INFO] No NBA games today")

except Exception as e:
    print(f"[ERROR] NBA API failed: {e}")

# Test 4: Database - Check tables exist
print("\n[TEST 4] Database Connection & Tables")
print("-" * 80)
try:
    import psycopg2
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="magnus",
        user="postgres",
        password="postgres123"
    )
    cursor = conn.cursor()

    print("[OK] Database connected")

    # Check key tables
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND (table_name LIKE '%game%' OR table_name LIKE '%kalshi%')
        ORDER BY table_name
    """)

    tables = [row[0] for row in cursor.fetchall()]
    print(f"[OK] Found {len(tables)} game/kalshi related tables")

    # Check if key tables have data
    key_tables = ['nfl_games', 'kalshi_markets', 'game_watchlist']
    for table in key_tables:
        if table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   {table}: {count} rows")
        else:
            print(f"   {table}: [MISSING]")

except Exception as e:
    print(f"[ERROR] Database check failed: {e}")

# Test 5: Kalshi Client
print("\n[TEST 5] Kalshi API Connection")
print("-" * 80)
try:
    from src.kalshi_db_manager import KalshiDBManager
    kalshi_db = KalshiDBManager()

    print("[OK] Kalshi DB Manager initialized")

    # Check markets in database
    conn = kalshi_db.get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN status = 'open' THEN 1 END) as open_markets,
            COUNT(CASE WHEN close_time >= NOW() THEN 1 END) as active_markets
        FROM kalshi_markets
    """)

    total, open_markets, active = cursor.fetchone()
    print(f"   Total markets: {total}")
    print(f"   Open markets: {open_markets}")
    print(f"   Active (not expired): {active}")

    # Check markets by category if possible
    cursor.execute("""
        SELECT sector, COUNT(*) as count
        FROM kalshi_markets
        WHERE close_time >= NOW()
        GROUP BY sector
        ORDER BY count DESC
        LIMIT 10
    """)

    sectors = cursor.fetchall()
    if sectors:
        print(f"\n   Top sectors with active markets:")
        for sector, count in sectors:
            print(f"     {sector}: {count} markets")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"[ERROR] Kalshi check failed: {e}")

# Test 6: ESPN-Kalshi Matching
print("\n[TEST 6] ESPN-Kalshi Matching System")
print("-" * 80)
try:
    from src.espn_kalshi_matcher_optimized import enrich_games_with_kalshi_odds_optimized

    print("[OK] Matcher module imported")

    # Try to match NCAA games if we have them
    if 'ncaa_games' in locals() and ncaa_games:
        enriched = enrich_games_with_kalshi_odds_optimized(ncaa_games[:5], sport='ncaaf')
        matched = sum(1 for g in enriched if g.get('kalshi_odds'))
        print(f"   NCAA Match Test: {matched}/5 games matched with Kalshi odds")

    # Try to match NFL games if we have them
    if 'nfl_games' in locals() and nfl_games:
        enriched = enrich_games_with_kalshi_odds_optimized(nfl_games[:5], sport='nfl')
        matched = sum(1 for g in enriched if g.get('kalshi_odds'))
        print(f"   NFL Match Test: {matched}/5 games matched with Kalshi odds")

except Exception as e:
    print(f"[ERROR] Matcher test failed: {e}")

# Test 7: Check Streamlit cache status
print("\n[TEST 7] Streamlit Integration Check")
print("-" * 80)
try:
    import streamlit as st
    print("[OK] Streamlit available (running in app context)")
except:
    print("[INFO] Streamlit not available (running standalone - this is normal)")

# Final Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

issues_found = []
recommendations = []

# Check each component
if 'nfl_games' in locals():
    if not nfl_games:
        issues_found.append("No NFL games found from ESPN API")
        recommendations.append("Check if it's NFL off-season or bye week")
else:
    issues_found.append("Could not fetch NFL games")
    recommendations.append("Check ESPN NFL API connection")

if 'ncaa_games' in locals():
    if not ncaa_games:
        issues_found.append("No NCAA games found from ESPN API")
        recommendations.append("Check if it's NCAA off-season or no games scheduled")
else:
    issues_found.append("Could not fetch NCAA games")
    recommendations.append("Check ESPN NCAA API connection")

if 'total' in locals():
    if total == 0:
        issues_found.append("No Kalshi markets in database")
        recommendations.append("Run: python sync_kalshi_team_winners.py")
    elif active == 0:
        issues_found.append("Kalshi markets exist but none are active")
        recommendations.append("Kalshi markets may be expired - resync markets")
else:
    issues_found.append("Could not check Kalshi markets")
    recommendations.append("Check database connection and Kalshi tables")

# Display results
if issues_found:
    print("\n[!] ISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")

    print("\n[>] RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
else:
    print("\n[OK] ALL SYSTEMS OPERATIONAL")
    print("     ESPN APIs working, Kalshi markets available, matching system ready")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("""
1. If ESPN APIs are working but games not showing in UI:
   - Clear Streamlit cache (press 'C' in app or Settings > Clear cache)
   - Refresh the page
   - Check date filters aren't hiding games

2. If Kalshi markets are missing:
   - Run: python sync_kalshi_team_winners.py
   - Check .env for KALSHI_EMAIL and KALSHI_PASSWORD
   - Verify Kalshi account is active

3. If matching is failing:
   - Check team name mappings in espn_kalshi_matcher_optimized.py
   - Verify Kalshi markets exist for the games you want
   - Some NCAA games may not have Kalshi markets (focus on ranked teams)

4. If nothing works:
   - Check network connectivity
   - Verify ESPN.com is accessible
   - Check Kalshi.com is accessible
   - Review application logs for detailed errors
""")

print("=" * 80)
