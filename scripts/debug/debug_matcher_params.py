"""Debug the exact parameters being passed"""
from datetime import datetime, timedelta
from src.espn_live_data import get_espn_client
from src.espn_kalshi_matcher import ESPNKalshiMatcher

# Get one game
espn_client = get_espn_client()
games = espn_client.get_scoreboard()

if games:
    game = games[0]
    print(f"Game: {game['away_team']} @ {game['home_team']}")
    print(f"Game time: {game.get('game_time')}")

    # Manual matching to see parameters
    away_team = game.get('away_team', '')
    home_team = game.get('home_team', '')
    game_time = game.get('game_time', '')

    print(f"\nVariables:")
    print(f"  away_team: {away_team} (type: {type(away_team)})")
    print(f"  home_team: {home_team} (type: {type(home_team)})")
    print(f"  game_time: {game_time} (type: {type(game_time)})")

    # Calculate game_date
    if isinstance(game_time, datetime):
        game_date = game_time.date()
    else:
        game_date = datetime.now().date()

    date_start = game_date - timedelta(days=3)
    date_end = game_date + timedelta(days=3)

    print(f"\nDate calculations:")
    print(f"  game_date: {game_date} (type: {type(game_date)})")
    print(f"  date_start: {date_start} (type: {type(date_start)})")
    print(f"  date_end: {date_end} (type: {type(date_end)})")

    # Get team variations
    matcher = ESPNKalshiMatcher()
    away_variations = matcher.get_team_variations(away_team)
    home_variations = matcher.get_team_variations(home_team)

    print(f"\nTeam variations:")
    print(f"  away: {away_variations}")
    print(f"  home: {home_variations}")

    # Test with first variation
    away_var = away_variations[0]
    home_var = home_variations[0]

    print(f"\nUsing variations: {away_var} vs {home_var}")

    # Build params tuple
    params = (
        f'%{away_var}%',
        f'%{home_var}%',
        date_start,
        date_end,
        date_start,
        date_end
    )

    print(f"\nParameters tuple:")
    for i, p in enumerate(params, 1):
        print(f"  {i}. {p} (type: {type(p)})")

    print(f"\nTotal parameters: {len(params)}")

    # Count %s in query
    from src.espn_kalshi_matcher import ESPNKalshiMatcher
    import inspect
    source = inspect.getsource(ESPNKalshiMatcher.match_game_to_kalshi)

    # Extract the query
    import re
    query_match = re.search(r'query = """(.+?)"""', source, re.DOTALL)
    if query_match:
        query_text = query_match.group(1)
        placeholder_count = query_text.count('%s')
        print(f"\n%s placeholders in query: {placeholder_count}")

    # Try executing the query
    print("\nAttempting query execution...")
    import psycopg2.extras
    from src.kalshi_db_manager import KalshiDBManager

    db = KalshiDBManager()
    conn = db.get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    query = """
    SELECT
        ticker,
        title,
        yes_price,
        no_price,
        volume,
        close_time,
        market_type,
        raw_data->>'expected_expiration_time' as game_time_str
    FROM kalshi_markets
    WHERE
        (
            title ILIKE %s AND title ILIKE %s
        )
        AND (
            market_type IN ('nfl', 'cfb', 'winner', 'all')
            OR raw_data->>'market_type' IN ('nfl', 'cfb', 'winner')
            OR ticker LIKE 'KXNFLGAME%'
            OR ticker LIKE 'KXNCAAFGAME%'
        )
        AND (
            -- Match by expected_expiration_time (actual game time) if available
            (raw_data->>'expected_expiration_time' IS NOT NULL
             AND (raw_data->>'expected_expiration_time')::timestamp >= %s
             AND (raw_data->>'expected_expiration_time')::timestamp <= %s)
            OR
            -- Fallback to close_time for older data
            (raw_data->>'expected_expiration_time' IS NULL
             AND close_time >= %s
             AND close_time <= %s)
        )
        AND status != 'closed'
        AND yes_price IS NOT NULL
    ORDER BY volume DESC, close_time ASC
    LIMIT 1
    """

    try:
        cur.execute(query, params)
        result = cur.fetchone()
        if result:
            print(f"SUCCESS! Found: {result['ticker']}")
        else:
            print("No match found (but query executed successfully)")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    cur.close()
    db.release_connection(conn)
