import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Check watchlist table structure
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'game_watchlist'
    ORDER BY ordinal_position
""")
print("Game Watchlist Table Schema:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

print("\n" + "="*60 + "\n")

# Check active subscriptions
cur.execute("""
    SELECT id, user_id, game_id, sport, home_team, away_team, selected_team, added_at
    FROM game_watchlist
    WHERE is_active = true
    ORDER BY added_at DESC
""")
rows = cur.fetchall()

print(f"Active subscriptions: {len(rows)}\n")

if rows:
    print("Recent subscriptions:")
    for row in rows:
        print(f"  ID: {row[0]}")
        print(f"  User: {row[1]}")
        print(f"  Game ID: {row[2]}")
        print(f"  Sport: {row[3]}")
        print(f"  Teams: {row[5]} @ {row[4]}")
        print(f"  Selected: {row[6]}")
        print(f"  Added: {row[7]}")
        print()
else:
    print("No active subscriptions found")

cur.close()
conn.close()
