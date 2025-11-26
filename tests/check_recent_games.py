import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

cur.execute("""
    SELECT home_team, away_team, game_status, week, game_time
    FROM nfl_games
    WHERE week IN (11, 12) AND game_status <> 'scheduled'
    ORDER BY game_time DESC
""")

print('Recent completed/live games (Weeks 11-12):')
for row in cur.fetchall():
    print(f'Week {row[3]}: {row[1]} @ {row[0]} - {row[2]} - {row[4]}')

cur.close()
conn.close()
