"""
Fix incorrectly categorized subscriptions in database
NCAA teams that were saved as NFL need to be corrected to CFB
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# College teams that were incorrectly saved as NFL
NCAA_TEAMS = [
    "Clemson Tigers",
    "Louisville Cardinals",
    "Minnesota Golden Gophers",
    "Oregon Ducks",
    "Wisconsin Badgers",
    "Indiana Hoosiers",
    "Oklahoma Sooners",
    "Alabama Crimson Tide",
    "Florida Atlantic Owls",
    "Tulane Green Wave",
    "Florida Gators",
    "Ole Miss Rebels",
    "Miami Hurricanes",
    "Virginia Tech Hokies"
]

def fix_subscriptions():
    """Fix NCAA games that were saved with sport='NFL'"""
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123'),
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432')
    )

    try:
        cur = conn.cursor()

        # Find subscriptions with college teams marked as NFL
        print("Checking for NCAA games marked as NFL...")
        print("=" * 60)

        fixed_count = 0
        for team in NCAA_TEAMS:
            # Check if any subscriptions have this team with sport='NFL'
            cur.execute("""
                SELECT game_id, away_team, home_team
                FROM game_watchlist
                WHERE sport = 'NFL'
                AND (away_team LIKE %s OR home_team LIKE %s)
            """, (f'%{team}%', f'%{team}%'))

            rows = cur.fetchall()

            for row in rows:
                game_id, away_team, home_team = row
                print(f"  Found: {away_team} @ {home_team}")

                # Update to CFB
                cur.execute("""
                    UPDATE game_watchlist
                    SET sport = 'CFB'
                    WHERE game_id = %s AND sport = 'NFL'
                """, (game_id,))

                fixed_count += cur.rowcount
                if cur.rowcount > 0:
                    print(f"    -> Fixed: Changed sport to CFB")

        conn.commit()

        print("\n" + "=" * 60)
        print(f"Fixed {fixed_count} subscriptions")

        # Show final counts by sport
        cur.execute("""
            SELECT sport, COUNT(*)
            FROM game_watchlist
            WHERE user_id = '7957298119'
            GROUP BY sport
            ORDER BY sport
        """)

        print("\nFinal subscription counts:")
        for sport, count in cur.fetchall():
            print(f"  {sport}: {count}")

        print("\n>> Database cleanup complete!")

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    fix_subscriptions()
