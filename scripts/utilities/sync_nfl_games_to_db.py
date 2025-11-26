"""
Quick NFL Database Population Script
Fetches current NFL games from ESPN and populates nfl_games table
"""

import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
from src.espn_live_data import get_espn_client
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def normalize_game_status(espn_status):
    """Convert ESPN status format to database format"""
    status_map = {
        'STATUS_SCHEDULED': 'scheduled',
        'STATUS_IN_PROGRESS': 'live',
        'STATUS_HALFTIME': 'halftime',
        'STATUS_FINAL': 'final',
        'STATUS_POSTPONED': 'postponed',
        'STATUS_CANCELED': 'cancelled',
        'STATUS_CANCELLED': 'cancelled'
    }
    return status_map.get(espn_status, 'scheduled')


def sync_nfl_games():
    """Fetch NFL games from ESPN and populate database"""

    # Connect to database
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()

    try:
        # Fetch games from ESPN
        logger.info("Fetching NFL games from ESPN...")
        espn = get_espn_client()

        all_games = []
        # Fetch weeks 11-18 (rest of season + playoffs)
        for week in range(11, 19):
            try:
                week_games = espn.get_scoreboard(week=week)
                if week_games:
                    # Add week number to each game since ESPN doesn't include it
                    for game in week_games:
                        game['week'] = week
                    all_games.extend(week_games)
                    logger.info(f"Week {week}: {len(week_games)} games")
            except Exception as e:
                logger.debug(f"Week {week} not available: {e}")

        logger.info(f"\nTotal games fetched: {len(all_games)}")

        # Insert/update games in database
        inserted = 0
        updated = 0

        for game in all_games:
            game_id = game.get('game_id')
            if not game_id:
                continue

            # Check if game exists
            cur.execute("SELECT id FROM nfl_games WHERE game_id = %s", (game_id,))
            exists = cur.fetchone()

            # Normalize game status
            normalized_status = normalize_game_status(game.get('status'))

            if exists:
                # Update existing game
                cur.execute("""
                    UPDATE nfl_games SET
                        season = %s,
                        home_team = %s,
                        away_team = %s,
                        home_score = %s,
                        away_score = %s,
                        game_status = %s,
                        is_live = %s,
                        game_time = %s,
                        week = %s,
                        quarter = %s,
                        time_remaining = %s,
                        last_updated = NOW()
                    WHERE game_id = %s
                """, (
                    2025,  # Current season
                    game.get('home_team'),
                    game.get('away_team'),
                    game.get('home_score', 0),
                    game.get('away_score', 0),
                    normalized_status,
                    game.get('is_live', False),
                    game.get('game_time'),
                    game.get('week'),
                    game.get('period', 0),
                    game.get('clock', '0:00'),
                    game_id
                ))
                updated += 1
            else:
                # Insert new game
                cur.execute("""
                    INSERT INTO nfl_games (
                        game_id, season, home_team, away_team, home_score, away_score,
                        game_status, is_live, game_time, week, quarter, time_remaining,
                        created_at, last_updated
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """, (
                    game_id,
                    2025,  # Current season
                    game.get('home_team'),
                    game.get('away_team'),
                    game.get('home_score', 0),
                    game.get('away_score', 0),
                    normalized_status,
                    game.get('is_live', False),
                    game.get('game_time'),
                    game.get('week'),
                    game.get('period', 0),
                    game.get('clock', '0:00')
                ))
                inserted += 1

        conn.commit()

        logger.info(f"\nSync Complete!")
        logger.info(f"   Inserted: {inserted} new games")
        logger.info(f"   Updated: {updated} existing games")

        # Show sample of what was synced
        cur.execute("""
            SELECT home_team, away_team, game_status, game_time, week
            FROM nfl_games
            ORDER BY game_time DESC
            LIMIT 10
        """)

        print("\nSample of synced games:")
        for row in cur.fetchall():
            home, away, status, game_time, week = row
            print(f"   Week {week}: {away} @ {home} - {status} - {game_time}")

        return True

    except Exception as e:
        logger.error(f"Error syncing games: {e}")
        conn.rollback()
        return False

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    print("="*60)
    print("NFL GAMES DATABASE SYNC")
    print("="*60)
    print()

    success = sync_nfl_games()

    if success:
        print("\nDatabase successfully populated!")
        print("\nNext steps:")
        print("   1. Refresh your Game Hub page")
        print("   2. Check your subscriptions in Settings tab")
        print("   3. Run background monitoring service if needed")
    else:
        print("\nSync failed. Check logs above for details.")
