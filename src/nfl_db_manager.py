"""
NFL Database Manager
Handles all database operations for NFL real-time data pipeline
"""

import os
import psycopg2
import psycopg2.extras
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import json
from src.database.connection_pool import DatabaseConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLDBManager:
    """Manages database operations for NFL real-time data pipeline"""

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': '5432',
            'database': 'magnus',
            'user': 'postgres',
            'password': os.getenv('DB_PASSWORD')
        }
        self.initialize_database()

    def get_connection(self):
        """Get database connection from shared pool"""
        try:
            pool = DatabaseConnectionPool()
            return pool._pool.getconn()
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            return psycopg2.connect(**self.db_config)

    def release_connection(self, conn):
        """Release connection back to shared pool"""
        if not conn:
            return

        try:
            pool = DatabaseConnectionPool()
            if pool._pool and not conn.closed:
                pool._pool.putconn(conn)
            elif not conn.closed:
                conn.close()
        except Exception as e:
            logger.error(f"Error releasing connection: {e}")
            try:
                if conn and not conn.closed:
                    conn.close()
            except:
                pass

    def initialize_database(self):
        """Initialize database tables from schema file"""
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'nfl_data_schema.sql')

            if not os.path.exists(schema_path):
                logger.warning(f"Schema file not found: {schema_path}")
                return

            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()

            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(schema_sql)
            conn.commit()
            cur.close()
            self.release_connection(conn)

            logger.info("NFL database tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    # ========================================================================
    # GAME OPERATIONS
    # ========================================================================

    def upsert_game(self, game_data: Dict[str, Any]) -> int:
        """
        Insert or update NFL game data

        Args:
            game_data: Dictionary containing game information

        Returns:
            Game database ID
        """
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO nfl_games (
                    game_id, season, week,
                    home_team, away_team, home_team_abbr, away_team_abbr,
                    game_time, venue, is_outdoor,
                    home_score, away_score, quarter, time_remaining, possession,
                    game_status, is_live, started_at, finished_at,
                    spread_home, spread_odds_home, spread_odds_away,
                    moneyline_home, moneyline_away,
                    over_under, over_odds, under_odds,
                    temperature, weather_condition, wind_speed, precipitation_chance,
                    raw_game_data, raw_weather_data
                ) VALUES (
                    %(game_id)s, %(season)s, %(week)s,
                    %(home_team)s, %(away_team)s, %(home_team_abbr)s, %(away_team_abbr)s,
                    %(game_time)s, %(venue)s, %(is_outdoor)s,
                    %(home_score)s, %(away_score)s, %(quarter)s, %(time_remaining)s, %(possession)s,
                    %(game_status)s, %(is_live)s, %(started_at)s, %(finished_at)s,
                    %(spread_home)s, %(spread_odds_home)s, %(spread_odds_away)s,
                    %(moneyline_home)s, %(moneyline_away)s,
                    %(over_under)s, %(over_odds)s, %(under_odds)s,
                    %(temperature)s, %(weather_condition)s, %(wind_speed)s, %(precipitation_chance)s,
                    %(raw_game_data)s, %(raw_weather_data)s
                )
                ON CONFLICT (game_id) DO UPDATE SET
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    quarter = EXCLUDED.quarter,
                    time_remaining = EXCLUDED.time_remaining,
                    possession = EXCLUDED.possession,
                    game_status = EXCLUDED.game_status,
                    is_live = EXCLUDED.is_live,
                    started_at = COALESCE(EXCLUDED.started_at, nfl_games.started_at),
                    finished_at = EXCLUDED.finished_at,
                    spread_home = COALESCE(EXCLUDED.spread_home, nfl_games.spread_home),
                    moneyline_home = COALESCE(EXCLUDED.moneyline_home, nfl_games.moneyline_home),
                    moneyline_away = COALESCE(EXCLUDED.moneyline_away, nfl_games.moneyline_away),
                    over_under = COALESCE(EXCLUDED.over_under, nfl_games.over_under),
                    temperature = EXCLUDED.temperature,
                    weather_condition = EXCLUDED.weather_condition,
                    wind_speed = EXCLUDED.wind_speed,
                    precipitation_chance = EXCLUDED.precipitation_chance,
                    raw_game_data = EXCLUDED.raw_game_data,
                    raw_weather_data = EXCLUDED.raw_weather_data,
                    last_synced = NOW()
                RETURNING id
            """, {
                **game_data,
                'raw_game_data': json.dumps(game_data.get('raw_game_data', {})),
                'raw_weather_data': json.dumps(game_data.get('raw_weather_data', {}))
            })

            game_id = cur.fetchone()[0]
            conn.commit()

            logger.debug(f"Upserted game {game_data['game_id']} -> DB ID {game_id}")
            return game_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error upserting game {game_data.get('game_id')}: {e}")
            raise
        finally:
            cur.close()
            self.release_connection(conn)

    def get_live_games(self) -> List[Dict]:
        """Get all currently live games"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            cur.execute("SELECT * FROM v_nfl_live_games")
            games = cur.fetchall()
            return [dict(g) for g in games]
        finally:
            cur.close()
            self.release_connection(conn)

    def get_upcoming_games(self, hours_ahead: int = 24) -> List[Dict]:
        """Get games starting in the next N hours"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM nfl_games
                WHERE game_status = 'scheduled'
                  AND game_time > NOW()
                  AND game_time <= NOW() + INTERVAL '%s hours'
                ORDER BY game_time ASC
            """, (hours_ahead,))

            games = cur.fetchall()
            return [dict(g) for g in games]
        finally:
            cur.close()
            self.release_connection(conn)

    def get_game_by_external_id(self, external_game_id: str) -> Optional[Dict]:
        """Get game by external API game_id"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM nfl_games WHERE game_id = %s
            """, (external_game_id,))

            game = cur.fetchone()
            return dict(game) if game else None
        finally:
            cur.close()
            self.release_connection(conn)

    # ========================================================================
    # PLAY OPERATIONS
    # ========================================================================

    def insert_play(self, play_data: Dict[str, Any]) -> int:
        """
        Insert play-by-play data

        Args:
            play_data: Dictionary containing play information

        Returns:
            Play database ID
        """
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO nfl_plays (
                    game_id, play_id, sequence_number, quarter, time_remaining,
                    play_type, description, down, yards_to_go, yard_line,
                    yards_gained, is_scoring_play, is_turnover, is_penalty,
                    offense_team, defense_team, player_name, player_position,
                    points_home, points_away, raw_play_data
                ) VALUES (
                    %(game_id)s, %(play_id)s, %(sequence_number)s, %(quarter)s, %(time_remaining)s,
                    %(play_type)s, %(description)s, %(down)s, %(yards_to_go)s, %(yard_line)s,
                    %(yards_gained)s, %(is_scoring_play)s, %(is_turnover)s, %(is_penalty)s,
                    %(offense_team)s, %(defense_team)s, %(player_name)s, %(player_position)s,
                    %(points_home)s, %(points_away)s, %(raw_play_data)s
                )
                ON CONFLICT (play_id) DO NOTHING
                RETURNING id
            """, {
                **play_data,
                'raw_play_data': json.dumps(play_data.get('raw_play_data', {}))
            })

            result = cur.fetchone()
            conn.commit()

            if result:
                play_id = result[0]
                logger.debug(f"Inserted play {play_data['play_id']} -> DB ID {play_id}")
                return play_id
            else:
                logger.debug(f"Play {play_data['play_id']} already exists, skipped")
                return None

        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting play {play_data.get('play_id')}: {e}")
            raise
        finally:
            cur.close()
            self.release_connection(conn)

    def get_recent_significant_plays(self, limit: int = 50) -> List[Dict]:
        """Get recent high-impact plays"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM v_nfl_significant_plays LIMIT %s
            """, (limit,))

            plays = cur.fetchall()
            return [dict(p) for p in plays]
        finally:
            cur.close()
            self.release_connection(conn)

    # ========================================================================
    # PLAYER STATS OPERATIONS
    # ========================================================================

    def upsert_player_stats(self, stats_data: Dict[str, Any]) -> int:
        """Insert or update player statistics"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO nfl_player_stats (
                    game_id, player_name, player_id, team, position,
                    passing_attempts, passing_completions, passing_yards,
                    passing_touchdowns, passing_interceptions,
                    rushing_attempts, rushing_yards, rushing_touchdowns,
                    receptions, receiving_yards, receiving_touchdowns, targets,
                    tackles, sacks, interceptions, forced_fumbles,
                    field_goals_made, field_goals_attempted,
                    extra_points_made, extra_points_attempted,
                    raw_stats_data
                ) VALUES (
                    %(game_id)s, %(player_name)s, %(player_id)s, %(team)s, %(position)s,
                    %(passing_attempts)s, %(passing_completions)s, %(passing_yards)s,
                    %(passing_touchdowns)s, %(passing_interceptions)s,
                    %(rushing_attempts)s, %(rushing_yards)s, %(rushing_touchdowns)s,
                    %(receptions)s, %(receiving_yards)s, %(receiving_touchdowns)s, %(targets)s,
                    %(tackles)s, %(sacks)s, %(interceptions)s, %(forced_fumbles)s,
                    %(field_goals_made)s, %(field_goals_attempted)s,
                    %(extra_points_made)s, %(extra_points_attempted)s,
                    %(raw_stats_data)s
                )
                ON CONFLICT (game_id, player_id, player_name) DO UPDATE SET
                    passing_attempts = EXCLUDED.passing_attempts,
                    passing_completions = EXCLUDED.passing_completions,
                    passing_yards = EXCLUDED.passing_yards,
                    passing_touchdowns = EXCLUDED.passing_touchdowns,
                    passing_interceptions = EXCLUDED.passing_interceptions,
                    rushing_attempts = EXCLUDED.rushing_attempts,
                    rushing_yards = EXCLUDED.rushing_yards,
                    rushing_touchdowns = EXCLUDED.rushing_touchdowns,
                    receptions = EXCLUDED.receptions,
                    receiving_yards = EXCLUDED.receiving_yards,
                    receiving_touchdowns = EXCLUDED.receiving_touchdowns,
                    targets = EXCLUDED.targets,
                    tackles = EXCLUDED.tackles,
                    sacks = EXCLUDED.sacks,
                    interceptions = EXCLUDED.interceptions,
                    forced_fumbles = EXCLUDED.forced_fumbles,
                    field_goals_made = EXCLUDED.field_goals_made,
                    field_goals_attempted = EXCLUDED.field_goals_attempted,
                    extra_points_made = EXCLUDED.extra_points_made,
                    extra_points_attempted = EXCLUDED.extra_points_attempted,
                    raw_stats_data = EXCLUDED.raw_stats_data,
                    last_updated = NOW()
                RETURNING id
            """, {
                **stats_data,
                'raw_stats_data': json.dumps(stats_data.get('raw_stats_data', {}))
            })

            stat_id = cur.fetchone()[0]
            conn.commit()
            return stat_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error upserting player stats: {e}")
            raise
        finally:
            cur.close()
            self.release_connection(conn)

    # ========================================================================
    # INJURY OPERATIONS
    # ========================================================================

    def insert_injury_report(self, injury_data: Dict[str, Any]) -> int:
        """Insert new injury report"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO nfl_injuries (
                    player_name, player_id, team, position,
                    injury_type, injury_status, description,
                    game_id, week, reported_at, is_active, source,
                    raw_injury_data
                ) VALUES (
                    %(player_name)s, %(player_id)s, %(team)s, %(position)s,
                    %(injury_type)s, %(injury_status)s, %(description)s,
                    %(game_id)s, %(week)s, %(reported_at)s, %(is_active)s, %(source)s,
                    %(raw_injury_data)s
                )
                RETURNING id
            """, {
                **injury_data,
                'raw_injury_data': json.dumps(injury_data.get('raw_injury_data', {}))
            })

            injury_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Inserted injury report for {injury_data['player_name']}: {injury_data['injury_status']}")
            return injury_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting injury report: {e}")
            raise
        finally:
            cur.close()
            self.release_connection(conn)

    def get_active_injuries(self, team: Optional[str] = None) -> List[Dict]:
        """Get active injury reports, optionally filtered by team"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            query = "SELECT * FROM nfl_injuries WHERE is_active = true"
            params = []

            if team:
                query += " AND team = %s"
                params.append(team)

            query += " ORDER BY reported_at DESC"

            cur.execute(query, params)
            injuries = cur.fetchall()
            return [dict(i) for i in injuries]
        finally:
            cur.close()
            self.release_connection(conn)

    # ========================================================================
    # KALSHI CORRELATION OPERATIONS
    # ========================================================================

    def insert_kalshi_correlation(self, correlation_data: Dict[str, Any]) -> int:
        """Record correlation between NFL event and Kalshi price movement"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO nfl_kalshi_correlations (
                    game_id, play_id, event_type, event_timestamp,
                    kalshi_market_id, market_ticker,
                    price_before, price_after, price_change_pct,
                    volume_before, volume_after, volume_spike_pct,
                    correlation_strength, impact_level
                ) VALUES (
                    %(game_id)s, %(play_id)s, %(event_type)s, %(event_timestamp)s,
                    %(kalshi_market_id)s, %(market_ticker)s,
                    %(price_before)s, %(price_after)s, %(price_change_pct)s,
                    %(volume_before)s, %(volume_after)s, %(volume_spike_pct)s,
                    %(correlation_strength)s, %(impact_level)s
                )
                RETURNING id
            """, correlation_data)

            corr_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Recorded Kalshi correlation: {correlation_data['event_type']} -> {correlation_data['price_change_pct']:.2f}%")
            return corr_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting Kalshi correlation: {e}")
            raise
        finally:
            cur.close()
            self.release_connection(conn)

    # ========================================================================
    # ALERT OPERATIONS
    # ========================================================================

    def log_alert(self, alert_data: Dict[str, Any]) -> int:
        """Log sent alert to history"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO nfl_alert_history (
                    trigger_id, alert_type, subject, message,
                    game_id, play_id, kalshi_market_id,
                    notification_channel, delivery_status,
                    telegram_message_id, error_message, alert_data
                ) VALUES (
                    %(trigger_id)s, %(alert_type)s, %(subject)s, %(message)s,
                    %(game_id)s, %(play_id)s, %(kalshi_market_id)s,
                    %(notification_channel)s, %(delivery_status)s,
                    %(telegram_message_id)s, %(error_message)s, %(alert_data)s
                )
                RETURNING id
            """, {
                **alert_data,
                'alert_data': json.dumps(alert_data.get('alert_data', {}))
            })

            alert_id = cur.fetchone()[0]
            conn.commit()
            return alert_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error logging alert: {e}")
            raise
        finally:
            cur.close()
            self.release_connection(conn)

    def get_active_alert_triggers(self) -> List[Dict]:
        """Get all active alert configurations"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM nfl_alert_triggers
                WHERE is_active = true
                ORDER BY notification_priority DESC
            """)

            triggers = cur.fetchall()
            return [dict(t) for t in triggers]
        finally:
            cur.close()
            self.release_connection(conn)

    # ========================================================================
    # SYNC LOGGING
    # ========================================================================

    def start_sync_log(self, sync_type: str, sync_scope: str = 'all_games') -> int:
        """Start a new sync operation log"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO nfl_data_sync_log (sync_type, sync_scope, sync_status)
                VALUES (%s, %s, 'running')
                RETURNING id
            """, (sync_type, sync_scope))

            sync_id = cur.fetchone()[0]
            conn.commit()
            return sync_id
        finally:
            cur.close()
            self.release_connection(conn)

    def complete_sync_log(
        self,
        sync_id: int,
        records_fetched: int,
        records_inserted: int,
        records_updated: int,
        records_failed: int = 0,
        api_calls: int = 0,
        api_errors: int = 0,
        status: str = 'completed',
        error_msg: Optional[str] = None
    ):
        """Complete a sync operation log"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            started_time = None
            cur.execute("SELECT started_at FROM nfl_data_sync_log WHERE id = %s", (sync_id,))
            result = cur.fetchone()
            if result:
                started_time = result[0]

            duration_ms = None
            if started_time:
                duration = datetime.now(started_time.tzinfo) - started_time
                duration_ms = int(duration.total_seconds() * 1000)

            cur.execute("""
                UPDATE nfl_data_sync_log SET
                    records_fetched = %s,
                    records_inserted = %s,
                    records_updated = %s,
                    records_failed = %s,
                    api_calls_made = %s,
                    api_errors = %s,
                    duration_ms = %s,
                    sync_status = %s,
                    error_message = %s,
                    completed_at = NOW()
                WHERE id = %s
            """, (
                records_fetched, records_inserted, records_updated, records_failed,
                api_calls, api_errors, duration_ms, status, error_msg, sync_id
            ))

            conn.commit()
        finally:
            cur.close()
            self.release_connection(conn)

    # ========================================================================
    # ANALYTICS & REPORTING
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            stats = {}

            # Total games
            cur.execute("SELECT COUNT(*) as total FROM nfl_games")
            stats['total_games'] = cur.fetchone()['total']

            # Live games
            cur.execute("SELECT COUNT(*) as live FROM nfl_games WHERE is_live = true")
            stats['live_games'] = cur.fetchone()['live']

            # Total plays
            cur.execute("SELECT COUNT(*) as total FROM nfl_plays")
            stats['total_plays'] = cur.fetchone()['total']

            # Active injuries
            cur.execute("SELECT COUNT(*) as active FROM nfl_injuries WHERE is_active = true")
            stats['active_injuries'] = cur.fetchone()['active']

            # Alerts sent today
            cur.execute("""
                SELECT COUNT(*) as today
                FROM nfl_alert_history
                WHERE sent_at > CURRENT_DATE
            """)
            stats['alerts_today'] = cur.fetchone()['today']

            # Recent sync performance
            cur.execute("""
                SELECT
                    sync_type,
                    AVG(duration_ms) as avg_duration_ms,
                    MAX(completed_at) as last_run
                FROM nfl_data_sync_log
                WHERE sync_status = 'completed'
                  AND completed_at > NOW() - INTERVAL '24 hours'
                GROUP BY sync_type
            """)
            stats['sync_performance'] = [dict(row) for row in cur.fetchall()]

            return stats

        finally:
            cur.close()
            self.release_connection(conn)


if __name__ == "__main__":
    # Test database manager
    db = NFLDBManager()

    print("\n" + "="*80)
    print("NFL DATABASE MANAGER - Test")
    print("="*80)

    # Get stats
    stats = db.get_stats()
    print("\nDatabase Stats:")
    print(f"  Total Games: {stats['total_games']}")
    print(f"  Live Games: {stats['live_games']}")
    print(f"  Total Plays: {stats['total_plays']}")
    print(f"  Active Injuries: {stats['active_injuries']}")
    print(f"  Alerts Today: {stats['alerts_today']}")

    if stats['sync_performance']:
        print("\nSync Performance (Last 24h):")
        for sync in stats['sync_performance']:
            print(f"  {sync['sync_type']}: {sync['avg_duration_ms']:.0f}ms avg, last run {sync['last_run']}")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
