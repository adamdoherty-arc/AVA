"""
Kalshi Database Manager
Handles all database operations for Kalshi football markets
"""

import os
import psycopg2
import psycopg2.extras
import logging
from typing import List, Dict, Optional
from datetime import datetime
import json
from src.database.connection_pool import DatabaseConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalshiDBManager:
    """Manages database operations for Kalshi markets"""

    def __init__(self) -> None:
        self.db_config = {
            'host': 'localhost',
            'port': '5432',
            'database': 'magnus',
            'user': 'postgres',
            'password': os.getenv('DB_PASSWORD')
        }
        # Initialize shared pool
        DatabaseConnectionPool()
        self.initialize_database()

    def get_connection(self) -> None:
        """Get database connection from shared pool"""
        try:
            pool = DatabaseConnectionPool()
            return pool._pool.getconn()
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            # Fallback to direct connection if pool fails
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

    def initialize_database(self) -> None:
        """Initialize database tables from schema file"""
        conn = None
        cur = None
        try:
            # Check if tables already exist
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'kalshi_markets'
                )
            """)
            tables_exist = cur.fetchone()[0]

            if tables_exist:
                logger.info("Kalshi database tables already initialized")
                return

            # Tables don't exist, run full schema
            schema_path = os.path.join(os.path.dirname(__file__), 'kalshi_schema.sql')

            if not os.path.exists(schema_path):
                logger.warning(f"Schema file not found: {schema_path}")
                return

            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()

            # Execute schema SQL in a transaction
            cur.execute(schema_sql)
            conn.commit()

            logger.info("Kalshi database tables initialized successfully")

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error initializing database: {e}")
            logger.error(f"Schema path attempted: {schema_path if 'schema_path' in locals() else 'Unknown'}")
            # Don't raise - allow the application to continue even if schema init fails
            logger.warning("Continuing with existing database schema")
        finally:
            if cur:
                cur.close()
            if conn:
                self.release_connection(conn)

    def store_markets(self, markets: List[Dict], market_type: str) -> int:
        """
        Store or update markets in database

        Args:
            markets: List of market dictionaries from Kalshi API
            market_type: 'nfl' or 'college'

        Returns:
            Number of markets stored/updated
        """
        if not markets:
            return 0

        conn = self.get_connection()
        cur = conn.cursor()
        stored_count = 0

        try:
            for market in markets:
                # Extract market data
                ticker = market.get('ticker')
                title = market.get('title', '')
                subtitle = market.get('subtitle', '')
                series_ticker = market.get('series_ticker', '')

                # Parse teams from title (if possible)
                home_team, away_team = self._extract_teams(title)

                # Extract prices
                last_price = market.get('last_price')
                yes_price = last_price / 100 if last_price else None
                no_price = (100 - last_price) / 100 if last_price else None

                # Extract other fields
                volume = market.get('volume', 0)
                open_interest = market.get('open_interest', 0)
                status = market.get('status', 'open')
                close_time = market.get('close_time')
                expiration_time = market.get('expiration_time')

                # Truncate fields to database limits to avoid errors
                ticker = ticker[:100] if ticker else None
                series_ticker = series_ticker[:100] if series_ticker else ''
                home_team = home_team[:100] if home_team else None
                away_team = away_team[:100] if away_team else None

                # Store market
                cur.execute("""
                    INSERT INTO kalshi_markets (
                        ticker, title, subtitle, market_type, series_ticker,
                        home_team, away_team,
                        yes_price, no_price, volume, open_interest,
                        status, close_time, expiration_time,
                        raw_data, synced_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, NOW()
                    )
                    ON CONFLICT (ticker) DO UPDATE SET
                        title = EXCLUDED.title,
                        subtitle = EXCLUDED.subtitle,
                        yes_price = EXCLUDED.yes_price,
                        no_price = EXCLUDED.no_price,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest,
                        status = EXCLUDED.status,
                        close_time = EXCLUDED.close_time,
                        expiration_time = EXCLUDED.expiration_time,
                        raw_data = EXCLUDED.raw_data,
                        synced_at = NOW(),
                        last_updated = NOW()
                """, (
                    ticker, title, subtitle, market_type, series_ticker,
                    home_team, away_team,
                    yes_price, no_price, volume, open_interest,
                    status, close_time, expiration_time,
                    json.dumps(market)
                ))

                stored_count += 1

            conn.commit()
            logger.info(f"Stored/updated {stored_count} {market_type} markets")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing markets: {e}")
            raise

        finally:
            cur.close()
            conn.close()

        return stored_count

    def update_market_prices(self, ticker: str, yes_price: float, no_price: float) -> bool:
        """
        Update market prices only (for real-time price monitoring)

        Args:
            ticker: Market ticker
            yes_price: Updated YES price (0-1)
            no_price: Updated NO price (0-1)

        Returns:
            True if updated successfully, False otherwise
        """
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                UPDATE kalshi_markets
                SET yes_price = %s,
                    no_price = %s,
                    last_updated = NOW()
                WHERE ticker = %s
            """, (yes_price, no_price, ticker))

            conn.commit()
            return cur.rowcount > 0

        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating prices for {ticker}: {e}")
            return False

        finally:
            cur.close()
            conn.close()

    def _extract_teams(self, title: str) -> tuple:
        """
        Extract team names from Kalshi market title using robust pattern matching
        and validation against known team databases.

        This method handles multi-word team names (e.g., "New England", "Tampa Bay",
        "Kansas City") correctly by using regex pattern matching and validating
        against NFL/NCAA team databases.

        Args:
            title: Market title from Kalshi

        Returns:
            Tuple of (home_team, away_team) or (None, None) if extraction fails

        Examples:
            >>> _extract_teams("Will New England beat Cincinnati?")
            ('Cincinnati', 'New England')

            >>> _extract_teams("New England at Cincinnati Winner?")
            ('Cincinnati', 'New England')

            >>> _extract_teams("Will the Chiefs beat the Bills?")
            ('Bills', 'Chiefs')
        """
        import re
        from difflib import get_close_matches
        from typing import Optional

        # Import team databases for validation
        try:
            # Build comprehensive team name list
            all_team_names = set()

            # Try to import NFL teams
            try:
                from src.nfl_team_database import NFL_TEAMS
                for team_key, team_info in NFL_TEAMS.items():
                    all_team_names.add(team_key)
                    if isinstance(team_info, dict):
                        all_team_names.add(team_info.get('full_name', ''))
                        all_team_names.add(team_info.get('city', ''))
            except ImportError:
                pass

            # Try to import NCAA teams
            try:
                from src.ncaa_team_database import NCAA_TEAMS
                all_team_names.update(NCAA_TEAMS.keys())
            except ImportError:
                pass

            # Try to import NBA teams
            try:
                from src.nba_team_database import NBA_TEAMS as NBA_TEAMS_DB
                for team_abbr, team_info in NBA_TEAMS_DB.items():
                    if isinstance(team_info, dict):
                        all_team_names.add(team_info.get('full_name', ''))
                        all_team_names.add(team_info.get('city', ''))
                        all_team_names.add(team_info.get('name', ''))
            except ImportError:
                pass

            # Try to import MLB teams
            try:
                from src.mlb_team_database import MLB_TEAMS as MLB_TEAMS_DB
                for team_abbr, team_info in MLB_TEAMS_DB.items():
                    if isinstance(team_info, dict):
                        all_team_names.add(team_info.get('full_name', ''))
                        all_team_names.add(team_info.get('city', ''))
                        all_team_names.add(team_info.get('name', ''))
                        all_team_names.add(team_info.get('nickname', ''))
            except ImportError:
                pass

        except Exception as e:
            logger.warning(f"Could not import team databases: {e}")
            all_team_names = set()

        # Define indicators and their regex patterns
        indicator_patterns = [
            (r'\s+vs\.?\s+', 'vs'),
            (r'\s+v\.?\s+', 'v'),
            (r'\s+@\s+', '@'),
            (r'\s+at\s+', 'at'),
            (r'\s+against\s+', 'against'),
        ]

        def normalize_ncaa_abbreviations(text: str) -> str:
            """Normalize NCAA-specific abbreviations and team names."""
            # Normalize "St." to "State" for common state universities
            # But preserve "St." for Saint schools (St. John's, St. Bonaventure, etc.)

            # Handle common state university patterns
            text = re.sub(r'\bFla\.?\s+St', 'Florida State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bOhio\s+St', 'Ohio State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bMich\.?\s+St', 'Michigan State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bKansas\s+St', 'Kansas State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bOklahoma\s+St', 'Oklahoma State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bIowa\s+St', 'Iowa State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bMiss\.?\s+St', 'Mississippi State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bN\.?C\.?\s+St', 'NC State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bAriz\.?\s+St', 'Arizona State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bColo\.?\s+St', 'Colorado State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bFresno\s+St', 'Fresno State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bBoise\s+St', 'Boise State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bSan Diego\s+St', 'San Diego State', text, flags=re.IGNORECASE)
            text = re.sub(r'\bSan Jose\s+St', 'San Jose State', text, flags=re.IGNORECASE)

            # Don't modify "St." when it's part of Saint schools
            # (St. John's, St. Bonaventure, St. Joseph's, etc.)
            # These will be preserved as-is

            return text

        def clean_team_name(text: str) -> str:
            """Clean extracted team name by removing articles and punctuation."""
            # First normalize NCAA abbreviations
            text = normalize_ncaa_abbreviations(text)

            text = re.sub(r'^(will\s+|the\s+|a\s+)', '', text.strip(), flags=re.IGNORECASE)
            text = re.sub(r'[?.!]+$', '', text.strip())

            # Only remove possessive 's if followed by a verb (like "Chiefs's beat")
            # Preserve it for school names like "St. John's"
            text = re.sub(r"'s\s+(beat|win|defeat|play)", r" \1", text, flags=re.IGNORECASE)

            text = re.sub(r'\s+(win|beat|defeat|winner|game)\s*$', '', text, flags=re.IGNORECASE)
            text = ' '.join(text.split())
            return text.strip()

        def validate_team_name(name: str) -> Optional[str]:
            """Validate and normalize team name against known databases."""
            if not name:
                return None

            # If no team database available, return name as-is
            if not all_team_names:
                return name

            # Direct match (case-insensitive)
            for known_team in all_team_names:
                if name.lower() == known_team.lower():
                    return known_team

            # Partial match - but only if it's likely a valid match
            # Avoid partial matches that would truncate multi-word names
            for known_team in all_team_names:
                # Only use partial match if the known team contains the input
                # This prevents "Chicago" from matching when we have "Chicago St."
                if known_team.lower().startswith(name.lower() + ' '):
                    # Input is a prefix of known team (e.g., "Florida" in "Florida State")
                    return known_team
                elif name.lower().startswith(known_team.lower() + ' '):
                    # Known team is a prefix of input (e.g., "Miami" in "Miami (OH)")
                    # Use the more specific input
                    return name

            # Fuzzy match as fallback for typos
            matches = get_close_matches(name, all_team_names, n=1, cutoff=0.85)
            if matches:
                logger.debug(f"Fuzzy matched '{name}' to '{matches[0]}'")
                return matches[0]

            # NCAA team not in database - preserve full name as extracted
            # This prevents truncation of smaller schools not in our database
            if len(name) >= 3:  # Minimum 3 chars to avoid abbreviations
                logger.debug(f"NCAA team not in database, preserving: '{name}'")
                return name

            return None

        # Try each indicator pattern
        for pattern, indicator_name in indicator_patterns:
            match = re.search(pattern, title, flags=re.IGNORECASE)

            if match:
                before_indicator = title[:match.start()]
                after_indicator = title[match.end():]

                team1_raw = clean_team_name(before_indicator)
                team2_raw = clean_team_name(after_indicator)

                team1 = validate_team_name(team1_raw)
                team2 = validate_team_name(team2_raw)

                if team1 and team2:
                    if indicator_name in ['@', 'at']:
                        home_team = team2
                        away_team = team1
                    else:
                        home_team = team2
                        away_team = team1

                    logger.debug(f"Extracted teams: home={home_team}, away={away_team}")
                    return (home_team, away_team)

        # Fallback: Try "beat" pattern
        beat_pattern = r'will\s+(.+?)\s+beat\s+(.+?)(?:\?|$)'
        beat_match = re.search(beat_pattern, title, flags=re.IGNORECASE)

        if beat_match:
            team1_raw = clean_team_name(beat_match.group(1))
            team2_raw = clean_team_name(beat_match.group(2))

            team1 = validate_team_name(team1_raw)
            team2 = validate_team_name(team2_raw)

            if team1 and team2:
                return (team2, team1)

        logger.warning(f"Could not extract teams from title: '{title}'")
        return (None, None)

    def get_active_markets(self, market_type: Optional[str] = None) -> List[Dict]:
        """
        Get all active markets

        Args:
            market_type: Filter by 'nfl', 'college', or None for all

        Returns:
            List of market dictionaries
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            if market_type:
                query = """
                    SELECT *
                    FROM kalshi_markets
                    WHERE status IN ('open', 'active')
                    AND market_type = %s
                    ORDER BY close_time ASC
                """
                cur.execute(query, (market_type,))
            else:
                query = """
                    SELECT *
                    FROM kalshi_markets
                    WHERE status IN ('open', 'active')
                    ORDER BY close_time ASC
                """
                cur.execute(query)

            markets = cur.fetchall()
            return [dict(m) for m in markets]

        except Exception as e:
            logger.error(f"Error fetching active markets: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def get_markets_with_predictions(self, market_type: Optional[str] = None,
                                     limit: int = 50) -> List[Dict]:
        """
        Get markets with AI predictions, ranked by opportunity

        Args:
            market_type: Filter by 'nfl', 'college', or None for all
            limit: Maximum number of markets to return

        Returns:
            List of market dictionaries with prediction data
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            if market_type:
                query = """
                    SELECT
                        m.id,
                        m.ticker,
                        m.title,
                        m.market_type,
                        m.home_team,
                        m.away_team,
                        m.game_date,
                        m.yes_price,
                        m.no_price,
                        m.volume,
                        m.close_time,
                        p.predicted_outcome,
                        p.confidence_score,
                        p.edge_percentage,
                        p.overall_rank,
                        p.recommended_action,
                        p.recommended_stake_pct,
                        p.reasoning
                    FROM kalshi_markets m
                    LEFT JOIN kalshi_predictions p ON m.id = p.market_id
                    WHERE m.status IN ('open', 'active')
                    AND m.market_type = %s
                    ORDER BY p.overall_rank ASC NULLS LAST
                    LIMIT %s
                """
                cur.execute(query, (market_type, limit))
            else:
                query = """
                    SELECT
                        m.id,
                        m.ticker,
                        m.title,
                        m.market_type,
                        m.home_team,
                        m.away_team,
                        m.game_date,
                        m.yes_price,
                        m.no_price,
                        m.volume,
                        m.close_time,
                        p.predicted_outcome,
                        p.confidence_score,
                        p.edge_percentage,
                        p.overall_rank,
                        p.recommended_action,
                        p.recommended_stake_pct,
                        p.reasoning
                    FROM kalshi_markets m
                    LEFT JOIN kalshi_predictions p ON m.id = p.market_id
                    WHERE m.status IN ('open', 'active')
                    ORDER BY p.overall_rank ASC NULLS LAST
                    LIMIT %s
                """
                cur.execute(query, (limit,))

            markets = cur.fetchall()
            return [dict(m) for m in markets]

        except Exception as e:
            logger.error(f"Error fetching markets with predictions: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def get_top_opportunities(self, limit: int = 20) -> List[Dict]:
        """
        Get top betting opportunities across all markets

        Args:
            limit: Maximum number of opportunities to return

        Returns:
            List of top opportunity dictionaries
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute("""
                SELECT * FROM v_kalshi_top_opportunities
                LIMIT %s
            """, (limit,))

            opportunities = cur.fetchall()
            return [dict(o) for o in opportunities]

        except Exception as e:
            logger.error(f"Error fetching top opportunities: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def store_predictions(self, predictions: List[Dict]) -> int:
        """
        Store AI predictions for markets

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Number of predictions stored
        """
        if not predictions:
            return 0

        conn = self.get_connection()
        cur = conn.cursor()
        stored_count = 0

        try:
            for pred in predictions:
                # Get market_id from ticker
                cur.execute("SELECT id FROM kalshi_markets WHERE ticker = %s", (pred['ticker'],))
                result = cur.fetchone()

                if not result:
                    logger.warning(f"Market not found for ticker: {pred['ticker']}")
                    continue

                market_id = result[0]

                # Debug: Check for values that might overflow
                for key in ['confidence_score', 'edge_percentage', 'value_score', 'liquidity_score',
                           'timing_score', 'matchup_score', 'sentiment_score', 'recommended_stake_pct']:
                    value = pred.get(key)
                    if value is not None and (value > 999.99 or value < -999.99):
                        logger.warning(f"Value overflow for {key} in {pred['ticker']}: {value}")
                        # Cap the value
                        pred[key] = min(max(value, -999.99), 999.99)

                # Store prediction
                cur.execute("""
                    INSERT INTO kalshi_predictions (
                        market_id, ticker,
                        predicted_outcome, confidence_score, edge_percentage,
                        overall_rank, type_rank,
                        value_score, liquidity_score, timing_score,
                        matchup_score, sentiment_score,
                        recommended_action, recommended_stake_pct, max_price,
                        reasoning, key_factors,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (market_id) DO UPDATE SET
                        predicted_outcome = EXCLUDED.predicted_outcome,
                        confidence_score = EXCLUDED.confidence_score,
                        edge_percentage = EXCLUDED.edge_percentage,
                        overall_rank = EXCLUDED.overall_rank,
                        type_rank = EXCLUDED.type_rank,
                        value_score = EXCLUDED.value_score,
                        liquidity_score = EXCLUDED.liquidity_score,
                        timing_score = EXCLUDED.timing_score,
                        matchup_score = EXCLUDED.matchup_score,
                        sentiment_score = EXCLUDED.sentiment_score,
                        recommended_action = EXCLUDED.recommended_action,
                        recommended_stake_pct = EXCLUDED.recommended_stake_pct,
                        max_price = EXCLUDED.max_price,
                        reasoning = EXCLUDED.reasoning,
                        key_factors = EXCLUDED.key_factors,
                        updated_at = NOW()
                """, (
                    market_id, pred['ticker'],
                    pred.get('predicted_outcome'), pred.get('confidence_score'),
                    pred.get('edge_percentage'),
                    pred.get('overall_rank'), pred.get('type_rank'),
                    pred.get('value_score'), pred.get('liquidity_score'),
                    pred.get('timing_score'), pred.get('matchup_score'),
                    pred.get('sentiment_score'),
                    pred.get('recommended_action'), pred.get('recommended_stake_pct'),
                    pred.get('max_price'),
                    pred.get('reasoning'), json.dumps(pred.get('key_factors', []))
                ))

                stored_count += 1

            conn.commit()
            logger.info(f"Stored {stored_count} predictions")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing predictions: {e}")
            raise

        finally:
            cur.close()
            conn.close()

        return stored_count

    def store_price_snapshot(self, ticker: str, yes_price: float, no_price: float,
                            volume: float, open_interest: int):
        """Store a price snapshot for historical charting"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            # Get market_id
            cur.execute("SELECT id FROM kalshi_markets WHERE ticker = %s", (ticker,))
            result = cur.fetchone()

            if not result:
                logger.warning(f"Market not found for ticker: {ticker}")
                return

            market_id = result[0]

            cur.execute("""
                INSERT INTO kalshi_price_history (
                    market_id, ticker, yes_price, no_price,
                    volume, open_interest, snapshot_time
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (market_id, ticker, yes_price, no_price, volume, open_interest))

            conn.commit()

        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing price snapshot: {e}")

        finally:
            cur.close()
            conn.close()

    def log_sync(self, sync_type: str, market_type: str, total: int,
                successful: int, failed: int, duration: int,
                status: str = 'completed', error_msg: Optional[str] = None) -> int:
        """
        Log a sync operation

        Returns:
            Sync log ID
        """
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO kalshi_sync_log (
                    sync_type, market_type, total_processed,
                    successful, failed, duration_seconds,
                    status, error_message, completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id
            """, (sync_type, market_type, total, successful, failed,
                  duration, status, error_msg))

            sync_id = cur.fetchone()[0]
            conn.commit()

            return sync_id

        finally:
            cur.close()
            conn.close()

    def get_markets_by_team(self, team: str, market_type: Optional[str] = None) -> List[Dict]:
        """
        Get all markets involving a specific team

        Args:
            team: Team name
            market_type: Filter by 'nfl', 'college', or None for all

        Returns:
            List of market dictionaries
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            query = """
                SELECT
                    m.*,
                    p.confidence_score,
                    p.edge_percentage,
                    p.recommended_action,
                    p.reasoning
                FROM kalshi_markets m
                LEFT JOIN kalshi_predictions p ON m.id = p.market_id
                WHERE (m.home_team = %s OR m.away_team = %s)
                AND m.status = 'open'
            """

            params = [team, team]

            if market_type:
                query += " AND m.market_type = %s"
                params.append(market_type)

            query += " ORDER BY m.close_time ASC"

            cur.execute(query, params)
            markets = cur.fetchall()

            return [dict(m) for m in markets]

        finally:
            cur.close()
            conn.close()

    def get_markets_closing_soon(self, hours: int = 24, market_type: Optional[str] = None) -> List[Dict]:
        """
        Get markets closing within specified hours

        Args:
            hours: Number of hours to look ahead
            market_type: Filter by 'nfl', 'college', or None for all

        Returns:
            List of market dictionaries
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            query = """
                SELECT
                    m.*,
                    p.confidence_score,
                    p.edge_percentage,
                    p.recommended_action,
                    p.reasoning
                FROM kalshi_markets m
                LEFT JOIN kalshi_predictions p ON m.id = p.market_id
                WHERE m.status = 'open'
                AND m.close_time <= NOW() + INTERVAL '%s hours'
                AND m.close_time > NOW()
            """

            params = [hours]

            if market_type:
                query += " AND m.market_type = %s"
                params.append(market_type)

            query += " ORDER BY m.close_time ASC"

            cur.execute(query, params)
            markets = cur.fetchall()

            return [dict(m) for m in markets]

        finally:
            cur.close()
            conn.close()

    def get_high_confidence_markets(self, min_confidence: float = 75.0,
                                   min_edge: float = 0.0,
                                   market_type: Optional[str] = None) -> List[Dict]:
        """
        Get markets with high AI confidence and positive edge

        Args:
            min_confidence: Minimum confidence score (0-100)
            min_edge: Minimum edge percentage
            market_type: Filter by 'nfl', 'college', or None for all

        Returns:
            List of market dictionaries
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            query = """
                SELECT
                    m.*,
                    p.confidence_score,
                    p.edge_percentage,
                    p.overall_rank,
                    p.recommended_action,
                    p.recommended_stake_pct,
                    p.reasoning
                FROM kalshi_markets m
                INNER JOIN kalshi_predictions p ON m.id = p.market_id
                WHERE m.status IN ('open', 'active')
                AND p.confidence_score >= %s
                AND p.edge_percentage >= %s
            """

            params = [min_confidence, min_edge]

            if market_type:
                query += " AND m.market_type = %s"
                params.append(market_type)

            query += " ORDER BY p.overall_rank ASC"

            cur.execute(query, params)
            markets = cur.fetchall()

            return [dict(m) for m in markets]

        finally:
            cur.close()
            conn.close()

    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            stats = {}

            # Total markets
            cur.execute("SELECT COUNT(*) as total FROM kalshi_markets")
            stats['total_markets'] = cur.fetchone()['total']

            # Active markets
            cur.execute("SELECT COUNT(*) as active FROM kalshi_markets WHERE status IN ('open', 'active')")
            stats['active_markets'] = cur.fetchone()['active']

            # Markets by type
            cur.execute("""
                SELECT market_type, COUNT(*) as count
                FROM kalshi_markets
                WHERE status IN ('open', 'active')
                GROUP BY market_type
            """)
            type_counts = cur.fetchall()
            stats['markets_by_type'] = {row['market_type']: row['count'] for row in type_counts}

            # Total predictions
            cur.execute("SELECT COUNT(*) as total FROM kalshi_predictions")
            stats['total_predictions'] = cur.fetchone()['total']

            # High confidence opportunities
            cur.execute("""
                SELECT COUNT(*) as count
                FROM kalshi_predictions p
                JOIN kalshi_markets m ON p.market_id = m.id
                WHERE p.confidence_score >= 80
                AND m.status = 'open'
            """)
            stats['high_confidence_count'] = cur.fetchone()['count']

            # Average edge
            cur.execute("""
                SELECT AVG(p.edge_percentage) as avg_edge
                FROM kalshi_predictions p
                JOIN kalshi_markets m ON p.market_id = m.id
                WHERE m.status = 'open'
            """)
            result = cur.fetchone()
            stats['avg_edge'] = float(result['avg_edge']) if result['avg_edge'] else 0.0

            # Last sync
            cur.execute("""
                SELECT sync_type, market_type, completed_at
                FROM kalshi_sync_log
                WHERE status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
            """)
            last_sync = cur.fetchone()
            stats['last_sync'] = dict(last_sync) if last_sync else None

            return stats

        finally:
            cur.close()
            conn.close()

    def store_market_prices(self, ticker: str, prices: Dict) -> bool:
        """
        Store market prices from orderbook data

        Args:
            ticker: Market ticker
            prices: Price data from Kalshi API (orderbook response)

        Returns:
            True if successful, False otherwise
        """
        conn = None
        cur = None
        try:
            # Extract yes/no prices from orderbook
            # Kalshi orderbook format: {'yes': [{'price': cents}], 'no': [{'price': cents}]}
            yes_orders = prices.get('yes', [])
            no_orders = prices.get('no', [])

            if not yes_orders or not no_orders:
                logger.warning(f"Empty orderbook for {ticker}")
                return False

            # Get best bid prices (first element in array)
            yes_price = yes_orders[0].get('price', 0) / 100 if yes_orders else None
            no_price = no_orders[0].get('price', 0) / 100 if no_orders else None

            if yes_price is None or no_price is None:
                return False

            # Update market prices
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE kalshi_markets
                SET yes_price = %s,
                    no_price = %s,
                    last_updated = NOW()
                WHERE ticker = %s
            """, (yes_price, no_price, ticker))

            # Also store in price history
            cur.execute("""
                INSERT INTO kalshi_price_history (
                    market_id, ticker, yes_price, no_price,
                    volume, open_interest, snapshot_time
                )
                SELECT id, %s, %s, %s, volume, open_interest, NOW()
                FROM kalshi_markets
                WHERE ticker = %s
            """, (ticker, yes_price, no_price, ticker))

            conn.commit()
            return cur.rowcount > 0

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error storing market prices for {ticker}: {e}")
            return False
        finally:
            if cur:
                cur.close()
            if conn:
                self.release_connection(conn)


if __name__ == "__main__":
    # Test database manager
    db = KalshiDBManager()

    print("\n" + "="*80)
    print("KALSHI DATABASE MANAGER - Test")
    print("="*80)

    # Get stats
    stats = db.get_stats()
    print("\nDatabase Stats:")
    print(f"  Total Markets: {stats['total_markets']}")
    print(f"  Active Markets: {stats['active_markets']}")
    print(f"  Markets by Type: {stats['markets_by_type']}")
    print(f"  Total Predictions: {stats['total_predictions']}")

    if stats['last_sync']:
        print(f"\nLast Sync:")
        print(f"  Type: {stats['last_sync']['sync_type']}")
        print(f"  Market Type: {stats['last_sync']['market_type']}")
        print(f"  Completed: {stats['last_sync']['completed_at']}")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
