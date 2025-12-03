"""
ESPN to Kalshi Odds Matcher
Matches ESPN live games to Kalshi prediction markets
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import psycopg2.extras
from src.kalshi_db_manager import KalshiDBManager

logger = logging.getLogger(__name__)


class ESPNKalshiMatcher:
    """
    Match ESPN live games to Kalshi prediction markets
    """

    # NFL team name variations mapping
    NFL_TEAM_VARIATIONS = {
        'Arizona Cardinals': ['Arizona', 'Cardinals', 'ARI'],
        'Atlanta Falcons': ['Atlanta', 'Falcons', 'ATL'],
        'Baltimore Ravens': ['Baltimore', 'Ravens', 'BAL'],
        'Buffalo Bills': ['Buffalo', 'Bills', 'BUF'],
        'Carolina Panthers': ['Carolina', 'Panthers', 'CAR'],
        'Chicago Bears': ['Chicago', 'Bears', 'CHI'],
        'Cincinnati Bengals': ['Cincinnati', 'Bengals', 'CIN'],
        'Cleveland Browns': ['Cleveland', 'Browns', 'CLE'],
        'Dallas Cowboys': ['Dallas', 'Cowboys', 'DAL'],
        'Denver Broncos': ['Denver', 'Broncos', 'DEN'],
        'Detroit Lions': ['Detroit', 'Lions', 'DET'],
        'Green Bay Packers': ['Green Bay', 'Packers', 'GB'],
        'Houston Texans': ['Houston', 'Texans', 'HOU'],
        'Indianapolis Colts': ['Indianapolis', 'Colts', 'IND'],
        'Jacksonville Jaguars': ['Jacksonville', 'Jaguars', 'JAX', 'Jags'],
        'Kansas City Chiefs': ['Kansas City', 'Chiefs', 'KC'],
        'Las Vegas Raiders': ['Las Vegas', 'Raiders', 'LV', 'Oakland Raiders'],
        'Los Angeles Chargers': ['Los Angeles Chargers', 'LA Chargers', 'Chargers', 'LAC', 'Los Angeles C'],
        'Los Angeles Rams': ['Los Angeles Rams', 'LA Rams', 'Rams', 'LAR', 'Los Angeles R'],
        'Miami Dolphins': ['Miami', 'Dolphins', 'MIA'],
        'Minnesota Vikings': ['Minnesota', 'Vikings', 'MIN'],
        'New England Patriots': ['New England', 'Patriots', 'NE'],
        'New Orleans Saints': ['New Orleans', 'Saints', 'NO'],
        'New York Giants': ['New York Giants', 'NY Giants', 'Giants', 'NYG'],
        'New York Jets': ['New York Jets', 'NY Jets', 'Jets', 'NYJ'],
        'Philadelphia Eagles': ['Philadelphia', 'Eagles', 'PHI'],
        'Pittsburgh Steelers': ['Pittsburgh', 'Steelers', 'PIT'],
        'San Francisco 49ers': ['San Francisco', '49ers', 'SF'],
        'Seattle Seahawks': ['Seattle', 'Seahawks', 'SEA'],
        'Tampa Bay Buccaneers': ['Tampa Bay', 'Buccaneers', 'Bucs', 'TB'],
        'Tennessee Titans': ['Tennessee', 'Titans', 'TEN'],
        'Washington Commanders': ['Washington', 'Commanders', 'WAS', 'Washington Football Team']
    }

    # Common CFB team name variations
    CFB_TEAM_VARIATIONS = {
        'Alabama Crimson Tide': ['Alabama', 'Crimson Tide', 'Bama'],
        'Ohio State Buckeyes': ['Ohio State', 'Buckeyes', 'OSU'],
        'Georgia Bulldogs': ['Georgia', 'Bulldogs', 'UGA'],
        'Michigan Wolverines': ['Michigan', 'Wolverines'],
        'Clemson Tigers': ['Clemson', 'Tigers'],
        'Texas Longhorns': ['Texas', 'Longhorns', 'UT'],
        'Oklahoma Sooners': ['Oklahoma', 'Sooners', 'OU'],
        'Notre Dame Fighting Irish': ['Notre Dame', 'Fighting Irish', 'ND'],
        'USC Trojans': ['USC', 'Trojans', 'Southern California'],
        'Penn State Nittany Lions': ['Penn State', 'Nittany Lions', 'PSU'],
        'Florida Gators': ['Florida', 'Gators', 'UF'],
        'LSU Tigers': ['LSU', 'Tigers', 'Louisiana State'],
        'Auburn Tigers': ['Auburn', 'Tigers'],
        'Oregon Ducks': ['Oregon', 'Ducks'],
        'Tennessee Volunteers': ['Tennessee', 'Volunteers', 'Vols'],
        'Texas A&M Aggies': ['Texas A&M', 'Aggies', 'TAMU'],
        'Wisconsin Badgers': ['Wisconsin', 'Badgers'],
        'Miami Hurricanes': ['Miami', 'Hurricanes', 'The U'],
        'Iowa Hawkeyes': ['Iowa', 'Hawkeyes']
    }

    def __init__(self) -> None:
        self.db = KalshiDBManager()
        # Combine all team variations
        self.all_team_variations = {**self.NFL_TEAM_VARIATIONS, **self.CFB_TEAM_VARIATIONS}

    def get_team_variations(self, team_name: str) -> List[str]:
        """
        Get all possible variations of a team name

        Args:
            team_name: Original team name from ESPN

        Returns:
            List of variations to try matching
        """
        # Start with the original name
        variations = [team_name]

        # Check if we have predefined variations
        for full_name, var_list in self.all_team_variations.items():
            # If ESPN name matches any variation, return all variations
            if team_name.lower() in [v.lower() for v in var_list] or \
               full_name.lower() == team_name.lower():
                return var_list + [team_name, full_name]

        # If no predefined variations, generate common ones
        parts = team_name.split()
        if len(parts) >= 2:
            variations.append(parts[-1])  # Just mascot (e.g., "Bills")
            variations.append(' '.join(parts[:-1]))  # Just city (e.g., "Buffalo")

        return list(set(variations))  # Remove duplicates

    def match_game_to_kalshi(self, espn_game: Dict) -> Optional[Dict]:
        """
        Find Kalshi market odds for an ESPN game

        Args:
            espn_game: ESPN game dict with away_team, home_team, game_time

        Returns:
            Dict with away_win_price, home_win_price, volume, ticker
        """
        away_team = espn_game.get('away_team', '')
        home_team = espn_game.get('home_team', '')
        game_time = espn_game.get('game_time', '')

        if not away_team or not home_team:
            return None

        # Get all team name variations
        away_variations = self.get_team_variations(away_team)
        home_variations = self.get_team_variations(home_team)

        # Extract date from game_time (can be datetime object or string)
        try:
            if game_time:
                if isinstance(game_time, datetime):
                    game_date = game_time.date()
                elif isinstance(game_time, str):
                    game_date = datetime.strptime(game_time[:10], '%Y-%m-%d').date()
                else:
                    game_date = datetime.now().date()
            else:
                # If no game time, use today
                game_date = datetime.now().date()
        except Exception as e:
            logger.warning(f"Could not parse game_time {game_time}: {e}")
            game_date = datetime.now().date()

        # Query Kalshi database for matching market
        conn = None
        cur = None
        try:
            conn = self.db.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Search within +/- 3 days of game time
            date_start = game_date - timedelta(days=3)
            date_end = game_date + timedelta(days=3)

            # Try matching with team name variations
            result = None
            for away_var in away_variations:
                for home_var in home_variations:
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
                            OR ticker LIKE 'KXNFLGAME%%'
                            OR ticker LIKE 'KXNCAAFGAME%%'
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

                    cur.execute(query, (
                        f'%{away_var}%',
                        f'%{home_var}%',
                        date_start,  # For expected_expiration_time start
                        date_end,    # For expected_expiration_time end
                        date_start,  # For close_time start (fallback)
                        date_end     # For close_time end (fallback)
                    ))

                    result = cur.fetchone()
                    if result:
                        logger.debug(f"Matched using variations: {away_var} vs {home_var}")
                        break

                if result:
                    break

            if result:
                # Determine which team is "yes" and which is "no"
                ticker = result['ticker']
                title = result['title'].lower()

                # Parse ticker to find which team is the "yes" option
                # Kalshi tickers end with team abbreviation: KXNFLGAME-25NOV17DALLV-DAL or -LV
                ticker_suffix = ticker.split('-')[-1].lower() if '-' in ticker else ''

                # Get team name variations for matching
                away_variations = [v.lower() for v in self.get_team_variations(away_team)]
                home_variations = [v.lower() for v in self.get_team_variations(home_team)]

                # Check if ticker suffix matches away team or home team
                away_is_yes = False
                if ticker_suffix:
                    # Check if suffix matches any away team variation
                    for var in away_variations:
                        if ticker_suffix == var.lower() or ticker_suffix in var.lower():
                            away_is_yes = True
                            break

                    # If not matched to away, check home team
                    if not away_is_yes:
                        for var in home_variations:
                            if ticker_suffix == var.lower() or ticker_suffix in var.lower():
                                away_is_yes = False
                                break

                if away_is_yes:
                    away_price = result['yes_price']
                    home_price = result['no_price']
                else:
                    away_price = result['no_price']
                    home_price = result['yes_price']

                return {
                    'away_win_price': away_price,
                    'home_win_price': home_price,
                    'volume': result['volume'],
                    'ticker': result['ticker'],
                    'market_title': result['title'],
                    'close_time': result['close_time']
                }

            return None

        except Exception as e:
            logger.error(f"Error matching game to Kalshi: {e}", exc_info=True)
            return None
        finally:
            if cur:
                cur.close()
            if conn:
                self.db.release_connection(conn)

    def enrich_espn_games_with_kalshi(self, espn_games: List[Dict]) -> List[Dict]:
        """
        Add Kalshi odds to all ESPN games

        Args:
            espn_games: List of ESPN game dicts

        Returns:
            Same list with kalshi_odds added to each game
        """
        enriched_games = []

        for game in espn_games:
            kalshi_odds = self.match_game_to_kalshi(game)

            if kalshi_odds:
                game['kalshi_odds'] = kalshi_odds
                logger.info(f"Matched Kalshi odds for {game['away_team']} @ {game['home_team']}: "
                          f"{kalshi_odds['away_win_price']:.2%} / {kalshi_odds['home_win_price']:.2%}")
            else:
                game['kalshi_odds'] = None
                logger.debug(f"No Kalshi market found for {game['away_team']} @ {game['home_team']}")

            enriched_games.append(game)

        matched_count = sum(1 for g in enriched_games if g.get('kalshi_odds'))
        logger.info(f"Enriched {len(espn_games)} ESPN games with Kalshi odds. "
                   f"Matched: {matched_count}, Unmatched: {len(espn_games) - matched_count}")

        return enriched_games

    def get_team_name_variations(self, team_name: str) -> List[str]:
        """
        Generate common variations of team names for fuzzy matching

        Examples:
            "Buffalo Bills" -> ["Buffalo Bills", "Bills", "Buffalo"]
            "Ohio State Buckeyes" -> ["Ohio State Buckeyes", "Buckeyes", "Ohio State", "OSU"]
        """
        variations = [team_name]

        # Split team name
        parts = team_name.split()

        if len(parts) >= 2:
            # Add just the mascot (last word)
            variations.append(parts[-1])

            # Add everything except mascot
            variations.append(' '.join(parts[:-1]))

            # Common abbreviations for college teams
            if len(parts) >= 2:
                # Ohio State -> OSU
                if len(parts) == 2:
                    abbrev = ''.join([p[0] for p in parts]).upper()
                    variations.append(abbrev)

        return variations

    def search_kalshi_markets(self, search_term: str, limit: int = 10) -> List[Dict]:
        """
        Search Kalshi markets by keyword

        Args:
            search_term: Team name or search query
            limit: Max number of results

        Returns:
            List of matching markets
        """
        conn = None
        cur = None
        try:
            conn = self.db.get_connection()
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
                status
            FROM kalshi_markets
            WHERE title ILIKE %s
              AND status != 'closed'
            ORDER BY volume DESC, close_time ASC
            LIMIT %s
            """

            cur.execute(query, (f'%{search_term}%', limit))
            results = cur.fetchall()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error searching Kalshi markets: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()


# Convenience function for easy importing
def enrich_games_with_kalshi_odds(espn_games: List[Dict]) -> List[Dict]:
    """
    Quick function to enrich ESPN games with Kalshi odds

    Usage:
        from src.espn_kalshi_matcher import enrich_games_with_kalshi_odds

        espn_games = get_espn_games()
        games_with_odds = enrich_games_with_kalshi_odds(espn_games)
    """
    matcher = ESPNKalshiMatcher()
    return matcher.enrich_espn_games_with_kalshi(espn_games)


def enrich_games_with_kalshi_odds_nba(nba_games: List[Dict]) -> List[Dict]:
    """
    Enrich NBA games with Kalshi odds

    Args:
        nba_games: List of NBA game dicts from ESPN

    Returns:
        Same list with kalshi_odds added
    """
    from src.kalshi_db_manager import KalshiDBManager

    db = KalshiDBManager()
    conn = None
    cur = None

    try:
        conn = db.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        enriched = []
        for game in nba_games:
            away_team = game.get('away_team', '')
            home_team = game.get('home_team', '')

            # Query for NBA Kalshi markets for this matchup
            # NBA markets use format like: KXNBAGAME-25NOV19NYKDAL-NYK
            cur.execute("""
                SELECT *
                FROM kalshi_markets
                WHERE status = 'active'
                AND ticker LIKE 'KXNBAGAME%%'
                AND (title ILIKE %s OR title ILIKE %s)
                ORDER BY volume DESC NULLS LAST
                LIMIT 1
            """, (f'%{away_team}%', f'%{home_team}%'))

            market = cur.fetchone()

            if market and market.get('yes_price') is not None:
                # Determine which team is "yes" based on ticker suffix
                ticker = market['ticker']
                ticker_suffix = ticker.split('-')[-1].lower() if '-' in ticker else ''

                # Simple matching: if suffix contains team abbreviation, that's the "yes" team
                away_abbr = game.get('away_abbr', '').lower()
                home_abbr = game.get('home_abbr', '').lower()

                # Check which team is "yes"
                if ticker_suffix == away_abbr or away_abbr in ticker_suffix:
                    away_price = float(market['yes_price']) if market['yes_price'] else 0
                    home_price = float(market['no_price']) if market['no_price'] else 0
                else:
                    away_price = float(market['no_price']) if market['no_price'] else 0
                    home_price = float(market['yes_price']) if market['yes_price'] else 0

                game['kalshi_odds'] = {
                    'away_win_price': away_price,
                    'home_win_price': home_price,
                    'ticker': market['ticker'],
                    'title': market['title'],
                    'volume': market.get('volume', 0)
                }

            enriched.append(game)

        return enriched

    except Exception as e:
        logger.error(f"Error enriching NBA games with Kalshi odds: {e}")
        return nba_games

    finally:
        if cur:
            cur.close()
        if conn:
            db.release_connection(conn)


if __name__ == "__main__":
    # Test matching
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    from src.espn_live_data import get_espn_client
    from src.espn_ncaa_live_data import get_espn_ncaa_client

    print("Testing ESPN to Kalshi Matcher")
    print("=" * 60)

    # Get ESPN games
    espn = get_espn_client()
    nfl_games = espn.get_live_scores()

    print(f"\nFound {len(nfl_games)} NFL games")

    # Enrich with Kalshi
    matcher = ESPNKalshiMatcher()
    enriched = matcher.enrich_espn_games_with_kalshi(nfl_games[:5])  # Test first 5

    print("\nResults:")
    for game in enriched:
        odds = game.get('kalshi_odds')
        if odds:
            print(f"✅ {game['away_team']} @ {game['home_team']}")
            print(f"   Away: {odds['away_win_price']:.1%}, Home: {odds['home_win_price']:.1%}")
            print(f"   Market: {odds['ticker']}")
        else:
            print(f"❌ {game['away_team']} @ {game['home_team']} - No Kalshi market found")
        print()
