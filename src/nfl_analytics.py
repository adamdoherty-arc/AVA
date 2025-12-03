"""
NFL Analytics using nfl_data_py
Provides historical and advanced statistics for NFL games and players

Data source: nfl_data_py (nflverse/nfl_data_py on GitHub)
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import - only import nfl_data_py when needed
# This allows the app to run even if nfl_data_py isn't installed yet
try:
    import nfl_data_py as nfl
    NFL_DATA_PY_AVAILABLE = True
except ImportError:
    NFL_DATA_PY_AVAILABLE = False
    logger.warning("nfl_data_py not installed. Run: pip install nfl-data-py")


class NFLAnalytics:
    """
    Enhanced NFL analytics using nfl_data_py

    Provides:
    - Historical play-by-play data
    - Advanced team statistics
    - Player performance metrics
    - Win probability analysis
    - Expected Points Added (EPA)
    """

    def __init__(self) -> None:
        """Initialize NFL analytics with current season"""
        if not NFL_DATA_PY_AVAILABLE:
            logger.error("nfl_data_py is not available. Install it: pip install nfl-data-py")
            return

        self.current_year = datetime.now().year
        logger.info(f"NFL Analytics initialized for {self.current_year} season")

    def is_available(self) -> bool:
        """Check if nfl_data_py is available"""
        return NFL_DATA_PY_AVAILABLE

    def get_team_season_stats(self, team_abbr: str, year: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get season statistics for a team

        Args:
            team_abbr: Team abbreviation (e.g., 'PIT', 'BUF', 'KC')
            year: Season year (defaults to current year)

        Returns:
            DataFrame with weekly team statistics or None if error
        """
        if not self.is_available():
            return None

        year = year or self.current_year

        try:
            # Get weekly data
            weekly = nfl.import_weekly_data([year])

            # Filter by team
            team_stats = weekly[
                (weekly['recent_team'] == team_abbr)
            ].copy()

            if team_stats.empty:
                logger.warning(f"No data found for team {team_abbr} in {year}")
                return None

            logger.info(f"Retrieved {len(team_stats)} weeks of data for {team_abbr}")
            return team_stats

        except Exception as e:
            logger.error(f"Error getting team stats for {team_abbr}: {e}")
            return None

    def get_win_probability_timeline(self, game_id: str, year: Optional[int] = None) -> Optional[List[Dict]]:
        """
        Get play-by-play with win probability for a specific game

        Args:
            game_id: NFL game ID (format: YYYY_WW_AWAY_HOME, e.g., '2025_12_BUF_PIT')
            year: Season year (defaults to current year)

        Returns:
            List of play-by-play records with win probability or None if error
        """
        if not self.is_available():
            return None

        year = year or self.current_year

        try:
            # Get play-by-play data
            pbp = nfl.import_pbp_data([year])

            # Filter by game
            game_plays = pbp[pbp['game_id'] == game_id].copy()

            if game_plays.empty:
                logger.warning(f"No play-by-play data found for game {game_id}")
                return None

            # Extract win probability timeline
            wp_timeline = game_plays[[
                'play_id', 'quarter', 'time', 'down', 'ydstogo',
                'home_wp', 'away_wp', 'home_wp_post', 'away_wp_post',
                'score_differential', 'desc'
            ]].to_dict('records')

            logger.info(f"Retrieved {len(wp_timeline)} plays for game {game_id}")
            return wp_timeline

        except Exception as e:
            logger.error(f"Error getting win probability for game {game_id}: {e}")
            return None

    def get_player_advanced_stats(self, player_name: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Get advanced player statistics

        Args:
            player_name: Player display name (e.g., 'Josh Allen', 'Patrick Mahomes')
            year: Season year (defaults to current year)

        Returns:
            Dictionary with aggregated player stats or None if error
        """
        if not self.is_available():
            return None

        year = year or self.current_year

        try:
            # Get weekly player data
            weekly = nfl.import_weekly_data([year])

            # Search for player
            player_stats = weekly[
                weekly['player_display_name'].str.contains(player_name, na=False, case=False)
            ].copy()

            if player_stats.empty:
                logger.warning(f"No data found for player {player_name} in {year}")
                return None

            # Determine position type
            first_row = player_stats.iloc[0]
            position = first_row.get('position', 'UNKNOWN')

            # Aggregate stats based on position
            if position in ['QB', 'RB', 'WR', 'TE']:
                stats = {
                    'player_name': first_row['player_display_name'],
                    'team': first_row.get('recent_team', 'N/A'),
                    'position': position,
                    'games_played': len(player_stats),

                    # Passing stats (QB)
                    'passing_yards': player_stats['passing_yards'].sum(),
                    'passing_tds': player_stats['passing_tds'].sum(),
                    'interceptions': player_stats['interceptions'].sum(),
                    'passing_epa': player_stats['passing_epa'].mean(),

                    # Rushing stats
                    'rushing_yards': player_stats['rushing_yards'].sum(),
                    'rushing_tds': player_stats['rushing_tds'].sum(),
                    'rushing_epa': player_stats['rushing_epa'].mean(),

                    # Receiving stats (WR/TE/RB)
                    'receiving_yards': player_stats['receiving_yards'].sum(),
                    'receptions': player_stats['receptions'].sum(),
                    'receiving_tds': player_stats['receiving_tds'].sum(),
                    'receiving_epa': player_stats['receiving_epa'].mean(),

                    # Overall EPA
                    'total_epa': (
                        player_stats['passing_epa'].sum() +
                        player_stats['rushing_epa'].sum() +
                        player_stats['receiving_epa'].sum()
                    )
                }
            else:
                # Defensive/other positions
                stats = {
                    'player_name': first_row['player_display_name'],
                    'team': first_row.get('recent_team', 'N/A'),
                    'position': position,
                    'games_played': len(player_stats),
                }

            logger.info(f"Retrieved stats for {player_name} ({position})")
            return stats

        except Exception as e:
            logger.error(f"Error getting player stats for {player_name}: {e}")
            return None

    def get_team_matchup_history(self, team1: str, team2: str, years: int = 5) -> Optional[List[Dict]]:
        """
        Get historical matchup data between two teams

        Args:
            team1: First team abbreviation
            team2: Second team abbreviation
            years: Number of years of history to retrieve (default: 5)

        Returns:
            List of historical matchup results or None if error
        """
        if not self.is_available():
            return None

        try:
            # Get schedule data for recent years
            year_range = list(range(self.current_year - years, self.current_year + 1))
            schedules = nfl.import_schedules(year_range)

            # Filter for matchups between these teams
            matchups = schedules[
                ((schedules['home_team'] == team1) & (schedules['away_team'] == team2)) |
                ((schedules['home_team'] == team2) & (schedules['away_team'] == team1))
            ].copy()

            if matchups.empty:
                logger.warning(f"No matchup history found between {team1} and {team2}")
                return None

            # Format results
            results = []
            for _, game in matchups.iterrows():
                results.append({
                    'game_id': game['game_id'],
                    'season': game['season'],
                    'week': game['week'],
                    'game_date': game.get('gameday', 'N/A'),
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_score': game.get('home_score', 0),
                    'away_score': game.get('away_score', 0),
                    'winner': self._determine_winner(game)
                })

            logger.info(f"Found {len(results)} matchups between {team1} and {team2}")
            return results

        except Exception as e:
            logger.error(f"Error getting matchup history for {team1} vs {team2}: {e}")
            return None

    def _determine_winner(self, game_row: pd.Series) -> str:
        """Determine winner from game row"""
        home_score = game_row.get('home_score', 0)
        away_score = game_row.get('away_score', 0)

        if home_score > away_score:
            return game_row['home_team']
        elif away_score > home_score:
            return game_row['away_team']
        else:
            return 'TIE'

    def get_current_season_schedule(self, team_abbr: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get current season schedule

        Args:
            team_abbr: Optional team abbreviation to filter by

        Returns:
            DataFrame with schedule or None if error
        """
        if not self.is_available():
            return None

        try:
            schedule = nfl.import_schedules([self.current_year])

            if team_abbr:
                schedule = schedule[
                    (schedule['home_team'] == team_abbr) | (schedule['away_team'] == team_abbr)
                ].copy()

            logger.info(f"Retrieved {len(schedule)} games for {team_abbr or 'all teams'}")
            return schedule

        except Exception as e:
            logger.error(f"Error getting schedule: {e}")
            return None


# Global instance
nfl_analytics = NFLAnalytics()
