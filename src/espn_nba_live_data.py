"""
ESPN NBA Live Game Data Integration
Fetches real-time scores and game status for NBA basketball
"""

import requests
import logging
from typing import Dict, Optional, List
from datetime import datetime
from src.espn_rate_limiter import rate_limited

logger = logging.getLogger(__name__)


class ESPNNBALiveData:
    """Fetch live NBA game data from ESPN"""

    # ESPN NBA API endpoint
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    @rate_limited
    def get_scoreboard(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get current NBA scoreboard

        Args:
            date: Date in YYYYMMDD format (optional, defaults to current date)

        Returns:
            List of game dictionaries with scores and status
        """
        try:
            url = f"{self.BASE_URL}/scoreboard"
            
            params = {}
            if date:
                params['dates'] = date
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            games = []
            for event in data.get('events', []):
                game = self._parse_game(event)
                if game:
                    games.append(game)
            
            logger.info(f"Fetched {len(games)} NBA games from ESPN scoreboard")
            return games
        
        except requests.RequestException as e:
            logger.error(f"Error fetching ESPN NBA scoreboard: {e}")
            return []

    def get_todays_games(self) -> List[Dict]:
        """Get today's NBA games"""
        return self.get_scoreboard()

    def get_live_games(self) -> List[Dict]:
        """Get currently live NBA games"""
        all_games = self.get_scoreboard()
        return [game for game in all_games if game.get('is_live', False)]

    def _parse_game(self, event: Dict) -> Optional[Dict]:
        """Parse game data from ESPN event"""
        
        try:
            # Extract basic info
            game_id = event.get('id')
            name = event.get('name', '')
            short_name = event.get('shortName', '')
            
            # Status
            status = event.get('status', {})
            status_type = status.get('type', {}).get('name', 'Unknown')
            status_detail = status.get('type', {}).get('shortDetail', '')
            is_completed = status.get('type', {}).get('completed', False)
            is_in_progress = status.get('type', {}).get('state', '') == 'in'
            clock = status.get('displayClock', '0:00')
            period = status.get('period', 0)
            
            # Teams (home and away)
            competitions = event.get('competitions', [])
            if not competitions:
                return None
            
            competition = competitions[0]
            competitors = competition.get('competitors', [])
            
            if len(competitors) < 2:
                return None
            
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            
            if not home_team or not away_team:
                return None
            
            # Extract team data
            home_name = home_team.get('team', {}).get('displayName', '')
            home_abbr = home_team.get('team', {}).get('abbreviation', '')
            home_score = int(home_team.get('score', 0))
            home_logo = home_team.get('team', {}).get('logo', '')
            home_record = home_team.get('records', [{}])[0].get('summary', '') if home_team.get('records') else ''
            
            away_name = away_team.get('team', {}).get('displayName', '')
            away_abbr = away_team.get('team', {}).get('abbreviation', '')
            away_score = int(away_team.get('score', 0))
            away_logo = away_team.get('team', {}).get('logo', '')
            away_record = away_team.get('records', [{}])[0].get('summary', '') if away_team.get('records') else ''
            
            # Game time
            game_date_str = event.get('date', '')
            game_time = datetime.fromisoformat(game_date_str.replace('Z', '+00:00')) if game_date_str else None
            
            # Venue
            venue = competition.get('venue', {})
            venue_name = venue.get('fullName', '')
            venue_city = venue.get('address', {}).get('city', '')
            
            # Broadcast
            broadcasts = competition.get('broadcasts', [])
            tv_network = broadcasts[0].get('names', [''])[0] if broadcasts else ''
            
            # Build game dictionary
            game_data = {
                'game_id': game_id,
                'name': name,
                'short_name': short_name,
                
                # Teams
                'home_team': home_name,
                'home_abbr': home_abbr,
                'home_score': home_score,
                'home_logo': home_logo,
                'home_record': home_record,
                
                'away_team': away_name,
                'away_abbr': away_abbr,
                'away_score': away_score,
                'away_logo': away_logo,
                'away_record': away_record,
                
                # Status
                'status': status_type,
                'status_detail': status_detail,
                'is_completed': is_completed,
                'is_live': is_in_progress,
                'clock': clock,
                'period': period,
                'quarter': period if period <= 4 else 'OT' + str(period - 4) if period > 4 else str(period),
                
                # Time and venue
                'game_time': game_time.strftime('%Y-%m-%d %H:%M') if game_time else '',
                'venue': venue_name,
                'city': venue_city,
                'tv': tv_network,
                
                # Additional metadata
                'sport': 'NBA',
                'league': 'NBA'
            }
            
            # Add team statistics if available (for live games)
            if is_in_progress:
                home_stats = home_team.get('statistics', [])
                away_stats = away_team.get('statistics', [])
                
                if home_stats:
                    game_data['home_stats'] = self._parse_team_stats(home_stats)
                if away_stats:
                    game_data['away_stats'] = self._parse_team_stats(away_stats)
            
            return game_data
        
        except Exception as e:
            logger.error(f"Error parsing NBA game: {e}")
            return None

    def _parse_team_stats(self, stats_list: List) -> Dict:
        """Parse team statistics from ESPN format"""
        stats = {}
        
        for stat in stats_list:
            name = stat.get('name', '')
            value = stat.get('displayValue', '')
            
            # Map common stats
            stat_mapping = {
                'fieldGoalsMade': 'fg_made',
                'fieldGoalsAttempted': 'fg_attempted',
                'fieldGoalPct': 'fg_pct',
                'threePointFieldGoalsMade': '3pt_made',
                'threePointFieldGoalsAttempted': '3pt_attempted',
                'threePointFieldGoalPct': '3pt_pct',
                'freeThrowsMade': 'ft_made',
                'freeThrowsAttempted': 'ft_attempted',
                'freeThrowPct': 'ft_pct',
                'totalRebounds': 'rebounds',
                'assists': 'assists',
                'steals': 'steals',
                'blocks': 'blocks',
                'turnovers': 'turnovers',
                'totalTurnovers': 'turnovers',
                'fouls': 'fouls',
            }
            
            if name in stat_mapping:
                stats[stat_mapping[name]] = value
        
        return stats

    def get_team_info(self, team_abbr: str) -> Optional[Dict]:
        """
        Get team information
        
        Args:
            team_abbr: Team abbreviation (e.g., 'LAL', 'BOS')
        
        Returns:
            Dictionary with team info or None
        """
        try:
            url = f"{self.BASE_URL}/teams/{team_abbr}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            team = data.get('team', {})
            
            return {
                'id': team.get('id'),
                'name': team.get('displayName'),
                'abbreviation': team.get('abbreviation'),
                'logo': team.get('logos', [{}])[0].get('href', ''),
                'color': team.get('color', ''),
                'alternateColor': team.get('alternateColor', ''),
            }
        
        except requests.RequestException as e:
            logger.error(f"Error fetching team info for {team_abbr}: {e}")
            return None


# Helper function to get ESPN NBA client
def get_espn_nba_client() -> ESPNNBALiveData:
    """Get ESPN NBA client singleton"""
    return ESPNNBALiveData()


# Testing
if __name__ == "__main__":
    print("Testing ESPN NBA Live Data Integration")
    print("=" * 60)
    
    client = ESPNNBALiveData()
    
    # Test 1: Get today's games
    print("\nTest 1: Today's NBA Games")
    games = client.get_todays_games()
    print(f"Found {len(games)} games today")
    
    if games:
        game = games[0]
        print(f"\nSample Game:")
        print(f"  {game['away_team']} ({game['away_record']}) @ {game['home_team']} ({game['home_record']})")
        print(f"  Score: {game['away_score']} - {game['home_score']}")
        print(f"  Status: {game['status_detail']}")
        print(f"  Venue: {game['venue']}, {game['city']}")
        if game.get('tv'):
            print(f"  TV: {game['tv']}")
    
    # Test 2: Get live games
    print("\nTest 2: Live NBA Games")
    live_games = client.get_live_games()
    print(f"Found {len(live_games)} live games")
    
    if live_games:
        for game in live_games[:3]:  # Show up to 3
            print(f"  🏀 {game['away_abbr']} {game['away_score']} - {game['home_score']} {game['home_abbr']} ({game['quarter']} {game['clock']})")
    
    # Test 3: Get team info
    print("\nTest 3: Team Info")
    teams_to_test = ['LAL', 'BOS', 'GSW']
    for team_abbr in teams_to_test:
        team_info = client.get_team_info(team_abbr)
        if team_info:
            print(f"  {team_abbr}: {team_info['name']}")
    
    print("\n" + "=" * 60)
    print("ESPN NBA integration test complete!")

