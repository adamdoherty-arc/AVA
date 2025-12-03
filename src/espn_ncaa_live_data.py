"""
ESPN NCAA Football Live Game Data Integration
Fetches real-time scores and game status for college football
"""

import requests
import logging
from typing import Dict, Optional, List
from datetime import datetime
from src.espn_rate_limiter import rate_limited

logger = logging.getLogger(__name__)


class ESPNNCAALiveData:
    """Fetch live NCAA football game data from ESPN"""

    # ESPN College Football API endpoint
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    @rate_limited
    def get_scoreboard(self, group: Optional[str] = None, week: Optional[int] = None) -> List[Dict]:
        """
        Get current NCAA football scoreboard

        Args:
            group: Conference group (e.g., '80' for FBS, '81' for FCS)
            week: Week number (optional, defaults to current week)

        Returns:
            List of game dictionaries with scores and status
        """
        try:
            url = f"{self.BASE_URL}/scoreboard"

            params = {}
            if group:
                params['groups'] = group
            if week:
                params['week'] = week

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            games = []
            for event in data.get('events', []):
                game = self._parse_game(event)
                if game:
                    games.append(game)

            logger.info(f"Fetched {len(games)} NCAA games from ESPN scoreboard")
            return games

        except requests.RequestException as e:
            logger.error(f"Error fetching ESPN NCAA scoreboard: {e}")
            return []

    def get_top_25_games(self) -> List[Dict]:
        """Get games involving Top 25 teams"""
        all_games = self.get_scoreboard()

        # Filter for games with ranked teams
        top_25_games = []
        for game in all_games:
            home_rank = game.get('home_rank')
            away_rank = game.get('away_rank')

            if home_rank or away_rank:
                top_25_games.append(game)

        # Sort by highest ranked team
        top_25_games.sort(key=lambda g: min(
            g.get('home_rank', 999),
            g.get('away_rank', 999)
        ))

        return top_25_games

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

            # Extract team data with NCAA-specific fields
            home_team_data = home_team.get('team', {})
            away_team_data = away_team.get('team', {})

            home_name = home_team_data.get('displayName', '')
            home_abbr = home_team_data.get('abbreviation', '')
            home_score = int(home_team.get('score', 0))
            home_logo = home_team_data.get('logo', '')
            home_id = home_team_data.get('id', '')
            home_color = home_team_data.get('color', '')
            home_rank = home_team.get('curatedRank', {}).get('current')  # AP Poll ranking
            home_record = home_team.get('records', [{}])[0].get('summary', '') if home_team.get('records') else ''

            away_name = away_team_data.get('displayName', '')
            away_abbr = away_team_data.get('abbreviation', '')
            away_score = int(away_team.get('score', 0))
            away_logo = away_team_data.get('logo', '')
            away_id = away_team_data.get('id', '')
            away_color = away_team_data.get('color', '')
            away_rank = away_team.get('curatedRank', {}).get('current')
            away_record = away_team.get('records', [{}])[0].get('summary', '') if away_team.get('records') else ''

            # Game time
            game_date = event.get('date')
            if game_date:
                game_datetime = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
            else:
                game_datetime = None

            # Venue info (useful for NCAA)
            venue = competition.get('venue', {})
            venue_name = venue.get('fullName', '')

            # Conference info (useful for NCAA)
            home_conference = home_team_data.get('conferenceId', '')
            away_conference = away_team_data.get('conferenceId', '')

            # Determine if live
            is_live = status_type in ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME']

            # TV broadcast info
            broadcast = competition.get('broadcasts', [])
            tv_network = broadcast[0].get('names', [''])[0] if broadcast else ''

            # Extract situation data (possession, down & distance) - only available during live games
            situation = competition.get('situation', {})
            possession = situation.get('possession')  # Team ID that has possession
            down_distance_text = situation.get('downDistanceText', '')  # e.g., "1st & 10"
            short_down_distance = situation.get('shortDownDistanceText', '')  # e.g., "1st & 10"
            possession_text = situation.get('possessionText', '')  # e.g., "ALA"
            is_red_zone = situation.get('isRedZone', False)
            home_timeouts = situation.get('homeTimeouts', 3)
            away_timeouts = situation.get('awayTimeouts', 3)
            last_play = situation.get('lastPlay', {})
            last_play_text = last_play.get('text', '') if last_play else ''

            # Extract leaders (passing, rushing, receiving)
            leaders_data = competition.get('leaders', [])
            passing_leader = None
            rushing_leader = None
            receiving_leader = None

            for leader_category in leaders_data:
                category_name = leader_category.get('name', '')
                leaders_list = leader_category.get('leaders', [])

                if leaders_list:
                    leader_info = leaders_list[0]  # Get top leader
                    athlete = leader_info.get('athlete', {})
                    display_value = leader_info.get('displayValue', '')
                    team = leader_info.get('team', {})
                    team_abbr = team.get('abbreviation', '') if team else ''

                    leader_dict = {
                        'name': athlete.get('shortName', ''),
                        'stats': display_value,
                        'team': team_abbr
                    }

                    if category_name == 'passingYards':
                        passing_leader = leader_dict
                    elif category_name == 'rushingYards':
                        rushing_leader = leader_dict
                    elif category_name == 'receivingYards':
                        receiving_leader = leader_dict

            # Extract venue city
            venue_city = venue.get('address', {}).get('city', '')

            # Extract notes (injuries, weather, etc.)
            notes = competition.get('notes', [])
            game_notes = [note.get('headline', '') for note in notes] if notes else []

            # Extract headlines
            headlines = competition.get('headlines', [])
            headline_text = headlines[0].get('shortLinkText', '') if headlines else ''

            return {
                'game_id': game_id,
                'name': name,
                'short_name': short_name,
                'home_team': home_name,
                'home_abbr': home_abbr,
                'home_score': home_score,
                'home_logo': home_logo,
                'home_id': home_id,
                'home_color': home_color,
                'home_rank': home_rank,
                'home_record': home_record,
                'home_conference': home_conference,
                'away_team': away_name,
                'away_abbr': away_abbr,
                'away_score': away_score,
                'away_logo': away_logo,
                'away_id': away_id,
                'away_color': away_color,
                'away_rank': away_rank,
                'away_record': away_record,
                'away_conference': away_conference,
                'status': status_type,
                'status_detail': status_detail,
                'is_live': is_live,
                'is_completed': is_completed,
                'clock': clock,
                'period': period,
                'game_time': game_datetime,
                'venue': venue_name,
                'tv_network': tv_network,
                # Enhanced live game data
                'possession': possession_text,
                'down_distance': short_down_distance or down_distance_text,
                'is_red_zone': is_red_zone,
                'home_timeouts': home_timeouts,
                'away_timeouts': away_timeouts,
                'last_play': last_play_text,
                'passing_leader': passing_leader,
                'rushing_leader': rushing_leader,
                'receiving_leader': receiving_leader,
                'venue_city': venue_city,
                'broadcast': tv_network,
                'notes': game_notes,
                'headline': headline_text
            }

        except Exception as e:
            logger.error(f"Error parsing NCAA game: {e}")
            return None

    def get_game_status(self, team_name: str) -> Optional[Dict]:
        """
        Get live status for a specific team's game

        Args:
            team_name: Team name (e.g., "Alabama", "Ohio State")

        Returns:
            Game dictionary if found, None otherwise
        """
        games = self.get_scoreboard()

        for game in games:
            if team_name.lower() in game['home_team'].lower() or \
               team_name.lower() in game['away_team'].lower():
                return game

        return None

    def get_live_games(self) -> List[Dict]:
        """Get only games currently in progress"""
        all_games = self.get_scoreboard()
        return [g for g in all_games if g.get('is_live', False)]

    def get_conference_games(self, conference_id: str) -> List[Dict]:
        """
        Get games for a specific conference

        Args:
            conference_id: ESPN conference ID (e.g., '5' for SEC, '4' for Big Ten)

        Returns:
            List of games involving teams from that conference
        """
        all_games = self.get_scoreboard()

        return [
            g for g in all_games
            if g.get('home_conference') == conference_id or
               g.get('away_conference') == conference_id
        ]


# Conference IDs for reference
NCAA_CONFERENCES = {
    'sec': '5',
    'big_ten': '4',
    'big_12': '12',
    'acc': '1',
    'pac_12': '9',
    'aac': '151',
    'sun_belt': '37',
    'mac': '15',
    'mountain_west': '17',
    'cusa': '12',
    'independent': '18'
}


# Singleton instance
_espn_ncaa_client = None


def get_espn_ncaa_client() -> ESPNNCAALiveData:
    """Get or create ESPN NCAA client singleton"""
    global _espn_ncaa_client
    if _espn_ncaa_client is None:
        _espn_ncaa_client = ESPNNCAALiveData()
    return _espn_ncaa_client


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    client = ESPNNCAALiveData()

    print("="*80)
    print("ESPN NCAA FOOTBALL SCOREBOARD")
    print("="*80)

    print("\nFetching all FBS games...")
    games = client.get_scoreboard(group='80')  # 80 = FBS

    print(f"\nFound {len(games)} games:\n")

    for game in games:
        # Show ranking if available
        away_display = f"#{game['away_rank']} {game['away_abbr']}" if game.get('away_rank') else game['away_abbr']
        home_display = f"#{game['home_rank']} {game['home_abbr']}" if game.get('home_rank') else game['home_abbr']

        print(f"{away_display} @ {home_display}")
        print(f"  Score: {game['away_score']} - {game['home_score']}")
        print(f"  Status: {game['status_detail']}")

        if game['is_live']:
            print(f"  [LIVE] - {game['clock']} Q{game['period']}")

        if game.get('tv_network'):
            print(f"  TV: {game['tv_network']}")

        if game.get('venue'):
            print(f"  Venue: {game['venue']}")

        print()

    # Show Top 25 games
    print("\n" + "="*80)
    print("TOP 25 MATCHUPS")
    print("="*80)

    top_25 = client.get_top_25_games()

    if top_25:
        for i, game in enumerate(top_25[:10], 1):
            away_display = f"#{game['away_rank']} {game['away_team']}" if game.get('away_rank') else game['away_team']
            home_display = f"#{game['home_rank']} {game['home_team']}" if game.get('home_rank') else game['home_team']

            print(f"\n{i}. {away_display} @ {home_display}")
            print(f"   Score: {game['away_score']} - {game['home_score']}")
            print(f"   Status: {game['status_detail']}")

            if game['is_live']:
                print(f"   🔴 LIVE")
    else:
        print("\nNo Top 25 games found")

    print("\n" + "="*80)
