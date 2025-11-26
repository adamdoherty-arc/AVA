"""
Check NCAA games for Virginia Tech and Miami
"""
from src.espn_ncaa_live_data import get_espn_ncaa_client

print("Fetching NCAA games...")
espn_ncaa = get_espn_ncaa_client()

# Try multiple weeks
for week in [12, 13, 14]:
    print(f"\n{'='*60}")
    print(f"WEEK {week}")
    print('='*60)

    games = espn_ncaa.get_scoreboard(week=week)
    print(f"Total games: {len(games)}")

    # Find Miami and Virginia Tech
    miami_games = [g for g in games if 'Miami' in g.get('home_team', '') or 'Miami' in g.get('away_team', '')]
    vt_games = [g for g in games if 'Virginia Tech' in g.get('home_team', '') or 'Virginia Tech' in g.get('away_team', '')]

    if miami_games:
        print(f"\nMIAMI GAMES ({len(miami_games)}):")
        for g in miami_games:
            print(f"  {g.get('away_team')} @ {g.get('home_team')} - {g.get('status')}")

    if vt_games:
        print(f"\nVIRGINIA TECH GAMES ({len(vt_games)}):")
        for g in vt_games:
            print(f"  {g.get('away_team')} @ {g.get('home_team')} - {g.get('status')}")
