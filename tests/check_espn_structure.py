from src.espn_live_data import get_espn_client

espn = get_espn_client()
games = espn.get_scoreboard(week=12)

print('Game keys:')
print(list(games[0].keys()))

print('\nGame data sample:')
for k, v in games[0].items():
    print(f'{k}: {v}')
