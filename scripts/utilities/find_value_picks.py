"""Find value picks with different filter settings"""
from src.kalshi_db_manager import KalshiDBManager
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent

print("Searching for value picks...")
kalshi_db = KalshiDBManager()
ai_agent = AdvancedBettingAIAgent()

# Get NFL markets
markets = kalshi_db.get_active_markets(market_type='nfl')
print(f"Total NFL markets: {len(markets)}")

# Stats
stats = {
    'total': 0,
    'lopsided_85': 0,
    'lopsided_80': 0,
    'good_odds': 0,
    'ev_5_plus': 0,
    'ev_3_plus': 0,
    'ev_1_plus': 0,
}

value_picks = []

for market in markets[:100]:  # Check first 100
    stats['total'] += 1

    home_team = market.get('home_team', '')
    away_team = market.get('away_team', '')

    if not home_team or not away_team:
        continue

    try:
        yes_price = float(market.get('yes_price') or 50)
        no_price = float(market.get('no_price') or 50)
    except:
        continue

    # Check lopsided at 85
    if yes_price > 85 or yes_price < 15 or no_price > 85 or no_price < 15:
        stats['lopsided_85'] += 1
        continue

    # Check lopsided at 80
    if yes_price > 80 or yes_price < 20 or no_price > 80 or no_price < 20:
        stats['lopsided_80'] += 1

    stats['good_odds'] += 1

    # Analyze
    try:
        game_data = {
            'id': market.get('id'),
            'away_team': away_team,
            'home_team': home_team,
            'away_score': 0,
            'home_score': 0,
            'status': 'scheduled',
            'is_live': False
        }

        market_data = {
            'yes_price': yes_price,
            'no_price': no_price,
            'volume': float(market.get('volume') or 0),
            'ticker': market.get('ticker', ''),
            'title': market.get('title', '')
        }

        prediction = ai_agent.analyze_betting_opportunity(game_data, market_data)
        ev = prediction.get('expected_value', 0)
        conf = prediction.get('confidence_score', 0)

        if ev >= 5.0:
            stats['ev_5_plus'] += 1
            value_picks.append((ev, conf, away_team, home_team, yes_price))
        elif ev >= 3.0:
            stats['ev_3_plus'] += 1
            value_picks.append((ev, conf, away_team, home_team, yes_price))
        elif ev >= 1.0:
            stats['ev_1_plus'] += 1

    except Exception as e:
        pass

print(f"\n{'='*60}")
print("Statistics from first 100 markets:")
print(f"  Total checked: {stats['total']}")
print(f"  Lopsided (>85/<15): {stats['lopsided_85']}")
print(f"  Lopsided (>80/<20): {stats['lopsided_80']}")
print(f"  Good odds (15-85): {stats['good_odds']}")
print(f"  EV >= 5%: {stats['ev_5_plus']}")
print(f"  EV >= 3%: {stats['ev_3_plus']}")
print(f"  EV >= 1%: {stats['ev_1_plus']}")

if value_picks:
    print(f"\nTop value picks found:")
    value_picks.sort(reverse=True)
    for ev, conf, away, home, odds in value_picks[:5]:
        print(f"  EV={ev:.1f}% Conf={conf:.0f}% {away} @ {home} (odds: {odds:.0f}c)")
else:
    print("\n  No value picks found in first 100 markets")
