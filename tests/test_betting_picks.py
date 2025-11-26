"""Test betting picks widget"""
from src.components.top_betting_picks_widget import fetch_top_value_picks

# Test NFL picks
print("Testing NFL picks...")
nfl_picks = fetch_top_value_picks(min_ev=5.0, min_confidence=50, limit=10, market_type='nfl')
print(f"NFL picks found: {len(nfl_picks)}")

if nfl_picks:
    top = nfl_picks[0]
    print(f"\nTop NFL pick:")
    print(f"  {top.get('away_team')} @ {top.get('home_team')}")
    print(f"  EV: {top.get('expected_value', 0):.1f}%")
    print(f"  Confidence: {top.get('confidence_score', 0):.0f}%")
    print(f"  Recommendation: {top.get('recommendation')}")

# Test NCAA picks
print("\n" + "="*50)
print("Testing NCAA picks...")
ncaa_picks = fetch_top_value_picks(min_ev=5.0, min_confidence=50, limit=10, market_type='college')
print(f"NCAA picks found: {len(ncaa_picks)}")

if ncaa_picks:
    top = ncaa_picks[0]
    print(f"\nTop NCAA pick:")
    print(f"  {top.get('away_team')} @ {top.get('home_team')}")
    print(f"  EV: {top.get('expected_value', 0):.1f}%")
    print(f"  Confidence: {top.get('confidence_score', 0):.0f}%")
    print(f"  Recommendation: {top.get('recommendation')}")
