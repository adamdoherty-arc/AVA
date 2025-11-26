"""
Test AI predictions to find the 54% issue
"""
import sys
sys.path.insert(0, 'c:\\code\\Magnus')

from src.prediction_agents.nfl_predictor import NFLPredictor
from src.prediction_agents.ncaa_predictor import NCAAPredictor

print("=" * 100)
print("TESTING AI PREDICTIONS - 54% ISSUE")
print("=" * 100)

# Test NCAA predictor
print("\n1. Testing NCAA Predictor...")
print("-" * 100)

ncaa_pred = NCAAPredictor()

test_games = [
    ("Florida State Seminoles", "NC State Wolfpack"),
    ("Texas A&M Aggies", "Samford Bulldogs"),
    ("Ohio State Buckeyes", "Rutgers Scarlet Knights"),
    ("Oklahoma Sooners", "Missouri Tigers"),
    ("Georgia Bulldogs", "Charlotte 49ers"),
]

for away, home in test_games:
    prediction = ncaa_pred.predict_winner(home_team=home, away_team=away)

    if prediction:
        winner = prediction.get('winner', 'Unknown')
        prob = prediction.get('probability', 0)
        conf = prediction.get('confidence', 'unknown')
        spread = prediction.get('spread', 0)

        print(f"\n{away} @ {home}")
        print(f"  Winner: {winner}")
        print(f"  Probability: {prob:.3f} ({prob*100:.1f}%)")
        print(f"  Confidence: {conf}")
        print(f"  Spread: {spread:.1f}")

        # Check if it's the problematic 54%
        if abs(prob - 0.54) < 0.01:
            print(f"  [!] FOUND 54% ISSUE!")
    else:
        print(f"\n{away} @ {home}: No prediction")

# Test NFL predictor
print("\n\n2. Testing NFL Predictor...")
print("-" * 100)

nfl_pred = NFLPredictor()

test_nfl_games = [
    ("Buffalo Bills", "Houston Texans"),
    ("Baltimore Ravens", "Cleveland Browns"),
    ("Kansas City Chiefs", "Denver Broncos"),
]

for away, home in test_nfl_games:
    prediction = nfl_pred.predict_winner(home_team=home, away_team=away)

    if prediction:
        winner = prediction.get('winner', 'Unknown')
        prob = prediction.get('probability', 0)
        conf = prediction.get('confidence', 'unknown')

        print(f"\n{away} @ {home}")
        print(f"  Winner: {winner}")
        print(f"  Probability: {prob:.3f} ({prob*100:.1f}%)")
        print(f"  Confidence: {conf}")

        if abs(prob - 0.54) < 0.01:
            print(f"  [!] FOUND 54% ISSUE!")
    else:
        print(f"\n{away} @ {home}: No prediction")

print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)
print("""
If all predictions show ~54%, the issue is in the predictor logic.
If predictions are varied, the issue is in the UI caching or display logic.
""")
print("=" * 100)
