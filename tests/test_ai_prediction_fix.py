"""Test that AI predictions now work correctly with fixed odds"""
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent

ai_agent = AdvancedBettingAIAgent()

# Simulate Buffalo Bills @ Houston Texans
# Kalshi odds: Bills (away) 72%, Texans (home) 28%
game = {
    'game_id': '12345',
    'away_team': 'Buffalo Bills',
    'home_team': 'Houston Texans',
    'away_score': 0,
    'home_score': 0,
    'kalshi_odds': {
        'away_win_price': 0.72,  # 72% Bills favored
        'home_win_price': 0.28   # 28% Texans underdog
    }
}

print("Testing AI prediction with:")
print(f"  Game: {game['away_team']} @ {game['home_team']}")
print(f"  Market odds: {game['kalshi_odds']['away_win_price']*100:.0f}% / {game['kalshi_odds']['home_win_price']*100:.0f}%")

prediction = ai_agent.analyze_betting_opportunity(game, {})

print(f"\nAI Prediction:")
print(f"  Predicted winner: {prediction.get('predicted_winner')}")
print(f"  Win probability: {prediction.get('win_probability', 0)*100:.1f}%")
print(f"  Confidence: {prediction.get('confidence_score', 0):.0f}%")
print(f"  Expected value: {prediction.get('expected_value', 0):+.1f}%")
print(f"  Recommendation: {prediction.get('recommendation')}")

# Verify it makes sense
if prediction.get('predicted_winner') in ['Buffalo Bills', 'away']:
    print("\n✅ CORRECT: AI correctly predicts Bills (the favorite) to win!")
else:
    print("\n❌ WRONG: AI should predict Bills to win, not Texans")
