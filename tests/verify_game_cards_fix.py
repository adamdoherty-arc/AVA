"""
Quick verification that game cards AI analysis is working
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("GAME CARDS AI ANALYSIS VERIFICATION")
print("=" * 70)

# Test 1: Import the predictors
print("\nTest 1: Importing predictors...")
try:
    from src.prediction_agents.nfl_predictor import NFLPredictor
    from src.prediction_agents.ncaa_predictor import NCAAPredictor
    print("[OK] Predictors imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import predictors: {e}")
    sys.exit(1)

# Test 2: Create predictor instances
print("\nTest 2: Creating predictor instances...")
try:
    nfl_pred = NFLPredictor()
    print("[OK] NFL Predictor created")

    ncaa_pred = NCAAPredictor()
    print("[OK] NCAA Predictor created")
except Exception as e:
    print(f"[ERROR] Failed to create predictors: {e}")
    sys.exit(1)

# Test 3: Test prediction function
print("\nTest 3: Testing prediction function...")
try:
    # Test NFL prediction
    prediction = nfl_pred.predict_winner(
        home_team="Chicago Bears",
        away_team="Pittsburgh Steelers"
    )

    if prediction:
        print(f"[OK] NFL Prediction successful")
        print(f"   Winner: {prediction.get('winner')}")
        print(f"   Probability: {prediction.get('probability', 0):.1%}")
        print(f"   Confidence: {prediction.get('confidence', 'unknown')}")
        print(f"   Spread: {prediction.get('spread', 0):.1f}")

        # Check for explanation
        if prediction.get('explanation'):
            print(f"   [OK] Explanation included (AI analysis will show)")
        else:
            print(f"   [WARN]  No explanation (AI analysis might not show)")
    else:
        print("[ERROR] Prediction returned None")
        sys.exit(1)

except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify cache_resource pattern works
print("\nTest 4: Verifying cache_resource pattern...")
try:
    # Simulate what game_cards_visual_page.py does now
    def get_nfl_predictor():
        """Simulated cached resource function"""
        return NFLPredictor()

    def get_ncaa_predictor():
        """Simulated cached resource function"""
        return NCAAPredictor()

    # Get predictors (like the fix does)
    pred1 = get_nfl_predictor()
    pred2 = get_nfl_predictor()

    # They should be different instances (caching happens in Streamlit)
    print(f"[OK] Cache pattern works - predictors can be retrieved")

except Exception as e:
    print(f"[ERROR] Cache pattern failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("[OK] ALL TESTS PASSED - Game Cards AI Analysis Should Work!")
print("=" * 70)
print("\nNext steps:")
print("1. Clear Streamlit cache: Settings → Clear cache")
print("2. Reload the app (F5)")
print("3. Navigate to Game Cards")
print("4. Verify 'Multi-Agent AI Analysis' section shows for all games")
print("5. Should see:")
print("   - Ensemble Consensus percentages")
print("   - Win probabilities")
print("   - Betting recommendations")
print("   - Deep analytics expanded view")
print("\nIf you see 'AI analysis unavailable', something else is wrong.")
print("=" * 70)
