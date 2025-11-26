import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Verifying integration...")

# 1. Test Predictors & LLM
print("\n--- Testing Predictors & LLM ---")
try:
    from src.prediction_agents.nfl_predictor import NFLPredictor
    from src.prediction_agents.ncaa_predictor import NCAAPredictor
    
    print("Predictors imported successfully.")
    
    # Test NFL Predictor
    nfl = NFLPredictor()
    print("NFL Predictor initialized.")
    
    # Mock data for prediction
    pred = nfl.predict_winner("Kansas City Chiefs", "Buffalo Bills")
    print(f"NFL Prediction: {pred['winner']} ({pred['probability']:.2f})")
    print(f"Explanation: {pred['explanation'][:100]}...")
    
    if "probability" in pred['explanation'] or "favored" in pred['explanation']:
         print("[OK] NFL Explanation generated")
    else:
         print("[WARNING] NFL Explanation might be empty or malformed")

    # Test NCAA Predictor
    ncaa = NCAAPredictor()
    print("NCAA Predictor initialized.")
    
    pred_ncaa = ncaa.predict_winner("Georgia", "Alabama")
    print(f"NCAA Prediction: {pred_ncaa['winner']} ({pred_ncaa['probability']:.2f})")
    print(f"Explanation: {pred_ncaa['explanation'][:100]}...")
    
    if "probability" in pred_ncaa['explanation']:
         print("[OK] NCAA Explanation generated")

    # Test NBA Predictor
    from src.prediction_agents.nba_predictor import NBAPredictor
    nba = NBAPredictor()
    print("NBA Predictor initialized.")
    
    pred_nba = nba.predict_game("LAL", "BOS")
    if pred_nba:
        print(f"NBA Prediction: {pred_nba['winner']} ({pred_nba['probability']:.2f})")
        print(f"Explanation: {pred_nba['explanation'][:100]}...")
        
        if "probability" in pred_nba['explanation']:
             print("[OK] NBA Explanation generated")
    else:
        print("[WARNING] NBA Prediction failed")

except Exception as e:
    print(f"[ERROR] Predictor test failed: {e}")
    import traceback
    traceback.print_exc()

# 2. Test Page Imports
print("\n--- Testing Page Imports ---")
try:
    # We can't run streamlit pages, but we can import them to check for syntax/import errors
    # Note: Streamlit pages might run code on import, so we wrap in try/except
    # and we might need to mock streamlit
    import streamlit as st
    
    # Mock st.set_page_config to avoid error
    if not hasattr(st, 'set_page_config_original'):
        st.set_page_config_original = st.set_page_config
        st.set_page_config = lambda **kwargs: None
    
    print("Importing game_cards_visual_page...")
    import game_cards_visual_page
    print("[OK] game_cards_visual_page imported")
    
    print("Importing kalshi_nfl_markets_page...")
    import kalshi_nfl_markets_page
    print("[OK] kalshi_nfl_markets_page imported")
    
    print("Importing best_bets_unified_page...")
    import best_bets_unified_page
    print("[OK] best_bets_unified_page imported")
    
except Exception as e:
    print(f"[ERROR] Page import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nVerification complete.")
