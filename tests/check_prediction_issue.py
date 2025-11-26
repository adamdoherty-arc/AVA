"""Check what's wrong with AI predictions"""
from src.kalshi_db_manager import KalshiDBManager
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent

kalshi_db = KalshiDBManager()
ai_agent = AdvancedBettingAIAgent()

# Get one NFL market
markets = kalshi_db.get_active_markets(market_type='nfl')
if markets:
    market = markets[0]
    print(f"Sample market from database:")
    print(f"  Teams: {market.get('away_team')} @ {market.get('home_team')}")
    print(f"  yes_price: {market.get('yes_price')} (type: {type(market.get('yes_price'))})")
    print(f"  no_price: {market.get('no_price')} (type: {type(market.get('no_price'))})")

    # Check what values we're actually getting
    yes_price = float(market.get('yes_price') or 0.5)
    no_price = float(market.get('no_price') or 0.5)

    print(f"\n  After conversion:")
    print(f"  yes_price: {yes_price}")
    print(f"  no_price: {no_price}")

    # Try AI analysis
    game_data = {
        'id': market.get('id'),
        'away_team': market.get('away_team'),
        'home_team': market.get('home_team'),
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

    print(f"\n  Market data passed to AI:")
    print(f"  yes_price: {market_data['yes_price']}")
    print(f"  no_price: {market_data['no_price']}")

    prediction = ai_agent.analyze_betting_opportunity(game_data, market_data)

    print(f"\n  AI Prediction:")
    print(f"  Predicted winner: {prediction.get('predicted_winner')}")
    print(f"  Win probability: {prediction.get('win_probability')}")
    print(f"  Expected value: {prediction.get('expected_value')}")
    print(f"  Confidence: {prediction.get('confidence_score')}")
