"""Debug betting picks to see why nothing is showing"""
from src.kalshi_db_manager import KalshiDBManager
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent

print("Initializing...")
kalshi_db = KalshiDBManager()
ai_agent = AdvancedBettingAIAgent()

# Get NFL markets
markets = kalshi_db.get_active_markets(market_type='nfl')
print(f"Total NFL markets: {len(markets)}")

if markets:
    print(f"\nChecking first 20 markets...")

    lopsided_count = 0
    no_odds_count = 0
    analyzed_count = 0
    passed_filter_count = 0

    for i, market in enumerate(markets[:20]):
        home_team = market.get('home_team', '')
        away_team = market.get('away_team', '')
        yes_price = market.get('yes_price')
        no_price = market.get('no_price')

        # Convert to float safely
        try:
            yes_price = float(yes_price or 50)
            no_price = float(no_price or 50)
        except:
            print(f"{i+1}. SKIP - Invalid odds: {away_team} @ {home_team}")
            no_odds_count += 1
            continue

        # Check if lopsided
        if yes_price > 85 or yes_price < 15 or no_price > 85 or no_price < 15:
            print(f"{i+1}. LOPSIDED - {away_team} @ {home_team}: YES {yes_price:.0f}¢ / NO {no_price:.0f}¢")
            lopsided_count += 1
            continue

        print(f"{i+1}. GOOD - {away_team} @ {home_team}: YES {yes_price:.0f}¢ / NO {no_price:.0f}¢")
        analyzed_count += 1

        # Try AI analysis
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

            if ev >= 5.0 and conf >= 50:
                print(f"   ✅ PASSED: EV={ev:.1f}%, Conf={conf:.0f}%")
                passed_filter_count += 1
            else:
                print(f"   ❌ FILTERED: EV={ev:.1f}%, Conf={conf:.0f}%")

        except Exception as e:
            print(f"   ERROR analyzing: {e}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Lopsided (filtered): {lopsided_count}")
    print(f"  No odds data: {no_odds_count}")
    print(f"  Analyzed: {analyzed_count}")
    print(f"  Passed all filters: {passed_filter_count}")
