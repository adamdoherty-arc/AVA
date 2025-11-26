#!/usr/bin/env python
"""Test complete system with AI analysis"""

from src.robinhood_fixed import *
from src.ai_trade_analyzer import AITradeAnalyzer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("COMPLETE SYSTEM TEST")
print("="*60)

if login_robinhood():
    # Get positions
    stocks = get_positions()
    options = get_options()
    wheel_positions = identify_wheel_positions(stocks, options)

    print(f"\nFound {len(wheel_positions)} wheel positions")

    if wheel_positions:
        # Convert to dashboard format
        positions = []
        for wp in wheel_positions:
            if wp['strategy'] == 'CSP':
                positions.append({
                    'Symbol': wp['symbol'],
                    'Type': 'CSP',
                    'Strike': wp.get('strike', 0),
                    'Expiration': wp.get('expiration', ''),
                    'Premium': wp.get('premium', 0),
                    'Current Value': wp.get('current_value', 0),
                    'Days to Expiry': wp.get('days_to_expiry', 0),
                    'Status': 'Open'
                })

        # Test AI analysis
        print("\n" + "="*60)
        print("AI ANALYSIS")
        print("="*60)

        analyzer = AITradeAnalyzer()
        portfolio_analysis = analyzer.get_portfolio_recommendations(positions)

        print(f"\nPortfolio Action: {portfolio_analysis['suggested_action']}")

        if portfolio_analysis['buyback_candidates']:
            print("\nBuyback Candidates:")
            for c in portfolio_analysis['buyback_candidates']:
                print(f"  {c['symbol']}: {c['profit_pct']:.1f}% profit (${c['profit']:.2f})")

        print("\nIndividual Recommendations:")
        for rec in portfolio_analysis['recommendations']:
            print(f"\n{rec['symbol']}:")
            print(f"  Action: {rec['recommendation']['action']}")
            print(f"  Reason: {rec['recommendation']['reason']}")

    # Logout
    import robin_stocks.robinhood as rh
    rh.authentication.logout()

else:
    print("Login failed")