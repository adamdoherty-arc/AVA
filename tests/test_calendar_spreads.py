"""
Test Calendar Spread Finder
"""

import time
from datetime import datetime
from src.strategies.calendar_spread_finder import CalendarSpreadFinder

def test_calendar_spreads():
    """Test the calendar spread finder with real data"""

    finder = CalendarSpreadFinder()

    # Test with multiple symbols
    test_symbols = ['AAPL', 'MSFT', 'SPY']

    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Testing Calendar Spreads for {symbol}")
        print(f"{'='*60}")

        # Time the scan
        start_time = time.time()

        # Find opportunities
        opportunities = finder.find_opportunities(
            symbol,
            near_dte_range=(25, 45),
            far_dte_range=(50, 90),
            option_type='call'
        )

        elapsed_time = time.time() - start_time

        print(f"\nFound {len(opportunities)} opportunities in {elapsed_time:.2f} seconds")

        if opportunities:
            print(f"\nTop 5 Calendar Spread Opportunities for {symbol}:")
            print("-" * 80)

            for i, opp in enumerate(opportunities[:5], 1):
                print(f"\n{i}. Strike: ${opp.near_strike:.2f}")
                print(f"   Near: {opp.near_expiration} ({opp.near_dte} DTE) - Premium: ${opp.near_premium:.2f}")
                print(f"   Far:  {opp.far_expiration} ({opp.far_dte} DTE) - Premium: ${opp.far_premium:.2f}")
                print(f"   Net Debit: ${opp.net_debit:.2f}")
                print(f"   Max Profit: ${opp.max_profit:.2f} ({opp.profit_potential*100:.1f}% potential)")
                print(f"   Probability of Profit: {opp.probability_profit:.1f}%")
                print(f"   Theta Advantage: {opp.net_theta:.4f}")
                print(f"   IV Differential: {opp.iv_differential:.1f}%")
                print(f"   Liquidity Score: {opp.liquidity_score:.0f}/100")
                print(f"   Opportunity Score: {opp.opportunity_score:.1f}/100")

            # Summary statistics
            print(f"\n{'-'*40}")
            print("Summary Statistics:")
            print(f"Average Max Profit: ${sum(o.max_profit for o in opportunities) / len(opportunities):.2f}")
            print(f"Average Probability: {sum(o.probability_profit for o in opportunities) / len(opportunities):.1f}%")
            print(f"Average Score: {sum(o.opportunity_score for o in opportunities) / len(opportunities):.1f}/100")
            print(f"Best Score: {opportunities[0].opportunity_score:.1f}/100")

        else:
            print("No opportunities found matching criteria")

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_calendar_spreads()