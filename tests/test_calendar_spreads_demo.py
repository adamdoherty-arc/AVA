"""
Test Calendar Spread Finder with Demo Data
"""

import time
from datetime import datetime, date, timedelta
from src.strategies.calendar_spread_models import CalendarSpreadOpportunity, OptionContract

def create_demo_opportunities():
    """Create demo calendar spread opportunities for testing"""

    opportunities = []
    today = date.today()

    # Demo opportunity 1: AAPL near ATM calendar
    opp1 = CalendarSpreadOpportunity(
        symbol="AAPL",
        stock_price=175.50,
        near_strike=175.0,
        near_expiration=today + timedelta(days=35),
        near_dte=35,
        near_premium=325.0,  # $3.25 per share
        near_iv=28.5,
        near_theta=-0.085,
        near_volume=1250,
        far_strike=175.0,
        far_expiration=today + timedelta(days=70),
        far_dte=70,
        far_premium=485.0,  # $4.85 per share
        far_iv=26.2,
        far_theta=-0.055,
        far_volume=850,
        net_debit=160.0,  # $1.60 per share
        max_profit=95.0,
        max_loss=160.0,
        profit_potential=0.594,  # 59.4% potential
        probability_profit=68.5,
        breakeven_lower=172.60,
        breakeven_upper=177.40,
        net_theta=0.030,
        net_vega=0.15,
        liquidity_score=85.0,
        iv_differential=2.3,
        opportunity_score=72.5,
        rank=1
    )
    opportunities.append(opp1)

    # Demo opportunity 2: SPY calendar spread
    opp2 = CalendarSpreadOpportunity(
        symbol="SPY",
        stock_price=445.25,
        near_strike=445.0,
        near_expiration=today + timedelta(days=30),
        near_dte=30,
        near_premium=425.0,
        near_iv=16.8,
        near_theta=-0.095,
        near_volume=5500,
        far_strike=445.0,
        far_expiration=today + timedelta(days=60),
        far_dte=60,
        far_premium=610.0,
        far_iv=15.2,
        far_theta=-0.068,
        far_volume=3200,
        net_debit=185.0,
        max_profit=125.0,
        max_loss=185.0,
        profit_potential=0.676,
        probability_profit=71.2,
        breakeven_lower=442.20,
        breakeven_upper=447.80,
        net_theta=0.027,
        net_vega=0.18,
        liquidity_score=100.0,
        iv_differential=1.6,
        opportunity_score=78.3,
        rank=2
    )
    opportunities.append(opp2)

    # Demo opportunity 3: MSFT calendar
    opp3 = CalendarSpreadOpportunity(
        symbol="MSFT",
        stock_price=380.50,
        near_strike=380.0,
        near_expiration=today + timedelta(days=32),
        near_dte=32,
        near_premium=520.0,
        near_iv=22.5,
        near_theta=-0.110,
        near_volume=2100,
        far_strike=380.0,
        far_expiration=today + timedelta(days=67),
        far_dte=67,
        far_premium=745.0,
        far_iv=21.1,
        far_theta=-0.072,
        far_volume=1500,
        net_debit=225.0,
        max_profit=140.0,
        max_loss=225.0,
        profit_potential=0.622,
        probability_profit=65.8,
        breakeven_lower=376.75,
        breakeven_upper=383.25,
        net_theta=0.038,
        net_vega=0.20,
        liquidity_score=92.0,
        iv_differential=1.4,
        opportunity_score=69.8,
        rank=3
    )
    opportunities.append(opp3)

    # Demo opportunity 4: QQQ calendar
    opp4 = CalendarSpreadOpportunity(
        symbol="QQQ",
        stock_price=385.75,
        near_strike=385.0,
        near_expiration=today + timedelta(days=28),
        near_dte=28,
        near_premium=475.0,
        near_iv=19.8,
        near_theta=-0.105,
        near_volume=4200,
        far_strike=385.0,
        far_expiration=today + timedelta(days=56),
        far_dte=56,
        far_premium=680.0,
        far_iv=18.5,
        far_theta=-0.075,
        far_volume=2800,
        net_debit=205.0,
        max_profit=115.0,
        max_loss=205.0,
        profit_potential=0.561,
        probability_profit=63.5,
        breakeven_lower=382.05,
        breakeven_upper=387.95,
        net_theta=0.030,
        net_vega=0.16,
        liquidity_score=95.0,
        iv_differential=1.3,
        opportunity_score=66.2,
        rank=4
    )
    opportunities.append(opp4)

    return opportunities

def display_demo_opportunities():
    """Display demo calendar spread opportunities"""

    print("\n" + "="*80)
    print("CALENDAR SPREAD EVALUATOR - DEMO DATA")
    print("="*80)

    opportunities = create_demo_opportunities()

    print(f"\nFound {len(opportunities)} Calendar Spread Opportunities")
    print("-" * 80)

    for opp in opportunities:
        print(f"\n{opp.rank}. {opp.symbol} - ${opp.near_strike:.2f} Strike")
        print("   " + "="*50)

        # Position details
        print(f"   Stock Price: ${opp.stock_price:.2f}")
        print(f"   Near Leg: {opp.near_expiration} ({opp.near_dte} DTE) - Premium: ${opp.near_premium:.2f}")
        print(f"   Far Leg:  {opp.far_expiration} ({opp.far_dte} DTE) - Premium: ${opp.far_premium:.2f}")

        # Financial metrics
        print(f"\n   Financial Metrics:")
        print(f"   - Net Debit: ${opp.net_debit:.2f}")
        print(f"   - Max Profit: ${opp.max_profit:.2f}")
        print(f"   - Profit Potential: {opp.profit_potential*100:.1f}%")
        print(f"   - Probability of Profit: {opp.probability_profit:.1f}%")

        # Breakeven range
        print(f"\n   Breakeven Range:")
        print(f"   - Lower: ${opp.breakeven_lower:.2f}")
        print(f"   - Upper: ${opp.breakeven_upper:.2f}")

        # Greeks & Quality
        print(f"\n   Greeks & Quality:")
        print(f"   - Net Theta: {opp.net_theta:.4f}")
        print(f"   - Net Vega: {opp.net_vega:.3f}")
        print(f"   - IV Differential: {opp.iv_differential:.1f}%")
        print(f"   - Liquidity Score: {opp.liquidity_score:.0f}/100")

        # Score
        print(f"\n   OPPORTUNITY SCORE: {opp.opportunity_score:.1f}/100")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    avg_profit = sum(o.max_profit for o in opportunities) / len(opportunities)
    avg_prob = sum(o.probability_profit for o in opportunities) / len(opportunities)
    avg_score = sum(o.opportunity_score for o in opportunities) / len(opportunities)
    avg_theta = sum(o.net_theta for o in opportunities) / len(opportunities)

    print(f"Average Max Profit: ${avg_profit:.2f}")
    print(f"Average Probability: {avg_prob:.1f}%")
    print(f"Average Net Theta: {avg_theta:.4f}")
    print(f"Average Opportunity Score: {avg_score:.1f}/100")
    print(f"Best Opportunity: {opportunities[0].symbol} ${opportunities[0].near_strike:.2f} (Score: {opportunities[0].opportunity_score:.1f})")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    print("\n1. SPY shows the best opportunity score (78.3/100) with:")
    print("   - Highest profit potential: 67.6%")
    print("   - Strong probability of profit: 71.2%")
    print("   - Excellent liquidity score: 100/100")

    print("\n2. AAPL offers balanced risk/reward with:")
    print("   - Moderate net debit: $160")
    print("   - Good theta advantage: 0.030")
    print("   - Higher IV differential: 2.3%")

    print("\n3. All spreads show positive theta advantage")
    print("   - Profitable time decay characteristics")
    print("   - Near-term options decay faster than far-term")

    print("\n" + "="*80)

if __name__ == "__main__":
    display_demo_opportunities()