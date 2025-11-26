"""
Comprehensive Test of Calendar Spread System
"""

import pandas as pd
from datetime import date, timedelta
from src.strategies.calendar_spread_models import CalendarSpreadOpportunity
from src.strategies.calendar_spread_display import display_spread_details


def create_comprehensive_test_data():
    """Create comprehensive test data for calendar spreads"""

    opportunities = []
    today = date.today()

    # Test data with various scenarios
    test_data = [
        # High probability, moderate profit
        {
            'symbol': 'SPY',
            'stock_price': 445.25,
            'strike': 445.0,
            'near_dte': 30,
            'far_dte': 60,
            'near_premium': 4.25,
            'far_premium': 6.10,
            'near_iv': 0.168,
            'far_iv': 0.152,
            'near_volume': 5500,
            'far_volume': 3200,
            'probability': 71.2,
            'liquidity': 100,
        },
        # High profit potential, lower probability
        {
            'symbol': 'TSLA',
            'stock_price': 225.50,
            'strike': 225.0,
            'near_dte': 35,
            'far_dte': 70,
            'near_premium': 8.50,
            'far_premium': 12.25,
            'near_iv': 0.452,
            'far_iv': 0.398,
            'near_volume': 2800,
            'far_volume': 1900,
            'probability': 58.5,
            'liquidity': 75,
        },
        # Balanced opportunity
        {
            'symbol': 'AAPL',
            'stock_price': 175.50,
            'strike': 175.0,
            'near_dte': 35,
            'far_dte': 70,
            'near_premium': 3.25,
            'far_premium': 4.85,
            'near_iv': 0.285,
            'far_iv': 0.262,
            'near_volume': 1250,
            'far_volume': 850,
            'probability': 68.5,
            'liquidity': 85,
        },
        # High IV differential opportunity
        {
            'symbol': 'AMD',
            'stock_price': 125.75,
            'strike': 125.0,
            'near_dte': 28,
            'far_dte': 63,
            'near_premium': 4.15,
            'far_premium': 5.95,
            'near_iv': 0.385,
            'far_iv': 0.325,
            'near_volume': 1800,
            'far_volume': 1200,
            'probability': 62.8,
            'liquidity': 70,
        },
    ]

    for idx, data in enumerate(test_data, 1):
        # Calculate derived metrics
        net_debit = (data['far_premium'] - data['near_premium']) * 100

        # Calculate max profit (simplified model)
        days_between = data['far_dte'] - data['near_dte']
        time_decay_factor = (days_between / data['far_dte']) ** 0.5
        far_value_at_near_exp = data['far_premium'] * time_decay_factor * 100
        max_profit = far_value_at_near_exp - net_debit

        # Calculate theta values
        near_theta = -data['near_premium'] / data['near_dte'] if data['near_dte'] > 0 else -0.1
        far_theta = -data['far_premium'] / data['far_dte'] if data['far_dte'] > 0 else -0.05

        # Calculate opportunity score
        profit_potential = max_profit / net_debit if net_debit > 0 else 0
        score = (
            min(100, (profit_potential / 1.5) * 100) * 0.35 +  # Profit potential
            data['probability'] * 0.30 +  # Probability
            min(100, abs(near_theta - far_theta) / 0.05 * 100) * 0.20 +  # Theta advantage
            data['liquidity'] * 0.10 +  # Liquidity
            min(100, abs(data['near_iv'] - data['far_iv']) * 100 / 10 * 100) * 0.05  # IV differential
        )

        opp = CalendarSpreadOpportunity(
            symbol=data['symbol'],
            stock_price=data['stock_price'],
            near_strike=data['strike'],
            near_expiration=today + timedelta(days=data['near_dte']),
            near_dte=data['near_dte'],
            near_premium=data['near_premium'] * 100,
            near_iv=data['near_iv'] * 100,
            near_theta=near_theta,
            near_volume=data['near_volume'],
            far_strike=data['strike'],
            far_expiration=today + timedelta(days=data['far_dte']),
            far_dte=data['far_dte'],
            far_premium=data['far_premium'] * 100,
            far_iv=data['far_iv'] * 100,
            far_theta=far_theta,
            far_volume=data['far_volume'],
            net_debit=net_debit,
            max_profit=max_profit,
            max_loss=net_debit,
            profit_potential=profit_potential,
            probability_profit=data['probability'],
            breakeven_lower=data['strike'] - (net_debit / 100 * 1.5),
            breakeven_upper=data['strike'] + (net_debit / 100 * 1.5),
            net_theta=abs(near_theta) - abs(far_theta),
            net_vega=0.15,  # Simplified
            liquidity_score=data['liquidity'],
            iv_differential=abs(data['near_iv'] - data['far_iv']) * 100,
            opportunity_score=score,
            rank=idx
        )
        opportunities.append(opp)

    # Sort by score and re-rank
    opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
    for idx, opp in enumerate(opportunities, 1):
        opp.rank = idx

    return opportunities


def test_calendar_spread_system():
    """Test the complete calendar spread system"""

    print("\n" + "="*80)
    print("COMPREHENSIVE CALENDAR SPREAD SYSTEM TEST")
    print("="*80)

    # Create test data
    opportunities = create_comprehensive_test_data()

    print(f"\nGenerated {len(opportunities)} test opportunities")
    print("-" * 80)

    # Display detailed analysis
    for opp in opportunities:
        print(f"\n{opp.rank}. {opp.symbol} - ${opp.near_strike:.2f} Strike")
        print("   " + "="*50)

        # Position details
        print(f"   Stock Price: ${opp.stock_price:.2f} (Strike {'ATM' if abs(opp.stock_price - opp.near_strike) < 1 else 'OTM' if opp.stock_price < opp.near_strike else 'ITM'})")
        print(f"   Near: {opp.near_expiration} ({opp.near_dte}d) @ ${opp.near_premium:.2f} | IV: {opp.near_iv:.1f}% | Vol: {opp.near_volume}")
        print(f"   Far:  {opp.far_expiration} ({opp.far_dte}d) @ ${opp.far_premium:.2f} | IV: {opp.far_iv:.1f}% | Vol: {opp.far_volume}")

        # Key metrics
        print(f"\n   Metrics:")
        print(f"   - Net Debit:    ${opp.net_debit:.2f}")
        print(f"   - Max Profit:   ${opp.max_profit:.2f} ({opp.profit_potential*100:.1f}% return)")
        print(f"   - Probability:  {opp.probability_profit:.1f}%")
        print(f"   - Breakeven:    ${opp.breakeven_lower:.2f} - ${opp.breakeven_upper:.2f}")

        # Greeks
        print(f"\n   Greeks:")
        print(f"   - Net Theta:    {opp.net_theta:.4f} (daily decay advantage)")
        print(f"   - IV Diff:      {opp.iv_differential:.1f}% (volatility premium)")

        # Scores
        print(f"\n   Quality:")
        print(f"   - Liquidity:    {opp.liquidity_score:.0f}/100")
        print(f"   - Score:        {opp.opportunity_score:.1f}/100")

    # Create DataFrame for analysis
    df_data = []
    for opp in opportunities:
        df_data.append({
            'Rank': opp.rank,
            'Symbol': opp.symbol,
            'Strike': opp.near_strike,
            'Stock Price': opp.stock_price,
            'Near DTE': opp.near_dte,
            'Far DTE': opp.far_dte,
            'Net Debit': opp.net_debit,
            'Max Profit': opp.max_profit,
            'Return %': opp.profit_potential * 100,
            'Prob %': opp.probability_profit,
            'Theta': opp.net_theta,
            'IV Diff': opp.iv_differential,
            'Liquidity': opp.liquidity_score,
            'Score': opp.opportunity_score
        })

    df = pd.DataFrame(df_data)

    print("\n" + "="*80)
    print("DATAFRAME ANALYSIS")
    print("="*80)
    print("\nOpportunities Table:")
    print(df.to_string())

    print("\n" + "-"*80)
    print("Statistical Summary:")
    print("-"*80)
    print(df[['Net Debit', 'Max Profit', 'Return %', 'Prob %', 'Score']].describe())

    # Correlation analysis
    print("\n" + "-"*80)
    print("Correlation Analysis:")
    print("-"*80)
    correlations = df[['Return %', 'Prob %', 'Theta', 'IV Diff', 'Liquidity', 'Score']].corr()['Score'].sort_values(ascending=False)
    print("Correlation with Score:")
    for metric, corr in correlations.items():
        if metric != 'Score':
            print(f"  {metric:12s}: {corr:+.3f}")

    # Best opportunity by category
    print("\n" + "="*80)
    print("BEST OPPORTUNITIES BY CATEGORY")
    print("="*80)

    best_return = df.loc[df['Return %'].idxmax()]
    print(f"\nHighest Return Potential:")
    print(f"  {best_return['Symbol']} ${best_return['Strike']:.0f}: {best_return['Return %']:.1f}% potential return")

    best_prob = df.loc[df['Prob %'].idxmax()]
    print(f"\nHighest Probability:")
    print(f"  {best_prob['Symbol']} ${best_prob['Strike']:.0f}: {best_prob['Prob %']:.1f}% probability")

    best_liquidity = df.loc[df['Liquidity'].idxmax()]
    print(f"\nMost Liquid:")
    print(f"  {best_liquidity['Symbol']} ${best_liquidity['Strike']:.0f}: {best_liquidity['Liquidity']:.0f}/100 liquidity")

    best_overall = df.loc[df['Score'].idxmax()]
    print(f"\nBest Overall Score:")
    print(f"  {best_overall['Symbol']} ${best_overall['Strike']:.0f}: {best_overall['Score']:.1f}/100 score")

    # Risk/Reward Analysis
    print("\n" + "="*80)
    print("RISK/REWARD ANALYSIS")
    print("="*80)

    for _, row in df.iterrows():
        risk_reward = row['Max Profit'] / row['Net Debit']
        expected_value = (row['Max Profit'] * row['Prob %'] / 100) - (row['Net Debit'] * (100 - row['Prob %']) / 100)
        print(f"\n{row['Symbol']} ${row['Strike']:.0f}:")
        print(f"  Risk/Reward Ratio: {risk_reward:.2f}:1")
        print(f"  Expected Value: ${expected_value:.2f}")
        print(f"  Kelly Fraction: {(row['Prob %']/100 - (1-row['Prob %']/100)/risk_reward):.1%}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_calendar_spread_system()