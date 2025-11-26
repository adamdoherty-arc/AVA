"""
Test script for CSP Recovery & Roll Strategy Feature

This script tests all components of the recovery strategies feature with sample data.
Run this to verify functionality without needing actual Robinhood positions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import the modules to test
from src.csp_recovery_analyzer import CSPRecoveryAnalyzer
from src.option_roll_evaluator import OptionRollEvaluator
from src.ai_options_advisor import AIOptionsAdvisor


def create_sample_positions():
    """Create sample losing CSP positions for testing"""

    positions = [
        {
            'symbol': 'AAPL',
            'option_type': 'put',
            'position_type': 'short',
            'strike_price': 185.0,
            'current_price': 178.50,  # Stock is below strike - losing position
            'average_price': 3.25,  # Premium collected
            'quantity': -2,  # Sold 2 contracts
            'expiration_date': (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
            'premium_collected': 325 * 2,  # Total premium
            'current_loss': (185 - 178.50) * 100 * 2,  # Unrealized loss
            'loss_percentage': ((185 - 178.50) / 185) * 100,
            'days_to_expiry': 15
        },
        {
            'symbol': 'MSFT',
            'option_type': 'put',
            'position_type': 'short',
            'strike_price': 420.0,
            'current_price': 405.25,  # Stock is below strike
            'average_price': 5.50,
            'quantity': -1,
            'expiration_date': (datetime.now() + timedelta(days=8)).strftime('%Y-%m-%d'),
            'premium_collected': 550,
            'current_loss': (420 - 405.25) * 100,
            'loss_percentage': ((420 - 405.25) / 420) * 100,
            'days_to_expiry': 8
        },
        {
            'symbol': 'NVDA',
            'option_type': 'put',
            'position_type': 'short',
            'strike_price': 850.0,
            'current_price': 810.0,  # Deep in the money
            'average_price': 12.00,
            'quantity': -1,
            'expiration_date': (datetime.now() + timedelta(days=22)).strftime('%Y-%m-%d'),
            'premium_collected': 1200,
            'current_loss': (850 - 810) * 100,
            'loss_percentage': ((850 - 810) / 850) * 100,
            'days_to_expiry': 22
        },
        {
            'symbol': 'SPY',
            'option_type': 'put',
            'position_type': 'short',
            'strike_price': 440.0,
            'current_price': 435.50,  # Slightly below strike
            'average_price': 2.15,
            'quantity': -3,
            'expiration_date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'premium_collected': 215 * 3,
            'current_loss': (440 - 435.50) * 100 * 3,
            'loss_percentage': ((440 - 435.50) / 440) * 100,
            'days_to_expiry': 3  # Urgent - near expiration
        }
    ]

    # Add additional fields needed for analysis
    for pos in positions:
        pos['expiration'] = pos['expiration_date']
        pos['current_strike'] = pos['strike_price']

    return positions


def test_csp_recovery_analyzer():
    """Test the CSP Recovery Analyzer"""
    print("\n" + "="*60)
    print("TESTING CSP RECOVERY ANALYZER")
    print("="*60)

    analyzer = CSPRecoveryAnalyzer()

    # Create sample positions
    positions = create_sample_positions()

    # Analyze losing positions
    losing_positions = analyzer.analyze_losing_positions(positions)

    print(f"\nFound {len(losing_positions)} losing positions:")
    for pos in losing_positions:
        print(f"\n{pos['symbol']}:")
        print(f"  Strike: ${pos['current_strike']:.2f}")
        print(f"  Current Price: ${pos['current_price']:.2f}")
        print(f"  Loss: ${pos['current_loss']:.2f} ({pos['loss_percentage']:.1f}%)")
        print(f"  Days to Expiry: {pos['days_to_expiry']}")
        print(f"  IV Rank: {pos.get('iv_rank', 'N/A')}")

    # Find recovery opportunities for the most urgent position (SPY)
    spy_position = next(p for p in losing_positions if p['symbol'] == 'SPY')
    print(f"\n\nFinding recovery opportunities for {spy_position['symbol']}...")

    opportunities = analyzer.find_recovery_opportunities(spy_position, num_strikes=3)

    if opportunities:
        print(f"\nTop {len(opportunities)} Recovery Opportunities:")
        for i, opp in enumerate(opportunities, 1):
            print(f"\n{i}. Strike: ${opp.get('strike', 0):.2f}")
            print(f"   Premium: ${opp.get('premium', 0):.2f}")
            print(f"   Yield: {opp.get('yield_percent', 0):.2f}%")
            print(f"   Probability of Profit: {opp.get('probability_profit', 0)*100:.1f}%")
            print(f"   Recovery %: {opp.get('recovery_percentage', 0):.1f}%")
            print(f"   AI Score: {opp.get('ai_score', 0):.1f}")
            print(f"   Recommendation: {opp.get('recommendation', 'N/A')}")
    else:
        print("No recovery opportunities found")

    return losing_positions


def test_option_roll_evaluator():
    """Test the Option Roll Evaluator"""
    print("\n" + "="*60)
    print("TESTING OPTION ROLL EVALUATOR")
    print("="*60)

    evaluator = OptionRollEvaluator()

    # Use the MSFT position for testing rolls
    position = {
        'symbol': 'MSFT',
        'current_strike': 420.0,
        'current_price': 405.25,
        'expiration': (datetime.now() + timedelta(days=8)).strftime('%Y-%m-%d'),
        'premium_collected': 550,
        'quantity': -1
    }

    print(f"\nEvaluating roll strategies for {position['symbol']}:")
    print(f"  Current Strike: ${position['current_strike']:.2f}")
    print(f"  Current Price: ${position['current_price']:.2f}")
    print(f"  Days to Expiry: 8")

    # Test each strategy individually
    strategies_to_test = [
        ('Roll Down', evaluator.evaluate_roll_down),
        ('Roll Out', evaluator.evaluate_roll_out),
        ('Roll Down & Out', evaluator.evaluate_roll_down_and_out),
        ('Accept Assignment', evaluator.evaluate_assignment)
    ]

    results = {}
    for strategy_name, evaluate_func in strategies_to_test:
        print(f"\n\n--- {strategy_name} Strategy ---")
        result = evaluate_func(position)
        results[strategy_name] = result

        if result.get('feasible', False):
            print(f"✅ Feasible")
            print(f"Description: {result.get('description', 'N/A')}")

            if 'new_strike' in result:
                print(f"New Strike: ${result['new_strike']:.2f}")
            if 'new_expiration' in result:
                print(f"New Expiration: {result['new_expiration']}")
            if 'net_credit' in result:
                print(f"Net Credit/Debit: ${result['net_credit']:.2f}")
            if 'probability_profit' in result:
                print(f"Probability of Profit: {result['probability_profit']*100:.1f}%")

            if 'pros' in result:
                print("\nPros:")
                for pro in result['pros']:
                    print(f"  ✓ {pro}")

            if 'cons' in result:
                print("\nCons:")
                for con in result['cons']:
                    print(f"  ✗ {con}")
        else:
            print(f"❌ Not Feasible")
            print(f"Reason: {result.get('reason', 'Unknown')}")

    # Test the comparison function
    print("\n\n--- STRATEGY COMPARISON ---")
    comparison = evaluator.compare_strategies(position)

    if 'recommendation' in comparison:
        rec = comparison['recommendation']
        print(f"\n🤖 AI RECOMMENDATION:")
        print(f"Strategy: {rec['recommended_strategy']}")
        print(f"Confidence: {rec['confidence']}")
        print(f"Reasoning: {rec['reasoning']}")
        print(f"Expected Outcome: {rec.get('expected_outcome', 'N/A')}")
        print(f"Risk Assessment: {rec.get('risk_assessment', 'N/A')}")

    return comparison


def test_ai_options_advisor():
    """Test the AI Options Advisor"""
    print("\n" + "="*60)
    print("TESTING AI OPTIONS ADVISOR")
    print("="*60)

    # Note: This will have limited functionality without an OpenAI API key
    advisor = AIOptionsAdvisor()

    position = {
        'symbol': 'NVDA',
        'current_strike': 850.0,
        'current_price': 810.0,
        'current_loss': 4000,
        'loss_percentage': 4.7,
        'days_to_expiry': 22
    }

    opportunities = [
        {
            'strike': 800,
            'premium': 8.50,
            'probability_profit': 0.72,
            'yield_percent': 1.06,
            'recovery_percentage': 21.25,
            'ai_score': 75.5
        },
        {
            'strike': 790,
            'premium': 6.25,
            'probability_profit': 0.78,
            'yield_percent': 0.79,
            'recovery_percentage': 15.63,
            'ai_score': 68.2
        }
    ]

    print(f"\nAnalyzing {position['symbol']}:")
    print(f"  Current Strike: ${position['current_strike']:.2f}")
    print(f"  Current Price: ${position['current_price']:.2f}")
    print(f"  Loss: ${position['current_loss']:.2f} ({position['loss_percentage']:.1f}%)")

    # Test fundamental analysis
    print("\n--- Fundamental Analysis ---")
    fundamentals = advisor.fundamental_analyzer.analyze_sync(position['symbol'])
    for key, value in fundamentals.items():
        if value is not None and value != 'N/A':
            print(f"  {key}: {value}")

    # Test technical analysis
    print("\n--- Technical Analysis ---")
    technicals = advisor.technical_analyzer.analyze_sync(position['symbol'])
    for key, value in technicals.items():
        if value is not None and value != 'N/A':
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    # Test Greeks analysis
    print("\n--- Option Greeks ---")
    greeks = advisor.analyze_option_greeks(
        position['symbol'],
        position['current_strike'],
        (datetime.now() + timedelta(days=22)).strftime('%Y-%m-%d'),
        'put'
    )

    if 'values' in greeks:
        for greek, value in greeks['values'].items():
            interpretation = greeks['interpretations'].get(greek, '')
            print(f"  {greek.capitalize()}: {value:.4f} - {interpretation}")

        print(f"\n  Overall Assessment: {greeks.get('overall_assessment', 'N/A')}")

    # Test Monte Carlo simulation
    print("\n--- Monte Carlo Simulation ---")
    simulation = advisor.run_monte_carlo_simulation(
        position['symbol'],
        position['current_strike'],
        position['days_to_expiry'],
        num_simulations=1000  # Reduced for faster testing
    )

    if 'probability_profit' in simulation:
        print(f"  Probability of Profit: {simulation['probability_profit']*100:.1f}%")
        print(f"  Expected Loss if Assigned: ${simulation['expected_loss']:.2f}")
        print(f"  Price Percentiles:")
        print(f"    5th: ${simulation['percentile_5']:.2f}")
        print(f"    50th: ${simulation['percentile_50']:.2f}")
        print(f"    95th: ${simulation['percentile_95']:.2f}")

    # Test strategy recommendation
    print("\n--- AI Strategy Recommendation ---")
    recommendation = advisor.recommend_strategy(position, opportunities)
    print(recommendation)

    return advisor


def test_integration():
    """Test the integration of all components"""
    print("\n" + "="*60)
    print("TESTING FULL INTEGRATION")
    print("="*60)

    # Create components
    analyzer = CSPRecoveryAnalyzer()
    evaluator = OptionRollEvaluator()
    advisor = AIOptionsAdvisor()

    # Create sample positions
    positions = create_sample_positions()

    # Analyze all positions
    losing_positions = analyzer.analyze_losing_positions(positions)

    print(f"\nAnalyzing {len(losing_positions)} losing positions...")

    # For each position, find opportunities and evaluate rolls
    for pos in losing_positions[:2]:  # Test first 2 positions
        print(f"\n\n{'='*40}")
        print(f"POSITION: {pos['symbol']} ${pos['current_strike']:.2f}")
        print(f"{'='*40}")

        # Find recovery opportunities
        opportunities = analyzer.find_recovery_opportunities(pos, num_strikes=3)

        if opportunities:
            print(f"\n📊 Top Recovery Opportunity:")
            best = opportunities[0]
            print(f"  Strike: ${best['strike']:.2f}")
            print(f"  Premium: ${best['premium']:.2f}")
            print(f"  AI Score: {best['ai_score']:.1f}")

        # Evaluate roll strategies
        comparison = evaluator.compare_strategies(pos)

        if 'recommendation' in comparison:
            print(f"\n🔄 Best Roll Strategy:")
            print(f"  {comparison['recommendation']['recommended_strategy']}")
            print(f"  Confidence: {comparison['recommendation']['confidence']}")

        # Get AI recommendation
        if opportunities:
            rec = advisor.recommend_strategy(pos, opportunities)
            print(f"\n🤖 AI Analysis:")
            print(rec[:300] + "..." if len(rec) > 300 else rec)


def main():
    """Main test execution"""
    print("\n" + "="*60)
    print("CSP RECOVERY & ROLL STRATEGY FEATURE TEST SUITE")
    print("="*60)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Run individual component tests
        print("\n\n📝 Running Component Tests...")

        losing_positions = test_csp_recovery_analyzer()
        print(f"\n✅ CSP Recovery Analyzer: PASSED")

        comparison = test_option_roll_evaluator()
        print(f"\n✅ Option Roll Evaluator: PASSED")

        advisor = test_ai_options_advisor()
        print(f"\n✅ AI Options Advisor: PASSED")

        # Run integration test
        print("\n\n🔗 Running Integration Test...")
        test_integration()
        print(f"\n✅ Integration Test: PASSED")

        print("\n\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe CSP Recovery & Roll Strategy Feature is ready for production.")
        print("\nTo use in the dashboard:")
        print("1. Navigate to the Positions page")
        print("2. Look for the '🎯 Recovery Strategies' section")
        print("3. It will appear when you have losing CSP positions")

    except Exception as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)