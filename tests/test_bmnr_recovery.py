"""
Test script to diagnose why BMNR recovery strategies aren't showing
"""

import yfinance as yf
from src.option_roll_evaluator import OptionRollEvaluator
from datetime import datetime

# Test BMNR position data
bmnr_position = {
    'symbol': 'BMNR',
    'current_strike': 47.50,
    'current_price': 42.30,
    'expiration': '2025-01-17',  # Try common date format
    'premium_collected': 200,  # $2.00 per share
    'quantity': -1
}

print("="*80)
print("Testing BMNR Recovery Strategies")
print("="*80)

# Test 1: Check if Yahoo Finance has BMNR options data
print("\n1. Checking Yahoo Finance options data for BMNR...")
try:
    ticker = yf.Ticker('BMNR')
    print(f"   Ticker exists: Yes")

    # Get available expiration dates
    exp_dates = ticker.options
    print(f"   Available expirations: {len(exp_dates)}")
    if exp_dates:
        print(f"   First 5 expirations: {exp_dates[:5]}")
    else:
        print("   ERROR: No expiration dates available!")
        print("   This means Yahoo Finance doesn't have options data for BMNR")
        print("   BMNR may not have listed options trading")

    # Try to get current stock price
    info = ticker.info
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    print(f"   Current price from Yahoo: ${current_price}")

except Exception as e:
    print(f"   ERROR getting BMNR data: {e}")

# Test 2: Try to evaluate roll strategies
print("\n2. Testing roll strategy evaluator...")
try:
    evaluator = OptionRollEvaluator()

    # Test each strategy individually
    print("\n   A. Testing Roll Down...")
    try:
        roll_down = evaluator.evaluate_roll_down(bmnr_position)
        print(f"      Feasible: {roll_down.get('feasible', False)}")
        if not roll_down.get('feasible'):
            print(f"      Reason: {roll_down.get('reason', 'Unknown')}")
    except Exception as e:
        print(f"      ERROR: {e}")

    print("\n   B. Testing Roll Out...")
    try:
        roll_out = evaluator.evaluate_roll_out(bmnr_position)
        print(f"      Feasible: {roll_out.get('feasible', False)}")
        if not roll_out.get('feasible'):
            print(f"      Reason: {roll_out.get('reason', 'Unknown')}")
    except Exception as e:
        print(f"      ERROR: {e}")

    print("\n   C. Testing Roll Down & Out...")
    try:
        roll_down_out = evaluator.evaluate_roll_down_and_out(bmnr_position)
        print(f"      Feasible: {roll_down_out.get('feasible', False)}")
        if not roll_down_out.get('feasible'):
            print(f"      Reason: {roll_down_out.get('reason', 'Unknown')}")
    except Exception as e:
        print(f"      ERROR: {e}")

    print("\n   D. Testing Assignment...")
    try:
        assignment = evaluator.evaluate_assignment(bmnr_position)
        print(f"      Feasible: {assignment.get('feasible', False)}")
        if not assignment.get('feasible'):
            print(f"      Reason: {assignment.get('reason', 'Unknown')}")
        else:
            print(f"      Cost Basis: ${assignment.get('cost_basis_per_share', 0):.2f}")
            print(f"      Immediate Loss: ${assignment.get('immediate_loss', 0):.2f}")
    except Exception as e:
        print(f"      ERROR: {e}")

    # Test full comparison
    print("\n   E. Testing full strategy comparison...")
    comparison = evaluator.compare_strategies(bmnr_position)

    feasible_strategies = [s for s in comparison['strategies'].values()
                          if s.get('feasible', False)]
    print(f"      Feasible strategies found: {len(feasible_strategies)}")

    if feasible_strategies:
        print(f"      Ranked strategies:")
        for i, strategy in enumerate(comparison['ranked_strategies'], 1):
            print(f"         {i}. {strategy['strategy']} (Score: {strategy['score']:.2f})")
    else:
        print("      No feasible strategies found!")
        print("      Reasons:")
        for key, strategy in comparison['strategies'].items():
            print(f"         {strategy['strategy']}: {strategy.get('reason', 'No reason given')}")

    # Show recommendation
    print(f"\n      Recommendation: {comparison['recommendation']['recommended_strategy']}")
    print(f"      Reasoning: {comparison['recommendation']['reasoning']}")

except Exception as e:
    print(f"   ERROR running evaluator: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Alternative expiration formats
print("\n3. Testing different expiration date formats...")
test_dates = [
    '2025-01-17',
    '01/17/2025',
    '2025-1-17',
    '17-Jan-2025'
]

for test_date in test_dates:
    test_pos = bmnr_position.copy()
    test_pos['expiration'] = test_date
    try:
        result = evaluator.evaluate_assignment(test_pos)
        print(f"   {test_date}: Feasible={result.get('feasible', False)}")
    except Exception as e:
        print(f"   {test_date}: ERROR - {str(e)[:50]}")

print("\n" + "="*80)
print("Diagnostic Complete")
print("="*80)
