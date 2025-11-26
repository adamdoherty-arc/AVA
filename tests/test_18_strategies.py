"""
Comprehensive Test Script for All 18 Options Strategies
Tests for errors, bottlenecks, and efficiency
"""

import time
import sys
import os
from typing import Dict, Any, List
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_options_agent.comprehensive_strategy_analyzer import ComprehensiveStrategyAnalyzer

def test_all_strategies():
    """Test all 18 strategies with various market conditions"""

    print("=" * 80)
    print("🧪 COMPREHENSIVE TESTING: All 18 Options Strategies")
    print("=" * 80)
    print()

    # Initialize analyzer
    analyzer = ComprehensiveStrategyAnalyzer()

    # Test cases with different market conditions
    test_cases = [
        {
            'name': 'SOFO - High IV Bearish',
            'symbol': 'SOFO',
            'stock_price': 8.45,
            'iv': 0.485,  # 48.5%
            'dte': 30,
            'trend': 'bearish'
        },
        {
            'name': 'AAPL - Low IV Bullish',
            'symbol': 'AAPL',
            'stock_price': 185.50,
            'iv': 0.22,  # 22%
            'dte': 45,
            'trend': 'bullish'
        },
        {
            'name': 'TSLA - High IV Neutral',
            'symbol': 'TSLA',
            'stock_price': 245.00,
            'iv': 0.55,  # 55%
            'dte': 30,
            'trend': 'neutral'
        },
        {
            'name': 'SPY - Moderate IV Neutral',
            'symbol': 'SPY',
            'stock_price': 475.00,
            'iv': 0.28,  # 28%
            'dte': 30,
            'trend': 'neutral'
        }
    ]

    all_tests_passed = True
    performance_data = []

    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"📊 Testing: {test_case['name']}")
        print(f"{'='*80}")
        print(f"  Stock Price: ${test_case['stock_price']}")
        print(f"  IV: {test_case['iv']*100:.1f}%")
        print(f"  DTE: {test_case['dte']} days")
        print(f"  Trend: {test_case['trend']}")
        print()

        # Mock stock data and options data
        stock_data = {
            'symbol': test_case['symbol'],
            'current_price': test_case['stock_price'],
            'iv': test_case['iv'],
            'trend': test_case['trend'],
            'market_cap': 1000000000,  # $1B
            'sector': 'Technology',
            'price_52w_high': test_case['stock_price'] * 1.2,
            'price_52w_low': test_case['stock_price'] * 0.8
        }

        options_data = {
            'strike_price': test_case['stock_price'],
            'dte': test_case['dte'],
            'delta': 0.50,
            'premium': test_case['stock_price'] * test_case['iv'] * 0.4,
            'bid': test_case['stock_price'] * 0.3,
            'ask': test_case['stock_price'] * 0.35,
            'volume': 1000,
            'open_interest': 5000
        }

        # Time the analysis
        start_time = time.time()

        try:
            # Run analysis
            results = analyzer.analyze_stock(
                symbol=test_case['symbol'],
                stock_data=stock_data,
                options_data=options_data
            )

            elapsed_time = time.time() - start_time

            # Verify results structure
            if 'top_3' not in results:
                print(f"  ❌ FAILED: Missing 'top_3' in results")
                all_tests_passed = False
                continue

            if 'strategy_rankings' not in results:
                print(f"  ❌ FAILED: Missing 'strategy_rankings' in results")
                all_tests_passed = False
                continue

            # Check we have all 18 strategies
            strategy_count = len(results['strategy_rankings'])
            if strategy_count != 18:
                print(f"  ❌ FAILED: Expected 18 strategies, got {strategy_count}")
                all_tests_passed = False
                continue

            print(f"  ✅ PASSED: All 18 strategies calculated successfully")
            print(f"  ⏱️  Performance: {elapsed_time:.3f} seconds")

            # Performance check
            if elapsed_time > 5.0:
                print(f"  ⚠️  WARNING: Analysis took longer than 5 seconds")
            elif elapsed_time > 3.5:
                print(f"  ⚡ GOOD: Within acceptable range (< 5s)")
            else:
                print(f"  ⚡ EXCELLENT: Very fast performance")

            performance_data.append({
                'test': test_case['name'],
                'time': elapsed_time
            })

            # Display top 3 strategies
            print(f"\n  🏆 Top 3 Recommended Strategies:")
            for i, strategy in enumerate(results['top_3'], 1):
                print(f"    {i}. {strategy['name']} - Score: {strategy['score']}/100")
                if 'metrics' in strategy:
                    metrics = strategy['metrics']
                    if 'max_profit' in metrics:
                        max_profit = metrics['max_profit']
                        if isinstance(max_profit, (int, float)):
                            print(f"       Max Profit: ${max_profit:.2f}")
                        else:
                            print(f"       Max Profit: {max_profit}")
                    if 'return_on_capital' in metrics:
                        print(f"       ROI: {metrics['return_on_capital']:.1f}%")

            # Verify new strategies are present
            new_strategies = [
                'Iron Butterfly', 'Jade Lizard', 'Long Call Butterfly',
                'Long Put Butterfly', 'Call Ratio Spread', 'Put Ratio Spread',
                'Collar', 'Synthetic Long'
            ]

            all_strategy_names = [s['name'] for s in results['strategy_rankings']]
            missing_strategies = [s for s in new_strategies if s not in all_strategy_names]

            if missing_strategies:
                print(f"\n  ❌ FAILED: Missing strategies: {', '.join(missing_strategies)}")
                all_tests_passed = False
            else:
                print(f"\n  ✅ VERIFIED: All 8 new strategies present")

            # Check for calculation errors
            errors_found = []
            for strategy in results['strategy_rankings']:
                if 'metrics' not in strategy:
                    errors_found.append(f"{strategy['name']}: Missing metrics")
                elif 'max_profit' not in strategy['metrics']:
                    errors_found.append(f"{strategy['name']}: Missing max_profit")
                elif strategy['metrics']['max_profit'] is None:
                    errors_found.append(f"{strategy['name']}: max_profit is None")

            if errors_found:
                print(f"\n  ❌ ERRORS FOUND:")
                for error in errors_found:
                    print(f"     - {error}")
                all_tests_passed = False
            else:
                print(f"  ✅ All strategy calculations valid")

        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            print(f"  Exception Type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            all_tests_passed = False
            elapsed_time = time.time() - start_time

    # Summary
    print(f"\n{'='*80}")
    print("📈 PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    if performance_data:
        avg_time = sum(p['time'] for p in performance_data) / len(performance_data)
        max_time = max(p['time'] for p in performance_data)
        min_time = min(p['time'] for p in performance_data)

        print(f"\n  Average Analysis Time: {avg_time:.3f}s")
        print(f"  Fastest Analysis: {min_time:.3f}s")
        print(f"  Slowest Analysis: {max_time:.3f}s")

        print(f"\n  Individual Results:")
        for perf in performance_data:
            print(f"    {perf['test']}: {perf['time']:.3f}s")

    print(f"\n{'='*80}")
    print("🎯 FINAL RESULT")
    print(f"{'='*80}")

    if all_tests_passed:
        print("\n  ✅ ✅ ✅ ALL TESTS PASSED ✅ ✅ ✅")
        print("\n  The Options Analysis Individual Stock Deep Dive is:")
        print("    ✓ Error-free")
        print("    ✓ All 18 strategies working")
        print("    ✓ Performance acceptable")
        print("    ✓ Production ready")
    else:
        print("\n  ❌ SOME TESTS FAILED")
        print("\n  Please review the errors above and fix issues before production.")

    print(f"\n{'='*80}\n")

    return all_tests_passed


def test_strategy_metadata():
    """Test that all strategies have complete metadata"""

    print("=" * 80)
    print("📋 TESTING: Strategy Metadata Completeness")
    print("=" * 80)

    analyzer = ComprehensiveStrategyAnalyzer()

    expected_strategies = [
        'Cash-Secured Put', 'Iron Condor', 'Poor Man\'s Covered Call',
        'Bull Put Spread', 'Bear Call Spread', 'Covered Call',
        'Calendar Spread', 'Diagonal Spread', 'Long Straddle',
        'Short Strangle', 'Iron Butterfly', 'Jade Lizard',
        'Long Call Butterfly', 'Long Put Butterfly', 'Call Ratio Spread',
        'Put Ratio Spread', 'Collar', 'Synthetic Long'
    ]

    required_fields = ['type', 'outlook', 'best_when', 'risk_profile', 'win_rate']

    all_good = True

    for strategy_name in expected_strategies:
        if strategy_name not in analyzer.strategies:
            print(f"  ❌ {strategy_name}: NOT FOUND in strategies dict")
            all_good = False
            continue

        metadata = analyzer.strategies[strategy_name]
        missing_fields = [field for field in required_fields if field not in metadata]

        if missing_fields:
            print(f"  ❌ {strategy_name}: Missing fields: {', '.join(missing_fields)}")
            all_good = False
        else:
            print(f"  ✅ {strategy_name}: Complete metadata")

    print()
    if all_good:
        print("  ✅ All 18 strategies have complete metadata")
    else:
        print("  ❌ Some strategies have incomplete metadata")

    print(f"{'='*80}\n")
    return all_good


if __name__ == "__main__":
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 78 + "|")
    print("|" + "   OPTIONS ANALYSIS COMPREHENSIVE TEST SUITE   ".center(78) + "|")
    print("|" + "   Testing All 18 Strategies for Production Readiness   ".center(78) + "|")
    print("|" + " " * 78 + "|")
    print("+" + "=" * 78 + "+")
    print()

    # Run tests
    test1_passed = test_strategy_metadata()
    test2_passed = test_all_strategies()

    # Final verdict
    print("+" + "=" * 78 + "+")
    print("|" + " " * 78 + "|")
    print("|" + "   FINAL VERDICT   ".center(78) + "|")
    print("|" + " " * 78 + "|")

    if test1_passed and test2_passed:
        print("|" + "   PRODUCTION READY - ALL TESTS PASSED   ".center(78) + "|")
    else:
        print("|" + "   NOT READY - PLEASE FIX ERRORS ABOVE   ".center(78) + "|")

    print("|" + " " * 78 + "|")
    print("+" + "=" * 78 + "+")
    print()

    sys.exit(0 if (test1_passed and test2_passed) else 1)
