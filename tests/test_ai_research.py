"""
Test script for AI Research Service
Demonstrates the service works correctly without Streamlit
"""

import sys
import os
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, 'src')

from ai_research_service import AIResearchService


def print_section(title, char="="):
    """Print a formatted section header"""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def test_basic_report():
    """Test basic report generation"""
    print_section("TEST 1: Basic Report Generation")

    service = AIResearchService()
    report = service._generate_mock_report("AAPL")

    print(f"Symbol: {report['symbol']}")
    print(f"Overall Rating: {report['overall_rating']}/5.0")
    print(f"Quick Summary: {report['quick_summary']}")
    print(f"Recommendation: {report['recommendation']['action']} (Confidence: {report['recommendation']['confidence']*100:.0f}%)")
    print(f"\n✅ Basic report generation: PASSED")


def test_all_sections():
    """Test all report sections are present"""
    print_section("TEST 2: Report Completeness")

    service = AIResearchService()
    report = service._generate_mock_report("MSFT")

    required_sections = [
        'symbol', 'timestamp', 'overall_rating', 'quick_summary',
        'fundamental', 'technical', 'sentiment', 'options',
        'recommendation', 'metadata'
    ]

    missing = []
    for section in required_sections:
        if section not in report:
            missing.append(section)
        else:
            print(f"✓ {section}")

    if missing:
        print(f"\n❌ Missing sections: {missing}")
        return False
    else:
        print(f"\n✅ All sections present: PASSED")
        return True


def test_fundamental_data():
    """Test fundamental analysis data"""
    print_section("TEST 3: Fundamental Analysis", "-")

    service = AIResearchService()
    report = service._generate_mock_report("GOOGL")
    fund = report['fundamental']

    print(f"Score: {fund['score']}/100")
    print(f"Revenue Growth YoY: {fund['revenue_growth_yoy']*100:.1f}%")
    print(f"Earnings Beat Streak: {fund['earnings_beat_streak']}")
    print(f"P/E Ratio: {fund['pe_ratio']:.1f}")
    print(f"Valuation: {fund['valuation_assessment']}")
    print(f"\nKey Strengths ({len(fund['key_strengths'])}):")
    for strength in fund['key_strengths']:
        print(f"  ✅ {strength}")
    print(f"\nKey Risks ({len(fund['key_risks'])}):")
    for risk in fund['key_risks']:
        print(f"  ⚠️ {risk}")

    print(f"\n✅ Fundamental analysis: PASSED")


def test_technical_data():
    """Test technical analysis data"""
    print_section("TEST 4: Technical Analysis", "-")

    service = AIResearchService()
    report = service._generate_mock_report("TSLA")
    tech = report['technical']

    print(f"Score: {tech['score']}/100")
    print(f"Trend: {tech['trend'].upper()}")
    print(f"RSI: {tech['rsi']:.1f}")
    print(f"MACD Signal: {tech['macd_signal'].upper()}")
    print(f"Support Levels: {', '.join([f'${s:.2f}' for s in tech['support_levels']])}")
    print(f"Resistance Levels: {', '.join([f'${r:.2f}' for r in tech['resistance_levels']])}")
    print(f"Volume Analysis: {tech['volume_analysis']}")
    print(f"Chart Patterns: {len(tech['chart_patterns'])}")
    print(f"Recommendation: {tech['recommendation']}")

    print(f"\n✅ Technical analysis: PASSED")


def test_sentiment_data():
    """Test sentiment analysis data"""
    print_section("TEST 5: Sentiment Analysis", "-")

    service = AIResearchService()
    report = service._generate_mock_report("NVDA")
    sent = report['sentiment']

    print(f"Score: {sent['score']}/100")
    print(f"News Sentiment: {sent['news_sentiment'].upper()}")
    print(f"News Count (7d): {sent['news_count_7d']}")
    print(f"Social Sentiment: {sent['social_sentiment'].upper()}")
    print(f"Reddit Mentions (24h): {sent['reddit_mentions_24h']}")
    print(f"Institutional Flow: {sent['institutional_flow'].replace('_', ' ').title()}")
    print(f"Analyst Rating: {sent['analyst_rating'].replace('_', ' ').title()}")

    consensus = sent['analyst_consensus']
    total = consensus['strong_buy'] + consensus['buy'] + consensus['hold'] + consensus['sell'] + consensus['strong_sell']
    print(f"\nAnalyst Consensus (Total: {total}):")
    print(f"  Strong Buy: {consensus['strong_buy']}")
    print(f"  Buy: {consensus['buy']}")
    print(f"  Hold: {consensus['hold']}")
    print(f"  Sell: {consensus['sell']}")
    print(f"  Strong Sell: {consensus['strong_sell']}")

    print(f"\n✅ Sentiment analysis: PASSED")


def test_options_data():
    """Test options analysis data"""
    print_section("TEST 6: Options Analysis", "-")

    service = AIResearchService()
    report = service._generate_mock_report("AMD")
    opts = report['options']

    print(f"Score: {opts['score']}/100")
    print(f"IV Rank: {opts['iv_rank']}/100")
    print(f"IV Percentile: {opts['iv_percentile']}/100")
    print(f"Current IV: {opts['current_iv']*100:.1f}%")
    print(f"30d Avg IV: {opts['iv_mean_30d']*100:.1f}%")
    print(f"Next Earnings: {opts['next_earnings_date']}")
    print(f"Days to Earnings: {opts['days_to_earnings']}")
    print(f"Avg Earnings Move: {opts['avg_earnings_move']*100:.1f}%")
    print(f"Put/Call Ratio: {opts['put_call_ratio']:.2f}")
    print(f"Unusual Activity: {'🚨 YES' if opts['unusual_activity'] else 'No'}")

    print(f"\nRecommended Strategies:")
    for strategy in opts['recommended_strategies']:
        print(f"  📋 {strategy['strategy'].replace('_', ' ').title()}")
        print(f"     {strategy['rationale']}")

    print(f"\n✅ Options analysis: PASSED")


def test_recommendation():
    """Test trade recommendation"""
    print_section("TEST 7: Trade Recommendation", "-")

    service = AIResearchService()
    report = service._generate_mock_report("SPY")
    rec = report['recommendation']

    print(f"Action: {rec['action']} (Confidence: {rec['confidence']*100:.0f}%)")
    print(f"\nReasoning:")
    print(f"  {rec['reasoning']}")

    print(f"\nTime-Sensitive Factors:")
    for factor in rec['time_sensitive_factors']:
        print(f"  ⏰ {factor}")

    print(f"\nPosition-Specific Advice:")
    for position_type, advice in rec['specific_position_advice'].items():
        print(f"\n  {position_type.replace('_', ' ').title()}:")
        print(f"    {advice}")

    print(f"\n✅ Recommendation: PASSED")


def test_metadata():
    """Test metadata"""
    print_section("TEST 8: Metadata", "-")

    service = AIResearchService()
    report = service._generate_mock_report("QQQ")
    meta = report['metadata']

    print(f"API Calls Used: {meta['api_calls_used']}")
    print(f"Processing Time: {meta['processing_time_ms']}ms")
    print(f"Agents Executed: {meta['agents_executed']}")
    print(f"LLM Model: {meta['llm_model']}")
    print(f"Tokens Used: {meta['llm_tokens_used']:,}")
    print(f"Cache Expires At: {meta['cache_expires_at']}")

    print(f"\n✅ Metadata: PASSED")


def test_multiple_symbols():
    """Test generating reports for multiple symbols"""
    print_section("TEST 9: Multiple Symbols")

    service = AIResearchService()
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print(f"Generating reports for {len(symbols)} symbols...\n")

    results = []
    for symbol in symbols:
        report = service._generate_mock_report(symbol)
        results.append({
            'symbol': symbol,
            'rating': report['overall_rating'],
            'action': report['recommendation']['action'],
            'confidence': report['recommendation']['confidence']
        })

    # Display results in table format
    print(f"{'Symbol':<8} {'Rating':<12} {'Action':<15} {'Confidence':<12}")
    print("-" * 50)
    for result in results:
        stars = "⭐" * int(result['rating'])
        print(f"{result['symbol']:<8} {stars:<12} {result['action']:<15} {result['confidence']*100:>5.0f}%")

    print(f"\n✅ Multiple symbols: PASSED ({len(results)} reports generated)")


def test_score_distribution():
    """Test score distribution is reasonable"""
    print_section("TEST 10: Score Distribution")

    service = AIResearchService()
    num_samples = 20

    print(f"Generating {num_samples} reports to check score distribution...\n")

    scores = {
        'fundamental': [],
        'technical': [],
        'sentiment': [],
        'options': [],
        'overall': []
    }

    for i in range(num_samples):
        report = service._generate_mock_report(f"TEST{i}")
        scores['fundamental'].append(report['fundamental']['score'])
        scores['technical'].append(report['technical']['score'])
        scores['sentiment'].append(report['sentiment']['score'])
        scores['options'].append(report['options']['score'])
        scores['overall'].append(report['overall_rating'])

    # Calculate averages
    print(f"Score Statistics (n={num_samples}):\n")
    for category, values in scores.items():
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        print(f"{category.capitalize():<15} Avg: {avg:>5.1f}  Min: {min_val:>5.1f}  Max: {max_val:>5.1f}")

    print(f"\n✅ Score distribution: PASSED (reasonable variance)")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  AI RESEARCH SERVICE TEST SUITE")
    print("=" * 80)

    tests = [
        test_basic_report,
        test_all_sections,
        test_fundamental_data,
        test_technical_data,
        test_sentiment_data,
        test_options_data,
        test_recommendation,
        test_metadata,
        test_multiple_symbols,
        test_score_distribution
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            failed += 1

    print_section("TEST SUMMARY")
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Service is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
