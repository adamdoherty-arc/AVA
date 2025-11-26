"""
Test script for Premium Options Flow feature

This script tests all components of the Premium Options Flow system:
- Database migration
- Options flow data collection
- AI analysis
- Database integration
"""

import sys
import os
import logging
from datetime import date

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.options_flow_tracker import OptionsFlowTracker, get_popular_optionable_symbols
from src.ai_flow_analyzer import AIFlowAnalyzer
from src.tradingview_db_manager import TradingViewDBManager

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_migration():
  """Test that migration can be executed"""
  print("\n" + "="*60)
  print("TEST 1: Database Migration")
  print("="*60)

  try:
    tv_manager = TradingViewDBManager()
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    # Check if tables exist
    cur.execute("""
      SELECT table_name
      FROM information_schema.tables
      WHERE table_name IN ('options_flow', 'options_flow_analysis',
                'premium_flow_opportunities', 'options_flow_alerts')
    """)

    tables = cur.fetchall()
    cur.close()
    conn.close()

    if len(tables) == 4:
      print("[PASS] All 4 tables exist")
      return True
    else:
      print(f"[WARN] Only {len(tables)} tables found. Run migration manually:")
      print("  psql -U postgres -d magnus -f migrations/add_premium_options_flow.sql")
      return False

  except Exception as e:
    print(f"[FAIL] Migration test failed: {e}")
    return False


def test_flow_data_collection():
  """Test options flow data collection"""
  print("\n" + "="*60)
  print("TEST 2: Options Flow Data Collection")
  print("="*60)

  try:
    tracker = OptionsFlowTracker()

    # Test single symbol
    test_symbol = 'AAPL'
    print(f"\n Testing flow collection for {test_symbol}...")

    flow_data = tracker.fetch_options_flow(test_symbol)

    if flow_data:
      print(f"[PASS] Successfully fetched flow data for {test_symbol}")
      print(f"  Call Volume: {flow_data.call_volume:,}")
      print(f"  Put Volume: {flow_data.put_volume:,}")
      print(f"  Net Premium Flow: ${flow_data.net_premium_flow:,.2f}")
      print(f"  Put/Call Ratio: {flow_data.put_call_ratio:.2f}")
      print(f"  Sentiment: {flow_data.flow_sentiment}")

      # Test saving to database
      if tracker.save_flow_data(flow_data):
        print(f"[PASS] Successfully saved flow data to database")
      else:
        print(f"[FAIL] Failed to save flow data")
        return False

      # Test metrics calculation
      metrics = tracker.calculate_flow_metrics(test_symbol)
      print(f"\nFlow Metrics:")
      print(f"  Average Volume: {metrics['avg_volume']:,.0f}")
      print(f"  Volume Percentile: {metrics['volume_percentile']:.0f}%")
      print(f"  Unusual Activity: {metrics['is_unusual']}")
      print(f"  7-Day Trend: {metrics['trend_7d']}")

      return True
    else:
      print(f"[FAIL] Failed to fetch flow data for {test_symbol}")
      return False

  except Exception as e:
    print(f"[FAIL] Flow data collection test failed: {e}")
    import traceback
    traceback.print_exc()
    return False


def test_ai_analysis():
  """Test AI flow analysis"""
  print("\n" + "="*60)
  print("TEST 3: AI Flow Analysis")
  print("="*60)

  try:
    analyzer = AIFlowAnalyzer()
    test_symbol = 'AAPL'

    # Create sample flow data
    flow_data = {
      'symbol': test_symbol,
      'put_call_ratio': 0.65,
      'net_premium_flow': 15000000,
      'call_premium': 25000000,
      'put_premium': 10000000,
      'total_volume': 50000,
      'flow_sentiment': 'Bullish',
      'trend_7d': 'Increasing',
      'unusual_activity': True,
      'volume_percentile': 85
    }

    print(f"\n Testing AI analysis for {test_symbol}...")

    # Test sentiment analysis
    sentiment = analyzer.analyze_flow_sentiment(flow_data)
    print(f" Sentiment Analysis: {sentiment}")

    # Test opportunity scoring
    score = analyzer.score_flow_opportunity(test_symbol, flow_data)
    print(f" Opportunity Score: {score:.1f}/100")

    # Test action recommendation
    action = analyzer.recommend_best_action(flow_data, [flow_data])
    print(f" Recommended Action: {action}")

    # Test risk assessment
    risk = analyzer.assess_flow_risk(test_symbol, flow_data)
    print(f" Risk Level: {risk}")

    # Test full analysis
    print(f"\n Running full flow analysis...")
    result = analyzer.generate_flow_insights(test_symbol, flow_data, [flow_data])

    print(f"\n Full Analysis Results:")
    print(f"  Opportunity Score: {result.opportunity_score:.1f}/100")
    print(f"  Best Action: {result.best_action}")
    print(f"  Risk Level: {result.risk_level}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"\n  AI Recommendation:")
    print(f"  {result.ai_recommendation}")
    print(f"\n  Key Insights:")
    for insight in result.key_insights:
      print(f"  - {insight}")

    # Test saving analysis
    if analyzer.save_analysis(result):
      print(f"\n Successfully saved AI analysis to database")
      return True
    else:
      print(f"\n Failed to save AI analysis")
      return False

  except Exception as e:
    print(f" AI analysis test failed: {e}")
    import traceback
    traceback.print_exc()
    return False


def test_batch_operations():
  """Test batch data collection and analysis"""
  print("\n" + "="*60)
  print("TEST 4: Batch Operations")
  print("="*60)

  try:
    tracker = OptionsFlowTracker()
    analyzer = AIFlowAnalyzer()

    # Test batch flow collection
    print("\n Testing batch flow collection (5 symbols)...")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    results = tracker.batch_update_flow(test_symbols, limit=5)
    print(f" Batch update results:")
    print(f"  Success: {results['success']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Skipped: {results['skipped']}")

    if results['success'] > 0:
      # Test batch analysis
      print(f"\n Testing batch AI analysis...")
      analyzed = analyzer.batch_analyze(test_symbols)
      print(f" Analyzed {analyzed} symbols")

      # Test getting top opportunities
      print(f"\n Testing top opportunities retrieval...")
      opportunities = tracker.get_top_flow_opportunities(limit=10)
      print(f" Retrieved {len(opportunities)} opportunities")

      if opportunities:
        print(f"\n  Top Opportunity:")
        opp = opportunities[0]
        print(f"  Symbol: {opp['symbol']}")
        print(f"  Score: {opp['opportunity_score']:.1f}")
        print(f"  Action: {opp['best_action']}")

      return True
    else:
      print(f" No successful flow collections in batch")
      return False

  except Exception as e:
    print(f" Batch operations test failed: {e}")
    import traceback
    traceback.print_exc()
    return False


def test_market_summary():
  """Test market flow summary"""
  print("\n" + "="*60)
  print("TEST 5: Market Flow Summary")
  print("="*60)

  try:
    tracker = OptionsFlowTracker()

    summary = tracker.get_market_flow_summary()

    if summary:
      print(f" Market Flow Summary:")
      print(f"  Total Symbols: {summary.get('total_symbols', 0)}")
      print(f"  Total Call Premium: ${summary.get('total_call_premium', 0):,.0f}")
      print(f"  Total Put Premium: ${summary.get('total_put_premium', 0):,.0f}")
      print(f"  Net Flow: ${summary.get('total_net_flow', 0):,.0f}")
      print(f"  Avg P/C Ratio: {summary.get('avg_put_call_ratio', 0):.2f}")
      print(f"  Bullish Symbols: {summary.get('bullish_count', 0)}")
      print(f"  Bearish Symbols: {summary.get('bearish_count', 0)}")
      print(f"  Neutral Symbols: {summary.get('neutral_count', 0)}")
      print(f"  Unusual Activity: {summary.get('unusual_count', 0)}")
      return True
    else:
      print(f" No market summary data available")
      return False

  except Exception as e:
    print(f" Market summary test failed: {e}")
    import traceback
    traceback.print_exc()
    return False


def main():
  """Run all tests"""
  print("\n" + "="*60)
  print("PREMIUM OPTIONS FLOW - COMPREHENSIVE TEST SUITE")
  print("="*60)

  results = {
    'migration': test_database_migration(),
    'data_collection': test_flow_data_collection(),
    'ai_analysis': test_ai_analysis(),
    'batch_operations': test_batch_operations(),
    'market_summary': test_market_summary()
  }

  print("\n" + "="*60)
  print("TEST RESULTS SUMMARY")
  print("="*60)

  total_tests = len(results)
  passed_tests = sum(1 for v in results.values() if v)

  for test_name, result in results.items():
    status = " PASSED" if result else " FAILED"
    print(f"{test_name.upper()}: {status}")

  print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

  if passed_tests == total_tests:
    print("\n All tests passed! Premium Options Flow feature is ready to use.")
    print("\nNext steps:")
    print("1. Start the dashboard: streamlit run dashboard.py")
    print("2. Navigate to 'Premium Options Flow' in the sidebar")
    print("3. Click 'Refresh Flow Data' to collect live data")
    print("4. Click 'Run AI Analysis' to generate recommendations")
  else:
    print(f"\n {total_tests - passed_tests} test(s) failed. Review errors above.")

  return passed_tests == total_tests


if __name__ == "__main__":
  success = main()
  sys.exit(0 if success else 1)
