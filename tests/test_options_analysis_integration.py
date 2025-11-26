"""
Comprehensive Integration Test Suite for AI Options Analysis System

Tests all components:
1. Import validation
2. Database connectivity
3. Component functionality (PaginatedTable, StockDropdown, WatchlistSelector)
4. Scoring engine (all 5 scorers)
5. AI Options Agent
6. Database manager
7. End-to-end workflow

Run this test to validate the entire Options Analysis implementation.
"""

import sys
import os
import traceback
from typing import Dict, Any, List
from datetime import datetime, date
import pandas as pd

# Test result tracking
test_results = []


class TestResult:
    """Track individual test results"""
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.passed = False
        self.error = None
        self.duration = 0
        self.details = None


def log_test(test_name: str, category: str = "General"):
    """Decorator to log test results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = TestResult(test_name, category)
            start_time = datetime.now()

            try:
                print(f"\n{'='*80}")
                print(f"TEST: {test_name}")
                print(f"Category: {category}")
                print(f"{'='*80}")

                output = func(*args, **kwargs)
                result.passed = True
                result.details = output

                print(f"[PASS] {test_name}")

            except Exception as e:
                result.passed = False
                result.error = str(e)

                print(f"[FAIL] {test_name}")
                print(f"Error: {e}")
                print(f"Traceback:\n{traceback.format_exc()}")

            finally:
                result.duration = (datetime.now() - start_time).total_seconds()
                test_results.append(result)
                print(f"Duration: {result.duration:.2f}s")

            return result

        return wrapper
    return decorator


# ============================================================================
# IMPORT TESTS
# ============================================================================

@log_test("Import Core Components", "Imports")
def test_imports_core():
    """Test importing core AI Options Agent components"""
    from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager
    from src.ai_options_agent.scoring_engine import (
        MultiCriteriaScorer,
        FundamentalScorer,
        TechnicalScorer,
        GreeksScorer,
        RiskScorer,
        SentimentScorer
    )

    return {
        'OptionsAnalysisAgent': OptionsAnalysisAgent,
        'AIOptionsDBManager': AIOptionsDBManager,
        'MultiCriteriaScorer': MultiCriteriaScorer
    }


@log_test("Import UI Components", "Imports")
def test_imports_ui_components():
    """Test importing UI components"""
    from src.components.paginated_table import PaginatedTable, create_paginated_dataframe
    from src.components.stock_dropdown import StockDropdown, WatchlistSelector
    from src.ai_options_agent.shared.display_helpers import (
        display_score_gauge,
        display_recommendation_badge
    )

    return {
        'PaginatedTable': PaginatedTable,
        'StockDropdown': StockDropdown,
        'WatchlistSelector': WatchlistSelector,
        'display_score_gauge': display_score_gauge,
        'display_recommendation_badge': display_recommendation_badge
    }


@log_test("Import Main Page", "Imports")
def test_import_main_page():
    """Test importing options_analysis_page.py"""
    import options_analysis_page

    assert hasattr(options_analysis_page, 'render_options_analysis_page'), \
        "Main render function not found"
    assert hasattr(options_analysis_page, 'render_batch_analysis_mode'), \
        "Batch analysis mode not found"
    assert hasattr(options_analysis_page, 'render_individual_stock_mode'), \
        "Individual stock mode not found"

    return {'module': options_analysis_page}


# ============================================================================
# DATABASE TESTS
# ============================================================================

@log_test("Database Connection", "Database")
def test_database_connection():
    """Test database connectivity"""
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

    db_manager = AIOptionsDBManager()
    conn = db_manager.get_connection()

    assert conn is not None, "Database connection failed"

    # Test query
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()

    cur.close()
    conn.close()

    return {'postgres_version': version[0]}


@log_test("Get Opportunities Query", "Database")
def test_get_opportunities():
    """Test get_opportunities database query"""
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

    db_manager = AIOptionsDBManager()

    # Test basic query
    opportunities = db_manager.get_opportunities(
        symbols=None,
        dte_range=(20, 40),
        delta_range=(-0.45, -0.15),
        min_premium=50,
        limit=10
    )

    assert isinstance(opportunities, list), "Expected list of opportunities"

    if opportunities:
        # Validate structure
        first_opp = opportunities[0]
        required_fields = ['symbol', 'strike_price', 'dte', 'delta', 'premium']

        for field in required_fields:
            assert field in first_opp, f"Missing required field: {field}"

    return {
        'count': len(opportunities),
        'sample': opportunities[0] if opportunities else None
    }


@log_test("Get Watchlist Symbols", "Database")
def test_get_watchlist_symbols():
    """Test getting watchlist symbols from database"""
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

    db_manager = AIOptionsDBManager()

    # First, get all watchlists
    watchlists = db_manager.get_all_watchlists()

    if watchlists:
        # Test getting symbols for first watchlist
        watchlist_name = watchlists[0]['name']
        symbols = db_manager.get_watchlist_symbols(watchlist_name)

        return {
            'watchlists_count': len(watchlists),
            'test_watchlist': watchlist_name,
            'symbols_count': len(symbols),
            'symbols': symbols[:5] if symbols else []
        }
    else:
        return {
            'watchlists_count': 0,
            'note': 'No watchlists found in database'
        }


# ============================================================================
# COMPONENT TESTS
# ============================================================================

@log_test("PaginatedTable Class Structure", "Components")
def test_paginated_table_structure():
    """Test PaginatedTable class structure"""
    from src.components.paginated_table import PaginatedTable

    # Create sample DataFrame
    df = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        'Score': [85, 78, 92, 65, 88],
        'Premium': [2.50, 3.25, 4.10, 1.75, 3.50]
    })

    # Initialize PaginatedTable
    table = PaginatedTable(
        df=df,
        key_prefix="test_table",
        page_size=2,
        show_export=True,
        show_page_size_selector=True
    )

    # Verify attributes
    assert table.df is not None, "DataFrame not set"
    assert table.key_prefix == "test_table", "Key prefix not set"
    assert table.page_size == 2, "Page size not set"
    assert hasattr(table, 'render'), "render() method not found"
    assert hasattr(table, '_get_sorted_df'), "_get_sorted_df() method not found"
    assert hasattr(table, '_render_controls'), "_render_controls() method not found"

    return {
        'df_rows': len(df),
        'df_columns': list(df.columns),
        'methods': [m for m in dir(table) if not m.startswith('_')]
    }


@log_test("StockDropdown Class Structure", "Components")
def test_stock_dropdown_structure():
    """Test StockDropdown class structure"""
    from src.components.stock_dropdown import StockDropdown
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

    db_manager = AIOptionsDBManager()
    dropdown = StockDropdown(db_manager=db_manager)

    # Verify methods
    assert hasattr(dropdown, 'render'), "render() method not found"
    assert hasattr(dropdown, 'render_multiselect'), "render_multiselect() method not found"
    assert hasattr(dropdown, '_get_stock_list'), "_get_stock_list() method not found"
    assert hasattr(dropdown, '_format_stock_option'), "_format_stock_option() method not found"

    return {
        'methods': [m for m in dir(dropdown) if not m.startswith('__')]
    }


@log_test("WatchlistSelector Class Structure", "Components")
def test_watchlist_selector_structure():
    """Test WatchlistSelector class structure"""
    from src.components.stock_dropdown import WatchlistSelector
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

    db_manager = AIOptionsDBManager()
    selector = WatchlistSelector(db_manager=db_manager)

    # Verify methods
    assert hasattr(selector, 'render'), "render() method not found"
    assert hasattr(selector, '_get_watchlists'), "_get_watchlists() method not found"
    assert hasattr(selector, '_get_watchlist_symbols'), "_get_watchlist_symbols() method not found"

    return {
        'methods': [m for m in dir(selector) if not m.startswith('__')]
    }


# ============================================================================
# SCORING ENGINE TESTS
# ============================================================================

@log_test("FundamentalScorer", "Scoring Engine")
def test_fundamental_scorer():
    """Test FundamentalScorer"""
    from src.ai_options_agent.scoring_engine import FundamentalScorer

    scorer = FundamentalScorer()

    # Test opportunity
    opportunity = {
        'symbol': 'AAPL',
        'pe_ratio': 25.5,
        'eps': 6.15,
        'market_cap': 2_700_000_000_000,
        'sector': 'Technology',
        'dividend_yield': 0.52
    }

    score = scorer.score(opportunity)

    assert isinstance(score, int), "Score should be integer"
    assert 0 <= score <= 100, f"Score should be 0-100, got {score}"

    return {
        'score': score,
        'opportunity': opportunity
    }


@log_test("TechnicalScorer", "Scoring Engine")
def test_technical_scorer():
    """Test TechnicalScorer"""
    from src.ai_options_agent.scoring_engine import TechnicalScorer

    scorer = TechnicalScorer()

    opportunity = {
        'symbol': 'AAPL',
        'stock_price': 175.50,
        'strike_price': 165.00,
        'volume': 1250,
        'oi': 3500,
        'bid': 2.80,
        'ask': 2.90
    }

    score = scorer.score(opportunity)

    assert isinstance(score, int), "Score should be integer"
    assert 0 <= score <= 100, f"Score should be 0-100, got {score}"

    return {
        'score': score,
        'opportunity': opportunity
    }


@log_test("GreeksScorer", "Scoring Engine")
def test_greeks_scorer():
    """Test GreeksScorer"""
    from src.ai_options_agent.scoring_engine import GreeksScorer

    scorer = GreeksScorer()

    opportunity = {
        'symbol': 'AAPL',
        'delta': -0.28,
        'iv': 0.32,
        'premium': 285.0,
        'strike_price': 165.00,
        'dte': 30
    }

    score = scorer.score(opportunity)

    assert isinstance(score, int), "Score should be integer"
    assert 0 <= score <= 100, f"Score should be 0-100, got {score}"

    return {
        'score': score,
        'opportunity': opportunity
    }


@log_test("RiskScorer", "Scoring Engine")
def test_risk_scorer():
    """Test RiskScorer"""
    from src.ai_options_agent.scoring_engine import RiskScorer

    scorer = RiskScorer()

    opportunity = {
        'symbol': 'AAPL',
        'strike_price': 165.00,
        'premium': 285.0,
        'delta': -0.28,
        'stock_price': 175.50,
        'breakeven': 162.15,
        'annual_return': 20.76
    }

    score = scorer.score(opportunity)

    assert isinstance(score, int), "Score should be integer"
    assert 0 <= score <= 100, f"Score should be 0-100, got {score}"

    return {
        'score': score,
        'opportunity': opportunity
    }


@log_test("SentimentScorer", "Scoring Engine")
def test_sentiment_scorer():
    """Test SentimentScorer (stub)"""
    from src.ai_options_agent.scoring_engine import SentimentScorer

    scorer = SentimentScorer()

    opportunity = {
        'symbol': 'AAPL'
    }

    score = scorer.score(opportunity)

    # Should return 70 (neutral) for stub implementation
    assert score == 70, f"Stub should return 70, got {score}"

    return {
        'score': score,
        'note': 'Stub implementation returns neutral score'
    }


@log_test("MultiCriteriaScorer", "Scoring Engine")
def test_multi_criteria_scorer():
    """Test MultiCriteriaScorer (MCDM)"""
    from src.ai_options_agent.scoring_engine import MultiCriteriaScorer

    scorer = MultiCriteriaScorer()

    # Complete test opportunity
    opportunity = {
        'symbol': 'AAPL',
        'stock_price': 175.50,
        'strike_price': 165.00,
        'expiration_date': date(2025, 12, 19),
        'dte': 30,
        'premium': 285.0,
        'delta': -0.28,
        'monthly_return': 1.73,
        'annual_return': 20.76,
        'iv': 0.32,
        'bid': 2.80,
        'ask': 2.90,
        'volume': 1250,
        'oi': 3500,
        'breakeven': 162.15,
        'pe_ratio': 28.5,
        'eps': 6.15,
        'market_cap': 2_700_000_000_000,
        'sector': 'Technology',
        'dividend_yield': 0.52
    }

    result = scorer.score_opportunity(opportunity)

    # Validate result structure
    required_fields = [
        'symbol', 'fundamental_score', 'technical_score', 'greeks_score',
        'risk_score', 'sentiment_score', 'final_score', 'recommendation', 'confidence'
    ]

    for field in required_fields:
        assert field in result, f"Missing field in result: {field}"

    # Validate scores
    assert 0 <= result['final_score'] <= 100, "Final score out of range"
    assert result['recommendation'] in ['STRONG_BUY', 'BUY', 'HOLD', 'CAUTION', 'AVOID'], \
        f"Invalid recommendation: {result['recommendation']}"

    return result


# ============================================================================
# AI OPTIONS AGENT TESTS
# ============================================================================

@log_test("OptionsAnalysisAgent Initialization", "Agent")
def test_agent_initialization():
    """Test OptionsAnalysisAgent initialization"""
    from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent

    agent = OptionsAnalysisAgent()

    assert agent.db_manager is not None, "DB manager not initialized"
    assert agent.scorer is not None, "Scorer not initialized"
    assert hasattr(agent, 'analyze_opportunity'), "analyze_opportunity method not found"
    assert hasattr(agent, 'analyze_watchlist'), "analyze_watchlist method not found"
    assert hasattr(agent, 'analyze_all_stocks'), "analyze_all_stocks method not found"

    return {
        'db_manager': str(type(agent.db_manager)),
        'scorer': str(type(agent.scorer)),
        'llm_manager': str(type(agent.llm_manager)) if agent.llm_manager else None
    }


@log_test("Analyze Single Opportunity", "Agent")
def test_analyze_opportunity():
    """Test analyzing a single opportunity"""
    from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent

    agent = OptionsAnalysisAgent()

    # Test opportunity
    opportunity = {
        'symbol': 'AAPL',
        'stock_price': 175.50,
        'current_price': 175.50,
        'strike_price': 165.00,
        'expiration_date': date(2025, 12, 19),
        'dte': 30,
        'premium': 285.0,
        'delta': -0.28,
        'monthly_return': 1.73,
        'annual_return': 20.76,
        'iv': 0.32,
        'bid': 2.80,
        'ask': 2.90,
        'volume': 1250,
        'oi': 3500,
        'breakeven': 162.15,
        'pe_ratio': 28.5,
        'eps': 6.15,
        'market_cap': 2_700_000_000_000,
        'sector': 'Technology',
        'dividend_yield': 0.52
    }

    analysis = agent.analyze_opportunity(opportunity, save_to_db=False, use_llm=False)

    # Validate analysis structure
    required_fields = [
        'symbol', 'final_score', 'recommendation', 'reasoning',
        'key_risks', 'key_opportunities', 'strategy', 'llm_model'
    ]

    for field in required_fields:
        assert field in analysis, f"Missing field in analysis: {field}"

    assert analysis['llm_model'] == 'rule_based_v1', "Should use rule-based reasoning"

    return analysis


@log_test("Analyze from Database", "Agent")
def test_analyze_from_database():
    """Test analyzing opportunities from database"""
    from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent

    agent = OptionsAnalysisAgent()

    # Get opportunities from database
    opportunities = agent.db_manager.get_opportunities(
        symbols=None,
        dte_range=(20, 40),
        delta_range=(-0.45, -0.15),
        min_premium=50,
        limit=3
    )

    if not opportunities:
        return {
            'note': 'No opportunities in database',
            'analyzed': 0
        }

    # Analyze first opportunity
    analysis = agent.analyze_opportunity(opportunities[0], save_to_db=False)

    return {
        'opportunities_found': len(opportunities),
        'analyzed_symbol': analysis['symbol'],
        'final_score': analysis['final_score'],
        'recommendation': analysis['recommendation']
    }


# ============================================================================
# END-TO-END WORKFLOW TESTS
# ============================================================================

@log_test("End-to-End Batch Analysis", "Workflow")
def test_e2e_batch_analysis():
    """Test complete batch analysis workflow"""
    from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent

    agent = OptionsAnalysisAgent()

    # Run batch analysis on limited set
    analyses = agent.analyze_all_stocks(
        dte_range=(20, 40),
        delta_range=(-0.45, -0.15),
        min_premium=100,
        limit=5,
        use_llm=False
    )

    if not analyses:
        return {
            'note': 'No opportunities found for analysis',
            'analyzed': 0
        }

    # Validate results
    for analysis in analyses:
        assert 'final_score' in analysis, "Missing final_score"
        assert 'recommendation' in analysis, "Missing recommendation"

    # Check sorting (should be by final_score DESC)
    if len(analyses) > 1:
        assert analyses[0]['final_score'] >= analyses[1]['final_score'], \
            "Results not sorted correctly"

    return {
        'opportunities_analyzed': len(analyses),
        'top_pick': {
            'symbol': analyses[0]['symbol'],
            'score': analyses[0]['final_score'],
            'recommendation': analyses[0]['recommendation']
        },
        'score_distribution': {
            'strong_buy': len([a for a in analyses if a['recommendation'] == 'STRONG_BUY']),
            'buy': len([a for a in analyses if a['recommendation'] == 'BUY']),
            'hold': len([a for a in analyses if a['recommendation'] == 'HOLD']),
            'caution': len([a for a in analyses if a['recommendation'] == 'CAUTION']),
            'avoid': len([a for a in analyses if a['recommendation'] == 'AVOID'])
        }
    }


@log_test("Save and Retrieve Analysis", "Workflow")
def test_save_retrieve_analysis():
    """Test saving and retrieving analysis from database"""
    from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent
    from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

    agent = OptionsAnalysisAgent()
    db_manager = AIOptionsDBManager()

    # Get an opportunity
    opportunities = db_manager.get_opportunities(limit=1)

    if not opportunities:
        return {'note': 'No opportunities to test save/retrieve'}

    # Analyze and save
    analysis = agent.analyze_opportunity(opportunities[0], save_to_db=True)

    assert 'id' in analysis, "Analysis ID not returned after save"

    # Retrieve recent analyses
    recent = db_manager.get_recent_analyses(days=7, limit=10)

    assert len(recent) > 0, "No recent analyses found"

    return {
        'saved_id': analysis['id'],
        'recent_count': len(recent),
        'saved_symbol': analysis['symbol']
    }


# ============================================================================
# TEST RUNNER
# ============================================================================

def generate_report():
    """Generate comprehensive test report"""
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)

    # Group by category
    categories = {}
    for result in test_results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)

    # Overall stats
    total_tests = len(test_results)
    passed_tests = len([r for r in test_results if r.passed])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r.duration for r in test_results)

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests} [PASS]")
    print(f"Failed: {failed_tests} [FAIL]")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    print(f"Total Duration: {total_duration:.2f}s")

    # Category breakdown
    print("\n" + "-"*80)
    print("RESULTS BY CATEGORY")
    print("-"*80)

    for category, results in sorted(categories.items()):
        cat_passed = len([r for r in results if r.passed])
        cat_total = len(results)

        print(f"\n{category} ({cat_passed}/{cat_total})")
        print("-" * 40)

        for result in results:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"  {status} | {result.name} ({result.duration:.2f}s)")
            if not result.passed and result.error:
                print(f"          Error: {result.error}")

    # Failed tests detail
    failed = [r for r in test_results if not r.passed]
    if failed:
        print("\n" + "="*80)
        print("FAILED TESTS DETAIL")
        print("="*80)

        for result in failed:
            print(f"\n[FAIL] {result.name}")
            print(f"   Category: {result.category}")
            print(f"   Error: {result.error}")

    # Final verdict
    print("\n" + "="*80)
    if failed_tests == 0:
        print("SUCCESS: ALL TESTS PASSED! System is ready for production.")
    else:
        print(f"WARNING: {failed_tests} TEST(S) FAILED - Review errors above")
    print("="*80 + "\n")

    return {
        'total': total_tests,
        'passed': passed_tests,
        'failed': failed_tests,
        'duration': total_duration,
        'categories': {cat: len(results) for cat, results in categories.items()}
    }


def main():
    """Run all tests"""
    print("="*80)
    print("AI OPTIONS ANALYSIS - COMPREHENSIVE INTEGRATION TEST SUITE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Import Tests
    test_imports_core()
    test_imports_ui_components()
    test_import_main_page()

    # Database Tests
    test_database_connection()
    test_get_opportunities()
    test_get_watchlist_symbols()

    # Component Tests
    test_paginated_table_structure()
    test_stock_dropdown_structure()
    test_watchlist_selector_structure()

    # Scoring Engine Tests
    test_fundamental_scorer()
    test_technical_scorer()
    test_greeks_scorer()
    test_risk_scorer()
    test_sentiment_scorer()
    test_multi_criteria_scorer()

    # Agent Tests
    test_agent_initialization()
    test_analyze_opportunity()
    test_analyze_from_database()

    # Workflow Tests
    test_e2e_batch_analysis()
    test_save_retrieve_analysis()

    # Generate report
    report = generate_report()

    return report


if __name__ == "__main__":
    try:
        report = main()

        # Exit with appropriate code
        sys.exit(0 if report['failed'] == 0 else 1)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test suite crashed")
        print(f"Error: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(2)
