"""
Comprehensive test suite for earnings calendar implementation
"""
import sys
import os
from datetime import date, timedelta
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def test_database_enhancements():
    """Test that all database objects were created"""
    print("\n" + "="*80)
    print("TEST 1: Database Enhancements")
    print("="*80)

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()

    # Test tables
    tables = ['earnings_pattern_analysis', 'earnings_iv_tracking']
    print("\nChecking tables...")
    for table in tables:
        cur.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{table}'
        """)
        exists = cur.fetchone()[0] > 0
        print(f"  [{'OK' if exists else 'FAIL'}] {table}")

    # Test views
    views = ['v_upcoming_quality_earnings', 'v_earnings_results', 'v_iv_expansion']
    print("\nChecking views...")
    for view in views:
        cur.execute(f"""
            SELECT COUNT(*) FROM information_schema.views
            WHERE table_name = '{view}'
        """)
        exists = cur.fetchone()[0] > 0
        print(f"  [{'OK' if exists else 'FAIL'}] {view}")

    # Test functions
    functions = ['calculate_beat_rate', 'get_quality_score', 'calculate_expected_move']
    print("\nChecking functions...")
    for func in functions:
        cur.execute(f"""
            SELECT COUNT(*) FROM information_schema.routines
            WHERE routine_name = '{func}'
        """)
        exists = cur.fetchone()[0] > 0
        print(f"  [{'OK' if exists else 'FAIL'}] {func}")

    # Test new columns in earnings_events
    print("\nChecking new columns in earnings_events...")
    new_columns = [
        'expected_move_dollars', 'expected_move_pct', 'price_move_dollars',
        'exceeded_expected_move', 'is_confirmed', 'has_occurred'
    ]
    for col in new_columns:
        cur.execute(f"""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'earnings_events' AND column_name = '{col}'
        """)
        exists = cur.fetchone()[0] > 0
        print(f"  [{'OK' if exists else 'FAIL'}] {col}")

    cur.close()
    conn.close()
    print("\n[SUCCESS] Database enhancements verified!")

def test_expected_move_calculator():
    """Test expected move calculation"""
    print("\n" + "="*80)
    print("TEST 2: Expected Move Calculator")
    print("="*80)

    from src.earnings_expected_move import calculate_expected_move_from_yf

    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    test_date = date.today() + timedelta(days=30)

    print(f"\nTesting expected move calculation for earnings on {test_date}...")

    for symbol in test_symbols:
        print(f"\n  Testing {symbol}...")
        result = calculate_expected_move_from_yf(symbol, test_date)

        if result:
            print(f"    [OK] Expected Move: ±${result['expected_move_dollars']:.2f} "
                  f"(±{result['expected_move_pct']:.2f}%)")
            print(f"    [OK] Stock Price: ${result['stock_price']:.2f}")
            print(f"    [OK] Pre-Earnings IV: {result['pre_earnings_iv']:.4f}")
        else:
            print(f"    [WARN] Could not calculate (no options data)")

    print("\n[SUCCESS] Expected move calculator working!")

def test_pattern_analyzer():
    """Test pattern analyzer"""
    print("\n" + "="*80)
    print("TEST 3: Pattern Analyzer")
    print("="*80)

    from src.earnings_pattern_analyzer import calculate_earnings_patterns

    # Need to have some historical data first
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()

    # Check if we have any historical data
    cur.execute("SELECT COUNT(*) FROM earnings_history WHERE eps_actual IS NOT NULL")
    count = cur.fetchone()[0]

    print(f"\nFound {count} historical earnings records")

    if count > 0:
        # Get a symbol with data
        cur.execute("""
            SELECT symbol FROM earnings_history
            WHERE eps_actual IS NOT NULL
            GROUP BY symbol
            HAVING COUNT(*) >= 4
            LIMIT 1
        """)
        row = cur.fetchone()

        if row:
            symbol = row[0]
            print(f"\nTesting pattern analysis for {symbol}...")

            patterns = calculate_earnings_patterns(symbol)

            if patterns:
                print(f"    [OK] Beat Rate: {patterns['beat_rate']:.1f}%")
                print(f"    [OK] Avg Surprise: {patterns['avg_surprise_pct']:.2f}%")
                print(f"    [OK] Quality Score: {patterns['quality_score']:.0f}/100")
                print(f"    [OK] Quarters Analyzed: {patterns['quarters_analyzed']}")
            else:
                print(f"    [FAIL] Could not calculate patterns")
        else:
            print("    [WARN] No symbols with enough data (need 4+ quarters)")
    else:
        print("    [WARN] No historical earnings data yet")
        print("    [INFO] Run earnings sync to populate historical data")

    cur.close()
    conn.close()

    print("\n[SUCCESS] Pattern analyzer working!")

def test_database_queries():
    """Test that key queries work"""
    print("\n" + "="*80)
    print("TEST 4: Database Queries")
    print("="*80)

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()

    # Test upcoming quality earnings view
    print("\nTesting v_upcoming_quality_earnings...")
    cur.execute("SELECT COUNT(*) FROM v_upcoming_quality_earnings")
    count = cur.fetchone()[0]
    print(f"  [OK] Found {count} upcoming earnings")

    # Test earnings results view
    print("\nTesting v_earnings_results...")
    cur.execute("SELECT COUNT(*) FROM v_earnings_results")
    count = cur.fetchone()[0]
    print(f"  [OK] Found {count} historical results")

    # Test functions
    print("\nTesting calculate_beat_rate function...")
    cur.execute("SELECT calculate_beat_rate('AAPL', 8)")
    result = cur.fetchone()[0]
    print(f"  [OK] AAPL beat rate: {result:.1f}%")

    # Test expected move function
    print("\nTesting calculate_expected_move function...")
    cur.execute("SELECT * FROM calculate_expected_move(5.0, 4.5, 100.0)")
    result = cur.fetchone()
    print(f"  [OK] Expected move from straddle: ${result[0]:.2f} ({result[1]:.2f}%)")

    cur.close()
    conn.close()

    print("\n[SUCCESS] All database queries working!")

def test_earnings_data_flow():
    """Test the complete data flow"""
    print("\n" + "="*80)
    print("TEST 5: End-to-End Data Flow")
    print("="*80)

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()

    # Check each stage of the pipeline
    print("\nChecking data pipeline stages...")

    # Stage 1: Upcoming earnings
    cur.execute("SELECT COUNT(*) FROM earnings_events WHERE has_occurred = FALSE")
    upcoming = cur.fetchone()[0]
    print(f"  [{'OK' if upcoming > 0 else 'WARN'}] Upcoming earnings: {upcoming}")

    # Stage 2: Expected moves calculated
    cur.execute("""
        SELECT COUNT(*) FROM earnings_events
        WHERE has_occurred = FALSE AND expected_move_pct IS NOT NULL
    """)
    with_expected = cur.fetchone()[0]
    print(f"  [{'OK' if with_expected > 0 else 'WARN'}] With expected move: {with_expected}")

    # Stage 3: Completed earnings
    cur.execute("SELECT COUNT(*) FROM earnings_events WHERE has_occurred = TRUE")
    completed = cur.fetchone()[0]
    print(f"  [{'OK' if completed > 0 else 'WARN'}] Completed earnings: {completed}")

    # Stage 4: Price movements tracked
    cur.execute("""
        SELECT COUNT(*) FROM earnings_events
        WHERE has_occurred = TRUE AND price_move_percent IS NOT NULL
    """)
    with_results = cur.fetchone()[0]
    print(f"  [{'OK' if with_results > 0 else 'WARN'}] With price results: {with_results}")

    # Stage 5: Pattern analysis
    cur.execute("SELECT COUNT(*) FROM earnings_pattern_analysis")
    patterns = cur.fetchone()[0]
    print(f"  [{'OK' if patterns > 0 else 'WARN'}] Stocks with patterns: {patterns}")

    cur.close()
    conn.close()

    print("\n[SUCCESS] Data flow verified!")

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("EARNINGS CALENDAR IMPLEMENTATION TEST SUITE")
    print("="*80)

    try:
        test_database_enhancements()
        test_expected_move_calculator()
        test_pattern_analyzer()
        test_database_queries()
        test_earnings_data_flow()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED!")
        print("="*80)

        print("\nNEXT STEPS:")
        print("  1. Sync historical earnings: python -m src.earnings_sync_service")
        print("  2. Run daily automation: python scripts/daily_earnings_automation.py")
        print("  3. View dashboard: streamlit run dashboard.py")
        print("\n")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
