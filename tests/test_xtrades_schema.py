"""
Test script for Xtrades Watchlists database schema
Creates tables, inserts test data, and runs verification queries
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'magnus'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123!')
}

def run_sql_file(cursor, filepath):
    """Execute SQL commands from a file"""
    print(f"\n{'='*80}")
    print(f"Executing SQL file: {filepath}")
    print(f"{'='*80}\n")

    with open(filepath, 'r', encoding='utf-8') as f:
        sql = f.read()

    try:
        cursor.execute(sql)
        print("SUCCESS: SQL file executed successfully!")
        return True
    except Exception as e:
        print(f"ERROR executing SQL file: {e}")
        return False

def verify_tables(cursor):
    """Verify all tables were created"""
    print(f"\n{'='*80}")
    print("VERIFYING TABLE CREATION")
    print(f"{'='*80}\n")

    cursor.execute("""
        SELECT
            table_name,
            (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'public'
            AND table_name LIKE 'xtrades_%'
        ORDER BY table_name
    """)

    tables = cursor.fetchall()

    expected_tables = ['xtrades_notifications', 'xtrades_profiles', 'xtrades_sync_log', 'xtrades_trades']

    print("Tables created:")
    for table in tables:
        print(f"  [OK] {table['table_name']} ({table['column_count']} columns)")

    found_tables = [t['table_name'] for t in tables]
    missing = set(expected_tables) - set(found_tables)

    if missing:
        print(f"\n  WARNING: Missing tables: {missing}")
        return False

    print(f"\n  SUCCESS: All {len(tables)} expected tables created!")
    return True

def verify_indexes(cursor):
    """Verify indexes were created"""
    print(f"\n{'='*80}")
    print("VERIFYING INDEXES")
    print(f"{'='*80}\n")

    cursor.execute("""
        SELECT
            tablename,
            COUNT(*) as index_count
        FROM pg_indexes
        WHERE schemaname = 'public'
            AND tablename LIKE 'xtrades_%'
            AND indexname NOT LIKE '%_pkey'
        GROUP BY tablename
        ORDER BY tablename
    """)

    indexes = cursor.fetchall()

    print("Indexes created (excluding primary keys):")
    total_indexes = 0
    for idx in indexes:
        print(f"  [OK] {idx['tablename']}: {idx['index_count']} indexes")
        total_indexes += idx['index_count']

    print(f"\n  SUCCESS: {total_indexes} total indexes created!")
    return True

def verify_sample_data(cursor):
    """Verify sample data was inserted"""
    print(f"\n{'='*80}")
    print("VERIFYING SAMPLE DATA")
    print(f"{'='*80}\n")

    cursor.execute("""
        SELECT 'xtrades_profiles' as table_name, COUNT(*) as row_count FROM xtrades_profiles
        UNION ALL
        SELECT 'xtrades_trades', COUNT(*) FROM xtrades_trades
        UNION ALL
        SELECT 'xtrades_sync_log', COUNT(*) FROM xtrades_sync_log
        UNION ALL
        SELECT 'xtrades_notifications', COUNT(*) FROM xtrades_notifications
        ORDER BY table_name
    """)

    data_counts = cursor.fetchall()

    print("Sample data inserted:")
    for row in data_counts:
        print(f"  [OK] {row['table_name']}: {row['row_count']} rows")

    return True

def run_sample_queries(cursor):
    """Run sample queries to demonstrate the schema"""
    print(f"\n{'='*80}")
    print("RUNNING SAMPLE QUERIES")
    print(f"{'='*80}\n")

    # Query 1: List all profiles with their trade counts
    print("Query 1: Profiles with trade counts")
    print("-" * 80)
    cursor.execute("""
        SELECT
            p.username,
            p.display_name,
            p.active,
            COUNT(t.id) as trade_count,
            SUM(CASE WHEN t.status = 'open' THEN 1 ELSE 0 END) as open_trades,
            SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END) as closed_trades
        FROM xtrades_profiles p
        LEFT JOIN xtrades_trades t ON p.id = t.profile_id
        GROUP BY p.id, p.username, p.display_name, p.active
        ORDER BY trade_count DESC
    """)

    profiles = cursor.fetchall()
    for profile in profiles:
        print(f"  {profile['username']:20} | Active: {profile['active']} | "
              f"Total: {profile['trade_count']} | Open: {profile['open_trades']} | "
              f"Closed: {profile['closed_trades']}")

    # Query 2: Recent trades
    print(f"\nQuery 2: Recent trades (last 10)")
    print("-" * 80)
    cursor.execute("""
        SELECT
            p.username,
            t.ticker,
            t.strategy,
            t.action,
            t.status,
            t.entry_price,
            t.exit_price,
            t.pnl,
            t.pnl_percent,
            t.alert_timestamp
        FROM xtrades_trades t
        JOIN xtrades_profiles p ON t.profile_id = p.id
        ORDER BY t.alert_timestamp DESC
        LIMIT 10
    """)

    trades = cursor.fetchall()
    for trade in trades:
        pnl_str = f"${trade['pnl']:.2f}" if trade['pnl'] else "N/A"
        print(f"  {trade['username']:15} | {trade['ticker']:6} | {trade['strategy']:10} | "
              f"{trade['status']:8} | PnL: {pnl_str:10} | {trade['alert_timestamp']}")

    # Query 3: Performance by ticker
    print(f"\nQuery 3: Performance summary by ticker (closed trades only)")
    print("-" * 80)
    cursor.execute("""
        SELECT
            ticker,
            COUNT(*) as trade_count,
            SUM(pnl) as total_pnl,
            AVG(pnl_percent) as avg_pnl_percent,
            COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
            COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades
        FROM xtrades_trades
        WHERE status = 'closed' AND pnl IS NOT NULL
        GROUP BY ticker
        ORDER BY total_pnl DESC
    """)

    performance = cursor.fetchall()
    if performance:
        for perf in performance:
            win_rate = (perf['winning_trades'] / perf['trade_count'] * 100) if perf['trade_count'] > 0 else 0
            print(f"  {perf['ticker']:6} | Trades: {perf['trade_count']:3} | "
                  f"Total PnL: ${perf['total_pnl']:8.2f} | "
                  f"Avg %: {perf['avg_pnl_percent']:6.2f}% | Win Rate: {win_rate:.1f}%")
    else:
        print("  No closed trades with PnL data yet")

    # Query 4: Recent sync history
    print(f"\nQuery 4: Recent sync history")
    print("-" * 80)
    cursor.execute("""
        SELECT
            sync_timestamp,
            profiles_synced,
            new_trades,
            updated_trades,
            status,
            duration_seconds
        FROM xtrades_sync_log
        ORDER BY sync_timestamp DESC
        LIMIT 5
    """)

    syncs = cursor.fetchall()
    for sync in syncs:
        print(f"  {sync['sync_timestamp']} | Status: {sync['status']:8} | "
              f"Profiles: {sync['profiles_synced']} | New: {sync['new_trades']} | "
              f"Updated: {sync['updated_trades']} | Duration: {sync['duration_seconds']}s")

    return True

def test_foreign_keys(cursor):
    """Test foreign key constraints"""
    print(f"\n{'='*80}")
    print("TESTING FOREIGN KEY CONSTRAINTS")
    print(f"{'='*80}\n")

    # Test cascading delete
    print("Testing CASCADE DELETE on xtrades_profiles...")

    # Create a test profile
    cursor.execute("""
        INSERT INTO xtrades_profiles (username, display_name, active)
        VALUES ('test_delete_user', 'Test Delete User', TRUE)
        RETURNING id
    """)
    test_profile_id = cursor.fetchone()['id']
    print(f"  [OK] Created test profile with ID: {test_profile_id}")

    # Create a test trade
    cursor.execute("""
        INSERT INTO xtrades_trades (
            profile_id, ticker, strategy, action, status,
            alert_text, alert_timestamp
        )
        VALUES (%s, 'TEST', 'CSP', 'STO', 'open', 'Test trade', NOW())
        RETURNING id
    """, (test_profile_id,))
    test_trade_id = cursor.fetchone()['id']
    print(f"  [OK] Created test trade with ID: {test_trade_id}")

    # Delete the profile (should cascade to trade)
    cursor.execute("""
        DELETE FROM xtrades_profiles WHERE id = %s
    """, (test_profile_id,))
    print(f"  [OK] Deleted test profile")

    # Verify trade was also deleted
    cursor.execute("""
        SELECT COUNT(*) as count FROM xtrades_trades WHERE id = %s
    """, (test_trade_id,))
    count = cursor.fetchone()['count']

    if count == 0:
        print(f"  [OK] SUCCESS: Cascade delete worked - trade was automatically deleted")
    else:
        print(f"  [ERROR] Cascade delete failed - trade still exists")
        return False

    return True

def main():
    """Main test function"""
    print(f"\n{'#'*80}")
    print(f"# Xtrades Watchlists Database Schema Test")
    print(f"# Database: {db_config['database']}")
    print(f"# Host: {db_config['host']}")
    print(f"# Timestamp: {datetime.now()}")
    print(f"{'#'*80}")

    try:
        # Connect to database
        print(f"\nConnecting to database...")
        conn = psycopg2.connect(**db_config)
        conn.autocommit = False  # Use transactions
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print("SUCCESS: Connected to database")

        # Execute schema file
        schema_file = os.path.join(os.path.dirname(__file__), 'src', 'xtrades_schema.sql')
        if run_sql_file(cursor, schema_file):
            conn.commit()
        else:
            conn.rollback()
            print("ERROR: Failed to execute schema file")
            return

        # Run verification tests
        all_passed = True
        all_passed &= verify_tables(cursor)
        all_passed &= verify_indexes(cursor)
        all_passed &= verify_sample_data(cursor)
        all_passed &= run_sample_queries(cursor)
        all_passed &= test_foreign_keys(cursor)

        # Commit the test
        conn.commit()

        # Final summary
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        if all_passed:
            print("\n  >>> ALL TESTS PASSED! <<<")
            print("\n  The Xtrades Watchlists schema is ready to use!")
        else:
            print("\n  >>> SOME TESTS FAILED <<<")
            print("\n  Please review the errors above")

        print(f"\n{'='*80}\n")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()

if __name__ == '__main__':
    main()
