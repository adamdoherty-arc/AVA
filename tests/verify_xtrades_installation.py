"""
Quick verification script to check Xtrades schema installation
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'magnus'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123!')
}

def verify_installation():
    """Quick check that Xtrades schema is installed"""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        print("="*60)
        print("XTRADES SCHEMA VERIFICATION")
        print("="*60)

        # Check tables
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'xtrades_%'
        """)
        table_count = cursor.fetchone()['count']

        if table_count == 4:
            print(f"\n[OK] All 4 Xtrades tables found")
        else:
            print(f"\n[ERROR] Expected 4 tables, found {table_count}")
            return False

        # Check indexes
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename LIKE 'xtrades_%'
            AND indexname NOT LIKE '%_pkey'
        """)
        index_count = cursor.fetchone()['count']

        if index_count >= 15:
            print(f"[OK] {index_count} indexes created")
        else:
            print(f"[WARNING] Only {index_count} indexes found (expected 15+)")

        # Check data
        cursor.execute("SELECT COUNT(*) as count FROM xtrades_profiles")
        profile_count = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM xtrades_trades")
        trade_count = cursor.fetchone()['count']

        print(f"[OK] {profile_count} profiles in database")
        print(f"[OK] {trade_count} trades in database")

        # Summary stats
        cursor.execute("""
            SELECT
                (SELECT COUNT(*) FROM xtrades_profiles WHERE active = TRUE) as active_profiles,
                (SELECT COUNT(*) FROM xtrades_trades WHERE status = 'open') as open_trades,
                (SELECT COUNT(*) FROM xtrades_trades WHERE status = 'closed') as closed_trades,
                (SELECT COALESCE(SUM(pnl), 0) FROM xtrades_trades WHERE status = 'closed') as total_pnl
        """)
        stats = cursor.fetchone()

        print("\n" + "="*60)
        print("SYSTEM STATISTICS")
        print("="*60)
        print(f"Active Profiles:    {stats['active_profiles']}")
        print(f"Open Positions:     {stats['open_trades']}")
        print(f"Closed Trades:      {stats['closed_trades']}")
        print(f"Total P&L:          ${stats['total_pnl']:.2f}")

        print("\n[SUCCESS] Xtrades schema is properly installed!\n")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        return False

if __name__ == '__main__':
    verify_installation()
