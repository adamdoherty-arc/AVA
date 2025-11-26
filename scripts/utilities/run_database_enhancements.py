"""
Run database enhancements for earnings calendar
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def run_database_enhancements():
    """Execute the database enhancement SQL script"""

    # Read the SQL file
    sql_file = 'database_earnings_enhancements_safe.sql'

    print(f"Reading SQL file: {sql_file}")
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_script = f.read()

    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123!')
    )

    # Set autocommit for DDL operations
    conn.set_session(autocommit=True)
    cur = conn.cursor()

    print("Executing database enhancements...")
    print("=" * 80)

    try:
        # Execute the script
        cur.execute(sql_script)

        print("=" * 80)
        print("SUCCESS: Database enhancements completed successfully!")
        print()
        print("New tables created:")
        print("  [+] earnings_pattern_analysis")
        print("  [+] earnings_iv_tracking")
        print()
        print("New views created:")
        print("  [+] v_upcoming_quality_earnings")
        print("  [+] v_earnings_results")
        print("  [+] v_iv_expansion")
        print()
        print("New functions created:")
        print("  [+] calculate_beat_rate()")
        print("  [+] get_quality_score()")
        print("  [+] calculate_expected_move()")
        print()

    except Exception as e:
        print(f"ERROR executing SQL: {e}")
        raise

    finally:
        cur.close()
        conn.close()
        print("Database connection closed")

if __name__ == "__main__":
    run_database_enhancements()
