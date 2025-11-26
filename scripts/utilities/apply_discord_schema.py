"""
Apply Discord schema to magnus database
Checks if tables exist and creates them if needed
"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def check_and_apply_discord_schema():
    """Check if Discord tables exist and apply schema if needed"""

    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '')
    )

    cur = conn.cursor()

    try:
        # Check if Discord tables exist
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'discord_%'
            ORDER BY table_name;
        """)

        existing_tables = [row[0] for row in cur.fetchall()]

        print(f"[OK] Connected to database: {os.getenv('DB_NAME', 'magnus')}")
        print(f"\nExisting Discord tables: {existing_tables if existing_tables else 'None'}")

        # Read and apply schema
        schema_file = r'c:\code\Magnus\src\discord_schema.sql'
        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        print(f"\n[APPLY] Applying Discord schema from: {schema_file}")

        # Execute schema
        cur.execute(schema_sql)
        conn.commit()

        print("[SUCCESS] Discord schema applied successfully!")

        # Verify tables were created
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'discord_%'
            ORDER BY table_name;
        """)

        final_tables = [row[0] for row in cur.fetchall()]
        print(f"\n[OK] Discord tables now in database:")
        for table in final_tables:
            print(f"  - {table}")

        # Check view
        cur.execute("""
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'public'
            AND table_name LIKE 'discord_%'
        """)

        views = [row[0] for row in cur.fetchall()]
        if views:
            print(f"\n[OK] Discord views:")
            for view in views:
                print(f"  - {view}")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Error: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    check_and_apply_discord_schema()
