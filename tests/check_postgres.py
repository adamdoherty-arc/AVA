"""
Check PostgreSQL connection and database status
"""
import psycopg2
from dotenv import load_dotenv
import os
import sys

load_dotenv(override=True)  # Override system environment variables

# Database credentials from .env
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': 'postgres',  # Connect to default db first
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123!')
}

print("=" * 70)
print("PostgreSQL Connection Test")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Host: {DB_CONFIG['host']}")
print(f"  Port: {DB_CONFIG['port']}")
print(f"  User: {DB_CONFIG['user']}")
print(f"  Password: {'*' * len(DB_CONFIG['password'])}")
print()

# Test connection
print("Testing connection to PostgreSQL...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Get PostgreSQL version
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"[OK] Connected successfully!")
    print(f"[OK] PostgreSQL version: {version.split(',')[0]}")

    # List all databases
    cursor.execute("""
        SELECT datname FROM pg_database
        WHERE datistemplate = false
        ORDER BY datname;
    """)
    databases = cursor.fetchall()

    print(f"\n[OK] Available databases ({len(databases)}):")
    for db in databases:
        print(f"  - {db[0]}")

    # Check if magnus database exists
    magnus_exists = any(db[0] == 'magnus' for db in databases)

    if magnus_exists:
        print(f"\n[OK] 'magnus' database exists")

        # Connect to magnus and check tables
        cursor.close()
        conn.close()

        DB_CONFIG['database'] = 'magnus'
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()

        if tables:
            print(f"[OK] Found {len(tables)} tables in 'magnus' database:")
            for table in tables[:15]:  # Show first 15
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
                count = cursor.fetchone()[0]
                print(f"  - {table[0]}: {count} rows")

            if len(tables) > 15:
                print(f"  ... and {len(tables) - 15} more tables")

            print(f"\n[OK] Database has been populated with data!")
        else:
            print(f"[WARNING] 'magnus' database exists but has no tables")
            print(f"[INFO] You may need to restore from backup or run schema")
    else:
        print(f"\n[WARNING] 'magnus' database does not exist")
        print(f"[INFO] Creating 'magnus' database...")

        cursor.execute("CREATE DATABASE magnus;")
        conn.commit()
        print(f"[OK] 'magnus' database created")

    cursor.close()
    conn.close()

    print(f"\n{'=' * 70}")
    print("PostgreSQL Status: READY")
    print("=" * 70)
    print(f"\nConnection string:")
    print(f"postgresql://{DB_CONFIG['user']}:****@{DB_CONFIG['host']}:{DB_CONFIG['port']}/magnus")
    print()

    sys.exit(0)

except psycopg2.OperationalError as e:
    error_str = str(e)
    print(f"[ERROR] Connection failed!")
    print(f"[ERROR] {error_str}")

    if "password authentication failed" in error_str:
        print(f"\n[FIX] Password mismatch detected!")
        print(f"\nOptions to fix:")
        print(f"1. Reset PostgreSQL password to match .env:")
        print(f"   a. Open pgAdmin")
        print(f"   b. Right-click postgres user -> Properties -> Definition")
        print(f"   c. Set password to: {DB_CONFIG['password']}")
        print(f"\n2. Or update .env with your current PostgreSQL password")

    sys.exit(1)

except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
