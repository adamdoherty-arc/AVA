"""
Restore PostgreSQL database backup
"""
import subprocess
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Set environment variables for pg_restore
os.environ['PGPASSWORD'] = os.getenv('DB_PASSWORD', 'postgres123')
os.environ['PGHOST'] = os.getenv('DB_HOST', 'localhost')
os.environ['PGPORT'] = os.getenv('DB_PORT', '5432')
os.environ['PGUSER'] = os.getenv('DB_USER', 'postgres')

print("Restoring database backup to 'magnus' database...")
print("=" * 70)
print(f"Backup file: trading_backup_20251120_214418.dump")
print(f"Host: {os.environ['PGHOST']}")
print(f"Port: {os.environ['PGPORT']}")
print(f"User: {os.environ['PGUSER']}")
print(f"Database: magnus")
print("=" * 70)

# Run pg_restore
pg_restore_path = r"C:\Program Files\PostgreSQL\16\bin\pg_restore.exe"

cmd = [
    pg_restore_path,
    '--dbname=magnus',
    '--verbose',
    '--clean',
    '--if-exists',
    'trading_backup_20251120_214418.dump'
]

print("\nRestoring backup...")
result = subprocess.run(cmd, capture_output=True, text=True)

# Print output
if result.stdout:
    print("\nOutput:")
    print(result.stdout)

if result.stderr:
    # pg_restore outputs to stderr even on success
    print("\nRestore log:")
    lines = result.stderr.split('\n')
    # Show first 30 and last 10 lines to avoid spam
    if len(lines) > 40:
        print('\n'.join(lines[:30]))
        print(f"\n... ({len(lines) - 40} lines omitted) ...\n")
        print('\n'.join(lines[-10:]))
    else:
        print(result.stderr)

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("Backup restored successfully!")
    print("=" * 70)
else:
    print(f"\n[WARNING] pg_restore exited with code {result.returncode}")
    print("This may be normal if there were warnings. Checking database...")

# Verify restoration
import psycopg2

print("\nVerifying database restoration...")
try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database='magnus',
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD')
    )
    cursor = conn.cursor()

    # Count tables
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()

    print(f"\n[OK] Database has {len(tables)} tables:")

    total_rows = 0
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
            count = cursor.fetchone()[0]
            total_rows += count
            if count > 0:  # Only show tables with data
                print(f"  - {table[0]}: {count:,} rows")
        except:
            pass

    print(f"\n[OK] Total rows across all tables: {total_rows:,}")

    cursor.close()
    conn.close()

    print("\n" + "=" * 70)
    print("Database restoration verified successfully!")
    print("=" * 70)

except Exception as e:
    print(f"[ERROR] Verification failed: {e}")
