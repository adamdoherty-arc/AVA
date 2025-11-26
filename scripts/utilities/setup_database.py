#!/usr/bin/env python
"""Setup PostgreSQL database for Wheel Strategy System"""

import os
import sys
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_database():
    """Create database and run schema"""

    # Database credentials from .env
    db_config = {
        'host': os.getenv('PGHOST', 'localhost'),
        'port': os.getenv('PGPORT', '5432'),
        'user': os.getenv('PGUSER', 'postgres'),
        'password': os.getenv('PGPASSWORD', 'postgres123!')
    }

    print("Setting up PostgreSQL database...")
    print(f"   Host: {db_config['host']}")
    print(f"   Port: {db_config['port']}")
    print(f"   User: {db_config['user']}")
    print()

    try:
        # First connect to default 'postgres' database to create our database
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database='postgres'
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'wheel_strategy'"
        )
        exists = cursor.fetchone()

        if not exists:
            print("Creating database 'wheel_strategy'...")
            cursor.execute('CREATE DATABASE wheel_strategy')
            print("Database created successfully!")
        else:
            print("Database 'wheel_strategy' already exists")

        cursor.close()
        conn.close()

        # Now connect to the wheel_strategy database
        print("\nConnecting to wheel_strategy database...")
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database='wheel_strategy'
        )
        cursor = conn.cursor()

        # Check if TimescaleDB extension is available
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            conn.commit()
            print("TimescaleDB extension enabled")
        except Exception as e:
            print(f"WARNING: TimescaleDB not available (optional): {e}")
            print("   System will work without TimescaleDB")
            conn.rollback()

        # Check if schema already exists by looking for a key table
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'stocks'
            );
        """)
        schema_exists = cursor.fetchone()[0]

        if not schema_exists:
            print("\nRunning database schema...")

            # Check if schema file exists
            if os.path.exists('database_schema.sql'):
                with open('database_schema.sql', 'r') as f:
                    schema_sql = f.read()

                # Split by semicolon and execute each statement
                statements = [s.strip() for s in schema_sql.split(';') if s.strip()]

                for i, statement in enumerate(statements, 1):
                    try:
                        # Skip TimescaleDB-specific commands if not available
                        if 'timescaledb' in statement.lower() or 'create_hypertable' in statement.lower():
                            try:
                                cursor.execute(statement)
                                conn.commit()
                            except:
                                # Skip TimescaleDB specific features
                                conn.rollback()
                                continue
                        else:
                            cursor.execute(statement)
                            conn.commit()

                        # Show progress
                        if i % 10 == 0:
                            print(f"   Executed {i}/{len(statements)} statements...")

                    except Exception as e:
                        print(f"   Warning on statement {i}: {str(e)[:50]}")
                        conn.rollback()
                        continue

                print(f"Schema created successfully! ({len(statements)} statements)")
            else:
                print("ERROR: database_schema.sql not found!")
                print("   Please ensure database_schema.sql is in the current directory")
        else:
            print("Database schema already exists")

        # Test the connection
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
        table_count = cursor.fetchone()[0]
        print(f"\nDatabase ready with {table_count} tables")

        cursor.close()
        conn.close()

        print("\nDatabase setup complete!")
        print("\nConnection details for your application:")
        print(f"  DATABASE_URL=postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/wheel_strategy")

        return True

    except psycopg2.OperationalError as e:
        print(f"\nERROR: Could not connect to PostgreSQL:")
        print(f"   {e}")
        print("\nPlease ensure PostgreSQL is running and accessible:")
        print("  Windows: Check Services for 'postgresql-x64-15' (or similar)")
        print("  Mac: brew services start postgresql")
        print("  Linux: sudo service postgresql start")
        return False

    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Load .env file
    if os.path.exists('.env'):
        print("Loading configuration from .env file")
        load_dotenv()

    # Run setup
    success = setup_database()

    if not success:
        sys.exit(1)

    print("\nYou can now run: python start.py")