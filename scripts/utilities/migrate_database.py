"""
Database Migration Script for AVA Trading Platform

Migrates PostgreSQL database from one server to another with verification.

Usage:
    python migrate_database.py --mode export
    python migrate_database.py --mode import --file trading_backup.dump
    python migrate_database.py --mode verify
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

load_dotenv()


class DatabaseMigrator:
    """Handles database migration operations."""

    def __init__(self):
        """Initialize migrator with database credentials."""
        self.source_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'trading'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

    def export_database(self, output_file: str = None) -> str:
        """
        Export database to backup file.

        Args:
            output_file: Path to output file (default: auto-generated)

        Returns:
            Path to backup file
        """
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"trading_backup_{timestamp}.dump"

        print(f"Exporting database to {output_file}...")

        cmd = [
            'pg_dump',
            '-h', self.source_config['host'],
            '-p', self.source_config['port'],
            '-U', self.source_config['user'],
            '-d', self.source_config['database'],
            '-F', 'c',  # Custom format (compressed)
            '-v',  # Verbose
            '-f', output_file
        ]

        # Set password environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = self.source_config['password']

        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ Export successful: {output_file}")

            # Get file size
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"   Backup size: {size_mb:.2f} MB")

            return output_file

        except subprocess.CalledProcessError as e:
            print(f"❌ Export failed: {e}")
            sys.exit(1)

    def import_database(self, backup_file: str, clean: bool = True) -> bool:
        """
        Import database from backup file.

        Args:
            backup_file: Path to backup file
            clean: Whether to drop existing objects first

        Returns:
            Success status
        """
        if not os.path.exists(backup_file):
            print(f"❌ Backup file not found: {backup_file}")
            return False

        print(f"Importing database from {backup_file}...")

        cmd = [
            'pg_restore',
            '-h', self.source_config['host'],
            '-p', self.source_config['port'],
            '-U', self.source_config['user'],
            '-d', self.source_config['database'],
            '-v'  # Verbose
        ]

        if clean:
            cmd.append('--clean')  # Drop objects before recreating

        cmd.append(backup_file)

        # Set password environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = self.source_config['password']

        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ Import successful")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Import failed: {e}")
            return False

    def verify_migration(self) -> bool:
        """
        Verify database migration by checking table counts and key data.

        Returns:
            Success status
        """
        print("\nVerifying database migration...")

        try:
            conn = psycopg2.connect(**self.source_config)
            cursor = conn.cursor()

            # Check critical tables exist
            critical_tables = [
                'robinhood_positions',
                'options_data',
                'tradingview_watchlists',
                'watchlist_stocks',
                'kalshi_markets',
                'kalshi_nfl_games',
                'kalshi_nba_games',
                'supply_demand_zones',
                'xtrades_alerts'
            ]

            print("\n📊 Table verification:")
            all_exist = True

            for table in critical_tables:
                cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_name = %s
                """, (table,))

                exists = cursor.fetchone()[0] > 0

                if exists:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"   ✅ {table}: {count:,} rows")
                else:
                    print(f"   ❌ {table}: MISSING")
                    all_exist = False

            # Check indexes
            cursor.execute("""
                SELECT COUNT(*)
                FROM pg_indexes
                WHERE schemaname = 'public'
            """)
            index_count = cursor.fetchone()[0]
            print(f"\n📑 Indexes: {index_count} total")

            # Check for recent data
            cursor.execute("""
                SELECT
                    table_name,
                    pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
                LIMIT 10
            """)

            print("\n💾 Largest tables:")
            for row in cursor.fetchall():
                print(f"   {row[0]}: {row[1]}")

            cursor.close()
            conn.close()

            if all_exist:
                print("\n✅ Database verification successful!")
                return True
            else:
                print("\n⚠️  Some tables are missing - check migration logs")
                return False

        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

    def export_schema_only(self, output_file: str = None) -> str:
        """
        Export database schema only (no data).

        Args:
            output_file: Path to output file

        Returns:
            Path to schema file
        """
        if not output_file:
            output_file = "schema_only.sql"

        print(f"Exporting schema to {output_file}...")

        cmd = [
            'pg_dump',
            '-h', self.source_config['host'],
            '-p', self.source_config['port'],
            '-U', self.source_config['user'],
            '-d', self.source_config['database'],
            '--schema-only',
            '-f', output_file
        ]

        env = os.environ.copy()
        env['PGPASSWORD'] = self.source_config['password']

        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ Schema export successful: {output_file}")
            return output_file

        except subprocess.CalledProcessError as e:
            print(f"❌ Schema export failed: {e}")
            sys.exit(1)

    def create_migration_script(self) -> None:
        """Create a SQL script with all schema definitions."""
        print("\nGenerating migration SQL script...")

        # Export all schema files
        schema_files = [
            ('database_schema.sql', 'Main trading schema'),
            ('src/kalshi_schema.sql', 'Kalshi prediction markets'),
            ('src/analytics_schema.sql', 'Analytics and performance'),
            ('src/xtrades_schema.sql', 'XTrades integration'),
            ('src/supply_demand_schema.sql', 'Supply/demand zones'),
            ('migrations/performance_indexes_migration.sql', 'Performance indexes')
        ]

        print("\n📄 Available schema files:")
        for file, description in schema_files:
            if os.path.exists(file):
                print(f"   ✅ {file} - {description}")
            else:
                print(f"   ⚠️  {file} - NOT FOUND")


def main():
    """Main migration workflow."""
    parser = argparse.ArgumentParser(
        description='Migrate AVA Trading Platform database'
    )
    parser.add_argument(
        '--mode',
        choices=['export', 'import', 'verify', 'schema'],
        required=True,
        help='Migration mode'
    )
    parser.add_argument(
        '--file',
        help='Backup file path (for import mode)'
    )
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Do not drop existing objects during import'
    )

    args = parser.parse_args()

    migrator = DatabaseMigrator()

    print("=" * 70)
    print("AVA Trading Platform - Database Migration Tool")
    print("=" * 70)
    print(f"\nDatabase: {migrator.source_config['database']}")
    print(f"Host: {migrator.source_config['host']}")
    print(f"User: {migrator.source_config['user']}")
    print()

    if args.mode == 'export':
        backup_file = migrator.export_database()
        print(f"\n✅ Database exported successfully!")
        print(f"\n📋 Next steps:")
        print(f"   1. Copy {backup_file} to destination computer")
        print(f"   2. Update .env on destination with new DB credentials")
        print(f"   3. Run: python migrate_database.py --mode import --file {backup_file}")

    elif args.mode == 'import':
        if not args.file:
            print("❌ Error: --file argument required for import mode")
            sys.exit(1)

        success = migrator.import_database(
            args.file,
            clean=not args.no_clean
        )

        if success:
            print("\n✅ Database imported successfully!")
            print("\n📋 Next steps:")
            print("   1. Run: python migrate_database.py --mode verify")
            print("   2. Update any API keys/credentials in .env")
            print("   3. Test dashboard: streamlit run dashboard.py")

    elif args.mode == 'verify':
        migrator.verify_migration()

    elif args.mode == 'schema':
        migrator.export_schema_only()
        migrator.create_migration_script()


if __name__ == '__main__':
    main()
