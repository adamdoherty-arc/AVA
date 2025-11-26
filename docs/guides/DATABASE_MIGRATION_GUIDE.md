# Database Migration Guide

Complete guide to migrating your AVA Trading Platform PostgreSQL database to a new computer.

## Quick Start

### On Source Computer (Current)
```bash
# Export database
python migrate_database.py --mode export
# Creates: trading_backup_YYYYMMDD_HHMMSS.dump

# Copy backup file to USB drive or transfer via network
```

### On Destination Computer (New)
```bash
# 1. Install PostgreSQL
# 2. Create database
# 3. Update .env with new credentials
# 4. Import backup
python migrate_database.py --mode import --file trading_backup_YYYYMMDD_HHMMSS.dump

# 5. Verify migration
python migrate_database.py --mode verify
```

---

## Detailed Migration Steps

### Step 1: Prepare Source Computer

**1.1 Verify current database status**
```bash
# Connect to database
psql -U postgres -d trading

# Check database size
SELECT pg_size_pretty(pg_database_size('trading'));

# List all tables
\dt

# Check row counts for critical tables
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;

# Exit
\q
```

**1.2 Export using migration script**
```bash
# Export full database (recommended)
python migrate_database.py --mode export

# Output: trading_backup_20250120_143022.dump
```

**Alternative: Manual export commands**
```bash
# Full backup (custom format - compressed)
pg_dump -h localhost -U postgres -d trading -F c -f trading_full.dump

# Full backup (SQL format - human readable)
pg_dump -h localhost -U postgres -d trading -f trading_full.sql

# Schema only
pg_dump -h localhost -U postgres -d trading --schema-only -f schema.sql

# Data only
pg_dump -h localhost -U postgres -d trading --data-only -f data.sql

# Specific tables only
pg_dump -h localhost -U postgres -d trading -t robinhood_positions -t options_data -f specific_tables.dump
```

### Step 2: Transfer Backup File

**Option A: USB Drive**
```bash
# Windows
copy trading_backup_*.dump E:\backup\

# Linux/Mac
cp trading_backup_*.dump /media/usb/backup/
```

**Option B: Network Transfer (if computers on same network)**
```bash
# Using SCP (Linux/Mac)
scp trading_backup_*.dump user@new-computer:/home/user/backup/

# Using Windows share
# Map network drive and copy file
```

**Option C: Cloud Storage**
```bash
# Upload to Dropbox, Google Drive, OneDrive, etc.
# Then download on destination computer
```

**Option D: Direct Network Connection**
```bash
# On destination, connect directly to source database
pg_dump -h source-computer-ip -U postgres -d trading -F c -f trading_backup.dump
```

### Step 3: Set Up Destination Computer

**3.1 Install PostgreSQL**

**Windows:**
```bash
# Download from: https://www.postgresql.org/download/windows/
# Or use Chocolatey
choco install postgresql

# Verify installation
psql --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS:**
```bash
# Using Homebrew
brew install postgresql

# Start PostgreSQL
brew services start postgresql
```

**3.2 Create database and user**
```bash
# Connect as postgres superuser
psql -U postgres

# Create database
CREATE DATABASE trading;

# Create user (if needed)
CREATE USER trading_user WITH PASSWORD 'your_secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE trading TO trading_user;

# Exit
\q
```

**3.3 Configure .env file**

Create or update `.env` on destination computer:
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=your_password

# Copy all other settings from source .env
# (Robinhood, TradingView, Kalshi credentials, etc.)
```

### Step 4: Import Database

**4.1 Using migration script (recommended)**
```bash
# Import database
python migrate_database.py --mode import --file trading_backup_20250120_143022.dump

# This will:
# - Drop existing objects (if any)
# - Recreate schema
# - Import all data
# - Show progress and errors
```

**4.2 Manual import**
```bash
# Custom format (.dump)
pg_restore -h localhost -U postgres -d trading --clean -v trading_backup.dump

# SQL format (.sql)
psql -h localhost -U postgres -d trading < trading_backup.sql

# With specific options
pg_restore -h localhost -U postgres -d trading \
    --clean \              # Drop objects before creating
    --if-exists \          # Use IF EXISTS when dropping
    --no-owner \           # Don't set ownership
    --no-privileges \      # Don't restore access privileges
    -v \                   # Verbose output
    trading_backup.dump
```

### Step 5: Verify Migration

**5.1 Using migration script**
```bash
python migrate_database.py --mode verify
```

**5.2 Manual verification**
```bash
# Connect to database
psql -U postgres -d trading

# Check table counts
SELECT
    schemaname,
    COUNT(*) as table_count
FROM pg_tables
WHERE schemaname = 'public'
GROUP BY schemaname;

# Verify critical tables exist with data
SELECT 'robinhood_positions' as table_name, COUNT(*) FROM robinhood_positions
UNION ALL
SELECT 'options_data', COUNT(*) FROM options_data
UNION ALL
SELECT 'tradingview_watchlists', COUNT(*) FROM tradingview_watchlists
UNION ALL
SELECT 'kalshi_markets', COUNT(*) FROM kalshi_markets;

# Check indexes
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

# Verify database size matches source
SELECT pg_size_pretty(pg_database_size('trading'));
```

### Step 6: Post-Migration Tasks

**6.1 Install Python dependencies**
```bash
# Install requirements
pip install -r requirements.txt

# Verify installation
pip list | grep -E "psycopg2|streamlit|pandas"
```

**6.2 Test database connections**
```bash
# Test basic connection
python -c "
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM robinhood_positions')
print(f'Positions count: {cursor.fetchone()[0]}')
conn.close()
print('✅ Database connection successful!')
"
```

**6.3 Run initial data sync (if needed)**
```bash
# Sync latest stock data
python sync_database_stocks_daily.py

# Sync TradingView watchlists
python src/tradingview_api_sync.py

# Sync Kalshi markets
python sync_kalshi_markets.py
```

**6.4 Test the dashboard**
```bash
# Start Streamlit dashboard
streamlit run dashboard.py

# Open browser to http://localhost:8501
# Verify all pages load correctly
```

---

## Migration Scenarios

### Scenario 1: Complete Fresh Install

```bash
# Source computer
python migrate_database.py --mode export

# Destination computer
# 1. Install PostgreSQL
# 2. Create database
createdb -U postgres trading

# 3. Import
python migrate_database.py --mode import --file trading_backup.dump

# 4. Verify
python migrate_database.py --mode verify
```

### Scenario 2: Selective Table Migration

```bash
# Export specific tables only
pg_dump -h localhost -U postgres -d trading \
    -t robinhood_positions \
    -t options_data \
    -t tradingview_watchlists \
    -F c -f critical_tables.dump

# Import on destination
pg_restore -h localhost -U postgres -d trading critical_tables.dump
```

### Scenario 3: Schema Migration (Rebuild Data)

```bash
# Export schema only
python migrate_database.py --mode schema

# On destination, import schema
psql -U postgres -d trading < schema_only.sql

# Then rebuild data using sync scripts
python sync_database_stocks_daily.py
python src/tradingview_api_sync.py
# etc.
```

### Scenario 4: Live Replication (Advanced)

For minimal downtime, set up streaming replication:

**On source (primary):**
```bash
# Edit postgresql.conf
wal_level = replica
max_wal_senders = 3
wal_keep_size = 64

# Edit pg_hba.conf
# Add line for replication user
host replication replication_user destination_ip/32 md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

**On destination (replica):**
```bash
# Stop PostgreSQL
sudo systemctl stop postgresql

# Create base backup
pg_basebackup -h source_ip -D /var/lib/postgresql/data -U replication_user -P

# Start PostgreSQL
sudo systemctl start postgresql
```

---

## Troubleshooting

### Error: "database 'trading' does not exist"
```bash
# Create database first
createdb -U postgres trading

# Then retry import
```

### Error: "permission denied"
```bash
# Grant proper permissions
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE trading TO your_user"
```

### Error: "relation already exists"
```bash
# Use --clean flag to drop existing objects
pg_restore --clean -d trading backup.dump

# Or manually drop database and recreate
dropdb -U postgres trading
createdb -U postgres trading
pg_restore -d trading backup.dump
```

### Import is very slow
```bash
# Disable triggers during import
pg_restore --disable-triggers -d trading backup.dump

# Or import in single transaction
pg_restore --single-transaction -d trading backup.dump
```

### Some tables are missing after import
```bash
# Check for errors in import log
pg_restore -v -d trading backup.dump 2>&1 | tee import.log

# Import with continue-on-error
pg_restore --exit-on-error=false -d trading backup.dump
```

### Database size is much larger than source
```bash
# Rebuild indexes and analyze
psql -U postgres -d trading -c "VACUUM FULL ANALYZE"

# Rebuild specific tables
psql -U postgres -d trading -c "VACUUM FULL ANALYZE robinhood_positions"
```

---

## Performance Optimization After Migration

### Rebuild Indexes
```bash
# Run performance indexes migration
psql -U postgres -d trading < migrations/performance_indexes_migration.sql

# Or rebuild all indexes
psql -U postgres -d trading -c "REINDEX DATABASE trading"
```

### Update Statistics
```bash
psql -U postgres -d trading -c "ANALYZE"
```

### Configure PostgreSQL for Performance
```bash
# Edit postgresql.conf
shared_buffers = 256MB          # 25% of RAM
effective_cache_size = 1GB      # 50-75% of RAM
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB

# Restart PostgreSQL
sudo systemctl restart postgresql
```

---

## Verification Checklist

After migration, verify these items:

- [ ] All tables exist and have correct row counts
- [ ] All indexes are present
- [ ] Database size matches source (approximately)
- [ ] Can connect from Python application
- [ ] Dashboard loads without errors
- [ ] All pages display data correctly
- [ ] Watchlists are populated
- [ ] Kalshi markets are present
- [ ] Options data is current
- [ ] Supply/demand zones exist
- [ ] XTrades alerts are synced
- [ ] Performance is acceptable
- [ ] Backup/sync scripts work

---

## Rollback Plan

If migration fails, you can rollback:

```bash
# On destination, drop failed migration
dropdb -U postgres trading

# Recreate empty database
createdb -U postgres trading

# Try import again with different options
pg_restore --no-owner --no-privileges -d trading backup.dump
```

---

## Automated Migration Script

For automated migration with logging:

```bash
#!/bin/bash
# migrate.sh

set -e  # Exit on error

BACKUP_FILE="trading_backup_$(date +%Y%m%d_%H%M%S).dump"
LOG_FILE="migration_$(date +%Y%m%d_%H%M%S).log"

echo "Starting migration..." | tee -a $LOG_FILE

# Export
echo "Exporting database..." | tee -a $LOG_FILE
pg_dump -h localhost -U postgres -d trading -F c -f $BACKUP_FILE 2>&1 | tee -a $LOG_FILE

# Verify backup file
if [ ! -f $BACKUP_FILE ]; then
    echo "ERROR: Backup file not created" | tee -a $LOG_FILE
    exit 1
fi

echo "Export complete: $BACKUP_FILE" | tee -a $LOG_FILE
echo "Size: $(du -h $BACKUP_FILE | cut -f1)" | tee -a $LOG_FILE

echo "Migration complete! Transfer $BACKUP_FILE to destination computer." | tee -a $LOG_FILE
```

---

## Additional Resources

- PostgreSQL Documentation: https://www.postgresql.org/docs/
- pg_dump Manual: https://www.postgresql.org/docs/current/app-pgdump.html
- pg_restore Manual: https://www.postgresql.org/docs/current/app-pgrestore.html
- AVA Platform Documentation: See README.md

---

**Need Help?**

If you encounter issues:
1. Check the troubleshooting section above
2. Review PostgreSQL logs: `/var/log/postgresql/`
3. Check migration log files
4. Verify .env configuration
5. Test database connection manually
