# Xtrades Sync Service - Testing Guide

## Pre-Flight Checklist

Before testing, ensure you have:

- [ ] Python virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] PostgreSQL running with Xtrades schema installed
- [ ] `.env` file configured with credentials
- [ ] Chrome browser installed
- [ ] At least one profile added to database

---

## Test Suite

### Test 1: Environment Verification

**Purpose:** Verify all dependencies and configuration

```bash
cd C:\Code\WheelStrategy
call venv\Scripts\activate.bat
```

**1.1 Check Python packages:**
```bash
python -c "import selenium, psycopg2, telegram, undetected_chromedriver; print('All packages installed')"
```

**Expected:** `All packages installed`

**1.2 Check database connection:**
```bash
python -c "from src.xtrades_db_manager import XtradesDBManager; db = XtradesDBManager(); conn = db.get_connection(); print('Database connected'); conn.close()"
```

**Expected:** `Database connected`

**1.3 Check environment variables:**
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('XTRADES_USERNAME:', os.getenv('XTRADES_USERNAME')); print('DB_NAME:', os.getenv('DB_NAME')); print('TELEGRAM_ENABLED:', os.getenv('TELEGRAM_ENABLED'))"
```

**Expected:** Your credentials displayed (verify they're correct)

---

### Test 2: Database Setup

**Purpose:** Verify database schema and add test profiles

**2.1 Check tables exist:**
```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'xtrades_%'
ORDER BY table_name;
```

**Expected:** 4 tables
- `xtrades_notifications`
- `xtrades_profiles`
- `xtrades_sync_log`
- `xtrades_trades`

**2.2 Add test profile:**
```sql
INSERT INTO xtrades_profiles (username, display_name, active, notes)
VALUES ('behappy', 'BeHappy Trader', TRUE, 'Test profile for sync service')
ON CONFLICT (username) DO UPDATE
  SET active = TRUE,
      display_name = EXCLUDED.display_name;

SELECT * FROM xtrades_profiles WHERE username = 'behappy';
```

**Expected:** One row returned with your profile

**2.3 Verify indexes:**
```sql
SELECT indexname
FROM pg_indexes
WHERE tablename = 'xtrades_trades'
ORDER BY indexname;
```

**Expected:** Multiple indexes including `idx_xtrades_trades_profile_id`, `idx_xtrades_trades_ticker`, etc.

---

### Test 3: Scraper Component Test

**Purpose:** Test the Xtrades scraper in isolation

**3.1 Test login (visible browser):**
```bash
python -c "
from src.xtrades_scraper import XtradesScraper
scraper = XtradesScraper(headless=False)
try:
    print('Logging in...')
    success = scraper.login()
    print(f'Login successful: {success}')
    if success:
        print('Waiting 5 seconds to verify...')
        import time
        time.sleep(5)
finally:
    scraper.close()
"
```

**Expected:**
- Chrome browser opens
- Navigates to Xtrades.net
- Logs in via Discord
- Returns to Xtrades
- Prints `Login successful: True`

**3.2 Test login (headless):**
```bash
python -c "
from src.xtrades_scraper import XtradesScraper
scraper = XtradesScraper(headless=True)
try:
    success = scraper.login()
    print(f'Headless login successful: {success}')
finally:
    scraper.close()
"
```

**Expected:** `Headless login successful: True` (no browser visible)

**3.3 Test profile scraping:**
```bash
python -c "
from src.xtrades_scraper import XtradesScraper
scraper = XtradesScraper(headless=False)
try:
    scraper.login()
    alerts = scraper.get_profile_alerts('behappy', max_alerts=10)
    print(f'Found {len(alerts)} alerts')
    if alerts:
        first = alerts[0]
        print(f'First alert: {first.get(\"ticker\")} - {first.get(\"strategy\")} - {first.get(\"action\")}')
finally:
    scraper.close()
"
```

**Expected:**
- Found X alerts (where X >= 0)
- If alerts found, displays first alert details

**3.4 Test alert parsing:**
```bash
python -c "
from src.xtrades_scraper import XtradesScraper

scraper = XtradesScraper()
test_alert = 'AAPL CSP: STO 1x \$170 PUT @ \$2.50 exp 12/15/2024'
parsed = scraper.parse_alert(test_alert)
print('Parsed alert:', parsed)
print('Ticker:', parsed.get('ticker'))
print('Strategy:', parsed.get('strategy'))
print('Action:', parsed.get('action'))
print('Price:', parsed.get('entry_price'))
scraper.close()
"
```

**Expected:**
- Ticker: AAPL
- Strategy: CSP
- Action: STO
- Price: 2.5

---

### Test 4: Database Manager Component Test

**Purpose:** Test database CRUD operations

**4.1 Test get active profiles:**
```bash
python -c "
from src.xtrades_db_manager import XtradesDBManager
db = XtradesDBManager()
profiles = db.get_active_profiles()
print(f'Found {len(profiles)} active profiles')
for p in profiles:
    print(f'  - {p[\"username\"]} ({p[\"display_name\"]})')
"
```

**Expected:** List of active profiles

**4.2 Test add trade:**
```bash
python -c "
from src.xtrades_db_manager import XtradesDBManager
from datetime import datetime

db = XtradesDBManager()
profile = db.get_profile_by_username('behappy')

trade_data = {
    'profile_id': profile['id'],
    'ticker': 'TEST',
    'strategy': 'CSP',
    'action': 'STO',
    'entry_price': 2.50,
    'quantity': 1,
    'strike_price': 170.00,
    'expiration_date': '2024-12-15',
    'alert_text': 'TEST trade for sync service',
    'alert_timestamp': datetime.now(),
    'status': 'open'
}

trade_id = db.add_trade(trade_data)
print(f'Added test trade ID: {trade_id}')
"
```

**Expected:** `Added test trade ID: XXX`

**4.3 Test find existing trade:**
```bash
python -c "
from src.xtrades_db_manager import XtradesDBManager
from datetime import datetime

db = XtradesDBManager()
profile = db.get_profile_by_username('behappy')

# Should find the trade we just added
trade_id = db.find_existing_trade(profile['id'], 'TEST', datetime.now())
print(f'Found existing trade: {trade_id}')
"
```

**Expected:** `Found existing trade: XXX` (same ID as previous test)

**4.4 Test update trade:**
```bash
python -c "
from src.xtrades_db_manager import XtradesDBManager
from datetime import datetime

db = XtradesDBManager()
profile = db.get_profile_by_username('behappy')
trade_id = db.find_existing_trade(profile['id'], 'TEST', datetime.now())

update_data = {
    'exit_price': 1.25,
    'pnl': 125.00,
    'pnl_percent': 50.0,
    'status': 'closed'
}

success = db.update_trade(trade_id, update_data)
print(f'Trade updated: {success}')

# Verify
trade = db.get_trade_by_id(trade_id)
print(f'Exit price: {trade[\"exit_price\"]}')
print(f'Status: {trade[\"status\"]}')
"
```

**Expected:**
- Trade updated: True
- Exit price: 1.25
- Status: closed

**4.5 Test sync logging:**
```bash
python -c "
from src.xtrades_db_manager import XtradesDBManager

db = XtradesDBManager()

# Start sync
sync_id = db.log_sync_start()
print(f'Started sync log ID: {sync_id}')

# Complete sync
stats = {
    'profiles_synced': 1,
    'trades_found': 5,
    'new_trades': 2,
    'updated_trades': 1,
    'errors': None,
    'duration_seconds': 45.3,
    'status': 'success'
}

success = db.log_sync_complete(sync_id, stats)
print(f'Sync log completed: {success}')

# Verify
latest = db.get_latest_sync()
print(f'Latest sync status: {latest[\"status\"]}')
print(f'New trades: {latest[\"new_trades\"]}')
"
```

**Expected:**
- Started sync log ID: XXX
- Sync log completed: True
- Latest sync status: success
- New trades: 2

---

### Test 5: Telegram Notifier Component Test

**Purpose:** Test Telegram notifications

**5.1 Test connection:**
```bash
python -c "
from src.telegram_notifier import TelegramNotifier

notifier = TelegramNotifier()
if notifier.enabled:
    print('Telegram is enabled')
    success = notifier.test_connection()
    print(f'Connection test: {success}')
else:
    print('Telegram is disabled (set TELEGRAM_ENABLED=true in .env)')
"
```

**Expected:**
- If enabled: `Connection test: True` and you receive a test message on Telegram
- If disabled: `Telegram is disabled...`

**5.2 Test new trade alert:**
```bash
python -c "
from src.telegram_notifier import TelegramNotifier
from datetime import datetime

notifier = TelegramNotifier()

test_trade = {
    'profile_username': 'behappy',
    'ticker': 'AAPL',
    'strategy': 'CSP',
    'action': 'STO',
    'entry_price': 2.50,
    'quantity': 1,
    'strike_price': 170.00,
    'expiration_date': '2024-12-15',
    'alert_timestamp': datetime.now(),
    'alert_text': 'Test trade alert'
}

if notifier.enabled:
    message_id = notifier.send_new_trade_alert(test_trade)
    print(f'Sent new trade alert, message ID: {message_id}')
else:
    print('Telegram disabled')
"
```

**Expected:** Telegram message received with trade details

**5.3 Test closed trade alert:**
```bash
python -c "
from src.telegram_notifier import TelegramNotifier
from datetime import datetime, timedelta

notifier = TelegramNotifier()

test_trade = {
    'profile_username': 'behappy',
    'ticker': 'AAPL',
    'strategy': 'CSP',
    'entry_price': 2.50,
    'exit_price': 1.25,
    'pnl': 125.00,
    'pnl_percent': 50.0,
    'entry_date': datetime.now() - timedelta(days=7),
    'exit_date': datetime.now()
}

if notifier.enabled:
    message_id = notifier.send_trade_closed_alert(test_trade)
    print(f'Sent closed trade alert, message ID: {message_id}')
else:
    print('Telegram disabled')
"
```

**Expected:** Telegram message received showing profit/loss

---

### Test 6: Full Sync Service Test

**Purpose:** Test the complete sync service end-to-end

**6.1 Dry run (visible browser):**
```bash
python xtrades_sync_service.py --no-headless
```

**Expected output:**
```
======================================================================
XTRADES SYNC - 2025-11-02 14:30:00
======================================================================
Found X active profile(s) to sync

Sync log ID: XX

Initializing browser...
Logging in to Xtrades.net...
Login successful!

[1/X] Processing behappy...
Syncing profile: behappy (ID: 1)
  Found XX alerts for behappy
    Added TICKER - STRATEGY @ $X.XX (ID: XXX)
  Completed behappy: X new, X updated

======================================================================
SYNC SUMMARY
======================================================================
Profiles Synced:   X/X
Total Alerts:      XX
New Trades:        X
Updated Trades:    X
Errors:            0
Duration:          XX.Xs
Status:            SUCCESS
======================================================================
```

**Verify:**
- Browser opens and logs in
- Scrapes each profile
- Displays found alerts
- Adds new trades to database
- Sends Telegram notifications (if enabled)
- Completes successfully

**6.2 Production run (headless):**
```bash
python xtrades_sync_service.py
```

**Expected:**
- No browser visible
- Same output as 6.1
- Completes successfully

**6.3 Verify database changes:**
```sql
-- Check latest sync log
SELECT * FROM xtrades_sync_log
ORDER BY sync_timestamp DESC
LIMIT 1;

-- Check new trades
SELECT
    t.ticker,
    t.strategy,
    t.action,
    t.entry_price,
    t.status,
    t.scraped_at
FROM xtrades_trades t
WHERE t.scraped_at >= NOW() - INTERVAL '5 minutes'
ORDER BY t.scraped_at DESC;

-- Check notifications sent
SELECT
    n.notification_type,
    n.status,
    t.ticker
FROM xtrades_notifications n
JOIN xtrades_trades t ON n.trade_id = t.id
WHERE n.sent_at >= NOW() - INTERVAL '5 minutes'
ORDER BY n.sent_at DESC;
```

**Expected:**
- Sync log shows recent successful sync
- New trades appear in database
- Notifications logged (if Telegram enabled)

---

### Test 7: Batch File Test

**Purpose:** Test Windows batch file execution

**7.1 Run batch file:**
```batch
cd C:\Code\WheelStrategy
xtrades_sync.bat
```

**Expected:**
- Activates virtual environment
- Runs sync service
- Outputs to console
- Appends to `logs\xtrades_sync.log`

**7.2 Verify log file:**
```batch
type logs\xtrades_sync.log
```

**Expected:** Recent sync output appended to log

---

### Test 8: Task Scheduler Test

**Purpose:** Test Windows Task Scheduler integration

**8.1 Create test task manually:**
```powershell
# Open Task Scheduler
taskschd.msc
```

- Create a new task named "Xtrades Sync Test"
- Set action to run `C:\Code\WheelStrategy\xtrades_sync.bat`
- Set trigger to run once, 1 minute from now
- Save

**8.2 Monitor task execution:**

Wait for task to run, then check:

```powershell
# Get last run time and result
Get-ScheduledTaskInfo -TaskName "Xtrades Sync Test"

# View history
Get-WinEvent -LogName Microsoft-Windows-TaskScheduler/Operational |
  Where-Object {$_.Message -like "*Xtrades Sync Test*"} |
  Select TimeCreated, Message -First 5
```

**Expected:**
- Task runs automatically
- LastRunTime shows recent execution
- LastTaskResult = 0 (success)

**8.3 Verify sync ran:**
```sql
SELECT * FROM xtrades_sync_log
ORDER BY sync_timestamp DESC
LIMIT 1;
```

**Expected:** Sync log shows execution at scheduled time

**8.4 Delete test task:**
```powershell
Unregister-ScheduledTask -TaskName "Xtrades Sync Test" -Confirm:$false
```

---

### Test 9: Error Handling Tests

**Purpose:** Verify graceful error handling

**9.1 Test invalid profile:**
```sql
-- Add fake profile
INSERT INTO xtrades_profiles (username, display_name, active)
VALUES ('nonexistent_user_12345', 'Fake User', TRUE);
```

Run sync:
```bash
python xtrades_sync_service.py
```

**Expected:**
- Sync continues despite error
- Error logged: "Profile not found: nonexistent_user_12345"
- Other profiles still sync successfully
- Final status: "partial" or "success"

Clean up:
```sql
DELETE FROM xtrades_profiles WHERE username = 'nonexistent_user_12345';
```

**9.2 Test database disconnection:**
```bash
# Stop PostgreSQL temporarily
net stop postgresql-x64-14

# Run sync
python xtrades_sync_service.py

# Restart PostgreSQL
net start postgresql-x64-14
```

**Expected:**
- Error: "Error connecting to database" or similar
- Service exits gracefully
- Exit code = 1 (failure)

**9.3 Test network error (disconnect internet):**

1. Disconnect internet or enable airplane mode
2. Run: `python xtrades_sync_service.py`
3. Reconnect internet

**Expected:**
- Login fails with network error
- Service logs error and exits gracefully
- No crashes or hangs

**9.4 Test invalid credentials:**

Edit `.env` temporarily:
```bash
XTRADES_PASSWORD=wrong_password
```

Run sync:
```bash
python xtrades_sync_service.py
```

**Expected:**
- Login fails after retries
- Error: "Failed to login after X attempts"
- Exit code = 1

Restore correct password in `.env`.

---

### Test 10: Performance & Load Test

**Purpose:** Test sync performance under load

**10.1 Add multiple profiles:**
```sql
INSERT INTO xtrades_profiles (username, display_name, active, notes)
VALUES
    ('profile1', 'Profile 1', TRUE, 'Load test 1'),
    ('profile2', 'Profile 2', TRUE, 'Load test 2'),
    ('profile3', 'Profile 3', TRUE, 'Load test 3')
ON CONFLICT (username) DO UPDATE SET active = TRUE;
```

**10.2 Run sync and measure time:**
```bash
# Windows
powershell -Command "Measure-Command { python xtrades_sync_service.py }"

# Or manually time it
python xtrades_sync_service.py
```

**Expected:**
- Completes within reasonable time (< 5 minutes for 5 profiles)
- Memory usage stays reasonable (< 1GB)
- No crashes or hangs

**10.3 Check database performance:**
```sql
-- Check query performance
EXPLAIN ANALYZE
SELECT * FROM xtrades_trades
WHERE profile_id = 1
  AND status = 'open'
ORDER BY alert_timestamp DESC
LIMIT 50;
```

**Expected:** Query executes in < 100ms

**10.4 Concurrent run test:**

Open two terminals and run simultaneously:
```bash
# Terminal 1
python xtrades_sync_service.py

# Terminal 2 (start immediately)
python xtrades_sync_service.py
```

**Expected:**
- Both run without conflicts
- Database handles concurrent inserts
- No duplicate trades created
- Both complete successfully

---

## Test Results Checklist

After completing all tests, verify:

- [ ] All dependencies installed correctly
- [ ] Database connection works
- [ ] Scraper can login successfully (both headless and visible)
- [ ] Scraper can retrieve alerts from profiles
- [ ] Alert parsing works correctly
- [ ] Database CRUD operations work
- [ ] Duplicate detection prevents re-adding trades
- [ ] Trade updates work correctly
- [ ] Sync logging records all operations
- [ ] Telegram notifications send successfully (if enabled)
- [ ] Full sync service runs end-to-end
- [ ] Batch file executes correctly
- [ ] Task Scheduler can run the service
- [ ] Error handling works gracefully
- [ ] Performance is acceptable
- [ ] No memory leaks or resource issues

---

## Cleanup After Testing

**Remove test data:**
```sql
-- Remove test trades
DELETE FROM xtrades_trades WHERE ticker = 'TEST';

-- Remove test profiles (optional)
DELETE FROM xtrades_profiles WHERE notes LIKE '%test%' OR notes LIKE '%Load test%';

-- Remove old sync logs (optional)
DELETE FROM xtrades_sync_log WHERE sync_timestamp < NOW() - INTERVAL '7 days';
```

**Clear logs:**
```bash
# Keep only last 7 days
del logs\xtrades_sync_2025*.log
```

---

## Production Deployment Checklist

Before deploying to production:

- [ ] All tests pass successfully
- [ ] Credentials are correct in `.env`
- [ ] Database is backed up
- [ ] Task Scheduler configured for desired frequency (every 5 minutes)
- [ ] Telegram notifications tested and working
- [ ] Log rotation configured (manual or automated)
- [ ] Monitoring alerts set up (email/Telegram for failures)
- [ ] Documentation reviewed
- [ ] Backup/recovery procedure tested
- [ ] Team trained on troubleshooting

---

## Manual Test Script (All-in-One)

Save as `test_xtrades_sync.py`:

```python
#!/usr/bin/env python3
"""
Automated test suite for Xtrades Sync Service
Run this to verify all components work correctly
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all required packages can be imported"""
    print("Testing imports...")
    try:
        import selenium
        import psycopg2
        import telegram
        import undetected_chromedriver
        from bs4 import BeautifulSoup
        from dotenv import load_dotenv
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_database():
    """Test database connection and tables"""
    print("\nTesting database...")
    try:
        from xtrades_db_manager import XtradesDBManager
        db = XtradesDBManager()
        conn = db.get_connection()

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM xtrades_profiles")
        profile_count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        print(f"  ✓ Database connected ({profile_count} profiles)")
        return True
    except Exception as e:
        print(f"  ✗ Database test failed: {e}")
        return False

def test_scraper_login():
    """Test scraper login"""
    print("\nTesting scraper login (headless)...")
    try:
        from xtrades_scraper import XtradesScraper
        scraper = XtradesScraper(headless=True)
        success = scraper.login()
        scraper.close()

        if success:
            print("  ✓ Login successful")
            return True
        else:
            print("  ✗ Login failed")
            return False
    except Exception as e:
        print(f"  ✗ Login test failed: {e}")
        return False

def test_telegram():
    """Test Telegram notifications"""
    print("\nTesting Telegram...")
    try:
        from telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier()

        if notifier.enabled:
            success = notifier.test_connection()
            if success:
                print("  ✓ Telegram connected")
                return True
            else:
                print("  ✗ Telegram connection failed")
                return False
        else:
            print("  - Telegram disabled (skipped)")
            return True
    except Exception as e:
        print(f"  ✗ Telegram test failed: {e}")
        return False

def main():
    print("="*60)
    print("XTRADES SYNC SERVICE - TEST SUITE")
    print("="*60)

    results = {
        'Imports': test_imports(),
        'Database': test_database(),
        'Scraper': test_scraper_login(),
        'Telegram': test_telegram()
    }

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    total = len(results)
    passed = sum(results.values())

    print("="*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*60)

    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

Run it:
```bash
python test_xtrades_sync.py
```

---

**Version:** 1.0
**Last Updated:** 2025-11-02
