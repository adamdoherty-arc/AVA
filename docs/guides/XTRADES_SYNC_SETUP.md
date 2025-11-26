# Xtrades Sync Service - Setup & Configuration Guide

## Overview

The Xtrades Sync Service automatically scrapes trade alerts from Xtrades.net profiles every 5 minutes and syncs them to your PostgreSQL database with Telegram notifications.

**Features:**
- Automated scraping every 5 minutes via Windows Task Scheduler
- Login once per sync (efficient browser session management)
- Duplicate detection prevents re-adding trades
- Automatic trade updates when positions are closed
- Telegram notifications for new trades and closed positions
- Comprehensive logging to database and file
- Graceful error handling (continues on profile errors)
- Profile-level sync tracking

---

## Prerequisites

### 1. Required Software
- Python 3.8+ with virtual environment (`venv`)
- PostgreSQL database (with Xtrades schema installed)
- Chrome browser (for Selenium/undetected-chromedriver)
- Windows Task Scheduler (built into Windows)

### 2. Required Python Packages
```bash
pip install selenium undetected-chromedriver beautifulsoup4 psycopg2-binary python-dotenv python-telegram-bot
```

### 3. Environment Variables
Add these to your `.env` file:

```bash
# Xtrades.net Credentials (Discord OAuth)
XTRADES_USERNAME=your_discord_email@example.com
XTRADES_PASSWORD=your_discord_password

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_db_password
DB_NAME=magnus

# Telegram Configuration (Optional)
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 4. Database Setup
Ensure the Xtrades schema is installed:

```bash
psql -U postgres -d magnus -f src/xtrades_schema.sql
```

---

## Installation

### Step 1: Verify Files

Ensure these files exist in your project root:
- `xtrades_sync_service.py` - Main sync service
- `xtrades_sync.bat` - Batch runner for Task Scheduler
- `src/xtrades_scraper.py` - Web scraper
- `src/xtrades_db_manager.py` - Database manager
- `src/telegram_notifier.py` - Telegram notifications

### Step 2: Add Active Profiles

Add Xtrades.net profiles to monitor using the database:

```sql
-- Add profiles to monitor
INSERT INTO xtrades_profiles (username, display_name, active, notes)
VALUES
    ('behappy', 'BeHappy Trader', TRUE, 'Conservative wheel strategy trader'),
    ('username2', 'Display Name', TRUE, 'Notes about this trader')
ON CONFLICT (username) DO NOTHING;
```

Or use Python:

```python
from src.xtrades_db_manager import XtradesDBManager

db = XtradesDBManager()
profile_id = db.add_profile('behappy', 'BeHappy Trader', 'Conservative wheel strategy')
print(f"Added profile ID: {profile_id}")
```

### Step 3: Test Manual Sync

Before scheduling, test the sync manually:

```bash
cd C:\Code\WheelStrategy
call venv\Scripts\activate.bat
python xtrades_sync_service.py
```

**Expected Output:**
```
======================================================================
XTRADES SYNC - 2025-11-02 14:30:00
======================================================================
Found 2 active profile(s) to sync

Sync log ID: 42

Initializing browser...
Logging in to Xtrades.net...
Login successful!

[1/2] Processing behappy...
Syncing profile: behappy (ID: 1)
  Found 15 alerts for behappy
    Added AAPL - CSP @ $2.50 (ID: 101)
    Added TSLA - CC @ $1.75 (ID: 102)
  Completed behappy: 2 new, 0 updated

[2/2] Processing username2...
...

======================================================================
SYNC SUMMARY
======================================================================
Profiles Synced:   2/2
Total Alerts:      25
New Trades:        5
Updated Trades:    1
Errors:            0
Duration:          45.3s
Status:            SUCCESS
======================================================================
```

### Step 4: Test with Visible Browser (Optional)

For debugging, run with browser visible:

```bash
python xtrades_sync_service.py --no-headless
```

---

## Windows Task Scheduler Configuration

### Method 1: Using Task Scheduler GUI

1. **Open Task Scheduler**
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create New Task**
   - Click "Create Task..." (not "Create Basic Task")
   - Name: `Xtrades Sync Service`
   - Description: `Syncs Xtrades.net trade alerts to database every 5 minutes`
   - Check "Run whether user is logged on or not"
   - Check "Run with highest privileges"
   - Configure for: Windows 10/11

3. **Triggers Tab**
   - Click "New..."
   - Begin the task: On a schedule
   - Settings: Daily
   - Recur every: 1 days
   - Start: `9:00:00 AM` (adjust to your timezone)
   - Advanced settings:
     - Check "Repeat task every: 5 minutes"
     - For a duration of: 8 hours (or market hours duration)
     - Check "Enabled"
   - Click OK

   **Note:** You can add multiple triggers for different trading sessions:
   - Trigger 1: 9:00 AM - 5:00 PM (regular market)
   - Trigger 2: 4:00 AM - 9:00 AM (pre-market)
   - Trigger 3: 5:00 PM - 8:00 PM (after-hours)

4. **Actions Tab**
   - Click "New..."
   - Action: Start a program
   - Program/script: `C:\Code\WheelStrategy\xtrades_sync.bat`
   - Start in: `C:\Code\WheelStrategy`
   - Click OK

5. **Conditions Tab**
   - Uncheck "Start the task only if the computer is on AC power"
   - Uncheck "Stop if the computer switches to battery power"
   - Check "Wake the computer to run this task" (optional)

6. **Settings Tab**
   - Check "Allow task to be run on demand"
   - Check "Run task as soon as possible after a scheduled start is missed"
   - If the task fails, restart every: 1 minute
   - Attempt to restart up to: 3 times
   - Check "Stop the task if it runs longer than: 10 minutes"
   - If the running task does not end when requested: "Stop the existing instance"

7. **Save**
   - Click OK
   - Enter your Windows password when prompted

### Method 2: Using Command Line (PowerShell as Administrator)

Create a task with a single command:

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Code\WheelStrategy\xtrades_sync.bat" -WorkingDirectory "C:\Code\WheelStrategy"

$trigger = New-ScheduledTaskTrigger -Daily -At "9:00AM" -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Hours 8)

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName "Xtrades Sync Service" -Action $action -Trigger $trigger -Settings $settings -Description "Syncs Xtrades.net trade alerts every 5 minutes" -RunLevel Highest
```

### Method 3: Import XML Configuration

Save this as `xtrades_sync_task.xml`:

```xml
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Syncs Xtrades.net trade alerts to database every 5 minutes</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <Repetition>
        <Interval>PT5M</Interval>
        <Duration>PT8H</Duration>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2025-11-02T09:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>StopExisting</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <Command>C:\Code\WheelStrategy\xtrades_sync.bat</Command>
      <WorkingDirectory>C:\Code\WheelStrategy</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

Import it:
```powershell
schtasks /create /tn "Xtrades Sync Service" /xml "xtrades_sync_task.xml" /ru "SYSTEM"
```

---

## Monitoring & Maintenance

### 1. Check Sync Logs

**Database Logs:**
```sql
-- View recent sync history
SELECT
    id,
    sync_timestamp,
    profiles_synced,
    new_trades,
    updated_trades,
    duration_seconds,
    status,
    errors
FROM xtrades_sync_log
ORDER BY sync_timestamp DESC
LIMIT 20;

-- Check for failed syncs
SELECT * FROM xtrades_sync_log
WHERE status IN ('failed', 'partial')
ORDER BY sync_timestamp DESC;
```

**File Logs:**
```bash
# View today's log
type logs\xtrades_sync_20251102.log

# View combined log (all syncs appended)
type logs\xtrades_sync.log

# Tail live log (using PowerShell)
Get-Content -Path "logs\xtrades_sync.log" -Wait -Tail 50
```

### 2. Monitor Profile Sync Status

```sql
-- Check profile sync status
SELECT
    username,
    display_name,
    active,
    last_sync,
    last_sync_status,
    total_trades_scraped
FROM xtrades_profiles
ORDER BY last_sync DESC;

-- Find profiles that haven't synced recently
SELECT
    username,
    last_sync,
    last_sync_status
FROM xtrades_profiles
WHERE active = TRUE
  AND (last_sync IS NULL OR last_sync < NOW() - INTERVAL '1 hour');
```

### 3. View Recent Trades

```sql
-- Recent trades across all profiles
SELECT
    p.username,
    t.ticker,
    t.strategy,
    t.action,
    t.entry_price,
    t.status,
    t.alert_timestamp
FROM xtrades_trades t
JOIN xtrades_profiles p ON t.profile_id = p.id
ORDER BY t.alert_timestamp DESC
LIMIT 20;
```

### 4. Check Telegram Notifications

```sql
-- View notification history
SELECT
    n.id,
    t.ticker,
    t.strategy,
    n.notification_type,
    n.status,
    n.sent_at,
    n.error_message
FROM xtrades_notifications n
JOIN xtrades_trades t ON n.trade_id = t.id
ORDER BY n.sent_at DESC
LIMIT 20;

-- Check for failed notifications
SELECT * FROM xtrades_notifications
WHERE status = 'failed'
ORDER BY sent_at DESC;
```

### 5. Task Scheduler Monitoring

**Check task status:**
```powershell
# View task details
Get-ScheduledTask -TaskName "Xtrades Sync Service" | Select *

# View last run result
Get-ScheduledTaskInfo -TaskName "Xtrades Sync Service"

# View task history (last 10 runs)
Get-WinEvent -LogName Microsoft-Windows-TaskScheduler/Operational |
  Where-Object {$_.Message -like "*Xtrades Sync Service*"} |
  Select TimeCreated, Message -First 10
```

---

## Troubleshooting

### Problem: Sync Not Running

**Check if task is enabled:**
```powershell
Get-ScheduledTask -TaskName "Xtrades Sync Service"
```

**Run manually to test:**
```powershell
Start-ScheduledTask -TaskName "Xtrades Sync Service"
```

**Check last run result:**
```powershell
Get-ScheduledTaskInfo -TaskName "Xtrades Sync Service" | Select LastRunTime, LastTaskResult
```

Result codes:
- `0`: Success
- `1`: General failure
- `0x41301`: Task is currently running
- `0x41303`: Task has not yet run

### Problem: Login Failures

**Symptoms:** Logs show "Failed to login to Xtrades.net"

**Solutions:**

1. **Check credentials in `.env`:**
   ```bash
   # Verify XTRADES_USERNAME and XTRADES_PASSWORD are correct
   # These should be your Discord credentials
   ```

2. **Test login manually:**
   ```bash
   python -c "from src.xtrades_scraper import XtradesScraper; s = XtradesScraper(headless=False); s.login()"
   ```

3. **Clear cookies cache:**
   ```bash
   # Delete cached session
   del %USERPROFILE%\.xtrades_cache\cookies.pkl
   ```

4. **Discord 2FA:** If you have 2FA enabled, you may need to:
   - Temporarily disable 2FA, or
   - Generate an app-specific password

### Problem: Profile Not Found

**Symptoms:** "Profile not found: username"

**Solutions:**

1. **Verify profile exists on Xtrades.net:**
   - Navigate to `https://app.xtrades.net/profile/username`
   - Ensure the profile is public and accessible

2. **Check username spelling:**
   ```sql
   SELECT username FROM xtrades_profiles WHERE active = TRUE;
   ```

3. **Update profile status:**
   ```sql
   UPDATE xtrades_profiles SET active = FALSE WHERE username = 'problematic_user';
   ```

### Problem: No Trades Found

**Symptoms:** "Found 0 alerts for username"

**Possible Causes:**

1. **Profile has no recent alerts** - This is normal if the trader hasn't posted
2. **Scraper selectors changed** - Xtrades.net may have updated their HTML structure
3. **Profile is private or requires login** - Check access permissions

**Debug:**
```bash
# Run with visible browser to see what's happening
python xtrades_sync_service.py --no-headless
```

### Problem: Duplicate Trades

**Symptoms:** Same trade appearing multiple times in database

**Solutions:**

1. **Check timestamp precision:**
   - Duplicate detection uses 1-minute tolerance
   - Ensure alert timestamps are consistent

2. **Manually remove duplicates:**
   ```sql
   -- Find duplicates
   SELECT ticker, alert_timestamp, COUNT(*)
   FROM xtrades_trades
   GROUP BY ticker, alert_timestamp
   HAVING COUNT(*) > 1;

   -- Delete duplicates (keep lowest ID)
   DELETE FROM xtrades_trades t1
   USING xtrades_trades t2
   WHERE t1.id > t2.id
     AND t1.ticker = t2.ticker
     AND t1.alert_timestamp = t2.alert_timestamp;
   ```

### Problem: Telegram Notifications Not Sending

**Symptoms:** Trades added but no Telegram messages

**Check:**

1. **Telegram enabled:**
   ```bash
   # In .env file
   TELEGRAM_ENABLED=true
   ```

2. **Valid credentials:**
   ```bash
   # Test Telegram connection
   python -c "from src.telegram_notifier import TelegramNotifier; n = TelegramNotifier(); n.test_connection()"
   ```

3. **Check notification logs:**
   ```sql
   SELECT * FROM xtrades_notifications
   WHERE status = 'failed'
   ORDER BY sent_at DESC
   LIMIT 10;
   ```

### Problem: Browser Crashes or Hangs

**Symptoms:** Task runs forever or Chrome crashes

**Solutions:**

1. **Update Chrome and chromedriver:**
   ```bash
   pip install --upgrade undetected-chromedriver
   ```

2. **Increase timeout in Task Scheduler:**
   - Settings > Stop the task if it runs longer than: 15 minutes

3. **Check system resources:**
   - Ensure sufficient RAM (Chrome needs ~500MB-1GB)
   - Close other Chrome instances during sync

4. **Run in headless mode:**
   - Edit `xtrades_sync.bat` to ensure headless mode is used

### Problem: Database Connection Errors

**Symptoms:** "Error connecting to database" or "psycopg2 errors"

**Solutions:**

1. **Verify PostgreSQL is running:**
   ```powershell
   Get-Service postgresql*
   ```

2. **Test connection:**
   ```bash
   psql -U postgres -d magnus
   ```

3. **Check `.env` configuration:**
   ```bash
   DB_HOST=localhost
   DB_PORT=5432
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_NAME=magnus
   ```

4. **Check network access:**
   - Ensure PostgreSQL accepts local connections
   - Check `pg_hba.conf` for authentication settings

---

## Manual Operations

### Run Sync Manually

```bash
cd C:\Code\WheelStrategy
call venv\Scripts\activate.bat
python xtrades_sync_service.py
```

### Run with Visible Browser (Debug Mode)

```bash
python xtrades_sync_service.py --no-headless
```

### Force Sync Single Profile

```python
from src.xtrades_scraper import XtradesScraper
from src.xtrades_db_manager import XtradesDBManager

# Initialize
db = XtradesDBManager()
scraper = XtradesScraper(headless=False)

# Login
scraper.login()

# Get profile
profile = db.get_profile_by_username('behappy')

# Scrape alerts
alerts = scraper.get_profile_alerts('behappy', max_alerts=50)
print(f"Found {len(alerts)} alerts")

# Close
scraper.close()
```

### Add Profile Manually

```sql
INSERT INTO xtrades_profiles (username, display_name, active, notes)
VALUES ('newtrader', 'New Trader', TRUE, 'Newly added trader to monitor')
ON CONFLICT (username) DO UPDATE
  SET active = TRUE,
      display_name = EXCLUDED.display_name,
      notes = EXCLUDED.notes;
```

### Deactivate/Reactivate Profile

```sql
-- Deactivate (stop syncing)
UPDATE xtrades_profiles SET active = FALSE WHERE username = 'username';

-- Reactivate
UPDATE xtrades_profiles SET active = TRUE WHERE username = 'username';
```

### Delete Old Sync Logs

```sql
-- Keep only last 30 days
DELETE FROM xtrades_sync_log
WHERE sync_timestamp < NOW() - INTERVAL '30 days';
```

### Clear All Trades for Profile (Caution!)

```sql
-- This will delete all trades for a specific profile
DELETE FROM xtrades_trades
WHERE profile_id = (SELECT id FROM xtrades_profiles WHERE username = 'username');

-- Reset trade counter
UPDATE xtrades_profiles
SET total_trades_scraped = 0
WHERE username = 'username';
```

---

## Performance Optimization

### 1. Database Indexing

Indexes are already created by the schema, but verify:

```sql
-- Check existing indexes
SELECT schemaname, tablename, indexname
FROM pg_indexes
WHERE tablename LIKE 'xtrades_%'
ORDER BY tablename, indexname;
```

### 2. Adjust Sync Frequency

For different trading styles:

- **Day traders:** Every 1-2 minutes during market hours
- **Swing traders:** Every 5-10 minutes
- **Long-term:** Every 15-30 minutes or hourly

Edit the Task Scheduler trigger interval accordingly.

### 3. Limit Alerts Per Profile

In `xtrades_sync_service.py`, adjust:

```python
# Line ~191
alerts = self.scraper.get_profile_alerts(username, max_alerts=50)

# Change to 25 for faster syncs:
alerts = self.scraper.get_profile_alerts(username, max_alerts=25)
```

### 4. Browser Performance

For faster scraping:
- Use headless mode (default)
- Disable images: Add to scraper options
- Reduce scroll count: Modify `_scroll_page()` in scraper

---

## Security Considerations

### 1. Protect Credentials

- **Never commit `.env` to Git**
  ```bash
  # Ensure .env is in .gitignore
  echo ".env" >> .gitignore
  ```

- **Use strong passwords** for Discord/Xtrades and database

- **Limit database user permissions:**
  ```sql
  -- Create dedicated user for sync service
  CREATE USER xtrades_sync WITH PASSWORD 'strong_password';
  GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO xtrades_sync;
  ```

### 2. Task Scheduler Security

- Run task as a specific user (not SYSTEM)
- Use "Run whether user is logged on or not" for background operation
- Store credentials securely in Windows Credential Manager

### 3. Network Security

- Use localhost for database connections when possible
- Enable SSL for database connections in production
- Whitelist IPs in `pg_hba.conf`

---

## Backup & Recovery

### 1. Backup Xtrades Data

```bash
# Backup all Xtrades tables
pg_dump -U postgres -d magnus -t xtrades_profiles -t xtrades_trades -t xtrades_sync_log -t xtrades_notifications > xtrades_backup.sql

# Restore
psql -U postgres -d magnus < xtrades_backup.sql
```

### 2. Export Trades to CSV

```sql
-- Export all trades
COPY (
  SELECT
    p.username,
    t.ticker,
    t.strategy,
    t.action,
    t.entry_price,
    t.entry_date,
    t.exit_price,
    t.exit_date,
    t.pnl,
    t.status,
    t.alert_timestamp
  FROM xtrades_trades t
  JOIN xtrades_profiles p ON t.profile_id = p.id
  ORDER BY t.alert_timestamp DESC
) TO 'C:/exports/xtrades_all_trades.csv' WITH CSV HEADER;
```

### 3. Scheduled Database Backups

Add to Task Scheduler (daily at 2 AM):

```batch
@echo off
set BACKUP_DIR=C:\Backups\Xtrades
set BACKUP_FILE=%BACKUP_DIR%\xtrades_%date:~-4,4%%date:~-10,2%%date:~-7,2%.sql

pg_dump -U postgres -d magnus -t xtrades_* > "%BACKUP_FILE%"
```

---

## Advanced Configuration

### 1. Multi-Environment Setup

For dev/staging/production:

```bash
# .env.development
XTRADES_USERNAME=dev@example.com
DB_NAME=magnus_dev
TELEGRAM_ENABLED=false

# .env.production
XTRADES_USERNAME=prod@example.com
DB_NAME=magnus
TELEGRAM_ENABLED=true

# Load specific environment
python xtrades_sync_service.py --env production
```

### 2. Custom Notification Rules

Modify `_process_alert()` in `xtrades_sync_service.py`:

```python
# Only notify for trades above $1000
if alert.get('entry_price', 0) * alert.get('quantity', 0) * 100 >= 1000:
    self.notifier.send_new_trade_alert(trade_data)
```

### 3. Email Notifications (Alternative to Telegram)

Install:
```bash
pip install sendgrid
```

Add to sync service:
```python
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_email_alert(trade_data):
    message = Mail(
        from_email='alerts@example.com',
        to_emails='your_email@example.com',
        subject=f"New Trade: {trade_data['ticker']}",
        html_content=f"<strong>{trade_data['strategy']}</strong>"
    )
    sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
    response = sg.send(message)
```

---

## FAQ

**Q: How many profiles can I monitor?**
A: Technically unlimited, but for practical purposes, 5-10 profiles work well. Each profile adds ~30-60 seconds to sync time.

**Q: What happens if Xtrades.net changes their website?**
A: The scraper may break. You'll need to update the selectors in `xtrades_scraper.py`. Check logs for "Found 0 alerts" or scraping errors.

**Q: Can I run multiple instances simultaneously?**
A: Not recommended. The Task Scheduler is configured to stop existing instances before starting new ones to prevent conflicts.

**Q: How much disk space does this use?**
A: Minimal. Logs rotate daily (~1-5MB per day). Database grows slowly (~1KB per trade). Budget ~100MB for logs, ~10MB for database per year.

**Q: Can I sync historical data?**
A: Yes, but manually. The scraper only gets recent alerts (last 50 by default). For historical data, you'd need to scrape the profile page extensively or use Xtrades' API if available.

**Q: What if I miss a sync (computer off)?**
A: Task Scheduler's "Run task as soon as possible after a scheduled start is missed" setting will catch up on the next run.

**Q: How do I know if everything is working?**
A: Check:
1. Database sync logs show recent successful syncs
2. Telegram receives test notifications
3. Task Scheduler shows last result = 0 (success)
4. Log files show no errors

---

## Support & Debugging

### Enable Verbose Logging

Edit `xtrades_sync_service.py`:

```python
# Change line ~48
logger.setLevel(logging.DEBUG)
```

### Get Help

1. **Check logs first:** `logs\xtrades_sync_YYYYMMDD.log`
2. **Run manual test:** `python xtrades_sync_service.py --no-headless`
3. **Check database logs:** `SELECT * FROM xtrades_sync_log ORDER BY sync_timestamp DESC LIMIT 5;`
4. **Review Task Scheduler history:** Task Scheduler > History tab

---

## Appendix

### File Structure
```
WheelStrategy/
├── xtrades_sync_service.py    # Main sync service
├── xtrades_sync.bat            # Batch runner
├── XTRADES_SYNC_SETUP.md       # This file
├── src/
│   ├── xtrades_scraper.py      # Web scraper
│   ├── xtrades_db_manager.py   # Database manager
│   ├── telegram_notifier.py    # Telegram notifications
│   ├── xtrades_schema.sql      # Database schema
│   └── ...
├── logs/
│   ├── xtrades_sync_20251102.log  # Daily detailed logs
│   └── xtrades_sync.log           # Combined log
└── venv/                       # Virtual environment
```

### Database Schema Quick Reference

**Tables:**
- `xtrades_profiles` - Profiles to monitor
- `xtrades_trades` - Individual trades
- `xtrades_sync_log` - Sync operation history
- `xtrades_notifications` - Notification tracking

**Key Columns:**
- `xtrades_profiles.active` - Enable/disable syncing
- `xtrades_trades.status` - 'open', 'closed', 'expired'
- `xtrades_sync_log.status` - 'success', 'partial', 'failed'

### Useful SQL Queries

```sql
-- Today's new trades
SELECT COUNT(*) FROM xtrades_trades
WHERE scraped_at >= CURRENT_DATE;

-- Sync success rate (last 100 runs)
SELECT
    status,
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM (SELECT status FROM xtrades_sync_log ORDER BY sync_timestamp DESC LIMIT 100) s
GROUP BY status;

-- Most active traders
SELECT
    p.username,
    COUNT(t.id) as total_trades,
    COUNT(CASE WHEN t.status = 'open' THEN 1 END) as open_trades,
    COUNT(CASE WHEN t.status = 'closed' THEN 1 END) as closed_trades
FROM xtrades_profiles p
LEFT JOIN xtrades_trades t ON p.id = t.profile_id
GROUP BY p.username
ORDER BY total_trades DESC;
```

---

**Version:** 1.0
**Last Updated:** 2025-11-02
**Maintained By:** Magnus Trading Dashboard Team
