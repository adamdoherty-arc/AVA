# Xtrades Sync Service - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies (1 min)

```bash
cd C:\Code\WheelStrategy
call venv\Scripts\activate.bat
pip install selenium undetected-chromedriver beautifulsoup4 psycopg2-binary python-dotenv python-telegram-bot
```

### 2. Configure Environment (1 min)

Edit `.env` file:

```bash
# Xtrades Credentials (your Discord account)
XTRADES_USERNAME=your_discord_email@gmail.com
XTRADES_PASSWORD=your_discord_password

# Database (should already be configured)
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_db_password
DB_NAME=magnus

# Telegram (optional)
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

### 3. Add Profiles to Monitor (1 min)

```sql
-- Connect to database
psql -U postgres -d magnus

-- Add profiles
INSERT INTO xtrades_profiles (username, display_name, active)
VALUES
    ('behappy', 'BeHappy Trader', TRUE),
    ('other_trader', 'Other Trader', TRUE)
ON CONFLICT (username) DO NOTHING;
```

### 4. Test Sync Manually (1 min)

```bash
python xtrades_sync_service.py --no-headless
```

Watch the browser log in and scrape. You should see output like:

```
======================================================================
XTRADES SYNC - 2025-11-02 14:30:00
======================================================================
Found 2 active profile(s) to sync
...
Status: SUCCESS
======================================================================
```

### 5. Schedule with Task Scheduler (1 min)

**PowerShell (as Administrator):**

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Code\WheelStrategy\xtrades_sync.bat" -WorkingDirectory "C:\Code\WheelStrategy"

$trigger = New-ScheduledTaskTrigger -Daily -At "9:00AM" -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Hours 8)

Register-ScheduledTask -TaskName "Xtrades Sync Service" -Action $action -Trigger $trigger -Description "Syncs Xtrades trade alerts every 5 minutes" -RunLevel Highest
```

**Done!** The service will now run every 5 minutes from 9 AM to 5 PM daily.

---

## Daily Operations

### Check Last Sync

```sql
SELECT
    sync_timestamp,
    profiles_synced,
    new_trades,
    status
FROM xtrades_sync_log
ORDER BY sync_timestamp DESC
LIMIT 5;
```

### View Recent Trades

```sql
SELECT
    p.username,
    t.ticker,
    t.strategy,
    t.entry_price,
    t.status,
    t.alert_timestamp
FROM xtrades_trades t
JOIN xtrades_profiles p ON t.profile_id = p.id
ORDER BY t.alert_timestamp DESC
LIMIT 20;
```

### Check Logs

```bash
# View today's detailed log
type logs\xtrades_sync_20251102.log

# View combined log
type logs\xtrades_sync.log

# Live tail (PowerShell)
Get-Content -Path "logs\xtrades_sync.log" -Wait -Tail 20
```

### Manual Sync

```bash
cd C:\Code\WheelStrategy
call venv\Scripts\activate.bat
python xtrades_sync_service.py
```

---

## Common Commands

### Add New Profile

```sql
INSERT INTO xtrades_profiles (username, display_name, active, notes)
VALUES ('new_trader', 'New Trader', TRUE, 'Just added')
ON CONFLICT (username) DO UPDATE SET active = TRUE;
```

### Deactivate Profile

```sql
UPDATE xtrades_profiles SET active = FALSE WHERE username = 'trader_name';
```

### Reactivate Profile

```sql
UPDATE xtrades_profiles SET active = TRUE WHERE username = 'trader_name';
```

### View All Active Profiles

```sql
SELECT username, display_name, last_sync, total_trades_scraped
FROM xtrades_profiles
WHERE active = TRUE
ORDER BY username;
```

### Delete Old Sync Logs

```sql
DELETE FROM xtrades_sync_log
WHERE sync_timestamp < NOW() - INTERVAL '30 days';
```

---

## Task Scheduler Operations

### Check Task Status

```powershell
Get-ScheduledTaskInfo -TaskName "Xtrades Sync Service"
```

### Run Task Manually

```powershell
Start-ScheduledTask -TaskName "Xtrades Sync Service"
```

### Disable Task

```powershell
Disable-ScheduledTask -TaskName "Xtrades Sync Service"
```

### Enable Task

```powershell
Enable-ScheduledTask -TaskName "Xtrades Sync Service"
```

### View Task History

```powershell
Get-WinEvent -LogName Microsoft-Windows-TaskScheduler/Operational |
  Where-Object {$_.Message -like "*Xtrades Sync*"} |
  Select TimeCreated, Message -First 10
```

---

## Troubleshooting Quick Fixes

### Problem: Login Fails

```bash
# Clear cookies cache
del %USERPROFILE%\.xtrades_cache\cookies.pkl

# Test login with visible browser
python xtrades_sync_service.py --no-headless
```

### Problem: No Trades Found

```sql
-- Check profile is active
SELECT * FROM xtrades_profiles WHERE username = 'your_username';

-- Update if needed
UPDATE xtrades_profiles SET active = TRUE WHERE username = 'your_username';
```

### Problem: Task Not Running

```powershell
# Check task exists and is enabled
Get-ScheduledTask -TaskName "Xtrades Sync Service"

# Check last result
Get-ScheduledTaskInfo -TaskName "Xtrades Sync Service" | Select LastRunTime, LastTaskResult

# Run manually to test
Start-ScheduledTask -TaskName "Xtrades Sync Service"
```

### Problem: Database Errors

```bash
# Check PostgreSQL is running
net start | findstr postgres

# Test connection
psql -U postgres -d magnus -c "SELECT 1"
```

### Problem: Telegram Not Working

```bash
# Test Telegram
python -c "from src.telegram_notifier import TelegramNotifier; TelegramNotifier().test_connection()"

# Check .env
echo %TELEGRAM_ENABLED%
echo %TELEGRAM_BOT_TOKEN%
```

---

## Performance Tips

1. **Adjust sync frequency** based on activity:
   - Day traders: Every 2-3 minutes
   - Swing traders: Every 5-10 minutes
   - Position traders: Every 15-30 minutes

2. **Limit alerts per profile** in `xtrades_sync_service.py`:
   ```python
   # Line ~191
   alerts = self.scraper.get_profile_alerts(username, max_alerts=25)
   ```

3. **Monitor only active profiles** - Deactivate inactive traders

4. **Clean up old data** monthly:
   ```sql
   DELETE FROM xtrades_sync_log WHERE sync_timestamp < NOW() - INTERVAL '90 days';
   ```

---

## Monitoring Dashboard (SQL Views)

### Sync Health Dashboard

```sql
-- Last 24 hours sync performance
SELECT
    DATE_TRUNC('hour', sync_timestamp) as hour,
    COUNT(*) as total_syncs,
    SUM(new_trades) as new_trades,
    AVG(duration_seconds) as avg_duration,
    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful
FROM xtrades_sync_log
WHERE sync_timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', sync_timestamp)
ORDER BY hour DESC;
```

### Profile Activity

```sql
-- Most active profiles
SELECT
    p.username,
    COUNT(t.id) as total_trades,
    COUNT(CASE WHEN t.status = 'open' THEN 1 END) as open_trades,
    MAX(t.alert_timestamp) as last_alert
FROM xtrades_profiles p
LEFT JOIN xtrades_trades t ON p.id = t.profile_id
WHERE p.active = TRUE
GROUP BY p.username
ORDER BY total_trades DESC;
```

### Today's Activity

```sql
SELECT
    COUNT(DISTINCT profile_id) as active_profiles,
    COUNT(*) as total_trades,
    COUNT(CASE WHEN status = 'open' THEN 1 END) as new_positions,
    COUNT(CASE WHEN status = 'closed' THEN 1 END) as closed_positions,
    COALESCE(SUM(pnl), 0) as total_pnl
FROM xtrades_trades
WHERE scraped_at >= CURRENT_DATE;
```

---

## File Locations

- **Service:** `C:\Code\WheelStrategy\xtrades_sync_service.py`
- **Batch file:** `C:\Code\WheelStrategy\xtrades_sync.bat`
- **Logs:** `C:\Code\WheelStrategy\logs\xtrades_sync*.log`
- **Config:** `C:\Code\WheelStrategy\.env`
- **Documentation:**
  - Setup: `XTRADES_SYNC_SETUP.md`
  - Testing: `XTRADES_SYNC_TEST.md`
  - Quick Start: `XTRADES_SYNC_QUICK_START.md` (this file)

---

## Support Resources

### Check Service Status

```bash
# Quick health check
python -c "from src.xtrades_db_manager import XtradesDBManager; db = XtradesDBManager(); print('Last sync:', db.get_latest_sync())"
```

### Full System Test

```bash
python test_xtrades_sync.py
```

### View Full Documentation

- **Setup Guide:** `XTRADES_SYNC_SETUP.md` - Complete setup instructions
- **Test Guide:** `XTRADES_SYNC_TEST.md` - Comprehensive testing procedures
- **This Guide:** `XTRADES_SYNC_QUICK_START.md` - Quick reference

---

## Success Indicators

Your sync service is working correctly if:

1. ✅ Task Scheduler shows recent successful runs (LastTaskResult = 0)
2. ✅ Database shows recent sync logs with status 'success'
3. ✅ New trades appear in database within 5 minutes of posting
4. ✅ Telegram notifications arrive for new trades (if enabled)
5. ✅ Log files show no errors
6. ✅ Profile sync timestamps update every 5 minutes

---

**Need Help?** Refer to `XTRADES_SYNC_SETUP.md` for detailed troubleshooting.
