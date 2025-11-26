# Windows Service Deployment Guide

**Magnus Background Sync Agent - Windows Service Installation**

---

## Overview

This guide explains how to deploy the Magnus Background Sync Agent as a Windows service that:
- Starts automatically on boot
- Runs continuously in the background
- Syncs Robinhood positions every 5 minutes
- Restarts automatically if it crashes
- Runs even when you're not logged in

---

## Prerequisites

### 1. NSSM (Non-Sucking Service Manager)

NSSM is a lightweight service manager for Windows that makes it easy to install and manage services.

**Download NSSM:**

1. Visit: https://nssm.cc/download
2. Download `nssm-2.24.zip` (or latest version)
3. Extract the ZIP file
4. Copy `nssm.exe` from the `win64` folder to one of:
   - `C:\Code\Magnus\` (recommended - same folder as the installation script)
   - OR any folder in your system PATH (e.g., `C:\Windows\System32`)

**Verify NSSM installation:**
```bash
nssm version
```

If you see the version number, NSSM is installed correctly.

### 2. Administrator Privileges

The installation script requires Administrator privileges to install Windows services.

**How to run as Administrator:**
- Right-click the `.bat` file
- Select "Run as Administrator"

### 3. Working Magnus Installation

Ensure your Magnus installation is working:
- Virtual environment exists at `C:\Code\Magnus\venv`
- Dashboard runs successfully (`streamlit run dashboard.py`)
- Robinhood credentials configured in `.env`

---

## Installation

### Step 1: Download NSSM

If you haven't already:
1. Download NSSM from https://nssm.cc/download
2. Extract `nssm.exe` from the `win64` folder
3. Copy to `C:\Code\Magnus\`

### Step 2: Run Installation Script

**Option 1: Double-click**
```
Right-click: install_background_sync_service.bat
Select: "Run as Administrator"
```

**Option 2: Command line**
```bash
cd C:\Code\Magnus
install_background_sync_service.bat
```

### Step 3: Follow Prompts

The script will:
1. Check for Administrator privileges
2. Verify Python and NSSM are available
3. Check if service already exists
4. Install and configure the service
5. Start the service automatically

**Expected output:**
```
============================================================
Magnus Background Sync - Service Installation
============================================================

[OK] Running with Administrator privileges
[OK] Found Python at: C:\Code\Magnus\venv\Scripts\python.exe
[OK] Found NSSM

Installing service...
[OK] Service installed

Configuring service...
[OK] Service configured

Starting service...
[OK] Service started successfully

============================================================
Service Installed Successfully!
============================================================

Service Name: MagnusBackgroundSync
Display Name: Magnus Background Sync
Startup Type: Automatic (starts on boot)

The service will now sync Robinhood positions every 5 minutes
automatically, even when you're not logged in!
```

---

## Service Management

### Check Service Status

**Script:**
```bash
check_service_status.bat
```

**Manual command:**
```bash
sc query MagnusBackgroundSync
```

**Expected output (running service):**
```
SERVICE_NAME: MagnusBackgroundSync
TYPE               : 10  WIN32_OWN_PROCESS
STATE              : 4  RUNNING
                        (STOPPABLE, NOT_PAUSABLE, ACCEPTS_SHUTDOWN)
WIN32_EXIT_CODE    : 0  (0x0)
SERVICE_EXIT_CODE  : 0  (0x0)
CHECKPOINT         : 0x0
WAIT_HINT          : 0x0
```

### Start Service

```bash
nssm start MagnusBackgroundSync
```

### Stop Service

```bash
nssm stop MagnusBackgroundSync
```

### Restart Service

```bash
nssm restart MagnusBackgroundSync
```

### Uninstall Service

**Script:**
```bash
uninstall_background_sync_service.bat
```

**Manual command:**
```bash
nssm stop MagnusBackgroundSync
nssm remove MagnusBackgroundSync confirm
```

---

## Service Configuration

### What Gets Configured

The installation script configures:

- **Executable:** `C:\Code\Magnus\venv\Scripts\python.exe`
- **Arguments:** `-m src.services.background_sync_agent`
- **Working Directory:** `C:\Code\Magnus`
- **Startup Type:** Automatic (starts on boot)
- **Display Name:** Magnus Background Sync
- **Description:** Magnus Background Position Sync Service - Syncs Robinhood positions every 5 minutes
- **Restart Policy:** Restart on failure (5-second delay)

### Log Files

The service generates 3 log files:

1. **background_sync.log** - Main application log (sync cycles, errors, statistics)
2. **background_sync_stdout.log** - Standard output from Python
3. **background_sync_stderr.log** - Error output from Python

**View logs:**
```bash
# Main log (recommended)
type background_sync.log

# Last 20 lines
powershell -Command "Get-Content background_sync.log -Tail 20"

# Follow log in real-time
powershell -Command "Get-Content background_sync.log -Wait -Tail 20"
```

---

## Verification

### After Installation

1. **Check service is running:**
   ```bash
   check_service_status.bat
   ```

2. **Check log file for sync cycles:**
   ```bash
   type background_sync.log
   ```

   You should see output like:
   ```
   2025-11-21 14:30:00 - BackgroundSyncAgent - INFO - Starting sync cycle #1
   2025-11-21 14:30:01 - BackgroundSyncAgent - INFO - Syncing positions...
   2025-11-21 14:30:03 - BackgroundSyncAgent - INFO - ✓ Synced 15 stock positions
   2025-11-21 14:30:03 - BackgroundSyncAgent - INFO - ✓ Synced 8 option positions
   2025-11-21 14:30:05 - BackgroundSyncAgent - INFO - ✓ Sync cycle #1 completed successfully
   ```

3. **Verify positions in dashboard:**
   - Open Magnus dashboard (`streamlit run dashboard.py`)
   - Navigate to Positions page
   - Positions should show fresh data (updated within last 5 minutes)

---

## Troubleshooting

### Issue: "NSSM not found in PATH"

**Cause:** NSSM is not installed or not in PATH

**Solution:**
1. Download NSSM from https://nssm.cc/download
2. Extract `nssm.exe` from `win64` folder
3. Copy to `C:\Code\Magnus\` OR add to system PATH
4. Run installation script again

### Issue: "This script requires Administrator privileges"

**Cause:** Script not run as Administrator

**Solution:**
- Right-click the `.bat` file
- Select "Run as Administrator"

### Issue: Service fails to start

**Symptoms:**
- Service shows status `STOPPED`
- Error in `background_sync_stderr.log`

**Solution:**
1. Check log files:
   ```bash
   type background_sync_stderr.log
   ```

2. Common issues:
   - **Missing dependencies:** Install with `pip install robin-stocks`
   - **Invalid credentials:** Check `.env` file has valid `ROBINHOOD_USERNAME`, `ROBINHOOD_PASSWORD`, `ROBINHOOD_MFA_CODE`
   - **Database connection:** Ensure PostgreSQL is running

3. Test agent manually:
   ```bash
   python -m src.services.background_sync_agent --once
   ```

### Issue: Service not syncing

**Symptoms:**
- Service shows `RUNNING` status
- But no new log entries
- Positions not updating

**Solution:**
1. Restart service:
   ```bash
   nssm restart MagnusBackgroundSync
   ```

2. Check for errors in logs:
   ```bash
   type background_sync.log | findstr /i error
   ```

3. Verify Robinhood connection:
   ```bash
   python -c "from src.services.robinhood_client import RobinhoodClient; client = RobinhoodClient.get_instance(); client.login()"
   ```

### Issue: Multiple services running

**Symptoms:**
- Duplicate sync log entries
- Higher than expected API usage

**Solution:**
1. Check for duplicate processes:
   ```bash
   tasklist | findstr python
   ```

2. Stop all instances:
   ```bash
   nssm stop MagnusBackgroundSync
   taskkill /IM python.exe /F
   ```

3. Restart service:
   ```bash
   nssm start MagnusBackgroundSync
   ```

---

## Advanced Configuration

### Change Sync Interval

By default, syncs run every 5 minutes (300 seconds). To change:

1. Stop service:
   ```bash
   nssm stop MagnusBackgroundSync
   ```

2. Edit service parameters:
   ```bash
   nssm set MagnusBackgroundSync AppParameters "-m src.services.background_sync_agent --interval 180"
   ```

3. Start service:
   ```bash
   nssm start MagnusBackgroundSync
   ```

**Common intervals:**
- 3 minutes: `--interval 180`
- 5 minutes: `--interval 300` (default)
- 10 minutes: `--interval 600`

### Custom Log File Location

To store logs in a different location:

1. Edit service:
   ```bash
   nssm set MagnusBackgroundSync AppStdout "D:\Logs\magnus_sync.log"
   nssm set MagnusBackgroundSync AppStderr "D:\Logs\magnus_sync_error.log"
   ```

2. Restart service:
   ```bash
   nssm restart MagnusBackgroundSync
   ```

### Disable Auto-Start

To prevent service from starting on boot:

```bash
nssm set MagnusBackgroundSync Start SERVICE_DEMAND_START
```

To re-enable auto-start:

```bash
nssm set MagnusBackgroundSync Start SERVICE_AUTO_START
```

---

## Service vs Manual Operation

### Windows Service (Recommended)

**Pros:**
- Starts automatically on boot
- Runs continuously 24/7
- Restarts on failure
- No need to keep terminal open
- Runs even when not logged in

**Cons:**
- Requires NSSM installation
- Requires Administrator privileges to install
- Slightly harder to debug (logs only)

### Manual Operation (Development)

**Pros:**
- Real-time console output
- Easier to debug
- No installation required
- Can stop with Ctrl+C

**Cons:**
- Must manually start after reboot
- Stops when terminal closes
- Doesn't restart on error
- Requires active user session

**How to run manually:**
```bash
cd C:\Code\Magnus
start_background_sync.bat
```

---

## Monitoring

### Windows Event Viewer

The service logs to Windows Event Viewer:

1. Open Event Viewer (`eventvwr.msc`)
2. Navigate to: Windows Logs → Application
3. Filter by source: MagnusBackgroundSync

### Performance Monitor

Monitor service resource usage:

1. Open Task Manager
2. Go to Details tab
3. Find `python.exe` process
4. Right-click → Set Priority → Normal

Expected resource usage:
- CPU: < 1% average, 5% during sync
- Memory: 150-200MB
- Network: ~6MB/hour

### Database Monitoring

Check sync status from database:

```python
from src.trade_history_sync import TradeHistorySyncService

sync_service = TradeHistorySyncService()
last_sync = sync_service.get_last_sync_time()
print(f"Last sync: {last_sync}")
```

---

## Security Considerations

### Credentials

The service uses credentials from `.env` file:
- `ROBINHOOD_USERNAME`
- `ROBINHOOD_PASSWORD`
- `ROBINHOOD_MFA_CODE`

**Security tips:**
- Use file system permissions to restrict `.env` access
- Don't commit `.env` to version control
- Consider using Windows Credential Manager for sensitive data

### Session Cache

Robinhood session is cached in `~/.xtrades_cache/cookies.pkl`:
- Prevents repeated MFA prompts
- Uses pickle format (binary)
- Delete to force re-login

### Log Files

Log files may contain sensitive information:
- Account numbers
- Portfolio values
- Position details

**Security tips:**
- Restrict log file permissions
- Don't expose logs publicly
- Consider log rotation and encryption

---

## Comparison with Daily Sync Script

### Background Sync Service (This Service)

- **Frequency:** Every 5 minutes (288 times/day)
- **Type:** Continuous service
- **Purpose:** Real-time position tracking
- **Startup:** Automatic (Windows Service)
- **Use Case:** Active trading, live dashboard

### Daily Sync Script (daily_trade_sync.py)

- **Frequency:** Once per day
- **Type:** Scheduled task
- **Purpose:** Historical trade archive
- **Startup:** Task Scheduler or manual
- **Use Case:** EOD reporting, historical analysis

**Recommendation:** Run both
- Background service for real-time data
- Daily script as backup/archive

---

## FAQ

**Q: Will this increase API usage significantly?**

A: At 5-minute intervals, you'll make ~288 syncs per day (~12 requests/hour). Robinhood allows 60 requests/minute, so this is well within limits.

**Q: Can I run this on a remote server?**

A: Yes! The service works on any Windows machine. Just ensure:
1. PostgreSQL is accessible (can be remote)
2. Robinhood credentials are configured
3. Service has network access

**Q: What happens if Robinhood is down?**

A: The service logs the error and continues. It will retry on the next sync cycle (5 minutes later). No manual intervention needed.

**Q: Can I pause syncing temporarily?**

A: Yes, stop the service:
```bash
nssm stop MagnusBackgroundSync
```

Resume when ready:
```bash
nssm start MagnusBackgroundSync
```

**Q: How do I know if positions are fresh?**

A: Check the log file for recent sync cycles:
```bash
powershell -Command "Get-Content background_sync.log -Tail 20"
```

Look for timestamps within last 5 minutes.

**Q: Does this work with paper trading accounts?**

A: Yes, as long as the account is supported by `robin_stocks` library.

---

## Support

### Log Files to Check

1. **background_sync.log** - Main log, check first
2. **background_sync_stderr.log** - Python errors
3. **background_sync_stdout.log** - Debug output

### Useful Commands

```bash
# Service status
check_service_status.bat

# View recent logs
powershell -Command "Get-Content background_sync.log -Tail 50"

# Test connection
python -c "from src.services.robinhood_client import RobinhoodClient; client = RobinhoodClient.get_instance(); client.login()"

# Test single sync
python -m src.services.background_sync_agent --once
```

### Getting Help

1. Check [BACKGROUND_SYNC_GUIDE.md](BACKGROUND_SYNC_GUIDE.md) for detailed usage
2. Check [BACKGROUND_SYNC_SUMMARY.md](BACKGROUND_SYNC_SUMMARY.md) for quick reference
3. Review service logs for error messages
4. Test agent in manual mode for debugging

---

## Summary

✅ **Service Benefits:**
- Automatic startup on boot
- Continuous 24/7 operation
- Self-healing (restarts on failure)
- Background operation (no terminal needed)
- Fresh position data every 5 minutes

✅ **Installation Steps:**
1. Download NSSM from https://nssm.cc/download
2. Copy `nssm.exe` to `C:\Code\Magnus`
3. Run `install_background_sync_service.bat` as Administrator
4. Verify service is running with `check_service_status.bat`

✅ **Management:**
- Start: `nssm start MagnusBackgroundSync`
- Stop: `nssm stop MagnusBackgroundSync`
- Status: `check_service_status.bat`
- Logs: `type background_sync.log`

**Ready to deploy!** The service will keep your Robinhood positions automatically synchronized 24/7.

---

**Last Updated:** November 21, 2025
**Version:** 1.0.0
**Status:** Production Ready
