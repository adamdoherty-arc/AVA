# Background Sync Agent - User Guide

**Continuous Robinhood Position Synchronization Service**

---

## Overview

The Background Sync Agent runs continuously in the background to keep your Robinhood data fresh and up-to-date. It syncs every 5 minutes (configurable) without requiring manual intervention.

### What Gets Synced

Every 5 minutes:
- ✅ **Stock Positions** - Quantities, prices, P&L
- ✅ **Option Positions** - Strikes, expirations, Greeks, P&L
- ✅ **Account Info** - Buying power, portfolio value
- ✅ **Portfolio Balance** - Daily snapshots

Every hour:
- ✅ **Trade History** - Closed positions and completed trades

---

## Quick Start

### Windows

**Double-click to start:**
```
start_background_sync.bat
```

Or from command line:
```bash
cd C:\Code\Magnus
start_background_sync.bat
```

### Command Line (Any OS)

**Standard mode (5-minute sync):**
```bash
python -m src.services.background_sync_agent
```

**Custom interval (e.g., 3 minutes = 180 seconds):**
```bash
python -m src.services.background_sync_agent --interval 180
```

**Run once (test mode):**
```bash
python -m src.services.background_sync_agent --once
```

---

## How It Works

### Sync Cycle

Each sync cycle performs these steps:

1. **Connect to Robinhood** - Uses cached session (no MFA required)
2. **Fetch Positions** - Get current stock and option positions
3. **Fetch Account Info** - Get buying power and portfolio value
4. **Sync Trades** - Sync closed trades to database (hourly)
5. **Record Balance** - Save portfolio balance snapshot
6. **Invalidate Cache** - Clear position cache for fresh data

### Timing

- **Default interval:** 5 minutes (300 seconds)
- **Trade sync:** Every 1 hour (to reduce API load)
- **First sync:** Immediate on startup
- **Graceful shutdown:** Ctrl+C to stop

### Logging

All activity is logged to:
- **Console output** - Real-time sync status
- **background_sync.log** - Persistent log file

Log format:
```
2025-11-21 14:30:00 - BackgroundSyncAgent - INFO - Starting sync cycle #12
2025-11-21 14:30:01 - BackgroundSyncAgent - INFO - Syncing positions...
2025-11-21 14:30:03 - BackgroundSyncAgent - INFO - ✓ Synced 15 stock positions
2025-11-21 14:30:03 - BackgroundSyncAgent - INFO - ✓ Synced 8 option positions
2025-11-21 14:30:04 - BackgroundSyncAgent - INFO - ✓ Buying Power: $5,432.10
2025-11-21 14:30:04 - BackgroundSyncAgent - INFO - ✓ Portfolio Value: $45,678.90
2025-11-21 14:30:05 - BackgroundSyncAgent - INFO - ✓ Portfolio balance recorded
2025-11-21 14:30:05 - BackgroundSyncAgent - INFO - ✓ Sync cycle #12 completed successfully
2025-11-21 14:30:05 - BackgroundSyncAgent - INFO - Cycle duration: 5.23s
2025-11-21 14:30:05 - BackgroundSyncAgent - INFO - Next sync in 300s
```

---

## Configuration

### Environment Variables

Required (already set if dashboard works):
- `ROBINHOOD_USERNAME` - Your Robinhood username
- `ROBINHOOD_PASSWORD` - Your Robinhood password
- `ROBINHOOD_MFA_CODE` - Your TOTP secret for 2FA

Optional:
- `DATABASE_URL` - PostgreSQL connection string

### Sync Interval

Change sync frequency with `--interval`:

```bash
# Every 3 minutes (180 seconds)
python -m src.services.background_sync_agent --interval 180

# Every 10 minutes (600 seconds)
python -m src.services.background_sync_agent --interval 600

# Every 1 minute (60 seconds) - NOT RECOMMENDED (rate limits)
python -m src.services.background_sync_agent --interval 60
```

**Recommended intervals:**
- **5 minutes (default)** - Good balance of freshness and API limits
- **3 minutes** - More frequent updates, higher API usage
- **10 minutes** - Conservative, lower API usage

⚠️ **Warning:** Intervals < 2 minutes may trigger Robinhood rate limits

---

## Usage Scenarios

### Scenario 1: Desktop Trading Workstation

**Goal:** Keep positions fresh while using dashboard

**Setup:**
1. Start background sync agent in separate window
2. Open dashboard in browser
3. Positions update automatically every 5 minutes

**Commands:**
```bash
# Terminal 1 - Start sync agent
cd C:\Code\Magnus
start_background_sync.bat

# Terminal 2 - Start dashboard
cd C:\Code\Magnus
streamlit run dashboard.py --server.port 8502
```

---

### Scenario 2: 24/7 Server Deployment

**Goal:** Continuous sync on server/VPS

**Setup:**
1. Install as Windows Service (Windows) or systemd service (Linux)
2. Agent runs automatically on startup
3. Dashboard accesses fresh data from database

**Windows Service (with NSSM):**
```bash
# Install NSSM (Non-Sucking Service Manager)
# Download from: https://nssm.cc/download

# Install service
nssm install MagnusBackgroundSync "C:\Code\Magnus\venv\Scripts\python.exe" "-m src.services.background_sync_agent"
nssm set MagnusBackgroundSync AppDirectory "C:\Code\Magnus"
nssm set MagnusBackgroundSync Description "Magnus Background Position Sync Service"
nssm set MagnusBackgroundSync Start SERVICE_AUTO_START

# Start service
nssm start MagnusBackgroundSync

# Check status
nssm status MagnusBackgroundSync

# Stop service
nssm stop MagnusBackgroundSync

# Uninstall service
nssm remove MagnusBackgroundSync confirm
```

**Linux systemd Service:**
```bash
# Create service file
sudo nano /etc/systemd/system/magnus-sync.service
```

```ini
[Unit]
Description=Magnus Background Position Sync
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/Magnus
Environment="PATH=/home/youruser/Magnus/venv/bin"
ExecStart=/home/youruser/Magnus/venv/bin/python -m src.services.background_sync_agent
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable magnus-sync
sudo systemctl start magnus-sync

# Check status
sudo systemctl status magnus-sync

# View logs
sudo journalctl -u magnus-sync -f
```

---

### Scenario 3: Test & Debug

**Goal:** Test sync without running continuously

**Commands:**
```bash
# Run single sync cycle
python -m src.services.background_sync_agent --once

# Run with verbose logging
python -m src.services.background_sync_agent --once 2>&1 | tee sync_test.log

# Check if services initialize correctly
python -c "from src.services.background_sync_agent import BackgroundSyncAgent; agent = BackgroundSyncAgent(); agent._initialize_services()"
```

---

## Monitoring

### Check Sync Status

**View log file:**
```bash
# Windows
type background_sync.log

# Linux/Mac
tail -f background_sync.log
```

**Check last sync time:**
```python
from src.trade_history_sync import TradeHistorySyncService

sync_service = TradeHistorySyncService()
last_sync = sync_service.get_last_sync_time()
print(f"Last sync: {last_sync}")
```

**Check position cache:**
```python
from src.services.positions_connector import PositionsConnector

connector = PositionsConnector()
positions = connector.fetch_positions()
print(f"Cached positions: {len(positions)}")
```

### Performance Metrics

The agent tracks:
- **Sync count** - Total successful sync cycles
- **Error count** - Total errors encountered
- **Last sync time** - Timestamp of last successful sync
- **Cycle duration** - How long each sync takes

View statistics on shutdown:
```
============================================================
Agent Statistics:
  Total sync cycles: 48
  Total errors: 0
  Last successful sync: 2025-11-21 15:30:00
============================================================
```

---

## Troubleshooting

### Issue: Agent won't start

**Symptoms:**
- "Failed to initialize services" error
- Agent exits immediately

**Solutions:**
1. Check Robinhood credentials in `.env` file
2. Verify virtual environment is activated
3. Test Robinhood connection manually:
   ```bash
   python -c "from src.services.robinhood_client import RobinhoodClient; client = RobinhoodClient.get_instance(); client.login()"
   ```

---

### Issue: MFA errors

**Symptoms:**
- "MFA required" error
- Agent stops after initial login

**Solutions:**
1. Ensure `ROBINHOOD_MFA_CODE` is set in `.env`
2. Test MFA manually:
   ```bash
   python -c "import pyotp; print(pyotp.TOTP('YOUR_MFA_SECRET').now())"
   ```
3. Login manually once to cache session:
   ```bash
   python show_my_positions.py
   ```

---

### Issue: Rate limit errors

**Symptoms:**
- "Too many requests" errors
- 429 HTTP status codes

**Solutions:**
1. Increase sync interval:
   ```bash
   python -m src.services.background_sync_agent --interval 600  # 10 minutes
   ```
2. Check if multiple agents are running:
   ```bash
   # Windows
   tasklist | findstr python

   # Linux/Mac
   ps aux | grep background_sync_agent
   ```
3. Wait 5-10 minutes before restarting

---

### Issue: Database connection errors

**Symptoms:**
- "Could not connect to database" errors
- PostgreSQL errors in log

**Solutions:**
1. Check PostgreSQL is running:
   ```bash
   # Windows
   sc query postgresql-x64-14

   # Linux
   sudo systemctl status postgresql
   ```
2. Verify `DATABASE_URL` in `.env`
3. Test database connection:
   ```bash
   python -c "from src.trade_history_sync import TradeHistorySyncService; service = TradeHistorySyncService(); print('DB connected')"
   ```

---

### Issue: High memory usage

**Symptoms:**
- Agent using > 500MB RAM
- System slowdown

**Solutions:**
1. Restart agent (clears cached data)
2. Increase sync interval to reduce frequency
3. Check for memory leaks:
   ```bash
   # Monitor memory usage
   python -m memory_profiler src/services/background_sync_agent.py --once
   ```

---

## Advanced Configuration

### Multiple Accounts

To sync multiple Robinhood accounts, run separate agent instances:

```bash
# Account 1
python -m src.services.background_sync_agent --interval 300 &

# Account 2 (different credentials in .env.account2)
python -m src.services.background_sync_agent --interval 300 --config .env.account2 &
```

### Custom Sync Logic

Extend the agent by subclassing:

```python
from src.services.background_sync_agent import BackgroundSyncAgent

class CustomSyncAgent(BackgroundSyncAgent):
    def _sync_positions(self):
        # Add custom logic here
        results = super()._sync_positions()

        # Your custom code
        # E.g., send Telegram notification, update custom database, etc.

        return results
```

### Integration with Other Services

The agent can trigger webhooks or notifications:

```python
# In _run_sync_cycle():
import requests

# Send webhook on successful sync
if all_success:
    requests.post('https://your-webhook-url.com/sync', json={
        'sync_count': self.sync_count,
        'timestamp': datetime.now().isoformat()
    })
```

---

## Best Practices

### ✅ DO:
- Run agent continuously for best data freshness
- Monitor logs regularly for errors
- Use default 5-minute interval (good balance)
- Set up as system service for 24/7 operation
- Test with `--once` before running continuously

### ❌ DON'T:
- Don't run multiple agents simultaneously (duplicate syncs)
- Don't use intervals < 2 minutes (rate limits)
- Don't ignore errors in logs (investigate promptly)
- Don't run without proper Robinhood credentials
- Don't expose log files publicly (contain sensitive info)

---

## FAQ

**Q: Does this replace the daily sync script?**

A: No, but it makes it less necessary. The agent syncs more frequently (every 5 min vs once daily), but the daily script can still run as a backup.

**Q: Will this use up my Robinhood API quota?**

A: At 5-minute intervals, you'll make ~12 requests/hour (well within limits). Robinhood allows ~60 requests/minute with rate limiting.

**Q: Can I run this on a Raspberry Pi?**

A: Yes! The agent is lightweight and works well on low-power devices. Just ensure PostgreSQL is accessible (can be remote).

**Q: Does this work with Robinhood's new API?**

A: The agent uses `robin_stocks` library which tracks Robinhood's API changes. Update `robin_stocks` regularly:
```bash
pip install --upgrade robin-stocks
```

**Q: Can I pause/resume the agent?**

A: Press Ctrl+C to stop gracefully. Restart with `start_background_sync.bat` to resume.

**Q: How do I know if positions are fresh?**

A: Check `last_sync_time` in logs. If it's recent (< 5 minutes ago), data is fresh. The positions page will show live data without needing manual refresh.

---

## Performance & Resource Usage

### Typical Resource Usage

**Per Sync Cycle:**
- CPU: < 5% for 2-5 seconds
- Memory: ~100-150MB
- Network: ~500KB download, ~50KB upload
- Database: 5-10 queries

**Continuous Operation:**
- CPU: < 1% average (spikes during sync)
- Memory: ~150-200MB
- Disk I/O: Minimal (log writing)

### Optimization Tips

1. **Reduce memory usage:**
   - Clear position cache less frequently
   - Use database more, cache less

2. **Reduce network usage:**
   - Increase sync interval
   - Only sync changed positions (delta sync)

3. **Reduce database load:**
   - Batch database updates
   - Use connection pooling

---

## Support & Feedback

**Log Issues:**
- Check [Issues](https://github.com/yourusername/Magnus/issues) for known problems
- Report new issues with log file attached

**Feature Requests:**
- Request features in [Discussions](https://github.com/yourusername/Magnus/discussions)

**Questions:**
- Ask in [Slack/Discord channel]
- Email: support@magnus-trading.com

---

**Last Updated:** November 21, 2025
**Version:** 1.0.0
**Status:** Production Ready
