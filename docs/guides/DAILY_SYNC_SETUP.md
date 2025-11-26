# Daily Trade Sync Automation Setup

This guide helps you set up automatic daily syncing of trades from Robinhood to the Magnus database.

## Files Created

1. `daily_trade_sync.py` - Python script that performs the sync
2. `daily_trade_sync.bat` - Windows batch file to execute the Python script
3. `logs/` - Directory for sync logs
4. `logs/trade_sync.log` - Log file with sync history

---

## Option 1: Windows Task Scheduler (Recommended)

### Step 1: Open Task Scheduler
1. Press `Win + R`
2. Type `taskschd.msc` and press Enter
3. Task Scheduler window opens

### Step 2: Create New Task
1. In the right panel, click **"Create Task..."** (not "Create Basic Task")
2. A "Create Task" dialog appears

### Step 3: General Tab
- **Name**: `Magnus Daily Trade Sync`
- **Description**: `Syncs trades from Robinhood to Magnus database once per day`
- **Security Options**:
  - Select **"Run whether user is logged on or not"**
  - Check **"Run with highest privileges"**

### Step 4: Triggers Tab
1. Click **"New..."**
2. Configure trigger:
   - **Begin the task**: `On a schedule`
   - **Settings**: `Daily`
   - **Start**: Choose time (recommend 2:00 AM)
   - **Recur every**: `1 days`
   - Check **"Enabled"**
3. Click **OK**

### Step 5: Actions Tab
1. Click **"New..."**
2. Configure action:
   - **Action**: `Start a program`
   - **Program/script**: `C:\Code\WheelStrategy\daily_trade_sync.bat`
   - **Start in (optional)**: `C:\Code\WheelStrategy`
3. Click **OK**

### Step 6: Conditions Tab
- **Power**:
  - Uncheck **"Start the task only if the computer is on AC power"**
  - Check **"Wake the computer to run this task"** (if you want it to wake from sleep)

### Step 7: Settings Tab
- Check **"Allow task to be run on demand"**
- Check **"Run task as soon as possible after a scheduled start is missed"**
- **If the task fails, restart every**: `15 minutes`
- **Attempt to restart up to**: `3 times`

### Step 8: Finish Setup
1. Click **OK**
2. Enter your Windows password if prompted
3. Task is now created!

---

## Option 2: Manual Scheduling (PowerShell)

Run this PowerShell command as Administrator:

```powershell
$Action = New-ScheduledTaskAction -Execute "C:\Code\WheelStrategy\daily_trade_sync.bat"
$Trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun
Register-ScheduledTask -TaskName "Magnus Daily Trade Sync" -Action $Action -Trigger $Trigger -Settings $Settings -Description "Syncs trades from Robinhood to Magnus database"
```

---

## Testing the Setup

### Test 1: Run Manually
1. Double-click `daily_trade_sync.bat`
2. Watch the console output
3. Check `logs/trade_sync.log` for results

### Test 2: Run from Task Scheduler
1. Open Task Scheduler (`Win + R` → `taskschd.msc`)
2. Find "Magnus Daily Trade Sync" in the list
3. Right-click → **Run**
4. Check `logs/trade_sync.log` for results

---

## Viewing Logs

### View Latest Log
```cmd
type C:\Code\WheelStrategy\logs\trade_sync.log
```

### View in Real-Time (tail equivalent)
```powershell
Get-Content C:\Code\WheelStrategy\logs\trade_sync.log -Tail 50 -Wait
```

---

## Troubleshooting

### Issue: Task doesn't run
**Solution**: Check Task Scheduler history
1. Open Task Scheduler
2. Click **"View"** → **"Show All Running Tasks"**
3. Check **"History"** tab for errors

### Issue: Python not found
**Solution**: Use full Python path in batch file
```batch
"C:\Users\YourUsername\AppData\Local\Programs\Python\Python312\python.exe" daily_trade_sync.py
```

### Issue: Database connection fails
**Solution**: Check `.env` file has correct database credentials
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=magnus
DB_USER=postgres
DB_PASSWORD=postgres123!
```

### Issue: Robinhood login fails
**Solution**: Check Robinhood credentials in `.env`
```
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
```

---

## Monitoring Sync Status

### From Positions Page
- The Positions page shows **"Last synced: [time]"** in the Trade History section
- Click **"Sync Now"** button to manually trigger sync anytime

### From Logs
- All sync attempts are logged to `logs/trade_sync.log`
- Each entry shows:
  - Timestamp
  - Number of trades synced
  - Success/failure status
  - Any errors

---

## Customization

### Change Sync Time
Edit the trigger in Task Scheduler:
1. Right-click task → **Properties**
2. Go to **Triggers** tab
3. Edit trigger and change time
4. Click **OK**

### Change Sync Frequency
To sync more often:
1. Edit trigger in Task Scheduler
2. Change from `Daily` to `Weekly` or create multiple triggers

### Disable Auto-Sync
1. Open Task Scheduler
2. Right-click "Magnus Daily Trade Sync"
3. Select **"Disable"**

---

## Uninstalling

To remove the scheduled task:

```powershell
Unregister-ScheduledTask -TaskName "Magnus Daily Trade Sync" -Confirm:$false
```

Or use Task Scheduler:
1. Open Task Scheduler
2. Find "Magnus Daily Trade Sync"
3. Right-click → **Delete**

---

## Summary

✅ **What happens daily**:
1. At 2:00 AM (or your chosen time), Windows runs `daily_trade_sync.bat`
2. Script logs into Robinhood
3. Fetches all closed trades
4. Syncs new trades to Magnus database
5. Logs results to `logs/trade_sync.log`
6. UI automatically shows updated data

✅ **Benefits**:
- No manual syncing needed
- Always see latest closed trades
- Fast page loads (data from database, not API)
- Historical tracking of sync operations

---

## Support

If you encounter issues:
1. Check `logs/trade_sync.log` for error messages
2. Test running `daily_trade_sync.bat` manually
3. Verify database connection and credentials
4. Ensure Robinhood credentials are correct in `.env`
