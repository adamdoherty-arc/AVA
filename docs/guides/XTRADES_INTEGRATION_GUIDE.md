# Xtrades Watchlists Integration Guide

## Overview

The Xtrades Watchlists page monitors option trades from Discord-connected Xtrades.net profiles, providing comprehensive tracking, analytics, and management features.

## Files Created

### 1. `xtrades_watchlists_page.py` (Main UI)
**Location:** Root directory
**Purpose:** Streamlit page with 6 tabs for complete Xtrades monitoring

**Features:**
- **Tab 1: Active Trades** - Show all open trades from monitored profiles
- **Tab 2: Closed Trades** - Historical trades with P/L calculations
- **Tab 3: Performance Analytics** - Stats by profile, strategy, ticker
- **Tab 4: Manage Profiles** - Add, activate, deactivate, sync profiles
- **Tab 5: Sync History** - Audit log of all sync operations
- **Tab 6: Settings** - Configure intervals, notifications, scraper

## Integration Steps

### Step 1: Dashboard Navigation (✅ COMPLETE)

The following changes have been made to `dashboard.py`:

```python
# Navigation button added (line ~113):
if st.sidebar.button("📱 Xtrades Watchlists", width='stretch'):
    st.session_state.page = "Xtrades Watchlists"

# Page routing added (line ~1838):
elif page == "Xtrades Watchlists":
    from xtrades_watchlists_page import show_xtrades_page
    show_xtrades_page()
```

### Step 2: Database Setup

The Xtrades tables should already exist from previous setup:

```sql
-- Tables required (should already exist):
xtrades_profiles        -- Profile management
xtrades_trades          -- Trade data
xtrades_sync_log        -- Sync history
xtrades_notifications   -- Notification tracking
```

To verify tables exist:

```bash
python -c "from src.xtrades_db_manager import XtradesDBManager; db = XtradesDBManager(); print('DB connected')"
```

### Step 3: Environment Variables

Ensure `.env` contains Discord credentials:

```bash
# Discord OAuth for Xtrades.net login
XTRADES_USERNAME=your_discord_email@example.com
XTRADES_PASSWORD=your_discord_password

# Telegram (for notifications)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Step 4: Dependencies

Required packages (should already be installed):

```bash
pip install selenium
pip install undetected-chromedriver
pip install beautifulsoup4
pip install psycopg2-binary
pip install python-dotenv
pip install plotly
```

### Step 5: ChromeDriver Setup

The scraper uses `undetected-chromedriver` which auto-downloads ChromeDriver.

**First run:** May take 30-60 seconds to download driver
**Subsequent runs:** Uses cached driver

## Usage Instructions

### 1. Add Profiles to Monitor

1. Navigate to **Xtrades Watchlists** → **Manage Profiles** tab
2. Enter Xtrades username (e.g., "behappy")
3. Optional: Add display name
4. Click "➕ Add Profile"
5. Profile is now active and ready to sync

### 2. Manual Sync

From **Manage Profiles** tab:
1. Expand profile
2. Click "🔄 Sync Now"
3. Scraper logs in via Discord OAuth
4. Retrieves latest alerts
5. New trades are added to database

### 3. View Active Trades

**Active Trades** tab shows:
- All open positions from monitored profiles
- Filter by: Profile, Strategy, Ticker
- Sort by: Date, Ticker, P/L
- Real-time P/L estimates (color-coded)

### 4. View Closed Trades

**Closed Trades** tab shows:
- Historical closed positions
- Actual P/L calculations
- Filter by: Profile, Strategy, Date Range
- Summary metrics: Total P/L, Win Rate, Avg P/L

### 5. Performance Analytics

**Performance Analytics** tab provides:
- **By Profile:** Total trades, win rate, P/L per trader
- **By Strategy:** CSP, CC, Long Calls/Puts performance
- **By Ticker:** Top winners and losers
- Interactive charts (Plotly)

### 6. Sync History

**Sync History** tab tracks:
- All sync operations (timestamp, duration, status)
- Profiles synced, trades found, new vs updated
- Error messages for failed syncs
- Filter by date range and status

### 7. Settings

**Settings** tab configures:
- Auto-sync interval (5/10/15/30 min)
- Max alerts per sync (default: 50)
- Telegram notification preferences
- Headless browser mode
- Test scraper connection

## Data Flow

```
1. User adds profile → Stored in xtrades_profiles
2. Manual/Auto sync triggered
3. XtradesScraper logs in via Discord OAuth
4. Scrapes profile page for alerts
5. Parses trade data (ticker, strategy, prices, dates)
6. XtradesDBManager stores in xtrades_trades
7. Sync logged in xtrades_sync_log
8. TelegramNotifier sends alerts (optional)
9. UI displays trades in real-time
```

## UI Components Used

Following existing dashboard patterns:

- `st.tabs()` - Navigation between sections
- `st.dataframe()` - Sortable tables with styling
- `st.metric()` - Summary statistics
- `st.selectbox()` / `st.multiselect()` - Filters
- `st.button()` - Actions (sync, add, etc.)
- `st.expander()` - Expandable details
- `st.success()` / `st.error()` - User feedback
- `plotly` charts - Performance visualization

## Color Coding

Following positions_page_improved.py pattern:

- **Green text** - Positive P/L (profit)
- **Red text** - Negative P/L (loss)
- **Bold text** - P/L columns highlighted
- **Emojis** - Status indicators (✅ Active, ❌ Inactive, etc.)

## Error Handling

Comprehensive error handling includes:

- Database connection failures → User-friendly message
- Login failures → Retry with exponential backoff
- Profile not found → Specific error message
- Scraper timeout → Graceful degradation
- Empty data → Helpful guidance messages

## Example Workflow

### First-Time Setup:

1. Navigate to **Xtrades Watchlists**
2. Go to **Manage Profiles** tab
3. Add profile: username="behappy", display="BeHappy Trader"
4. Click "🔄 Sync Now" to test connection
5. Check **Active Trades** to see results
6. Enable auto-sync in **Settings** (optional)

### Daily Monitoring:

1. Check **Active Trades** for current positions
2. Filter by ticker to track specific symbols
3. Review **Performance Analytics** for trends
4. Check **Sync History** for any errors
5. Add/remove profiles as needed

## Telegram Notifications (Optional)

Enable in **Settings** tab:

- ✅ Notify on New Trades
- ✅ Notify on Closed Trades
- ✅ Notify on Large P/L (>$500)

Example notification:
```
🔥 New Trade Alert!
Profile: BeHappy Trader (@behappy)
Ticker: NVDA
Strategy: CSP
Entry: $500.00
Strike: $475.00
Expiration: 2025-12-20
```

## Performance Considerations

- **Initial sync:** May take 60-90 seconds per profile
- **Subsequent syncs:** 30-45 seconds per profile
- **Database queries:** Optimized with indexes
- **UI rendering:** Uses pagination (500 max per view)
- **Scraper:** Headless mode recommended for speed

## Troubleshooting

### Issue: Scraper fails to login
**Solution:** Check Discord credentials in .env, ensure 2FA is disabled on Discord

### Issue: Profile not found
**Solution:** Verify username is correct, check if profile is public

### Issue: No trades showing
**Solution:** Profile may not have recent alerts, try different profile

### Issue: Slow sync
**Solution:** Reduce max_alerts_per_sync in Settings, enable headless mode

### Issue: Database connection error
**Solution:** Verify PostgreSQL is running, check .env credentials

## Testing

### Test Scraper Connection:
1. Go to **Settings** tab
2. Click "🧪 Test Scraper Connection"
3. Should see "✅ Scraper connected successfully!"

### Test Telegram Notifications:
1. Go to **Settings** tab
2. Enable Telegram notifications
3. Click "📤 Send Test Notification"
4. Check Telegram for test message

### Test Profile Sync:
1. Go to **Manage Profiles** tab
2. Select a known active profile
3. Click "🔄 Sync Now"
4. Verify new trades appear in **Active Trades**

## Screenshots/Tab Descriptions

### Tab 1: Active Trades
```
📋 Active Trades (42 shown)
╔══════════╦═══════╦══════════╦══════╦═══════╦════════╦═══════════╗
║ Profile  ║Ticker ║ Strategy ║Strike║ DTE   ║Days Open║Current P/L║
╠══════════╬═══════╬══════════╬══════╬═══════╬════════╬═══════════╣
║BeHappy   ║NVDA   ║CSP       ║$475  ║ 14    ║    3   ║  +$45.00  ║
║ThetaGang ║AAPL   ║CC        ║$190  ║ 21    ║    7   ║  -$12.50  ║
╚══════════╩═══════╩══════════╩══════╩═══════╩════════╩═══════════╝
```

### Tab 2: Closed Trades
```
Summary: Total P/L: $2,450.00 | Win Rate: 68.5% | Avg P/L: $18.75

📋 Closed Trades (127 shown)
╔══════════╦═══════╦══════════╦═══════╦══════╦════════╦═════════╗
║ Profile  ║Ticker ║ Strategy ║ P/L   ║ P/L% ║Duration║CloseDate║
╠══════════╬═══════╬══════════╬═══════╬══════╬════════╬═════════╣
║BeHappy   ║TSLA   ║CSP       ║+$125  ║+25%  ║  12d   ║2025-11-01║
║ThetaGang ║SPY    ║CC        ║-$45   ║-15%  ║   7d   ║2025-10-28║
╚══════════╩═══════╩══════════╩═══════╩══════╩════════╩═════════╝
```

### Tab 3: Performance Analytics
```
👥 Performance by Profile
╔══════════╦═══════╦═══════╦══════════╦═════════╗
║ Profile  ║Trades ║Win Rate║Total P/L ║ Avg P/L ║
╠══════════╬═══════╬═══════╬══════════╬═════════╣
║BeHappy   ║  85   ║ 72.5% ║ $1,850   ║ $21.76  ║
║ThetaGang ║  42   ║ 61.9% ║   $600   ║ $14.29  ║
╚══════════╩═══════╩═══════╩══════════╩═════════╝

[Interactive Charts: P/L by Profile, Strategy Distribution, Monthly Trends]
```

### Tab 4: Manage Profiles
```
➕ Add New Profile
[Username] [Display Name] [Add Profile Button]

📋 Existing Profiles

✅ BeHappy Trader (@behappy)
  ├─ Last Sync: 2025-11-02 14:30
  ├─ Status: success
  ├─ Total Trades: 85
  └─ Actions: [Deactivate] [Sync Now]

✅ Theta Gang (@thetagang)
  ├─ Last Sync: 2025-11-02 14:25
  ├─ Status: success
  ├─ Total Trades: 42
  └─ Actions: [Deactivate] [Sync Now]
```

### Tab 5: Sync History
```
📋 Sync Operations (50 shown)

╔═══════════════╦══════════╦════════╦═════╦════════╦═════════╦════════╗
║  Timestamp    ║Profiles  ║ Found  ║ New ║Updated ║Duration ║ Status ║
╠═══════════════╬══════════╬════════╬═════╬════════╬═════════╬════════╣
║2025-11-02 14:30║    2    ║   15   ║  3  ║   12   ║  45.2s  ║✅success║
║2025-11-02 14:15║    2    ║   12   ║  2  ║   10   ║  42.8s  ║✅success║
║2025-11-02 14:00║    2    ║    8   ║  1  ║    7   ║  38.5s  ║✅success║
╚═══════════════╩══════════╩════════╩═════╩════════╩═════════╩════════╝
```

### Tab 6: Settings
```
🔄 Sync Configuration
Auto-Sync Interval: [15 minutes ▼]
Max Alerts per Profile: [50]

📱 Telegram Notifications
☑ Enable Telegram Notifications
☑ Notify on New Trades
☑ Notify on Closed Trades
☑ Notify on Large P/L (>$500)
[📤 Send Test Notification]

🌐 Discord/Scraper Settings
☑ Run Scraper in Headless Mode
Discord Login Status: Connected
[🧪 Test Scraper Connection]

[💾 Save All Settings]
```

## Next Steps

1. **Test the integration:**
   ```bash
   streamlit run dashboard.py
   ```

2. **Navigate to Xtrades Watchlists** from sidebar

3. **Add your first profile** in Manage Profiles tab

4. **Sync and verify** trades appear correctly

5. **Configure settings** for your preferences

6. **Optional:** Set up auto-sync cron job for background monitoring

## Support

For issues or questions:
- Check error details in expandable sections
- Review sync history for patterns
- Test scraper connection in Settings
- Verify database tables exist
- Check Discord credentials in .env

---

**Created:** 2025-11-02
**Version:** 1.0
**Status:** ✅ Ready for Integration
