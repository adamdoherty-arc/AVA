# Integrating Earnings Calendar into Main Dashboard

## Quick Integration Guide

### Option 1: Add Navigation Button (Recommended)

Edit `dashboard.py` to add the Earnings Calendar page to your navigation:

```python
# Around line 85-105 where navigation buttons are defined

# Add this button with the others
if st.sidebar.button("📅 Earnings Calendar", use_container_width=True):
    st.session_state.page = "Earnings Calendar"
```

Then in the main content area where pages are rendered:

```python
# Around line 800+ where page routing happens

if st.session_state.page == "Earnings Calendar":
    import pages.earnings_calendar as earnings_cal
    earnings_cal.main()
```

### Option 2: Standalone Page

Run the earnings calendar as a separate Streamlit app:

```bash
streamlit run pages/earnings_calendar.py
```

### Option 3: Multi-Page App Structure

If using Streamlit's multi-page feature, the page is already in `pages/` directory and will appear automatically in the sidebar.

## Setup Steps

### 1. Install Dependencies

Ensure you have all required packages:

```bash
pip install streamlit pandas plotly robin-stocks-py python-dotenv psycopg2-binary
```

### 2. Configure Environment

Your `.env` should already have database settings. Add Robinhood credentials if not present:

```env
# Database (already configured)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=magnus
DB_USER=postgres
DB_PASSWORD=postgres123!

# Robinhood API (add if missing)
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
```

### 3. Initialize Database Tables

Run the demo script to create tables and add sample data:

```bash
python sync_earnings_demo.py
```

Or just create tables without demo data:

```bash
python -c "from src.earnings_manager import EarningsManager; EarningsManager(); print('Tables created')"
```

### 4. Verify Setup

Check that tables were created:

```bash
python check_earnings_tables.py
```

(Create this file if needed - see below)

### 5. Test the Page

Run standalone first to test:

```bash
streamlit run pages/earnings_calendar.py
```

### 6. Integrate into Dashboard

Add navigation button in `dashboard.py` as shown in Option 1.

## File Structure

After integration, your structure should be:

```
WheelStrategy/
├── dashboard.py                 # Main dashboard (add button here)
├── pages/
│   └── earnings_calendar.py     # New earnings calendar page
├── src/
│   ├── earnings_manager.py      # New earnings data manager
│   └── tradingview_db_manager.py
├── sync_earnings_demo.py        # Demo data sync script
├── EARNINGS_CALENDAR_README.md  # Full documentation
└── INTEGRATE_EARNINGS_CALENDAR.md  # This file
```

## Code Snippets

### Full Navigation Integration

In `dashboard.py`, add this to your sidebar section:

```python
# Around line 85
st.sidebar.title("⚡ Magnus")

# Navigation as buttons/links
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# Existing buttons
if st.sidebar.button("📈 Dashboard", use_container_width=True):
    st.session_state.page = "Dashboard"
if st.sidebar.button("🎯 Opportunities", use_container_width=True):
    st.session_state.page = "Opportunities"
if st.sidebar.button("💼 Positions", use_container_width=True):
    st.session_state.page = "Positions"
if st.sidebar.button("🔍 Premium Scanner", use_container_width=True):
    st.session_state.page = "Premium Scanner"
if st.sidebar.button("📊 TradingView Watchlists", use_container_width=True):
    st.session_state.page = "TradingView Watchlists"

# ADD THIS NEW BUTTON
if st.sidebar.button("📅 Earnings Calendar", use_container_width=True):
    st.session_state.page = "Earnings Calendar"
```

Then in the page routing section:

```python
# Main content area - around line 800+

if st.session_state.page == "Dashboard":
    # Your dashboard code
    pass

elif st.session_state.page == "Opportunities":
    # Your opportunities code
    pass

# ... other pages ...

# ADD THIS NEW PAGE HANDLER
elif st.session_state.page == "Earnings Calendar":
    # Import and run earnings calendar
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "earnings_calendar",
        "pages/earnings_calendar.py"
    )
    earnings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(earnings_module)
    earnings_module.main()
```

### Alternative: Simpler Import

If you prefer a cleaner approach:

```python
elif st.session_state.page == "Earnings Calendar":
    from pages.earnings_calendar import main as earnings_main
    earnings_main()
```

But this requires modifying `pages/earnings_calendar.py` to ensure `main()` is the entry point.

## Verification Script

Create `check_earnings_tables.py`:

```python
"""Verify earnings calendar tables are set up correctly"""

from src.tradingview_db_manager import TradingViewDBManager
import pandas as pd

print("Checking earnings calendar setup...")

tv = TradingViewDBManager()
conn = tv.get_connection()
cur = conn.cursor()

# Check earnings_events table
print("\n1. earnings_events table:")
cur.execute("SELECT COUNT(*) FROM earnings_events")
count = cur.fetchone()[0]
print(f"   Rows: {count}")

if count > 0:
    cur.execute("SELECT symbol, earnings_date, eps_estimate FROM earnings_events LIMIT 5")
    rows = cur.fetchall()
    print("   Sample data:")
    for row in rows:
        print(f"     {row[0]} - {row[1]} - EPS: ${row[2]}")

# Check earnings_history table
print("\n2. earnings_history table:")
cur.execute("SELECT COUNT(*) FROM earnings_history")
count = cur.fetchone()[0]
print(f"   Rows: {count}")

if count > 0:
    cur.execute("SELECT symbol, report_date, eps_actual FROM earnings_history LIMIT 5")
    rows = cur.fetchall()
    print("   Sample data:")
    for row in rows:
        print(f"     {row[0]} - {row[1]} - EPS: ${row[2]}")

# Check indexes
print("\n3. Indexes:")
cur.execute("""
    SELECT indexname
    FROM pg_indexes
    WHERE tablename IN ('earnings_events', 'earnings_history')
""")
indexes = cur.fetchall()
for idx in indexes:
    print(f"   ✓ {idx[0]}")

cur.close()
conn.close()

print("\n✓ Setup verification complete!")
```

Run it:

```bash
python check_earnings_tables.py
```

## Usage Examples

### Programmatic Access

Use `EarningsManager` in your own scripts:

```python
from src.earnings_manager import EarningsManager
from datetime import date, timedelta

# Initialize
em = EarningsManager()

# Get this week's earnings
this_week = em.get_earnings_events(
    start_date=date.today(),
    end_date=date.today() + timedelta(days=7),
    time_filter='AMC'
)

print(f"Found {len(this_week)} earnings this week after market close")

# Get analytics
analytics = em.get_analytics(this_week)
print(f"Beat rate: {analytics['beat_rate']:.1f}%")

# Close
em.close()
```

### Add to Existing Dashboard Logic

In your `dashboard.py`, you can add earnings insights to other sections:

```python
from src.earnings_manager import EarningsManager

# In your portfolio or opportunities section
em = EarningsManager()

# Check if any portfolio stocks have earnings this week
portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Your holdings
upcoming = em.get_earnings_events(
    start_date=date.today(),
    end_date=date.today() + timedelta(days=7),
    symbols=portfolio_symbols
)

if not upcoming.empty:
    st.warning(f"⚠️ {len(upcoming)} of your holdings have earnings this week!")
    st.dataframe(upcoming[['symbol', 'earnings_date', 'earnings_time']])

em.close()
```

## Troubleshooting

### Import Errors

If you get import errors:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

Add this at the top of `pages/earnings_calendar.py` if needed.

### Database Connection Issues

Check your `.env` file has correct credentials:

```bash
python -c "from src.tradingview_db_manager import TradingViewDBManager; tv = TradingViewDBManager(); print('Connected!')"
```

### Robinhood Authentication

Test Robinhood login separately:

```python
import robin_stocks.robinhood as rh
import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')

try:
    rh.login(username, password)
    print("✓ Robinhood login successful")
    rh.logout()
except Exception as e:
    print(f"✗ Robinhood login failed: {e}")
```

### Page Not Appearing

If using Streamlit multi-page structure and page doesn't show:

1. Ensure file is in `pages/` directory
2. Check filename doesn't start with underscore
3. Verify file has `.py` extension
4. Restart Streamlit server

## Next Steps

After integration:

1. **Populate Data**: Run `sync_earnings_demo.py`
2. **Test Page**: Navigate to Earnings Calendar in dashboard
3. **Sync More Data**: Use sync button to fetch more stocks
4. **Customize**: Adjust filters, styling, metrics per your needs
5. **Automate**: Set up scheduled sync (see below)

## Automated Sync (Optional)

Create a scheduled task to sync earnings daily:

### Windows Task Scheduler

1. Create `sync_earnings_scheduled.bat`:

```batch
@echo off
cd C:\Code\WheelStrategy
call venv\Scripts\activate
python -c "from src.earnings_manager import EarningsManager; em = EarningsManager(); symbols = [...]; em.sync_robinhood_earnings(symbols); em.close()"
```

2. Add to Task Scheduler to run daily at 6 AM

### Linux/Mac Cron

```bash
0 6 * * * cd /path/to/WheelStrategy && source venv/bin/activate && python sync_earnings_scheduled.py
```

## Support

For issues:
1. Check logs in terminal/console
2. Review EARNINGS_CALENDAR_README.md for detailed docs
3. Test components individually (database, Robinhood, Streamlit)
4. Check Streamlit documentation for page navigation issues

## Complete Example

Here's a complete minimal integration:

**dashboard.py** (add to existing code):

```python
# At top with other imports
import sys
import os

# In sidebar navigation section
if st.sidebar.button("📅 Earnings", use_container_width=True):
    st.session_state.page = "Earnings"

# In main content routing
elif st.session_state.page == "Earnings":
    # Load and run earnings page
    exec(open('pages/earnings_calendar.py').read())
```

That's it! The page should now be accessible from your main dashboard.
