# Earnings Calendar - Comprehensive Guide

## Overview

The Earnings Calendar is a full-featured Streamlit page for tracking and analyzing company earnings events. It integrates with your database and Robinhood API to provide real-time earnings data, historical analysis, and actionable insights.

## Features

### 1. Calendar View
- **Monthly Calendar Grid** - Visual representation of earnings by date
- **Color-Coded Events** - Quick identification of BMO/AMC earnings
- **Event Count** - See number of earnings per day
- **Click Navigation** - Jump to specific dates

### 2. List View
- **Sortable Table** - Sort by any column (date, symbol, sector, etc.)
- **Color-Coded Rows**:
  - Green: Beat estimates
  - Red: Missed estimates
  - Gray: Pending earnings
  - Yellow: Inline with estimates
- **Export to CSV** - Download filtered data
- **Detailed Metrics** - EPS, Revenue, Surprise %, IV, Price Move

### 3. Historical Analysis
- **Earnings Charts** - Visual history of EPS actual vs estimate
- **Beat/Miss Tracking** - Historical success rate
- **Average Surprise %** - Track consistent outperformers
- **Price Move Analysis** - See typical post-earnings moves

### 4. Analytics Dashboard
- **Total Events** - Count of upcoming earnings
- **Beat Rate** - % of companies that beat estimates
- **Average Surprise** - Mean surprise percentage
- **Beat/Miss Ratio** - Visual comparison
- **Sector Breakdown** - Earnings by industry

### 5. Filters
- **Date Range**:
  - This Week
  - Next Week
  - This Month
  - Next Month
  - Custom Range
- **Time of Day**: BMO (Before Market Open), AMC (After Market Close), All
- **Sector Filter**: Technology, Healthcare, Finance, etc.
- **Market Cap Filter**: Large, Mid, Small cap

## Data Sources

### Database Tables

#### `earnings_events`
Main table for upcoming and past earnings events.

```sql
CREATE TABLE earnings_events (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    earnings_date TIMESTAMP WITH TIME ZONE NOT NULL,
    earnings_time VARCHAR(10),  -- BMO, AMC
    eps_estimate NUMERIC(10, 4),
    eps_actual NUMERIC(10, 4),
    revenue_estimate NUMERIC(20, 2),
    revenue_actual NUMERIC(20, 2),
    surprise_percent NUMERIC(10, 2),
    pre_earnings_iv NUMERIC(10, 4),
    post_earnings_iv NUMERIC(10, 4),
    pre_earnings_price NUMERIC(10, 2),
    post_earnings_price NUMERIC(10, 2),
    price_move_percent NUMERIC(10, 2),
    volume_ratio NUMERIC(10, 2),
    options_volume INTEGER,
    whisper_number NUMERIC(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, earnings_date)
);
```

#### `earnings_history`
Historical earnings data synced from Robinhood API.

```sql
CREATE TABLE earnings_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    quarter INTEGER,
    year INTEGER,
    eps_actual NUMERIC(10, 4),
    eps_estimate NUMERIC(10, 4),
    call_datetime TIMESTAMP WITH TIME ZONE,
    call_replay_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, report_date)
);
```

### Robinhood API

The `get_earnings(symbol)` function provides:
- **Historical Data**: Up to 8 quarters per stock
- **EPS Metrics**: Actual vs Estimate
- **Report Dates**: When earnings were released
- **Call Information**: Earnings call datetime and replay URL
- **Quarter/Year**: Q1-Q4 and fiscal year

## Installation & Setup

### 1. Install Dependencies

```bash
pip install streamlit pandas plotly robin-stocks-py python-dotenv psycopg2-binary
```

### 2. Configure Environment

Add to your `.env` file:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=magnus
DB_USER=postgres
DB_PASSWORD=your_password

# Robinhood (for earnings sync)
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
```

### 3. Initialize Database

The tables will be created automatically on first run. To manually create:

```bash
python -c "from src.earnings_manager import EarningsManager; em = EarningsManager(); print('Tables created')"
```

### 4. Run the Dashboard

#### Option A: Standalone Page
```bash
streamlit run pages/earnings_calendar.py
```

#### Option B: Add to Main Dashboard

In your `dashboard.py`, add navigation:

```python
if st.sidebar.button("📅 Earnings Calendar", use_container_width=True):
    st.session_state.page = "Earnings Calendar"

# In main section
if st.session_state.page == "Earnings Calendar":
    exec(open('pages/earnings_calendar.py').read())
```

## Usage Guide

### Syncing Earnings Data

1. **Navigate to Sidebar** - Look for "Data Sync" section
2. **Click "Sync from Robinhood"** - Fetches latest earnings data
3. **Progress Indicator** - Shows sync progress
4. **Results** - Displays number of stocks synced

The sync:
- Pulls from all active stocks in your `stocks` table
- Fetches last 8 quarters per stock
- Updates both `earnings_events` and `earnings_history` tables
- Handles duplicates automatically

### Viewing Earnings

#### Calendar View
1. Select month using date picker
2. View earnings organized by date
3. Hover over events for details
4. Events show symbol and time (BMO/AMC)

#### List View
1. View all earnings in sortable table
2. Click column headers to sort
3. Color indicates status:
   - **Green Row**: Beat estimates
   - **Red Row**: Missed estimates
   - **Gray Row**: Pending
4. Export filtered data to CSV

#### Historical Analysis
1. Select a symbol from dropdown
2. View chart of EPS actual vs estimate
3. See metrics:
   - Beat rate (% of times beat)
   - Average surprise %
   - Average price move
4. Identify patterns for trading

### Filtering Data

#### Date Range
- **Presets**: This Week, Next Week, This Month, Next Month
- **Custom**: Select specific start and end dates

#### Time Filter
- **BMO**: Before Market Open (typically 7-8 AM ET)
- **AMC**: After Market Close (typically 4-5 PM ET)
- **All**: No time filter

#### Sector Filter
- Select specific sector (Technology, Healthcare, etc.)
- Filter to focus on industry trends

### Analytics

The analytics section shows:
- **Total Events**: Count in selected date range
- **Pending**: Upcoming earnings not yet reported
- **Beat Rate**: Historical success rate
- **Avg Surprise**: Mean surprise percentage
- **Beat/Miss**: Visual ratio of outcomes

## Key Metrics Explained

### EPS (Earnings Per Share)
- **Estimate**: Analyst consensus
- **Actual**: Reported result
- **Surprise %**: (Actual - Estimate) / |Estimate| × 100

### Expected Move
Calculated from options implied volatility:
```
Expected Move = Stock Price × IV
```

This represents the market's expectation for price movement around earnings.

### Volume Ratio
```
Volume Ratio = Post-Earnings Volume / Average Volume
```

Higher ratios indicate increased trading activity.

### IV (Implied Volatility)
- **Pre-Earnings IV**: Before announcement
- **Post-Earnings IV**: After announcement
- **IV Crush**: The drop in IV after earnings

## Trading Strategies

### 1. Earnings Plays
- **High IV**: Sell options before earnings (premium collection)
- **Low IV**: Buy options if expecting big move
- **Expected Move**: Compare to historical moves

### 2. Beat/Miss Patterns
- **Consistent Beaters**: Look for stocks with 75%+ beat rate
- **Surprise Leaders**: Track stocks with high average surprise %
- **Sector Trends**: Identify sectors with strong earnings

### 3. Post-Earnings Trading
- **IV Crush**: Buy stock after earnings when IV drops
- **Price Dislocation**: Look for overreactions
- **Follow Through**: Trade momentum on beats

### 4. Wheel Strategy Integration
- **Before Earnings**: Close short puts to avoid assignment
- **After Earnings**: Sell covered calls on elevated IV
- **Earnings Week**: Avoid opening new positions

## API Integration

### Robinhood API Methods

```python
import robin_stocks.robinhood as rh

# Login
rh.login(username, password)

# Get earnings for a symbol
earnings = rh.get_earnings('AAPL')

# Returns list of dicts:
[
    {
        'symbol': 'AAPL',
        'year': 2024,
        'quarter': 1,
        'eps': {'estimate': 1.43, 'actual': 1.52},
        'report': {'date': '2024-01-25', 'timing': 'pm'},
        'call': {
            'datetime': '2024-01-25T17:00:00Z',
            'replay_url': 'https://...'
        }
    },
    # ... more quarters
]

# Logout
rh.logout()
```

### Using EarningsManager

```python
from src.earnings_manager import EarningsManager

# Initialize
em = EarningsManager()

# Get upcoming earnings
df = em.get_earnings_events(
    start_date=date.today(),
    end_date=date.today() + timedelta(days=7),
    time_filter='AMC',
    sector_filter='Technology'
)

# Sync from Robinhood
result = em.sync_robinhood_earnings(['AAPL', 'MSFT', 'GOOGL'])

# Get historical data
hist_df = em.get_historical_earnings('AAPL', limit=12)

# Get analytics
analytics = em.get_analytics(df)

# Close connection
em.close()
```

## Customization

### Adding Custom Metrics

Edit `src/earnings_manager.py`:

```python
def _calculate_custom_metric(self, row) -> float:
    """Add your custom calculation"""
    # Example: Revenue surprise
    if pd.notna(row['revenue_actual']) and pd.notna(row['revenue_estimate']):
        return ((row['revenue_actual'] - row['revenue_estimate'])
                / row['revenue_estimate'] * 100)
    return 0.0

# Add to get_earnings_events():
if not df.empty:
    df['custom_metric'] = df.apply(self._calculate_custom_metric, axis=1)
```

### Styling

Modify CSS in `pages/earnings_calendar.py`:

```python
st.markdown("""
<style>
    /* Your custom styles */
    .earnings-beat {
        background-color: #your-color;
    }
</style>
""", unsafe_allow_html=True)
```

### Adding Filters

Add new filter in sidebar:

```python
# Market cap filter
market_cap_filter = st.sidebar.selectbox(
    "Market Cap",
    ["All", "Large Cap (>$10B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)"]
)

# Apply in query
if market_cap_filter != "All":
    if "Large" in market_cap_filter:
        query += " AND s.market_cap > 10000000000"
    # ... etc
```

## Troubleshooting

### No Earnings Data

**Issue**: Empty table or "No earnings events found"

**Solutions**:
1. Run sync from Robinhood
2. Check database has stocks in `stocks` table
3. Verify date range includes earnings dates
4. Check filters aren't too restrictive

### Sync Errors

**Issue**: Robinhood sync fails or shows errors

**Solutions**:
1. Verify credentials in `.env`
2. Check Robinhood login (may need 2FA)
3. Reduce number of symbols (API rate limits)
4. Wait and retry (temporary API issues)

### Database Connection

**Issue**: Can't connect to database

**Solutions**:
1. Verify PostgreSQL is running
2. Check DB credentials in `.env`
3. Ensure database `magnus` exists
4. Check firewall/network settings

### Missing Historical Data

**Issue**: Historical chart shows no data

**Solutions**:
1. Sync earnings for that symbol
2. Check symbol exists in `earnings_history` table
3. Verify Robinhood has data for that stock

## Performance Tips

### Database Indexes

Already created, but verify:
```sql
CREATE INDEX idx_earnings_events_symbol ON earnings_events(symbol);
CREATE INDEX idx_earnings_events_date ON earnings_events(earnings_date);
CREATE INDEX idx_earnings_history_symbol ON earnings_history(symbol);
```

### Caching

Add Streamlit caching for expensive operations:

```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_earnings_data(start_date, end_date):
    return manager.get_earnings_events(start_date, end_date)
```

### Batch Sync

For large numbers of stocks:
```python
# Sync in batches of 50
symbols = get_all_symbols()
batch_size = 50

for i in range(0, len(symbols), batch_size):
    batch = symbols[i:i+batch_size]
    manager.sync_robinhood_earnings(batch)
    time.sleep(1)  # Rate limit
```

## Future Enhancements

### Planned Features
- [ ] Earnings alerts/notifications
- [ ] Pre-earnings scanner (high IV opportunities)
- [ ] Post-earnings analysis automation
- [ ] Calendar export (iCal format)
- [ ] Earnings call transcript integration
- [ ] Analyst rating changes tracking
- [ ] Revenue surprise tracking
- [ ] Guidance tracking
- [ ] Institutional ownership changes
- [ ] Options flow tracking around earnings

### API Expansion
- [ ] Polygon.io earnings endpoint
- [ ] Alpha Vantage earnings calendar
- [ ] Yahoo Finance earnings scraper
- [ ] Benzinga earnings API

## Support

### Documentation
- [Streamlit Docs](https://docs.streamlit.io)
- [Robinhood API](https://robin-stocks.readthedocs.io)
- [Plotly Charts](https://plotly.com/python)

### Contact
For issues or feature requests, see main project README.

## License

Part of Magnus Trading Platform - See main LICENSE file.
