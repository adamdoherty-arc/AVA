# TradingView Watchlist Sync - Complete Guide

## Overview

Your TradingView watchlists are now automatically synced to PostgreSQL database with **automatic session refresh** when authentication expires.

## Current Status

✅ **Fully Operational**
- Session ID: `a1xrjh6rt0ovi42np45b7lz4ik5m3et2`
- 8 watchlists synced
- 281 total symbols
- Auto-refresh enabled

## Your Watchlists

| Watchlist | Symbols | Description |
|-----------|---------|-------------|
| **NVDA** | 152 | AI/Tech stocks (NVDA, SMCI, ARM, AMZN, TSLA) |
| **MAIN** | 99 | Crypto (BTC, ETH, SOL, DOT, etc.) |
| **Investment** | 19 | Crypto holdings (ADA, BNB, ETH, etc.) |
| **Track** | 5 | Crypto tracking (SOL, BTC, GMT, ETH, APE) |
| **Stocks** | 3 | US Stocks (COIN, AAPL, URA) |
| **Red List** | 1 | Colored list (ABVE) |
| **Green List** | 1 | Colored list (RNDRUSDT) |
| **Purple List** | 1 | Colored list (FTMUSDT) |

## How It Works

### Automatic Session Management

The system automatically handles session expiration:

```
┌─────────────────────────────────────────────────────────┐
│  Your Code: Run sync                                    │
│  python src/tradingview_api_sync.py                     │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Check Session ID                                       │
│  • If valid → Sync watchlists ✅                        │
│  • If expired → Auto-refresh ⚡                         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼ (if expired)
┌─────────────────────────────────────────────────────────┐
│  Auto-Refresh Process                                   │
│  1. Detect auth failure (401/403)                       │
│  2. Launch browser automatically                        │
│  3. Fill username/password                              │
│  4. Wait for YOU to complete 2FA                        │
│  5. Extract new session cookie                          │
│  6. Save to .env file                                   │
│  7. Retry sync with new session ✅                      │
└─────────────────────────────────────────────────────────┘
```

## Usage

### Daily Sync (Recommended)

```bash
python src/tradingview_api_sync.py
```

This will:
- Sync all your watchlists
- Update symbol prices
- Handle expired sessions automatically

### Manual Session Refresh

If you want to manually get a fresh session ID:

```bash
python src/get_session_interactive.py
```

### Query Watchlists from Database

```python
from src.tradingview_api_sync import TradingViewAPISync

syncer = TradingViewAPISync()

# Get all watchlists
watchlists = syncer.sync_to_database()

# Find which watchlists contain NVDA
nvda_lists = syncer.find_watchlist_with_symbol('NVDA')
print(f"NVDA found in: {nvda_lists}")
```

### Get Symbols for Analysis

```python
from src.tradingview_db_manager import TradingViewDBManager

manager = TradingViewDBManager()

# Get all symbols from NVDA watchlist
nvda_symbols = manager.get_watchlist_symbols('NVDA')
print(f"Found {len(nvda_symbols)} symbols in NVDA watchlist")

# Get detailed info including prices
details = manager.get_watchlist_details('NVDA')
for stock in details:
    print(f"{stock['symbol']}: ${stock['last_price']}")
```

## Database Schema

### Tables

**tv_watchlists_api** - Watchlist metadata
- `watchlist_id` (varchar) - TradingView ID
- `name` (varchar) - Watchlist name
- `symbols` (text[]) - Array of symbols
- `symbol_count` (integer)
- `last_synced` (timestamp)

**tv_symbols_api** - Individual symbols
- `watchlist_id` (varchar) - FK to watchlists
- `symbol` (varchar) - Stock symbol
- `exchange` (varchar) - Exchange name
- `full_symbol` (varchar) - Full symbol with exchange

### SQL Queries

```sql
-- Get all watchlists
SELECT name, symbol_count, last_synced
FROM tv_watchlists_api
ORDER BY symbol_count DESC;

-- Get NVDA watchlist symbols
SELECT symbol
FROM tv_symbols_api
WHERE watchlist_id = '44325911'
ORDER BY symbol;

-- Find watchlists containing AAPL
SELECT DISTINCT w.name
FROM tv_watchlists_api w
JOIN tv_symbols_api s ON w.watchlist_id = s.watchlist_id
WHERE s.symbol = 'AAPL';
```

## Automation

### Scheduled Sync (Windows Task Scheduler)

Create a batch file `sync_tradingview.bat`:

```batch
@echo off
cd C:\Code\WheelStrategy
python src/tradingview_api_sync.py >> logs/tradingview_sync.log 2>&1
```

Schedule it to run daily at 9 AM:
```bash
schtasks /create /tn "TradingView Sync" /tr "C:\Code\WheelStrategy\sync_tradingview.bat" /sc daily /st 09:00
```

### Python Scheduler

```python
import schedule
import time
from src.tradingview_api_sync import TradingViewAPISync

def sync_job():
    print("Starting TradingView sync...")
    syncer = TradingViewAPISync()
    watchlists = syncer.sync_to_database()
    print(f"Synced {len(watchlists)} watchlists")

# Run every day at 9 AM
schedule.every().day.at("09:00").do(sync_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Session ID Expired

**Symptom**: Sync returns empty watchlists or 401/403 errors

**Solution**: Automatic! Just run:
```bash
python src/tradingview_api_sync.py
```

The system will:
1. Detect the expired session
2. Open a browser window
3. Wait for you to complete 2FA
4. Save the new session ID
5. Continue syncing

### No Watchlists Found

**Check**:
1. Session ID is in .env
2. TradingView account has watchlists
3. Credentials are correct

**Test**:
```bash
python test_auto_session_refresh.py
```

### Database Connection Error

**Check**:
1. PostgreSQL is running
2. Database "magnus" exists
3. Credentials in .env are correct

```bash
psql -h localhost -U postgres -d magnus -c "SELECT 1"
```

## Files

### Core Scripts

- `src/tradingview_api_sync.py` - Main sync script with auto-refresh
- `src/get_session_interactive.py` - Interactive session ID retrieval
- `src/tradingview_db_manager.py` - Database operations layer

### Configuration

- `.env` - Contains session ID and credentials
- `GET_TRADINGVIEW_SESSION.md` - Manual session retrieval guide

### Testing

- `test_auto_session_refresh.py` - Test automatic refresh
- `test_tradingview.py` - General integration tests

## Security

### Session ID Security

⚠️ Your session ID is like a password. Keep it secure:

- ✅ Never commit `.env` to git
- ✅ Don't share your session ID
- ✅ Regenerate if compromised
- ✅ Session expires automatically after weeks

### Best Practices

```bash
# Add .env to .gitignore
echo ".env" >> .gitignore

# Keep backup of working session
cp .env .env.backup
```

## Integration with Wheel Strategy

### Get Symbols for Premium Scanning

```python
from src.tradingview_db_manager import TradingViewDBManager
from src.premium_scanner import PremiumScanner

# Get NVDA watchlist symbols
manager = TradingViewDBManager()
symbols = manager.get_watchlist_symbols('NVDA')

# Scan for best option premiums
scanner = PremiumScanner()
opportunities = scanner.scan_premiums(
    symbols=symbols,
    max_price=50,
    min_premium_pct=1.0,
    dte=30
)

# Show top 10 opportunities
for opp in opportunities[:10]:
    print(f"{opp['symbol']}: {opp['premium_pct']:.2f}% premium")
```

### Filter by Price

```python
# Get symbols under $50 from watchlist
import yfinance as yf

symbols = manager.get_watchlist_symbols('NVDA')
affordable = []

for symbol in symbols:
    ticker = yf.Ticker(symbol)
    price = ticker.info.get('regularMarketPrice', 0)
    if 0 < price <= 50:
        affordable.append(symbol)

print(f"Found {len(affordable)} stocks under $50")
```

## FAQ

**Q: How long does the session ID last?**
A: Typically 2-4 weeks, but it varies. The system auto-refreshes when it expires.

**Q: Do I need to manually update the session ID?**
A: No! The system automatically handles it. Just complete 2FA when the browser opens.

**Q: Can I run this on a server without a display?**
A: Session retrieval needs a browser (for 2FA), but you can:
1. Get session ID on your local machine
2. Copy it to the server's .env file
3. Sync runs headless after that

**Q: What happens if I have 2FA enabled?**
A: Perfect! The interactive script handles 2FA. It opens a browser, fills credentials, and waits for you to complete 2FA.

**Q: Can I sync only specific watchlists?**
A: Currently it syncs all. To filter, query the database after sync.

## Support

If you encounter issues:

1. Check logs: `logs/tradingview_sync.log`
2. Test sync: `python test_auto_session_refresh.py`
3. Verify session: Session ID should be 32 characters
4. Check database: Ensure PostgreSQL is running

## Summary

🎉 **You're all set!**

- ✅ Session ID configured and working
- ✅ 8 watchlists synced (281 symbols)
- ✅ Automatic session refresh enabled
- ✅ NVDA watchlist with 152 stocks ready
- ✅ Database integration complete

**Just run**: `python src/tradingview_api_sync.py` whenever you want to sync!
