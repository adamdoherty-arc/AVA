# Xtrades Watchlists Implementation Plan
## Comprehensive Feature Development Report

Generated: November 2, 2025
Author: Magnus Main Agent

---

## Executive Summary

### Feature Overview
The Xtrades Watchlists feature will enable the Magnus dashboard to automatically scrape trade alerts from professional traders on https://app.xtrades.net, store them in the PostgreSQL database, and send real-time Telegram notifications for new trades and updates.

### Complexity Assessment
- **Overall Complexity: 7/10**
- **Primary Challenges**: Discord OAuth authentication, dynamic content scraping, rate limiting
- **Technical Risk**: Medium-High

### Time Estimate
- **Total Implementation Time: 5-7 days**
- **Testing & Refinement: 2-3 days**
- **Total Project Duration: 7-10 days**

### Key Risks
1. **Discord OAuth Complexity**: The site uses Discord for authentication, which may require browser automation
2. **Anti-Bot Detection**: The site may have protection against automated access
3. **Rate Limiting**: Need to balance sync frequency with site policies
4. **DOM Changes**: Website structure changes could break scraping logic

---

## Technology Recommendations

### Web Scraping Library Selection

#### **Recommended: Selenium (Already in requirements.txt)**
**Justification:**
- Already installed in the project (selenium==4.16.0)
- Handles JavaScript-heavy sites and SPAs effectively
- Can manage Discord OAuth flow through browser automation
- Mature ecosystem with extensive documentation
- Chrome/Firefox WebDriver readily available

**Pros:**
- Handles complex authentication flows
- Can interact with dynamic content
- Screenshots for debugging
- Proven track record with similar projects

**Cons:**
- Resource intensive (requires browser instance)
- Slower than headless alternatives
- Requires WebDriver management

#### Alternative Options (Not Recommended)

**Playwright**
- More modern but would require additional dependency
- Better performance but learning curve for team
- Not currently in project dependencies

**BeautifulSoup + Requests**
- Already in requirements (beautifulsoup4==4.12.2)
- Won't work with Discord OAuth or JavaScript rendering
- Only suitable for static content

### Background Job Implementation

#### **Recommended: Python Script with Windows Task Scheduler**
**Justification:**
- Follows existing pattern (daily_trade_sync.py)
- Simple, reliable, OS-native scheduling
- No additional dependencies required
- Easy monitoring and logs

**Implementation:**
```python
# xtrades_sync_service.py - runs every 5 minutes
# Scheduled via Task Scheduler (Windows) or cron (Linux)
```

### Telegram Bot Library

#### **Recommended: python-telegram-bot**
```bash
pip install python-telegram-bot==20.6
```
**Justification:**
- Most popular and maintained library
- Async support
- Rich feature set
- Extensive documentation

---

## Database Schema Design

### Complete SQL Schema

```sql
-- Xtrades Profiles Table
CREATE TABLE IF NOT EXISTS xtrades_profiles (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    profile_url VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_sync TIMESTAMP,
    total_trades INTEGER DEFAULT 0,
    successful_trades INTEGER DEFAULT 0,
    failed_trades INTEGER DEFAULT 0,
    avg_return DECIMAL(10,2),
    notes TEXT
);

-- Xtrades Trades Table
CREATE TABLE IF NOT EXISTS xtrades_trades (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES xtrades_profiles(id) ON DELETE CASCADE,
    trade_id VARCHAR(100), -- Unique ID from xtrades if available
    ticker VARCHAR(20) NOT NULL,
    strategy VARCHAR(100), -- 'Long', 'Short', 'Options', etc
    trade_type VARCHAR(50), -- 'Stock', 'Option Call', 'Option Put'
    entry_price DECIMAL(10,2),
    exit_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    take_profit DECIMAL(10,2),
    shares INTEGER,
    entry_date TIMESTAMP,
    exit_date TIMESTAMP,
    status VARCHAR(50) DEFAULT 'open', -- 'open', 'closed', 'stopped_out', 'partial'
    pnl_amount DECIMAL(10,2),
    pnl_percentage DECIMAL(10,2),
    alert_text TEXT, -- Original alert message
    notes TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notified BOOLEAN DEFAULT FALSE,
    UNIQUE(profile_id, trade_id)
);

-- Xtrades Sync Log Table
CREATE TABLE IF NOT EXISTS xtrades_sync_log (
    id SERIAL PRIMARY KEY,
    sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sync_type VARCHAR(50), -- 'scheduled', 'manual', 'error_retry'
    profiles_synced INTEGER DEFAULT 0,
    trades_found INTEGER DEFAULT 0,
    new_trades INTEGER DEFAULT 0,
    updated_trades INTEGER DEFAULT 0,
    errors TEXT,
    duration_seconds DECIMAL(10,2),
    status VARCHAR(50) -- 'success', 'partial', 'failed'
);

-- Telegram Notifications Table
CREATE TABLE IF NOT EXISTS telegram_notifications (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER REFERENCES xtrades_trades(id) ON DELETE CASCADE,
    notification_type VARCHAR(50), -- 'new_trade', 'trade_update', 'trade_closed', 'error'
    message TEXT,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    telegram_message_id VARCHAR(100),
    status VARCHAR(50), -- 'sent', 'failed', 'pending'
    error_message TEXT
);

-- Create Indexes for Performance
CREATE INDEX idx_xtrades_trades_profile ON xtrades_trades(profile_id);
CREATE INDEX idx_xtrades_trades_ticker ON xtrades_trades(ticker);
CREATE INDEX idx_xtrades_trades_status ON xtrades_trades(status);
CREATE INDEX idx_xtrades_trades_entry_date ON xtrades_trades(entry_date);
CREATE INDEX idx_xtrades_sync_timestamp ON xtrades_sync_log(sync_timestamp);
CREATE INDEX idx_telegram_sent_at ON telegram_notifications(sent_at);
```

---

## File Structure

```
C:\Code\WheelStrategy\
├── .env (UPDATE with new credentials)
├── requirements.txt (UPDATE with new dependencies)
├── dashboard.py (UPDATE navigation)
├── xtrades_watchlists_page.py (NEW - Streamlit page)
├── xtrades_sync.py (NEW - Standalone sync script)
├── xtrades_sync.bat (NEW - Windows scheduler script)
├── setup_xtrades_scheduler.ps1 (NEW - PowerShell setup)
│
├── src/
│   ├── xtrades_scraper.py (NEW - Web scraping logic)
│   ├── xtrades_sync_service.py (NEW - Sync orchestrator)
│   ├── xtrades_db_manager.py (NEW - Database operations)
│   └── telegram_notifier.py (NEW - Telegram integration)
│
└── features/
    └── xtrades_watchlists/ (NEW)
        ├── README.md
        ├── AGENT.md
        ├── SPEC.md
        ├── ARCHITECTURE.md
        └── tests/
            └── test_xtrades.py
```

---

## Implementation Steps (Detailed)

### Phase 1: Environment Setup (Day 1)

1. **Update .env file**
```bash
# Add to .env
XTRADES_USERNAME=sureadam
XTRADES_PASSWORD=aadam420
TELEGRAM_BOT_TOKEN=  # User to create via @BotFather
TELEGRAM_CHAT_ID=    # User to get from bot
CHROME_DRIVER_PATH=C:/chromedriver/chromedriver.exe
```

2. **Install dependencies**
```bash
pip install python-telegram-bot==20.6
pip install webdriver-manager==4.0.1
```

3. **Create database tables**
```python
# Run setup_xtrades_tables.py
python setup_xtrades_tables.py
```

### Phase 2: Discord OAuth Research (Day 1-2)

1. **Manual Test Flow**
```python
# src/test_xtrades_login.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def test_discord_login():
    driver = webdriver.Chrome()
    driver.get("https://app.xtrades.net")

    # Find and click Discord login button
    discord_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Discord')]"))
    )
    discord_btn.click()

    # Handle Discord OAuth in new window
    driver.switch_to.window(driver.window_handles[-1])

    # Enter credentials
    email_input = driver.find_element(By.NAME, "email")
    email_input.send_keys("sureadam")

    password_input = driver.find_element(By.NAME, "password")
    password_input.send_keys("aadam420")

    # Submit and handle 2FA if needed
    # Save cookies for reuse
```

### Phase 3: Web Scraper Development (Day 2-3)

```python
# src/xtrades_scraper.py
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional

class XtradesScraper:
    """Scrapes trade alerts from xtrades.net profiles"""

    def __init__(self):
        self.driver = None
        self.cookies_file = "xtrades_cookies.pkl"
        self.username = os.getenv('XTRADES_USERNAME')
        self.password = os.getenv('XTRADES_PASSWORD')

    def setup_driver(self):
        """Initialize Chrome driver with options"""
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    def login_with_discord(self) -> bool:
        """Handle Discord OAuth login"""
        try:
            # Load saved cookies if available
            if os.path.exists(self.cookies_file):
                self.load_cookies()
                if self.is_logged_in():
                    return True

            # Perform fresh login
            self.driver.get("https://app.xtrades.net")
            # ... Discord OAuth flow ...

            # Save cookies for future use
            self.save_cookies()
            return True

        except Exception as e:
            print(f"Login failed: {e}")
            return False

    def scrape_profile(self, username: str) -> List[Dict]:
        """Scrape trades from a profile"""
        try:
            url = f"https://app.xtrades.net/profile/{username}"
            self.driver.get(url)

            # Wait for alerts div to load
            alerts_container = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "alerts"))
            )

            trades = []
            alert_elements = alerts_container.find_elements(By.CLASS_NAME, "alert-item")

            for alert in alert_elements:
                trade = self.parse_alert(alert)
                if trade:
                    trades.append(trade)

            return trades

        except TimeoutException:
            print(f"No trades found for {username}")
            return []

    def parse_alert(self, element) -> Optional[Dict]:
        """Parse individual alert element"""
        try:
            # Extract trade details from DOM
            ticker = element.find_element(By.CLASS_NAME, "ticker").text
            strategy = element.find_element(By.CLASS_NAME, "strategy").text
            entry = element.find_element(By.CLASS_NAME, "entry").text

            return {
                'ticker': ticker,
                'strategy': strategy,
                'entry_price': float(entry.replace('$', '')),
                'alert_text': element.text,
                'timestamp': datetime.now()
            }
        except:
            return None
```

### Phase 4: Database Manager (Day 3)

```python
# src/xtrades_db_manager.py
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class XtradesDBManager:
    """Manages Xtrades data in PostgreSQL"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!')
        }

    def get_connection(self):
        return psycopg2.connect(**self.db_config)

    def add_profile(self, username: str) -> int:
        """Add a new profile to track"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO xtrades_profiles (username, profile_url)
                VALUES (%s, %s)
                ON CONFLICT (username) DO UPDATE
                SET last_sync = CURRENT_TIMESTAMP
                RETURNING id
            """, (username, f"https://app.xtrades.net/profile/{username}"))

            profile_id = cur.fetchone()[0]
            conn.commit()
            return profile_id

        finally:
            cur.close()
            conn.close()

    def save_trades(self, profile_id: int, trades: List[Dict]) -> tuple:
        """Save trades and return (new_count, updated_count)"""
        conn = self.get_connection()
        cur = conn.cursor()
        new_trades = []
        updated_trades = []

        try:
            for trade in trades:
                # Check if trade exists
                cur.execute("""
                    SELECT id, status FROM xtrades_trades
                    WHERE profile_id = %s AND ticker = %s
                    AND entry_date::date = %s::date
                """, (profile_id, trade['ticker'], trade.get('entry_date', datetime.now())))

                existing = cur.fetchone()

                if not existing:
                    # Insert new trade
                    cur.execute("""
                        INSERT INTO xtrades_trades
                        (profile_id, ticker, strategy, entry_price, alert_text, status)
                        VALUES (%s, %s, %s, %s, %s, 'open')
                        RETURNING id
                    """, (profile_id, trade['ticker'], trade['strategy'],
                          trade['entry_price'], trade['alert_text']))

                    trade_id = cur.fetchone()[0]
                    new_trades.append(trade_id)

                elif existing[1] == 'open' and trade.get('exit_price'):
                    # Update closed trade
                    cur.execute("""
                        UPDATE xtrades_trades
                        SET exit_price = %s, exit_date = %s, status = 'closed',
                            pnl_amount = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (trade['exit_price'], trade.get('exit_date'),
                          trade.get('pnl_amount'), existing[0]))

                    updated_trades.append(existing[0])

            conn.commit()
            return new_trades, updated_trades

        finally:
            cur.close()
            conn.close()

    def get_active_profiles(self) -> List[Dict]:
        """Get all active profiles for syncing"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM xtrades_profiles
                WHERE is_active = TRUE
                ORDER BY username
            """)
            return cur.fetchall()

        finally:
            cur.close()
            conn.close()
```

### Phase 5: Sync Service (Day 4)

```python
# src/xtrades_sync_service.py
from src.xtrades_scraper import XtradesScraper
from src.xtrades_db_manager import XtradesDBManager
from src.telegram_notifier import TelegramNotifier
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XtradesSyncService:
    """Orchestrates syncing of Xtrades data"""

    def __init__(self):
        self.scraper = XtradesScraper()
        self.db = XtradesDBManager()
        self.notifier = TelegramNotifier()

    def sync_all_profiles(self):
        """Sync all active profiles"""
        start_time = datetime.now()
        sync_stats = {
            'profiles_synced': 0,
            'trades_found': 0,
            'new_trades': 0,
            'updated_trades': 0,
            'errors': []
        }

        try:
            # Setup scraper
            self.scraper.setup_driver()
            if not self.scraper.login_with_discord():
                raise Exception("Failed to login to Xtrades")

            # Get profiles to sync
            profiles = self.db.get_active_profiles()
            logger.info(f"Syncing {len(profiles)} profiles")

            for profile in profiles:
                try:
                    # Scrape profile
                    trades = self.scraper.scrape_profile(profile['username'])
                    sync_stats['trades_found'] += len(trades)

                    # Save to database
                    new, updated = self.db.save_trades(profile['id'], trades)
                    sync_stats['new_trades'] += len(new)
                    sync_stats['updated_trades'] += len(updated)

                    # Send notifications
                    for trade_id in new:
                        self.notifier.notify_new_trade(trade_id)

                    for trade_id in updated:
                        self.notifier.notify_trade_update(trade_id)

                    sync_stats['profiles_synced'] += 1

                    # Rate limiting
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Error syncing {profile['username']}: {e}")
                    sync_stats['errors'].append(str(e))

            # Log sync results
            duration = (datetime.now() - start_time).total_seconds()
            self.db.log_sync(sync_stats, duration)

            logger.info(f"Sync complete: {sync_stats}")

        finally:
            if self.scraper.driver:
                self.scraper.driver.quit()

if __name__ == "__main__":
    service = XtradesSyncService()
    service.sync_all_profiles()
```

### Phase 6: Telegram Notifier (Day 4)

```python
# src/telegram_notifier.py
import os
from telegram import Bot
from telegram.error import TelegramError
import asyncio
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    """Sends notifications via Telegram"""

    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bot = Bot(token=self.bot_token) if self.bot_token else None

    def notify_new_trade(self, trade_id: int):
        """Send notification for new trade"""
        if not self.bot:
            return

        trade = self.get_trade_details(trade_id)

        message = f"""
🚨 NEW TRADE ALERT 🚨

Trader: {trade['profile_username']}
Ticker: ${trade['ticker']}
Strategy: {trade['strategy']}
Entry: ${trade['entry_price']}
Time: {trade['entry_date']}

{trade['alert_text']}
        """

        try:
            asyncio.run(self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            ))
        except TelegramError as e:
            print(f"Telegram error: {e}")
```

### Phase 7: UI Implementation (Day 5)

```python
# xtrades_watchlists_page.py
import streamlit as st
import pandas as pd
from src.xtrades_db_manager import XtradesDBManager
from src.xtrades_sync_service import XtradesSyncService

def render_xtrades_watchlists():
    st.title("🎯 Xtrades Watchlists")
    st.markdown("Track professional traders from xtrades.net")

    db = XtradesDBManager()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "👥 Profiles",
        "📈 Trades",
        "⚙️ Settings"
    ])

    with tab1:
        st.subheader("Trading Activity Overview")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        stats = db.get_overview_stats()

        with col1:
            st.metric("Active Profiles", stats['active_profiles'])
        with col2:
            st.metric("Total Trades", stats['total_trades'])
        with col3:
            st.metric("Open Positions", stats['open_trades'])
        with col4:
            st.metric("Win Rate", f"{stats['win_rate']:.1%}")

        # Recent trades table
        st.subheader("Recent Trades")
        recent_trades = db.get_recent_trades(limit=20)

        if recent_trades:
            df = pd.DataFrame(recent_trades)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No trades yet. Add profiles to start tracking.")

    with tab2:
        st.subheader("Manage Profiles")

        # Add new profile
        with st.form("add_profile"):
            new_username = st.text_input("Profile Username")
            if st.form_submit_button("Add Profile"):
                profile_id = db.add_profile(new_username)
                st.success(f"Added profile: {new_username}")
                st.rerun()

        # List existing profiles
        profiles = db.get_all_profiles()

        for profile in profiles:
            with st.expander(f"{profile['username']} - {profile['total_trades']} trades"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Win Rate", f"{profile['win_rate']:.1%}")
                with col2:
                    st.metric("Avg Return", f"{profile['avg_return']:.1%}")
                with col3:
                    if st.button(f"Remove", key=f"remove_{profile['id']}"):
                        db.deactivate_profile(profile['id'])
                        st.rerun()

    with tab3:
        st.subheader("All Trades")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            profile_filter = st.selectbox(
                "Profile",
                ["All"] + [p['username'] for p in profiles]
            )

        with col2:
            status_filter = st.selectbox(
                "Status",
                ["All", "Open", "Closed", "Stopped Out"]
            )

        with col3:
            date_range = st.date_input(
                "Date Range",
                value=[]
            )

        # Get filtered trades
        trades = db.get_trades(
            profile=profile_filter if profile_filter != "All" else None,
            status=status_filter.lower() if status_filter != "All" else None,
            date_range=date_range if date_range else None
        )

        if trades:
            df = pd.DataFrame(trades)
            st.dataframe(df, use_container_width=True)

            # Export button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export CSV",
                data=csv,
                file_name="xtrades_export.csv",
                mime="text/csv"
            )

    with tab4:
        st.subheader("Settings")

        # Manual sync button
        if st.button("🔄 Sync Now"):
            with st.spinner("Syncing trades..."):
                service = XtradesSyncService()
                service.sync_all_profiles()
                st.success("Sync completed!")
                st.rerun()

        # Telegram setup
        st.subheader("Telegram Notifications")

        with st.form("telegram_settings"):
            bot_token = st.text_input(
                "Bot Token",
                value=os.getenv('TELEGRAM_BOT_TOKEN', ''),
                type="password"
            )
            chat_id = st.text_input(
                "Chat ID",
                value=os.getenv('TELEGRAM_CHAT_ID', '')
            )

            if st.form_submit_button("Save Settings"):
                # Save to .env
                st.success("Settings saved!")

        # Sync status
        st.subheader("Sync History")
        sync_logs = db.get_sync_logs(limit=10)

        if sync_logs:
            df = pd.DataFrame(sync_logs)
            st.dataframe(df, use_container_width=True)
```

### Phase 8: Integration & Testing (Day 5-6)

1. **Update dashboard.py navigation**
```python
# Add after line 117 in dashboard.py
if st.sidebar.button("🎯 Xtrades Watchlists", width='stretch'):
    st.session_state.page = "Xtrades Watchlists"

# Add in page routing section
elif page == "Xtrades Watchlists":
    from xtrades_watchlists_page import render_xtrades_watchlists
    render_xtrades_watchlists()
```

2. **Create Windows Task Scheduler script**
```batch
# xtrades_sync.bat
@echo off
cd /d C:\Code\WheelStrategy
call venv\Scripts\activate
python src\xtrades_sync_service.py
```

3. **Setup PowerShell scheduler**
```powershell
# setup_xtrades_scheduler.ps1
$action = New-ScheduledTaskAction -Execute "C:\Code\WheelStrategy\xtrades_sync.bat"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName "Magnus-Xtrades-Sync" -Action $action -Trigger $trigger -Principal $principal
```

---

## Database Overview Fix

### Root Cause
The `DatabaseScanner` class in `src/database_scanner.py` is connecting to the wrong database name.

**Line 31** currently reads:
```python
'database': os.getenv('DB_NAME', 'wheel_strategy'),
```

Should be:
```python
'database': os.getenv('DB_NAME', 'magnus'),
```

### Fix Implementation
```python
# File: C:\Code\WheelStrategy\src\database_scanner.py
# Line: 31
# Change from: 'wheel_strategy' to 'magnus'
```

This simple one-line fix will restore the Database Scan functionality to show the 1,205 stocks correctly.

---

## Integration Points

### Navigation Button
Add button after line 117 in `dashboard.py`:
```python
if st.sidebar.button("🎯 Xtrades Watchlists", width='stretch'):
    st.session_state.page = "Xtrades Watchlists"
```

### Page Routing
Add routing logic around line 870 in `dashboard.py`:
```python
elif page == "Xtrades Watchlists":
    from xtrades_watchlists_page import render_xtrades_watchlists
    render_xtrades_watchlists()
```

### Environment Variables
Add to `.env`:
```bash
# Xtrades Configuration
XTRADES_USERNAME=sureadam
XTRADES_PASSWORD=aadam420
TELEGRAM_BOT_TOKEN=  # Get from @BotFather
TELEGRAM_CHAT_ID=    # Get from bot
```

---

## Testing Strategy

### 1. Unit Tests
```python
# features/xtrades_watchlists/tests/test_xtrades.py
import pytest
from src.xtrades_db_manager import XtradesDBManager

def test_add_profile():
    db = XtradesDBManager()
    profile_id = db.add_profile("testuser")
    assert profile_id is not None

def test_save_trades():
    # Test trade saving logic
    pass
```

### 2. Integration Tests
- Test Discord OAuth flow with saved cookies
- Test profile scraping with mock data
- Test database operations
- Test Telegram notifications

### 3. Load Testing
- Implement rate limiting (2-3 second delays between requests)
- Test with multiple profiles
- Monitor resource usage

### 4. Error Handling
- Handle website structure changes gracefully
- Implement retry logic for failed requests
- Log all errors to database

---

## Risk Mitigation

### 1. Discord OAuth Complexity
- **Mitigation**: Save and reuse cookies to minimize logins
- **Fallback**: Manual cookie extraction if automation fails

### 2. Anti-Bot Detection
- **Mitigation**:
  - Use undetected-chromedriver
  - Randomize user agents
  - Add human-like delays
  - Rotate IP addresses if needed

### 3. Website Changes
- **Mitigation**:
  - Use multiple selector strategies
  - Implement fallback parsing logic
  - Alert on parsing failures
  - Version control scraping logic

### 4. Rate Limiting
- **Mitigation**:
  - Implement exponential backoff
  - Respect robots.txt
  - Monitor for 429 responses
  - Queue requests appropriately

---

## Success Metrics

1. **Technical Metrics**
   - Sync success rate > 95%
   - Average sync time < 30 seconds per profile
   - Telegram notification delivery > 99%
   - Zero data loss incidents

2. **Business Metrics**
   - Trade capture rate > 98%
   - Notification latency < 1 minute
   - User engagement with alerts
   - Portfolio performance improvement

---

## Maintenance & Operations

### Daily Tasks
- Monitor sync logs for failures
- Check Telegram delivery status
- Review new trades for accuracy

### Weekly Tasks
- Update profile list
- Review error logs
- Performance optimization
- Cookie refresh if needed

### Monthly Tasks
- Analyze trader performance
- Database cleanup
- Update scraping logic if needed
- Security audit

---

## Conclusion

The Xtrades Watchlists feature is a valuable addition to the Magnus platform that will provide users with professional trading insights and real-time notifications. While the Discord OAuth presents some complexity, the overall implementation is achievable within 7-10 days using existing patterns and technologies already present in the codebase.

The Database Scan fix is trivial (one line change) and should be implemented immediately to restore functionality for the existing 1,205 stocks.

## Next Steps

1. Fix Database Scan issue (immediate)
2. Create Telegram bot and get credentials
3. Set up development environment for Xtrades
4. Begin Phase 1 implementation
5. Daily progress reviews and adjustments

---

**Document Version**: 1.0
**Last Updated**: November 2, 2025
**Author**: Magnus Main Agent
**Status**: Ready for Implementation