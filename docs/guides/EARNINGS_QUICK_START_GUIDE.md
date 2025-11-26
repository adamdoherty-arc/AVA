# Earnings Calendar Enhancement - Quick Start Guide

## 🚀 Get Started in 30 Minutes

This guide gets you implementing the most impactful earnings calendar improvements immediately.

---

## 📋 Prerequisites

```bash
# Install required packages
pip install finance-calendars
pip install schedule
```

---

## ⚡ Quick Win #1: Add Free Calendar API (10 minutes)

### Step 1: Create Sync Script

Create [src/earnings_calendar_sync.py](src/earnings_calendar_sync.py):

```python
"""
Sync earnings calendar from NASDAQ (free, comprehensive)
Run daily to keep calendar updated
"""
from finance_calendars import finance_calendars as fc
from datetime import datetime, timedelta
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def sync_earnings_calendar(days_ahead=30):
    """
    Sync upcoming earnings from NASDAQ public API

    Args:
        days_ahead: Number of days to fetch ahead

    Returns:
        Number of earnings events synced
    """
    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123!')
    )
    cur = conn.cursor()

    synced_count = 0
    today = datetime.now()

    print(f"📅 Syncing earnings calendar for next {days_ahead} days...")

    for day_offset in range(days_ahead):
        target_date = today + timedelta(days=day_offset)

        try:
            # Fetch earnings for this date
            earnings = fc.get_earnings_by_date(target_date)

            if earnings is None or earnings.empty:
                continue

            # Insert each earnings event
            for _, row in earnings.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO earnings_events (
                            symbol,
                            earnings_date,
                            is_confirmed,
                            created_at,
                            updated_at
                        )
                        VALUES (%s, %s, TRUE, NOW(), NOW())
                        ON CONFLICT (symbol, earnings_date)
                        DO UPDATE SET
                            is_confirmed = TRUE,
                            updated_at = NOW()
                    """, (
                        row['symbol'],
                        target_date.date()
                    ))

                    synced_count += 1

                except Exception as e:
                    print(f"⚠️  Error syncing {row['symbol']}: {e}")
                    continue

            conn.commit()
            print(f"✅ Synced {len(earnings)} events for {target_date.strftime('%Y-%m-%d')}")

        except Exception as e:
            print(f"❌ Error fetching date {target_date}: {e}")
            continue

    cur.close()
    conn.close()

    print(f"\n🎉 Total synced: {synced_count} earnings events")
    return synced_count

if __name__ == "__main__":
    sync_earnings_calendar(days_ahead=30)
```

### Step 2: Test It

```bash
python src/earnings_calendar_sync.py
```

**Result:** Your calendar now has comprehensive, reliable data for next 30 days!

---

## ⚡ Quick Win #2: Calculate Expected Move (15 minutes)

### Option A: Using Your Existing Options Data

If you already have options chain data:

```python
"""
Calculate expected move from options prices
Add to src/earnings_expected_move.py
"""
import math
from datetime import datetime, timedelta

def calculate_expected_move_from_options(symbol, earnings_date, options_chain, stock_price):
    """
    Calculate expected move from ATM straddle

    Args:
        symbol: Stock ticker
        earnings_date: Date of earnings
        options_chain: Options chain DataFrame
        stock_price: Current stock price

    Returns:
        Dictionary with expected move metrics
    """
    # Find options expiring AFTER earnings (typically weekly)
    expiration = find_next_expiration_after(earnings_date, options_chain)

    # Find ATM strike
    atm_strike = round_to_nearest_strike(stock_price)

    # Get ATM call and put
    atm_call = options_chain[
        (options_chain['strike'] == atm_strike) &
        (options_chain['type'] == 'call') &
        (options_chain['expiration'] == expiration)
    ]

    atm_put = options_chain[
        (options_chain['strike'] == atm_strike) &
        (options_chain['type'] == 'put') &
        (options_chain['expiration'] == expiration)
    ]

    if atm_call.empty or atm_put.empty:
        return None

    # Calculate straddle price
    call_price = atm_call.iloc[0]['last_price']
    put_price = atm_put.iloc[0]['last_price']
    straddle_price = call_price + put_price

    # Expected move = 85% of straddle
    expected_move_dollars = straddle_price * 0.85
    expected_move_pct = (expected_move_dollars / stock_price) * 100

    # Get IV
    pre_earnings_iv = atm_call.iloc[0]['implied_volatility']

    return {
        'symbol': symbol,
        'earnings_date': earnings_date,
        'expected_move_dollars': round(expected_move_dollars, 2),
        'expected_move_pct': round(expected_move_pct, 2),
        'pre_earnings_iv': round(pre_earnings_iv, 4),
        'straddle_price': round(straddle_price, 2),
        'atm_strike': atm_strike,
        'stock_price': stock_price
    }

def round_to_nearest_strike(price, strike_interval=5):
    """Round price to nearest option strike"""
    if price < 50:
        strike_interval = 2.5
    elif price < 100:
        strike_interval = 5
    else:
        strike_interval = 10

    return round(price / strike_interval) * strike_interval

def find_next_expiration_after(earnings_date, options_chain):
    """Find next expiration after earnings"""
    expirations = sorted(options_chain['expiration'].unique())

    for exp in expirations:
        if exp > earnings_date:
            return exp

    return None
```

### Option B: Using Yahoo Finance (Free Alternative)

```python
"""
Get expected move from Yahoo Finance implied volatility
Add this to src/earnings_expected_move.py
"""
import yfinance as yf
import math

def calculate_expected_move_from_yf(symbol, earnings_date):
    """
    Calculate expected move using Yahoo Finance IV

    Free alternative when you don't have options chain
    """
    try:
        # Get stock data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        stock_price = info.get('currentPrice') or info.get('regularMarketPrice')

        # Get options chain for next expiration after earnings
        expirations = ticker.options

        # Find expiration after earnings
        expiration = None
        for exp in expirations:
            if datetime.strptime(exp, '%Y-%m-%d').date() > earnings_date:
                expiration = exp
                break

        if not expiration:
            return None

        # Get options chain
        opt_chain = ticker.option_chain(expiration)

        # Find ATM options
        atm_strike = round_to_nearest_strike(stock_price)

        # Get ATM call and put
        atm_call = opt_chain.calls[opt_chain.calls['strike'] == atm_strike]
        atm_put = opt_chain.puts[opt_chain.puts['strike'] == atm_strike]

        if atm_call.empty or atm_put.empty:
            return None

        # Calculate expected move
        call_price = atm_call.iloc[0]['lastPrice']
        put_price = atm_put.iloc[0]['lastPrice']
        straddle_price = call_price + put_price
        expected_move_dollars = straddle_price * 0.85
        expected_move_pct = (expected_move_dollars / stock_price) * 100

        # Get IV
        iv = atm_call.iloc[0]['impliedVolatility']

        return {
            'symbol': symbol,
            'earnings_date': earnings_date,
            'expected_move_dollars': round(expected_move_dollars, 2),
            'expected_move_pct': round(expected_move_pct, 2),
            'pre_earnings_iv': round(iv, 4),
            'stock_price': stock_price
        }

    except Exception as e:
        print(f"Error calculating expected move for {symbol}: {e}")
        return None

# Install if needed: pip install yfinance
```

### Step 3: Store Expected Move Before Earnings

```python
"""
Run this 1-2 days before earnings
Add to src/earnings_pre_earnings_collector.py
"""
from src.earnings_expected_move import calculate_expected_move_from_yf
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def collect_pre_earnings_data():
    """
    Collect expected move for earnings in next 2 days

    Schedule this to run daily at market close
    """
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123!')
    )
    cur = conn.cursor()

    # Get earnings in next 2 days without expected move data
    cur.execute("""
        SELECT symbol, earnings_date, id
        FROM earnings_events
        WHERE earnings_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '2 days'
          AND pre_earnings_iv IS NULL
        ORDER BY earnings_date
    """)

    earnings = cur.fetchall()

    print(f"📊 Collecting expected move for {len(earnings)} upcoming earnings...")

    for symbol, earnings_date, event_id in earnings:
        print(f"Processing {symbol}...")

        # Calculate expected move
        result = calculate_expected_move_from_yf(symbol, earnings_date)

        if result:
            # Update database
            cur.execute("""
                UPDATE earnings_events
                SET
                    pre_earnings_iv = %s,
                    expected_move_dollars = %s,
                    expected_move_pct = %s,
                    pre_earnings_price = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                result['pre_earnings_iv'],
                result['expected_move_dollars'],
                result['expected_move_pct'],
                result['stock_price'],
                event_id
            ))

            conn.commit()
            print(f"✅ {symbol}: Expected move ±{result['expected_move_pct']:.2f}%")

        else:
            print(f"⚠️  {symbol}: Could not calculate expected move")

    cur.close()
    conn.close()

    print("\n🎉 Pre-earnings data collection complete!")

if __name__ == "__main__":
    collect_pre_earnings_data()
```

**Schedule this to run daily:**

```python
# Add to your scheduler or crontab
# Daily at 3:30 PM ET (after market close)
import schedule
import time

schedule.every().day.at("15:30").do(collect_pre_earnings_data)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## ⚡ Quick Win #3: Track Post-Earnings Results (15 minutes)

```python
"""
Collect actual price movement after earnings
Add to src/earnings_post_earnings_collector.py
"""
import yfinance as yf
from datetime import datetime, timedelta
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def collect_post_earnings_data():
    """
    Collect actual price movement 1 day after earnings

    Run this daily to capture post-earnings metrics
    """
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123!')
    )
    cur = conn.cursor()

    # Get earnings from yesterday that need post-data
    cur.execute("""
        SELECT symbol, earnings_date, pre_earnings_price, expected_move_pct, id
        FROM earnings_events
        WHERE earnings_date = CURRENT_DATE - INTERVAL '1 day'
          AND post_earnings_price IS NULL
          AND pre_earnings_price IS NOT NULL
    """)

    earnings = cur.fetchall()

    print(f"📈 Collecting post-earnings data for {len(earnings)} events...")

    for symbol, earnings_date, pre_price, expected_move, event_id in earnings:
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)

            # Get price 1 day after earnings
            post_date = earnings_date + timedelta(days=1)

            # Fetch historical data
            hist = ticker.history(start=post_date, end=post_date + timedelta(days=2))

            if hist.empty:
                print(f"⚠️  {symbol}: No price data available yet")
                continue

            post_price = hist.iloc[0]['Close']

            # Calculate actual move
            price_move_dollars = post_price - pre_price
            price_move_pct = (price_move_dollars / pre_price) * 100

            # Did it exceed expected move?
            exceeded_expected = abs(price_move_pct) > expected_move if expected_move else False

            # Get volume data
            post_volume = hist.iloc[0]['Volume']

            # Get average volume (20 days before earnings)
            hist_volume = ticker.history(
                start=earnings_date - timedelta(days=30),
                end=earnings_date
            )
            avg_volume = hist_volume['Volume'].mean()
            volume_ratio = post_volume / avg_volume if avg_volume > 0 else None

            # Update database
            cur.execute("""
                UPDATE earnings_events
                SET
                    post_earnings_price = %s,
                    price_move_dollars = %s,
                    price_move_percent = %s,
                    exceeded_expected_move = %s,
                    volume_ratio = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                post_price,
                price_move_dollars,
                price_move_pct,
                exceeded_expected,
                volume_ratio,
                event_id
            ))

            conn.commit()

            print(f"✅ {symbol}: {price_move_pct:+.2f}% "
                  f"(expected ±{expected_move:.2f}%, "
                  f"{'EXCEEDED' if exceeded_expected else 'within'})")

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            continue

    cur.close()
    conn.close()

    print("\n🎉 Post-earnings data collection complete!")

if __name__ == "__main__":
    collect_post_earnings_data()
```

---

## ⚡ Quick Win #4: Historical Pattern Analysis (10 minutes)

```python
"""
Analyze historical earnings patterns for each stock
Add to src/earnings_pattern_analyzer.py
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def calculate_earnings_patterns(symbol):
    """
    Calculate beat rate, consistency, and quality metrics

    Returns:
        Dictionary with pattern analysis
    """
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123!')
    )
    cur = conn.cursor()

    # Get last 12 quarters of earnings
    cur.execute("""
        SELECT
            eps_actual,
            eps_estimate,
            eps_surprise_percent,
            beat_miss,
            price_move_percent,
            expected_move_pct,
            ABS(price_move_percent) > expected_move_pct as exceeded_expected
        FROM earnings_history
        WHERE symbol = %s
          AND eps_actual IS NOT NULL
          AND eps_estimate IS NOT NULL
        ORDER BY report_date DESC
        LIMIT 12
    """, (symbol,))

    history = cur.fetchall()

    if not history:
        return None

    # Calculate metrics
    total = len(history)
    beats = sum(1 for h in history if h[3] == 'beat')  # beat_miss column
    misses = sum(1 for h in history if h[3] == 'miss')
    meets = sum(1 for h in history if h[3] == 'meet')

    beat_rate = (beats / total) * 100

    # Average surprise
    surprises = [h[2] for h in history if h[2] is not None]  # eps_surprise_percent
    avg_surprise = sum(surprises) / len(surprises) if surprises else 0

    # Consistency (standard deviation)
    import statistics
    surprise_std = statistics.stdev(surprises) if len(surprises) > 1 else 0

    # Exceed expected move rate
    exceeded_count = sum(1 for h in history if h[6] and h[5] is not None)  # exceeded_expected
    valid_count = sum(1 for h in history if h[5] is not None)  # has expected_move_pct
    exceed_rate = (exceeded_count / valid_count * 100) if valid_count > 0 else 0

    # Calculate quality score (0-100)
    quality_score = calculate_quality_score(beat_rate, avg_surprise, surprise_std)

    result = {
        'symbol': symbol,
        'quarters_analyzed': total,
        'beat_rate': round(beat_rate, 2),
        'miss_rate': round((misses / total) * 100, 2),
        'meet_rate': round((meets / total) * 100, 2),
        'avg_surprise_pct': round(avg_surprise, 2),
        'surprise_std': round(surprise_std, 2),
        'exceed_expected_move_rate': round(exceed_rate, 2),
        'quality_score': round(quality_score, 2)
    }

    cur.close()
    conn.close()

    return result

def calculate_quality_score(beat_rate, avg_surprise, consistency):
    """
    Calculate 0-100 quality score

    High score = consistent beater, good trading opportunity
    """
    score = 0

    # Beat rate component (0-40 points)
    score += (beat_rate / 100) * 40

    # Positive surprise component (0-30 points)
    score += min(abs(avg_surprise), 10) * 3

    # Consistency component (0-30 points)
    # Lower std dev = better
    consistency_score = max(0, 30 - consistency)
    score += consistency_score

    return min(100, score)

def update_all_patterns():
    """
    Calculate patterns for all stocks with earnings history
    """
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'magnus'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123!')
    )
    cur = conn.cursor()

    # Get all symbols with earnings history
    cur.execute("""
        SELECT DISTINCT symbol
        FROM earnings_history
        WHERE eps_actual IS NOT NULL
    """)

    symbols = [row[0] for row in cur.fetchall()]

    print(f"📊 Calculating patterns for {len(symbols)} symbols...")

    for symbol in symbols:
        patterns = calculate_earnings_patterns(symbol)

        if patterns:
            # Store in pattern analysis table
            cur.execute("""
                INSERT INTO earnings_pattern_analysis (
                    symbol, beat_rate_8q, avg_surprise_pct_8q,
                    surprise_std_8q, exceed_expected_move_rate,
                    quality_score, last_updated
                )
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (symbol)
                DO UPDATE SET
                    beat_rate_8q = EXCLUDED.beat_rate_8q,
                    avg_surprise_pct_8q = EXCLUDED.avg_surprise_pct_8q,
                    surprise_std_8q = EXCLUDED.surprise_std_8q,
                    exceed_expected_move_rate = EXCLUDED.exceed_expected_move_rate,
                    quality_score = EXCLUDED.quality_score,
                    last_updated = NOW()
            """, (
                symbol,
                patterns['beat_rate'],
                patterns['avg_surprise_pct'],
                patterns['surprise_std'],
                patterns['exceed_expected_move_rate'],
                patterns['quality_score']
            ))

            conn.commit()

            print(f"✅ {symbol}: Beat Rate {patterns['beat_rate']:.1f}%, "
                  f"Quality Score {patterns['quality_score']:.0f}/100")

    cur.close()
    conn.close()

    print("\n🎉 Pattern analysis complete!")

if __name__ == "__main__":
    update_all_patterns()
```

---

## 🎯 Putting It All Together: Daily Automation

Create [scripts/daily_earnings_automation.py](scripts/daily_earnings_automation.py):

```python
"""
Daily automation for earnings calendar system
Run this via cron or Windows Task Scheduler

Schedule: Daily at 4:00 PM ET (after market close)
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.earnings_calendar_sync import sync_earnings_calendar
from src.earnings_pre_earnings_collector import collect_pre_earnings_data
from src.earnings_post_earnings_collector import collect_post_earnings_data
from src.earnings_pattern_analyzer import update_all_patterns

def main():
    """
    Run all daily earnings tasks
    """
    print("=" * 80)
    print("DAILY EARNINGS AUTOMATION")
    print("=" * 80)
    print()

    # 1. Sync calendar (get upcoming earnings)
    print("STEP 1: Syncing earnings calendar...")
    print("-" * 80)
    sync_earnings_calendar(days_ahead=30)
    print()

    # 2. Collect pre-earnings data (expected move)
    print("STEP 2: Collecting pre-earnings data...")
    print("-" * 80)
    collect_pre_earnings_data()
    print()

    # 3. Collect post-earnings data (actual results)
    print("STEP 3: Collecting post-earnings results...")
    print("-" * 80)
    collect_post_earnings_data()
    print()

    # 4. Update pattern analysis (weekly - check day of week)
    from datetime import datetime
    if datetime.now().weekday() == 5:  # Saturday
        print("STEP 4: Updating pattern analysis (weekly)...")
        print("-" * 80)
        update_all_patterns()
        print()

    print("=" * 80)
    print("✅ ALL TASKS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

### Windows Task Scheduler Setup

Create [scripts/schedule_daily_earnings.bat](scripts/schedule_daily_earnings.bat):

```batch
@echo off
cd /d "C:\code\Magnus"
call venv\Scripts\activate
python scripts/daily_earnings_automation.py >> logs\earnings_automation.log 2>&1
```

**Schedule in Task Scheduler:**
- Trigger: Daily at 4:00 PM
- Action: Run `schedule_daily_earnings.bat`
- Start in: `C:\code\Magnus`

---

## 📊 Dashboard Integration

Add to your existing [earnings_calendar_page.py](earnings_calendar_page.py):

```python
# At the top of display_earnings_table()

def display_earnings_table():
    """Enhanced earnings table with patterns"""

    # ... existing filter code ...

    # Add pattern metrics
    st.markdown("### 📊 Top Quality Earnings Opportunities")

    # Get upcoming earnings with patterns
    cur.execute("""
        SELECT
            e.symbol,
            e.earnings_date,
            e.expected_move_pct,
            p.beat_rate_8q,
            p.avg_surprise_pct_8q,
            p.quality_score,
            s.company_name
        FROM earnings_events e
        LEFT JOIN earnings_pattern_analysis p ON e.symbol = p.symbol
        LEFT JOIN stocks s ON e.symbol = s.symbol
        WHERE e.earnings_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
          AND p.quality_score > 70
        ORDER BY p.quality_score DESC, e.earnings_date
        LIMIT 10
    """)

    opportunities = cur.fetchall()

    if opportunities:
        df_opp = pd.DataFrame(opportunities, columns=[
            'Symbol', 'Date', 'Expected Move %', 'Beat Rate %',
            'Avg Surprise %', 'Quality Score', 'Company'
        ])

        st.dataframe(
            df_opp,
            hide_index=True,
            column_config={
                "Quality Score": st.column_config.ProgressColumn(
                    "Quality",
                    min_value=0,
                    max_value=100,
                    format="%d"
                )
            }
        )
    else:
        st.info("No high-quality opportunities in next 7 days")

    st.markdown("---")

    # ... rest of your existing code ...
```

---

## 🎉 Success Checklist

After implementing these quick wins, you should have:

- ✅ **Comprehensive calendar** from free NASDAQ API
- ✅ **Expected move calculations** stored before each earnings
- ✅ **Actual move tracking** after each earnings
- ✅ **Historical pattern analysis** (beat rates, quality scores)
- ✅ **Daily automation** to keep everything updated
- ✅ **Enhanced dashboard** showing high-quality opportunities

---

## 🚀 Next Steps

Once you have these basics running smoothly:

1. **Add Whisper Numbers** (if budget allows)
   - Subscribe to EarningsWhispers.com or Estimize
   - Store in `whisper_number` field

2. **Build ML Predictor**
   - See comprehensive guide for full implementation
   - Train on your growing dataset

3. **Add Telegram Alerts**
   - Notify on high-quality opportunities
   - Alert when earnings dates change

4. **Track IV Time Series**
   - Monitor IV expansion before earnings
   - Identify unusual patterns

---

## 📚 Additional Resources

- **Full Research Report:** [docs/EARNINGS_CALENDAR_RESEARCH_COMPREHENSIVE.md](docs/EARNINGS_CALENDAR_RESEARCH_COMPREHENSIVE.md)
- **Database Schema:** [database_schema.sql](database_schema.sql)
- **Existing Manager:** [src/earnings_manager.py](src/earnings_manager.py)

---

## 💡 Pro Tips

1. **Start small:** Run the calendar sync manually first, then automate
2. **Test with paper trading:** Validate patterns before risking capital
3. **Focus on quality:** 10 high-quality earnings plays > 100 random ones
4. **Track performance:** Log every trade based on earnings data
5. **Iterate:** Your ML models will improve as you collect more data

---

**Estimated Time to Implement:** 1-2 hours
**Estimated Value:** Potentially thousands in improved trading decisions
**Maintenance:** 5 minutes daily (automated)

Happy Trading! 📈
