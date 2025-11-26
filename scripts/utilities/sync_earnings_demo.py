"""
Earnings Sync Demo - Populate earnings calendar with sample data

This script:
1. Syncs earnings from Robinhood for popular stocks
2. Adds sample upcoming earnings events
3. Demonstrates EarningsManager usage
"""

import sys
import os
from datetime import datetime, timedelta, date
from src.earnings_manager import EarningsManager
import random

print("=" * 80)
print("EARNINGS CALENDAR - DEMO DATA SYNC")
print("=" * 80)

# Initialize manager
print("\n1. Initializing Earnings Manager...")
manager = EarningsManager()
print("✓ Connected to database")
print("✓ Tables verified/created")

# Popular stocks to sync
POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'TSLA', 'META', 'AMD', 'NFLX', 'INTC',
    'BABA', 'DIS', 'PYPL', 'SHOP', 'UBER',
    'SQ', 'COIN', 'ROKU', 'SNAP', 'PINS'
]

print(f"\n2. Syncing historical earnings from Robinhood for {len(POPULAR_STOCKS)} stocks...")
print("   This may take 1-2 minutes...")

def progress_callback(current, total, symbol):
    """Progress indicator"""
    pct = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r   [{bar}] {pct:.1f}% - {symbol:<6}', end='', flush=True)

result = manager.sync_robinhood_earnings(POPULAR_STOCKS, progress_callback)

print("\n")

if result['success']:
    print(f"✓ Sync completed successfully")
    print(f"  - Synced: {result['synced']} stocks")
    print(f"  - Errors: {result['errors']} stocks")

    if result['errors'] > 0 and result.get('error_symbols'):
        print(f"  - Failed symbols: {', '.join(result['error_symbols'][:5])}")
else:
    print(f"✗ Sync failed: {result.get('error', 'Unknown error')}")

# Add sample upcoming earnings events
print("\n3. Adding sample upcoming earnings events...")

upcoming_earnings = [
    # This week
    {
        'symbol': 'AAPL',
        'earnings_date': datetime.now() + timedelta(days=2),
        'earnings_time': 'AMC',
        'eps_estimate': 2.10,
        'pre_earnings_iv': 0.25,
        'whisper_number': 2.15
    },
    {
        'symbol': 'MSFT',
        'earnings_date': datetime.now() + timedelta(days=3),
        'earnings_time': 'AMC',
        'eps_estimate': 2.65,
        'pre_earnings_iv': 0.22,
        'whisper_number': 2.68
    },
    {
        'symbol': 'GOOGL',
        'earnings_date': datetime.now() + timedelta(days=4),
        'earnings_time': 'AMC',
        'eps_estimate': 1.45,
        'pre_earnings_iv': 0.28,
        'whisper_number': 1.50
    },
    # Next week
    {
        'symbol': 'AMZN',
        'earnings_date': datetime.now() + timedelta(days=7),
        'earnings_time': 'AMC',
        'eps_estimate': 1.20,
        'pre_earnings_iv': 0.30,
        'whisper_number': 1.25
    },
    {
        'symbol': 'NVDA',
        'earnings_date': datetime.now() + timedelta(days=8),
        'earnings_time': 'AMC',
        'eps_estimate': 5.50,
        'pre_earnings_iv': 0.35,
        'whisper_number': 5.75
    },
    {
        'symbol': 'TSLA',
        'earnings_date': datetime.now() + timedelta(days=9),
        'earnings_time': 'AMC',
        'eps_estimate': 0.85,
        'pre_earnings_iv': 0.45,
        'whisper_number': 0.90
    },
    {
        'symbol': 'META',
        'earnings_date': datetime.now() + timedelta(days=10),
        'earnings_time': 'AMC',
        'eps_estimate': 4.50,
        'pre_earnings_iv': 0.32,
        'whisper_number': 4.65
    },
    # Following week
    {
        'symbol': 'AMD',
        'earnings_date': datetime.now() + timedelta(days=14),
        'earnings_time': 'AMC',
        'eps_estimate': 0.92,
        'pre_earnings_iv': 0.38,
        'whisper_number': 0.95
    },
    {
        'symbol': 'NFLX',
        'earnings_date': datetime.now() + timedelta(days=15),
        'earnings_time': 'AMC',
        'eps_estimate': 3.20,
        'pre_earnings_iv': 0.40,
        'whisper_number': 3.30
    },
    {
        'symbol': 'INTC',
        'earnings_date': datetime.now() + timedelta(days=16),
        'earnings_time': 'AMC',
        'eps_estimate': 0.45,
        'pre_earnings_iv': 0.35,
        'whisper_number': 0.48
    },
]

added = 0
for event in upcoming_earnings:
    if manager.add_earnings_event(**event):
        added += 1

print(f"✓ Added {added} upcoming earnings events")

# Add some historical earnings with actuals (for demo analytics)
print("\n4. Adding historical earnings with results...")

historical_earnings = [
    {
        'symbol': 'AAPL',
        'earnings_date': datetime.now() - timedelta(days=90),
        'earnings_time': 'AMC',
        'eps_estimate': 1.95,
        'eps_actual': 2.05,  # Beat
        'pre_earnings_price': 178.50,
        'post_earnings_price': 185.20,
        'price_move_percent': 3.75
    },
    {
        'symbol': 'MSFT',
        'earnings_date': datetime.now() - timedelta(days=85),
        'earnings_time': 'AMC',
        'eps_estimate': 2.50,
        'eps_actual': 2.55,  # Beat
        'pre_earnings_price': 375.00,
        'post_earnings_price': 382.50,
        'price_move_percent': 2.00
    },
    {
        'symbol': 'GOOGL',
        'earnings_date': datetime.now() - timedelta(days=80),
        'earnings_time': 'AMC',
        'eps_estimate': 1.40,
        'eps_actual': 1.35,  # Miss
        'pre_earnings_price': 140.00,
        'post_earnings_price': 135.00,
        'price_move_percent': -3.57
    },
    {
        'symbol': 'AMZN',
        'earnings_date': datetime.now() - timedelta(days=75),
        'earnings_time': 'AMC',
        'eps_estimate': 1.15,
        'eps_actual': 1.30,  # Beat
        'pre_earnings_price': 145.00,
        'post_earnings_price': 152.00,
        'price_move_percent': 4.83
    },
    {
        'symbol': 'NVDA',
        'earnings_date': datetime.now() - timedelta(days=70),
        'earnings_time': 'AMC',
        'eps_estimate': 5.00,
        'eps_actual': 5.80,  # Beat big
        'pre_earnings_price': 480.00,
        'post_earnings_price': 520.00,
        'price_move_percent': 8.33
    },
]

added_hist = 0
for event in historical_earnings:
    # Calculate surprise percent
    surprise = ((event['eps_actual'] - event['eps_estimate']) /
                abs(event['eps_estimate']) * 100)
    event['surprise_percent'] = surprise

    if manager.add_earnings_event(**event):
        added_hist += 1

print(f"✓ Added {added_hist} historical earnings events")

# Display summary statistics
print("\n5. Summary Statistics:")
print("=" * 80)

# Get all earnings
import pandas as pd
all_earnings = manager.get_earnings_events(
    start_date=date.today() - timedelta(days=100),
    end_date=date.today() + timedelta(days=30)
)

if not all_earnings.empty:
    analytics = manager.get_analytics(all_earnings)

    print(f"Total Events:      {analytics['total_events']}")
    print(f"Pending:           {analytics['pending']}")
    print(f"Reported:          {analytics['beat'] + analytics['miss'] + analytics['inline']}")
    print(f"  - Beats:         {analytics['beat']}")
    print(f"  - Misses:        {analytics['miss']}")
    print(f"  - Inline:        {analytics['inline']}")
    print(f"Beat Rate:         {analytics['beat_rate']:.1f}%")
    print(f"Avg Surprise:      {analytics['avg_surprise']:.2f}%")

    # Upcoming earnings by week
    upcoming = all_earnings[all_earnings['earnings_date'] >= datetime.now()]
    this_week = upcoming[upcoming['earnings_date'] < datetime.now() + timedelta(days=7)]
    next_week = upcoming[
        (upcoming['earnings_date'] >= datetime.now() + timedelta(days=7)) &
        (upcoming['earnings_date'] < datetime.now() + timedelta(days=14))
    ]

    print(f"\nUpcoming Earnings:")
    print(f"  This Week:       {len(this_week)}")
    print(f"  Next Week:       {len(next_week)}")

    # Sectors
    if 'sector' in upcoming.columns:
        sector_counts = upcoming['sector'].value_counts()
        if not sector_counts.empty:
            print(f"\nBy Sector:")
            for sector, count in sector_counts.head(5).items():
                print(f"  {sector:20s} {count}")

else:
    print("No earnings data found")

print("\n" + "=" * 80)
print("SETUP COMPLETE!")
print("=" * 80)
print("\nNext Steps:")
print("1. Run the Streamlit dashboard:")
print("   streamlit run pages/earnings_calendar.py")
print("\n2. Or integrate into main dashboard:")
print("   python dashboard.py")
print("\n3. Use the Sync button in the app to fetch more data")
print("=" * 80)

# Close connection
manager.close()
