"""
Explore available earnings data sources
"""
import robin_stocks.robinhood as rh
import os
from dotenv import load_dotenv
import json
from src.tradingview_db_manager import TradingViewDBManager

load_dotenv()

print("=" * 80)
print("EARNINGS DATA EXPLORATION")
print("=" * 80)

# Check database first
print("\n1. DATABASE EARNINGS TABLES:")
print("=" * 80)

tv = TradingViewDBManager()
conn = tv.get_connection()
cur = conn.cursor()

# Check earnings_events table
cur.execute("SELECT COUNT(*) FROM earnings_events")
earnings_count = cur.fetchone()[0]
print(f"\nearnings_events table: {earnings_count} rows")

if earnings_count > 0:
    cur.execute("SELECT * FROM earnings_events LIMIT 5")
    rows = cur.fetchall()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'earnings_events'")
    columns = [row[0] for row in cur.fetchall()]
    print(f"Columns: {', '.join(columns)}")
    print("\nSample data:")
    for row in rows:
        print(f"  {dict(zip(columns, row))}")

# Check earnings_events_enhanced table
cur.execute("SELECT COUNT(*) FROM earnings_events_enhanced")
enhanced_count = cur.fetchone()[0]
print(f"\nearnings_events_enhanced table: {enhanced_count} rows")

# Check earnings_history table
cur.execute("SELECT COUNT(*) FROM earnings_history")
history_count = cur.fetchone()[0]
print(f"earnings_history table: {history_count} rows")

# Check earnings_alerts table
cur.execute("SELECT COUNT(*) FROM earnings_alerts")
alerts_count = cur.fetchone()[0]
print(f"earnings_alerts table: {alerts_count} rows")

cur.close()
conn.close()

# Check Robinhood API
print("\n\n2. ROBINHOOD EARNINGS API:")
print("=" * 80)

username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')

print("\nLogging in to Robinhood...")
rh.login(username, password)
print("Logged in successfully")

# Test get_earnings() for various stocks
test_symbols = ['AAPL', 'NVDA', 'TSLA', 'META', 'GOOGL']

print("\nTesting get_earnings() for popular stocks:")

for symbol in test_symbols:
    print(f"\n{symbol}:")
    try:
        earnings = rh.get_earnings(symbol)

        if earnings:
            if isinstance(earnings, list):
                print(f"  Found {len(earnings)} earnings records")
                if len(earnings) > 0:
                    print(f"\n  Latest earnings data:")
                    latest = earnings[0]
                    print(f"    Report Date: {latest.get('report', {}).get('date', 'N/A')}")
                    print(f"    EPS Actual: {latest.get('eps', {}).get('actual', 'N/A')}")
                    print(f"    EPS Estimate: {latest.get('eps', {}).get('estimate', 'N/A')}")
                    print(f"    Call Info: {latest.get('call', {})}")

                    # Show all available keys
                    print(f"\n  Available data fields:")
                    for key in latest.keys():
                        print(f"    - {key}")
            else:
                print(f"  Data type: {type(earnings)}")
                print(json.dumps(earnings, indent=4))
        else:
            print(f"  No earnings data found")

    except Exception as e:
        print(f"  Error: {e}")

rh.logout()

print("\n\n3. POLYGON API (if configured):")
print("=" * 80)

try:
    from src.polygon_client import PolygonClient
    polygon = PolygonClient()

    print("\nTesting Polygon earnings endpoint...")
    # Polygon may have earnings calendar endpoint
    # polygon.get_earnings_calendar()

    print("Polygon client available - check if earnings endpoints exist")

except Exception as e:
    print(f"Polygon client not available or configured: {e}")

print("\n\n" + "=" * 80)
print("SUMMARY OF AVAILABLE DATA:")
print("=" * 80)
print("""
1. DATABASE:
   - earnings_events: {earnings_count} records
   - earnings_events_enhanced: {enhanced_count} records
   - earnings_history: {history_count} records
   - earnings_alerts: {alerts_count} records

2. ROBINHOOD API:
   - get_earnings(symbol) - provides historical earnings data
   - Includes: EPS actual, EPS estimate, report date, call info

3. POTENTIAL ENHANCEMENTS:
   - Sync earnings from Robinhood for all stocks
   - Build earnings calendar view
   - Add filters by date range, sector, surprise %
   - Show earnings whispers/estimates
   - Track historical beat/miss patterns
""".format(
    earnings_count=earnings_count,
    enhanced_count=enhanced_count,
    history_count=history_count,
    alerts_count=alerts_count
))
