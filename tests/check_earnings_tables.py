"""
Verify earnings calendar tables are set up correctly
"""

from src.tradingview_db_manager import TradingViewDBManager
import pandas as pd

print("=" * 80)
print("EARNINGS CALENDAR - TABLE VERIFICATION")
print("=" * 80)

tv = TradingViewDBManager()
conn = tv.get_connection()
cur = conn.cursor()

# Check earnings_events table
print("\n1. earnings_events TABLE:")
print("-" * 80)

try:
    cur.execute("SELECT COUNT(*) FROM earnings_events")
    count = cur.fetchone()[0]
    print(f"✓ Table exists")
    print(f"✓ Total rows: {count}")

    if count > 0:
        # Sample data
        cur.execute("""
            SELECT symbol, earnings_date, earnings_time, eps_estimate, eps_actual
            FROM earnings_events
            ORDER BY earnings_date DESC
            LIMIT 5
        """)
        rows = cur.fetchall()

        print("\nSample data (last 5 records):")
        print(f"{'Symbol':<8} {'Date':<20} {'Time':<6} {'Est':<8} {'Actual':<8}")
        print("-" * 60)
        for row in rows:
            symbol, date, time, est, actual = row
            time_str = time or "N/A"
            est_str = f"${est:.2f}" if est else "N/A"
            actual_str = f"${actual:.2f}" if actual else "Pending"
            print(f"{symbol:<8} {str(date):<20} {time_str:<6} {est_str:<8} {actual_str:<8}")

        # Get date range
        cur.execute("""
            SELECT MIN(earnings_date), MAX(earnings_date)
            FROM earnings_events
        """)
        min_date, max_date = cur.fetchone()
        print(f"\nDate range: {min_date} to {max_date}")

        # Count by status
        cur.execute("""
            SELECT
                CASE
                    WHEN eps_actual IS NULL THEN 'Pending'
                    WHEN eps_actual > eps_estimate THEN 'Beat'
                    WHEN eps_actual < eps_estimate THEN 'Miss'
                    ELSE 'Inline'
                END as status,
                COUNT(*) as count
            FROM earnings_events
            WHERE eps_estimate IS NOT NULL
            GROUP BY status
        """)
        status_counts = cur.fetchall()

        if status_counts:
            print("\nBy Status:")
            for status, cnt in status_counts:
                print(f"  {status:<10} {cnt:>5} events")

    else:
        print("\n⚠️  Table is empty - run sync_earnings_demo.py to populate")

except Exception as e:
    print(f"✗ Error: {e}")

# Check earnings_history table
print("\n\n2. earnings_history TABLE:")
print("-" * 80)

try:
    cur.execute("SELECT COUNT(*) FROM earnings_history")
    count = cur.fetchone()[0]
    print(f"✓ Table exists")
    print(f"✓ Total rows: {count}")

    if count > 0:
        # Sample data
        cur.execute("""
            SELECT symbol, report_date, quarter, year, eps_actual, eps_estimate
            FROM earnings_history
            ORDER BY report_date DESC
            LIMIT 5
        """)
        rows = cur.fetchall()

        print("\nSample data (last 5 records):")
        print(f"{'Symbol':<8} {'Date':<12} {'Quarter':<10} {'Year':<6} {'Actual':<8} {'Estimate':<8}")
        print("-" * 65)
        for row in rows:
            symbol, date, quarter, year, actual, estimate = row
            q_str = f"Q{quarter}" if quarter else "N/A"
            y_str = str(year) if year else "N/A"
            act_str = f"${actual:.2f}" if actual else "N/A"
            est_str = f"${estimate:.2f}" if estimate else "N/A"
            print(f"{symbol:<8} {str(date):<12} {q_str:<10} {y_str:<6} {act_str:<8} {est_str:<8}")

        # Count by symbol
        cur.execute("""
            SELECT symbol, COUNT(*) as count
            FROM earnings_history
            GROUP BY symbol
            ORDER BY count DESC
            LIMIT 10
        """)
        symbol_counts = cur.fetchall()

        print("\nTop symbols by record count:")
        for symbol, cnt in symbol_counts:
            print(f"  {symbol:<8} {cnt:>3} quarters")

    else:
        print("\n⚠️  Table is empty - run sync_earnings_demo.py to populate")

except Exception as e:
    print(f"✗ Error: {e}")

# Check indexes
print("\n\n3. DATABASE INDEXES:")
print("-" * 80)

try:
    cur.execute("""
        SELECT
            tablename,
            indexname,
            indexdef
        FROM pg_indexes
        WHERE tablename IN ('earnings_events', 'earnings_history')
        ORDER BY tablename, indexname
    """)
    indexes = cur.fetchall()

    if indexes:
        current_table = None
        for table, index, definition in indexes:
            if table != current_table:
                print(f"\n{table}:")
                current_table = table
            print(f"  ✓ {index}")
    else:
        print("⚠️  No indexes found")

except Exception as e:
    print(f"✗ Error: {e}")

# Check stocks table (for joining data)
print("\n\n4. RELATED TABLES CHECK:")
print("-" * 80)

try:
    cur.execute("""
        SELECT COUNT(*)
        FROM stocks
        WHERE is_active = TRUE
    """)
    active_stocks = cur.fetchone()[0]
    print(f"✓ stocks table: {active_stocks} active stocks")

    # Check how many have earnings data
    cur.execute("""
        SELECT COUNT(DISTINCT e.symbol)
        FROM earnings_events e
        JOIN stocks s ON e.symbol = s.symbol
    """)
    with_earnings = cur.fetchone()[0]
    print(f"✓ Stocks with earnings data: {with_earnings}")

except Exception as e:
    print(f"✗ Error checking related tables: {e}")

# System recommendations
print("\n\n5. RECOMMENDATIONS:")
print("-" * 80)

try:
    cur.execute("SELECT COUNT(*) FROM earnings_events")
    ee_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM earnings_history")
    eh_count = cur.fetchone()[0]

    if ee_count == 0 and eh_count == 0:
        print("⚠️  No earnings data found")
        print("   → Run: python sync_earnings_demo.py")
        print("   → This will populate sample data and sync from Robinhood")

    elif ee_count == 0:
        print("⚠️  No upcoming earnings events")
        print("   → Add upcoming earnings manually or via API")
        print("   → Use EarningsManager.add_earnings_event()")

    elif eh_count == 0:
        print("⚠️  No historical earnings data")
        print("   → Run Robinhood sync from dashboard")
        print("   → Or run: python sync_earnings_demo.py")

    else:
        print("✓ Earnings calendar is set up and populated")
        print("✓ Ready to use!")

        # Check for upcoming earnings
        from datetime import datetime, timedelta
        cur.execute("""
            SELECT COUNT(*)
            FROM earnings_events
            WHERE earnings_date >= NOW()
            AND earnings_date <= NOW() + INTERVAL '7 days'
        """)
        upcoming = cur.fetchone()[0]

        if upcoming > 0:
            print(f"✓ {upcoming} earnings events in the next 7 days")
        else:
            print("ℹ️  No earnings in next 7 days - consider syncing more data")

except Exception as e:
    print(f"✗ Error generating recommendations: {e}")

# Next steps
print("\n\n6. NEXT STEPS:")
print("-" * 80)
print("1. Run the Streamlit page:")
print("   streamlit run pages/earnings_calendar.py")
print("\n2. Or integrate into main dashboard:")
print("   See INTEGRATE_EARNINGS_CALENDAR.md")
print("\n3. Sync more data:")
print("   Use the sync button in the dashboard")
print("   Or run: python sync_earnings_demo.py")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)

cur.close()
conn.close()
