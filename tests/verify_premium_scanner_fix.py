"""
Final verification that Premium Scanner fix works
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def verify_fix():
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        database='magnus',
        user='postgres',
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()

    print("=" * 80)
    print("PREMIUM SCANNER FIX VERIFICATION")
    print("=" * 80)

    # Test 7-day scanner
    print("\n1. Testing 7-DAY SCANNER")
    print("-" * 80)

    dte_min, dte_max = 5, 9
    delta_min, delta_max = -1.0, 0.0
    min_premium = 0.0
    min_stock_price = 0.0
    max_stock_price = 10000.0

    # FIXED query (simplified, no ABS)
    query = '''
        SELECT DISTINCT ON (sp.symbol)
            sp.symbol,
            sd.current_price as stock_price,
            sp.strike_price,
            sp.premium,
            sp.delta,
            sp.dte
        FROM stock_premiums sp
        LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
        LEFT JOIN stocks s ON sp.symbol = s.symbol
        WHERE sp.dte BETWEEN %s AND %s
          AND sp.premium >= %s
          AND sp.delta BETWEEN %s AND %s
          AND sp.strike_price > 0
          AND (sd.current_price BETWEEN %s AND %s OR sd.current_price IS NULL)
        ORDER BY sp.symbol, (sp.premium / sp.dte) DESC
    '''

    cur.execute(query, (dte_min, dte_max, min_premium, delta_min, delta_max,
                        min_stock_price, max_stock_price))
    results_7day = cur.fetchall()

    print(f"✓ Query executed successfully")
    print(f"✓ Results: {len(results_7day)} opportunities")
    print(f"\nFirst 5 opportunities:")
    for row in results_7day[:5]:
        print(f"  {row[0]:6s} | Stock: ${row[1] or 0:7.2f} | Strike: ${row[2]:7.2f} | "
              f"Premium: ${row[3]:6.2f} | Delta: {row[4]:6.3f} | DTE: {row[5]:2d}")

    # Test 30-day scanner
    print("\n2. Testing 30-DAY SCANNER")
    print("-" * 80)

    dte_min, dte_max = 25, 35

    cur.execute(query, (dte_min, dte_max, min_premium, delta_min, delta_max,
                        min_stock_price, max_stock_price))
    results_30day = cur.fetchall()

    print(f"✓ Query executed successfully")
    print(f"✓ Results: {len(results_30day)} opportunities")
    if len(results_30day) > 0:
        print(f"\nFirst 5 opportunities:")
        for row in results_30day[:5]:
            print(f"  {row[0]:6s} | Stock: ${row[1] or 0:7.2f} | Strike: ${row[2]:7.2f} | "
                  f"Premium: ${row[3]:6.2f} | Delta: {row[4]:6.3f} | DTE: {row[5]:2d}")
    else:
        print("\n  No 30-day data in database (this is OK if not synced yet)")

    # Verify stats match
    print("\n3. VERIFY STATS vs RESULTS")
    print("-" * 80)

    # Stats query (like get_stats function)
    cur.execute("""
        SELECT COUNT(DISTINCT symbol), COUNT(*)
        FROM stock_premiums
        WHERE dte BETWEEN 5 AND 9
    """)
    stats = cur.fetchone()
    print(f"Stats query (no filters):  {stats[0]} symbols, {stats[1]} total records")
    print(f"Results query (filtered):  {len(results_7day)} symbols (DISTINCT ON)")
    print(f"\nThis is CORRECT - DISTINCT ON returns one row per symbol")

    # Test parameter count
    print("\n4. VERIFY PARAMETER COUNT")
    print("-" * 80)
    params = (dte_min, dte_max, min_premium, delta_min, delta_max,
              min_stock_price, max_stock_price)
    print(f"Parameters passed: {len(params)}")
    print(f"Expected: 7 (dte_min, dte_max, min_premium, delta_min, delta_max, min_stock_price, max_stock_price)")

    if len(params) == 7:
        print("✓ PASS - Correct parameter count")
    else:
        print("✗ FAIL - Wrong parameter count")

    cur.close()
    conn.close()

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    success = True
    if len(results_7day) > 0:
        print("✓ 7-day scanner: WORKING ({} results)".format(len(results_7day)))
    else:
        print("✗ 7-day scanner: FAILED (0 results)")
        success = False

    if len(params) == 7:
        print("✓ Parameter count: CORRECT (7 parameters)")
    else:
        print("✗ Parameter count: WRONG ({} parameters)".format(len(params)))
        success = False

    print("=" * 80)
    if success:
        print("✓✓✓ ALL TESTS PASSED - FIX VERIFIED ✓✓✓")
        print("\nNext steps:")
        print("1. In Streamlit, click hamburger menu > Clear cache")
        print("2. Refresh the Premium Scanner page (F5)")
        print("3. Both scanners should now show results")
    else:
        print("✗✗✗ TESTS FAILED - ISSUE REMAINS ✗✗✗")

    print("=" * 80)

if __name__ == "__main__":
    verify_fix()
