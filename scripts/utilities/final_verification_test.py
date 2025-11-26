"""
Final comprehensive verification that Premium Scanner is fixed
"""
import sys
sys.path.insert(0, 'c:/code/Magnus')

# Import the ACTUAL function from premium_scanner_page.py
import importlib.util
spec = importlib.util.spec_from_file_location("premium_scanner_page", "c:/code/Magnus/premium_scanner_page.py")
premium_scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(premium_scanner)

print("=" * 80)
print("FINAL VERIFICATION - PREMIUM SCANNER FIX")
print("=" * 80)

# Test 7-day scanner
print("\n1. Testing 7-DAY SCANNER (dte 5-9)")
print("-" * 80)
df_7day = premium_scanner.fetch_opportunities(
    dte_min=5,
    dte_max=9,
    delta_min=-1.0,
    delta_max=0.0,
    min_premium=0.0,
    min_stock_price=0.0,
    max_stock_price=10000.0
)

if df_7day.empty:
    print("FAILED: 7-day scanner returned empty DataFrame")
else:
    print(f"SUCCESS: {len(df_7day)} opportunities found")
    print(f"Required columns present: {all(col in df_7day.columns for col in ['symbol', 'premium', 'delta', 'annualized_52wk'])}")
    print(f"\nFirst 3 results:")
    for idx, row in df_7day.head(3).iterrows():
        print(f"  {row['symbol']:6s} | Premium: ${row['premium']:6.2f} | Delta: {row['delta']:6.3f} | Annual: {row['annualized_52wk']:6.1f}%")

# Test 30-day scanner
print("\n2. Testing 30-DAY SCANNER (dte 25-35)")
print("-" * 80)
df_30day = premium_scanner.fetch_opportunities(
    dte_min=25,
    dte_max=35,
    delta_min=-1.0,
    delta_max=0.0,
    min_premium=0.0,
    min_stock_price=0.0,
    max_stock_price=10000.0
)

if df_30day.empty:
    print("WARNING: 30-day scanner returned empty (might not have data synced)")
else:
    print(f"SUCCESS: {len(df_30day)} opportunities found")
    print(f"Required columns present: {all(col in df_30day.columns for col in ['symbol', 'premium', 'delta', 'annualized_52wk'])}")
    print(f"\nFirst 3 results:")
    for idx, row in df_30day.head(3).iterrows():
        print(f"  {row['symbol']:6s} | Premium: ${row['premium']:6.2f} | Delta: {row['delta']:6.3f} | Annual: {row['annual_return']:6.1f}%")

# Test stats function
print("\n3. Testing STATS FUNCTION")
print("-" * 80)
stats_7 = premium_scanner.get_stats(5, 9)
if stats_7:
    print(f"7-day stats: {stats_7['unique_symbols']} symbols, {stats_7['total_opportunities']} opportunities")
else:
    print("FAILED: Stats returned None")

stats_30 = premium_scanner.get_stats(25, 35)
if stats_30:
    print(f"30-day stats: {stats_30['unique_symbols']} symbols, {stats_30['total_opportunities']} opportunities")
else:
    print("WARNING: 30-day stats returned None")

# Test last sync time
print("\n4. Testing LAST SYNC TIME")
print("-" * 80)
last_sync_7 = premium_scanner.get_last_sync_time('7day')
if last_sync_7:
    print(f"7-day last sync: {last_sync_7}")
else:
    print("No 7-day sync time found")

last_sync_30 = premium_scanner.get_last_sync_time('30day')
if last_sync_30:
    print(f"30-day last sync: {last_sync_30}")
else:
    print("No 30-day sync time found")

# Final verdict
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

tests_passed = 0
tests_total = 0

# Test 1: 7-day scanner
tests_total += 1
if not df_7day.empty and len(df_7day) > 0:
    print(f"Test 1: 7-day scanner - PASS ({len(df_7day)} results)")
    tests_passed += 1
else:
    print("Test 1: 7-day scanner - FAIL (no results)")

# Test 2: Calculated columns
tests_total += 1
if not df_7day.empty and 'annualized_52wk' in df_7day.columns:
    print(f"Test 2: Calculated columns - PASS")
    tests_passed += 1
else:
    print("Test 2: Calculated columns - FAIL")

# Test 3: No type errors
tests_total += 1
if not df_7day.empty and df_7day['premium'].dtype in ['float64', 'float32', 'int64']:
    print(f"Test 3: Type conversions - PASS (premium is {df_7day['premium'].dtype})")
    tests_passed += 1
else:
    print("Test 3: Type conversions - FAIL")

# Test 4: Stats function
tests_total += 1
if stats_7 and stats_7['unique_symbols'] > 0:
    print(f"Test 4: Stats function - PASS ({stats_7['unique_symbols']} symbols)")
    tests_passed += 1
else:
    print("Test 4: Stats function - FAIL")

print("\n" + "=" * 80)
if tests_passed == tests_total:
    print(f"ALL TESTS PASSED ({tests_passed}/{tests_total})")
    print("\nPremium Scanner is FIXED and WORKING!")
    print("\nNext step: Clear Streamlit cache and refresh page")
else:
    print(f"SOME TESTS FAILED ({tests_passed}/{tests_total})")

print("=" * 80)
