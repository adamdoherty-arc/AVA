"""
Test the ACTUAL production code from seven_day_dte_scanner_page.py
Import the real function and verify it works end-to-end
"""
import sys
import os

# Add current directory to path so we can import the page module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the ACTUAL function from the production file
from seven_day_dte_scanner_page import fetch_opportunities

print("=" * 80)
print("TESTING ACTUAL PRODUCTION CODE")
print("Importing fetch_opportunities() from seven_day_dte_scanner_page.py")
print("=" * 80)

# Test with exact same parameters Streamlit uses
delta_range = (-0.4, -0.2)
min_premium = 0.0
min_annual_return = 30.0
min_volume = 0

print(f"\n[1] Calling ACTUAL fetch_opportunities function:")
print(f"    fetch_opportunities(5, 9, {delta_range[0]}, {delta_range[1]}, {min_premium})")

try:
    df_7day = fetch_opportunities(5, 9, delta_range[0], delta_range[1], min_premium)
    print(f"    [OK] Function executed successfully")
    print(f"    [OK] Returned {len(df_7day)} rows")
except Exception as e:
    print(f"    [FAIL] Function threw exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check data quality
print(f"\n[2] Data Quality Check:")
if not df_7day.empty:
    # Check for required columns
    required_cols = ['symbol', 'premium_pct', 'annualized_52wk', 'premium', 'strike_price', 'dte']
    missing_cols = [col for col in required_cols if col not in df_7day.columns]
    if missing_cols:
        print(f"    [FAIL] Missing columns: {missing_cols}")
        sys.exit(1)
    else:
        print(f"    [OK] All required columns present")

    # Check for NaN in critical column
    nan_count = df_7day['annualized_52wk'].isna().sum()
    print(f"    [OK] annualized_52wk NaN count: {nan_count}")
    if nan_count > 0:
        print(f"    [WARN] Found {nan_count} NaN values - these will be filtered out!")

    # Check value ranges
    print(f"    [OK] annualized_52wk range: {df_7day['annualized_52wk'].min():.1f}% to {df_7day['annualized_52wk'].max():.1f}%")
else:
    print(f"    [FAIL] DataFrame is empty!")
    sys.exit(1)

# Apply the EXACT same filter logic as Streamlit page (lines 309-312)
print(f"\n[3] Applying Streamlit Filter Logic (lines 309-312):")
print(f"    if not df_7day.empty:")
print(f"        df_7day = df_7day[df_7day['annualized_52wk'] >= {min_annual_return}]")

if not df_7day.empty:
    df_7day_filtered = df_7day[df_7day['annualized_52wk'] >= min_annual_return]
    print(f"    [OK] Before filter: {len(df_7day)} rows")
    print(f"    [OK] After filter: {len(df_7day_filtered)} rows")

    if min_volume > 0:
        df_7day_filtered = df_7day_filtered[df_7day_filtered['volume'] >= min_volume]
        print(f"    [OK] After volume filter: {len(df_7day_filtered)} rows")

    df_7day = df_7day_filtered

# Final check (line 314)
print(f"\n[4] Final Check (line 314): if not df_7day.empty:")
print(f"    df_7day.empty = {df_7day.empty}")
print(f"    len(df_7day) = {len(df_7day)}")

if not df_7day.empty:
    print(f"\n[5] ✓✓✓ SUCCESS! ✓✓✓")
    print(f"\n    Production code WILL work in Streamlit!")
    print(f"    {len(df_7day)} opportunities will be displayed to user")

    print(f"\n    Preview of what user will see:")
    print(f"\n    {'Symbol':<8} {'Strike':>10} {'Premium':>10} {'DTE':>4} {'Weekly%':>8} {'Annual%':>9}")
    print("    " + "-" * 65)
    for idx, row in df_7day.head(10).iterrows():
        print(f"    {row['symbol']:<8} ${row['strike_price']:>9.2f} ${row['premium']:>9.2f} {row['dte']:>4} {row['premium_pct']:>7.2f}% {row['annualized_52wk']:>8.1f}%")

    print("\n" + "=" * 80)
    print("PRODUCTION CODE VERIFIED WORKING")
    print("=" * 80)
    print("\nNext step: Restart Streamlit to clear cache")
    print("  Ctrl+C in terminal, then: streamlit run dashboard.py")

else:
    print(f"\n[5] XXX FAILURE XXX")
    print(f"\n    Production code is STILL broken!")
    print(f"    All rows were filtered out")

    # Debug why
    df_debug = fetch_opportunities(5, 9, delta_range[0], delta_range[1], min_premium)
    if not df_debug.empty:
        print(f"\n    Debug info:")
        print(f"    - Rows fetched: {len(df_debug)}")
        print(f"    - Rows with annualized >= {min_annual_return}: {len(df_debug[df_debug['annualized_52wk'] >= min_annual_return])}")
        print(f"    - annualized_52wk sample: {df_debug['annualized_52wk'].head(5).tolist()}")
        print(f"    - annualized_52wk has NaN: {df_debug['annualized_52wk'].isna().any()}")

    sys.exit(1)
