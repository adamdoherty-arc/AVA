"""
Final Verification - Confirm 7-Day Scanner Fix is Ready
Run this to verify the fix before testing in Streamlit
"""
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("7-DAY SCANNER FIX - FINAL VERIFICATION")
print("=" * 80)

# Check 1: Database has data
print("\n[1] Database Check:")
conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

cur.execute("""
    SELECT COUNT(*)
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
      AND delta BETWEEN -0.4 AND -0.2
""")
count = cur.fetchone()[0]
print(f"   Rows in database (DTE 5-9, Delta -0.4 to -0.2): {count}")
if count >= 300:
    print("   [OK] Sufficient data available")
else:
    print("   [WARN] Low data count - may need sync")

# Check 2: annual_return column exists and has values
cur.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(annual_return) as has_annual,
        MIN(annual_return) as min_annual,
        MAX(annual_return) as max_annual
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
      AND delta BETWEEN -0.4 AND -0.2
""")
result = cur.fetchone()
print(f"\n[2] annual_return Column Check:")
print(f"   Total rows: {result[0]}")
print(f"   Rows with annual_return: {result[1]}")
print(f"   Range: {result[2]:.1f}% to {result[3]:.1f}%")
if result[1] == result[0]:
    print("   [OK] All rows have annual_return values")
else:
    print(f"   [WARN] {result[0] - result[1]} rows missing annual_return")

cur.close()
conn.close()

# Check 3: Code file has the fix
print("\n[3] Code Fix Verification:")
with open('seven_day_dte_scanner_page.py', 'r', encoding='utf-8') as f:
    code = f.read()

checks = [
    ("Decimal conversion", "pd.to_numeric(df[col], errors='coerce')"),
    ("premium_pct fillna", "df['premium_pct'].fillna"),
    ("annual_return usage", "df['annualized_52wk'] = df['annual_return']"),
]

all_checks_pass = True
for check_name, check_string in checks:
    if check_string in code:
        print(f"   [OK] {check_name}")
    else:
        print(f"   [FAIL] {check_name} - Missing from code!")
        all_checks_pass = False

# Final verdict
print("\n" + "=" * 80)
if all_checks_pass and count >= 300 and result[1] == result[0]:
    print("STATUS: ALL CHECKS PASSED")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Restart Streamlit dashboard (Ctrl+C then 'streamlit run dashboard.py')")
    print("2. Navigate to '7-Day DTE Scanner' page")
    print("3. Should see 333+ opportunities!")
    print("\nIf still seeing 'No opportunities found':")
    print("  - Wait 60 seconds (cache TTL)")
    print("  - OR use Streamlit menu > Clear cache")
    print("  - OR run: python clear_streamlit_cache.py")
else:
    print("STATUS: SOME CHECKS FAILED")
    print("=" * 80)
    print("\nPlease review the failed checks above")

print("\n" + "=" * 80)
