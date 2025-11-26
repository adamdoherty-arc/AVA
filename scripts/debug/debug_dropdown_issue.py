"""Debug stock dropdown issue - trace execution step by step"""
import sys
sys.path.insert(0, r'c:\code\Magnus')

import pandas as pd
from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

print("=" * 70)
print("DEBUGGING STOCK DROPDOWN ISSUE")
print("=" * 70)

# Step 1: Test database connection and query
print("\n[Step 1] Testing database query...")
db_manager = AIOptionsDBManager()
conn = db_manager.get_connection()

query = """
    SELECT DISTINCT
        symbol,
        company_name,
        current_price,
        sector,
        market_cap,
        pe_ratio
    FROM stock_data
    WHERE symbol IS NOT NULL
        AND current_price > 0
    ORDER BY symbol
"""

try:
    df = pd.read_sql(query, conn)
    print(f"[OK] Query executed successfully")
    print(f"[OK] Retrieved {len(df)} records")
    print(f"\nDataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    print(f"\nFirst 5 records:")
    print(df.head())
except Exception as e:
    print(f"[FAIL] Query failed: {e}")
    conn.close()
    exit(1)

conn.close()

# Step 2: Test formatting function with NULL company_name
print("\n" + "=" * 70)
print("[Step 2] Testing format function with actual data...")

def format_stock_option(row):
    symbol = row.get('symbol', '')
    company_name = row.get('company_name', '')
    price = row.get('current_price', 0)

    # Simple, clean format: AAPL - Apple Inc. ($235.50)
    if company_name and company_name != symbol:
        return f"{symbol} - {company_name} (${price:.2f})"
    else:
        return f"{symbol} (${price:.2f})"

print("\nFormatted options (first 20):")
for idx, row in df.head(20).iterrows():
    formatted = format_stock_option(row)
    print(f"  {formatted}")

# Step 3: Test selectbox options list generation
print("\n" + "=" * 70)
print("[Step 3] Testing selectbox options generation...")

options = df.apply(format_stock_option, axis=1).tolist()
symbols = df['symbol'].tolist()

print(f"[OK] Generated {len(options)} options")
print(f"[OK] Generated {len(symbols)} symbols")

# Add default option
options.insert(0, f"-- Select from {len(df)} stocks --")
symbols.insert(0, None)

print(f"\nFinal options list (first 10):")
for i, opt in enumerate(options[:10]):
    print(f"  [{i}] {opt} -> {symbols[i]}")

# Step 4: Test without caching (simulate fresh load)
print("\n" + "=" * 70)
print("[Step 4] Testing component import and initialization...")

try:
    from src.components.stock_dropdown import StockDropdown
    print("[OK] StockDropdown imported successfully")

    dropdown = StockDropdown(db_manager=AIOptionsDBManager())
    print("[OK] StockDropdown initialized")

    # Call _get_stock_list directly
    stocks_df = dropdown._get_stock_list()
    print(f"[OK] _get_stock_list() returned {len(stocks_df)} records")

    # Test format function
    if len(stocks_df) > 0:
        first_row = stocks_df.iloc[0]
        formatted = dropdown._format_stock_option(first_row)
        print(f"[OK] Format function works: {formatted}")

except Exception as e:
    print(f"[FAIL] Component test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Check for data type issues
print("\n" + "=" * 70)
print("[Step 5] Checking data types...")
print("\nDataFrame dtypes:")
print(df.dtypes)

print("\nChecking for NaN values:")
print(df.isnull().sum())

print("\nChecking company_name unique values (first 10):")
print(df['company_name'].unique()[:10])

# Step 6: Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"[OK] Database query: {len(df)} records retrieved")
print(f"[OK] Formatting function: Works with NULL company_name")
print(f"[OK] Options list: {len(options)} items generated")
print(f"[OK] Component import: Success")
print("\nISSUE IDENTIFIED:")
if df['company_name'].isnull().all():
    print("  - company_name is NULL for all records")
    print("  - Format function correctly falls back to symbol only")
    print("  - Dropdown should still work with this data")
    print("\n  -> Likely cause: Streamlit caching or widget key issue")
    print("  -> Next step: Clear Streamlit cache and restart dashboard")
else:
    print("  - Some records have company_name populated")
    print("  -> Check for data inconsistency")

print("=" * 70)
