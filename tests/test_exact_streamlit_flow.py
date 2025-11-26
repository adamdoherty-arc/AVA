"""
Test EXACT Streamlit Flow for 7-Day Scanner
Replicate the exact filtering logic from the page
"""
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_opportunities(dte_min, dte_max, delta_min=-0.4, delta_max=-0.2, min_premium=0):
    """EXACT copy of the fetch function from seven_day_dte_scanner_page.py"""
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        database='magnus',
        user='postgres',
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()

    query = '''
        SELECT
            symbol,
            strike_price,
            premium,
            dte,
            premium_pct,
            annual_return,
            delta,
            prob_profit,
            implied_volatility,
            volume,
            open_interest,
            strike_type,
            expiration_date,
            bid,
            ask
        FROM stock_premiums
        WHERE dte BETWEEN %s AND %s
          AND premium > %s
          AND delta BETWEEN %s AND %s
          AND strike_price > 0
        ORDER BY (premium / dte) DESC
    '''

    cur.execute(query, (dte_min, dte_max, min_premium, delta_min, delta_max))
    columns = [desc[0] for desc in cur.description]
    results = cur.fetchall()
    cur.close()
    conn.close()

    df = pd.DataFrame(results, columns=columns)

    # Calculate additional metrics (EXACT copy from file)
    if not df.empty:
        # Convert Decimal columns to float for calculations
        numeric_cols = ['premium', 'strike_price', 'dte', 'premium_pct', 'annual_return', 'delta', 'prob_profit', 'implied_volatility', 'bid', 'ask']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Use premium_pct if available, otherwise calculate from premium/strike
        if df['premium_pct'].isna().all() or (df['premium_pct'] == 0).all():
            print(f"   [CALC] Calculating premium_pct from premium/strike")
            df['premium_pct'] = (df['premium'] / df['strike_price']) * 100

        df['weekly_return'] = df['premium_pct']
        df['annualized_52wk'] = df['premium_pct'] * (365 / df['dte'])
        df['premium_per_day'] = df['premium'] / df['dte']
        df['risk_reward_ratio'] = df['premium'] / df['strike_price']
        df['bid_ask_spread'] = df.apply(lambda x: (x['ask'] - x['bid']) if pd.notna(x['bid']) and pd.notna(x['ask']) else 0, axis=1)

    return df

print("=" * 80)
print("TESTING EXACT STREAMLIT FLOW")
print("=" * 80)

# Default values from Streamlit UI
print("\n[1] UI Filter Values (from slider defaults):")
delta_range = (-0.4, -0.2)
min_premium = 0.0
min_annual_return = 30.0
min_volume = 0

print(f"   Delta Range: {delta_range}")
print(f"   Min Premium: ${min_premium}")
print(f"   Min Annualized Return: {min_annual_return}%")
print(f"   Min Volume: {min_volume}")

# Step 1: Fetch (line 306 in streamlit page)
print("\n[2] Fetch 7-day opportunities (line 306):")
print(f"   fetch_opportunities(5, 9, {delta_range[0]}, {delta_range[1]}, {min_premium})")
df_7day = fetch_opportunities(5, 9, delta_range[0], delta_range[1], min_premium)
print(f"   Fetched: {len(df_7day)} rows")

if not df_7day.empty:
    print(f"\n   Preview of fetched data:")
    print(f"   - premium_pct range: {df_7day['premium_pct'].min():.2f}% to {df_7day['premium_pct'].max():.2f}%")
    print(f"   - annualized_52wk range: {df_7day['annualized_52wk'].min():.1f}% to {df_7day['annualized_52wk'].max():.1f}%")
    print(f"   - First 3 annualized_52wk values: {df_7day['annualized_52wk'].head(3).tolist()}")

# Step 2: Apply additional filters (lines 309-312)
print("\n[3] Apply additional filters (lines 309-312):")
print(f"   Before filtering: {len(df_7day)} rows")

if not df_7day.empty:
    print(f"   Filter 1: df_7day[df_7day['annualized_52wk'] >= {min_annual_return}]")
    df_7day_filtered = df_7day[df_7day['annualized_52wk'] >= min_annual_return]
    print(f"   After annual return filter: {len(df_7day_filtered)} rows")

    if min_volume > 0:
        print(f"   Filter 2: df_7day[df_7day['volume'] >= {min_volume}]")
        df_7day_filtered = df_7day_filtered[df_7day_filtered['volume'] >= min_volume]
        print(f"   After volume filter: {len(df_7day_filtered)} rows")

    df_7day = df_7day_filtered

# Step 3: Check if empty (line 314)
print(f"\n[4] Final check (line 314): if not df_7day.empty:")
print(f"   df_7day.empty = {df_7day.empty}")
print(f"   len(df_7day) = {len(df_7day)}")

if not df_7day.empty:
    print("\n[5] SUCCESS! Data would be displayed to user")
    print(f"\n   Top 5 opportunities:")
    print(f"   {'Symbol':<8} {'Strike':>10} {'Premium':>10} {'DTE':>4} {'Weekly%':>8} {'Annual%':>9}")
    print("   " + "-" * 65)
    for idx, row in df_7day.head(5).iterrows():
        print(f"   {row['symbol']:<8} ${row['strike_price']:>9.2f} ${row['premium']:>9.2f} {row['dte']:>4} {row['premium_pct']:>7.2f}% {row['annualized_52wk']:>8.1f}%")
else:
    print("\n[5] FAILURE! Page would show: 'No 7-day opportunities found'")
    print("\n   Debugging empty result:")

    # Re-fetch without annualized filter to see what's happening
    df_debug = fetch_opportunities(5, 9, delta_range[0], delta_range[1], min_premium)
    if not df_debug.empty:
        print(f"\n   Data exists ({len(df_debug)} rows) but got filtered out")
        print(f"   annualized_52wk column exists: {'annualized_52wk' in df_debug.columns}")
        print(f"   annualized_52wk is NaN: {df_debug['annualized_52wk'].isna().all()}")
        print(f"   annualized_52wk sample values: {df_debug['annualized_52wk'].head(5).tolist()}")

        # Check how many rows pass the filter
        passing = df_debug[df_debug['annualized_52wk'] >= min_annual_return]
        print(f"   Rows passing annualized >= {min_annual_return}%: {len(passing)}")

        if len(passing) == 0:
            print(f"\n   ⚠️ PROBLEM: All rows filtered out by annualized_52wk >= {min_annual_return}")
            print(f"   Max annualized_52wk in data: {df_debug['annualized_52wk'].max():.1f}%")
            print(f"   Min annualized_52wk in data: {df_debug['annualized_52wk'].min():.1f}%")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
