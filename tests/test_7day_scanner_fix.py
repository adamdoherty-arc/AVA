"""
Test the 7-Day Scanner fix for premium_pct issue
"""
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Replicate the fixed fetch_opportunities function
def fetch_opportunities_test(dte_min, dte_max, delta_min=-0.4, delta_max=-0.2, min_premium=0):
    """Test version of fetch function with fix"""
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
        LIMIT 20
    '''

    cur.execute(query, (dte_min, dte_max, min_premium, delta_min, delta_max))
    columns = [desc[0] for desc in cur.description]
    results = cur.fetchall()
    cur.close()
    conn.close()

    df = pd.DataFrame(results, columns=columns)

    # THE FIX: Calculate premium_pct if it's NULL
    if not df.empty:
        # Convert Decimal columns to float for calculations
        numeric_cols = ['premium', 'strike_price', 'dte', 'premium_pct', 'annual_return', 'delta', 'prob_profit', 'implied_volatility', 'bid', 'ask']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Use premium_pct if available, otherwise calculate from premium/strike
        if df['premium_pct'].isna().all() or (df['premium_pct'] == 0).all():
            print("   Fixing NULL premium_pct values...")
            df['premium_pct'] = (df['premium'] / df['strike_price']) * 100

        df['weekly_return'] = df['premium_pct']
        df['annualized_52wk'] = df['premium_pct'] * (365 / df['dte'])
        df['premium_per_day'] = df['premium'] / df['dte']

    return df

print("=" * 80)
print("TESTING 7-DAY SCANNER FIX")
print("=" * 80)

# Test with default filters (delta -0.4 to -0.2)
print("\n[1] Fetching 7-day opportunities (DTE 5-9, Delta -0.4 to -0.2)")
df = fetch_opportunities_test(5, 9, -0.4, -0.2, 0)
print(f"   Fetched: {len(df)} rows")

if not df.empty:
    print("\n[2] Before annualized filter:")
    print(f"   premium_pct range: {df['premium_pct'].min():.2f}% to {df['premium_pct'].max():.2f}%")
    print(f"   annualized_52wk range: {df['annualized_52wk'].min():.1f}% to {df['annualized_52wk'].max():.1f}%")

    # Apply the same filter as the page
    min_annual_return = 30.0
    print(f"\n[3] Applying filter: annualized_52wk >= {min_annual_return}%")
    df_filtered = df[df['annualized_52wk'] >= min_annual_return]
    print(f"   After filter: {len(df_filtered)} rows")

    if not df_filtered.empty:
        print("\n[4] SUCCESS! Top 5 opportunities:")
        print(f"\n{'Symbol':<8} {'Strike':>10} {'Premium':>10} {'DTE':>4} {'Weekly%':>8} {'Annual%':>9}")
        print("-" * 65)
        for idx, row in df_filtered.head(5).iterrows():
            print(f"{row['symbol']:<8} ${row['strike_price']:>9.2f} ${row['premium']:>9.2f} {row['dte']:>4} {row['premium_pct']:>7.2f}% {row['annualized_52wk']:>8.1f}%")

        print("\n" + "=" * 80)
        print(f"FIX VERIFIED: {len(df_filtered)} opportunities will now display in the scanner!")
        print("=" * 80)
    else:
        print("\n   ERROR: All filtered out! Check annualized_52wk calculation")
else:
    print("\n   ERROR: No data returned from query")
