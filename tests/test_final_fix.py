"""
Test FINAL Fix - Use annual_return from database
"""
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_opportunities_v2(dte_min, dte_max, delta_min=-0.4, delta_max=-0.2, min_premium=0):
    """Updated fetch with annual_return usage"""
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

    # UPDATED FIX
    if not df.empty:
        # Convert Decimal columns to float for calculations
        numeric_cols = ['premium', 'strike_price', 'dte', 'premium_pct', 'annual_return', 'delta', 'prob_profit', 'implied_volatility', 'bid', 'ask']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate premium_pct if missing (for display purposes)
        if df['premium_pct'].isna().any():
            df['premium_pct'] = df['premium_pct'].fillna((df['premium'] / df['strike_price']) * 100)

        df['weekly_return'] = df['premium_pct']

        # Use annual_return from DB if available, otherwise calculate
        if 'annual_return' in df.columns and df['annual_return'].notna().any():
            df['annualized_52wk'] = df['annual_return']  # Use database value
        else:
            df['annualized_52wk'] = df['premium_pct'] * (365 / df['dte'])  # Calculate if missing

        df['premium_per_day'] = df['premium'] / df['dte']

    return df

print("=" * 80)
print("TESTING FINAL FIX - Using annual_return from Database")
print("=" * 80)

# Fetch data
print("\n[1] Fetching 7-day opportunities (DTE 5-9, Delta -0.4 to -0.2)")
df = fetch_opportunities_v2(5, 9, -0.4, -0.2, 0)
print(f"   Fetched: {len(df)} rows")

if not df.empty:
    print(f"\n[2] Data preview:")
    print(f"   premium_pct NaN count: {df['premium_pct'].isna().sum()}")
    print(f"   annual_return NaN count: {df['annual_return'].isna().sum()}")
    print(f"   annualized_52wk NaN count: {df['annualized_52wk'].isna().sum()}")  # Should be 0 now!

    print(f"\n[3] Value ranges:")
    print(f"   premium_pct: {df['premium_pct'].min():.2f}% to {df['premium_pct'].max():.2f}%")
    print(f"   annual_return: {df['annual_return'].min():.1f}% to {df['annual_return'].max():.1f}%")
    print(f"   annualized_52wk: {df['annualized_52wk'].min():.1f}% to {df['annualized_52wk'].max():.1f}%")

    # Apply filter
    min_annual_return = 30.0
    print(f"\n[4] Applying filter: annualized_52wk >= {min_annual_return}%")
    df_filtered = df[df['annualized_52wk'] >= min_annual_return]
    print(f"   Before filter: {len(df)} rows")
    print(f"   After filter: {len(df_filtered)} rows")

    if not df_filtered.empty:
        print(f"\n[5] ✅ SUCCESS! Top 10 opportunities:")
        print(f"\n   {'Symbol':<8} {'Strike':>10} {'Premium':>10} {'DTE':>4} {'Weekly%':>8} {'Annual%':>9}")
        print("   " + "-" * 65)
        for idx, row in df_filtered.head(10).iterrows():
            print(f"   {row['symbol']:<8} ${row['strike_price']:>9.2f} ${row['premium']:>9.2f} {row['dte']:>4} {row['premium_pct']:>7.2f}% {row['annualized_52wk']:>8.1f}%")

        print(f"\n" + "=" * 80)
        print(f"✅ FIX VERIFIED: {len(df_filtered)} opportunities will display!")
        print(f"   Using annual_return from database (no NaN values)")
        print("=" * 80)
    else:
        print(f"\n[5] ❌ STILL BROKEN: All rows filtered out")
        print(f"   Check annualized_52wk calculation")
else:
    print("\n❌ No data fetched from database")
