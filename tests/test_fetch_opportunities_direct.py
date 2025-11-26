"""
Test fetch_opportunities function directly to see what's failing
"""
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host='localhost',
        port='5432',
        database='magnus',
        user='postgres',
        password=os.getenv('DB_PASSWORD')
    )

def fetch_opportunities(dte_min, dte_max, delta_min=-0.4, delta_max=-0.2, min_premium=0,
                        min_stock_price=0, max_stock_price=10000):
    """
    EXACT copy of fetch_opportunities from premium_scanner_page.py
    """
    conn = None
    try:
        print(f"\n1. Creating connection...")
        conn = get_connection()
        cur = conn.cursor()
        print("   SUCCESS")

        query = '''
            SELECT DISTINCT ON (sp.symbol)
                sp.symbol,
                sd.current_price as stock_price,
                sp.strike_price,
                sp.premium,
                sp.dte,
                sp.premium_pct,
                sp.annual_return,
                sp.delta,
                sp.prob_profit,
                sp.implied_volatility,
                sp.volume,
                sp.open_interest,
                sp.strike_type,
                sp.expiration_date,
                sp.bid,
                sp.ask,
                s.company_name,
                s.sector
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

        print(f"\n2. Executing query with parameters:")
        print(f"   dte_min={dte_min}, dte_max={dte_max}")
        print(f"   delta_min={delta_min}, delta_max={delta_max}")
        print(f"   min_premium={min_premium}")
        print(f"   min_stock_price={min_stock_price}, max_stock_price={max_stock_price}")

        cur.execute(query, (dte_min, dte_max, min_premium, delta_min, delta_max,
                            min_stock_price, max_stock_price))
        print("   SUCCESS")

        print(f"\n3. Fetching results...")
        columns = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        cur.close()
        print(f"   SUCCESS - {len(results)} rows fetched")

        print(f"\n4. Creating DataFrame...")
        df = pd.DataFrame(results, columns=columns)
        print(f"   SUCCESS - DataFrame shape: {df.shape}")

        # THIS IS THE LINE THAT MIGHT BE FAILING
        print(f"\n5. Calculating additional metrics...")
        if not df.empty:
            print(f"   DataFrame is NOT empty, calculating metrics...")

            # LINE 97 - PROBLEMATIC LINE
            print(f"   5a. Calculating weekly_return...")
            try:
                df['weekly_return'] = df['premium_pct'] if dte_min < 15 else df['premium_pct']
                print(f"      SUCCESS - but this line is weird (always same value)")
            except Exception as e:
                print(f"      FAILED: {e}")
                traceback.print_exc()
                raise

            print(f"   5b. Calculating annualized_52wk...")
            try:
                df['annualized_52wk'] = df['premium_pct'] * (365 / df['dte'])
                print(f"      SUCCESS")
            except Exception as e:
                print(f"      FAILED: {e}")
                traceback.print_exc()
                raise

            print(f"   5c. Calculating premium_per_day...")
            try:
                df['premium_per_day'] = df['premium'] / df['dte']
                print(f"      SUCCESS")
            except Exception as e:
                print(f"      FAILED: {e}")
                traceback.print_exc()
                raise

            print(f"   5d. Calculating bid_ask_spread...")
            try:
                df['bid_ask_spread'] = df.apply(
                    lambda x: (x['ask'] - x['bid']) if pd.notna(x['bid']) and pd.notna(x['ask']) else 0,
                    axis=1
                )
                print(f"      SUCCESS")
            except Exception as e:
                print(f"      FAILED: {e}")
                traceback.print_exc()
                raise

        print(f"\n6. Returning DataFrame...")
        print(f"   Final shape: {df.shape}")
        return df

    except Exception as e:
        print(f"\nEXCEPTION CAUGHT: {e}")
        traceback.print_exc()
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
            print("\n7. Connection closed")

# Test with the EXACT parameters from the screenshot
print("=" * 80)
print("TESTING FETCH_OPPORTUNITIES WITH DEFAULT PARAMETERS")
print("=" * 80)

df = fetch_opportunities(
    dte_min=5,
    dte_max=9,
    delta_min=-1.0,  # From screenshot
    delta_max=0.0,   # From screenshot
    min_premium=0.0,
    min_stock_price=0.0,
    max_stock_price=10000.0
)

print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)
if df.empty:
    print("FAILED: DataFrame is EMPTY")
else:
    print(f"SUCCESS: DataFrame has {len(df)} rows")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3)[['symbol', 'stock_price', 'premium', 'delta', 'dte']])
print("=" * 80)
