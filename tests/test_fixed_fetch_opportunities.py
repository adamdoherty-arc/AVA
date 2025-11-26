"""
Test the FIXED fetch_opportunities function
"""
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host='localhost',
        port='5432',
        database='magnus',
        user='postgres',
        password=os.getenv('DB_PASSWORD')
    )

def fetch_opportunities_FIXED(dte_min, dte_max, delta_min=-0.4, delta_max=-0.2, min_premium=0,
                        min_stock_price=0, max_stock_price=10000):
    """
    FIXED version with Decimal to float conversion
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

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

        cur.execute(query, (dte_min, dte_max, min_premium, delta_min, delta_max,
                            min_stock_price, max_stock_price))

        columns = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        cur.close()

        df = pd.DataFrame(results, columns=columns)

        # Calculate additional metrics
        if not df.empty:
            # Convert Decimal columns to float for calculations
            numeric_cols = ['premium_pct', 'annual_return', 'premium', 'dte', 'delta',
                          'prob_profit', 'implied_volatility', 'volume', 'open_interest',
                          'stock_price', 'strike_price', 'bid', 'ask']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Calculate metrics
            df['annualized_52wk'] = df['premium_pct'] * (365 / df['dte'])
            df['premium_per_day'] = df['premium'] / df['dte']
            df['bid_ask_spread'] = df.apply(
                lambda x: (x['ask'] - x['bid']) if pd.notna(x['bid']) and pd.notna(x['ask']) else 0,
                axis=1
            )

        return df
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# Test with default parameters
print("=" * 80)
print("TESTING FIXED FETCH_OPPORTUNITIES")
print("=" * 80)

df = fetch_opportunities_FIXED(
    dte_min=5,
    dte_max=9,
    delta_min=-1.0,
    delta_max=0.0,
    min_premium=0.0,
    min_stock_price=0.0,
    max_stock_price=10000.0
)

print(f"\nResults: {len(df)} rows")

if not df.empty:
    print("\nSUCCESS! DataFrame created with all calculated columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    display_cols = ['symbol', 'stock_price', 'premium', 'delta', 'dte', 'premium_pct', 'annualized_52wk']
    print(df[display_cols].head())
    print("\nColumn types:")
    print(df[display_cols].dtypes)
else:
    print("\nFAILED: DataFrame is empty")

print("=" * 80)
