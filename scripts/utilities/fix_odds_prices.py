"""
Fix the odds prices - they're divided by 100 when they shouldn't be
"""
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="magnus",
    user="postgres",
    password="postgres123"
)

cursor = conn.cursor()

print("=" * 100)
print("FIXING ODDS PRICES")
print("=" * 100)

# The issue: stored prices are 0.35 when they should be 35
# The raw_data has correct values (35c) but yes_price/no_price columns are wrong

print("\n[FIX] Updating prices from raw_data...")
print("-" * 100)

# Update yes_price and no_price from raw_data
cursor.execute("""
    UPDATE kalshi_markets
    SET
        yes_price = (raw_data->>'last_price')::numeric,
        no_price = 100 - (raw_data->>'last_price')::numeric
    WHERE raw_data IS NOT NULL
    AND raw_data->>'last_price' IS NOT NULL
    AND raw_data->>'last_price' != 'null'
""")

rows_updated = cursor.rowcount
conn.commit()

print(f"[OK] Updated {rows_updated} markets with correct prices from raw_data")

# Verify the fix
print("\n[VERIFY] Checking sample prices after fix...")
print("-" * 100)

cursor.execute("""
    SELECT ticker, title, yes_price, no_price,
           raw_data->>'last_price' as raw_last_price
    FROM kalshi_markets
    WHERE ticker LIKE 'KXNCAAFGAME%'
    AND raw_data IS NOT NULL
    LIMIT 5
""")

print("Sample NCAA markets after fix:")
for row in cursor.fetchall():
    ticker, title, yes_p, no_p, raw_last = row
    print(f"\n{ticker}")
    print(f"  {title}")
    print(f"  yes_price: {yes_p}c (was 0.{int(float(raw_last)) if raw_last else 0}c)")
    print(f"  no_price: {no_p}c")
    print(f"  raw last_price: {raw_last}c")

# Show price distribution
print("\n[STATS] Price distribution after fix...")
print("-" * 100)

cursor.execute("""
    SELECT
        COUNT(*) as total,
        AVG(yes_price) as avg_yes,
        MIN(yes_price) as min_yes,
        MAX(yes_price) as max_yes
    FROM kalshi_markets
    WHERE yes_price IS NOT NULL
    AND ticker LIKE '%GAME%'
""")

row = cursor.fetchone()
print(f"Game markets: {row[0]} total")
print(f"Average yes_price: {row[1]:.1f}c")
print(f"Range: {row[2]:.1f}c to {row[3]:.1f}c")

cursor.close()
conn.close()

print("\n" + "=" * 100)
print("[OK] PRICES FIXED!")
print("=" * 100)
print("""
The issue was that prices were stored as decimals (0.35) instead of cents (35).
This caused the UI to display 0.35% instead of 35%.

Fixed by extracting correct prices from raw_data->>'last_price'.

Next step: Clear Streamlit cache and refresh to see correct odds!
""")
print("=" * 100)
