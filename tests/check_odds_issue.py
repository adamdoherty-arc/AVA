"""
Check the 54% odds issue
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
print("CHECKING ODDS DISPLAY ISSUE")
print("=" * 100)

# Check NCAA markets
print("\n1. NCAA MARKETS WITH PRICES:")
print("-" * 100)
cursor.execute("""
    SELECT ticker, title, home_team, away_team, yes_price, no_price
    FROM kalshi_markets
    WHERE ticker LIKE 'KXNCAAFGAME%'
    AND yes_price IS NOT NULL
    ORDER BY yes_price
    LIMIT 15
""")

for row in cursor.fetchall():
    ticker, title, home, away, yes_p, no_p = row
    print(f"\nTicker: {ticker}")
    print(f"Title: {title}")
    print(f"Home: {home} | Away: {away}")
    print(f"Yes Price: {yes_p}c | No Price: {no_p}c")

    # Convert to percentages
    yes_pct = float(yes_p) / 100 if yes_p else 0
    no_pct = float(no_p) / 100 if no_p else 0
    print(f"As %: Yes={yes_pct:.1f}% | No={no_pct:.1f}%")

# Check NFL markets
print("\n\n2. NFL MARKETS WITH PRICES:")
print("-" * 100)
cursor.execute("""
    SELECT ticker, title, home_team, away_team, yes_price, no_price
    FROM kalshi_markets
    WHERE ticker LIKE 'KXNFLGAME%'
    AND yes_price IS NOT NULL
    ORDER BY yes_price
    LIMIT 10
""")

for row in cursor.fetchall():
    ticker, title, home, away, yes_p, no_p = row
    print(f"\nTicker: {ticker}")
    print(f"Title: {title}")
    print(f"Home: {home} | Away: {away}")
    print(f"Yes Price: {yes_p}c | No Price: {no_p}c")

    # Convert to percentages
    yes_pct = float(yes_p) / 100 if yes_p else 0
    no_pct = float(no_p) / 100 if no_p else 0
    print(f"As %: Yes={yes_pct:.1f}% | No={no_pct:.1f}%")

# Check if all are the same
print("\n\n3. CHECKING FOR DUPLICATE PRICES:")
print("-" * 100)
cursor.execute("""
    SELECT yes_price, COUNT(*) as count
    FROM kalshi_markets
    WHERE ticker LIKE 'KXNCAAFGAME%'
    AND yes_price IS NOT NULL
    GROUP BY yes_price
    ORDER BY count DESC
    LIMIT 10
""")

print("Most common yes_price values in NCAA markets:")
for yes_p, count in cursor.fetchall():
    print(f"  {yes_p}c: {count} markets ({count/298*100:.1f}%)")

# Check the raw_data for accurate prices
print("\n\n4. CHECKING RAW_DATA FOR ACCURATE PRICES:")
print("-" * 100)
cursor.execute("""
    SELECT ticker, title,
           yes_price,
           raw_data->>'yes_ask' as raw_yes_ask,
           raw_data->>'yes_bid' as raw_yes_bid,
           raw_data->>'last_price' as raw_last_price
    FROM kalshi_markets
    WHERE ticker LIKE 'KXNCAAFGAME%'
    AND raw_data IS NOT NULL
    LIMIT 5
""")

print("Comparing stored prices vs raw_data prices:")
for row in cursor.fetchall():
    ticker, title, stored_yes, raw_yes_ask, raw_yes_bid, raw_last = row
    print(f"\n{ticker}: {title}")
    print(f"  Stored yes_price: {stored_yes}c")
    print(f"  Raw yes_ask: {raw_yes_ask}c")
    print(f"  Raw yes_bid: {raw_yes_bid}c")
    print(f"  Raw last_price: {raw_last}c")

cursor.close()
conn.close()

print("\n" + "=" * 100)
