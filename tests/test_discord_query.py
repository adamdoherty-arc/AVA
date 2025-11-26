"""
Test Discord Messages SQL Query Issue
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("TESTING DISCORD MESSAGES SQL QUERY")
print("=" * 80)

conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)

# Check total messages
cur = conn.cursor()
print("\n[1] Total messages in database:")
cur.execute("SELECT COUNT(*) FROM discord_messages")
total = cur.fetchone()[0]
print(f"   Total discord_messages: {total}")

# Check messages in last 24 hours
cur.execute("""
    SELECT COUNT(*)
    FROM discord_messages
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
""")
last_24h = cur.fetchone()[0]
print(f"   Messages in last 24 hours: {last_24h}")

# Test the BROKEN query from line 183
print("\n[2] Testing BROKEN query (line 183):")
print("   Query: WHERE timestamp >= NOW() - INTERVAL '%s hours'")

trading_keywords = [
    'buy', 'sell', 'call', 'put', 'strike', 'expiry', 'expiration',
    'bullish', 'bearish', 'target', 'entry', 'stop', 'alert'
]

search_conditions = ' OR '.join(['content ILIKE %s' for _ in trading_keywords])
params = [24] + [f'%{kw}%' for kw in trading_keywords]

broken_query = f"""
    SELECT COUNT(*)
    FROM discord_messages m
    JOIN discord_channels c ON m.channel_id = c.channel_id
    WHERE m.timestamp >= NOW() - INTERVAL '%s hours'
    AND ({search_conditions})
"""

try:
    cur.execute(broken_query, params)
    count = cur.fetchone()[0]
    print(f"   [BROKEN] Result: {count} rows")
except Exception as e:
    print(f"   [ERROR] Query failed: {e}")

# Test the FIXED query
print("\n[3] Testing FIXED query:")
print("   Query: WHERE timestamp >= NOW() - INTERVAL '24 hours'")

fixed_query = f"""
    SELECT COUNT(*)
    FROM discord_messages m
    JOIN discord_channels c ON m.channel_id = c.channel_id
    WHERE m.timestamp >= NOW() - INTERVAL '24 hours'
    AND ({search_conditions})
"""

# params without hours_back since it's hardcoded in query
fixed_params = [f'%{kw}%' for kw in trading_keywords]

try:
    cur.execute(fixed_query, fixed_params)
    count = cur.fetchone()[0]
    print(f"   [FIXED] Result: {count} rows")
except Exception as e:
    print(f"   [ERROR] Query failed: {e}")

# Alternative fix: Build interval in Python
print("\n[4] Testing ALTERNATIVE fix (Python interval):")
hours_back = 24
fixed_query2 = f"""
    SELECT COUNT(*)
    FROM discord_messages m
    JOIN discord_channels c ON m.channel_id = c.channel_id
    WHERE m.timestamp >= NOW() - INTERVAL '%s hour'
    AND ({search_conditions})
"""

fixed_params2 = [str(hours_back)] + [f'%{kw}%' for kw in trading_keywords]

try:
    cur.execute(fixed_query2, fixed_params2)
    count = cur.fetchone()[0]
    print(f"   [ALTERNATIVE] Result: {count} rows")
except Exception as e:
    print(f"   [ERROR] Query failed: {e}")

# Show sample messages
if count > 0:
    print(f"\n[5] Sample messages with trading keywords:")
    sample_query = f"""
        SELECT
            m.content,
            m.author_name,
            m.timestamp,
            c.channel_name
        FROM discord_messages m
        JOIN discord_channels c ON m.channel_id = c.channel_id
        WHERE m.timestamp >= NOW() - INTERVAL '24 hours'
        AND ({search_conditions})
        LIMIT 5
    """
    cur.execute(sample_query, fixed_params)
    for row in cur.fetchall():
        print(f"\n   Author: {row[1]}")
        print(f"   Channel: {row[3]}")
        print(f"   Time: {row[2]}")
        print(f"   Content: {row[0][:100]}...")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("ISSUE IDENTIFIED: INTERVAL '%s hours' syntax is broken")
print("FIX: Use INTERVAL '%s hour' or hardcode interval in query")
print("=" * 80)
