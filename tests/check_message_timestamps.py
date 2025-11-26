"""
Check Discord message timestamps
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

print("=" * 80)
print("DISCORD MESSAGE TIMESTAMPS")
print("=" * 80)

# Get message age distribution
cur.execute("""
    SELECT
        COUNT(*) as total_messages,
        MIN(timestamp) as oldest_message,
        MAX(timestamp) as newest_message,
        NOW() - MAX(timestamp) as time_since_last
    FROM discord_messages
""")
result = cur.fetchone()

print(f"\n[1] Message Age Summary:")
print(f"   Total messages: {result[0]}")
print(f"   Oldest message: {result[1]}")
print(f"   Newest message: {result[2]}")
print(f"   Time since last message: {result[3]}")

# Messages by hour range
print(f"\n[2] Messages by time range:")
time_ranges = [
    ("Last hour", 1),
    ("Last 6 hours", 6),
    ("Last 12 hours", 12),
    ("Last 24 hours", 24),
    ("Last 3 days", 72),
    ("Last 7 days", 168),
    ("Last 30 days", 720),
]

for label, hours in time_ranges:
    cur.execute(f"""
        SELECT COUNT(*)
        FROM discord_messages
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
    """)
    count = cur.fetchone()[0]
    print(f"   {label:20}: {count:3} messages")

# Check last sync time from channels table
cur.execute("""
    SELECT
        channel_name,
        server_name,
        last_sync,
        NOW() - last_sync as time_since_sync
    FROM discord_channels
    ORDER BY last_sync DESC NULLS LAST
""")

print(f"\n[3] Channel Sync Status:")
for row in cur.fetchall():
    if row[2]:
        print(f"   {row[1]} / {row[0]}")
        print(f"     Last sync: {row[2]}")
        print(f"     Time since: {row[3]}")
    else:
        print(f"   {row[1]} / {row[0]}: Never synced")

# Sample messages
cur.execute("""
    SELECT
        content,
        author_name,
        timestamp
    FROM discord_messages
    ORDER BY timestamp DESC
    LIMIT 5
""")

print(f"\n[4] Most Recent 5 Messages:")
for row in cur.fetchall():
    print(f"\n   Time: {row[2]}")
    print(f"   Author: {row[1]}")
    print(f"   Content: {row[0][:80]}...")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("If no messages in last 24 hours, you need to run sync again!")
print("Command: python sync_discord.py CHANNEL_ID 7")
print("=" * 80)
