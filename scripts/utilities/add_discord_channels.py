"""
Add Discord channels directly to database
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Channels to add
channels = [
    (990343144241500232, "XTrades Channel 1", "XTrades Server"),
    (1360173424495824896, "XTrades Channel 2", "XTrades Server"),
    (1307115332015755284, "XTrades Channel 3", "XTrades Server"),
]

print("=" * 80)
print("ADDING DISCORD CHANNELS")
print("=" * 80)

conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

for channel_id, channel_name, server_name in channels:
    print(f"\n[+] Adding: {server_name} / {channel_name}")
    print(f"    ID: {channel_id}")

    cur.execute("""
        INSERT INTO discord_channels (channel_id, channel_name, server_name, description, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (channel_id)
        DO UPDATE SET
            channel_name = EXCLUDED.channel_name,
            server_name = EXCLUDED.server_name
    """, (channel_id, channel_name, server_name, "XTrades trading signals"))

    # Check if this channel has messages
    cur.execute("""
        SELECT COUNT(*) FROM discord_messages WHERE channel_id = %s
    """, (channel_id,))
    msg_count = cur.fetchone()[0]
    print(f"    Messages in DB: {msg_count}")

conn.commit()

# Show all channels
print("\n" + "=" * 80)
print("ALL CHANNELS IN DATABASE")
print("=" * 80)

cur.execute("""
    SELECT
        c.channel_id,
        c.channel_name,
        c.server_name,
        c.last_sync,
        (SELECT COUNT(*) FROM discord_messages WHERE channel_id = c.channel_id) as msg_count
    FROM discord_channels c
    ORDER BY c.created_at DESC
""")

channels = cur.fetchall()
print(f"\nTotal channels: {len(channels)}\n")

for channel_id, name, server, last_sync, msg_count in channels:
    print(f"Channel: {server} / {name}")
    print(f"  ID: {channel_id}")
    print(f"  Messages: {msg_count:,}")
    if last_sync:
        print(f"  Last sync: {last_sync.strftime('%Y-%m-%d %H:%M')}")
    else:
        print(f"  Last sync: Never")
    print()

# Check all messages
cur.execute("SELECT COUNT(*) FROM discord_messages")
total_msgs = cur.fetchone()[0]

cur.execute("""
    SELECT COUNT(*)
    FROM discord_messages
    WHERE timestamp >= NOW() - INTERVAL '168 hours'
""")
last_7_days = cur.fetchone()[0]

print("=" * 80)
print(f"Total messages in database: {total_msgs:,}")
print(f"Messages in last 7 days: {last_7_days:,}")
print("=" * 80)

# If we have messages but they're old, show when they're from
if total_msgs > 0:
    cur.execute("""
        SELECT
            MIN(timestamp) as oldest,
            MAX(timestamp) as newest,
            NOW() - MAX(timestamp) as age
        FROM discord_messages
    """)
    oldest, newest, age = cur.fetchone()
    print(f"\nMessage age:")
    print(f"  Oldest: {oldest}")
    print(f"  Newest: {newest}")
    print(f"  Time since last: {age}")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("CHANNELS ADDED - Now refresh XTrade Messages page")
print("=" * 80)
