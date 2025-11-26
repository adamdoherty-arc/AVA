"""
Simple check - no Unicode issues
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

print("DISCORD STATUS")
print("=" * 60)

# Total
cur.execute("SELECT COUNT(*) FROM discord_messages")
total = cur.fetchone()[0]
print(f"\nTotal messages: {total}")

# By channel ID
cur.execute("""
    SELECT channel_id, COUNT(*) as count
    FROM discord_messages
    GROUP BY channel_id
    ORDER BY count DESC
""")
print(f"\nMessages by channel ID:")
for channel_id, count in cur.fetchall():
    print(f"  {channel_id}: {count} messages")

# Last 7 days
cur.execute("""
    SELECT COUNT(*)
    FROM discord_messages
    WHERE timestamp >= NOW() - INTERVAL '168 hours'
""")
last_7d = cur.fetchone()[0]
print(f"\nLast 7 days (168 hours): {last_7d} messages")

# Last 30 days
cur.execute("""
    SELECT COUNT(*)
    FROM discord_messages
    WHERE timestamp >= NOW() - INTERVAL '720 hours'
""")
last_30d = cur.fetchone()[0]
print(f"Last 30 days (720 hours): {last_30d} messages")

# Newest message
cur.execute("SELECT MAX(timestamp) FROM discord_messages")
newest = cur.fetchone()[0]
print(f"\nNewest message: {newest}")

# All channels
cur.execute("SELECT channel_id, channel_name FROM discord_channels ORDER BY channel_id")
print(f"\nAll channels in system:")
for cid, name in cur.fetchall():
    print(f"  {cid}: {name}")

cur.close()
conn.close()

print("\n" + "=" * 60)
print("DIAGNOSIS:")
if last_7d == 0:
    print("Messages are older than 7 days!")
    print("UI slider needs to be set to 30+ days")
elif last_7d == total:
    print("All messages in last 7 days - should show!")
else:
    print(f"{last_7d} messages in last 7 days")
    print(f"{total - last_7d} messages older than 7 days")
print("=" * 60)
