"""
Check existing Discord messages
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
print("DISCORD MESSAGES STATUS")
print("=" * 80)

# Total messages
cur.execute("SELECT COUNT(*) FROM discord_messages")
total = cur.fetchone()[0]
print(f"\nTotal messages in database: {total}")

if total == 0:
    print("\n[!] No messages in database!")
    print("[!] You need to click the Sync button on the XTrade Messages page")
    print("[!] Go to 'Channel Management' tab -> Click '🔄 Sync' for each channel")
else:
    # Messages by channel
    print(f"\nMessages by channel:")
    cur.execute("""
        SELECT
            c.channel_name,
            c.server_name,
            c.channel_id,
            COUNT(m.message_id) as msg_count,
            MIN(m.timestamp) as oldest,
            MAX(m.timestamp) as newest
        FROM discord_channels c
        LEFT JOIN discord_messages m ON c.channel_id = m.channel_id
        GROUP BY c.channel_id, c.channel_name, c.server_name
        ORDER BY msg_count DESC
    """)

    for name, server, channel_id, count, oldest, newest in cur.fetchall():
        print(f"\n  {server} / {name}")
        print(f"    ID: {channel_id}")
        print(f"    Messages: {count:,}")
        if oldest and newest:
            print(f"    Date range: {oldest.date()} to {newest.date()}")

    # Message age
    cur.execute("""
        SELECT
            MIN(timestamp) as oldest,
            MAX(timestamp) as newest,
            NOW() - MAX(timestamp) as time_since_last
        FROM discord_messages
    """)
    oldest, newest, age = cur.fetchone()

    print(f"\n\nAll messages age:")
    print(f"  Oldest: {oldest}")
    print(f"  Newest: {newest}")
    print(f"  Time since last: {age}")

    # Messages in different time ranges
    print(f"\n\nMessages by time range:")
    ranges = [
        ("Last 24 hours", 24),
        ("Last 3 days", 72),
        ("Last 7 days", 168),
        ("Last 14 days", 336),
        ("Last 30 days", 720),
    ]

    for label, hours in ranges:
        cur.execute(f"""
            SELECT COUNT(*)
            FROM discord_messages
            WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
        """)
        count = cur.fetchone()[0]
        print(f"  {label:20}: {count:,}")

    # Recent message sample
    print(f"\n\nMost recent 5 messages:")
    cur.execute("""
        SELECT
            m.timestamp,
            m.author_name,
            LEFT(m.content, 60) as preview,
            c.channel_name
        FROM discord_messages m
        JOIN discord_channels c ON m.channel_id = c.channel_id
        ORDER BY m.timestamp DESC
        LIMIT 5
    """)

    for timestamp, author, preview, channel in cur.fetchall():
        print(f"\n  {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"    From: {author} in {channel}")
        print(f"    Preview: {preview}...")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("\n1. Go to XTrade Messages page in Streamlit")
print("2. Click 'Channel Management' tab")
print("3. For each channel, click '🔄 Sync' button")
print("4. Wait for sync to complete")
print("5. Messages will appear in tabs above")
print("\n" + "=" * 80)
