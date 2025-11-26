"""
Quick script to add a Discord channel to tracking
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def add_channel(channel_id: int, channel_name: str = None, server_name: str = None, description: str = None):
    """Add a Discord channel to tracking"""

    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        cur = conn.cursor()

        # Use defaults if not provided
        if not channel_name:
            channel_name = f"channel_{channel_id}"
        if not server_name:
            server_name = "Discord Server"

        # Insert or update channel
        cur.execute("""
            INSERT INTO discord_channels (channel_id, channel_name, server_name, description, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (channel_id)
            DO UPDATE SET
                channel_name = EXCLUDED.channel_name,
                server_name = EXCLUDED.server_name,
                description = EXCLUDED.description
        """, (channel_id, channel_name, server_name, description))

        conn.commit()

        print(f"[OK] Channel added successfully!")
        print(f"     Channel ID: {channel_id}")
        print(f"     Name: {channel_name}")
        print(f"     Server: {server_name}")

        cur.close()
        conn.close()

        return True

    except Exception as e:
        print(f"[ERROR] Failed to add channel: {e}")
        return False


if __name__ == "__main__":
    # Add the requested channel
    channel_id = 991515360509571233

    print("=" * 70)
    print("  Adding Discord Channel to Magnus Database")
    print("=" * 70)
    print()

    # Prompt for details
    print(f"Channel ID: {channel_id}")
    channel_name = input("Channel name (press Enter for default): ").strip() or f"channel_{channel_id}"
    server_name = input("Server name (press Enter for 'Discord Server'): ").strip() or "Discord Server"
    description = input("Description (optional): ").strip() or None

    print()
    print("[INFO] Adding channel to database...")

    if add_channel(channel_id, channel_name, server_name, description):
        print()
        print("[SUCCESS] Channel added to tracking!")
        print()
        print("Next steps:")
        print(f"  1. Sync messages: python sync_discord.py {channel_id} 7")
        print(f"  2. View in dashboard: XTrade Messages > Messages tab")
        print()
    else:
        print()
        print("[FAILED] Could not add channel")
        sys.exit(1)
