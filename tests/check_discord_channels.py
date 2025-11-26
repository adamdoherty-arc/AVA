"""Check Discord channels in the database"""
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

def check_channels():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'discord_channels'
                );
            """)
            table_exists = cur.fetchone()['exists']
            print(f"discord_channels table exists: {table_exists}")

            if table_exists:
                # Count channels
                cur.execute("SELECT COUNT(*) as count FROM discord_channels;")
                count = cur.fetchone()['count']
                print(f"\nTotal channels in database: {count}")

                # Get all channels
                cur.execute("""
                    SELECT
                        channel_id,
                        channel_name,
                        server_name,
                        last_sync,
                        created_at
                    FROM discord_channels
                    ORDER BY created_at DESC;
                """)
                channels = cur.fetchall()

                if channels:
                    print("\nChannels:")
                    print("-" * 80)
                    for ch in channels:
                        print(f"ID: {ch['channel_id']}")
                        print(f"  Name: {ch['channel_name']}")
                        print(f"  Server: {ch['server_name']}")
                        print(f"  Last Sync: {ch['last_sync']}")
                        print(f"  Created: {ch['created_at']}")
                        print()
                else:
                    print("\nNo channels found in table.")

                # Check messages count per channel
                cur.execute("""
                    SELECT
                        c.channel_id,
                        c.channel_name,
                        c.server_name,
                        c.last_sync,
                        COALESCE(COUNT(m.message_id), 0) as message_count
                    FROM discord_channels c
                    LEFT JOIN discord_messages m ON c.channel_id = m.channel_id
                    GROUP BY c.channel_id, c.channel_name, c.server_name, c.last_sync
                    ORDER BY c.last_sync DESC NULLS LAST;
                """)
                results = cur.fetchall()

                print("\nQuery results (same as page uses):")
                print("-" * 80)
                for r in results:
                    print(f"{r['server_name']} / {r['channel_name']}: {r['message_count']} messages")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_channels()
