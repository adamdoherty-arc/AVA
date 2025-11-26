"""Clear Streamlit cache and test Discord channels query"""
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import shutil

load_dotenv()

def clear_streamlit_cache():
    """Clear all Streamlit cache directories"""
    cache_dirs = [
        '.streamlit/cache',
        os.path.expanduser('~/.streamlit/cache'),
        'C:/Users/New User/AppData/Local/Streamlit/Cache'
    ]

    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"[OK] Cleared cache: {cache_dir}")
            except Exception as e:
                print(f"[WARN] Could not clear {cache_dir}: {e}")

    print("\n[OK] Streamlit cache cleared!")

def test_query():
    """Test the channels query"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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
                ORDER BY c.last_sync DESC NULLS LAST
            """)
            channels = cur.fetchall()

            print(f"\n[OK] Query successful! Found {len(channels)} channels:")
            print("-" * 80)
            for ch in channels:
                try:
                    print(f"{ch['server_name']} / {ch['channel_name']}")
                    print(f"  Channel ID: {ch['channel_id']}")
                    print(f"  Messages: {ch['message_count']}")
                    print(f"  Last Sync: {ch['last_sync'] or 'Never'}")
                    print()
                except UnicodeEncodeError:
                    print(f"Channel ID: {ch['channel_id']} (contains special characters)")
                    print(f"  Messages: {ch['message_count']}")
                    print()

        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Discord Messages Page - Cache Fix\n")
    print("=" * 80)

    # Clear cache
    clear_streamlit_cache()

    # Test query
    print("\n" + "=" * 80)
    print("Testing database query...")
    print("=" * 80)
    test_query()

    print("\n" + "=" * 80)
    print("[OK] Fix complete! Please restart the Streamlit app:")
    print("   streamlit run dashboard.py")
    print("=" * 80)
