"""
Quick Discord Sync Helper
Uses your existing DISCORD_USER_TOKEN from .env
"""

import sys
from dotenv import load_dotenv
from src.discord_message_sync import DiscordMessageSync

load_dotenv()

def main():
    """Sync Discord channel messages"""

    if len(sys.argv) < 2:
        print("\nDiscord Message Sync")
        print("=" * 50)
        print("\nUsage:")
        print("  python sync_discord.py <channel_id> [days_back]")
        print("\nExamples:")
        print("  python sync_discord.py 1234567890 7    # Last 7 days")
        print("  python sync_discord.py 1234567890      # Last 7 days (default)")
        print("\nYour current channels:")

        # Show existing channels
        sync = DiscordMessageSync()
        try:
            conn = sync.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT channel_id, channel_name, server_name, last_sync FROM discord_channels ORDER BY last_sync DESC")
            channels = cur.fetchall()

            if channels:
                print("\n+---------------------+--------------------------+-------------------------+")
                print("| Channel ID          | Channel Name             | Last Sync               |")
                print("+---------------------+--------------------------+-------------------------+")
                for channel_id, name, server, last_sync in channels:
                    last_sync_str = last_sync.strftime("%Y-%m-%d %H:%M") if last_sync else "Never"
                    # Encode safely for Windows console
                    safe_name = name[:24].encode('ascii', 'replace').decode('ascii') if name else ""
                    print(f"| {channel_id:<19} | {safe_name:<24} | {last_sync_str:<23} |")
                print("+---------------------+--------------------------+-------------------------+")
            else:
                print("\n  No channels synced yet.")

            cur.close()
            conn.close()
        except Exception as e:
            print(f"\n  Error checking channels: {e}")

        print("\n[OK] Your DISCORD_USER_TOKEN is configured in .env")
        print("[OK] DiscordChatExporter path is set")
        print("\n")
        sys.exit(1)

    channel_id = sys.argv[1]
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    print(f"\nSyncing Discord Channel: {channel_id}")
    print(f"Days back: {days_back}")
    print("=" * 50)

    try:
        sync = DiscordMessageSync()

        print("\nStep 1/2: Exporting messages from Discord...")
        json_file = sync.export_channel(channel_id, days_back)
        print(f"[OK] Exported to: {json_file}")

        print("\nStep 2/2: Importing messages to database...")
        count = sync.import_messages(json_file)
        print(f"[OK] Imported {count} messages")

        print("\nSync complete!")
        print(f"{count} messages now available in XTrade Messages page")
        print("")

    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Make sure DISCORD_USER_TOKEN is set in .env")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
