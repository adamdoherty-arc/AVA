"""
Auto-sync all tracked Discord channels
Designed to run on a schedule (hourly via Task Scheduler)
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from src.discord_message_sync import DiscordMessageSync

# Setup logging
log_file = os.path.join(os.path.dirname(__file__), 'discord_sync.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


def sync_all_channels():
    """Sync all tracked Discord channels"""

    logger.info("=" * 70)
    logger.info("Discord Auto-Sync Started")
    logger.info("=" * 70)

    try:
        sync = DiscordMessageSync()

        # Get all tracked channels
        conn = sync.get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT channel_id, channel_name, server_name, last_sync
            FROM discord_channels
            ORDER BY channel_id
        """)
        channels = cur.fetchall()
        cur.close()
        conn.close()

        if not channels:
            logger.warning("No channels configured to sync")
            return

        logger.info(f"Found {len(channels)} channel(s) to sync")

        total_messages = 0
        successful_syncs = 0
        failed_syncs = 0

        # Sync each channel
        for channel_id, channel_name, server_name, last_sync in channels:
            logger.info("-" * 70)
            logger.info(f"Syncing: {server_name} / {channel_name} (ID: {channel_id})")

            try:
                # Export messages (last 7 days)
                logger.info("  [1/2] Exporting messages from Discord...")
                json_file = sync.export_channel(str(channel_id), days_back=7)
                logger.info(f"  [OK] Exported to: {json_file}")

                # Import to database
                logger.info("  [2/2] Importing messages to database...")
                count = sync.import_messages(json_file)
                logger.info(f"  [OK] Imported {count} messages")

                total_messages += count
                successful_syncs += 1

            except Exception as e:
                logger.error(f"  [ERROR] Failed to sync {channel_name}: {e}")
                failed_syncs += 1
                continue

        # Summary
        logger.info("=" * 70)
        logger.info("Sync Summary:")
        logger.info(f"  Total channels: {len(channels)}")
        logger.info(f"  Successful: {successful_syncs}")
        logger.info(f"  Failed: {failed_syncs}")
        logger.info(f"  Total messages synced: {total_messages}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Fatal error during sync: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        sync_all_channels()
        logger.info("Sync completed successfully")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("Sync interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
