"""
Discord Message Sync
Uses DiscordChatExporter to pull messages from Discord channels
"""

import os
import json
import subprocess
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DiscordMessageSync:
    """Sync Discord messages using DiscordChatExporter"""

    def __init__(self) -> None:
        self.token = os.getenv('DISCORD_USER_TOKEN')
        self.exporter_path = os.getenv('DISCORD_EXPORTER_PATH', 'DiscordChatExporter.Cli.exe')
        self.export_dir = Path('data/discord_exports')
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Database connection
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'magnus')  # Fixed: was 'trading'
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', '')

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def export_channel(self, channel_id: str, days_back: int = 7) -> Path:
        """
        Export Discord channel messages using DiscordChatExporter

        Args:
            channel_id: Discord channel ID
            days_back: How many days back to fetch

        Returns:
            Path to exported JSON file
        """
        if not self.token:
            raise ValueError("DISCORD_USER_TOKEN not set in .env")

        # Calculate date range
        after_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        # Output file
        output_file = self.export_dir / f"channel_{channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Build command
        cmd = [
            self.exporter_path,
            'export',
            '-t', self.token,
            '-c', channel_id,
            '-f', 'Json',
            '-o', str(output_file),
            '--after', after_date
        ]

        logger.info(f"Exporting channel {channel_id}...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Export successful: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Export failed: {e.stderr}")
            raise

    def import_messages(self, json_file: Path) -> int:
        """
        Import messages from exported JSON file into database

        Args:
            json_file: Path to DiscordChatExporter JSON export

        Returns:
            Number of messages imported
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract channel info
        channel_id = data.get('channel', {}).get('id')
        channel_name = data.get('channel', {}).get('name')
        server_name = data.get('guild', {}).get('name')
        server_id = data.get('guild', {}).get('id')

        conn = None
        cur = None
        imported = 0

        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Insert/update channel
            cur.execute("""
                INSERT INTO discord_channels (channel_id, channel_name, server_name, server_id, last_sync)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (channel_id) DO UPDATE SET
                    channel_name = EXCLUDED.channel_name,
                    server_name = EXCLUDED.server_name,
                    last_sync = EXCLUDED.last_sync
            """, (channel_id, channel_name, server_name, server_id, datetime.now()))

            # Import messages
            messages = data.get('messages', [])

            for msg in messages:
                try:
                    message_id = msg.get('id')
                    author = msg.get('author', {})
                    author_id = author.get('id')
                    author_name = author.get('name')
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp')
                    edited_timestamp = msg.get('timestampEdited')

                    # Skip messages that are only @everyone with no other content
                    if content.strip() == '@everyone':
                        continue

                    # Parse timestamp
                    if timestamp:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if edited_timestamp:
                        edited_timestamp = datetime.fromisoformat(edited_timestamp.replace('Z', '+00:00'))

                    # Extract attachments, embeds, reactions
                    attachments = json.dumps(msg.get('attachments', []))
                    embeds = json.dumps(msg.get('embeds', []))
                    reactions = json.dumps(msg.get('reactions', []))
                    mentions = json.dumps(msg.get('mentions', []))

                    cur.execute("""
                        INSERT INTO discord_messages (
                            message_id, channel_id, author_id, author_name,
                            content, timestamp, edited_timestamp,
                            attachments, embeds, reactions, mentions, raw_data
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (message_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            edited_timestamp = EXCLUDED.edited_timestamp,
                            reactions = EXCLUDED.reactions,
                            raw_data = EXCLUDED.raw_data
                    """, (
                        message_id, channel_id, author_id, author_name,
                        content, timestamp, edited_timestamp,
                        attachments, embeds, reactions, mentions,
                        json.dumps(msg)
                    ))

                    imported += 1

                except Exception as e:
                    logger.error(f"Error importing message {msg.get('id')}: {e}")
                    continue

            conn.commit()
            logger.info(f"Imported {imported} messages from {channel_name}")

            return imported

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error importing messages: {e}")
            raise

        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def sync_channel(self, channel_id: str, days_back: int = 7) -> int:
        """
        Full sync: export and import channel messages

        Args:
            channel_id: Discord channel ID
            days_back: How many days back to sync

        Returns:
            Number of messages imported
        """
        # Export
        json_file = self.export_channel(channel_id, days_back)

        # Import
        imported = self.import_messages(json_file)

        # Clean up export file (optional)
        # json_file.unlink()

        return imported

    def get_recent_messages(self, channel_id: str = None, limit: int = 100):
        """Get recent messages from database"""
        conn = None
        cur = None

        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            if channel_id:
                cur.execute("""
                    SELECT * FROM discord_recent_messages
                    WHERE channel_id = %s
                    LIMIT %s
                """, (channel_id, limit))
            else:
                cur.execute("""
                    SELECT * FROM discord_recent_messages
                    LIMIT %s
                """, (limit,))

            return cur.fetchall()

        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python discord_message_sync.py <channel_id> [days_back]")
        sys.exit(1)

    channel_id = sys.argv[1]
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    sync = DiscordMessageSync()
    count = sync.sync_channel(channel_id, days_back)

    print(f"Synced {count} messages from channel {channel_id}")
