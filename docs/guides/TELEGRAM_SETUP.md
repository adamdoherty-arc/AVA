# Telegram Bot Setup Guide

Complete guide to set up Telegram notifications for the Magnus Wheel Strategy Trading Dashboard.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Features](#advanced-features)

---

## Overview

The Telegram notification system provides real-time alerts for your Xtrades trading activity directly to your Telegram app. You'll receive notifications for:

- New trade alerts from followed profiles
- Trade updates (price changes, status updates)
- Trade closures with P&L information
- Sync errors and system alerts
- Daily trading summaries

---

## Prerequisites

1. **Telegram Account**: You need an active Telegram account
2. **Python Environment**: Python 3.7+ with virtual environment activated
3. **Required Package**: `python-telegram-bot` library

---

## Step-by-Step Setup

### Step 1: Create Your Telegram Bot

1. **Open Telegram** on your phone or desktop

2. **Search for BotFather**
   - In the Telegram search bar, type `@BotFather`
   - Start a chat with the official BotFather bot (verified with a blue checkmark)

3. **Create a New Bot**
   - Send the command: `/newbot`
   - BotFather will ask you to choose a name for your bot
   - Example: `Magnus Trading Alerts`

4. **Choose a Username**
   - BotFather will ask for a username (must end in 'bot')
   - Example: `magnus_trading_alerts_bot` or `YourName_trades_bot`
   - Must be unique across all Telegram

5. **Save Your Bot Token**
   - BotFather will provide a token that looks like this:
   ```
   123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
   ```
   - **IMPORTANT**: Keep this token secure! Anyone with this token can control your bot
   - Copy this token - you'll need it in Step 3

### Step 2: Get Your Chat ID

You need your personal Chat ID to receive messages.

**Method 1: Using @userinfobot (Easiest)**

1. Search for `@userinfobot` in Telegram
2. Start a chat and send any message
3. The bot will reply with your user information
4. Copy your Chat ID (it will be a number like `123456789`)

**Method 2: Using @RawDataBot**

1. Search for `@RawDataBot` in Telegram
2. Start a chat and send any message
3. Look for the `"id"` field in the response
4. Copy that number

**Method 3: Manual Method**

1. Start a chat with your bot (search for the username you created)
2. Send any message to your bot (e.g., "Hello")
3. Open this URL in your browser (replace `YOUR_BOT_TOKEN`):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Look for the `"chat":{"id":` field in the JSON response
5. Copy that ID number

### Step 3: Configure Environment Variables

1. **Open your `.env` file** in the project root directory

2. **Update the Telegram configuration** section:

```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
TELEGRAM_CHAT_ID=123456789
TELEGRAM_ENABLED=true
```

Replace:
- `TELEGRAM_BOT_TOKEN`: Your token from Step 1
- `TELEGRAM_CHAT_ID`: Your chat ID from Step 2
- `TELEGRAM_ENABLED`: Set to `true` to enable notifications

3. **Save the file**

### Step 4: Install Required Package

Make sure your virtual environment is activated, then install:

```bash
pip install python-telegram-bot
```

Or if you're updating `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from BotFather | `123456789:ABC...` |
| `TELEGRAM_CHAT_ID` | Yes | Your personal chat ID | `123456789` |
| `TELEGRAM_ENABLED` | No | Enable/disable notifications | `true` or `false` (default: `false`) |

### Optional Configuration in Code

You can also configure the notifier programmatically:

```python
from telegram_notifier import TelegramNotifier

# Custom configuration
notifier = TelegramNotifier(
    bot_token='your_token_here',
    chat_id='your_chat_id_here',
    enabled=True,
    max_retries=3,        # Number of retry attempts
    retry_delay=2.0       # Delay between retries (seconds)
)
```

---

## Testing

### Quick Test

Run the built-in test to verify your setup:

```bash
cd src
python telegram_notifier.py
```

This will:
1. Load your configuration from `.env`
2. Test the connection to Telegram
3. Send a test message to your chat
4. Display the result

### Expected Output

If successful:
```
Telegram Notifier - Example Usage

Testing connection...
Connection successful!

Sending test trade alert...
Alert sent! Message ID: 12345
```

### Manual Test

You can also test using Python:

```python
from telegram_notifier import TelegramNotifier
from datetime import datetime

# Initialize
notifier = TelegramNotifier()

# Test connection
if notifier.test_connection():
    print("Success! Check your Telegram app.")
else:
    print("Failed. Check your configuration.")

# Test trade alert
sample_trade = {
    'profile_username': 'testuser',
    'ticker': 'AAPL',
    'strategy': 'CSP',
    'action': 'STO',
    'entry_price': 2.50,
    'quantity': 1,
    'strike_price': 170.00,
    'expiration_date': datetime.now().date(),
    'alert_timestamp': datetime.now()
}

notifier.send_new_trade_alert(sample_trade)
```

### Run Unit Tests

Run the comprehensive test suite:

```bash
# Install pytest if needed
pip install pytest pytest-mock

# Run all tests
pytest tests/test_telegram_notifier.py -v

# Run specific test class
pytest tests/test_telegram_notifier.py::TestNewTradeAlert -v

# Run with coverage
pip install pytest-cov
pytest tests/test_telegram_notifier.py --cov=src.telegram_notifier --cov-report=html
```

---

## Troubleshooting

### Common Issues

#### 1. "Telegram bot not initialized"

**Problem**: Bot token or chat ID is missing/incorrect

**Solution**:
- Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
- Make sure there are no extra spaces or quotes
- Verify the token is still valid (test in browser)

#### 2. "python-telegram-bot package not installed"

**Problem**: Required library is missing

**Solution**:
```bash
pip install python-telegram-bot
```

#### 3. "Connection test failed"

**Problem**: Cannot connect to Telegram API

**Solution**:
- Check your internet connection
- Verify the bot token is correct
- Make sure the bot is not blocked
- Try the API URL test from Step 2, Method 3

#### 4. "Unauthorized" Error

**Problem**: Bot token is invalid or revoked

**Solution**:
- Create a new bot token from BotFather
- Send `/token` to BotFather to get a new token
- Update `.env` with the new token

#### 5. "Chat not found" Error

**Problem**: Chat ID is incorrect or you haven't started a chat with the bot

**Solution**:
- Verify your chat ID using @userinfobot
- Make sure you've sent at least one message to your bot
- For group chats, use the group chat ID (negative number)

#### 6. Messages not appearing

**Problem**: Notifications are disabled or bot is muted

**Solution**:
- Check `TELEGRAM_ENABLED=true` in `.env`
- Verify the bot is not muted in Telegram
- Check that you're looking at the correct chat
- Review logs for error messages

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from telegram_notifier import TelegramNotifier
notifier = TelegramNotifier()
notifier.test_connection()
```

### Verify Configuration

Check your current configuration:

```python
from telegram_notifier import TelegramNotifier

notifier = TelegramNotifier()

print(f"Enabled: {notifier.enabled}")
print(f"Bot configured: {notifier.bot is not None}")
print(f"Chat ID: {notifier.chat_id}")

# Get bot info (if configured)
if notifier.bot:
    bot_info = notifier.get_bot_info()
    print(f"Bot username: @{bot_info['username']}")
    print(f"Bot name: {bot_info['first_name']}")
```

---

## Advanced Features

### Custom Retry Configuration

Configure retry behavior for network issues:

```python
notifier = TelegramNotifier(
    max_retries=5,      # Try up to 5 times
    retry_delay=3.0     # Wait 3 seconds between retries
)
```

### Rate Limiting

The notifier automatically handles Telegram's rate limits:
- Waits when rate limited (HTTP 429)
- Retries with exponential backoff
- Respects `retry_after` headers

### Group Notifications

To send notifications to a group:

1. Create a Telegram group
2. Add your bot to the group
3. Make the bot an admin (for posting)
4. Get the group chat ID:
   - Send a message in the group
   - Visit: `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
   - Look for the group's negative chat ID (e.g., `-123456789`)
5. Use the group chat ID in your `.env`:
   ```bash
   TELEGRAM_CHAT_ID=-123456789
   ```

### Multiple Recipients

To send to multiple recipients, create multiple notifier instances:

```python
# Personal notifications
personal_notifier = TelegramNotifier(
    chat_id='your_personal_chat_id'
)

# Group notifications
group_notifier = TelegramNotifier(
    chat_id='your_group_chat_id'
)

# Send to both
personal_notifier.send_new_trade_alert(trade_data)
group_notifier.send_new_trade_alert(trade_data)
```

### Custom Message Formatting

Send custom formatted messages:

```python
# Markdown formatting
notifier.send_custom_message(
    "*Bold* _italic_ `code`\n[Link](https://example.com)",
    parse_mode='Markdown'
)

# HTML formatting
notifier.send_custom_message(
    "<b>Bold</b> <i>italic</i> <code>code</code>",
    parse_mode='HTML'
)
```

### Disable Web Page Previews

By default, link previews are disabled. To enable:

```python
message_id = notifier._send_message(
    "Check this out: https://example.com",
    disable_web_page_preview=False
)
```

### Integration with Database

Track sent notifications to avoid duplicates:

```python
import psycopg2
from datetime import datetime

def send_and_log_notification(trade_id, trade_data):
    # Send notification
    message_id = notifier.send_new_trade_alert(trade_data)

    # Log to database
    if message_id:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO xtrades_notifications
            (trade_id, notification_type, telegram_message_id, status)
            VALUES (%s, %s, %s, %s)
        """, (trade_id, 'new_trade', message_id, 'sent'))

        conn.commit()
        conn.close()

    return message_id
```

---

## Security Best Practices

1. **Never commit your bot token** to version control
2. **Use environment variables** for sensitive data
3. **Regenerate tokens** if accidentally exposed
4. **Limit bot permissions** to only what's needed
5. **Monitor bot usage** for unusual activity
6. **Use HTTPS** for all API calls (handled automatically)

---

## Support and Resources

### Official Documentation

- [Telegram Bot API](https://core.telegram.org/bots/api)
- [python-telegram-bot Library](https://github.com/python-telegram-bot/python-telegram-bot)
- [BotFather Commands](https://core.telegram.org/bots#6-botfather)

### Useful BotFather Commands

- `/newbot` - Create a new bot
- `/mybots` - List your bots
- `/setname` - Change bot name
- `/setdescription` - Change bot description
- `/setuserpic` - Set bot profile picture
- `/token` - Get a new token (revokes old one)
- `/revoke` - Revoke current token
- `/deletebot` - Delete a bot

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the logs for error messages
3. Test with the provided test scripts
4. Verify your `.env` configuration
5. Check Telegram API status

---

## Example Integration

Here's a complete example integrating notifications with Xtrades sync:

```python
from telegram_notifier import TelegramNotifier
import psycopg2
from datetime import datetime

def sync_xtrades_with_notifications():
    """Sync Xtrades and send notifications for new trades."""

    # Initialize notifier
    notifier = TelegramNotifier()

    # Connect to database
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor()

    try:
        # Get new trades (not yet notified)
        cursor.execute("""
            SELECT t.id, t.ticker, t.strategy, t.action, t.entry_price,
                   t.quantity, t.strike_price, t.expiration_date,
                   t.alert_timestamp, p.username
            FROM xtrades_trades t
            JOIN xtrades_profiles p ON t.profile_id = p.id
            LEFT JOIN xtrades_notifications n
                ON t.id = n.trade_id AND n.notification_type = 'new_trade'
            WHERE n.id IS NULL
            AND t.status = 'open'
        """)

        new_trades = cursor.fetchall()

        # Send notifications for each new trade
        for trade in new_trades:
            trade_data = {
                'id': trade[0],
                'ticker': trade[1],
                'strategy': trade[2],
                'action': trade[3],
                'entry_price': trade[4],
                'quantity': trade[5],
                'strike_price': trade[6],
                'expiration_date': trade[7],
                'alert_timestamp': trade[8],
                'profile_username': trade[9]
            }

            # Send notification
            message_id = notifier.send_new_trade_alert(trade_data)

            # Log notification
            if message_id:
                cursor.execute("""
                    INSERT INTO xtrades_notifications
                    (trade_id, notification_type, telegram_message_id, status)
                    VALUES (%s, %s, %s, %s)
                """, (trade_data['id'], 'new_trade', message_id, 'sent'))
            else:
                cursor.execute("""
                    INSERT INTO xtrades_notifications
                    (trade_id, notification_type, status, error_message)
                    VALUES (%s, %s, %s, %s)
                """, (trade_data['id'], 'new_trade', 'failed', 'Failed to send'))

        conn.commit()
        print(f"Sent {len(new_trades)} notifications")

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

        # Send error notification
        notifier.send_sync_error_alert(str(e))

    finally:
        conn.close()

if __name__ == '__main__':
    sync_xtrades_with_notifications()
```

---

## Changelog

### Version 1.0.0 (2025-11-02)
- Initial release
- New trade alerts
- Trade update alerts
- Trade closed alerts
- Sync error alerts
- Daily summary alerts
- Retry logic and error handling
- Comprehensive test suite

---

## License

This Telegram integration is part of the Magnus Wheel Strategy Trading Dashboard.

---

**Ready to get started?** Follow the [Step-by-Step Setup](#step-by-step-setup) guide above!
