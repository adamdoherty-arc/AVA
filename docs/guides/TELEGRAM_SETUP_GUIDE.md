# Telegram Alerts Setup Guide

## Current Status
- **Bot Token**: ✅ Configured (@ava_n8n_bot)
- **Chat ID**: ⚠️ Needs configuration

## Complete Setup in 3 Steps

### Step 1: Start a chat with your bot
1. Open Telegram on your phone or computer
2. Search for: `@ava_n8n_bot`
3. Click on the bot to open the chat
4. Send any message (for example: `/start` or `Hello`)

### Step 2: Run the automated setup
Open a command prompt in the Magnus directory and run:
```bash
python setup_telegram_alerts.py
```

This script will:
- Find your chat ID automatically from the message you sent
- Update your .env file with the chat ID
- Send a test alert to verify everything works

### Step 3: Test the Subscribe button
1. Start the dashboard: `streamlit run dashboard.py`
2. Go to the "Game Cards" page
3. Click the "Subscribe" button on any game
4. You should receive a Telegram alert instantly!

## What You'll Receive

When you subscribe to a game, you'll get instant Telegram notifications for:
- **Score updates** - Real-time score changes
- **Quarter changes** - When quarters/periods change
- **Game status changes** - Pre-game, live, final
- **AI prediction updates** - Multi-agent AI analysis updates

## Example Alert

```
🏈 GAME SUBSCRIPTION CONFIRMED

Oklahoma Sooners @ Missouri Tigers

📅 Date: 11/22 - 12:00 PM EST
📺 Status: Live Now

You'll receive notifications for:
• Score updates
• Quarter changes
• Game status changes
• AI prediction updates

🤖 Multi-Agent AI Analysis
🎯 Prediction: Oklahoma -13.8
✅ 72% win probability
💡 Medium Confidence

Powered by Magnus NFL Tracker
```

## Troubleshooting

### "Chat not found" error
- Make sure you sent a message to @ava_n8n_bot first
- The bot can't send messages until you've initiated a chat

### "Bot token invalid"
- Check your .env file has the correct TELEGRAM_BOT_TOKEN
- Token should start with a number followed by a colon

### Not receiving alerts
- Check TELEGRAM_ENABLED=true in your .env file
- Verify your chat ID is set correctly
- Make sure you've subscribed to a game on the Game Cards page

## Need Help?

Run this to check your configuration:
```bash
python check_telegram_config.py
```
