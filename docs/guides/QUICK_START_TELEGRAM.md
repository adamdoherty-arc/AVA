# 🚀 Quick Start: Activate Telegram Alerts

## You're 2 Minutes Away from Game Alerts!

Your bot `@ava_n8n_bot` is already configured. Just complete the setup:

---

## Step 1: Send a Message (30 seconds)
1. Open **Telegram** (phone or desktop)
2. Search for: **@ava_n8n_bot**
3. Send any message (try: `/start` or `Hello`)

---

## Step 2: Run Setup Script (30 seconds)
Open Command Prompt in this directory and run:
```bash
python setup_telegram_alerts.py
```

The script will:
- ✅ Find your chat ID automatically
- ✅ Update your .env file
- ✅ Send you a test alert

---

## Step 3: Test It! (1 minute)
1. Start dashboard:
   ```bash
   streamlit run dashboard.py
   ```
2. Go to **Game Cards** page
3. Click **Subscribe** on any game
4. Check Telegram for your alert!

---

## That's It! 🎉

Every time you subscribe to a game, you'll get instant alerts for:
- Score updates
- Quarter changes
- Game status changes
- AI predictions

---

## Need Help?

Check configuration:
```bash
python check_telegram_config.py
```

Send test alert:
```bash
python send_telegram_test.py
```

Full docs: [TELEGRAM_SETUP_GUIDE.md](TELEGRAM_SETUP_GUIDE.md)
