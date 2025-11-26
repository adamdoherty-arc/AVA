# Subscription Management - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Subscribe to Games
```
1. Open dashboard → Click "🏟️ Sports Game Hub"
2. Choose sport tab (NFL / NCAA / NBA)
3. Click "Subscribe" on any game card
4. ✅ Instant Telegram alert confirms subscription
```

### Step 2: Configure Monitoring
```
1. Go to "⚙️ Settings" tab
2. Choose update interval (1, 3, 5, 10, or 15 minutes)
3. Click "▶️ Start Monitoring"
4. Keep browser tab open
```

### Step 3: Receive Updates
```
✅ Telegram updates arrive automatically when:
   • Scores change
   • Quarters change
   • Odds shift >10¢
   • AI predictions change
   • Your team starts winning/losing
```

---

## 📊 Quick Access

### View Subscriptions
**Main Dashboard:**
- Check sidebar → "📋 My Subscriptions" widget
- Shows count and recent games

**Full Management:**
- Sports Game Hub → ⚙️ Settings tab
- Complete list organized by sport

### Unsubscribe
**Option 1:** Settings tab → Find game → Click 🗑️
**Option 2:** NFL/NCAA/NBA tab → Find game → Click "Unsubscribe"

---

## ⚡ Monitoring Options

### In-Browser (Simple)
```
Sports Game Hub → Settings → Start Monitoring
Keep browser tab open
```

### Background (Advanced)
```bash
python game_watchlist_monitor.py --interval 5
```

---

## 🎯 Recommended Settings

| Use Case | Interval | Why |
|----------|----------|-----|
| Live game watching | 3-5 min | Good balance |
| Tracking multiple games | 5-10 min | Standard |
| Battery saver | 10-15 min | Less frequent |
| Critical moments | 1-3 min | Fastest |

---

## 🔔 What You'll Receive

Every Telegram update includes:
```
🔔 GAME UPDATE

🏈 Miami Dolphins @ Buffalo Bills
21 - 17 ✅

📊 What Changed:
• Score changed: 14-17 → 21-17

🔥 Your Team (Miami): WINNING by 4

💰 Kalshi Odds:
   Miami: 62¢
   Buffalo: 38¢

🤖 AI Prediction: MIAMI wins
   Win Probability: 62%
   Recommendation: INCREASE_BET
```

---

## 🆘 Troubleshooting

**No Telegram alerts?**
→ Check `.env` has `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`

**Monitoring not working?**
→ Ensure browser tab is open OR run background script

**Not seeing updates?**
→ Only meaningful changes trigger updates (no spam!)

**Dashboard widget missing?**
→ Subscribe to at least one game first

---

## 📁 Key Files

- [game_cards_visual_page.py](game_cards_visual_page.py) - Sports Game Hub with Settings tab
- [dashboard.py](dashboard.py) - Main dashboard with My Subscriptions widget
- [game_watchlist_monitor.py](game_watchlist_monitor.py) - Background monitoring script
- [src/game_watchlist_manager.py](src/game_watchlist_manager.py) - Watchlist logic

---

## 📖 Full Documentation

See [SUBSCRIPTION_MANAGEMENT_COMPLETE.md](SUBSCRIPTION_MANAGEMENT_COMPLETE.md) for:
- Complete architecture details
- Database schema
- All features explained
- Advanced configuration
- Troubleshooting guide

---

**Ready to Start?** → Open Sports Game Hub and click Subscribe! 🎉
