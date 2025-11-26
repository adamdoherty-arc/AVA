# Live Game Monitoring System - Complete Guide 🎯

## Overview
Your Magnus platform has a **fully built live game monitoring system** that sends intelligent Telegram updates when games change. No spam - only meaningful updates!

---

## ✅ What's Already Built

### Background Monitoring Service
**File**: `game_watchlist_monitor.py`

This service runs continuously in the background and:
- ✅ Checks all your subscribed games every 5 minutes (configurable)
- ✅ Fetches live scores from ESPN
- ✅ Gets updated Kalshi betting odds
- ✅ Runs fresh AI predictions
- ✅ Detects what changed since last update
- ✅ Only sends updates when something meaningful happens

### What Gets Monitored

#### 1. Score Changes 🏈
```
Score changed: 14-10 → 21-10
```

#### 2. Period/Quarter Changes ⏱️
```
Period changed: 2nd Quarter → Halftime
```

#### 3. Game Status Changes 📺
```
Status: Live → Final
```

#### 4. Your Team Status 🎯
```
🎉 Oklahoma Sooners is now WINNING!
⚠️ Missouri Tigers is now LOSING
```

#### 5. Odds Changes 💰
Alerts when odds shift >10 cents:
```
Odds shift: Oklahoma 65% → 72%
```

#### 6. AI Prediction Changes 🤖
Alerts when:
- Winner prediction flips
- Confidence changes >10%

---

## 🎯 Smart Update Logic (No Spam!)

The system **only sends updates when something changes**:

### Initial Subscription
```
🔔 GAME SUBSCRIPTION CONFIRMED

Oklahoma Sooners @ Missouri Tigers

📊 Live Score: 14 - 10
📺 Status: Live - 2nd Quarter

You'll receive notifications for:
• Score updates
• Quarter changes
• Game status changes
• AI prediction updates

🤖 Multi-Agent AI Analysis
🎯 Prediction: Oklahoma -6.5
✅ 68% win probability
💡 High Confidence
```

### Score Update (Only when score changes)
```
🔔 GAME UPDATE

🏈 Oklahoma Sooners @ Missouri Tigers
**21 - 10**
_Live - 3rd Quarter 8:45_

📊 What Changed:
• Score changed: 14-10 → 21-10
• 🎉 Oklahoma Sooners is now WINNING by more!

🤖 AI Update:
🎯 Prediction: Oklahoma -8.5
✅ 75% win probability (+7% confidence)
💡 STRONG BET recommendation

💰 Kalshi Odds:
Oklahoma: 78¢ (+6¢)
Missouri: 22¢ (-6¢)

**Recommendation**: INCREASE BET - Oklahoma pulling away
```

### No Update Sent If:
- ❌ No score change
- ❌ No quarter change
- ❌ No significant odds movement
- ❌ No AI prediction shift

---

## 🚀 How to Use

### Step 1: Subscribe to Games
1. Open dashboard: `streamlit run dashboard.py`
2. Go to **Sports Game Hub** page
3. Use team filters to find games
4. Click **Subscribe** on any game
5. Get instant confirmation on Telegram

### Step 2: Start the Monitor
Open a new terminal and run:
```bash
python game_watchlist_monitor.py
```

Or run in background mode (recommended):
```bash
python game_watchlist_monitor.py --interval 5 &
```

The monitor will:
1. Send you a startup confirmation on Telegram
2. Check all your subscribed games every 5 minutes
3. Send updates only when something changes
4. Keep running until you stop it

### Step 3: Receive Smart Updates
You'll get Telegram alerts for:
- ✅ Score changes
- ✅ Quarter/period changes
- ✅ Your team winning/losing status changes
- ✅ Significant odds movements
- ✅ AI prediction changes
- ✅ Betting recommendations

---

## 📱 Example Update Flow

### 1. You Subscribe
**Action**: Click Subscribe on Oklahoma @ Missouri game
**Telegram**: Instant confirmation with current game state

### 2. Oklahoma Scores (14-10 → 21-10)
**System Detects**:
- Score changed
- Your team (Oklahoma) extended lead
- AI confidence increased 7%
- Odds moved 6 cents

**Telegram Update**:
```
🔔 GAME UPDATE

🏈 Oklahoma @ Missouri
21 - 10
_3rd Quarter 8:45_

📊 Changes:
• Score: 14-10 → 21-10
• 🎉 Oklahoma extending lead!

🤖 AI: 75% win probability (+7%)
💰 Odds: Oklahoma 78¢ (+6¢)
💡 Recommendation: INCREASE BET
```

### 3. Halftime (No Score Change)
**System**: Detects period change but no score/odds change
**Telegram**: No update (no meaningful change)

### 4. Missouri Scores (21-17)
**System Detects**:
- Score changed
- Lead shrinking
- AI confidence dropped
- Odds shifted

**Telegram Update**:
```
🔔 GAME UPDATE

21 - 17
_4th Quarter 12:00_

📊 Changes:
• Score: 21-10 → 21-17
• ⚠️ Lead shrinking!

🤖 AI: 62% win probability (-13%)
💰 Odds: Oklahoma 65¢ (-13¢)
💡 Recommendation: HEDGE BET - Missouri momentum
```

---

## ⚙️ Configuration

### Update Frequency
Default: 5 minutes

Change via command line:
```bash
python game_watchlist_monitor.py --interval 3  # Check every 3 minutes
python game_watchlist_monitor.py --interval 10 # Check every 10 minutes
```

### Thresholds (in game_watchlist_monitor.py)
```python
# Odds change threshold: 10 cents (0.10)
if away_change > 0.10:
    # Send update

# AI confidence threshold: 10%
if abs(new_confidence - old_confidence) > 10:
    # Send update
```

You can adjust these to get more or fewer updates.

---

## 🎯 What Makes This Smart

### 1. State Tracking
The system stores the last known state in the database:
- Last score
- Last period
- Last AI prediction
- Last odds

### 2. Change Detection
Compares current state vs last state:
- Did score change?
- Did quarter change?
- Did odds move significantly?
- Did AI flip prediction?

### 3. Smart Filtering
Only sends updates for meaningful changes:
- Score changes: Always notify
- Period changes: Only if other changes too
- Odds changes: >10 cents movement
- AI changes: >10% confidence swing

### 4. Deduplication
Never sends the same update twice:
- After sending update, saves new state as "last known"
- Next check compares against this new baseline
- Only new changes trigger new alerts

---

## 📊 Database Tables Used

### game_watchlist
Stores your subscribed games:
```sql
- game_id
- user_id
- sport (NFL/NCAA)
- away_team, home_team
- selected_team (your pick)
- is_active
- added_at
```

### game_state_history
Tracks last known state:
```sql
- game_id
- away_score, home_score
- status, period, clock
- ai_confidence, ai_predicted_winner
- kalshi_away_odds, kalshi_home_odds
- timestamp
```

### watchlist_updates
Logs all sent updates:
```sql
- game_id
- user_id
- update_type (score/period/odds/ai)
- message
- sent_at
```

---

## 🚨 Troubleshooting

### Not Receiving Updates?

**Check 1: Is monitor running?**
```bash
# Look for process
ps aux | grep game_watchlist_monitor

# Check log file
tail -f game_watchlist_monitor.log
```

**Check 2: Are games subscribed?**
- Open Sports Game Hub page
- Check "Watching" count in header
- Subscribe to at least one game

**Check 3: Is Telegram configured?**
```bash
python check_telegram_config.py
```

**Check 4: Are games live?**
- Monitor only sends updates for live/scheduled games
- Completed games won't trigger updates

### Monitor Stopped?

Restart it:
```bash
python game_watchlist_monitor.py --interval 5
```

Or run as background service (stays running):
```bash
nohup python game_watchlist_monitor.py --interval 5 &
```

---

## 🎊 Summary

### What You Have:
✅ **Fully operational live game monitoring system**
✅ **Smart change detection** (no spam)
✅ **Telegram integration** (instant updates)
✅ **AI predictions** included in updates
✅ **Betting recommendations** (increase/decrease/hedge)
✅ **Odds tracking** (Kalshi integration)
✅ **Your team status** (winning/losing alerts)

### How It Works:
1. You subscribe to games via Sports Game Hub
2. Background monitor checks games every 5 minutes
3. Detects meaningful changes (scores, odds, AI predictions)
4. Sends smart Telegram updates with recommendations
5. Never repeats same message (state tracking)

### To Start Using:
1. Subscribe to games in dashboard ✅ (already works)
2. Run: `python game_watchlist_monitor.py`
3. Receive smart updates on Telegram! 📱

---

**Status**: ✅ Fully Built and Ready to Use
**Last Verified**: 2025-11-22
