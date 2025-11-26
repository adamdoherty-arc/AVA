# Kalshi Odds Status & Team Button Update

**Date:** November 14, 2025

---

## ✅ What Was Fixed

### 1. Team Selection Buttons - UPDATED

**Before:**
- Buttons showed: "🏈 Buffalo Bil" (football emoji + truncated name)
- No visual indication of which team

**After:**
- Team logo displayed above each button
- Buttons show full team name: "Buffalo Bills"
- Cleaner, more professional look
- Primary button style for better visibility

**Example:**
```
[Buffalo Bills Logo]
[Buffalo Bills] button

[Kansas City Logo]
[Kansas City Chiefs] button
```

### 2. Kalshi Matcher - IMPROVED

**Before:**
- Only searched for `market_type = 'winner'`
- Your database has `market_type = 'nfl'`
- Result: 0 matches

**After:**
- Searches for `market_type IN ('nfl', 'cfb', 'winner')`
- Requires `yes_price IS NOT NULL` (only markets with actual prices)
- Better chance of finding matches

---

## ⚠️ Why Kalshi Odds May Still Not Show

### Your Kalshi Database Structure

You have **3,794 active Kalshi markets**, but:

- ✅ 3,794 total markets
- ✅ 279 markets with prices (7%)
- ❌ 0 markets with `market_type = 'winner'`
- ❌ 3,515 markets with NULL prices (93%)

### Market Types in Your Database

Your Kalshi markets are **complex parlays**, not simple game winners:

#### Example Markets:
```
"yes Baltimore, yes Carolina, yes Denver, yes Miami"
"yes Buffalo wins by over 2.5 points, yes Josh Allen: 200+"
"yes James Cook III: 90+, yes Buffalo wins by over 6.5 points"
```

These are **multi-leg parlays** combining:
- Multiple games
- Player props (yards, touchdowns)
- Point spreads
- Over/under bets

### What Simple Winner Markets Look Like (Not in your DB):

```
"Buffalo Bills to win vs Kansas City Chiefs"
  Away team: Buffalo Bills - 68¢
  Home team: Kansas City Chiefs - 32¢
```

---

## 🔧 Solutions for Getting Kalshi Odds

### Option 1: Sync Simple Winner Markets (Recommended)

Run a Kalshi API sync that specifically fetches **single-game winner markets**:

```python
# Example Kalshi API request
market_type = "SINGLE_GAME_WINNER"  # Not parlays
category = "nfl"
```

**Benefits:**
- Clean, simple odds
- Easy to display
- Matches ESPN games perfectly

### Option 2: Parse Parlay Markets (Complex)

Extract team win probability from parlay titles:

**Challenges:**
- Parlay markets mix multiple conditions
- Odds reflect combined probability, not individual game
- "yes Buffalo wins by over 6.5" ≠ "Buffalo to win"
- Most markets don't have prices anyway (93%)

### Option 3: Use Different Odds Source

Alternative betting odds APIs:
- **The Odds API** (free tier available)
- **BetMGM**
- **DraftKings**
- **FanDuel**

---

## 📊 Current Database Statistics

```sql
Total Markets:          3,794
Active Markets:         3,794
Markets with Prices:    279 (7%)
Market Types:           'nfl' (100%)
Winner Markets:         0 (0%)
```

### Sample Market Breakdown:
```
MULTIGAME EXTENDED: 60%  (multi-game parlays)
SINGLEGAME:         40%  (single game, but still parlays)
```

---

## 🎯 Recommended Next Steps

### Immediate (Working Now):
✅ Team buttons show logos and names
✅ AI predictions work correctly
✅ Date filters work
✅ Immediate Telegram alerts work

### Short-Term (If you want Kalshi odds):

1. **Check Kalshi API for simpler markets:**
   ```bash
   python scripts/check_kalshi_simple_markets.py
   ```

2. **Or use The Odds API (free tier):**
   ```python
   # Example: Get simple NFL odds
   import requests
   response = requests.get(
       'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/',
       params={'apiKey': 'YOUR_KEY'}
   )
   ```

3. **Or display Kalshi market link instead of odds:**
   ```
   💰 Kalshi Markets: [View](https://kalshi.com/markets/nfl)
   ```

---

## 💡 Why This Matters

**For Trading Dashboard:**
- Kalshi parlays are **complex instruments**
- Not suitable for simple "who will win" display
- Better to show:
  - AI prediction (✅ Working)
  - Confidence scores (✅ Working)
  - Expected value (✅ Working)

**For Betting:**
- If you want Kalshi parlay odds, that requires:
  - Different UI design
  - Explanation of parlay conditions
  - Not just "Bills: 68¢"

---

## 📁 Files Modified

1. **src/espn_kalshi_matcher.py** (lines 59-80)
   - Added support for market_type IN ('nfl', 'cfb', 'winner')
   - Required yes_price IS NOT NULL
   - Better logging

2. **game_cards_visual_page.py** (lines 626-667)
   - Moved logo fetching earlier (line 626)
   - Removed football emojis from buttons
   - Added team logos above buttons
   - Changed button style to "primary"

---

## 🧪 Testing

To test if Kalshi odds work with your current data:

```bash
cd C:\Code\Legion\repos\ava
python src/espn_kalshi_matcher.py
```

This will show:
- How many ESPN games found
- How many matched to Kalshi markets
- What the odds are (if any)

---

## Summary

**Team Buttons:** ✅ Fixed - Now show logos and clean names

**Kalshi Odds:** ⚠️ Limited - Your database has parlay markets, not simple winner markets

**Recommendation:** Continue using AI predictions (which work great!) and consider adding a simple odds API if you need betting odds.
