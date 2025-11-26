# Game Cards AI & Kalshi Odds Fix Plan

**Date:** November 15, 2025
**Status:** Ready to implement

---

## 🔍 Issues Identified

### 1. **AI Predictions All Look the Same**

**Current Behavior:**
- All games show: Win Prob 50%, Confidence 50%, EV 0%, Recommendation PASS
- No variation between games
- Generic reasoning like "No value detected"

**Root Causes:**
- ❌ No Kalshi odds available (see Issue #2)
- ❌ Default market_odds = 0.5 for all games
- ❌ Win probability calculation defaults to 50/50 without odds
- ❌ No score differential analysis when game is live

**What Should Happen:**
- Each game shows unique prediction based on:
  - Current score (if live)
  - Time remaining / game period
  - Team momentum
  - Historical performance (if available)
  - Kalshi odds (when available)

---

### 2. **Kalshi Odds Not Displaying**

**Current Behavior:**
- "Kalshi Odds: 0/15 games" in dashboard
- All game cards show no Kalshi odds
- Database has 3,794 Kalshi markets but **0 matches** to ESPN games

**Root Causes:**
- ❌ Database only has **combo/parlay markets** (e.g., "yes Baltimore,yes Carolina,yes Denver")
- ❌ Database missing **single-game team winner markets** (e.g., "Will Jacksonville beat LA?")
- ❌ ESPN-Kalshi matcher can't find simple team vs team markets
- ❌ Market titles don't match team names (Baltimore vs Ravens, etc.)

**What Kalshi Actually Has (per user):**
- Jacksonville Jaguars: 41% win probability
- Los Angeles (Chargers/Rams): 59% win probability
- These ARE available on Kalshi, just not in our database

**Fix Required:**
1. Sync fresh Kalshi markets from API (not just database)
2. Update matcher to handle team name variations
3. Store single-game winner markets properly

---

### 3. **Specific Bugs Found**

#### Bug #1: Confidence Display (RESOLVED)
- Initial report: "Confidence: 5000%"
- Actual: Confidence calculation is correct (50.0)
- **Display is correct** - false alarm from test output formatting

#### Bug #2: Win Probability Always 50%
- Location: `advanced_betting_ai_agent.py` lines 213-259
- Issue: Without Kalshi odds, both teams default to 50% probability
- No adjustment for current score or game state

#### Bug #3: Score Differential Not Considered
- Even when Jets are losing 14-27, prediction still shows 50/50
- Game state analysis exists but isn't weighted enough

#### Bug #4: LLM Predictions Not Used
- User can select GPT-4, Claude, etc. from dropdown
- But predictions still show generic local AI results
- LLM calls may be failing silently

---

## 🔧 Fixes to Implement

### **Fix #1: Improve Local AI Predictions (Without Odds)**

Even without Kalshi odds, AI should analyze:

```python
# Enhanced win probability calculation
def _calculate_win_probability_enhanced(self, game_state, odds_analysis):
    # If live game with score, use score differential
    if game_state['is_live']:
        away_score = game_state.get('away_score', 0)
        home_score = game_state.get('home_score', 0)
        score_diff = abs(away_score - home_score)
        time_weight = game_state['time_weight']  # 0-1 based on period

        # Calculate probability based on score and time
        if away_score > home_score:
            # Away team leading
            base_prob = 0.5 + (score_diff / 50) * time_weight  # Max 80% for 15pt lead in Q4
            away_prob = min(0.9, base_prob)  # Cap at 90%
            home_prob = 1.0 - away_prob
        elif home_score > away_score:
            # Home team leading
            base_prob = 0.5 + (score_diff / 50) * time_weight
            home_prob = min(0.9, base_prob)
            away_prob = 1.0 - home_prob
        else:
            # Tied game
            away_prob = 0.5
            home_prob = 0.5
    else:
        # Pre-game: use odds or default 50/50
        away_prob = odds_analysis.get('away_implied_prob', 0.5)
        home_prob = odds_analysis.get('home_implied_prob', 0.5)

    return away_prob, home_prob
```

### **Fix #2: Sync Real Kalshi Team Winner Markets**

**Option A: Use Kalshi Public API**
```python
# In kalshi_client.py
def get_nfl_team_winner_markets(self):
    """Fetch simple team winner markets for NFL games"""
    markets = self.get_all_markets(status='open')

    # Filter for team winner markets (not player props, not parlays)
    team_winner_markets = []
    for market in markets:
        title = market.get('title', '').lower()
        ticker = market.get('ticker', '')

        # Include if:
        # - Series is NFL
        # - Title contains team names
        # - NOT a player prop (no yards/touchdowns)
        # - NOT a combo bet (no commas in title)
        if ('nfl' in ticker.lower() and
            'win' in title and
            'yards' not in title and
            'touchdown' not in title and
            ',' not in title):
            team_winner_markets.append(market)

    return team_winner_markets
```

**Option B: Use Kalshi Website Scraping**
- Scrape https://kalshi.com/markets/sports/nfl
- Extract team winner markets directly
- Parse odds percentages

### **Fix #3: Improve ESPN-Kalshi Matcher**

```python
# In espn_kalshi_matcher.py
def match_espn_to_kalshi(espn_game, kalshi_markets):
    """Match ESPN game to Kalshi market with fuzzy matching"""
    espn_away = normalize_team_name(espn_game['away_team'])
    espn_home = normalize_team_name(espn_game['home_team'])

    for market in kalshi_markets:
        market_title = market['title'].lower()

        # Check if both teams mentioned in title
        if espn_away in market_title and espn_home in market_title:
            return market

        # Try variations (Ravens vs Baltimore, etc.)
        away_variations = get_team_variations(espn_away)
        home_variations = get_team_variations(espn_home)

        for away_var in away_variations:
            for home_var in home_variations:
                if away_var in market_title and home_var in market_title:
                    return market

    return None

def get_team_variations(team_name):
    """Get all name variations for a team"""
    variations = {
        'Baltimore Ravens': ['baltimore', 'ravens'],
        'New England Patriots': ['new england', 'patriots'],
        'Kansas City Chiefs': ['kansas city', 'kansas', 'chiefs', 'kc'],
        # ... all teams
    }
    return variations.get(team_name, [team_name.lower()])
```

### **Fix #4: Enable LLM Predictions**

Check why LLM calls aren't working:

```python
# In game_cards_visual_page.py line 828-900
# Verify llm_service is passed correctly
# Add error handling to show LLM failures
# Log which model is actually being used
```

---

## 📋 Implementation Order

1. ✅ **Fix AI Win Probability** (30 min)
   - Update score differential logic
   - Add time weighting
   - Better reasoning generation

2. ✅ **Sync Kalshi Markets** (45 min)
   - Update kalshi_client.py
   - Fetch team winner markets
   - Store in database properly

3. ✅ **Improve ESPN-Kalshi Matcher** (30 min)
   - Add team name variations
   - Fuzzy matching logic
   - Test with Jacksonville vs LA example

4. ✅ **Fix LLM Integration** (20 min)
   - Debug why LLM calls fail
   - Add error logging
   - Verify Claude/GPT-4 work

5. ✅ **Test & Verify** (15 min)
   - Check 5+ games show different predictions
   - Verify Kalshi odds display
   - Confirm LLM models work

**Total Time: ~2.5 hours**

---

## 🎯 Success Criteria

After fixes, game cards page should show:

✅ Each game has UNIQUE prediction (not all 50/50/0/PASS)
✅ Kalshi odds displayed for available games (e.g., "Jags 41¢, LA 59¢")
✅ Win probability reflects current score (e.g., if losing 14-27, show realistic prob)
✅ Confidence varies by game quality (50-90% range)
✅ Recommendations vary (PASS, BUY, STRONG_BUY)
✅ Reasoning is specific to each game
✅ LLM models (GPT-4, Claude) work when selected

---

## 🧪 Test Cases

**Test 1: Live Game with Score**
- Game: Jets 14 - Patriots 27 (Q4)
- Expected: Patriots 75-85% win probability
- Confidence: 70-80% (late in game, clear lead)
- Recommendation: Based on if Kalshi odds exist

**Test 2: Kalshi Odds Available**
- Game: Jaguars vs LA
- Expected: Shows "Jags 41¢, LA 59¢"
- AI adjusts prediction based on these odds
- Confidence higher with market data

**Test 3: Pre-game (No Score)**
- Game: Upcoming game, no score yet
- Expected: 50/50 or based on Kalshi odds if available
- Confidence: 50-60% (less certain without live data)
- Recommendation: Likely PASS unless great odds value

**Test 4: LLM Model Selected**
- Select "GPT-4 Turbo" from dropdown
- Expected: Prediction shows "✨ GPT-4" badge
- Reasoning is more detailed/conversational
- Different than local AI prediction

---

## 📝 Files to Modify

1. `src/advanced_betting_ai_agent.py` - Fix win probability calculation
2. `src/kalshi_client.py` - Add team winner market fetching
3. `src/espn_kalshi_matcher.py` - Improve matching logic
4. `game_cards_visual_page.py` - Fix LLM integration
5. `src/kalshi_db_manager.py` - Update schema if needed

---

Ready to implement? Let me know and I'll proceed with all fixes systematically.
