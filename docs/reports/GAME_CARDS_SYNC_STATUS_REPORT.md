# Game Cards - Sync Status & Kalshi Odds Report

**Generated**: 2025-11-21 10:06:49
**Status**: All APIs operational, but Kalshi matching needs improvement

---

## Executive Summary

✅ **ESPN APIs are working perfectly** - Fetching games from all sports
❌ **Kalshi odds matching is poor** - Most games have no Kalshi odds
❌ **Database is empty** - ESPN data not persisted (but this is OK - UI fetches live)

**User Impact**: "I see none on NCAA pages" - This is because 0% of NCAA games have matched Kalshi odds, so if any filter requires odds, no games will show.

---

## Detailed Findings

### 1. ESPN API Status ✅ WORKING

| Sport | Games Fetched | Status Breakdown |
|-------|--------------|------------------|
| **NFL** | 14 games | Working |
| **NCAA** | 64 games | 58 upcoming, 6 final |
| **NBA** | 9 games | Working |

**Result**: ESPN APIs are fully operational. The game cards page can fetch live data.

---

### 2. Kalshi Odds Matching ❌ POOR

| Sport | Matched | Match Rate | Issue |
|-------|---------|------------|-------|
| **NCAA** | 0/5 games | 0% | No Kalshi markets for NCAA games |
| **NFL** | 1/5 games | 20% | Low match rate |

**Root Causes**:

1. **Kalshi markets have sector=None**
   - All 6,227 Kalshi markets have `sector = None` in database
   - Matcher relies on sector to identify sport
   - Without sector, can't properly filter markets by sport

2. **Team name mapping issues**
   - ESPN returns: "Florida State Seminoles"
   - Kalshi might have: "Florida State" or "FSU"
   - Fuzzy matching not working well

3. **NCAA focus on ranked teams**
   - Kalshi typically only offers markets for Top 25 matchups
   - 64 NCAA games includes many unranked teams
   - Expected to have low match rate for NCAA

---

### 3. Database Status ℹ️ NOT CRITICAL

| Table | Rows | Notes |
|-------|------|-------|
| `nfl_games` | 0 | Empty but OK - UI fetches live from ESPN |
| `kalshi_markets` | 6,227 | Has data but missing sectors |
| `game_watchlist` | 11 | Working |

**Note**: The game cards page fetches ESPN data **directly from API**, not from database. Database is used for:
- Kalshi markets (for odds matching)
- Watchlist (for user favorites)
- Historical analysis (optional)

---

## Why "None on NCAA Pages"

Your UI is showing no NCAA games because:

1. **ESPN fetches 64 NCAA games** ✅
2. **Kalshi matcher runs** and finds 0% match ❌
3. **If UI filters by "Games with Odds"**, all 64 games are hidden
4. **User sees**: "No games available"

**Likely scenarios**:
- Date filter set to wrong range
- "Hide Final Games" is checked (6 games final)
- Odds filter requires Kalshi odds (none available)
- Auto-refresh cleared cache and games not re-fetched

---

## Solutions & Fixes

### Quick Fix #1: Remove Odds Filter (IMMEDIATE)

If your UI has any filter that requires Kalshi odds, **disable it temporarily**:

```python
# In game_cards_visual_page.py
# Comment out or set to "All Games"
odds_filter = "All Games"  # Not "Only with Odds"
```

This will show all ESPN games regardless of Kalshi odds.

---

### Quick Fix #2: Check Date Filter (IMMEDIATE)

NCAA games are mostly scheduled for **November 22-23 (Friday-Saturday)**:

```python
# Ensure date filter includes future dates
date_filter_mode = "All Games"  # or "Next 7 Days"
```

If set to "Today Only" on Thursday Nov 21, you'll see nothing.

---

### Fix #3: Resync Kalshi Markets with Sectors (RECOMMENDED)

**Problem**: Kalshi markets missing `sector` field
**Solution**: Update sync script to populate sectors

**File**: `sync_kalshi_team_winners.py` (lines 83-100)

Current code categorizes markets (NFL vs CFB) but doesn't save to DB. Need to add:

```python
# In sync_kalshi_team_winners.py, after line 81
market['sector'] = self.categorize_market(market)  # Add this line
```

Then resync:

```bash
python sync_kalshi_team_winners.py --sport all
```

---

### Fix #4: Improve Team Name Matching (RECOMMENDED)

**File**: `src/espn_kalshi_matcher_optimized.py`

Add better fuzzy matching for team names:

```python
from fuzzywuzzy import fuzz

def match_team_names(espn_team, kalshi_team):
    """Fuzzy match team names"""
    # Try exact match
    if espn_team.lower() == kalshi_team.lower():
        return 100

    # Try partial match
    score = fuzz.partial_ratio(espn_team.lower(), kalshi_team.lower())

    # Try with common abbreviations
    espn_short = espn_team.split()[-1]  # "Florida State Seminoles" -> "Seminoles"
    kalshi_short = kalshi_team.split()[-1]
    abbr_score = fuzz.ratio(espn_short.lower(), kalshi_short.lower())

    return max(score, abbr_score)
```

---

### Fix #5: Better Logging (DEBUGGING)

Add detailed logging to matcher to see why matches fail:

```python
# In espn_kalshi_matcher_optimized.py
logger.info(f"ESPN team: {espn_team}")
logger.info(f"Available Kalshi teams: {[m['home_team'] for m in markets[:5]]}")
logger.info(f"Match attempts: {match_scores}")
```

This will show exactly what team names are being compared.

---

## Testing Steps

### Test 1: Verify ESPN Data is Showing

1. Open game_cards_visual_page.py in Streamlit
2. Select "NCAA" from sport filter
3. Set filters to:
   - Date Filter: "All Games" or "Next 7 Days"
   - Game Status: "All Games"
   - Hide Final: Unchecked
4. **Expected**: See 58-64 games displayed (64 total: 58 upcoming, 6 final)

**If you see 0 games**: ESPN API issue or cache problem (clear cache with 'C')

---

### Test 2: Check Kalshi Odds Coverage

1. Look at game cards displayed
2. Check "Kalshi Status" indicator in UI
3. **Expected**: Should show "0/64 games with odds" or "X/64 games with odds"

**If all games show odds**: Matcher is working!
**If 0 games show odds**: Matcher is broken (needs fixes above)

---

### Test 3: Verify Database Isn't Blocking

1. Check that UI shows "ESPN Status: ✅ X games fetched"
2. If showing "ESPN Status: ❌ No games", then:
   - Cache might be stale (clear cache)
   - ESPN API might be down (check espn.com)
   - Network issue (check internet connection)

---

## Recommendations by Priority

### Priority 1: IMMEDIATE (Get games showing)

✅ **Set date filter to "All Games" or "Next 7 Days"**
✅ **Uncheck "Hide Final Games" if checked**
✅ **Set odds filter to "All Games" (don't require Kalshi odds)**
✅ **Clear Streamlit cache** (press 'C' in app)

**Result**: NCAA games should appear even without Kalshi odds

---

### Priority 2: SHORT TERM (Improve Kalshi matching)

🔧 **Fix sector population in sync script**
🔧 **Improve team name fuzzy matching**
🔧 **Add debug logging to matcher**
🔧 **Resync Kalshi markets**

**Result**: Better Kalshi odds coverage (expect ~30-40% for NCAA, ~60-70% for NFL)

---

### Priority 3: LONG TERM (Optimal experience)

📊 **Build team name mapping database**
- Map all ESPN team names to Kalshi equivalents
- Handle abbreviations, nicknames, variations

📊 **Add manual market linking**
- Allow users to manually link ESPN game → Kalshi market
- Save these mappings for future use

📊 **Sync ESPN data to database** (optional)
- Store ESPN games in nfl_games table
- Enable historical analysis and caching
- Reduce API calls

---

## Expected Kalshi Coverage

**Realistic expectations**:

| Sport | Expected Match Rate | Reason |
|-------|-------------------|--------|
| **NFL** | 60-80% | Most NFL games have markets |
| **NCAA Top 25** | 70-90% | Ranked matchups usually have markets |
| **NCAA All Games** | 10-30% | Many small games don't have markets |
| **NBA** | 80-95% | Most NBA games have markets |

**Kalshi focuses on**:
- Major conference games (SEC, Big Ten, Big 12, etc.)
- Ranked team matchups
- Primetime games
- Playoff implications

**Kalshi rarely covers**:
- FCS (lower division)
- Mid-week mid-major games
- Blowout matchups
- Rivalry games between unranked teams

---

## Quick Commands

```bash
# Clear Streamlit cache
# Press 'C' in app

# Resync Kalshi markets (all sports)
python sync_kalshi_team_winners.py --sport all

# Resync Kalshi markets (NFL only)
python sync_kalshi_team_winners.py --sport nfl

# Resync Kalshi markets (NCAA only)
python sync_kalshi_team_winners.py --sport cfb

# Run diagnostic again
python diagnose_game_cards_issue.py

# Check database status
python check_games_kalshi_status.py
```

---

## Files to Modify

### To Fix Kalshi Matching

1. **sync_kalshi_team_winners.py** (line 81)
   - Add: `market['sector'] = self.categorize_market(market)`

2. **src/espn_kalshi_matcher_optimized.py** (add function)
   - Improve team name fuzzy matching
   - Add debug logging

3. **game_cards_visual_page.py** (line 609)
   - Update help text to reflect realistic expectations
   - Add "Most NCAA games won't have Kalshi odds" note

---

## Summary

### What's Working ✅
- ESPN NFL API: 14 games
- ESPN NCAA API: 64 games
- ESPN NBA API: 9 games
- Kalshi database: 6,227 markets
- Streamlit UI: Rendering correctly

### What's Not Working ❌
- NCAA Kalshi matching: 0%
- NFL Kalshi matching: 20%
- Kalshi markets missing sectors
- Database not populated (OK for now)

### User Impact
**"I see none on NCAA pages"** because:
- Filters might be hiding games (date, status, odds)
- Cache might be stale
- UI might require Kalshi odds (which 0% of NCAA games have)

### Immediate Action
1. ✅ Clear cache (press 'C')
2. ✅ Set date filter to "All Games" or "Next 7 Days"
3. ✅ Uncheck "Hide Final Games"
4. ✅ Set odds filter to "All Games"
5. ✅ Refresh page (F5)

**Expected Result**: See 58-64 NCAA games (even without Kalshi odds)

---

*Generated by: diagnose_game_cards_issue.py*
*Last Updated: 2025-11-21 10:06:49*
