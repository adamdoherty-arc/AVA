# Phase 2: Unified Ranking System - COMPLETE ✅

**Date:** November 22, 2025
**Duration:** ~2 hours
**Status:** All components built and ready to test

---

## 🎯 **What Was Built**

### **The Problem:**
You requested: *"there should be some type of ranking system I can view to rank all games from all sports to know which ones have the best odds of making money"*

Before Phase 2, the system had:
- ❌ No way to compare NFL vs NCAA vs NBA bets
- ❌ No unified scoring system
- ❌ No cross-sport profitability ranking
- ❌ Manual work to find best opportunities

### **The Solution:**
Created a complete **Unified Ranking System** that:
- ✅ Scores ALL betting opportunities on the same 0-100 scale
- ✅ Ranks bets across ALL sports (NFL, NCAA, NBA, politics)
- ✅ Combines 5 factors: EV (40%), Confidence (25%), Edge (15%), Liquidity (10%), Recency (10%)
- ✅ Shows you the most profitable bets regardless of sport
- ✅ Color-coded ratings: Exceptional 🔥 / Great ✅ / Good 👍 / Decent ⚠️ / Poor ❌

---

## 📦 **Components Created**

### **1. OpportunityScorer Class**
**File:** `src/betting/opportunity_scorer.py` (400+ lines)

**Purpose:** Score any betting opportunity on a 0-100 scale

**Key Methods:**
- `score_opportunity()` - Score a single bet
- `rank_opportunities()` - Score and rank a list of bets
- `_score_ev()` - Convert EV to 0-100 score
- `_score_edge()` - Convert edge to 0-100 score
- `_score_liquidity()` - Score based on market volume
- `_score_recency()` - Score based on odds freshness

**Scoring Formula:**
```
Total Score = (EV × 0.40) + (Confidence × 0.25) + (Edge × 0.15) + (Liquidity × 0.10) + (Recency × 0.10)
```

**Ratings:**
- **90-100**: Exceptional - Strong BUY (top tier)
- **75-89**: Great - BUY (highly recommended)
- **60-74**: Good - BUY (recommended)
- **40-59**: Decent - CONSIDER
- **0-39**: Poor - AVOID

---

### **2. BestBetsRanker Service**
**File:** `src/betting/best_bets_ranker.py` (500+ lines)

**Purpose:** Fetch opportunities from ALL sources and rank them

**Key Methods:**
- `get_best_bets()` - Get top N ranked opportunities across all sports
- `get_sport_summary()` - Get count of opportunities by sport
- `_fetch_nfl_opportunities()` - Fetch NFL bets from database
- `_fetch_ncaa_opportunities()` - Fetch NCAA bets from database
- `_fetch_nba_opportunities()` - Fetch NBA bets from database

**Features:**
- Filters out lopsided odds (>85% or <15%)
- Supports minimum EV and confidence filters
- Supports sport-specific filtering
- Supports max odds age filtering
- Returns opportunities sorted by profitability

**Example Usage:**
```python
ranker = BestBetsRanker(db_connection_string)
best_bets = ranker.get_best_bets(
    top_n=20,
    min_ev=0.05,        # 5% minimum EV
    min_confidence=60,  # 60+ confidence
    sports_filter=['NFL', 'NCAA'],
    max_age_hours=24
)
```

---

### **3. Best Bets Across All Sports Page**
**File:** `best_bets_unified_page.py` (350+ lines)

**Purpose:** Streamlit page to view unified rankings

**Features:**

**Filters (Sidebar):**
- Number of bets to show (5-50)
- Minimum EV % (0-30%)
- Minimum confidence (0-100)
- Sports to include (multi-select)
- Max odds age in hours (1-48)

**Summary Metrics:**
- Total opportunities found
- Average Expected Value across all bets
- Average opportunity score
- Number of sports represented

**Bet Display:**
Each bet shows:
- Overall score (0-100) with color coding
- Rating and recommendation
- Game information and betting details
- AI prediction vs market price
- Expected value and edge
- Component score breakdown with progress bars
- Market volume and odds freshness
- Kalshi ticker for easy access

**Color Coding:**
- 🔥 Green: Exceptional/Great (90-75)
- 👍 Gold: Good (74-60)
- ⚠️ Orange: Decent (59-40)
- ❌ Red: Poor (39-0)

---

## 🔄 **How It Works**

### **Step 1: Fetch Opportunities**
BestBetsRanker queries the database for:
- NFL games with Kalshi odds and AI predictions
- NCAA games with Kalshi odds and AI predictions
- NBA games with Kalshi odds and AI predictions
- Future: Politics markets, MLB, NHL, etc.

Filters applied:
- Game time > NOW (future games only)
- Odds updated within last N hours
- Away/home price between 0.15 and 0.85 (no lopsided odds)
- AI prediction exists

### **Step 2: Score Each Opportunity**
OpportunityScorer calculates:

**EV Score (0-100):**
- 30%+ EV → 100 points
- 20% EV → 90 points
- 10% EV → 70 points
- 5% EV → 50 points
- 0% EV → 25 points

**Confidence Score (0-100):**
- Already calculated by AI prediction system

**Edge Score (0-100):**
- 30%+ edge → 100 points
- 20% edge → 85 points
- 10% edge → 65 points

**Liquidity Score (0-100):**
- $1M+ volume → 100 points
- $500K volume → 85 points
- $100K volume → 70 points
- Unknown → 50 points (neutral)

**Recency Score (0-100):**
- <5 min old → 100 points
- 5-15 min old → 85 points
- 15-30 min old → 70 points
- >3 hours old → 25 points

### **Step 3: Calculate Weighted Total**
```
Total = (EV × 40%) + (Confidence × 25%) + (Edge × 15%) + (Liquidity × 10%) + (Recency × 10%)
```

### **Step 4: Rank and Filter**
- Sort by total score (highest first)
- Filter by minimum EV and confidence
- Return top N opportunities

---

## 📊 **Example Output**

```
🏆 Best Bets Across All Sports

Filters Applied:
- Top 20 bets
- Min EV: 5%
- Min Confidence: 60
- Sports: NFL, NCAA, NBA
- Max age: 24 hours

Summary:
- Total Opportunities: 18
- Avg Expected Value: 12.3%
- Avg Opportunity Score: 78.5/100
- Sports Represented: 3

Rankings:
#1 🔥 NFL | Pittsburgh Steelers | Score: 92.1/100 (Exceptional) | EV: 18.2%
   Game: Pittsburgh Steelers @ Chicago Bears
   AI: 62.1% | Market: 52¢ | Edge: +10.1%
   Recommendation: Strong BUY - Top tier opportunity

#2 ✅ NCAA | Georgia Tech Yellow Jackets | Score: 81.4/100 (Great) | EV: 14.8%
   Game: Pittsburgh Panthers @ Georgia Tech Yellow Jackets
   AI: 69.9% | Market: 60¢ | Edge: +9.9%
   Recommendation: BUY - Highly recommended

#3 ✅ NBA | Los Angeles Lakers | Score: 76.3/100 (Great) | EV: 11.2%
   ...
```

---

## 🎯 **Benefits**

### **For the User:**
1. **Single View**: See best bets across ALL sports in one place
2. **Smart Ranking**: No more guessing - system tells you which are most profitable
3. **Time Saving**: Don't manually compare NFL game to NCAA game to NBA game
4. **Confidence**: Color-coded ratings make decisions easy
5. **Transparency**: See exactly how each bet is scored

### **For the System:**
1. **Consistency**: All bets scored using same formula
2. **Extensibility**: Easy to add new sports (politics, MLB, NHL, etc.)
3. **Maintainability**: Single scoring logic in OpportunityScorer class
4. **Reusability**: BestBetsRanker can be used by other components
5. **Testability**: Well-documented, type-hinted code

---

## 🔗 **Integration with Existing Components**

The unified ranking system integrates with:

### **Phase 1 Components:**
- ✅ Uses `UnifiedEVCalculator` for EV calculations
- ✅ Respects lopsided odds filters (>85% or <15%)
- ✅ Uses same confidence calculation methodology
- ✅ Sorts by EV first (profitability priority)

### **Database:**
- ✅ Queries `espn_games` table
- ✅ Joins with `kalshi_markets` table for odds
- ✅ Joins with `ai_predictions` table for predictions
- ✅ Filters by sport, game time, odds age

### **Dashboard:**
Can be integrated into dashboard by:
1. Adding page to navigation
2. Using `best_bets_unified_page.py` as main page
3. Optional: Add "Top 5 Opportunities" widget to main dashboard

---

## 📁 **Files Created/Modified**

### **New Files:**
1. **src/betting/opportunity_scorer.py** (400+ lines)
   - OpportunityScorer class
   - Scoring methods for each factor
   - Rating calculation logic

2. **src/betting/best_bets_ranker.py** (500+ lines)
   - BestBetsRanker service class
   - Database query methods
   - Sport-specific fetch methods

3. **best_bets_unified_page.py** (350+ lines)
   - Streamlit page UI
   - Filter controls
   - Bet display with expandable sections

### **Modified Files:**
1. **src/betting/__init__.py**
   - Added OpportunityScorer export
   - Added BestBetsRanker export

---

## 🧪 **Testing Instructions**

### **Manual Testing:**

1. **Access the Page:**
   - Add to dashboard navigation OR
   - Run directly: `streamlit run best_bets_unified_page.py`

2. **Test Filters:**
   - Adjust "Number of bets to show" slider
   - Change "Minimum EV %" slider
   - Change "Minimum Confidence" slider
   - Select different sports in multi-select
   - Adjust "Max odds age" slider

3. **Verify Results:**
   - Bets are sorted by total score (highest first)
   - Color coding matches score (Green for 90+, Gold for 75+, etc.)
   - EV calculations are reasonable (5-30% range)
   - No lopsided odds (>85% or <15%)
   - Component scores add up correctly

4. **Check Data:**
   - Game information is accurate
   - Market prices match Kalshi
   - AI predictions are reasonable
   - Odds freshness is recent

### **Database Testing:**

```sql
-- Check if there are opportunities available
SELECT
    g.sport,
    COUNT(DISTINCT g.game_id) as game_count
FROM espn_games g
LEFT JOIN kalshi_markets k ON g.kalshi_market_ticker = k.market_ticker
LEFT JOIN ai_predictions a ON g.game_id = a.game_id
WHERE g.game_time > NOW()
  AND k.last_updated > NOW() - INTERVAL '24 hours'
  AND k.away_win_price > 0.15 AND k.away_win_price < 0.85
  AND a.away_win_probability IS NOT NULL
GROUP BY g.sport;
```

---

## 🚀 **Next Steps (Future Enhancements)**

### **Phase 3: Add More Sports**
- Politics markets (presidential elections, etc.)
- MLB games
- NHL games
- Soccer matches
- Tennis matches

### **Phase 4: Advanced Features**
- Historical performance tracking
- Bankroll management integration
- Kelly criterion bet sizing
- Multi-leg parlay recommendations
- Live odds monitoring and alerts

### **Phase 5: Machine Learning**
- Learn from past bets to improve scoring
- Personalized scoring weights based on user preferences
- Automated bet placement (with user approval)

---

## ✅ **Phase 2 Complete!**

**The unified ranking system is ready to use. You now have:**

1. ✅ **OpportunityScorer** - Universal 0-100 scoring system
2. ✅ **BestBetsRanker** - Cross-sport opportunity fetcher
3. ✅ **Best Bets Page** - User-friendly interface
4. ✅ **Transparent Methodology** - See exactly how bets are scored
5. ✅ **Extensible Architecture** - Easy to add new sports

**You can now answer the question:**
> **"Which bets have the best odds of making money?"**

Just open the **Best Bets Across All Sports** page and see the top-ranked opportunities!

---

## 📖 **Quick Reference**

**To use in Python:**
```python
from src.betting.best_bets_ranker import BestBetsRanker

ranker = BestBetsRanker(db_connection_string)
top_20 = ranker.get_best_bets(top_n=20, min_ev=0.05, min_confidence=60)

for bet in top_20:
    print(f"{bet['sport']} | {bet['team']} | Score: {bet['total_score']}/100")
```

**To view in Streamlit:**
```bash
streamlit run best_bets_unified_page.py
```

---

**Ready to find the most profitable bets across all sports!** 🏆
