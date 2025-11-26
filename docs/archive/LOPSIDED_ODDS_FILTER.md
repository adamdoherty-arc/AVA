# 🎯 Hide Lopsided Odds Filter - New Feature

## What It Does

Filters out games where one team is heavily favored (like 96% odds), which means low payout potential. These games aren't worth betting on because even if you win, you won't make much money.

## How to Use

### 1. Enable the Filter

In the Sports Game Hub (NFL, NCAA, or NBA tabs):

1. Look for the **third row of filters**
2. Check the box: **"🎯 Hide Lopsided Odds"**

### 2. Adjust the Threshold (Optional)

When the filter is enabled, a slider appears:
- **Max Odds %**: Set the maximum odds percentage (default: 90%)
- Range: 70% to 99%

**Example**:
- Set to 90%: Hides games where either team has >90% odds
- Set to 80%: More aggressive - hides games with >80% odds (shows only competitive games)
- Set to 95%: Less aggressive - only hides extremely lopsided games

## Why This Matters

### Lopsided Games = Low Profit

**Example Game**:
```
Team A: 96% chance to win
Team B: 4% chance to win
```

**If you bet $100 on Team A (the favorite)**:
- Win probability: 96%
- Payout if you win: ~$104 ($4 profit)
- Risk: $100
- **Not worth it!** Only $4 profit for $100 risk

**Better games to bet on**:
```
Team A: 55% chance
Team B: 45% chance
```
- More competitive odds
- Higher potential payout
- Better risk/reward ratio

## Filter Location

### NFL & NCAA Tabs
```
Row 1: Sort By | Game Status | Money Filter | Min EV % | Cards/Row | Hide Final

Row 2: Date Filter | Custom Range | Auto-Refresh | Interval

Row 3: 🎯 Hide Lopsided Odds | Max Odds %  ← NEW FILTER HERE
```

### NBA Tab
```
Row 1: Sort By | Game Status | Money Filter | Min EV % | Cards/Row | Hide Final

Row 2: Date Filter | Custom Range

Row 3: 🎯 Hide Lopsided Odds | Max Odds %  ← NEW FILTER HERE
```

## Technical Details

**Filtering Logic**:
```python
# For each game, check both team odds
yes_price = game.get('yes_price', 0.5)  # Team A odds
no_price = game.get('no_price', 0.5)    # Team B odds

# If EITHER team has odds above threshold, hide the game
if yes_price > threshold or no_price > threshold:
    # Game is lopsided - hide it
    return False
```

**Example with 90% threshold**:
- Game 1: 96% vs 4% → **HIDDEN** (96% > 90%)
- Game 2: 85% vs 15% → **SHOWN** (both < 90%)
- Game 3: 55% vs 45% → **SHOWN** (both < 90%)
- Game 4: 92% vs 8% → **HIDDEN** (92% > 90%)

## Use Cases

### 1. Profit-Focused Betting
**Goal**: Find games with good payout potential
**Setting**: Enable filter, set threshold to 85-90%
**Result**: Only see competitive games worth betting on

### 2. Aggressive Filtering
**Goal**: Only see extremely competitive games
**Setting**: Enable filter, set threshold to 75-80%
**Result**: Very few games, but all highly competitive

### 3. Conservative Approach
**Goal**: Filter only the most extreme mismatches
**Setting**: Enable filter, set threshold to 95%
**Result**: Still see most games, but hide obvious blowouts

## Benefits

✅ **Save Time**: Don't waste time reviewing bad betting opportunities
✅ **Focus on Profit**: See only games with worthwhile payouts
✅ **Better Strategy**: Avoid low-value bets
✅ **Customizable**: Adjust threshold to your risk tolerance

## Combined with Other Filters

**Smart Betting Strategy**:
1. Enable "Hide Lopsided Odds" (90% threshold)
2. Set "Min EV %" to 10%
3. Sort by "🎯 Best Odds"

**Result**:
- Only competitive games (not lopsided)
- With 10%+ expected value
- Sorted by best opportunities first

## What's Filtered

**Automatically Hidden**:
- Blowout predictions (96-4%, 92-8%, etc.)
- Games with no competitive betting value
- Heavily favored matchups

**Still Shown**:
- Competitive games (55-45%, 60-40%, etc.)
- Underdogs with a real chance
- Games with good payout potential

## Status

✅ **Complete and Ready**
- Added to NFL tab
- Added to NCAA tab
- Added to NBA tab
- Filter logic implemented
- Threshold slider working

**Just restart Streamlit to use!**

---

**After restart, try it out**:
1. Go to NFL tab
2. Check "🎯 Hide Lopsided Odds"
3. See how many games disappear (the non-profitable ones)
4. Adjust slider to find your sweet spot
