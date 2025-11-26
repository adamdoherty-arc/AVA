# AVA Betting Picks Bug Fix Summary

## Problem
AVA Betting Picks page was showing "No betting markets found for current games" even though:
- ESPN API was returning 14 games
- Kalshi database had 6,227 active markets
- The page should have shown betting recommendations

## Root Cause
**SQL Parameter Escaping Bug in [espn_kalshi_matcher.py](src/espn_kalshi_matcher.py)**

The query contained unescaped `%` signs in LIKE clauses:
```sql
OR ticker LIKE 'KXNFLGAME%'
OR ticker LIKE 'KXNCAAFGAME%'
```

In psycopg2 (PostgreSQL Python driver), the `%` character is used for parameter placeholders (`%s`). When a literal `%` appears in the SQL string (like in `LIKE 'pattern%'`), it must be escaped as `%%`.

### Error Details
- **Error**: `IndexError: tuple index out of range`
- **Location**: Line 202 in `src/espn_kalshi_matcher.py`
- **Cause**: psycopg2 was counting the `%` in `'KXNFLGAME%'` as a parameter placeholder
- **Result**: Expected 6 parameters but found 8+ placeholders, causing the tuple index error

## Fixes Applied

### 1. Fixed SQL Parameter Escaping
**File**: `src/espn_kalshi_matcher.py`

Changed:
```sql
OR ticker LIKE 'KXNFLGAME%'
OR ticker LIKE 'KXNCAAFGAME%'
```

To:
```sql
OR ticker LIKE 'KXNFLGAME%%'
OR ticker LIKE 'KXNCAAFGAME%%'
```

### 2. Fixed DateTime Handling
Added proper handling for datetime objects from ESPN:

```python
# Extract date from game_time (can be datetime object or string)
try:
    if game_time:
        if isinstance(game_time, datetime):
            game_date = game_time.date()
        elif isinstance(game_time, str):
            game_date = datetime.strptime(game_time[:10], '%Y-%m-%d').date()
        else:
            game_date = datetime.now().date()
    else:
        game_date = datetime.now().date()
except Exception as e:
    logger.warning(f"Could not parse game_time {game_time}: {e}")
    game_date = datetime.now().date()
```

### 3. Enhanced Error Logging
Added `exc_info=True` to error logging for better debugging:
```python
logger.error(f"Error matching game to Kalshi: {e}", exc_info=True)
```

## Test Results
After fixes:
- ✅ 14 ESPN games fetched successfully
- ✅ 12/14 games matched with Kalshi markets (86% success rate)
- ✅ Kalshi odds retrieved and displayed correctly

## Matched Games Example
```
Buffalo Bills @ Houston Texans: 72.00% / 28.00%
Pittsburgh Steelers @ Chicago Bears: 42.00% / 58.00%
New England Patriots @ Cincinnati Bengals: 74.00% / 26.00%
Minnesota Vikings @ Green Bay Packers: 29.00% / 71.00%
... and 8 more
```

## How to Verify the Fix
1. Navigate to **AVA Betting Picks** page in the dashboard
2. You should now see active betting opportunities with:
   - Game matchups
   - Kalshi odds
   - AI-powered recommendations
   - Kelly Criterion bet sizing

## Files Modified
- [src/espn_kalshi_matcher.py](src/espn_kalshi_matcher.py) - Main fix for SQL escaping and datetime handling

## Technical Notes
- This is a common pitfall when using psycopg2 with pattern matching in SQL
- Always escape literal `%` signs as `%%` in psycopg2 queries
- The same issue would occur with `_` (underscore) which should be escaped as `\\_` if literal

## Status
✅ **FIXED** - AVA Betting Picks page is now fully functional
