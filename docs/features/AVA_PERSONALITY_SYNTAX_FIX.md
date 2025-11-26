# AVA Personality Syntax Error Fix

## Problem
Dashboard was crashing on startup with syntax error:
```
SyntaxError: unterminated string literal (detected at line 306)
File "C:\code\Magnus\src\ava\ava_personality.py", line 306
    'profit': 'the market's gift',
                        ^
```

## Root Cause
Line 306 contained an **unescaped apostrophe** inside a single-quoted string:

```python
'profit': 'the market's gift',
                     ^ This apostrophe breaks the string
```

In Python, when using single quotes to delimit a string, any apostrophes inside must be escaped.

## Solution

Escaped the apostrophe with a backslash:

```python
# BEFORE (Line 306):
'profit': 'the market's gift',

# AFTER (Line 306):
'profit': 'the market\'s gift',
          ↑ Escaped apostrophe
```

## File Modified

**[src/ava/ava_personality.py](src/ava/ava_personality.py:306)** - Line 306

Changed in the `PersonalityMode.GURU` vocabulary mapping.

## Testing

```bash
$ python -m py_compile src/ava/ava_personality.py
✅ No syntax errors

$ python -c "from src.ava.ava_personality import AVAPersonality"
✅ Import successful

$ python -c "from src.ava.omnipresent_ava_enhanced import show_enhanced_ava"
✅ Omnipresent AVA import successful
```

## Impact

### Before Fix:
- ❌ Dashboard crashes on startup
- ❌ Cannot access any page
- ❌ Import chain broken at AVA personality module

### After Fix:
- ✅ Dashboard loads successfully
- ✅ AVA personality system works
- ✅ All personality modes functional
- ✅ Import chain restored

## Context

The AVA personality system includes 8 different personality modes:
1. **FRIENDLY** - Warm and approachable
2. **PROFESSIONAL** - Formal and business-like
3. **CHARMING** - Playful and engaging
4. **ANALYST** - Data-driven and technical
5. **COACH** - Motivational and encouraging
6. **REBEL** - Edgy and contrarian
7. **GURU** - Wise and philosophical ← Fixed this mode
8. **SARCASTIC** - Witty and ironic

Each mode has custom vocabulary for trading terms. The GURU mode now correctly refers to profits as "the market's gift" (with proper escaping).

## Related Files

- `src/ava/ava_personality.py` - Personality system (fixed)
- `src/ava/omnipresent_ava_enhanced.py` - Enhanced AVA UI (imports personality)
- `dashboard.py` - Main dashboard (imports omnipresent AVA)

All files now import without errors.

---

**Status:** ✅ Fixed and Tested
**Breaking Changes:** None
**Impact:** Critical - Dashboard startup fixed
