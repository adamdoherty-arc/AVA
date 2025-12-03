# Chart Patterns for Options Traders

## Introduction to Chart Patterns

Chart patterns are visual formations on price charts that have historically preceded specific price movements. They represent the psychology of buyers and sellers and can help predict future direction.

## Categories of Patterns

### Reversal Patterns

Signal a change in trend direction:
- Head and Shoulders
- Double Top / Double Bottom
- Triple Top / Triple Bottom
- Rounding Top / Bottom

### Continuation Patterns

Signal trend will continue after consolidation:
- Flags and Pennants
- Triangles (Ascending, Descending, Symmetrical)
- Rectangles
- Wedges

## Major Reversal Patterns

### Head and Shoulders (Top)

```
Structure:
    Head
   /    \
  /      \
Left      Right
Shoulder  Shoulder
  |________|
  Neckline

Formation:
1. Left shoulder: Rally, pullback
2. Head: Higher rally, pullback to neckline
3. Right shoulder: Lower rally, break neckline

Target: Distance from head to neckline, projected down
```

**Options Application**:
```
Setup: When right shoulder forming
Strategy: Bear put spread or long puts
Entry: On neckline break with volume
Target: Measured move down
Stop: Above right shoulder
```

### Inverse Head and Shoulders (Bottom)

```
Structure (mirror image of top):
  Neckline
  |________|
Left      Right
Shoulder  Shoulder
  \      /
   \    /
    Head

Target: Distance from head to neckline, projected up
```

**Options Application**:
```
Setup: When right shoulder forming
Strategy: Bull call spread or CSP
Entry: On neckline break with volume
Target: Measured move up
Stop: Below right shoulder
```

### Double Top

```
Structure:
  Peak 1   Peak 2
    /\      /\
   /  \    /  \
  /    \  /    \
_/      \/      \_
      Support

Signals: Resistance proven twice, reversal likely
```

**Options Application**:
```
Setup: Second peak rejection
Strategy: Bear call spread
Entry: Break below support between peaks
Target: Distance between peaks and support, down
Stop: Above second peak
```

### Double Bottom

```
Structure:
_      Support      _
  \      /\      /
   \    /  \    /
    \  /    \  /
     \/      \/
  Bottom1  Bottom2

Signals: Support proven twice, reversal likely
```

**Options Application**:
```
Setup: Second bottom holding
Strategy: Bull put spread or CSP
Entry: Break above resistance between bottoms
Target: Distance between bottoms and resistance, up
Stop: Below second bottom
```

## Continuation Patterns

### Bull Flag

```
Structure:
           |
          /|
         / |
        /  |    <- Flag (downward slope)
   Pole/   |
      /    |
     /     |
    /

Formation: Strong move up (pole), shallow pullback (flag)
Signal: Trend continuation upward
```

**Options Application**:
```
Strategy: Bull call spread or long call
Entry: Break above flag resistance
Target: Length of pole, measured from breakout
Timing: Best in first half of flag formation
```

### Bear Flag

```
Structure:
    \
     \
      \    <- Flag (upward slope)
       \  |
        \ |
         \|
          |
          |Pole

Formation: Strong move down (pole), shallow bounce (flag)
Signal: Trend continuation downward
```

**Options Application**:
```
Strategy: Bear put spread or long put
Entry: Break below flag support
Target: Length of pole, measured from breakdown
```

### Ascending Triangle

```
Structure:
        ___________
       /          |
      /           |
     /____________|
        Flat Top

Formation: Flat resistance, rising support
Signal: Usually bullish breakout
```

**Options Application**:
```
Strategy: Bull call spread
Entry: Break above flat resistance with volume
Target: Height of triangle, measured from breakout
Wait: Don't enter in the triangle - wait for break
```

### Descending Triangle

```
Structure:
     ____________
     |          \
     |           \
     |____________\
     Flat Bottom

Formation: Flat support, falling resistance
Signal: Usually bearish breakdown
```

**Options Application**:
```
Strategy: Bear put spread
Entry: Break below flat support with volume
Target: Height of triangle, measured from breakdown
```

### Symmetrical Triangle

```
Structure:
      /\
     /  \
    /    \
   /      \
  /________\

Formation: Converging trendlines
Signal: Breakout direction determines trade (neutral pattern)
```

**Options Application**:
```
Wait for breakout - don't guess direction
Entry: Confirmed break with volume
Strategy: Depends on breakout direction
Target: Widest part of triangle
```

## Wedge Patterns

### Rising Wedge (Bearish)

```
Structure:
       /|
      / |
     /  |
    /   |
   /    |
  Both lines rising, converging

Signal: Bearish - breakdown likely
Context: Occurs in uptrends as reversal or downtrends as continuation
```

**Options Application**:
```
Strategy: Bear put spread when pattern completes
Entry: Break below lower trendline
Stop: Above pattern high
```

### Falling Wedge (Bullish)

```
Structure:
  |
  |\
  | \
  |  \
  |   \
  Both lines falling, converging

Signal: Bullish - breakout likely
Context: Occurs in downtrends as reversal or uptrends as continuation
```

**Options Application**:
```
Strategy: Bull call spread when pattern completes
Entry: Break above upper trendline
Stop: Below pattern low
```

## Rectangle Patterns

### Bullish Rectangle

```
Structure:
_______________
|              |
|  Trading    |
|   Range     |
|______________|
(in uptrend)

Signal: Consolidation before continuation up
```

**Options Application**:
```
Within pattern: Iron condors (range-bound)
On breakout: Bull call spread
Entry: Break above rectangle top with volume
Target: Height of rectangle, projected up
```

### Bearish Rectangle

```
Structure:
_______________
|              |
|  Trading    |
|   Range     |
|______________|
(in downtrend)

Signal: Consolidation before continuation down
```

## Cup and Handle

### Structure

```
         |
        /|
       / |  Handle
      |  |
      |  |
      |  |
       \_/ Cup

Formation:
1. Rounded bottom (cup) - 7-65 weeks typical
2. Small pullback (handle) - 1-4 weeks
3. Breakout above handle resistance
```

**Options Application**:
```
Strategy: Long calls or bull call spread
Entry: Break above handle with volume
Target: Depth of cup, projected up
Stop: Below handle low
Note: One of most reliable bullish patterns
```

## Pattern Recognition Guidelines

### Confirmation Requirements

```
For any pattern:
1. Clear, recognizable structure
2. Volume confirms (high on breaks, low during pattern)
3. Break with follow-through
4. Not too "perfect" (ideal patterns often fail)
```

### Time Frame Considerations

```
Pattern reliability by time frame:
- Weekly: Most reliable
- Daily: Very reliable
- 4-hour: Moderately reliable
- 1-hour and below: Less reliable

Longer time frame = More significant pattern
```

### Volume Patterns

```
Reversal patterns:
- Volume should decline during pattern formation
- Volume should spike on breakout/breakdown

Continuation patterns:
- Volume declines during consolidation
- Volume expands on breakout
```

## Measuring Pattern Targets

### The Measured Move

```
General rule:
Target = Breakout point ± Pattern height

Head & Shoulders:
Target = Neckline - (Head to Neckline distance)

Triangle:
Target = Breakout ± (Widest part of triangle)

Flag:
Target = Breakout + Pole length
```

### Success Rates

```
Approximate success rates (reaching target):
- Head & Shoulders: 65-70%
- Double Top/Bottom: 70-75%
- Triangles: 60-70%
- Flags/Pennants: 65-70%
- Cup & Handle: 75-80%

Note: Success improves with volume confirmation
```

## Options Strategy by Pattern

### Bullish Patterns

| Pattern | Best Strategy | Entry Point |
|---------|--------------|-------------|
| Inv H&S | Bull call spread | Neckline break |
| Double Bottom | CSP or bull put spread | Above resistance |
| Asc Triangle | Long call | Above flat top |
| Bull Flag | Bull call spread | Above flag |
| Cup & Handle | Long call | Above handle |

### Bearish Patterns

| Pattern | Best Strategy | Entry Point |
|---------|--------------|-------------|
| H&S Top | Bear put spread | Below neckline |
| Double Top | Bear call spread | Below support |
| Desc Triangle | Long put | Below flat bottom |
| Bear Flag | Bear put spread | Below flag |
| Rising Wedge | Bear put spread | Below lower line |

## Common Pattern Mistakes

### Mistake 1: Forcing Patterns

```
Wrong: Seeing patterns everywhere
Right: Wait for clear, obvious patterns

If you have to squint to see it, it's not there
```

### Mistake 2: Early Entry

```
Wrong: Entering before pattern completes
Right: Wait for confirmed breakout

Patience is key - let the pattern prove itself
```

### Mistake 3: Ignoring Volume

```
Wrong: Trading pattern without volume confirmation
Right: Require volume on breakout

Volume validates the pattern
```

### Mistake 4: Ignoring Context

```
Wrong: Trading pattern against the trend
Right: Pattern in direction of trend more reliable

Reversal patterns at trend extremes work best
Continuation patterns within trends work best
```

### Mistake 5: Fixed Targets

```
Wrong: Always expecting exact measured move
Right: Using measured move as guide, not guarantee

Be flexible - market doesn't know about your targets
```

## Pattern Trading Checklist

### Before Entry

```
□ Is the pattern clear and obvious?
□ Is it in the right context (trend)?
□ Is volume confirming?
□ Is the breakout/breakdown confirmed?
□ Have I defined my target and stop?
□ Does the risk/reward make sense?
```

### During Trade

```
□ Is volume following through?
□ Is price acting as expected?
□ Should I adjust stop?
□ Is target still realistic?
```

## Conclusion

Chart patterns are valuable tools when used correctly:

1. **Be selective** - Only trade clear patterns
2. **Wait for confirmation** - Don't anticipate
3. **Use volume** - It validates patterns
4. **Respect context** - Trend matters
5. **Define risk** - Always have a stop
6. **Be realistic** - Not all patterns work

**For options traders**: Patterns help with direction and timing. Use them to select strategies and time entries. The measured move helps with strike selection and profit targets.

**Remember**: Patterns are probabilities, not certainties. Proper risk management is still essential.
