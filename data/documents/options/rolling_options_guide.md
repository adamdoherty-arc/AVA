# Rolling Options: Complete Guide

## What is Rolling?

Rolling is the process of closing an existing options position and simultaneously opening a new one with different parameters (strike, expiration, or both). It's a critical skill for active options traders.

## Why Roll Options?

### Defensive Reasons

- **Avoid assignment**: Keep shares or avoid buying stock
- **Manage losers**: Reduce loss or extend time to recover
- **Delay decision**: Buy time when uncertain

### Offensive Reasons

- **Capture more premium**: Extend profitable trades
- **Improve position**: Better strike or expiration
- **Lock in gains**: Roll to safer position

## Types of Rolls

### Roll Out (Same Strike, Later Expiration)

**When to use**: Position is profitable or neutral, want more time

```
Example - Cash Secured Put:
Original: Sold AAPL $170 put expiring in 7 days for $2.00
Current value: $0.50 (profitable)

Roll to: Sell AAPL $170 put expiring in 37 days for $3.00

Transaction:
- Buy back current put: -$0.50
- Sell new put: +$3.00
- Net credit: $2.50

Total collected: $2.00 + $2.50 = $4.50
```

### Roll Up (Higher Strike, Same Expiration)

**When to use**: Stock has risen, want to capture upside

```
Example - Covered Call:
Original: Sold AAPL $175 call when stock at $170
Stock now at $180

Roll to: $180 or $185 call

Transaction:
- Buy back $175 call: -$6.00 (now ITM)
- Sell $185 call: +$3.00
- Net debit: $3.00

Result: Higher strike allows more upside participation
```

### Roll Down (Lower Strike, Same Expiration)

**When to use**: Stock has fallen, want better position

```
Example - Cash Secured Put:
Original: Sold AAPL $175 put when stock at $180
Stock now at $165

Roll to: $165 put (current ATM)

Transaction:
- Buy back $175 put: -$11.00 (now ITM)
- Sell $165 put: +$5.00
- Net debit: $6.00

Result: Lower strike means lower assignment price
Tradeoff: Realized $6 loss to reset position
```

### Roll Down and Out (Lower Strike, Later Expiration)

**When to use**: Defending losing put position

```
Example - Cash Secured Put:
Original: Sold AAPL $175 put, 14 DTE, for $3.00
Stock drops to $165, put now worth $11.00

Roll to: AAPL $170 put, 45 DTE

Transaction:
- Buy back $175 put: -$11.00
- Sell $170 put (45 DTE): +$9.00
- Net debit: $2.00

Analysis:
- Original credit: $3.00
- Roll debit: -$2.00
- Net credit: $1.00
- New strike: $170 (vs $175)
- New breakeven: $169 (vs $172)
- Additional time for recovery
```

### Roll Up and Out (Higher Strike, Later Expiration)

**When to use**: Defending covered call when stock rallies

```
Example - Covered Call:
Original: Sold AAPL $175 call when stock at $170
Stock rallies to $182, call worth $8.00

Roll to: AAPL $185 call, 45 DTE

Transaction:
- Buy back $175 call: -$8.00
- Sell $185 call (45 DTE): +$5.00
- Net debit: $3.00

Result: Keep shares, higher strike for potential exit
Trade-off: Pay $3 for opportunity to sell at $185 vs $175
```

## Rolling Rules

### The Credit Rule

**Best practice**: Only roll for a net credit or very small debit

```
Good roll: Net credit of any amount
Acceptable: Net debit less than 20% of new premium
Avoid: Large debit to roll

Why: Rolling for debit compounds losses
Exception: Rolling to take assignment at better price
```

### The "Don't Chase" Rule

**Know when to stop rolling**:
- Don't roll more than 2-3 times
- Accept loss if stock fundamentally changed
- Continuing to roll ties up capital

```
Example of bad rolling:
Trade 1: Sold $100 put, stock drops to $90
Roll 1: To $95 put for small credit
Stock drops to $85
Roll 2: To $90 put for small credit
Stock drops to $80
Roll 3: To $85 put for tiny credit

Problem: Chasing a falling stock
Better: Take assignment or close for loss
```

### The Time Rule

**Roll before expiration week** when possible:
- Gamma risk highest in final week
- More premium available further out
- Wider bid-ask spreads near expiration

### The Strike Rule

**Never roll to a worse strike just for credit**:
- Rolling up puts = accepting assignment at higher price
- Rolling down calls = capping gains at lower level
- Only roll to same or better strike

## Rolling Specific Strategies

### Rolling Cash-Secured Puts

**Profitable CSP**:
```
When: 50%+ profit achieved
Action: Roll out to same strike, later expiration
Result: Capture more premium on winning position
```

**Threatened CSP**:
```
When: Stock approaching strike, still above
Action: Roll down and out for credit if possible
Result: Lower strike, more time
```

**Losing CSP**:
```
When: Stock below strike
Options:
1. Take assignment (if still want stock)
2. Roll down and out for credit
3. Close for loss (if thesis broken)
```

### Rolling Covered Calls

**Profitable CC**:
```
When: Stock flat or down, call near worthless
Action: Roll out to same strike
Result: Collect more premium
```

**Threatened CC**:
```
When: Stock approaching or above strike
Options:
1. Let shares be called away (take profit)
2. Roll up and out for credit
3. Roll up and out for small debit (if want shares)
```

**ITM CC near expiration**:
```
When: Stock significantly above strike
Action: Roll up and out to delay assignment
Consideration: May need to pay debit
Alternative: Accept assignment, restart wheel
```

### Rolling Vertical Spreads

**Credit Spread Roll**:
```
When: Short strike threatened
Action: Roll entire spread down/up and out
Goal: Collect additional credit, move strikes
```

**Example - Bull Put Spread**:
```
Original: $95/$90 put spread for $1.50 credit
Stock drops, spread now worth $3.00

Roll to: $90/$85 put spread, 30 more DTE
- Buy back $95/$90 spread: -$3.00
- Sell $90/$85 spread: +$2.00
- Net debit: $1.00

New position at lower strikes with more time
```

### Rolling Iron Condors

**One Side Threatened**:
```
When: Stock moving toward one side
Action: Roll the threatened side away

Example:
Original: $95/$90 puts, $105/$110 calls
Stock rises to $104

Action: Roll call spread
- Buy back $105/$110 calls
- Sell $108/$113 calls
- Usually for credit or small debit
```

**Both Sides Threatened** (unusual):
```
When: Stock whipsawing violently
Action: Consider closing entirely
Note: Iron condors aren't designed for high volatility
```

## Rolling Mechanics

### Order Types

**Single order** (preferred):
- Use "Roll" or "Spread" order type
- Ensures simultaneous execution
- Better pricing than separate orders

**Separate orders** (if needed):
- Close existing position first
- Then open new position
- Risk of price movement between orders

### Timing Considerations

**Best time to roll**:
- Market open (after first 15-30 minutes)
- When bid-ask spreads are tightest
- Not during volatility spikes

**Avoid rolling**:
- Right at market open/close
- During news events
- In illiquid options

### Commission Impact

```
Rolling = 2 trades (close + open)
- 4 legs for vertical spread roll
- 8 legs for iron condor full roll

Factor commissions into roll decision
May not be worth rolling small positions
```

## When NOT to Roll

### Fundamental Change

Stock thesis has changed:
- Company in trouble
- Sector collapse
- Major negative news

**Action**: Close position, don't roll

### Better Opportunities

Capital tied up in underwater position could earn more elsewhere:
- Opportunity cost of rolling
- Fresh position may be better
- Don't get married to losing trades

### Already Rolled Multiple Times

After 2-3 rolls, consider:
- Taking the loss
- Taking assignment
- Moving on

Rolling becomes "hoping" rather than managing

### Wide Bid-Ask Spread

If roll pricing is unfavorable:
- Wait for better prices
- Accept current outcome
- Close for loss

## Roll Decision Matrix

### CSP Roll Decision

| Stock Position | Profit/Loss | Action |
|---------------|-------------|--------|
| Above strike | Profitable | Roll out or close |
| At strike | Break-even | Roll down/out or close |
| Below strike (small) | Small loss | Roll down/out for credit |
| Below strike (large) | Big loss | Take assignment or close |

### Covered Call Roll Decision

| Stock Position | Profit/Loss | Action |
|---------------|-------------|--------|
| Below strike | Profitable | Roll out or close |
| At strike | At max profit | Roll up/out or let assign |
| Above strike (small) | Max profit | Roll up/out or let assign |
| Above strike (large) | Max profit | Let assign or roll for debit |

## Rolling Cost Analysis

### Break-Even Calculation

```
Original trade: Sold $100 put for $3.00
Stock at $95, put worth $6.00

Roll to $97 put, 30 DTE:
- Buy back: -$6.00
- Sell new: +$5.00
- Net debit: $1.00

Total credits: $3.00 - $1.00 = $2.00
New breakeven: $97 - $2.00 = $95

If stock above $95 at new expiration = profit
```

### Opportunity Cost

```
Capital tied up in rolled position: $9,500 ($95 strike)
Days to new expiration: 30
Potential premium elsewhere: $300 (from fresh CSP)

If roll premium is less than $300,
fresh position may be better use of capital
```

## Practical Examples

### Example 1: Successful CSP Roll

```
Week 1: Sell AAPL $170 put for $3.00 (45 DTE)
Week 3: AAPL at $175, put worth $0.75

Action: Roll out
- Buy back $170 put: -$0.75
- Sell $170 put (45 DTE): +$3.50
- Net credit: $2.75

Total collected: $3.00 + $2.75 = $5.75
Days in trade: 90
Annualized return: ~25%
```

### Example 2: Defensive CSP Roll

```
Week 1: Sell NVDA $400 put for $8.00 (30 DTE)
Week 2: NVDA drops to $390, put worth $18.00

Analysis:
- Current loss: $10.00
- Stock still above strike
- Fundamentals unchanged

Action: Roll down and out
- Buy back $400 put: -$18.00
- Sell $390 put (45 DTE): +$15.00
- Net debit: $3.00

New position:
- Total credits: $8.00 - $3.00 = $5.00
- New strike: $390 (was $400)
- New breakeven: $385 (was $392)
- More time for recovery
```

### Example 3: When Not to Roll

```
Week 1: Sell XYZ $50 put for $2.00
Week 3: Company announces bad earnings, stock at $35

Put now worth $16.00
Roll would require significant debit
Fundamentals have changed

Action: Do NOT roll
- Close for $14.00 loss, or
- Take assignment and sell stock

Learning: Sometimes accepting loss is correct decision
```

## Conclusion

Rolling is an essential skill that separates successful options traders from beginners. Key principles:

1. **Roll for credit** when possible
2. **Roll early**, not at expiration
3. **Don't chase** losing positions endlessly
4. **Know when to stop** and accept outcomes
5. **Consider opportunity cost** of tied-up capital

Master rolling to manage positions effectively and improve long-term returns.
