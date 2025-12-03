# Maximum Loss Guidelines

## Why Max Loss Matters

Maximum loss rules protect you from the two greatest threats to trading capital: catastrophic single-trade losses and death by a thousand cuts from undisciplined trading.

## Defining Maximum Loss

### Per-Trade Max Loss

**The golden rule**: Never risk more than 1-2% of your account on a single trade.

```
Account Size: $100,000
1% max loss: $1,000 per trade
2% max loss: $2,000 per trade

Why this works:
- 10 consecutive losses at 1% = 9.6% drawdown (recoverable)
- 10 consecutive losses at 2% = 18.3% drawdown (challenging but manageable)
- 10 consecutive losses at 5% = 40.1% drawdown (dangerous)
- 10 consecutive losses at 10% = 65.1% drawdown (potentially fatal)
```

### Calculating Max Loss for Options

**For long options (buying)**:
```
Max loss = Premium paid × 100 × Contracts

Example:
Buy 3 calls at $2.50 each
Max loss = $2.50 × 100 × 3 = $750
```

**For short options (selling)**:

Cash-Secured Puts:
```
Theoretical max loss = (Strike - $0) × 100 × Contracts
Realistic max loss = (Strike × 30-50%) × 100 × Contracts

Example: Sell $100 put
Theoretical max: $10,000 if stock goes to $0
Realistic max: $3,000-$5,000 (30-50% drop)
```

Covered Calls:
```
Max loss = (Stock purchase price - $0 + Premium received) × Shares
Realistic max loss = Stock purchase price × 30-50%
```

Credit Spreads:
```
Max loss = (Spread width - Credit) × 100 × Contracts

Example: $5 wide spread, $2 credit
Max loss = ($5 - $2) × 100 = $300 per spread
```

## Stop Loss Placement

### Technical Stop Losses

**Support-based stops**:
```
For long positions:
- Place stop below nearest significant support
- Account for normal volatility (use ATR)
- Avoid round numbers (use $99.85 instead of $100)
```

**Volatility-based stops**:
```
Stop distance = ATR × Multiplier

Example:
Stock ATR: $2.50
Multiplier: 2
Stop distance: $5.00 below entry
```

**Percentage-based stops**:
```
Typical ranges:
- Swing trades: 5-10% stop
- Position trades: 10-15% stop
- Options: Define by premium at risk
```

### Options-Specific Stop Rules

**For long options**:
```
Rule: Close when option loses 50% of premium

Example:
Bought call for $4.00
Stop at: $2.00 option price
```

**For short options**:
```
Rule: Close when loss equals 2x premium received

Example:
Sold put for $3.00
Stop when put is worth: $6.00 (net loss of $3.00)
```

**For spreads**:
```
Rule: Close when reaching 50% of max loss

Example:
Credit spread max loss: $300
Stop at: $150 loss (spread worth $150 more than credit)
```

## Daily and Weekly Loss Limits

### Daily Loss Limit

**Rule**: Stop trading when daily loss reaches 2-3% of account

```
$100,000 account:
Daily loss limit: $2,000-$3,000

Implementation:
1. Track P&L in real-time
2. When limit hit, close all positions
3. No new trades for rest of day
4. Review what went wrong
```

**Why this works**:
- Prevents emotional "revenge trading"
- Limits damage from bad market conditions
- Forces discipline
- Preserves capital for better opportunities

### Weekly Loss Limit

**Rule**: Stop trading when weekly loss reaches 5-6% of account

```
$100,000 account:
Weekly loss limit: $5,000-$6,000

If hit:
1. Close all positions by Friday
2. No trading the following Monday
3. Comprehensive review required
4. Reduced position sizes upon return
```

### Monthly Drawdown Limits

**Drawdown response protocol**:

| Drawdown | Action |
|----------|--------|
| 5% | Review trades, continue normal |
| 10% | Reduce position sizes 50% |
| 15% | Reduce sizes 75%, weekly only |
| 20% | Stop trading, full strategy review |

## Emergency Exit Rules

### The "Stop Everything" Trigger

**When to close all positions immediately**:
1. Account down 20% from peak
2. Black swan event affecting your positions
3. Personal emergency affecting judgment
4. System/broker failure
5. Position concentrations exceeded

### Flash Crash Protocol

```
Market drops 5%+ in minutes:

Step 1: Do NOT panic sell
Step 2: Assess damage to positions
Step 3: Check if stops were hit (may have slippage)
Step 4: Evaluate if positions should be closed or held
Step 5: Document everything for review
```

### Position Liquidation Order

**If forced to liquidate, prioritize**:
1. Close largest losses first
2. Close most leveraged positions
3. Close positions with highest gamma risk
4. Close positions approaching expiration
5. Keep best-performing positions last

## Loss Recovery Rules

### The Math of Recovery

```
Recovery required for each loss level:
| Loss | Recovery Needed |
|------|-----------------|
| 5%   | 5.3%           |
| 10%  | 11.1%          |
| 15%  | 17.6%          |
| 20%  | 25%            |
| 25%  | 33%            |
| 30%  | 43%            |
| 40%  | 67%            |
| 50%  | 100%           |
```

This is why preventing large losses is crucial - recovery becomes exponentially harder.

### Post-Loss Trading Rules

**After a significant loss (>5%)**:
```
Week 1:
- Trade at 50% normal size
- Maximum 2-3 trades
- Focus on highest-conviction setups

Week 2:
- If profitable week 1, move to 75% size
- If losing, remain at 50%

Week 3:
- If two profitable weeks, return to normal
- Otherwise, continue reduced sizing
```

### Breaking Loss Streaks

**After 5 consecutive losses**:
1. Stop trading for 24 hours minimum
2. Review all 5 trades for common errors
3. Paper trade next 3 setups before real money
4. Return at 50% position size

## Position-Specific Max Loss

### By Strategy Type

**Cash-Secured Puts**:
```
Max loss per position: 3-5% of account
Reasoning: Can result in stock ownership at unfavorable prices
Example: $100K account, max $3K-$5K at risk per CSP
```

**Covered Calls**:
```
Max loss per position: Based on stock position size
Note: Main risk is opportunity cost, not loss
Focus: Don't sell calls that cap too much upside
```

**Vertical Spreads**:
```
Max loss per position: 1-2% of account
Example: Max loss $1,000-$2,000 per spread position
```

**Iron Condors**:
```
Max loss per position: 2-3% of account
Reasoning: Both sides can't be max loss simultaneously
Example: Max $2,000-$3,000 per iron condor position
```

### By Underlying Type

**Index ETFs (SPY, QQQ)**:
```
Higher limits acceptable: up to 3% per position
Reasoning: Diversified, less gap risk
```

**Large Cap Stocks (AAPL, MSFT)**:
```
Standard limits: 1-2% per position
Reasoning: Liquid, stable, but single-company risk
```

**Small/Mid Cap Stocks**:
```
Lower limits: 0.5-1% per position
Reasoning: Higher volatility, gap risk
```

**Speculative/Meme Stocks**:
```
Minimal exposure: 0.25-0.5% per position
Or avoid entirely
```

## Implementing Max Loss in Practice

### Before Trade Entry

```
Pre-trade checklist:
□ Define exact entry price
□ Calculate maximum shares/contracts for 1-2% risk
□ Set stop loss price
□ Calculate max dollar loss
□ Confirm max loss is acceptable
□ Enter position and stop simultaneously
```

### Max Loss Calculation Worksheet

```
Trade Planning Worksheet:

Account Size: $________
Max Risk %: ____%
Max Risk $: $________

Entry Price: $________
Stop Loss Price: $________
Risk Per Share: $________

Position Size = Max Risk $ / Risk Per Share
Position Size: ________ shares/contracts

Verification:
Position × Risk Per Share = $_______
Is this ≤ Max Risk $? □ Yes □ No

If No, reduce position size!
```

### Tracking Max Loss

```
Daily tracking spreadsheet:
| Trade | Entry | Stop | Risk $ | Current | P/L | % of Max |
|-------|-------|------|--------|---------|-----|----------|

Running totals:
- Total risk deployed: $______
- Daily P/L: $______
- Distance to daily limit: $______
```

## Special Situations

### Earnings Trades

**Before earnings, max loss should be lower**:
```
Regular max loss: 2%
Pre-earnings max loss: 1%
Reasoning: Higher volatility, unpredictable gaps
```

### High Volatility Environments

**When VIX is elevated (>25)**:
```
Reduce max loss limits by 50%
Example: 2% becomes 1%
Reasoning: Larger moves more likely
```

### End of Quarter/Year

**When approaching key dates**:
```
Consider tighter stops
Reduce position sizes
Reasoning: Increased volatility, portfolio rebalancing flows
```

## Max Loss Rules Summary

### The 10 Commandments of Max Loss

1. **Never risk more than 2% on any single trade**
2. **Define max loss before entering every trade**
3. **Use stop losses religiously**
4. **Have daily loss limits and honor them**
5. **Reduce size after losses, not increase**
6. **Account for correlation in total portfolio risk**
7. **Know your realistic max loss, not just theoretical**
8. **Don't average down on losers**
9. **Review every max loss hit for lessons**
10. **Protect capital first, seek profits second**

### Quick Reference Card

```
MAX LOSS QUICK REFERENCE

Per Trade: 1-2% of account
Per Day: 2-3% of account
Per Week: 5-6% of account
Per Month: 10% before mandatory review

Stop at:
- 50% premium loss (long options)
- 2x credit (short options)
- Defined technical level (stocks)

After 20% drawdown: STOP TRADING
```

## Conclusion

Maximum loss guidelines are your safety net. They won't prevent all losses, but they will prevent the catastrophic ones that end trading careers.

**Remember**: The goal isn't to never lose - it's to ensure that when you lose, the loss is small enough to be overcome with the next few winning trades.

A trader who risks 1% per trade needs only a 51% win rate with 1:1 risk/reward to be profitable. A trader who risks 10% per trade is just gambling.

**Control your losses, and the profits will take care of themselves.**
