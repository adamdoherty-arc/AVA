# Iron Condors: Complete Strategy Guide

## What is an Iron Condor?

An iron condor is a neutral options strategy that profits when the underlying stock stays within a defined price range. It combines a bull put spread and a bear call spread with the same expiration date.

## Structure

### Components

```
Iron Condor = Bull Put Spread + Bear Call Spread

Bull Put Spread (below current price):
- Sell OTM put (short strike)
- Buy further OTM put (long strike)

Bear Call Spread (above current price):
- Sell OTM call (short strike)
- Buy further OTM call (long strike)
```

### Visual Representation

```
Stock Price: $100

        Long Put    Short Put    Stock    Short Call   Long Call
           |           |           |           |           |
           $90        $95        $100       $105        $110

Profit Zone: $95 to $105 (between short strikes)
Max Loss Zone: Below $90 or above $110 (beyond long strikes)
```

## Example Trade Setup

### Standard Iron Condor

```
Stock: SPY at $450
Expiration: 30 days

Bull Put Spread:
- Sell $440 put for $3.00
- Buy $435 put for $1.50
- Net credit: $1.50

Bear Call Spread:
- Sell $460 call for $3.00
- Buy $465 call for $1.50
- Net credit: $1.50

Total credit: $3.00 ($300 per iron condor)

Risk/Reward:
- Max profit: $300 (if SPY between $440-$460 at expiration)
- Max loss: $200 ($5 spread width - $3 credit = $2 x 100)
- Breakeven: $437 and $463
```

## When to Use Iron Condors

### Ideal Market Conditions

- **Low expected volatility**: Expecting stock to stay range-bound
- **High IV rank**: Collecting elevated premium
- **No major catalysts**: No earnings, FDA, or significant events
- **Sideways trend**: Stock in consolidation pattern

### Best Underlyings

- **Index ETFs**: SPY, QQQ, IWM (diversified, less gap risk)
- **High-liquidity stocks**: AAPL, MSFT, AMZN
- **Stocks in ranges**: Identified support and resistance levels

### When to Avoid

- Trending markets (strong up or down)
- Before earnings or major announcements
- Low IV environments (poor premium)
- Illiquid options (wide bid-ask spreads)

## Strike Selection

### Width Between Short Strikes

**Narrow body** ($15-20 on SPY):
- Higher probability of profit
- Smaller credit received
- Less room for error

**Wide body** ($25-30 on SPY):
- Lower probability of profit
- Larger credit received
- More room for stock movement

### Delta-Based Selection

```
Short strikes at 15-20 delta each side:
- ~70-80% probability of profit
- Standard approach for most traders

Short strikes at 10-15 delta each side:
- ~85% probability of profit
- Lower premium but higher win rate

Short strikes at 25-30 delta each side:
- ~60% probability of profit
- Higher premium but more risk
```

### Wing Width (Long Strikes)

```
Standard: $5 wings
- Balanced risk/reward
- Efficient capital use

Narrow: $2.50 wings
- Lower max loss
- Requires more contracts for same risk
- Higher commission impact

Wide: $10 wings
- Higher max loss
- Better premium collection
- Capital efficient
```

## Expiration Selection

### Optimal DTE

**30-45 DTE**: Industry standard
- Optimal theta decay curve
- Time to adjust if needed
- Manageable gamma risk

**21-30 DTE**: More aggressive
- Faster theta decay
- Less adjustment time
- Higher gamma risk

**45-60 DTE**: Conservative
- More time to be right
- Lower gamma risk
- Slower theta decay

### Weekly vs Monthly

**Weekly expirations**:
- Higher theta decay rate
- More frequent management
- Higher gamma risk
- Good for small accounts

**Monthly expirations**:
- Standard approach
- Less frequent trading
- More predictable decay
- Lower gamma near expiration

## Management Strategies

### Profit Taking

**Close at 50% of max profit**:
```
Received $3.00 credit
When spread worth $1.50, close
Profit: $1.50 ($150 per iron condor)

Benefits:
- Lock in gains
- Free up capital
- Avoid gamma risk
- Higher annualized returns
```

**Close at 25% of max profit** (aggressive):
- Faster capital turnover
- Very high win rate
- Lower profit per trade

### Rolling the Untested Side

When one side is threatened but not breached:

```
Original iron condor: $440/$435 puts, $460/$465 calls
SPY rallies to $458

Action: Roll call spread up
- Buy back $460/$465 call spread
- Sell new $465/$470 call spread

Benefits:
- Collect additional credit
- Move short strike further away
- Maintain iron condor structure
```

### Closing One Side

If one spread is nearly worthless:
```
SPY drops to $442
Put spread at risk, call spread worth $0.10

Action: Buy back call spread for $0.10
- Remove upside risk
- Focus on defending put spread
- Small cost for risk reduction
```

### Stop Loss Rules

**Rule of thumb**: Close at 2x credit received
```
Received $3.00 credit
If iron condor worth $6.00+, close
Max loss: $3.00 ($300 per iron condor)
Prevents catastrophic loss
```

**Alternate rule**: Close if short strike breached
- Don't wait for max loss
- Defend capital early
- Accept smaller loss

## Risk Management

### Position Sizing

```
Max risk per trade: 1-2% of account
Account size: $100,000
Max risk: $1,000-$2,000

Iron condor max loss: $200
Position size: $1,000 / $200 = 5 iron condors maximum
```

### Portfolio Considerations

- Maximum 5% of account in single underlying
- Diversify across multiple iron condors
- Stagger expirations to spread risk
- Account for correlation between positions

### Margin Requirements

```
Margin = Greater of (put spread width, call spread width) - credit received

Example:
$5 put spread + $5 call spread
Max margin: $5 x 100 = $500
Credit received: $3.00 x 100 = $300
Net margin: $500 - $300 = $200

Buying power reduction: $200 per iron condor
```

## Greeks Analysis

### Delta

**Well-positioned iron condor**: Near-zero delta
- Balanced exposure
- No directional bias
- Profits from time decay

**Adjusting delta**:
- If delta positive, short strikes closer to stock
- If delta negative, stock moving toward put side

### Theta

**Positive theta**: Time decay works for you
```
Example: 30 DTE iron condor
Theta: +$8/day
Expected daily decay: $8 profit
Accelerates as expiration approaches
```

### Vega

**Negative vega**: Want IV to decrease
- Collected premium when IV high
- Spread shrinks when IV drops
- IV crush after events is beneficial

### Gamma

**Negative gamma**: Danger near expiration
- Delta changes rapidly
- ATM options spike in gamma
- Major risk if stock near short strikes

## Advanced Techniques

### Adjusting Wing Width

**Unbalanced iron condor**:
```
Bullish bias: Wider put spread, narrower call spread
Stock at $100:
- $85/$90 puts (wider)
- $105/$107.50 calls (narrower)

Bearish bias: Opposite configuration
```

### Jade Lizard Variation

**Iron condor without call spread wing**:
```
Sell put spread: $95/$90 puts for $2.00
Sell naked call: $110 call for $1.00
No upside risk if credit > call-to-stock distance
```

### Rolling to New Cycle

When approaching expiration with profit:
```
Current: Iron condor worth $1.00 (started at $3.00)
10 DTE remaining

Action: Close current, open new 30-45 DTE iron condor
- Realize $2.00 profit
- Start fresh cycle
- Avoid gamma risk
```

## Common Mistakes

### 1. Wrong Volatility Environment

**Mistake**: Selling iron condors when IV is low
**Fix**: Only trade when IV rank > 30%

### 2. Short Strikes Too Close

**Mistake**: Tight profit zone gets breached easily
**Fix**: Minimum $20 between short strikes on SPY

### 3. Holding Through Breach

**Mistake**: Hoping stock reverses after short strike breached
**Fix**: Close or roll when short strike is tested

### 4. Ignoring Earnings

**Mistake**: Iron condor through earnings announcement
**Fix**: Close before earnings or don't trade that cycle

### 5. Oversizing

**Mistake**: Too many contracts relative to account
**Fix**: Max 2% of account at risk per trade

## Performance Metrics

### Tracking Your Iron Condors

```
Per trade:
- Entry date and strikes
- Credit received
- Delta at entry
- IV rank at entry
- Exit date and price
- Profit/loss
- Days held
- What triggered exit

Monthly/Annual:
- Win rate (target 70-80%)
- Average profit per trade
- Average loss per trade
- Profit factor (gross wins / gross losses)
- Max drawdown
- Annualized return
```

### Realistic Expectations

```
Conservative approach:
- Win rate: 75-85%
- Average winner: $100-$150
- Average loser: $150-$250
- Monthly return: 2-4%
- Annual return: 15-30%

Aggressive approach:
- Win rate: 60-70%
- Average winner: $150-$250
- Average loser: $200-$350
- Monthly return: 4-8%
- Annual return: 25-50% (with more drawdowns)
```

## Iron Condor Checklist

### Before Entry

- [ ] IV rank > 30% (preferably > 50%)
- [ ] No earnings in next 30 days
- [ ] Stock in defined range (support/resistance identified)
- [ ] Short strikes at 15-20 delta
- [ ] Credit meets minimum target (typically 1/3 of width)
- [ ] Position size within risk limits
- [ ] Bid-ask spreads are tight (<10% of premium)

### During Trade

- [ ] Monitor at 50% profit for exit
- [ ] Watch if short strike approached
- [ ] Consider rolling untested side
- [ ] Check if underlying conditions changed

### Exit Conditions

- [ ] 50% of max profit achieved
- [ ] Short strike breached (close or roll)
- [ ] 21 DTE or less (close to avoid gamma)
- [ ] Fundamental change in underlying
- [ ] 2x credit loss (stop loss)

## Conclusion

Iron condors are a cornerstone strategy for neutral traders seeking consistent income. Success requires:

1. **Proper conditions**: High IV, range-bound underlying
2. **Smart sizing**: Never risk more than 2% per trade
3. **Active management**: Close at 50% profit, roll when tested
4. **Diversification**: Multiple iron condors across underlyings
5. **Discipline**: Follow rules, don't hope or pray

When executed systematically, iron condors can generate 15-30% annual returns with controlled risk, making them ideal for income-focused traders who expect sideways markets.
