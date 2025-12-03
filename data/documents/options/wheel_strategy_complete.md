# The Wheel Strategy: Complete Guide

## Overview

The Wheel Strategy is a systematic options trading approach designed to generate consistent income through premium collection while building long-term stock positions at favorable prices. It combines two foundational options strategies: Cash-Secured Puts (CSPs) and Covered Calls.

## The Wheel Cycle

### Phase 1: Selling Cash-Secured Puts

**Objective**: Collect premium while waiting to buy a stock at a discount.

**How it works**:
1. Identify a stock you want to own at a lower price
2. Sell a put option at a strike price where you'd be happy to buy
3. Collect the premium immediately
4. If the stock stays above your strike, keep the premium and repeat
5. If the stock falls below your strike, you'll be assigned shares

**Example**:
- Stock XYZ is trading at $100
- You sell a $95 put for $2.50 premium
- If XYZ stays above $95, you keep $250 per contract
- If XYZ falls to $90, you buy 100 shares at $95 but your effective cost basis is $92.50

### Phase 2: Assignment and Stock Ownership

**What happens at assignment**:
- You purchase 100 shares per contract at the strike price
- Your cost basis = Strike Price - Premium Received
- You now own the stock and can sell covered calls

**Managing assignment**:
- Have cash ready (hence "cash-secured")
- Consider the tax implications (short-term capital gains on premium)
- Update your tracking with the new cost basis

### Phase 3: Selling Covered Calls

**Objective**: Generate income while holding shares, potentially selling at a profit.

**How it works**:
1. With 100 shares, sell a call option above your cost basis
2. Collect premium immediately
3. If the stock stays below your strike, keep premium and repeat
4. If the stock rises above your strike, shares are called away at profit

**Example**:
- You own XYZ at $92.50 cost basis
- Stock is trading at $95
- Sell a $100 call for $1.50 premium
- If called away: $100 sale price + $1.50 premium - $92.50 cost = $9.00 profit per share
- If not called: Keep $150 premium and sell another call

### Phase 4: Called Away - Cycle Restarts

When shares are called away:
- You've sold at a profit (strike price > cost basis)
- You have cash again
- Return to Phase 1 and sell more puts

## Entry Criteria for the Wheel

### Stock Selection

**Ideal Wheel Candidates**:
- Stocks you genuinely want to own long-term
- Liquid options with tight bid-ask spreads
- Moderate to high implied volatility (for better premiums)
- Fundamentally sound companies
- Stocks with strong support levels
- Price range you can afford (100 shares)

**Avoid for Wheel**:
- Highly volatile meme stocks
- Stocks in downward trends
- Companies with poor fundamentals
- Illiquid options (wide spreads)
- Stocks facing major binary events

### Premium Targets

**Minimum Premium Targets by DTE**:

| DTE | Min Premium Target | Annualized Return |
|-----|-------------------|-------------------|
| 7 days | 0.5% | ~26% |
| 14 days | 0.75% | ~20% |
| 21 days | 1.0% | ~17% |
| 30 days | 1.25% | ~15% |
| 45 days | 1.75% | ~14% |

**Premium/Risk Calculation**:
```
Premium % = (Premium Received / Capital at Risk) x 100
Annualized Return = Premium % x (365 / DTE)
```

### Strike Selection

**For Cash-Secured Puts**:
- Select strikes at or below strong support levels
- Target 20-30 delta for balanced risk/reward
- Consider at-the-money (ATM) if you strongly want the stock
- Use 10-15 delta for more conservative approach

**For Covered Calls**:
- Strike should be above your cost basis
- Consider resistance levels
- 20-30 delta for balanced approach
- Higher delta if you want to exit the position

### Days to Expiration (DTE)

**Optimal DTE Range**: 30-45 days

**Why 30-45 DTE**:
- Theta decay accelerates but still has time for stock movement
- Better premium than weekly options
- Time to roll if needed
- Less frequent management

**When to use shorter DTE (7-21 days)**:
- Higher IV environment
- Want more frequent trades
- Stock is range-bound
- Near expiration roll

## Management Strategies

### Rolling Puts

**When to roll a put**:
- Stock has dropped and assignment is imminent
- You want more time or better strike
- IV has dropped (roll to capture new premium)

**Rolling mechanics**:
- Buy back current put
- Sell new put with different strike/expiration
- Try to collect net credit

**Roll down and out**:
- Lower your strike (down)
- Extend expiration (out)
- Often used when stock drops significantly

### Rolling Covered Calls

**When to roll a call**:
- Stock has risen and call is threatened
- You want to keep shares longer
- Good opportunity to capture more premium

**Roll up and out**:
- Higher strike (up) for more upside
- Extend expiration (out) for more premium
- Common when stock is rallying

### Early Closure

**Close for 50% profit**:
- If option reaches 50% of max profit before expiration
- Frees up capital for new trades
- Reduces gamma risk near expiration

**Example**:
- Sold put for $2.00
- Put is now worth $1.00
- Close for $100 profit (50%)
- Use capital for next trade

### Assignment Management

**Before assignment**:
- Ensure you have buying power
- Decide: accept or roll?
- Check ex-dividend dates

**After assignment**:
- Immediately sell covered call (if appropriate)
- Update cost basis tracking
- Consider tax lot selection

## Risk Management

### Position Sizing

**Capital Allocation Rules**:
- Maximum 5% of portfolio per position
- No more than 20% in a single sector
- Keep 20% cash reserve for opportunities/assignment

**Contract Sizing**:
```
Max Contracts = (Portfolio Size x Position Limit) / (Strike x 100)
Example: $100,000 portfolio, 5% limit, $50 strike
Max = ($100,000 x 0.05) / ($50 x 100) = 1 contract
```

### Stop-Loss Guidelines

**Mental stops for wheel**:
- Close put if loss exceeds 2x premium received
- Consider closing if stock drops 15-20% from entry
- Always have max loss defined before entry

**Example**:
- Sold put for $2.00 premium
- If put value reaches $4.00 (2x loss), consider closing
- Maximum loss: $200 vs potential assignment loss

### Diversification

**Spread across**:
- Multiple stocks (minimum 5-10)
- Different sectors
- Various expiration dates
- Different strike distances

## Tax Considerations

### Short-Term vs Long-Term

**Premium income**: Always short-term capital gains

**Stock holding period**:
- Starts when assigned, not when put sold
- Covered calls don't reset holding period
- Called away stock: holding period determines tax treatment

### Qualified Covered Calls

**Rules for qualified covered calls**:
- Strike must be above closing price on prior day
- Must meet specific delta/strike requirements
- Unqualified calls can terminate holding period

### Tax-Efficient Strategies

- Use tax-advantaged accounts (IRA) for wheel
- Consider tax-loss harvesting opportunities
- Track all trades carefully for Schedule D

## Advanced Wheel Techniques

### The Poor Man's Covered Call

**Alternative to owning shares**:
- Buy long-dated ITM call (LEAPS) instead of shares
- Sell short-term calls against it
- Lower capital requirement

**Setup**:
- Buy 70+ delta LEAPS (1-2 years out)
- Sell 30 delta monthly calls
- Roll as needed

### Jade Lizard

**Enhanced CSP strategy**:
- Sell put (cash-secured)
- Sell call spread above current price
- No upside risk, enhanced premium

**Example**:
- Stock at $100
- Sell $95 put for $2.00
- Sell $105/$110 call spread for $0.75
- Total premium: $2.75
- No risk if stock goes up

### Wheel with Adjustments

**Adding puts on dips**:
- If stock drops after put sale, sell additional put at lower strike
- Averages into position over time
- Increases premium income

**Multiple covered calls**:
- Split position into multiple calls
- Different strikes for different scenarios
- Allows partial assignment

## Performance Tracking

### Key Metrics to Track

**Per Trade**:
- Entry date and strike
- Premium received
- Days to expiration
- Delta at entry
- Outcome (expired, assigned, closed early)
- Profit/loss
- Annualized return

**Portfolio Level**:
- Win rate (% profitable trades)
- Average profit per trade
- Total premium collected
- Assignment rate
- Average holding period
- Sharpe ratio

### Expected Results

**Realistic Expectations**:
- Annual returns: 15-25% in normal markets
- Win rate: 70-85%
- Assignment rate: 20-40%
- Monthly income: 1-2% of capital

**Factors affecting performance**:
- Market conditions (sideways best)
- Stock selection quality
- Premium targets
- Management discipline

## Common Mistakes to Avoid

### Entry Mistakes

1. **Selling on stocks you don't want to own**
   - Solution: Only wheel stocks you'd buy anyway

2. **Chasing premium on risky stocks**
   - Solution: Stick to quality companies

3. **Over-leveraging**
   - Solution: Maintain position size limits

4. **Ignoring IV rank**
   - Solution: Sell when IV rank > 30

### Management Mistakes

1. **Not rolling when appropriate**
   - Solution: Have clear roll rules

2. **Letting winners run too long**
   - Solution: Close at 50% profit

3. **Fighting the trend**
   - Solution: Adjust strikes with market direction

4. **Ignoring cost basis**
   - Solution: Always track true cost basis

## Wheel Strategy Checklist

### Before Opening Position

- [ ] Stock passes fundamental screening
- [ ] Options are liquid (bid-ask < 10% of premium)
- [ ] IV rank > 30% (good premium environment)
- [ ] Clear support/resistance levels identified
- [ ] Position size within limits
- [ ] Capital available for potential assignment
- [ ] Premium target met
- [ ] No major events before expiration

### During Trade

- [ ] Monitor for 50% profit close opportunity
- [ ] Watch for roll opportunities
- [ ] Track any dividend dates
- [ ] Reassess if stock moves 10%+ against position

### After Trade

- [ ] Log trade details
- [ ] Calculate actual return
- [ ] Update position tracking
- [ ] Prepare for next cycle

## Conclusion

The Wheel Strategy is a proven method for generating consistent income while building stock positions at favorable prices. Success requires:

1. **Discipline**: Stick to entry criteria and management rules
2. **Patience**: Let theta work for you
3. **Capital**: Adequate funds for assignment
4. **Selection**: Only trade stocks you'd want to own
5. **Tracking**: Monitor performance and adjust

When executed properly, the wheel can generate 15-25% annual returns with lower risk than buy-and-hold strategies, making it ideal for income-focused investors.
