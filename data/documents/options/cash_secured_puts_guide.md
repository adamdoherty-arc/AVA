# Cash-Secured Puts (CSP): Complete Guide

## What is a Cash-Secured Put?

A Cash-Secured Put is an options strategy where you sell a put option while holding enough cash to purchase the underlying stock if assigned. It's the foundation of income-focused options trading and the entry point for the Wheel Strategy.

## How CSPs Work

### Mechanics

1. **You sell a put option** at a strike price below current stock price
2. **You receive premium** immediately (credit to your account)
3. **You hold cash** equal to strike × 100 as collateral
4. **At expiration**:
   - Stock above strike: Put expires worthless, keep premium
   - Stock below strike: You buy 100 shares at strike price

### Example Trade

```
Stock: AAPL trading at $175
Action: Sell 1 AAPL $170 put expiring in 30 days
Premium: $3.50 ($350 per contract)
Capital Required: $17,000 (170 × 100)

Scenarios at expiration:
1. AAPL at $180: Put expires worthless
   Profit: $350 (2.06% return on capital)

2. AAPL at $168: Assigned 100 shares at $170
   Effective cost basis: $170 - $3.50 = $166.50
   Paper loss: ($168 - $166.50) × 100 = $150 gain on cost basis

3. AAPL at $150: Assigned 100 shares at $170
   Effective cost basis: $166.50
   Paper loss: ($150 - $166.50) × 100 = -$1,650
```

## Why Sell CSPs?

### Advantages

1. **Income Generation**: Collect premium regardless of outcome
2. **Buy at Discount**: Acquire stocks below market price
3. **Lower Risk than Buying**: Premium reduces cost basis
4. **High Probability**: 70-80% win rate at appropriate strikes
5. **Defined Risk**: Maximum loss is known upfront

### Compared to Buying Stock

| Metric | Buy Stock | Sell CSP |
|--------|-----------|----------|
| Upside | Unlimited | Limited to premium |
| Downside | Full stock risk | Reduced by premium |
| Income | None | Premium collected |
| Cost Basis | Market price | Strike - premium |
| Capital Efficiency | 100% | 100% (or margin) |

## Strike Selection

### Delta-Based Selection

| Delta | Probability OTM | Risk Level | Premium |
|-------|-----------------|------------|---------|
| 10 | ~90% | Conservative | Low |
| 20 | ~80% | Moderate | Medium |
| 30 | ~70% | Standard | Good |
| 40 | ~60% | Aggressive | High |
| 50 | ~50% | Very Aggressive | Maximum |

### Strike Selection Strategies

**Conservative (10-15 Delta)**:
- Far OTM strikes
- Very high win rate
- Lower premium
- Best for volatile markets

**Standard (20-30 Delta)**:
- Optimal balance
- Good premium
- Acceptable assignment risk
- Most common approach

**Aggressive (40-50 Delta)**:
- Higher premium
- Likely assignment
- Use when you want the stock

### Support-Based Selection

**Identify technical support levels**:
- Previous lows
- Moving averages (50, 200 day)
- Round numbers
- Fibonacci retracements

**Sell puts at or below support** for higher safety margin.

## Days to Expiration (DTE)

### Optimal DTE: 30-45 Days

**Why 30-45 DTE?**:
- Theta decay is accelerating
- Still have time to manage/roll
- Good premium relative to time
- Less gamma risk than weeklies

### DTE Comparison

| DTE | Theta Decay | Management Time | Premium |
|-----|-------------|-----------------|---------|
| 7 | Fastest | Minimal | Lower |
| 14-21 | Fast | Some time | Medium |
| 30-45 | Moderate | Good time | Good |
| 60-90 | Slow | Lots of time | Higher |

### Weekly vs Monthly

**Weeklies (7 DTE)**:
- Higher annualized returns
- More trades = more commissions
- Less time to be wrong
- Higher gamma risk

**Monthly (30-45 DTE)**:
- More forgiving
- Better risk-adjusted returns
- Less management needed
- Recommended for most traders

## Premium Targets

### Minimum Premium Requirements

Calculate premium as percentage of capital at risk:

```
Premium % = (Premium / Strike) × 100

Example: $170 strike, $3.50 premium
Premium % = (3.50 / 170) × 100 = 2.06%
```

### Target Premium by DTE

| DTE | Minimum Premium % | Annualized |
|-----|-------------------|------------|
| 7 | 0.5% | 26% |
| 14 | 0.75% | 20% |
| 21 | 1.0% | 17% |
| 30 | 1.25% | 15% |
| 45 | 1.75% | 14% |

### IV Rank Consideration

**Sell CSPs when IV is elevated** (IV Rank > 30%):
- Higher premiums available
- Mean reversion works in your favor
- Better risk/reward

**Avoid selling when IV is low** (IV Rank < 20%):
- Insufficient premium
- Potential IV expansion against you

## Entry Checklist

### Before Opening Position

- [ ] Stock you want to own at strike price
- [ ] IV Rank > 30% preferred
- [ ] Premium meets minimum target (1%+ for 30 DTE)
- [ ] Liquid options (bid-ask < 10% of premium)
- [ ] No earnings in next 2 weeks
- [ ] No major binary events pending
- [ ] Capital available for assignment
- [ ] Position size within limits

### Technical Confirmation

- [ ] Stock in uptrend or range
- [ ] Strike at or below support
- [ ] Not fighting major downtrend
- [ ] Sector showing relative strength

## Management Strategies

### Profit Taking

**Close at 50% profit**:
```
Original Premium: $3.50
50% Target: $1.75
Close when put value falls to $1.75
Profit: $1.75 per share
```

**Why close early?**:
- Free up capital for new trades
- Remove tail risk
- Better risk-adjusted returns
- Compound gains faster

### Rolling

**Rolling Down**:
- Stock drops, put is threatened
- Buy back current put
- Sell new put at lower strike
- Try to collect net credit

**Rolling Out**:
- Stock near strike at expiration
- Buy back current put
- Sell new put at same strike, later expiration
- Collect additional premium, extend time

**Rolling Down and Out**:
- Combine both adjustments
- Lower strike AND extend time
- Often used for significant drops

### Rolling Example

```
Original: Sold $170 put for $3.50
Stock drops to $168
Put now worth $5.00

Roll to:
- $165 strike (down $5)
- Next month expiration (out 30 days)
- Receive $4.00 premium

Cost to close: -$5.00
New premium: +$4.00
Net debit: -$1.00

New position: $165 put
Combined premium: $3.50 - $1.00 = $2.50
Effective cost basis if assigned: $165 - $2.50 = $162.50
```

### Defending a Losing Position

**Options when stock drops significantly**:

1. **Take assignment**: Accept shares, sell covered calls
2. **Roll down and out**: Extend time, lower strike
3. **Close for loss**: Cut losses if thesis broken
4. **Do nothing**: Wait for bounce or expiration

**Decision framework**:
- Still want the stock? → Roll or take assignment
- Thesis broken? → Close for loss
- Need capital? → Close position

## Assignment

### What Happens at Assignment

1. Put is exercised by buyer
2. You purchase 100 shares per contract
3. Cash is deducted from account
4. Shares appear in portfolio
5. Short put position disappears

### Preparing for Assignment

**Before expiration**:
- Ensure adequate buying power
- Know your cost basis
- Have covered call plan ready

**Cost Basis Calculation**:
```
Cost Basis = Strike Price - Premium Received

Example: $170 strike, $3.50 premium
Cost Basis = $170 - $3.50 = $166.50
```

### Post-Assignment Actions

1. **Immediately sell covered call** (if appropriate)
2. **Update tracking** with new cost basis
3. **Reassess position** - still want to hold?
4. **Check ex-dividend dates**

## Risk Management

### Maximum Loss

**Theoretical max loss**: Strike × 100 - Premium (if stock goes to $0)

```
$170 put, $3.50 premium
Max Loss = ($170 × 100) - $350 = $16,650
```

**Realistic max loss consideration**: Stock rarely goes to $0
- Consider 30-50% drop as realistic worst case
- Build position sizing around realistic max loss

### Position Sizing for CSPs

```
Max Risk Per Trade = Account Size × Risk %

Example: $100,000 account, 2% risk = $2,000

If realistic max loss = 30% of strike:
$170 strike × 30% = $51/share
Risk per contract = $51 × 100 = $5,100

Contracts = $2,000 / $5,100 = 0.39

Result: Trade 0-1 contracts to stay within risk limits
```

### Portfolio Limits

- Maximum 5% of portfolio per CSP position
- Maximum 20% total in CSPs
- Diversify across sectors
- Keep cash reserves for opportunities

## Common Mistakes

### Mistake 1: Selling on Stocks You Don't Want

**Problem**: Taking assignment becomes painful
**Solution**: Only sell CSPs on stocks you'd buy anyway

### Mistake 2: Chasing Premium

**Problem**: Selling on high IV junk stocks
**Solution**: Premium is payment for risk, not free money

### Mistake 3: Wrong Position Size

**Problem**: Single position causes major account swing
**Solution**: Strict position sizing (1-2% risk max)

### Mistake 4: Not Rolling Early Enough

**Problem**: Position becomes too far ITM to roll effectively
**Solution**: Roll when delta reaches 60-70

### Mistake 5: Ignoring Cost of Management

**Problem**: Frequent adjustments eat into profits
**Solution**: Factor management costs into expected return

## CSP vs Other Strategies

### CSP vs Buying Stock

| CSP | Buying Stock |
|-----|--------------|
| Lower cost basis | Market price |
| Premium income | No income |
| Limited upside | Unlimited upside |
| Can expire worthless (profit) | Must sell to exit |

### CSP vs Naked Put

Both involve selling puts, but:
- CSP: Fully cash secured
- Naked put: Uses margin
- CSP safer for most traders

### CSP vs Put Credit Spread

| CSP | Put Credit Spread |
|-----|-------------------|
| Unlimited downside | Defined risk |
| Higher premium | Lower premium |
| Can take assignment | Close spread before expiration |
| Simpler | Requires spread management |

## Performance Expectations

### Realistic Returns

**Annual return range**: 12-24% depending on:
- Market conditions
- Strike selection
- Management style
- IV environment

### Win Rate

**Typical win rate at 30 delta**: 70-75%
**With proper management**: 80-85%

### Monthly Income Example

```
Account: $100,000
Allocation to CSPs: 50% = $50,000
Average position size: $10,000
Positions: 5 CSPs
Monthly premium: 1.5% average
Monthly income: $50,000 × 1.5% = $750
Annual income: $9,000 (18% on allocated capital)
```

## Quick Reference

### Ideal CSP Setup

- Stock: Quality company you'd own
- IV Rank: > 30%
- DTE: 30-45 days
- Delta: 20-30
- Strike: At or below support
- Premium: > 1% of strike

### Exit Rules

- Close at 50% profit
- Roll at 21 DTE if profitable
- Roll when delta > 60
- Close if thesis broken

### Don't Trade CSPs When

- IV Rank < 20%
- Stock in strong downtrend
- Earnings within 2 weeks
- Unable to take assignment
- Position would exceed limits
