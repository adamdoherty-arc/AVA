# Options Greeks: Comprehensive Guide

## Introduction to Greeks

The Greeks are mathematical measures that describe how option prices change in response to various factors. Understanding Greeks is essential for risk management, position sizing, and strategy selection.

## Delta (Δ)

### Definition

Delta measures the rate of change in option price for a $1 change in the underlying stock price.

### Delta Values

| Option Type | Delta Range | Interpretation |
|-------------|-------------|----------------|
| Long Call | 0 to +1.00 | Positive: gains when stock rises |
| Short Call | 0 to -1.00 | Negative: gains when stock falls |
| Long Put | -1.00 to 0 | Negative: gains when stock falls |
| Short Put | 0 to +1.00 | Positive: gains when stock rises |

### Delta by Moneyness

```
Deep ITM Call: ~0.90 to 1.00
ATM Call: ~0.50
OTM Call: ~0.10 to 0.40
Deep OTM Call: ~0.01 to 0.10

Deep ITM Put: ~-0.90 to -1.00
ATM Put: ~-0.50
OTM Put: ~-0.10 to -0.40
Deep OTM Put: ~-0.01 to -0.10
```

### Delta as Probability

Delta approximates the probability of the option expiring in-the-money.

**Example**:
- A 30 delta call has approximately 30% chance of expiring ITM
- A -20 delta put has approximately 20% chance of expiring ITM

### Delta Applications

**Position Sizing**:
```
Delta-Equivalent Shares = Option Delta × 100 × Number of Contracts
Example: 5 contracts of 0.40 delta calls = 5 × 0.40 × 100 = 200 delta-equivalent shares
```

**Hedging**:
To delta-hedge a long stock position:
```
Number of Puts = Shares Owned / (Put Delta × 100)
Example: 500 shares, using 0.50 delta puts = 500 / (0.50 × 100) = 10 put contracts
```

**Strike Selection**:
- For CSPs: 20-30 delta = 70-80% probability of profit
- For covered calls: 20-30 delta = balanced income vs called away risk

## Gamma (Γ)

### Definition

Gamma measures the rate of change in delta for a $1 change in the underlying stock price.

### Gamma Characteristics

**ATM options have highest gamma**:
- ATM delta changes fastest
- Gamma is maximized at ATM strikes
- Gamma increases as expiration approaches

**ITM and OTM options have lower gamma**:
- Delta is already close to 1 (ITM) or 0 (OTM)
- Less room for delta to change

### Gamma by Expiration

```
Long-dated options: Low gamma (delta changes slowly)
30-45 DTE: Moderate gamma
7-14 DTE: Higher gamma
0-3 DTE: Maximum gamma (gamma risk zone)
```

### Gamma Risk

**For Option Buyers** (Long Gamma):
- Gamma works in your favor
- As stock moves your direction, delta increases
- Profits accelerate on large moves

**For Option Sellers** (Short Gamma):
- Gamma works against you
- Large moves hurt disproportionately
- Risk accelerates near expiration

### Gamma Scalping

Advanced strategy to profit from gamma:
1. Buy ATM options (long gamma)
2. Delta hedge with stock
3. Re-hedge as delta changes
4. Profit from stock oscillation

## Theta (Θ)

### Definition

Theta measures the rate of option price decay per day due to time passage.

### Theta Characteristics

**Theta is always negative for long options**:
- Options lose value over time
- Theta accelerates near expiration

**ATM options have highest theta**:
- Most time value to decay
- Fastest absolute decay rate

### Theta Decay Curve

```
Days to Expiration vs Theta Decay Rate:

90 DTE: ~$0.02-0.03/day (slow)
60 DTE: ~$0.03-0.05/day
45 DTE: ~$0.04-0.06/day
30 DTE: ~$0.06-0.10/day
21 DTE: ~$0.08-0.15/day
14 DTE: ~$0.12-0.20/day
7 DTE: ~$0.20-0.40/day
3 DTE: ~$0.40-0.80/day (rapid)
1 DTE: ~$0.60-1.00+/day
```

### Theta for Premium Sellers

**Calculating daily income**:
```
Daily Theta Income = Theta × Number of Contracts × 100
Example: Sold 3 puts with -$0.05 theta
Daily income = $0.05 × 3 × 100 = $15/day
```

**Optimal DTE for theta capture**:
- 45 DTE: Best balance of theta vs gamma risk
- 30 DTE: Higher theta, more gamma risk
- 21 DTE: Theta accelerating rapidly

### Weekend Theta

Options don't trade on weekends, but theta still decays. Some traders sell options Friday to capture weekend decay.

**Weekend decay consideration**:
- 2.5 calendar days of decay
- But may already be priced in Friday
- Gamma risk increases Monday open

## Vega (ν)

### Definition

Vega measures the change in option price for a 1% change in implied volatility.

### Vega Characteristics

**Longer-dated options have higher vega**:
- More time for volatility to affect outcome
- LEAPS have maximum vega exposure

**ATM options have highest vega**:
- Maximum uncertainty = maximum sensitivity
- ITM/OTM have lower vega

### Vega Values by DTE

```
Typical ATM option vega per 1% IV change:

7 DTE: $0.02-0.04
14 DTE: $0.04-0.06
30 DTE: $0.06-0.10
45 DTE: $0.08-0.12
60 DTE: $0.10-0.15
90 DTE: $0.12-0.18
180 DTE: $0.18-0.25
```

### IV Rank and IV Percentile

**IV Rank**: Current IV relative to 52-week range
```
IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV) × 100

Example: Current IV 30%, Low 20%, High 50%
IV Rank = (30 - 20) / (50 - 20) × 100 = 33%
```

**IV Percentile**: Percentage of days IV was lower
```
IV Percentile = Days IV was lower / Total trading days × 100

Example: IV was lower on 200 of 252 trading days
IV Percentile = 200 / 252 × 100 = 79%
```

### Vega Strategies

**Long Vega** (Buy options when IV is low):
- Expect IV to increase
- Buy before earnings/events
- LEAPS during low volatility periods

**Short Vega** (Sell options when IV is high):
- Expect IV to decrease
- Sell after IV spikes
- Premium selling after uncertainty resolves

### IV Crush

**Definition**: Rapid IV decline after anticipated event

**Common IV crush events**:
- Earnings announcements
- FDA decisions
- FOMC meetings
- Product launches

**Managing IV crush**:
- Close long options before events
- Sell options before events (capture elevated IV)
- Use spreads to reduce vega exposure

## Rho (ρ)

### Definition

Rho measures the change in option price for a 1% change in interest rates.

### Rho Characteristics

**Calls have positive rho**: Higher rates increase call values
**Puts have negative rho**: Higher rates decrease put values

### Rho Impact

```
Rho is typically small:
- 30 DTE options: ~$0.01-0.02 per 1% rate change
- 365 DTE options: ~$0.10-0.15 per 1% rate change
```

**When rho matters**:
- Long-dated options (LEAPS)
- High interest rate environments
- Large positions

## Portfolio Greeks

### Position Greek Calculation

**Total Position Delta**:
```
Sum of (Each Option Delta × Contracts × 100)

Example:
Long 5 calls at 0.40 delta: 5 × 0.40 × 100 = +200
Short 3 puts at -0.30 delta: 3 × 0.30 × 100 = +90
Total Position Delta: +290 (equivalent to 290 long shares)
```

### Delta-Neutral Portfolios

**Definition**: Portfolio with zero net delta (no directional bias)

**Creating delta-neutral position**:
```
Position delta + Hedge = 0

Example: Long 10 calls at 0.50 delta = +500 delta
Hedge: Short 500 shares = -500 delta
Net delta: +500 - 500 = 0
```

### Gamma-Neutral Portfolios

More complex, requires balancing gamma exposure:
- Combine different strikes/expirations
- Used by market makers
- Requires frequent rebalancing

## Greek Sensitivity by Strategy

### Long Call

| Greek | Exposure | Impact |
|-------|----------|--------|
| Delta | Positive | Profits when stock rises |
| Gamma | Positive | Delta increases with stock |
| Theta | Negative | Loses value daily |
| Vega | Positive | Benefits from IV rise |

### Short Put (CSP)

| Greek | Exposure | Impact |
|-------|----------|--------|
| Delta | Positive | Profits when stock rises |
| Gamma | Negative | Losses accelerate on drops |
| Theta | Positive | Gains value daily |
| Vega | Negative | Benefits from IV fall |

### Covered Call

| Greek | Exposure | Impact |
|-------|----------|--------|
| Delta | Positive (reduced) | Profits when stock rises (capped) |
| Gamma | Negative | Losses if stock moves big |
| Theta | Positive (from call) | Short call gains daily |
| Vega | Negative (from call) | Benefits from IV fall |

### Iron Condor

| Greek | Exposure | Impact |
|-------|----------|--------|
| Delta | Near zero | Minimal directional exposure |
| Gamma | Negative | Hurts if stock moves big |
| Theta | Positive | Maximum time decay capture |
| Vega | Negative | Benefits from IV fall |

## Greek Management Rules

### Delta Management

**Position limits by account size**:
```
Conservative: Max delta = 30% of account value
Moderate: Max delta = 50% of account value
Aggressive: Max delta = 75% of account value
```

**Rebalancing triggers**:
- Delta exceeds target by 20%+
- Major market events
- Portfolio concentrations

### Theta Management

**Target theta per day**:
```
Conservative: 0.1% of account value
Moderate: 0.2% of account value
Aggressive: 0.3% of account value

Example: $100,000 account, moderate approach
Target theta: $100,000 × 0.002 = $200/day
```

### Gamma Management

**Avoiding gamma risk**:
- Close short options before 7 DTE
- Reduce position size near expiration
- Use spreads to cap gamma exposure

### Vega Management

**Sell high IV, buy low IV**:
- Check IV rank before trading
- Avoid buying options when IV > 50%
- Favor selling when IV rank > 30%

## Greek Formulas Summary

### Approximations

```
Delta: Change in Option Price / $1 Stock Move

Gamma: Change in Delta / $1 Stock Move

Theta: Daily Time Decay (shown as negative for longs)

Vega: Change in Option Price / 1% IV Change

Rho: Change in Option Price / 1% Interest Rate Change
```

### Quick Reference Table

| Greek | Range | Long Option | Short Option |
|-------|-------|-------------|--------------|
| Delta | -1 to +1 | Works for you | Works against |
| Gamma | 0 to +∞ | Works for you | Works against |
| Theta | -∞ to 0 | Works against | Works for you |
| Vega | 0 to +∞ | Benefits rising IV | Benefits falling IV |

## Practical Examples

### Example 1: CSP Greek Analysis

**Trade**: Sell AAPL $150 put, 30 DTE, for $3.00
**Greeks**: Delta +0.30, Gamma 0.02, Theta $0.08, Vega $0.15

**Analysis**:
- +30 delta = 30% chance ITM, position benefits from stock rise
- Positive theta = collecting $8/day per contract
- Negative vega exposure = want IV to fall

### Example 2: Covered Call Greek Analysis

**Trade**: Own 100 NVDA shares at $450, sell $480 call, 21 DTE
**Stock Delta**: +100 (own 100 shares)
**Call Delta**: -35 (short call)
**Net Delta**: +65

**Analysis**:
- Reduced delta exposure from +100 to +65
- Collecting theta on short call
- Capped upside at $480 strike
- Benefits if IV drops (short vega from call)

## Conclusion

Greeks are essential tools for:
1. Understanding position risk
2. Sizing positions appropriately
3. Managing portfolio exposure
4. Timing entries and exits
5. Selecting optimal strategies

Master the Greeks to move from gambling to strategic trading.
