# Portfolio Risk Management

## The Foundation of Successful Trading

Risk management is not about avoiding losses - it's about controlling their size while allowing winners to flourish. The best traders aren't those who avoid losses; they're those who manage them effectively.

## Core Risk Principles

### Principle 1: Preservation of Capital

**Capital is your ammunition** - without it, you can't trade.

```
Priorities in order:
1. Don't lose money (capital preservation)
2. Don't lose money (repeat for emphasis)
3. Make money
```

### Principle 2: Asymmetric Risk/Reward

**Always seek favorable odds**:
- Minimum 1:1.5 risk/reward ratio
- Prefer 1:2 or better
- Small losses, big wins = profitable system

### Principle 3: Position Independence

**Each trade should stand alone**:
- Don't average down hoping for recovery
- Don't let one position dominate portfolio
- Don't let emotions from one trade affect others

## Risk Metrics Every Trader Needs

### Maximum Drawdown

**Definition**: Largest peak-to-trough decline in account value

```
Calculating Max Drawdown:
Peak account value: $120,000
Lowest point before new peak: $95,000
Max Drawdown = ($120,000 - $95,000) / $120,000 = 20.8%
```

**Acceptable drawdowns by trader type**:
| Trader Type | Max Acceptable Drawdown |
|-------------|------------------------|
| Conservative | 10-15% |
| Moderate | 15-25% |
| Aggressive | 25-35% |
| Professional | Varies by mandate |

### Value at Risk (VaR)

**Definition**: Maximum expected loss at a given confidence level

```
95% VaR Example:
"There's a 95% chance we won't lose more than $5,000 in a day"

Calculation (simplified):
Portfolio value: $100,000
Daily volatility: 1.5%
95% VaR = $100,000 × 1.5% × 1.65 = $2,475

In 95% of days, loss won't exceed $2,475
```

### Sharpe Ratio

**Definition**: Risk-adjusted return measure

```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Std Deviation

Example:
Annual return: 25%
Risk-free rate: 5%
Standard deviation: 15%

Sharpe = (25% - 5%) / 15% = 1.33

Interpretation:
< 1.0 = Suboptimal
1.0-2.0 = Good
2.0-3.0 = Very Good
> 3.0 = Excellent
```

### Win Rate and Profit Factor

```
Win Rate = Winning Trades / Total Trades × 100

Profit Factor = Gross Profits / Gross Losses

Example:
100 trades, 65 winners, 35 losers
Total profits: $15,000
Total losses: $8,000

Win Rate: 65%
Profit Factor: $15,000 / $8,000 = 1.875
```

## Portfolio-Level Risk Controls

### Maximum Portfolio Heat

**Definition**: Total capital at risk across all open positions

```
Portfolio Heat Limit: 6-10% of account

Example ($100,000 account):
Position 1: $1,000 at risk (1%)
Position 2: $1,500 at risk (1.5%)
Position 3: $2,000 at risk (2%)
Position 4: $1,500 at risk (1.5%)
Total Heat: $6,000 (6%)

If opening new trade risks $2,000 more:
New total: 8% - within limit, proceed
```

### Correlation-Adjusted Risk

**Problem**: Multiple positions in correlated assets compound risk

```
Example:
Position 1: AAPL options (1% risk)
Position 2: MSFT options (1% risk)
Position 3: NVDA options (1% risk)
Position 4: QQQ options (1% risk)

Nominal risk: 4%
Correlated risk: ~7-8% (tech stocks move together)

Solution: Treat correlated positions as single exposure
Combined tech limit: 3-4% total
```

### Sector Limits

**Maximum sector exposure**:
- Single sector: 25% of portfolio
- Two sectors combined: 40%
- Maintain minimum 3-4 sectors

```
$100,000 portfolio allocation:
Technology: $25,000 max (25%)
Financials: $15,000 (15%)
Healthcare: $20,000 (20%)
Consumer: $15,000 (15%)
Energy: $10,000 (10%)
Cash: $15,000 (15%)
```

## Daily Risk Management

### Pre-Market Checklist

```
Before trading each day:
□ Check overnight news for held positions
□ Review economic calendar
□ Assess overall market sentiment
□ Calculate current portfolio heat
□ Review open order status
□ Set daily loss limit
□ Identify potential opportunities
```

### Daily Loss Limit

**Rule**: Stop trading when daily loss reaches 2-3% of account

```
$100,000 account:
Daily loss limit: $2,000-$3,000

Purpose:
- Prevents emotional trading
- Stops "revenge trading"
- Preserves capital for better days
- Enforces discipline
```

### Position Review Schedule

```
Intraday (if day trading):
- Check every 30-60 minutes
- Adjust stops if needed

Swing trading:
- Morning review
- End of day review
- Weekend full portfolio review

Options income:
- Daily theta check
- Monitor approaching strikes
- Weekly roll assessment
```

## Emergency Risk Protocols

### Black Swan Events

**Definition**: Rare, unpredictable events with severe impact

**Preparation strategies**:
1. Always have defined max loss
2. Use options for unlimited risk protection
3. Maintain cash reserves
4. Have hedges in place
5. Know your broker's emergency procedures

### Circuit Breakers

**Market-wide trading halts**:
- Level 1: 7% decline = 15 min halt
- Level 2: 13% decline = 15 min halt
- Level 3: 20% decline = trading stops for day

**Personal circuit breakers**:
- 3% daily loss = stop trading
- 10% monthly drawdown = reduce size 50%
- 20% drawdown = stop, reassess strategy

### Gap Risk Management

**Overnight gap protection**:
```
Strategies:
1. Reduce position size before events
2. Use spreads vs naked options
3. Maintain smaller positions in volatile stocks
4. Consider protective options (puts for long positions)
```

## Risk-Adjusted Position Sizing

### Volatility-Based Sizing

```
Position Size = Risk Amount / (ATR × Multiplier)

Example:
Risk per trade: $1,000
Stock ATR: $5
Multiplier: 2 (2x daily range)

Shares = $1,000 / ($5 × 2) = 100 shares
```

### Options-Specific Sizing

**For premium selling**:
```
CSP position size:
Max capital = Strike × 100 × Contracts
Risk per contract = Strike - Expected Support

Example:
$100 strike, support at $90
Risk per contract: ~$1,000 (10% decline)
$1,000 risk budget = 1 contract
```

**For option buying**:
```
Position size = Risk budget / Premium paid

Example:
Risk budget: $500
Option premium: $2.50
Contracts = $500 / $250 = 2 contracts
```

## Hedging Strategies

### Portfolio Hedging with Options

**Protective puts**:
```
Cost: 1-3% of portfolio per quarter
Protection: Limits downside to strike price
Best for: Concentrated positions, market uncertainty
```

**Collar strategy**:
```
Own stock + Buy put + Sell call
Cost: Reduced or zero (call premium offsets put)
Protection: Floor and ceiling on position value
```

### Tail Risk Hedging

**VIX-based hedging**:
- Buy VIX calls when VIX is low
- Provides insurance against market crashes
- Typically expires worthless (insurance premium)

**Put spread hedges**:
```
Instead of expensive protective put:
Buy OTM put
Sell further OTM put

Example:
Portfolio value: $100,000
Buy 10 SPY $380 puts
Sell 10 SPY $350 puts

Cost: ~$2,000
Protection: $30 of downside covered
```

## Risk Documentation

### Trading Journal Requirements

```
For each trade, record:
1. Date and time
2. Entry price and size
3. Stop loss level
4. Target price
5. Risk/reward ratio
6. Thesis/reason
7. Exit date and price
8. Profit/loss
9. What went right/wrong
10. Lessons learned
```

### Monthly Risk Report

```
Monthly metrics to track:
- Total P&L
- Win rate
- Average win/loss
- Profit factor
- Max drawdown
- Sharpe ratio (rolling)
- Position count
- Sector exposure
- Largest winner/loser
- Rule violations
```

### Annual Risk Review

```
Annual assessment:
1. Strategy performance by type
2. Risk-adjusted returns
3. Drawdown analysis
4. Correlation impact
5. Black swan exposure
6. System improvements needed
7. Rule updates required
```

## Common Risk Management Mistakes

### Mistake 1: Oversizing

**Symptom**: Single trade causes major account swing
**Fix**: Never exceed 2% risk per trade

### Mistake 2: Ignoring Correlation

**Symptom**: Multiple positions lose simultaneously
**Fix**: Track and limit correlated exposure

### Mistake 3: No Stop Losses

**Symptom**: Small losses become catastrophic
**Fix**: Define max loss before every trade

### Mistake 4: Averaging Down

**Symptom**: Adding to losers hoping for recovery
**Fix**: Never add to losing positions

### Mistake 5: Revenge Trading

**Symptom**: Increasing size after losses to "get even"
**Fix**: Daily loss limits, mandatory breaks

### Mistake 6: Neglecting Tail Risk

**Symptom**: System works until it doesn't (blowup)
**Fix**: Hedge against extreme events

## Risk Management Framework Summary

### The 5 Pillars of Risk Management

```
1. POSITION SIZING
   - Max 1-2% risk per trade
   - Volatility-adjusted sizing
   - Correlation-aware allocation

2. PORTFOLIO LIMITS
   - Max 6-10% portfolio heat
   - Sector diversification
   - Asset class balance

3. DEFINED EXITS
   - Stop losses on every trade
   - Profit targets defined
   - Time-based exits

4. MONITORING
   - Daily portfolio review
   - Real-time alerting
   - Regular rebalancing

5. EMERGENCY PLANS
   - Black swan protocols
   - Circuit breaker rules
   - Hedge maintenance
```

## Conclusion

Risk management separates successful traders from failed ones. The market will always present opportunities, but only if you preserve your capital to capture them.

**Key takeaways**:
1. Position sizing is everything
2. Define risk before entering any trade
3. Diversify across uncorrelated assets
4. Have emergency protocols ready
5. Review and adapt continuously

Remember: **Professional traders think about risk first, reward second.**
