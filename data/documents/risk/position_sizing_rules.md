# Position Sizing Rules for Options Trading

## Introduction

Position sizing is the most critical aspect of risk management. Proper position sizing ensures that no single trade can significantly damage your portfolio, while still allowing for meaningful profits.

## The Foundation: Risk Per Trade

### The 1-2% Rule

**Never risk more than 1-2% of your account on a single trade.**

```
Maximum Risk Per Trade = Account Size × Risk Percentage

Example: $100,000 account, 1% risk
Max Risk = $100,000 × 0.01 = $1,000 per trade
```

### Why 1-2%?

**Mathematical survival**:
- 10 consecutive losses at 2% = 18% drawdown (survivable)
- 10 consecutive losses at 5% = 40% drawdown (difficult recovery)
- 10 consecutive losses at 10% = 65% drawdown (potentially fatal)

**Recovery requirements**:
| Drawdown | Recovery Needed |
|----------|-----------------|
| 10% | 11% |
| 20% | 25% |
| 30% | 43% |
| 40% | 67% |
| 50% | 100% |

## Position Sizing Methods

### Method 1: Fixed Dollar Risk

**Simplest approach**: Risk same dollar amount per trade.

```
Contracts = Dollar Risk / Max Loss Per Contract

Example:
- Account: $100,000
- Risk: 1% = $1,000
- Option premium: $3.00
- Max loss per contract: $300

Contracts = $1,000 / $300 = 3.3 → 3 contracts
```

### Method 2: Fixed Percentage Risk

**Risk same percentage of current account balance.**

```
Current Account = $95,000 (after losses)
Risk = 1% = $950
Contracts = $950 / $300 = 3.16 → 3 contracts
```

**Benefit**: Position size adjusts with account swings.

### Method 3: Kelly Criterion

**Optimal sizing based on edge and win rate.**

```
Kelly % = (Win% × Avg Win / Avg Loss) - (Loss% / Avg Loss)

Simplified Kelly = Win% - (Loss% / Win:Loss Ratio)
```

**Example**:
- Win rate: 70%
- Average win: $150
- Average loss: $300
- Win:Loss ratio: 0.5

```
Kelly = 0.70 - (0.30 / 0.5) = 0.70 - 0.60 = 0.10 = 10%
```

**Important**: Use Half Kelly (5%) for safety margin.

### Method 4: Volatility-Based Sizing

**Adjust size based on underlying volatility.**

```
Position Size = Risk $ / (ATR × Multiplier)

Example:
- NVDA ATR: $15/day
- Risk: $1,000
- Multiplier: 2 (2× daily move stop)

Equivalent shares = $1,000 / ($15 × 2) = 33 shares
```

## Options-Specific Sizing

### Cash-Secured Puts

**Capital requirement**:
```
Capital Required = Strike Price × 100 × Contracts

Example: Sell 2 AAPL $150 puts
Capital = $150 × 100 × 2 = $30,000
```

**Position limit calculation**:
```
Max Position % = 5% of portfolio
Max Position = $100,000 × 0.05 = $5,000 at risk

If strike = $150, premium = $3.00
Max loss = $147 × 100 = $14,700 per contract
Contracts = $5,000 / $14,700 = 0.34 → careful!

Better approach: Use 50% of strike as max loss estimate
Max loss = $75 × 100 = $7,500 per contract
Contracts = $5,000 / $7,500 = 0.67 → 0-1 contracts
```

### Covered Calls

**Position already sized by shares owned.**

Focus on:
- Not selling more calls than shares owned
- Strike selection for income vs called-away risk

### Long Options

**Size based on maximum loss (premium paid).**

```
Contracts = Risk $ / (Premium × 100)

Example:
- Risk: $500
- Call premium: $2.50

Contracts = $500 / ($2.50 × 100) = 2 contracts
```

### Spreads

**Size based on max loss of spread.**

```
Vertical Spread Max Loss = (Width - Credit) × 100

Example: $5 wide put spread, $1.50 credit
Max loss = ($5 - $1.50) × 100 = $350 per spread

Contracts = $1,000 risk / $350 = 2.86 → 2 spreads
```

## Portfolio-Level Sizing

### Concentration Limits

**Single position limits**:
- Maximum 5% of portfolio per position
- Maximum 10% of portfolio per underlying
- Maximum 25% in single sector

**Example allocation for $100,000 account**:
```
Position A: $5,000 (5%)
Position B: $5,000 (5%)
Position C: $5,000 (5%)
...
Maximum 20 positions at 5% each
```

### Correlation Consideration

**Correlated positions compound risk.**

```
Tech positions: AAPL, MSFT, NVDA all correlate
If 5% each = 15% exposure to "tech drops" event

Solution: Treat correlated positions as single exposure
Combined tech limit: 15% total
```

### Delta Dollar Exposure

**Portfolio-wide directional risk.**

```
Delta Dollars = Sum of (Delta × 100 × Contracts × Stock Price)

Example Portfolio:
AAPL CSP: +30 delta × $150 = $4,500
NVDA CC: +50 delta × $450 = $22,500
SPY put: -20 delta × $420 = -$8,400

Total Delta Dollars: $18,600

As % of $100,000 account: 18.6% directional exposure
```

**Guidelines**:
- Conservative: <30% delta dollar exposure
- Moderate: 30-50%
- Aggressive: 50-75%

## Cash Management

### Reserve Requirements

**Always maintain cash reserves**:
- Minimum 20% cash for opportunities
- Additional cash for potential assignments
- Emergency fund separate from trading capital

### Assignment Capital

```
Assignment Reserve = Sum of (Strike × 100) for all short puts

Example:
3 AAPL $150 puts: 3 × $150 × 100 = $45,000
2 NVDA $400 puts: 2 × $400 × 100 = $80,000
Total: $125,000 needed if all assigned

Must have: $125,000 available or ability to close
```

### Margin Considerations

**Margin rules for options**:
- CSPs: Full cash secured or margin
- Naked calls: High margin requirement
- Spreads: Defined risk reduces margin

**Avoid**:
- Using >50% of available margin
- Margin calls force liquidation at bad times

## Position Sizing Workflow

### Step 1: Define Account Risk

```
Account Size: $100,000
Max Risk Per Trade: 1% = $1,000
Max Position Size: 5% = $5,000
```

### Step 2: Analyze Trade

```
Trade: Sell AAPL $150 put @ $3.00, 30 DTE
Delta: 0.25
Capital Required: $15,000
Max Loss: ~$14,700 (stock to $0)
Realistic Max Loss: $4,500 (30% stock drop)
```

### Step 3: Calculate Position Size

```
Using realistic max loss:
Contracts = $1,000 / $4,500 = 0.22

This suggests < 1 contract
Or accept higher risk: 1 contract = ~$4,500 at risk (4.5%)
```

### Step 4: Check Portfolio Limits

```
Current Portfolio:
- Tech exposure: 12%
- Total delta dollars: $25,000
- Cash: 35%

Adding AAPL ($15,000 capital):
- New tech exposure: 12% + 15% = 27% (above 25% limit)

Decision: Either skip trade or reduce existing tech
```

### Step 5: Execute Appropriate Size

Final decision based on all factors.

## Position Sizing Mistakes

### Over-Sizing

**Symptoms**:
- Stress watching positions
- Single trades cause major P&L swings
- Emotional trading decisions

**Solution**: Reduce to 1-2% risk max

### Under-Sizing

**Symptoms**:
- Profits don't meaningfully impact account
- Many trades but no account growth
- Transaction costs eat profits

**Solution**: Ensure position sizes are meaningful (0.5% minimum)

### Concentration

**Symptoms**:
- 3-5 positions represent >50% of account
- Single stock news causes large account swings
- Limited diversification

**Solution**: Limit individual positions, diversify across sectors

### Ignoring Correlation

**Symptoms**:
- Multiple positions move together
- Losses compound on down days
- False sense of diversification

**Solution**: Count correlated positions toward single limit

## Position Sizing Table

| Account Size | 1% Risk | 2% Risk | 5% Max Position |
|--------------|---------|---------|-----------------|
| $25,000 | $250 | $500 | $1,250 |
| $50,000 | $500 | $1,000 | $2,500 |
| $100,000 | $1,000 | $2,000 | $5,000 |
| $250,000 | $2,500 | $5,000 | $12,500 |
| $500,000 | $5,000 | $10,000 | $25,000 |

## Quick Reference Rules

### Conservative Approach
- 0.5-1% risk per trade
- 3% max position size
- 50% max invested
- 20 delta for income trades

### Moderate Approach
- 1-2% risk per trade
- 5% max position size
- 70% max invested
- 30 delta for income trades

### Aggressive Approach
- 2-3% risk per trade
- 7.5% max position size
- 85% max invested
- 40 delta for income trades

## Conclusion

Position sizing protects your capital and ensures long-term survival. The goal is not to maximize any single trade but to stay in the game long enough for your edge to play out.

**Key principles**:
1. Never risk more than you can afford to lose
2. Size positions to survive consecutive losses
3. Diversify across stocks, sectors, and strategies
4. Maintain adequate cash reserves
5. Adjust sizing based on volatility and correlation
