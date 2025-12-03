# Vertical Spreads: Complete Guide

## What Are Vertical Spreads?

A vertical spread involves buying and selling two options of the same type (both calls or both puts) with the same expiration date but different strike prices. They offer defined risk and defined reward.

## Types of Vertical Spreads

### Bull Call Spread (Debit)

**Setup**: Buy lower strike call, sell higher strike call
**Outlook**: Moderately bullish
**Risk**: Limited to debit paid
**Reward**: Limited to spread width minus debit

```
Example:
Stock at $100
Buy $100 call for $5.00
Sell $105 call for $2.00
Net debit: $3.00

Max profit: $5.00 - $3.00 = $2.00 ($200 per spread)
Max loss: $3.00 ($300 per spread)
Breakeven: $103.00
```

### Bull Put Spread (Credit)

**Setup**: Sell higher strike put, buy lower strike put
**Outlook**: Moderately bullish to neutral
**Risk**: Limited to spread width minus credit
**Reward**: Limited to credit received

```
Example:
Stock at $100
Sell $100 put for $4.00
Buy $95 put for $1.50
Net credit: $2.50

Max profit: $2.50 ($250 per spread)
Max loss: $5.00 - $2.50 = $2.50 ($250 per spread)
Breakeven: $97.50
```

### Bear Put Spread (Debit)

**Setup**: Buy higher strike put, sell lower strike put
**Outlook**: Moderately bearish
**Risk**: Limited to debit paid
**Reward**: Limited to spread width minus debit

```
Example:
Stock at $100
Buy $100 put for $4.00
Sell $95 put for $1.50
Net debit: $2.50

Max profit: $5.00 - $2.50 = $2.50 ($250 per spread)
Max loss: $2.50 ($250 per spread)
Breakeven: $97.50
```

### Bear Call Spread (Credit)

**Setup**: Sell lower strike call, buy higher strike call
**Outlook**: Moderately bearish to neutral
**Risk**: Limited to spread width minus credit
**Reward**: Limited to credit received

```
Example:
Stock at $100
Sell $100 call for $5.00
Buy $105 call for $2.00
Net credit: $3.00

Max profit: $3.00 ($300 per spread)
Max loss: $5.00 - $3.00 = $2.00 ($200 per spread)
Breakeven: $103.00
```

## Debit Spreads vs Credit Spreads

### Debit Spreads

| Characteristic | Description |
|----------------|-------------|
| Cost | Pay upfront (debit) |
| Time decay | Works against you |
| Need | Stock to move in your direction |
| IV preference | Low IV entry, IV expansion helps |
| Probability | Lower probability, higher reward ratio |

### Credit Spreads

| Characteristic | Description |
|----------------|-------------|
| Cost | Receive premium upfront (credit) |
| Time decay | Works for you |
| Need | Stock to stay above/below strike |
| IV preference | High IV entry, IV contraction helps |
| Probability | Higher probability, lower reward ratio |

## Spread Width Selection

### Narrow Spreads ($2.50-$5)

**Advantages**:
- Lower capital requirement
- More contracts possible
- Better percentage returns

**Disadvantages**:
- Lower absolute profit per spread
- Commission impact higher
- Need more precision

### Wide Spreads ($10-$20)

**Advantages**:
- Higher absolute profit potential
- Lower commission impact
- More room for stock movement

**Disadvantages**:
- More capital required
- Larger losses if wrong
- Lower percentage returns

### Width Selection Guidelines

```
Account size considerations:
- Under $25K: $2.50-$5 spreads
- $25K-$100K: $5-$10 spreads
- Over $100K: Any width based on conviction

Risk per trade (using $100K account, 2% risk):
$2K risk budget
- $5 wide spread at $2 risk = $200 risk per spread = 10 spreads max
- $10 wide spread at $4 risk = $400 risk per spread = 5 spreads max
```

## Strike Selection Strategy

### For Credit Spreads (High Probability)

**Short strike selection**:
- 70-80% probability of profit
- At or beyond 1 standard deviation
- Typically 20-30 delta

**Long strike selection**:
- 1-2 strikes beyond short strike
- Defines your maximum risk
- Determines capital requirement

### For Debit Spreads (Directional)

**Long strike selection**:
- ATM or slightly ITM for higher probability
- OTM for more leverage (lower probability)
- 40-60 delta common

**Short strike selection**:
- At your price target
- Caps your profit potential
- Reduces cost of trade

## Optimal Expiration Timing

### Credit Spreads

**Best DTE**: 30-45 days
- Optimal theta decay zone
- Time to be right
- Manageable gamma risk

**Close at**: 50% of max profit or 21 DTE

### Debit Spreads

**Best DTE**: 45-60 days
- Time for thesis to play out
- Reduced theta drag
- Better probability

**Close at**: 50-75% of max profit or if target reached

## Greeks Impact on Verticals

### Delta

**Credit spreads**:
- Bull put spread: Positive delta (want stock up)
- Bear call spread: Negative delta (want stock down)

**Debit spreads**:
- Bull call spread: Positive delta
- Bear put spread: Negative delta

### Theta

**Credit spreads**: Positive theta (time decay helps)
**Debit spreads**: Negative theta (time decay hurts)

### Vega

**Credit spreads**: Negative vega (IV contraction helps)
**Debit spreads**: Positive vega (IV expansion helps)

### Gamma

Near expiration, gamma increases for ATM spreads. This creates risk for credit spreads as delta can change rapidly.

## Management Rules

### Credit Spread Management

**Profit target**: Close at 50% of max profit
```
Example: Received $2.50 credit
When spread is worth $1.25, buy to close
Profit: $1.25 per spread
```

**Loss limit**: Close at 2x credit received or when short strike is breached
```
Example: Received $2.50 credit
If spread is worth $5.00, close for $2.50 loss
Maximum loss controlled
```

### Debit Spread Management

**Profit target**: Close at 50-75% of max profit
```
Example: $5 wide spread, paid $2.00
Max profit = $3.00
Close when spread worth $4.50+ (75% of max)
```

**Stop loss**: Close if underlying moves against by 1 ATR or defined percentage

### Rolling Verticals

**Roll credit spreads**:
- Down and out (puts) or up and out (calls)
- Only roll for additional credit
- Don't chase losing trades

**When to roll**:
- Short strike tested but not breached
- Can collect meaningful credit
- Still believe in trade thesis

## Position Sizing

### Risk-Based Sizing

```
Max risk per trade: 1-2% of account
Spreads to trade = Risk Budget / Max Loss per Spread

Example:
$100,000 account, 1% risk = $1,000 max loss
$5 wide spread at max $250 loss
Spreads = $1,000 / $250 = 4 spreads maximum
```

### Capital Requirement

**Credit spreads**: (Spread Width - Credit) x 100 x Contracts
**Debit spreads**: Debit Paid x 100 x Contracts

## Advanced Techniques

### Skip-Strike Spreads

**Broken wing butterfly alternative**:
- Buy ATM, skip a strike, sell OTM
- Wider spread, different risk profile

### Ratio Spreads

**1x2 ratio**:
- Buy 1 ATM, sell 2 OTM
- Can be done for credit
- Has undefined risk on one side

### Calendar + Vertical (Diagonal)

Combine time spread with strike differential for unique risk/reward profiles.

## Common Mistakes

### 1. Wrong Width for Account Size
- Too wide = too much risk
- Too narrow = commission drag
- Match width to risk tolerance

### 2. Fighting the Trend
- Bull put spreads in downtrends lose
- Bear call spreads in uptrends lose
- Trade with the trend

### 3. Ignoring IV
- Don't buy debit spreads when IV is high
- Don't sell credit spreads when IV is low
- Check IV rank before entry

### 4. Holding Too Long
- Close winners at 50%
- Don't let winners become losers
- Time is your friend (credit) or enemy (debit)

### 5. Overleveraging
- Multiple spreads compound risk
- Correlated underlyings multiply exposure
- Keep total portfolio risk manageable

## Vertical Spread Comparison

| Strategy | Outlook | Risk | Reward | Time Decay | Best When |
|----------|---------|------|--------|------------|-----------|
| Bull Call | Bullish | Limited | Limited | Against | Low IV, expect move |
| Bull Put | Bullish/Neutral | Limited | Limited | For | High IV, support holds |
| Bear Put | Bearish | Limited | Limited | Against | Low IV, expect drop |
| Bear Call | Bearish/Neutral | Limited | Limited | For | High IV, resistance holds |

## Selection Flowchart

```
Bullish on stock?
├── Yes → Want defined risk?
│   ├── Yes → Is IV high?
│   │   ├── Yes → Bull Put Spread (credit)
│   │   └── No → Bull Call Spread (debit)
│   └── No → Consider naked puts or long calls
└── No (Bearish) → Want defined risk?
    ├── Yes → Is IV high?
    │   ├── Yes → Bear Call Spread (credit)
    │   └── No → Bear Put Spread (debit)
    └── No → Consider naked calls or long puts
```

## Trade Example Walkthrough

### Bull Put Spread Example

**Thesis**: AAPL holding support at $170, expecting bounce

**Setup**:
- AAPL at $175
- Sell $170 put (25 delta) for $2.50
- Buy $165 put (15 delta) for $1.00
- Net credit: $1.50
- 30 DTE

**Risk/Reward**:
- Max profit: $150 per spread
- Max loss: $350 per spread ($5 width - $1.50 credit)
- Breakeven: $168.50
- Probability of profit: ~70%

**Management**:
- Close at $0.75 (50% profit)
- Close if AAPL breaks $170
- Roll if tested but holding

**Outcome scenarios**:
1. AAPL stays above $170: Keep $150 (100% of max)
2. AAPL at $168 at expiration: Lose $200 ($170 - $168 - $1.50 credit)
3. AAPL at $165 or below: Max loss $350

## Conclusion

Vertical spreads are the foundation of defined-risk options trading. Key principles:

1. **Match spread to outlook**: Debit for moves, credit for range
2. **Size appropriately**: Never risk more than 1-2% per trade
3. **Manage winners**: Close at 50% profit
4. **Respect IV**: Sell high IV, buy low IV
5. **Trade with trend**: Higher probability of success

Master verticals before moving to more complex strategies like iron condors or butterflies.
