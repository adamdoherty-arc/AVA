# Options Trading Fundamentals

## What Are Options?

Options are financial contracts that give the buyer the right, but not the obligation, to buy or sell an underlying asset at a predetermined price within a specified time period.

### Key Terms

- **Underlying Asset**: The stock, ETF, or index the option is based on
- **Strike Price**: The price at which the option can be exercised
- **Expiration Date**: The date when the option contract expires
- **Premium**: The price paid (or received) for the option
- **Contract Size**: Standard options represent 100 shares

## Types of Options

### Call Options

**Definition**: A call option gives the buyer the right to BUY 100 shares at the strike price.

**When to buy calls**:
- You expect the stock to rise
- You want leveraged exposure to upside
- You want to limit downside risk

**When to sell calls**:
- You own shares and want income (covered call)
- You expect the stock to stay flat or decline
- You want to generate premium

**Call Option Payoff at Expiration**:
```
For Buyer: Max(Stock Price - Strike Price, 0) - Premium Paid
For Seller: Premium Received - Max(Stock Price - Strike Price, 0)
```

### Put Options

**Definition**: A put option gives the buyer the right to SELL 100 shares at the strike price.

**When to buy puts**:
- You expect the stock to decline
- You want to hedge existing positions
- You want downside protection

**When to sell puts**:
- You want to buy stock at a lower price (CSP)
- You expect the stock to stay flat or rise
- You want to generate premium

**Put Option Payoff at Expiration**:
```
For Buyer: Max(Strike Price - Stock Price, 0) - Premium Paid
For Seller: Premium Received - Max(Strike Price - Stock Price, 0)
```

## Option Moneyness

### In-The-Money (ITM)

**Calls**: Stock price > Strike price
**Puts**: Stock price < Strike price

ITM options have intrinsic value and are more expensive. They have higher deltas and higher probability of profit.

### At-The-Money (ATM)

**Calls and Puts**: Stock price ≈ Strike price

ATM options have the most time value and highest theta decay rate. They have approximately 50 delta.

### Out-of-The-Money (OTM)

**Calls**: Stock price < Strike price
**Puts**: Stock price > Strike price

OTM options have no intrinsic value, only time value. They're cheaper but have lower probability of profit.

## Option Value Components

### Intrinsic Value

The amount an option would be worth if exercised immediately.

```
Call Intrinsic Value = Max(Stock Price - Strike, 0)
Put Intrinsic Value = Max(Strike - Stock Price, 0)
```

### Extrinsic Value (Time Value)

The portion of premium above intrinsic value, representing:
- Time until expiration
- Implied volatility
- Interest rates
- Dividends

```
Extrinsic Value = Option Premium - Intrinsic Value
```

**Example**:
- Stock at $105
- $100 Call trading at $7.00
- Intrinsic Value: $105 - $100 = $5.00
- Extrinsic Value: $7.00 - $5.00 = $2.00

## Strike Price Selection

### For Long Calls

| Goal | Recommended Strike |
|------|-------------------|
| Maximum leverage | Deep OTM (low probability) |
| Balanced approach | ATM or slightly OTM |
| Higher probability | ITM (pay for intrinsic) |
| Stock replacement | Deep ITM (high delta) |

### For Short Puts (CSPs)

| Goal | Recommended Strike |
|------|-------------------|
| Maximum premium | ATM (highest theta) |
| Higher probability | OTM 20-30 delta |
| Stock acquisition | ATM or slightly ITM |
| Conservative income | 10-15 delta OTM |

### For Covered Calls

| Goal | Recommended Strike |
|------|-------------------|
| Maximum premium | ATM or slightly OTM |
| Keep shares | Far OTM (10-15 delta) |
| Exit position | ITM or ATM |
| Balanced | 30 delta OTM |

## Expiration Dates

### Understanding Expiration Cycles

**Weekly Options**: Expire every Friday
**Monthly Options**: Expire third Friday of month
**LEAPS**: Long-term options (1-2 years out)

### DTE Selection Guide

| Strategy | Recommended DTE | Rationale |
|----------|-----------------|-----------|
| Day trading | 0-1 DTE | Maximum gamma |
| Swing trading | 7-21 DTE | Balance of theta/time |
| Income strategies | 30-45 DTE | Optimal theta decay |
| LEAPS buying | 6-24 months | Time for thesis |
| Hedging | Match position duration | Protect for needed time |

### Time Value Decay Pattern

```
Time value decays exponentially:
- 90 DTE: Slow decay (~1-2% per week)
- 45 DTE: Moderate decay (~3-5% per week)
- 21 DTE: Accelerating decay (~7-10% per week)
- 7 DTE: Rapid decay (~15-25% per week)
- 0-2 DTE: Maximum decay
```

## Option Pricing Basics

### Factors Affecting Option Prices

1. **Stock Price**: Primary driver of option value
2. **Strike Price**: Determines intrinsic value
3. **Time to Expiration**: More time = more premium
4. **Volatility**: Higher IV = higher premium
5. **Interest Rates**: Minor effect, raises calls, lowers puts
6. **Dividends**: Lowers calls, raises puts

### Black-Scholes Inputs

The standard option pricing model uses:
- Current stock price (S)
- Strike price (K)
- Time to expiration (T)
- Risk-free interest rate (r)
- Volatility (σ)
- Dividend yield (q)

## Exercise and Assignment

### Exercise

**Definition**: The option buyer uses their right to buy (call) or sell (put) shares.

**American-style options**: Can be exercised any time before expiration
**European-style options**: Can only be exercised at expiration

### Assignment

**Definition**: The option seller is obligated to fulfill the contract when buyer exercises.

**Call assignment**: Must sell 100 shares at strike price
**Put assignment**: Must buy 100 shares at strike price

### Automatic Exercise

Most brokers automatically exercise options that are ITM by $0.01 or more at expiration.

**To avoid unwanted exercise**:
- Close position before expiration
- Submit Do Not Exercise (DNE) instruction

## Long vs Short Options

### Long Options (Buying)

**Characteristics**:
- Pay premium upfront
- Limited risk (premium paid)
- Unlimited profit potential (calls)
- Time decay works against you
- Need stock to move in your direction

**Best for**:
- Directional speculation
- Hedging existing positions
- Leveraged exposure
- Defined risk trades

### Short Options (Selling)

**Characteristics**:
- Receive premium upfront
- Potentially unlimited risk (naked calls)
- Limited profit (premium received)
- Time decay works for you
- Can profit even if stock doesn't move

**Best for**:
- Income generation
- Non-directional strategies
- High probability trades
- Taking advantage of high IV

## Order Types

### Market Orders

Execute immediately at best available price. Use for liquid options when speed is priority.

### Limit Orders

Execute only at specified price or better. Recommended for most option trades.

### Spread Orders

Execute multi-leg strategies as a single order. Ensures all legs fill together.

### Good Till Canceled (GTC)

Order remains active until filled or canceled. Useful for limit orders on illiquid options.

## Bid-Ask Spread

### Understanding the Spread

- **Bid**: Price buyers are willing to pay
- **Ask**: Price sellers are willing to accept
- **Spread**: Difference between bid and ask

### Spread Impact on Trading

```
Immediate cost of trade = Spread / 2
Example: $2.00 bid / $2.20 ask = $0.20 spread
Cost to enter and exit = $0.20 total
```

### Tight vs Wide Spreads

**Tight Spread (Good)**: < 5% of option price
**Wide Spread (Caution)**: > 10% of option price

**Factors causing wide spreads**:
- Low volume
- Low open interest
- Illiquid underlying
- Complex strikes

## Options Chains

### Reading an Options Chain

An options chain displays all available options for an underlying, showing:
- Strike prices
- Bid/Ask prices
- Volume
- Open interest
- Greeks (delta, theta, etc.)
- Implied volatility

### Identifying Liquid Options

Look for:
- High open interest (>1000)
- High volume
- Tight bid-ask spreads
- Standard strike intervals

## Common Beginner Strategies

### Long Call

**Setup**: Buy a call option
**Max Profit**: Unlimited
**Max Loss**: Premium paid
**Breakeven**: Strike + Premium

### Long Put

**Setup**: Buy a put option
**Max Profit**: Strike - Premium (if stock goes to 0)
**Max Loss**: Premium paid
**Breakeven**: Strike - Premium

### Covered Call

**Setup**: Own 100 shares + Sell 1 call
**Max Profit**: (Strike - Stock Price) + Premium
**Max Loss**: Stock Price - Premium (if stock goes to 0)
**Breakeven**: Stock Price - Premium

### Cash-Secured Put

**Setup**: Have cash for assignment + Sell 1 put
**Max Profit**: Premium received
**Max Loss**: Strike - Premium (if stock goes to 0)
**Breakeven**: Strike - Premium

## Risk Considerations

### Greeks Overview

- **Delta**: Price change per $1 stock move
- **Gamma**: Delta change per $1 stock move
- **Theta**: Daily time decay
- **Vega**: Price change per 1% IV change

### Position Sizing

Never risk more than 2-5% of portfolio on a single trade.

```
Position Size = (Account Risk %) / (Max Loss per Contract)
Example: $100,000 account, 2% risk, $500 max loss
Contracts = ($100,000 × 0.02) / $500 = 4 contracts
```

### Common Risks

1. **Unlimited loss** (naked calls)
2. **Time decay** (long options)
3. **IV crush** (after events)
4. **Assignment risk** (short options)
5. **Liquidity risk** (illiquid options)
6. **Gap risk** (overnight moves)

## Getting Started Checklist

- [ ] Understand calls and puts
- [ ] Know the difference between buying and selling
- [ ] Learn to read options chains
- [ ] Practice with paper trading
- [ ] Start with defined-risk strategies
- [ ] Master one strategy before adding more
- [ ] Never trade more than you can afford to lose
- [ ] Track all trades and learn from results
