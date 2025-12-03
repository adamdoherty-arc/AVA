# Covered Calls Mastery Guide

## What is a Covered Call?

A covered call is an options strategy where you sell a call option against shares of stock you already own. The call is "covered" because you have the underlying shares to deliver if assigned.

## Basic Mechanics

### Setup Requirements

- Own at least 100 shares of the underlying stock
- Sell 1 call option per 100 shares owned
- Collect premium immediately
- Obligation to sell shares at strike price if called away

### Example Trade

```
Stock: AAPL at $175
You own: 200 shares
Action: Sell 2 AAPL $180 calls for $3.00 each
Premium received: $600 (2 x $3.00 x 100)

Outcomes at expiration:
1. AAPL at $172: Keep shares + $600 premium
2. AAPL at $180: Keep shares + $600 premium
3. AAPL at $190: Shares called away at $180, profit = $5/share + $3 premium = $8/share
```

## Strike Selection Strategy

### By Objective

| Goal | Strike Selection | Delta Target |
|------|------------------|--------------|
| Maximum income | ATM or slight OTM | 40-50 delta |
| Keep shares likely | Far OTM | 10-15 delta |
| Exit position | ITM or ATM | 50+ delta |
| Balanced approach | 1 strike OTM | 25-35 delta |

### Delta-Based Selection

**High Delta (50+)**:
- Higher premium but greater chance of assignment
- Good when you want to exit the position
- More downside protection

**Medium Delta (25-35)**:
- Balanced risk/reward
- Reasonable premium with good probability of keeping shares
- Most popular for income traders

**Low Delta (10-20)**:
- Lower premium but high probability of keeping shares
- Good for stocks you don't want to sell
- Less downside protection

## Optimal Expiration Timing

### Days to Expiration (DTE)

| DTE Range | Best For | Characteristics |
|-----------|----------|-----------------|
| 7-14 days | Weekly income | Highest theta decay, more management |
| 21-30 days | Balanced | Good theta, time for adjustment |
| 30-45 days | Standard | Optimal theta decay curve |
| 45-60 days | Less management | Lower theta but fewer trades |

### Theta Decay Sweet Spot

```
Theta decay accelerates significantly inside 45 DTE:
- 45 DTE: ~0.4% daily theta decay
- 30 DTE: ~0.6% daily theta decay
- 21 DTE: ~0.8% daily theta decay
- 14 DTE: ~1.2% daily theta decay
- 7 DTE: ~2.0% daily theta decay
```

**Recommendation**: Sell at 30-45 DTE, close at 50% profit or 7-14 DTE remaining.

## Premium Targets

### Minimum Return Expectations

```
Monthly call premium target: 1-2% of stock value
Annual target: 12-24% additional return

Example:
$175 stock, targeting 1.5% monthly
Target premium: $175 x 0.015 = $2.63 per share
Annualized: 18% extra return on capital
```

### Premium/Risk Calculation

```
Premium Yield = Premium / Stock Price x 100
If Called Return = ((Strike - Stock Price) + Premium) / Stock Price x 100
Max Profit = (Strike - Cost Basis) + Premium
```

## Management Strategies

### Close at 50% Profit

**Rule**: Close covered call when 50% of maximum profit is achieved.

```
Example:
Sold call for $3.00
When call is worth $1.50, buy to close
Profit: $1.50 per share
Free up capital to sell new call
```

**Benefits**:
- Lock in profits
- Reduce gamma risk
- More efficient capital use
- Higher annualized returns

### Rolling Covered Calls

**Roll Up**: Higher strike, same expiration
- Use when stock rises and you want to capture more upside
- May need to pay debit

**Roll Out**: Same strike, later expiration
- Use when approaching expiration and want to avoid assignment
- Collect more premium

**Roll Up and Out**: Higher strike, later expiration
- Most common roll when stock is threatening strike
- Best to do for net credit

```
Rolling Example:
Original: $180 call expiring Friday, worth $0.50
Stock at $179, want to keep shares

Roll to: $185 call, 30 days out
- Buy back $180 call: -$0.50
- Sell $185 call: +$2.00
- Net credit: $1.50
```

### When NOT to Roll

- Stock is significantly above strike (roll would be expensive)
- You actually want to sell the shares
- Better opportunity elsewhere
- Fundamentals have changed

## Adjustment Techniques

### Stock Drops Significantly

**Options when stock falls**:
1. Let call expire worthless, sell new call
2. Buy back call early if near $0, sell new ATM call
3. Do nothing - wait for expiration

**Mistake to avoid**: Selling calls below your cost basis just for premium.

### Stock Rises Above Strike

**Options when stock rallies**:
1. Let shares get called away (take the profit)
2. Roll up and out for credit
3. Buy back call and hold (if expecting more upside)

### Approaching Ex-Dividend Date

**Early assignment risk increases** when:
- Call is ITM
- Time value < dividend amount
- Expiration is after ex-date

**Action**: Consider closing ITM calls before ex-dividend date.

## Tax Considerations

### Qualified Covered Calls

To avoid resetting the holding period for long-term capital gains:
- Strike must be at least one strike below ATM for 30 DTE
- Strike must be ATM or higher for shorter durations
- Complex rules for deep ITM calls

### Tax Treatment

**Premium received**: Short-term capital gain when call expires or is closed
**Stock called away**: Capital gain/loss based on cost basis and holding period
**Combined**: Premium reduces effective cost basis for tax calculation

## Advanced Strategies

### Systematic Covered Calls

**Monthly rotation system**:
1. Week 1: Identify positions for new calls
2. Week 2: Sell calls on 30-45 DTE cycle
3. Week 3-4: Monitor and manage
4. Week 4-5: Roll or close at 50% profit

### Laddering Strikes

**Split position across multiple strikes**:
```
Own 500 shares of XYZ at $100
- Sell 2 contracts at $105 strike
- Sell 2 contracts at $110 strike
- Sell 1 contract at $115 strike

Benefits:
- Diversified risk
- Some shares called if stock rises moderately
- Keep some shares for bigger moves
```

### Covered Strangles

**Sell both covered call AND cash-secured put**:
```
Own 100 shares at $100
- Sell $105 call for $2.00
- Sell $95 put for $2.00 (cash-secured)
- Total premium: $4.00

Risk: Must buy 100 more shares if put assigned
Benefit: Double the premium income
```

## Common Mistakes

### 1. Selling Calls on Losers
- Don't sell calls on stocks you should sell
- Covered calls are for stocks you want to hold

### 2. Strikes Too Close to ATM
- Higher premium but more likely to lose shares
- Balance income vs keeping quality positions

### 3. Ignoring Cost Basis
- Never sell calls below your cost basis
- Track adjusted cost basis after rolling

### 4. Not Having an Exit Plan
- Define when you'll roll vs let shares go
- Know your price targets before selling

### 5. Over-Covering
- Don't sell calls on every share
- Keep some uncovered for upside participation

## Performance Metrics

### Tracking Your Covered Calls

```
Track per trade:
- Entry date
- Stock price at entry
- Strike and expiration
- Premium received
- Exit date and price
- Outcome (expired, closed, assigned)
- Profit/loss
- Days held
- Annualized return

Monthly summary:
- Total premium collected
- Assignment rate
- Win rate
- Average return per trade
```

### Realistic Expectations

```
Conservative (15 delta): 8-12% annual added return
Moderate (25-30 delta): 12-18% annual added return
Aggressive (40+ delta): 18-25% annual added return

Note: Higher returns come with more assignments
and potentially capped upside
```

## Stock Selection for Covered Calls

### Ideal Candidates

- Stocks you want to hold long-term
- Moderate IV (higher premiums)
- Liquid options market
- Strong fundamentals
- Clear support/resistance levels

### Avoid for Covered Calls

- High-growth stocks you don't want capped
- Stocks before major announcements
- Illiquid options (wide bid-ask)
- Stocks in strong downtrends

## Quick Reference

### Covered Call Checklist

Before selling:
- [ ] Stock is one I want to own
- [ ] Strike is above my cost basis
- [ ] Premium meets my return target
- [ ] No earnings before expiration (unless intentional)
- [ ] Checked ex-dividend date
- [ ] Position size is appropriate

After selling:
- [ ] Set alert at 50% profit
- [ ] Monitor for roll opportunities
- [ ] Track in trading journal
- [ ] Know my exit plan

## Conclusion

Covered calls are the most conservative options strategy for income generation. Success comes from:

1. **Stock selection**: Only cover stocks worth owning
2. **Consistent execution**: Regular monthly cycles
3. **Proper management**: Close winners early, roll when appropriate
4. **Realistic expectations**: 1-2% monthly is excellent

When executed systematically, covered calls can add 12-24% annually to your portfolio returns while reducing overall volatility.
