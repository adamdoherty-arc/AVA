# Sector Diversification for Options Traders

## Why Sector Diversification Matters

Concentrating positions in a single sector is one of the most common and dangerous mistakes traders make. Even great companies can suffer when their entire sector faces headwinds.

## The Case for Diversification

### Historical Sector Drawdowns

```
Major sector-specific crashes:

Technology (2000-2002): -78%
Financials (2008): -83%
Energy (2014-2016): -65%
Retail (2017-2019): -50%
Airlines (2020): -70%
Regional Banks (2023): -40%
```

**Lesson**: Every sector experiences major drawdowns. Concentration guarantees you'll experience one.

### Correlation During Crises

In normal markets, sectors have moderate correlation (~0.5-0.7). During crises, correlations spike to 0.9+, but sector differences still matter.

```
Example: March 2020 crash
Healthcare: -25%
Technology: -32%
Energy: -60%
Travel: -70%

A diversified portfolio fared better than energy-only
```

## GICS Sector Framework

### The 11 Sectors

| Sector | ETF | Key Characteristics |
|--------|-----|---------------------|
| Technology | XLK | High growth, high volatility |
| Healthcare | XLV | Defensive, regulatory risk |
| Financials | XLF | Interest rate sensitive |
| Consumer Discretionary | XLY | Economic cycle sensitive |
| Consumer Staples | XLP | Defensive, low volatility |
| Industrials | XLI | Economic cycle sensitive |
| Energy | XLE | Commodity-linked, volatile |
| Materials | XLB | Commodity-linked |
| Real Estate | XLRE | Interest rate sensitive |
| Utilities | XLU | Defensive, rate sensitive |
| Communication Services | XLC | Mixed (tech + traditional) |

### Sector Characteristics Matrix

```
                     Volatility  Growth   Defensiveness  Yield
Technology           High        High     Low            Low
Healthcare           Medium      Medium   Medium         Medium
Financials           High        Medium   Low            High
Cons Discretionary   High        High     Low            Low
Consumer Staples     Low         Low      High           High
Industrials          Medium      Medium   Low            Medium
Energy               Very High   Low      Low            High
Materials            High        Low      Low            Medium
Real Estate          Medium      Low      Medium         High
Utilities            Low         Low      High           High
Communication        Medium      Medium   Low            Medium
```

## Diversification Rules for Options

### Maximum Sector Exposure

**Conservative approach**:
```
Single sector: Max 15% of portfolio
Two sectors combined: Max 25%
Minimum sectors held: 5
```

**Moderate approach**:
```
Single sector: Max 25% of portfolio
Two sectors combined: Max 40%
Minimum sectors held: 4
```

**Aggressive approach**:
```
Single sector: Max 35% of portfolio
Two sectors combined: Max 50%
Minimum sectors held: 3
```

### Position Limits Within Sectors

```
Within any sector:
- Single stock: Max 5% of portfolio
- Two stocks in same sector: Max 8%
- Sector total: Follow sector limits above
```

## Building a Diversified Options Portfolio

### The Core-Satellite Approach

**Core positions (60-70%)**:
```
Broad market exposure via index options:
- SPY puts/calls
- QQQ puts/calls
- IWM puts/calls
- DIA puts/calls
```

**Satellite positions (30-40%)**:
```
Individual stock options across sectors:
- 2-3 positions in favorite sectors
- 1-2 positions in other sectors
- No sector > 25% including core
```

### Sample Diversified Portfolio

```
$100,000 Portfolio Example:

CORE (65%):
- SPY CSPs/CCs: $35,000 (35%)
- QQQ CSPs/CCs: $20,000 (20%)
- IWM CSPs/CCs: $10,000 (10%)

SATELLITE (35%):
Technology: $10,000 (10%)
- AAPL options: $5,000
- MSFT options: $5,000

Healthcare: $7,500 (7.5%)
- UNH options: $5,000
- JNJ options: $2,500

Financials: $7,500 (7.5%)
- JPM options: $5,000
- V options: $2,500

Energy: $5,000 (5%)
- XOM options: $5,000

Consumer: $5,000 (5%)
- HD options: $2,500
- COST options: $2,500
```

## Sector Rotation Strategy

### Understanding Sector Cycles

```
Economic Cycle and Sector Performance:

Early Recovery:
- Best: Financials, Consumer Discretionary, Industrials
- Worst: Utilities, Consumer Staples

Mid Cycle:
- Best: Technology, Industrials, Materials
- Worst: Utilities, Telecom

Late Cycle:
- Best: Energy, Materials, Healthcare
- Worst: Technology, Financials

Recession:
- Best: Utilities, Consumer Staples, Healthcare
- Worst: Financials, Consumer Discretionary, Industrials
```

### Tactical Allocation Adjustments

**When rotating sectors**:
```
Don't sell all positions immediately
Gradual shift over 2-4 weeks
Use options expiration as natural rotation point
Roll to new sectors rather than close-and-open
```

## Correlation Management

### High Correlation Traps

**Stocks that seem diversified but aren't**:
```
AAPL + MSFT + NVDA = All tech, all correlated
AMZN (consumer) + META (comm) + GOOGL (comm) = All big tech

Better diversification:
AAPL (tech) + JPM (financial) + JNJ (healthcare) + XOM (energy)
```

### Measuring Correlation

```
Portfolio correlation check:
1. Get 1-year daily returns for each position
2. Calculate correlation matrix
3. Average pairwise correlations

Target average correlation: < 0.50
Warning level: > 0.70
```

### Creating Uncorrelated Positions

**Mix these position types**:
```
Long delta + Short delta = Reduced directional correlation
Short vol + Long vol = Reduced volatility correlation
Different sectors = Reduced fundamental correlation
Different expirations = Reduced timing correlation
```

## Sector Analysis for Stock Selection

### Evaluating Sector Health

**Before adding sector exposure**:
```
Check:
1. Sector ETF trend (above/below 50-day MA)
2. Relative strength vs SPY
3. Sector earnings growth estimates
4. Interest rate sensitivity
5. Regulatory environment
6. Seasonal patterns
```

### Sector Red Flags

**Reduce or avoid sector when**:
- ETF below 200-day moving average
- Multiple stocks in sector have negative surprises
- Major regulatory changes announced
- Commodity price collapse (for commodity sectors)
- Interest rate changes hurt sector thesis

## Practical Implementation

### Weekly Sector Review

```
Weekly checklist:
□ Calculate current sector exposure
□ Compare to limits
□ Review sector performance rankings
□ Identify any concentration issues
□ Plan rebalancing if needed
```

### Sector Exposure Tracking

```
Sector Exposure Spreadsheet:

| Sector | Positions | Capital | % of Port | Limit | Status |
|--------|-----------|---------|-----------|-------|--------|
| Tech | AAPL,MSFT | $15,000 | 15% | 25% | OK |
| Health | UNH | $7,500 | 7.5% | 25% | OK |
| Finance | JPM,V | $10,000 | 10% | 25% | OK |
| Energy | XOM | $5,000 | 5% | 25% | OK |
| Index | SPY,QQQ | $55,000 | 55% | N/A | OK |
| Cash | - | $7,500 | 7.5% | - | - |
```

### Rebalancing Triggers

**Rebalance when**:
```
1. Single sector exceeds limit by 5%+
2. Portfolio heat exceeds comfortable level
3. Position has large gain/loss affecting allocation
4. Market conditions change requiring adjustment
```

## Sector-Specific Options Considerations

### High IV Sectors

**Energy, Biotech, Small Caps**:
- Higher premiums available
- Greater assignment risk
- More gap risk
- Smaller position sizes recommended

### Low IV Sectors

**Utilities, Consumer Staples**:
- Lower premiums
- More stable positions
- Can use larger position sizes
- Good for conservative strategies

### Interest Rate Sensitive Sectors

**Financials, Real Estate, Utilities**:
- Monitor Fed announcements
- Adjust exposure before FOMC
- Position accordingly for rate outlook

## Common Diversification Mistakes

### Mistake 1: False Diversification

```
Thinking you're diversified:
AAPL + GOOGL + AMZN + META + NVDA

Reality: 100% megacap tech exposure
```

### Mistake 2: Over-Diversification

```
20 positions across 10 sectors
Problem: Too many to manage, diluted returns
Solution: 8-12 positions, focused attention
```

### Mistake 3: Ignoring Sector Correlation

```
During crises, sectors correlate more
Your "diversified" portfolio drops together
Solution: Include truly uncorrelated assets (bonds, gold)
```

### Mistake 4: Chasing Hot Sectors

```
Rotating into whatever did well last month
Usually buy high, sell low
Solution: Systematic rules, not emotions
```

## Diversification Quick Reference

### The 80/20 Approach

```
80% of portfolio: Diversified core
- Index ETFs
- Blue chip stocks across 4+ sectors

20% of portfolio: Concentrated bets
- High conviction plays
- May violate normal sector rules
- Clear thesis required
```

### Emergency Concentration Check

```
If any of these are true, you're too concentrated:
□ Any stock > 10% of portfolio
□ Any sector > 30% of portfolio
□ Tech + Comm Services > 40%
□ Less than 4 sectors represented
□ Average position correlation > 0.7
```

## Conclusion

Sector diversification isn't about maximizing returns in good times - it's about surviving bad times. The trader who stays in the game longest wins.

**Key principles**:
1. No single sector > 25% of portfolio
2. Spread across at least 4 sectors
3. Monitor correlations, not just sector labels
4. Use index exposure as diversification foundation
5. Rebalance regularly, but not too frequently

**Remember**: The goal isn't to own the best-performing sector - it's to never be destroyed by the worst-performing one.
