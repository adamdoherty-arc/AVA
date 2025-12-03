# Correlation Risk Management

## What is Correlation Risk?

Correlation risk occurs when multiple positions in your portfolio move together more than expected, causing amplified gains or losses. During market stress, correlations typically increase, making diversification less effective precisely when it's needed most.

## Understanding Correlation

### Correlation Coefficient Basics

```
Correlation ranges from -1 to +1:

+1.0: Perfect positive correlation (move together)
+0.7 to +0.9: Strong positive correlation
+0.3 to +0.7: Moderate positive correlation
-0.3 to +0.3: Low/no correlation
-0.7 to -0.3: Moderate negative correlation
-0.9 to -0.7: Strong negative correlation
-1.0: Perfect negative correlation (move opposite)
```

### Why Correlation Matters

```
Example: Two positions, each with 1% risk

Low correlation (0.3):
- Combined risk: ~1.1% (diversification benefit)
- Losses unlikely to occur simultaneously

High correlation (0.9):
- Combined risk: ~1.9% (almost additive)
- Losses likely to compound
- Diversification benefit lost
```

## Common Correlation Traps

### Trap 1: Same-Sector "Diversification"

```
False diversification:
- AAPL + MSFT + NVDA + AMD
- All technology
- Correlation: 0.70-0.85
- When tech sells off, all lose

Real risk:
4 positions × 2% each = 8% nominal risk
Correlation-adjusted risk ≈ 6-7%
```

### Trap 2: Market-Cap Correlation

```
Large-cap stocks correlate heavily:
- AAPL, MSFT, AMZN, GOOGL, META
- All mega-cap, index-driven
- Institutional flows affect all similarly
- Correlation: 0.65-0.80
```

### Trap 3: Factor Correlation

```
Hidden correlations by factor:

Growth stocks: High correlation with each other
Value stocks: High correlation with each other
High-beta: All respond to market moves similarly
Dividend payers: Interest rate sensitivity links them
```

### Trap 4: Crisis Correlation

```
Normal times: Cross-sector correlation ~0.50
Crisis times: Cross-sector correlation ~0.85

Your "diversified" portfolio becomes highly correlated
exactly when you need diversification most.
```

## Measuring Portfolio Correlation

### Pairwise Correlation Matrix

```
Example matrix:
        AAPL   MSFT   JPM    XOM    JNJ
AAPL    1.00   0.75   0.45   0.20   0.35
MSFT    0.75   1.00   0.50   0.25   0.40
JPM     0.45   0.50   1.00   0.35   0.30
XOM     0.20   0.25   0.35   1.00   0.25
JNJ     0.35   0.40   0.30   0.25   1.00

Analysis:
- AAPL/MSFT highly correlated (tech)
- XOM least correlated with others
- JNJ moderately uncorrelated
```

### Average Portfolio Correlation

```
Calculation:
Sum of all pairwise correlations / Number of pairs

From matrix above:
Pairs: 10
Sum: 0.75+0.45+0.20+0.35+0.50+0.25+0.40+0.35+0.30+0.25 = 3.80
Average: 0.38

Target: < 0.50 average correlation
Warning: > 0.60 average correlation
Danger: > 0.75 average correlation
```

### Beta-Weighted Correlation

```
Portfolio Beta = Sum of (Position Size × Position Beta)

Example:
Position 1: 25% weight, 1.2 beta = 0.30
Position 2: 25% weight, 1.0 beta = 0.25
Position 3: 25% weight, 0.8 beta = 0.20
Position 4: 25% weight, 0.5 beta = 0.125

Portfolio Beta: 0.875

If market drops 10%:
Expected portfolio drop: ~8.75%
```

## Managing Correlation Risk

### Strategy 1: Sector Limits

```
Maximum sector exposure:
- Single sector: 25% of portfolio
- Two related sectors: 35%
- "Tech-adjacent" combined: 40%
  (Tech + Communication Services + Consumer Discretionary/AMZN)
```

### Strategy 2: Factor Diversification

```
Mix different factor exposures:
- Growth + Value
- Large cap + Small cap
- High beta + Low beta
- Momentum + Quality

Each factor behaves differently in various conditions
```

### Strategy 3: Correlation Budgets

```
Allocate correlation similar to risk:

High correlation pairs allowed: 2-3
Medium correlation pairs: 4-6
Low/no correlation pairs: Unlimited

Track and manage like position sizing
```

### Strategy 4: True Diversifiers

```
Add genuinely uncorrelated assets:
- Gold (negative correlation to dollar)
- Bonds (negative correlation in crisis)
- Utilities (defensive, different drivers)
- International (different economic cycles)
```

## Correlation-Adjusted Position Sizing

### The Basic Approach

```
Standard position size: Based on individual risk
Correlation adjustment: Reduce size for correlated positions

Formula:
Adjusted Size = Standard Size × (1 - Correlation Factor)

Example:
Standard size: $5,000
Adding position correlated at 0.7 with existing position
Correlation factor: 0.7 × 0.3 = 0.21
Adjusted size: $5,000 × (1 - 0.21) = $3,950
```

### Correlation Penalty Table

| Average Correlation | Size Reduction |
|--------------------|----------------|
| 0.30 or less | 0% (no adjustment) |
| 0.30 - 0.50 | 10-15% |
| 0.50 - 0.70 | 15-25% |
| 0.70 - 0.85 | 25-40% |
| 0.85+ | 40-50% or avoid |

### Portfolio-Wide Adjustment

```
Step 1: Calculate average portfolio correlation
Step 2: Determine overall adjustment factor
Step 3: Apply to total portfolio heat limit

Example:
Normal portfolio heat limit: 8%
Average correlation: 0.55
Adjustment: 20%
Adjusted heat limit: 8% × 0.80 = 6.4%
```

## Options-Specific Correlation Issues

### Implied Correlation

```
Options prices imply correlation:
- Index options vs component options
- During fear, implied correlation spikes
- Creates trading opportunities

High implied correlation = components expected to move together
Low implied correlation = more dispersed moves expected
```

### Volatility Correlation

```
Most stocks correlate with VIX:
- VIX up → stocks down (negative correlation)
- During stress, all volatility rises together
- Short volatility positions correlate highly

Risk: Multiple short vol positions amplify losses
```

### Delta Correlation

```
Portfolio delta correlation matters:
- Multiple positive delta positions = all lose when market drops
- Mix positive and negative delta for balance
- Monitor net portfolio delta as correlation measure
```

## Building a Low-Correlation Portfolio

### Core Holdings Selection

```
Choose positions that respond to different drivers:

Interest rates: Financials (+), Utilities (-)
Oil prices: Energy (+), Airlines (-)
Dollar strength: Exporters (-), Domestic (+)
Consumer spending: Discretionary (+), Staples (neutral)
```

### Ideal Portfolio Mix

```
Truly diversified options portfolio:

Index exposure (30%): SPY CSPs
- Market-level returns
- Diversified by definition

Tech (15%): AAPL or MSFT
- Growth exposure
- High quality

Financials (15%): JPM or BAC
- Interest rate play
- Different sector

Healthcare (15%): UNH or JNJ
- Defensive
- Different drivers

Energy (10%): XOM or CVX
- Commodity correlation
- Inflation hedge

Utilities (10%): NEE or SO
- True defensive
- Negative stock correlation

Cash (5%): Reserve
- Zero correlation
- Optionality
```

### Correlation Monitoring Dashboard

```
Weekly review metrics:

1. Pairwise correlations (heatmap)
2. Average portfolio correlation
3. Portfolio beta to SPY
4. Net portfolio delta
5. Sector concentration
6. Factor exposures

Action triggers:
- Average correlation > 0.60: Review positions
- Portfolio beta > 1.2: Reduce risk
- Any sector > 30%: Rebalance
```

## Crisis Correlation Management

### Preparing for Correlation Spikes

```
Before crisis:
- Maintain lower normal correlation
- Keep diversifiers in portfolio
- Have hedges in place
- Hold more cash

During crisis:
- Accept correlations will spike
- Don't panic about portfolio correlation
- Focus on absolute risk management
- Hedges should be working
```

### Hedging Correlation Risk

```
Strategies:

1. VIX calls: Profit from correlation spike
2. SPY puts: Direct market hedge
3. Dispersion trades: Short correlation directly
4. Sector puts: Hedge concentrated exposure
```

### Post-Crisis Opportunity

```
After correlation spike:
- Correlations normalize
- Relative value opportunities appear
- Add uncorrelated positions
- Capture mean reversion
```

## Correlation Risk Mistakes

### Mistake 1: Ignoring Hidden Correlation

```
AMZN seems like retail
But it's really cloud computing (AWS)
Correlates more with tech than retail
Check actual correlation, not sector label
```

### Mistake 2: Over-Relying on Historical Correlation

```
Problem: Past correlation ≠ future correlation
Especially during regime changes
Relationships can break down

Solution: Monitor for correlation changes
Use rolling correlation windows
Stay alert to structural changes
```

### Mistake 3: Perfect Negative Correlation Expectation

```
Myth: Bonds always protect against stock losses
Reality: 2022 - both stocks AND bonds fell
Correlations are dynamic

Solution: Don't assume any hedge is permanent
Monitor correlation between hedges and portfolio
```

### Mistake 4: Correlation as Only Metric

```
Low correlation doesn't mean low risk
Two uncorrelated positions can both lose 50%
Correlation reduces variance, not expected loss

Solution: Manage both individual risk AND correlation
Position sizing still matters for each position
```

## Practical Correlation Checklist

### Before Adding New Position

```
□ What's the correlation with existing positions?
□ Does it increase average portfolio correlation?
□ Is it in an already-heavy sector?
□ Does it share risk factors with current holdings?
□ Should position size be adjusted for correlation?
```

### Weekly Review

```
□ Calculate current average correlation
□ Identify highest correlated pair
□ Check sector concentrations
□ Review portfolio beta
□ Note any correlation changes from prior week
```

### Monthly Assessment

```
□ Full correlation matrix update
□ Factor exposure analysis
□ Correlation trend over past 3 months
□ Rebalancing needs
□ Hedge effectiveness review
```

## Conclusion

Correlation risk is the hidden multiplier that can make your portfolio much riskier than individual position analysis suggests. During normal times, correlation is manageable. During crises, it becomes critical.

**Key Principles**:

1. **Measure it**: You can't manage what you don't measure
2. **Limit it**: Set maximum correlation thresholds
3. **Adjust for it**: Reduce position sizes for correlated holdings
4. **Monitor it**: Correlations change over time
5. **Prepare for it**: Correlations spike during stress

**The goal**: A portfolio where not everything goes wrong at once.

**Remember**: True diversification means holding things that don't just have different names, but actually behave differently. A portfolio of 10 tech stocks is not diversified - it's concentrated with extra steps.
