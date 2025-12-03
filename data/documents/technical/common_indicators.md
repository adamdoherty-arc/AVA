# Common Technical Indicators for Options Trading

## Overview

Technical indicators are mathematical calculations based on price, volume, or open interest. They help traders identify trends, momentum, and potential reversal points. This guide covers the most useful indicators for options trading.

## Moving Averages

### Simple Moving Average (SMA)

```
Calculation:
SMA = Sum of closing prices over N periods / N

Common periods:
- 20 SMA: Short-term trend
- 50 SMA: Intermediate trend
- 200 SMA: Long-term trend
```

**Trading Applications**:
```
Trend identification:
- Price > 200 SMA = Bullish bias
- Price < 200 SMA = Bearish bias

Support/Resistance:
- MAs act as dynamic support/resistance
- Multiple bounces confirm significance

Crossovers:
- Golden Cross: 50 SMA crosses above 200 SMA (bullish)
- Death Cross: 50 SMA crosses below 200 SMA (bearish)
```

### Exponential Moving Average (EMA)

```
Difference from SMA:
- Weights recent prices more heavily
- Reacts faster to price changes
- More useful for short-term trading

Common periods:
- 9 EMA: Very short-term
- 20 EMA: Short-term trend
- 50 EMA: Intermediate trend
```

**Options Application**:
```
CSP strike selection:
- Strong stocks trade above 50 EMA
- Sell puts at or below 50 EMA as support

Entry timing:
- Pullback to rising 20 EMA = entry point
- Break below 50 EMA = caution
```

## Relative Strength Index (RSI)

### Definition

```
Calculation:
RSI = 100 - (100 / (1 + RS))
RS = Average gain / Average loss over N periods

Standard period: 14 days
Range: 0 to 100
```

### Interpretation

```
Traditional levels:
RSI > 70: Overbought (potential selling)
RSI < 30: Oversold (potential buying)

Refinements:
RSI > 80: Extremely overbought
RSI < 20: Extremely oversold

In strong trends:
- RSI can stay overbought/oversold for extended periods
- Don't fight the trend based solely on RSI
```

### RSI for Options

```
For selling puts (bullish strategies):
- Wait for RSI < 40 (pullback in uptrend)
- Avoid RSI > 75 (too extended)

For selling calls (bearish strategies):
- Wait for RSI > 60 (rally in downtrend)
- Avoid RSI < 25 (too beaten down)

Divergences:
- Price makes new high, RSI makes lower high = bearish
- Price makes new low, RSI makes higher low = bullish
```

## MACD (Moving Average Convergence Divergence)

### Components

```
MACD Line = 12 EMA - 26 EMA
Signal Line = 9 EMA of MACD Line
Histogram = MACD Line - Signal Line
```

### Trading Signals

```
Crossover signals:
- MACD crosses above Signal = Bullish
- MACD crosses below Signal = Bearish

Zero line:
- MACD above zero = Bullish momentum
- MACD below zero = Bearish momentum

Histogram:
- Growing histogram = Increasing momentum
- Shrinking histogram = Decreasing momentum
```

### MACD for Options

```
Entry timing:
- MACD bullish crossover near zero = Good long entry
- MACD bearish crossover near zero = Good short entry

Trend confirmation:
- MACD above zero supports bullish strategies
- MACD below zero supports bearish strategies

Warnings:
- Histogram shrinking warns of momentum change
- Divergences signal potential reversals
```

## Bollinger Bands

### Structure

```
Middle Band = 20-period SMA
Upper Band = Middle Band + (2 × Standard Deviation)
Lower Band = Middle Band - (2 × Standard Deviation)

Interpretation:
- ~95% of price action within bands
- Bands widen during volatility
- Bands contract during consolidation
```

### Trading Applications

```
Mean reversion:
- Price at upper band = Extended, potential pullback
- Price at lower band = Extended, potential bounce

Breakouts:
- Price breaks above band with volume = Bullish continuation
- Price breaks below band with volume = Bearish continuation

Squeeze (bands narrow):
- Low volatility precedes high volatility
- Prepare for breakout
- Don't predict direction, wait for break
```

### Options Application

```
For option selling:
- Sell when bands are wide (high IV)
- Avoid when bands are extremely narrow

For option buying:
- Buy when bands are narrow (low IV)
- Breakout direction determines strategy

Strike selection:
- Bands provide price targets
- Sell puts below lower band
- Sell calls above upper band
```

## Average True Range (ATR)

### Calculation

```
True Range = Greatest of:
1. Current High - Current Low
2. |Current High - Previous Close|
3. |Current Low - Previous Close|

ATR = Average of True Range over N periods
Standard period: 14 days
```

### Applications

```
Volatility measurement:
- High ATR = High volatility
- Low ATR = Low volatility

Position sizing:
- Risk = ATR × Multiplier
- Larger ATR = Smaller position

Stop placement:
- Stop distance = 1.5-2× ATR below entry
- Accounts for normal price fluctuation
```

### Options Application

```
Strike selection:
- Expected move = Current Price ± (ATR × Days)
- Sell options outside expected move

Premium assessment:
- High ATR stocks = Higher premiums
- May justify wider strikes
```

## Stochastic Oscillator

### Components

```
%K = (Current Close - Lowest Low) / (Highest High - Lowest Low) × 100
%D = 3-period SMA of %K

Standard settings: 14, 3, 3
Range: 0 to 100
```

### Interpretation

```
Overbought/Oversold:
- Above 80 = Overbought
- Below 20 = Oversold

Crossover signals:
- %K crosses above %D below 20 = Buy signal
- %K crosses below %D above 80 = Sell signal
```

### Options Application

```
Entry timing:
- Wait for oversold conditions before bullish entries
- Wait for overbought conditions before bearish entries

Confirmation:
- Use with trend (don't fight primary trend)
- Best in ranging markets
```

## Volume-Based Indicators

### On-Balance Volume (OBV)

```
Calculation:
If close > prior close: OBV + Volume
If close < prior close: OBV - Volume

Use: Confirms price trends
Rising OBV + Rising Price = Healthy trend
Divergence = Warning sign
```

### Volume Weighted Average Price (VWAP)

```
VWAP = Cumulative(Price × Volume) / Cumulative Volume

Use:
- Intraday benchmark
- Institutional execution target
- Dynamic support/resistance

Above VWAP = Bullish intraday
Below VWAP = Bearish intraday
```

## Momentum Indicators

### Rate of Change (ROC)

```
ROC = ((Current Price - Price N periods ago) / Price N periods ago) × 100

Interpretation:
- ROC > 0: Positive momentum
- ROC < 0: Negative momentum
- Extreme values may indicate overextension
```

### Momentum

```
Momentum = Current Price - Price N periods ago

Simple but effective
Measures speed of price change
Zero line crossovers significant
```

## Combining Indicators

### The Wrong Way

```
Using multiple indicators that measure the same thing:
- RSI + Stochastic + CCI (all momentum)
- Redundant information
- False confidence from "confirmation"
```

### The Right Way

```
Combine different types of indicators:

Trend: Moving Averages or MACD
Momentum: RSI or Stochastic
Volatility: Bollinger Bands or ATR
Volume: OBV or VWAP

Example setup:
1. 50/200 SMA for trend direction
2. RSI for overbought/oversold
3. ATR for stop placement
4. Volume for confirmation
```

## Indicator Settings

### Default vs Custom

```
General rule: Start with defaults
Defaults are popular, creating self-fulfilling prophecy

Adjustments:
- Shorter periods: More signals, more noise
- Longer periods: Fewer signals, more reliable
- Match time frame to your trading style
```

### Time Frame Considerations

```
Day trading: Shorter periods (5, 10, 20)
Swing trading: Standard periods (14, 20, 50)
Position trading: Longer periods (50, 100, 200)

Rule: Indicator period should match holding period
```

## Practical Application Checklist

### Before Entry

```
□ What does the trend say? (MAs, MACD)
□ Is momentum supporting? (RSI, Stochastic)
□ Is volatility favorable? (Bollinger, ATR)
□ Does volume confirm? (OBV, VWAP)
□ Are indicators aligned?
```

### Indicator Signals Matrix

| Indicator | Bullish Signal | Bearish Signal |
|-----------|---------------|----------------|
| 50/200 MA | Price above both | Price below both |
| RSI | Below 40, rising | Above 60, falling |
| MACD | Bullish crossover | Bearish crossover |
| Bollinger | Bounce off lower | Rejection at upper |
| Stochastic | %K crosses %D <20 | %K crosses %D >80 |

## Common Indicator Mistakes

### Mistake 1: Over-Reliance

```
Wrong: Trading solely on indicators
Right: Using indicators to support analysis

Indicators lag price
They don't predict, they describe
```

### Mistake 2: Too Many Indicators

```
Wrong: 10 indicators on chart
Right: 3-4 complementary indicators

More indicators = more confusion
Simplicity is key
```

### Mistake 3: Ignoring Context

```
Wrong: RSI < 30 = automatic buy
Right: RSI < 30 in uptrend = buying opportunity

Context (trend, sector, market) matters
Same reading means different things
```

### Mistake 4: Curve Fitting

```
Wrong: Adjusting settings to match past
Right: Using robust, standard settings

Optimized indicators fail forward
Simplicity works better long-term
```

## Conclusion

Technical indicators are tools, not crystal balls. They work best when:

1. **Combined thoughtfully** - Different types, not redundant
2. **Used with context** - Consider trend, market conditions
3. **Confirmed by price action** - Indicators support, not replace
4. **Applied consistently** - Same tools, same interpretation
5. **Kept simple** - 3-4 indicators maximum

**The best indicator is the one you understand deeply and apply consistently.**

For options trading specifically:
- Use trend indicators to determine strategy type
- Use momentum indicators for entry timing
- Use volatility indicators for strike selection
- Use volume indicators for confirmation
