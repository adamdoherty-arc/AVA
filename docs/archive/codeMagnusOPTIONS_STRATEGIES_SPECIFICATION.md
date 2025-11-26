# Options Strategies Specification

## Overview

The Comprehensive Strategy Analyzer evaluates **10 different option strategies** for any stock and ranks them based on current market conditions, volatility, trend, and risk profile.

---

## The 10 Strategies

### 1. Cash-Secured Put (CSP)
**Type**: Credit strategy, neutral to bullish
**Description**: Sell put option, hold cash to buy stock if assigned
**Legs**:
- SELL 1 PUT at strike K (typically 5% OTM)

**Profit/Loss**:
- Max Profit: Premium received
- Max Loss: (Strike - Premium) × 100
- Breakeven: Strike - Premium

**Best When**:
- Bullish to neutral on stock
- Want to own stock at lower price  
- Low to moderate volatility

**Capital Required**: Strike × 100

---

### 2. Iron Condor
**Type**: Credit strategy, neutral
**Description**: Sell OTM put spread + sell OTM call spread
**Legs**:
- SELL 1 PUT at K1 (5% below price)
- BUY 1 PUT at K2 (10% below price)  
- SELL 1 CALL at K3 (5% above price)
- BUY 1 CALL at K4 (10% above price)

**Profit/Loss**:
- Max Profit: Net credit received
- Max Loss: (Width of widest spread - Net credit) × 100
- Breakeven: Upper breakeven = Short call + Net credit, Lower breakeven = Short put - Net credit

**Best When**:
- Expect stock to stay range-bound
- High implied volatility (sell premium)
- Low actual volatility expected

**Capital Required**: (K1 - K2) × 100

---

### 3. Poor Man's Covered Call (PMCC)
**Type**: Debit spread, bullish
**Description**: Long-term deep ITM call + sell near-term OTM call
**Legs**:
- BUY 1 CALL at K1 (20% ITM, 180+ DTE)
- SELL 1 CALL at K2 (5% OTM, 30-45 DTE)

**Profit/Loss**:
- Max Profit: (K2 - K1 - Net debit) × 100
- Max Loss: Net debit paid
- Breakeven: K1 + Net debit

**Best When**:
- Bullish but can't afford 100 shares
- Want covered call-like returns with less capital
- Moderate volatility

**Capital Required**: Cost of long call - credit from short call

---

### 4. Bull Put Spread
**Type**: Credit strategy, bullish
**Description**: Sell put + buy lower strike put for protection
**Legs**:
- SELL 1 PUT at K1 (3% below price)
- BUY 1 PUT at K2 (7% below price)

**Profit/Loss**:
- Max Profit: Net credit received
- Max Loss: (K1 - K2 - Net credit) × 100
- Breakeven: K1 - Net credit

**Best When**:
- Moderately bullish
- Want defined risk (vs naked put)
- High IV environment

**Capital Required**: (K1 - K2) × 100

---

### 5. Bear Call Spread
**Type**: Credit strategy, bearish
**Description**: Sell call + buy higher strike call for protection
**Legs**:
- SELL 1 CALL at K1 (3% above price)
- BUY 1 CALL at K2 (7% above price)

**Profit/Loss**:
- Max Profit: Net credit received
- Max Loss: (K2 - K1 - Net credit) × 100
- Breakeven: K1 + Net credit

**Best When**:
- Moderately bearish
- Want defined risk
- High IV environment

**Capital Required**: (K2 - K1) × 100

---

### 6. Covered Call
**Type**: Income strategy, neutral to moderately bullish
**Description**: Own 100 shares + sell OTM call
**Legs**:
- OWN 100 shares
- SELL 1 CALL at K (5% above current price)

**Profit/Loss**:
- Max Profit: (K - Stock price) + Premium
- Max Loss: Stock price - Premium (if stock goes to $0)
- Breakeven: Stock cost basis - Premium

**Best When**:
- Already own stock
- Stock in consolidation/sideways
- Want extra income

**Capital Required**: Current stock price × 100

---

### 7. Calendar Spread (Time Spread)
**Type**: Debit strategy, neutral
**Description**: Buy long-dated option + sell near-dated option (same strike)
**Legs**:
- BUY 1 PUT/CALL at K (45-90 DTE)
- SELL 1 PUT/CALL at K (7-21 DTE, same strike)

**Profit/Loss**:
- Max Profit: Varies (when short expires worthless and long retains value)
- Max Loss: Net debit paid
- Breakeven: Complex (depends on volatility changes)

**Best When**:
- Expect stock to stay near strike
- Earnings or volatility event approaching
- Want to profit from time decay differential

**Capital Required**: Net debit paid

---

### 8. Diagonal Spread
**Type**: Debit strategy, bullish/bearish depending on calls/puts
**Description**: Like calendar spread but different strikes
**Legs**:
- BUY 1 CALL at K1 (45-90 DTE, ATM or slightly ITM)
- SELL 1 CALL at K2 (7-21 DTE, OTM)

**Profit/Loss**:
- Max Profit: (K2 - K1 - Net debit) × 100
- Max Loss: Net debit paid  
- Breakeven: K1 + Net debit

**Best When**:
- Moderately bullish (call diagonal) or bearish (put diagonal)
- Want some directionality + time decay
- Lower cost than straight long call

**Capital Required**: Net debit paid

---

### 9. Long Straddle
**Type**: Debit strategy, high volatility play
**Description**: Buy ATM call + buy ATM put (expect big move either direction)
**Legs**:
- BUY 1 CALL at K (ATM)
- BUY 1 PUT at K (ATM)

**Profit/Loss**:
- Max Profit: Unlimited (if stock moves significantly)
- Max Loss: Total premium paid
- Breakeven: K ± Total premium

**Best When**:
- Expect large price move but unsure of direction
- Before earnings or major catalyst
- IV is low (buying cheap options)

**Capital Required**: Total premium for both options

---

### 10. Short Strangle
**Type**: Credit strategy, neutral
**Description**: Sell OTM call + sell OTM put (profit if stock stays in range)
**Legs**:
- SELL 1 PUT at K1 (10% below price)
- SELL 1 CALL at K2 (10% above price)

**Profit/Loss**:
- Max Profit: Total premium received
- Max Loss: Unlimited (theoretically)
- Breakeven: K1 - Total premium, K2 + Total premium

**Best When**:
- Expect stock to stay range-bound
- High IV (sell expensive premium)
- Want wider range than iron condor

**Capital Required**: Margin requirement (varies by broker)

---

## Strategy Selection Criteria

### Market Environment Factors

1. **Volatility Regime**
   - Low IV (<20%): Favor debit strategies (PMCC, Long Straddle, Calendar)
   - Moderate IV (20-35%): Neutral strategies (CSP, Bull Put Spread)
   - High IV (>35%): Credit strategies (Iron Condor, Short Strangle)

2. **Trend**
   - Strong Bullish: CSP, Bull Put Spread, PMCC, Covered Call
   - Weak Bullish: Bull Put Spread, Calendar Spread
   - Neutral: Iron Condor, Short Strangle, Calendar Spread
   - Weak Bearish: Bear Call Spread
   - Strong Bearish: Avoid or use Bear Call Spread

3. **Time to Event** (earnings, etc.)
   - Before event: Long Straddle (if IV low), Calendar Spread
   - After event: CSP, Credit spreads

### Scoring System

Each strategy scored 0-100 based on:
- **Market Fit (40%)**: How well strategy fits current trend/volatility
- **Risk/Reward (30%)**: Profit potential vs risk
- **Win Probability (20%)**: Probability of profit  
- **Capital Efficiency (10%)**: Return on capital

---

## Implementation Notes

The analyzer will:
1. Fetch current stock price, IV, historical volatility
2. Determine market regime (bullish/bearish/neutral, low/high IV)
3. Score all 10 strategies
4. Rank by total score
5. Generate trade execution details for top 3 strategies
6. Provide AI reasoning (optional, via LLM)

