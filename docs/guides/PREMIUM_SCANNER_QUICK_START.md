# Premium Scanner - Quick Start Guide

## 🚀 Getting Started in 60 Seconds

### Step 1: Open Premium Scanner
Navigate to the Premium Scanner page in your Magnus dashboard.

### Step 2: Sync Data
Click "🔄 Sync" button for the scanner you want to use:
- **7-Day Scanner**: Weekly options for fast premium collection
- **30-Day Scanner**: Monthly options for higher premium per contract

### Step 3: Adjust Filters (Optional)
Open sidebar → Adjust filters → Click "🔍 Apply Filters"

### Step 4: Review Opportunities
Browse the table, click "📊 Chart" links to view on TradingView

### Step 5: Export (Optional)
Click "📥 CSV" or "📊 Excel" to download results

**Done!** You're scanning premium opportunities like a pro.

---

## 🎯 Filter Guide

### Basic Filters (Start Here)

#### Max Stock Price
```
What: Maximum stock price to include
Why: Control capital requirement
Example: $100 = need ~$10k per trade
Recommended: $50-$200
```

#### Delta Range
```
What: Option delta (probability of ITM)
Why: Control risk/reward ratio
Example: -0.30 = ~70% probability of profit
Recommended:
  - Conservative: -0.30 to -0.20 (70-80% PoP)
  - Aggressive: -0.40 to -0.30 (60-70% PoP)
```

#### Min Premium
```
What: Minimum premium per contract
Why: Ensure minimum profit worthwhile
Example: $50 minimum profit
Recommended: $25-$100
```

#### Min Annualized
```
What: Minimum annualized return %
Why: Target specific return goals
Example: 40% annual return
Recommended: 30-60%
```

#### Min Volume
```
What: Minimum option volume
Why: Ensure liquidity (easy to trade)
Example: 100 contracts/day
Recommended: 50-200
```

### Advanced Filters (Fine-Tune)

#### Sectors
```
What: Filter by specific industries
Why: Focus on sectors you know
Example: Tech, Finance, Healthcare
Recommended: Pick 2-3 favorite sectors
```

#### IV Range
```
What: Implied volatility range
Why: Control option pricing
Example: 40-80% IV
Recommended:
  - Low volatility: 20-40%
  - Medium volatility: 40-80%
  - High volatility: 80%+
```

#### Min Open Interest
```
What: Minimum outstanding contracts
Why: Additional liquidity check
Example: 100+ open contracts
Recommended: 50-200
```

#### Max Bid-Ask Spread
```
What: Maximum spread between bid/ask
Why: Ensure tight pricing
Example: $0.50 maximum spread
Recommended: $0.25-$1.00
```

---

## 📊 Understanding the Metrics

### Table Columns Explained

| Column | Meaning | Good Values |
|--------|---------|-------------|
| **Symbol** | Stock ticker | - |
| **📊 Chart** | TradingView link | Click to view chart |
| **Company** | Company name | - |
| **Stock $** | Current stock price | Depends on capital |
| **Strike** | Put strike price | ~3-7% below stock |
| **Premium** | Premium per contract | Higher is better |
| **DTE** | Days to expiration | 6-8 or 28-32 |
| **Weekly%** | % return for week | 1-3% is good |
| **Monthly%** | % return for month | 2-5% is good |
| **Annual%** | Annualized return | 40-80% is great |
| **Delta** | Prob. ITM (negative) | -0.30 to -0.20 |
| **IV** | Implied volatility % | 40-80% sweet spot |
| **Volume** | Daily volume | 100+ is good |
| **Open Int** | Outstanding contracts | 100+ is good |
| **Sector** | Industry | Use for filtering |

### Summary Metrics

```
📈 7-Day Summary

Opportunities: 150        ← How many matches found
Avg Weekly Return: 2.5%   ← Average premium for week
Avg Annualized: 65%       ← If repeated 52 weeks
Best Weekly: 4.2%         ← Highest single opportunity
```

---

## 🎓 Strategy Guide

### Conservative Strategy (Beginners)

**Goal:** High probability of profit, consistent income

**Recommended Settings:**
```
Max Stock Price: $100
Delta Range: -0.30 to -0.20 (70-80% PoP)
Min Premium: $25
Min Annual: 30%
Min Volume: 100
Min Open Interest: 100
Max Bid-Ask Spread: $0.50
```

**Expected Results:**
- Win rate: 70-80%
- Weekly return: 1-2%
- Monthly return: 3-5%
- Annual return: 40-60%

**Best For:**
- Small accounts ($5k-$25k)
- Risk-averse traders
- Consistent income seekers

---

### Balanced Strategy (Intermediate)

**Goal:** Good probability with decent premium

**Recommended Settings:**
```
Max Stock Price: $200
Delta Range: -0.35 to -0.25 (65-75% PoP)
Min Premium: $50
Min Annual: 40%
Min Volume: 75
Sectors: Tech, Finance, Healthcare
IV Range: 40-80%
```

**Expected Results:**
- Win rate: 65-75%
- Weekly return: 1.5-2.5%
- Monthly return: 4-6%
- Annual return: 50-75%

**Best For:**
- Medium accounts ($25k-$100k)
- Experienced traders
- Growth + income balance

---

### Aggressive Strategy (Advanced)

**Goal:** Higher premiums, accept more risk

**Recommended Settings:**
```
Max Stock Price: $300
Delta Range: -0.40 to -0.30 (60-70% PoP)
Min Premium: $100
Min Annual: 50%
Min Volume: 50
Sectors: Tech, Consumer, Energy
IV Range: 60-100%
```

**Expected Results:**
- Win rate: 60-70%
- Weekly return: 2-3%
- Monthly return: 5-8%
- Annual return: 60-100%+

**Best For:**
- Large accounts ($100k+)
- Aggressive traders
- Higher risk tolerance

---

## 💡 Pro Tips

### 1. Weekly vs Monthly
```
7-Day Scanner:
✅ Higher annualized returns
✅ Faster theta decay
✅ More active management
❌ More frequent rolling
❌ More commissions

30-Day Scanner:
✅ Higher premium per trade
✅ Less frequent management
✅ Fewer commissions
❌ Lower annualized returns
❌ More capital tied up
```

### 2. Best Times to Scan
```
📅 Weekly Options:
- Monday: See new weekly chain
- Thursday PM: Catch last-minute opportunities
- Avoid Friday (1 DTE too risky)

📅 Monthly Options:
- First week of month: New monthly chain
- Mid-month: Good mix of DTE available
- Last week: Avoid (too close to expiration)
```

### 3. Sector Rotation
```
Don't scan the same sectors every time!

Rotate through:
Week 1: Tech, Finance
Week 2: Healthcare, Consumer
Week 3: Energy, Industrials
Week 4: Communications, Utilities

This diversifies your portfolio.
```

### 4. IV Sweet Spots
```
❌ Too Low IV (< 30%):
- Premiums too small
- Not worth the capital

✅ Sweet Spot IV (40-80%):
- Good premiums
- Manageable risk
- Best risk/reward

❌ Too High IV (> 100%):
- High premiums BUT
- Very risky
- High chance of assignment
```

### 5. Volume & Open Interest
```
Always check BOTH:

Example 1:
Volume: 500  ✅
Open Int: 50 ❌  ← Red flag (might be hard to close)

Example 2:
Volume: 50   ❌
Open Int: 500 ❌  ← Red flag (low daily activity)

Example 3:
Volume: 200  ✅
Open Int: 300 ✅  ← Perfect (active and liquid)
```

---

## 📈 Analytics Tab Guide

### Premium Heatmap

**What it shows:**
Average premium % by sector and DTE

**How to use:**
1. Look for green areas (high premiums)
2. Compare sectors (which pays best?)
3. Compare DTEs (weekly vs monthly)

**Example Insights:**
```
"Tech sector pays 2.8% on 7-day but only 2.0% on 30-day"
→ Focus on tech weeklies

"Finance pays better on 30-day options"
→ Use monthlies for finance sector
```

### Risk vs Reward Scatter

**What it shows:**
Delta (risk) vs Annual Return (reward) for all opportunities

**How to use:**
1. Look for top-left quadrant (low risk, high reward)
2. Identify outliers (unusual risk/reward)
3. See sector clustering

**Example Insights:**
```
"AAPL is low risk (-0.25 delta) with high return (70%)"
→ Great opportunity

"TSLA is high risk (-0.45 delta) with medium return (50%)"
→ Skip this one
```

### Distribution Analysis

**What it shows:**
- How premiums are distributed
- How returns vary by sector

**How to use:**
1. Check if distribution is normal or skewed
2. Identify outliers
3. Compare sector boxes

---

## 🔄 Sync Guide

### When to Sync

```
✅ Sync when:
- First time using scanner
- Market just opened
- Want latest prices
- Before making trades
- Once per day minimum

❌ Don't need to sync:
- Multiple times per hour
- During market close
- When nothing changed
```

### Sync Frequency Recommendations

```
7-Day Scanner:
🔄 Monday: Sync for new weekly chain
🔄 Thursday: Sync for updated prices
📊 Rest of week: Use cached data

30-Day Scanner:
🔄 First Monday: Sync for new monthly chain
🔄 Mid-month: Sync for updated prices
📊 Rest of month: Use cached data
```

### Understanding Sync Results

```
✅ Synced 148/150 symbols successfully!
⚠️ 2/150 symbols failed to sync

Normal! Some symbols may:
- Be delisted
- Have no options
- Be experiencing data issues

If > 10% fail, might be API issue.
```

---

## 📥 Export Guide

### When to Export

**CSV Export - Best for:**
- Further analysis in Excel/Google Sheets
- Importing to other tools
- Sharing with others
- Lightweight file size

**Excel Export - Best for:**
- Professional reports
- Presentations
- Client deliverables
- Already formatted

### Export Tips

```
1. Filter first, then export
   (Only export what you need)

2. Export regularly
   (Track historical opportunities)

3. Compare exports
   (See how opportunities change)

4. Archive by date
   (Files include timestamp)
```

---

## ❓ FAQ

### Q: How often should I scan?
**A:** Once daily is usually enough. Market doesn't change dramatically hour-to-hour.

### Q: Why are some symbols missing?
**A:** Not all stocks have weekly options. Some only have monthly options available.

### Q: What's a good delta for beginners?
**A:** Start with -0.25 to -0.20 (75-80% probability of profit).

### Q: How many trades should I make?
**A:** Start with 3-5 positions. Diversify across sectors and DTEs.

### Q: What if premium looks too good?
**A:** Check IV, earnings date, and news. High premium often = high risk.

### Q: Should I use 7-day or 30-day?
**A:** Both! 50/50 split provides balance of frequency and premium size.

### Q: How do I avoid earnings?
**A:** Check earnings calendar. Avoid options expiring within 7 days of earnings.

### Q: What about assignment risk?
**A:** -0.30 delta = 30% chance of assignment. Manage by rolling or closing early.

### Q: When should I take profit?
**A:** General rule: Close at 50% profit (e.g., sold for $1.00, buy back at $0.50).

### Q: What's the minimum account size?
**A:** $5,000 minimum. More is better for proper diversification.

---

## 🎯 Quick Win Checklist

### Day 1: Setup
- [ ] Open Premium Scanner
- [ ] Sync 7-day scanner
- [ ] Sync 30-day scanner
- [ ] Try different filters
- [ ] Export a sample to CSV

### Day 2: Learn
- [ ] Review Analytics tab
- [ ] Compare sectors in heatmap
- [ ] Check risk/reward scatter
- [ ] Understand distributions
- [ ] Read this guide fully

### Day 3: Strategy
- [ ] Pick your strategy (conservative/balanced/aggressive)
- [ ] Set your filters accordingly
- [ ] Find 10 opportunities
- [ ] Export to Excel
- [ ] Review each one carefully

### Day 4: Execute
- [ ] Scan for best 5 opportunities
- [ ] Check charts on TradingView
- [ ] Verify earnings dates
- [ ] Check news/catalysts
- [ ] Execute trades

### Ongoing
- [ ] Scan Monday & Thursday
- [ ] Export weekly for tracking
- [ ] Adjust filters as needed
- [ ] Review Analytics monthly
- [ ] Refine strategy quarterly

---

## 🚀 Next Steps

1. **Start Small**: Begin with 1-2 trades using conservative settings
2. **Track Results**: Export and track every trade
3. **Analyze**: Use Analytics tab to improve strategy
4. **Scale Up**: Once comfortable, increase position size
5. **Diversify**: Mix 7-day and 30-day, multiple sectors
6. **Optimize**: Refine filters based on results

---

## 📞 Support

**Need Help?**
- Check "❓ Quick Help" in sidebar
- Review this guide
- Check error messages for guidance

**Found a Bug?**
- Note the error message
- Check what filters were active
- Try refreshing the page
- Clear cache if needed

---

**Happy Scanning!** 💎

Remember: Consistent, systematic scanning beats random opportunity-hunting every time. Use this tool daily, stick to your strategy, and watch your premium income grow!
