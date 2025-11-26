# Free Market Data Sources for WheelStrategy

## ✅ Completely FREE Options (No API Key Required)

### 1. Yahoo Finance (BEST FREE OPTION)
**Already implemented in `src/api/DataProvider.js`**
- ✅ **Real-time stock prices**
- ✅ **Options chains with Greeks**
- ✅ **Fundamentals data**
- ✅ **No API key needed**
- ✅ **No rate limits** (reasonable use)
- ⚠️ Unofficial API (may change)

**How to use:**
```javascript
const DataProvider = require('./src/api/DataProvider');
const provider = new DataProvider({});  // No API key needed!

// Get stock data
const stock = await provider.getStockData('AAPL');

// Get options chain
const options = await provider.getOptionsChain('AAPL');
```

### 2. yfinance Python Library
If you prefer Python:
```bash
pip install yfinance
```
```python
import yfinance as yf
ticker = yf.Ticker("AAPL")
options = ticker.option_chain('2024-12-20')
```

## 🆓 Free with Registration (API Key Required)

### 3. Alpha Vantage (RECOMMENDED)
- **Free tier:** 5 API calls/minute, 500/day
- **Includes:** Stock data, fundamentals
- **Sign up:** https://www.alphavantage.co/support/#api-key
- ⚠️ Options data requires premium ($49.99/month)

### 4. Polygon.io
- **Free tier:** 5 calls/minute
- **Includes:** End-of-day data
- **Sign up:** https://polygon.io/dashboard/signup
- ⚠️ Real-time data requires paid plan

### 5. IEX Cloud
- **Free tier:** 500,000 messages/month
- **Includes:** Stock data, fundamentals
- **Sign up:** https://iexcloud.io/console/
- ⚠️ No options data in free tier

### 6. Twelvedata
- **Free tier:** 800 API calls/day
- **Includes:** Stock data, technical indicators
- **Sign up:** https://twelvedata.com/apikey

## 📊 TradingView Integration Options

### Option 1: TradingView Webhook Alerts (Limited Free)
- Create alerts on TradingView
- Use webhook to send to your app
- **Free:** 1 alert
- **Pro:** Unlimited alerts ($14.95/month)

### Option 2: TradingView Widgets (FREE)
Add real-time charts to your web interface:
```html
<!-- TradingView Widget -->
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({
    "width": 980,
    "height": 610,
    "symbol": "NASDAQ:AAPL",
    "interval": "D",
    "timezone": "Etc/UTC",
    "theme": "light",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#f1f3f6",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "container_id": "tradingview_chart"
  });
  </script>
</div>
```

## 🔧 GitHub Projects for Free Data

### 1. ranaroussi/yfinance (Python)
```bash
git clone https://github.com/ranaroussi/yfinance
```

### 2. gadicc/node-yahoo-finance2 (Node.js)
```bash
npm install yahoo-finance2
```
```javascript
const yahooFinance = require('yahoo-finance2').default;
const quote = await yahooFinance.quote('AAPL');
const options = await yahooFinance.options('AAPL');
```

### 3. JECSand/yahoofinancials (Python)
```bash
pip install yahoofinancials
```

## 🚀 Quick Start with FREE Data

### Step 1: No API Keys Needed!
The project already works with Yahoo Finance - no configuration required:

```bash
# Just run the app
npm start
```

### Step 2: Test the Free Data Provider
```javascript
const WheelStrategy = require('./src/index');
const DataProvider = require('./src/api/DataProvider');

// Initialize with no API keys
const dataProvider = new DataProvider({});
const app = new WheelStrategy();

// Get real-time data
const appleStock = await dataProvider.getStockData('AAPL');
console.log('Apple Price:', appleStock.currentPrice);

// Get options chain
const options = await dataProvider.getOptionsChain('AAPL');
console.log('Available strikes:', options.puts.map(p => p.strike));
```

## 💰 Cost Comparison

| Provider | Free Tier | Paid | Options Data | Real-time |
|----------|-----------|------|--------------|-----------|
| **Yahoo Finance** | ✅ Unlimited* | N/A | ✅ Yes | ✅ Yes |
| **Alpha Vantage** | 500 calls/day | $49.99/mo | ❌ Paid only | ✅ Yes |
| **IEX Cloud** | 500K msgs/mo | Pay-as-you-go | ❌ No | ✅ Yes |
| **Polygon** | 5 calls/min | $99/mo | ✅ Paid | ❌ Delayed |
| **Finnhub** | 60 calls/min | $39.99/mo | ❌ No | ✅ Yes |
| **Twelvedata** | 800 calls/day | $29/mo | ❌ No | ✅ Yes |

\* Yahoo Finance has no official rate limits but should be used responsibly

## ⚠️ Important Notes

1. **Yahoo Finance** is unofficial but widely used. It's the best free option for options data.

2. **Rate Limiting**: Even free APIs should be used responsibly. The DataProvider class includes rate limiting.

3. **Data Quality**: Free data may have delays or gaps. For production trading, consider paid services.

4. **Terms of Service**: Always check the provider's ToS for commercial use restrictions.

5. **Backup Sources**: The DataProvider supports multiple sources for redundancy.

## 🎯 Recommendation for WheelStrategy

**For Development/Testing:**
- Use Yahoo Finance (already implemented, no setup needed!)

**For Production:**
1. Start with Yahoo Finance (free)
2. Add Alpha Vantage free tier for fundamentals
3. Consider IEX Cloud for reliability
4. Upgrade to paid tiers as needed

The project is **ready to use immediately** with Yahoo Finance - no API keys or costs required!