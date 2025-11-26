# API Keys Reference

## Market Data APIs (Currently Configured)

### Primary Data Sources
1. **Yahoo Finance** - DEFAULT (No key required)
   - Real-time stock prices
   - Options chains with Greeks
   - Company fundamentals
   - **Status:** ✅ Active (Primary source)

2. **Alpha Vantage** 
   - **Key:** ZW1RV8BMTZMUUZVJ
   - **Tier:** Free (5 calls/min, 500/day)
   - **Use:** Stock data, fundamentals
   - **Status:** ✅ Configured as backup

3. **Polygon.io**
   - **Key:** peRAMicTnZi6GEdxratGhkujvvSgzwmn
   - **Tier:** Free tier
   - **Use:** Historical data, end-of-day prices
   - **Status:** ✅ Configured as backup

4. **Finnhub**
   - **Key:** c39rsbqad3i9bcobhve0
   - **Tier:** Free (60 calls/min)
   - **Use:** Real-time quotes, basic fundamentals
   - **Status:** ✅ Configured as backup

5. **Alpaca Markets**
   - **API Key:** AKKBTT6R1HMG6BSYOVVL
   - **Secret:** eOdrdIxIwVHxo4fVmVipxaNmm09qppDNm3hbKMSv
   - **Use:** Trading data, market data
   - **Status:** 🔄 Ready for integration

## AI/ML APIs (For Future Enhancements)

### Available for Advanced Analysis
1. **Google Gemini**
   - **Key:** `YOUR_GEMINI_API_KEY_HERE`
   - **Use:** Advanced pattern recognition, sentiment analysis

2. **OpenAI GPT**
   - **Key:** `YOUR_OPENAI_API_KEY_HERE`
   - **Use:** Market analysis, news interpretation

3. **DeepSeek**
   - **Key:** `YOUR_DEEPSEEK_API_KEY_HERE`
   - **Use:** Deep learning models for prediction

4. **Grok (xAI)**
   - **Key:** `YOUR_XAI_API_KEY_HERE`
   - **Use:** Real-time market insights

5. **Kimi**
   - **Key:** `YOUR_KIMI_API_KEY_HERE`
   - **Use:** Document analysis, research

## Additional Integrations

### TradingView
- **Username:** h.adam.doherty@gmail.com
- **Password:** AA420dam!@
- **Use:** Advanced charting, technical indicators
- **Status:** 📊 Available for web interface

### Google Finance
- **No API key needed**
- **Use:** Additional market data via web scraping
- **Status:** ✅ Enabled

## Data Source Priority Order

The application uses data sources in this priority:
1. Yahoo Finance (primary - no limits)
2. Alpha Vantage (if Yahoo fails)
3. Finnhub (if both above fail)
4. Polygon.io (for historical data)

## Rate Limits Summary

| Provider | Free Limit | Current Usage |
|----------|------------|---------------|
| Yahoo Finance | Unlimited* | Primary |
| Alpha Vantage | 5/min, 500/day | Backup |
| Polygon | 5/min | Backup |
| Finnhub | 60/min | Backup |
| Alpaca | Varies | Not active |

*Yahoo Finance has no official limits but should be used responsibly

## Future Enhancements Possible

With these API keys, future features could include:
- AI-powered market sentiment analysis
- Pattern recognition for optimal entry/exit points
- News-based volatility predictions
- Advanced technical indicator combinations
- Automated risk assessment using ML models

## Security Note

These keys are stored in `.env` file which is:
- ✅ Excluded from git (.gitignore)
- ✅ Only accessible locally
- ✅ Not shared in Docker images
- ⚠️ Should be rotated periodically for security