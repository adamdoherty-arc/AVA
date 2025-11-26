# Quick Start Guide - Wheel Strategy Premium Optimizer

## Your Database Configuration
```
Database: magnus
Host: postgres (or localhost if running locally)
Port: 5432
User: postgres
Password: postgres123!
```

## Option 1: Docker Setup (Recommended)
```bash
# Start everything with Docker
docker-compose up -d

# The database will be automatically configured
# Tables will be created on first run
```

## Option 2: Local Setup
```bash
# 1. Install dependencies
npm install

# 2. Your .env file is already configured with your database

# 3. Setup database tables and sample data
npm run setup

# 4. Start scanning for opportunities
npm start
```

## Using the Application

### 1. Add Your Stock Holdings
```bash
# Add stocks you currently own (for covered calls)
node src/app.js add-holding AAPL 100 150.00
node src/app.js add-holding MSFT 200 250.00
node src/app.js add-holding VZ 400 38.50
```

### 2. Add Your Watchlist
```bash
# Add stocks you want to buy (for cash-secured puts)
node src/app.js add-watchlist NVDA 400 10   # Priority 10 (highest)
node src/app.js add-watchlist AMD 100 8      # Priority 8
node src/app.js add-watchlist TSLA 200 7     # Priority 7
```

### 3. Scan for Opportunities
```bash
# Find the best opportunities
npm start
# or
node src/app.js scan
```

### 4. View Your Portfolio
```bash
# Show your holdings
node src/app.js holdings

# Show your watchlist
node src/app.js watchlist
```

## Sample Output
```
=================================================================================
WHEEL STRATEGY OPPORTUNITY REPORT
=================================================================================
Generated: 12/1/2024, 2:30:45 PM
---------------------------------------------------------------------------------

📊 SUMMARY
Total Opportunities: 45
Strong Buys: 8 | Buys: 15
Average Confidence: 72%
Average Risk: 35%
Average Expected Return: 18.5%

🎯 TOP COVERED CALL OPPORTUNITIES
---------------------------------------------------------------------------------
Symbol | Strike | Premium | Exp    | Confidence | Risk | Return | Action
---------------------------------------------------------------------------------
AAPL   | $160.00 | $2.50   | 12/20  |        85% |  25% |  19.5% | STRONG BUY
  ↳ Excellent return of 19.5%, High probability of profit (78%), 6.7% upside potential
MSFT   | $380.00 | $4.75   | 12/20  |        82% |  30% |  17.8% | STRONG BUY
  ↳ High probability of profit (75%), 5.2% upside potential

💰 TOP CASH-SECURED PUT OPPORTUNITIES  
---------------------------------------------------------------------------------
Symbol | Strike | Premium | Exp    | Confidence | Risk | Return | Action
---------------------------------------------------------------------------------
NVDA   | $400.00 | $8.00   | 12/20  |        82% |  40% |  24.0% | STRONG BUY
  ↳ Excellent return of 24.0%, 7.5% discount to current price
AMD    | $95.00  | $2.25   | 12/20  |        78% |  35% |  21.3% | BUY
  ↳ Excellent return of 21.3%, 5.0% discount to current price

⚠️  RISK ANALYSIS
---------------------------------------------------------------------------------
Portfolio Diversification: 6 symbols
No significant concentration risks detected
```

## Understanding the Scores

### Confidence Score (0-100%)
- **85-100%**: STRONG BUY - Highly confident opportunity
- **70-84%**: BUY - Good opportunity
- **50-69%**: CONSIDER - Moderate opportunity
- **Below 50%**: AVOID - Too risky or low return

### Risk Score (0-100%)
- **0-30%**: Low risk
- **31-50%**: Moderate risk
- **51-70%**: High risk
- **71-100%**: Very high risk

### Factors Considered
- **Premium Yield**: Annualized return percentage
- **Probability of Profit**: Statistical chance of success
- **Liquidity**: Options volume and open interest
- **Volatility**: Implied volatility levels
- **Historical Success**: Past performance with symbol
- **Time Value**: Days to expiration optimization

## Database Tables Created

The application automatically creates these tables in your `magnus` database:

1. **stock_holdings** - Your owned stocks
2. **watchlist** - Stocks you want to buy
3. **opportunities** - Discovered opportunities with scores
4. **trade_history** - Past trades for learning
5. **positions** - Current open positions
6. **performance_metrics** - Performance tracking

## Troubleshooting

### Database Connection Issues
```bash
# Test connection
psql -h localhost -p 5432 -U postgres -d magnus

# Password: postgres123!
```

### No Opportunities Found
1. Make sure you've added holdings and watchlist stocks
2. Check that markets are open (options data updates during market hours)
3. Try popular stocks like AAPL, MSFT, NVDA

### Data Source Issues
- Yahoo Finance works without API keys
- If Yahoo is down, add Alpha Vantage key to .env (free tier available)

## Next Steps

1. **Review opportunities** and execute trades through your broker
2. **Track your trades** by adding to trade_history table
3. **Adjust parameters** in .env for your risk tolerance
4. **Set up cron job** to scan automatically:
   ```bash
   # Add to crontab for daily 9:30 AM scan
   30 9 * * 1-5 cd /path/to/WheelStrategy && npm start >> logs/scan.log 2>&1
   ```

## Support

- Check `logs/` directory for detailed logs
- Review `DATABASE_SCHEMA.md` for table details
- See `FREE_DATA_SOURCES.md` for data provider options