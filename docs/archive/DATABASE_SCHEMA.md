# Database Schema for Wheel Strategy

## Overview
This application uses PostgreSQL to store your stock holdings, watchlist, and track opportunities. It pulls data from YOUR database and uses free market data (Yahoo Finance) to find the best premium opportunities.

## Database Tables

### 1. `stock_holdings` - Your Current Stock Positions
```sql
CREATE TABLE stock_holdings (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL,        -- Stock ticker (AAPL, MSFT, etc.)
  shares INTEGER NOT NULL,            -- Number of shares you own
  cost_basis DECIMAL(10, 2),         -- Your average purchase price
  purchase_date DATE,                 -- When you bought the shares
  account_type VARCHAR(50),          -- IRA, Taxable, etc.
  notes TEXT,                         -- Any notes about the position
  active BOOLEAN DEFAULT true,        -- Is this position still active?
  created_at TIMESTAMP DEFAULT NOW()
);
```

### 2. `watchlist` - Stocks You Want to Buy (Cash-Secured Puts)
```sql
CREATE TABLE watchlist (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL UNIQUE, -- Stock ticker
  target_price DECIMAL(10, 2),       -- Price you'd like to buy at
  max_shares INTEGER DEFAULT 100,     -- Maximum shares you'd buy
  priority INTEGER DEFAULT 5,         -- 1-10 priority (10 = highest)
  notes TEXT,
  active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW()
);
```

### 3. `opportunities` - Discovered Premium Opportunities
```sql
CREATE TABLE opportunities (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL,
  strategy VARCHAR(20) NOT NULL,      -- 'covered-call' or 'cash-secured-put'
  strike DECIMAL(10, 2),              -- Strike price
  premium DECIMAL(10, 2),             -- Premium per share
  expiration DATE,                    -- Option expiration date
  confidence_score INTEGER,           -- 0-100 confidence score
  risk_score INTEGER,                 -- 0-100 risk score
  expected_return DECIMAL(10, 2),     -- Annualized return %
  analysis_data JSONB,                -- Detailed analysis metrics
  created_at TIMESTAMP DEFAULT NOW(),
  executed BOOLEAN DEFAULT false      -- Did you execute this trade?
);
```

### 4. `trade_history` - Your Past Trades (For Learning)
```sql
CREATE TABLE trade_history (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL,
  strategy VARCHAR(20) NOT NULL,
  strike DECIMAL(10, 2),
  premium DECIMAL(10, 2),
  quantity INTEGER DEFAULT 1,
  expiration DATE,
  entry_date DATE,
  exit_date DATE,
  profit_loss DECIMAL(10, 2),
  exit_reason VARCHAR(50),           -- 'expired', 'assigned', 'closed_early'
  notes TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

## How It Works

1. **You add your stocks to `stock_holdings`**
   ```sql
   INSERT INTO stock_holdings (symbol, shares, cost_basis) 
   VALUES ('AAPL', 200, 150.00);
   ```

2. **You add watchlist stocks for cash-secured puts**
   ```sql
   INSERT INTO watchlist (symbol, target_price, priority) 
   VALUES ('NVDA', 400.00, 10);
   ```

3. **The app scans for opportunities:**
   - For your holdings → Finds best covered calls
   - For your watchlist → Finds best cash-secured puts
   - Uses FREE Yahoo Finance data for real-time prices and options chains

4. **Results show confidence and risk scores:**
   - **Confidence Score (0-100)**: How likely this trade will be profitable
   - **Risk Score (0-100)**: How risky this trade is
   - **Expected Return**: Annualized return percentage

## Quick Setup

1. **Create your PostgreSQL database:**
```bash
psql -U postgres
CREATE DATABASE trading;
\c trading
```

2. **Configure .env file:**
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=yourpassword
```

3. **Run the app setup:**
```bash
npm install
npm run setup  # Creates tables and adds sample data
```

4. **Add your holdings:**
```bash
node src/app.js add-holding AAPL 100 150.00
node src/app.js add-holding MSFT 200 300.00
```

5. **Add your watchlist:**
```bash
node src/app.js add-watchlist NVDA 400 10
node src/app.js add-watchlist TSLA 200 8
```

6. **Scan for opportunities:**
```bash
npm start
# or
node src/app.js scan
```

## Sample Output
```
=================================================================================
WHEEL STRATEGY OPPORTUNITY REPORT
=================================================================================

📊 SUMMARY
Total Opportunities: 45
Strong Buys: 8 | Buys: 15
Average Confidence: 72%
Average Risk: 35%
Average Expected Return: 18.5%

🎯 TOP COVERED CALL OPPORTUNITIES
Symbol | Strike | Premium | Exp    | Confidence | Risk | Return | Action
AAPL   | $160.00 | $2.50   | 12/20  |        85% |  25% |  19.5% | STRONG BUY
  ↳ Excellent return of 19.5%, High probability of profit (78%), 6.7% upside potential

💰 TOP CASH-SECURED PUT OPPORTUNITIES  
Symbol | Strike | Premium | Exp    | Confidence | Risk | Return | Action
NVDA   | $400.00 | $8.00   | 12/20  |        82% |  40% |  24.0% | STRONG BUY
  ↳ Excellent return of 24.0%, 7.5% discount to current price
```

## Features

- ✅ Pulls YOUR stocks from PostgreSQL
- ✅ Uses FREE Yahoo Finance for real-time data
- ✅ Calculates confidence scores based on multiple factors
- ✅ Risk assessment for each opportunity
- ✅ Learns from your historical trades
- ✅ No API keys required for market data
- ✅ Works immediately after setup