# Setup Guide for Wheel Strategy Application

## Prerequisites
- Node.js 16+ installed
- Git installed
- Market data API access (see API Keys section below)

## Installation

1. **Clone and install dependencies:**
```bash
git clone <repository-url>
cd WheelStrategy
npm install
```

2. **Create environment file:**
```bash
cp .env.example .env
```

3. **Configure environment variables** (see Configuration section below)

4. **Initialize application:**
```bash
npm start
```

## Required API Keys

You need at least one market data provider API key:

### Option 1: Alpha Vantage (Recommended for beginners)
- **Free tier:** 5 API calls per minute, 500 per day
- **Cost:** Free tier available, premium starts at $49.99/month
- **Sign up:** https://www.alphavantage.co/
- **Add to .env:** `ALPHA_VANTAGE_API_KEY=your_key_here`

### Option 2: Polygon.io (Recommended for active trading)
- **Free tier:** 5 calls per minute, limited historical data
- **Cost:** Starter plan $99/month for real-time data
- **Sign up:** https://polygon.io/
- **Add to .env:** `POLYGON_API_KEY=your_key_here`

### Option 3: Finnhub (Good for basic data)
- **Free tier:** 60 calls per minute, limited features
- **Cost:** Basic plan $39.99/month
- **Sign up:** https://finnhub.io/
- **Add to .env:** `FINNHUB_API_KEY=your_key_here`

### Option 4: IEX Cloud (Developer friendly)
- **Free tier:** 500,000 free messages per month
- **Cost:** Pay-as-you-go, very affordable
- **Sign up:** https://iexcloud.io/
- **Add to .env:** `IEX_CLOUD_API_KEY=your_key_here`

## Configuration

### Required Configuration (.env file)

```bash
# Choose your market data provider
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Portfolio settings
INITIAL_CASH=100000
MAX_POSITION_SIZE=0.05

# Tax settings
TAX_YEAR=2024
FILING_STATUS=single
STATE_TAX_RATE=0.07
```

### Optional Configuration

```bash
# Email notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
NOTIFICATION_EMAIL=your_email@gmail.com

# Trading parameters
MIN_OPTIONS_VOLUME=1000
TARGET_ANNUAL_RETURN=15
```

## Database Setup

The application uses SQLite by default (no setup required). For production use:

### PostgreSQL Setup:
```bash
# Install PostgreSQL
# Create database: wheel_strategy
# Update .env:
DATABASE_URL=postgresql://username:password@localhost:5432/wheel_strategy
```

### MySQL Setup:
```bash
# Install MySQL
# Create database: wheel_strategy  
# Update .env:
DATABASE_URL=mysql://username:password@localhost:3306/wheel_strategy
```

## What's Included

### ✅ Core Components Created
- **Portfolio Management System**
- **Cash-Secured Put Strategy Engine**
- **Covered Call Strategy Engine**
- **Tax Calculator with 2024 tax brackets**
- **Risk Management System**
- **Stock Screening Tools**
- **Performance Analytics**
- **Configuration Management**

### ✅ File Structure
```
WheelStrategy/
├── src/
│   ├── models/          # Data models (Stock, Option, Position)
│   ├── strategies/      # Strategy implementations
│   ├── portfolio/       # Portfolio and risk management
│   ├── tax/            # Tax calculations
│   ├── analysis/       # Screening and performance tools
│   └── utils/          # Utilities and logging
├── config/             # Configuration files
├── tests/              # Test files
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Missing Components (What You Need to Implement)

### 🔑 Required for Full Functionality

1. **Market Data Provider Integration**
   - Implement `src/api/DataProvider.js`
   - Connect to your chosen API (Alpha Vantage, Polygon, etc.)
   - Real-time stock prices and options chains

2. **Database Layer** 
   - Implement data persistence
   - Position history storage
   - Performance tracking over time

3. **Options Data Processing**
   - Real options chain parsing
   - Greeks calculations
   - Real-time premium updates

### 💡 Optional Enhancements

4. **Web Interface**
   - Dashboard for portfolio monitoring
   - Position management UI
   - Performance charts

5. **Automated Trading Integration**
   - Broker API integration (TD Ameritrade, Interactive Brokers)
   - Automated position execution
   - Real-time monitoring

6. **Advanced Features**
   - Machine learning for premium prediction
   - Advanced backtesting
   - Multi-account management

## Quick Start Example

```javascript
const WheelStrategy = require('./src/index');

// Initialize application
const app = new WheelStrategy();
await app.initialize();

// Analyze a cash-secured put opportunity  
const analysis = await app.analyzePosition(
  'AAPL',           // symbol
  'cash-secured-put', // strategy
  220,              // strike
  '2024-12-20',     // expiration
  3.00              // premium
);

console.log('Analysis:', analysis);

// Add position if analysis looks good
if (analysis.recommendation === 'BUY') {
  const position = await app.addPosition(analysis);
  console.log('Position added:', position.id);
}
```

## Development Setup

```bash
# Install dependencies
npm install

# Run tests
npm test

# Start in development mode
npm run dev

# Lint code
npm run lint
```

## Production Deployment

1. **Set environment to production:**
```bash
NODE_ENV=production
```

2. **Use production database:**
```bash
DATABASE_URL=postgresql://user:password@host:5432/wheel_strategy
```

3. **Configure proper API keys with sufficient rate limits**

4. **Set up monitoring and alerting**

## Support and Documentation

- **API Documentation:** `docs/api/`
- **User Guides:** `docs/guides/`
- **Configuration Reference:** `config/config.js`
- **Log Files:** `logs/`

## Cost Estimates

### Monthly Costs for Full Operation:
- **Market Data API:** $40-$100/month
- **Database Hosting:** $20-$50/month (if using cloud)
- **Email Service:** $0-$20/month
- **Total:** $60-$170/month

### Free Tier Options:
- Alpha Vantage free tier
- SQLite database (local)
- Gmail SMTP
- **Total:** $0/month (with limitations)

Ready to start building your options income generation system!