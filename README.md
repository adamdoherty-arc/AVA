# Ava Trading Platform

**Advanced Options Trading Platform with AI-Powered Analysis**

Magnus is an intelligent options trading platform designed for cash-secured puts (CSP) and covered call (CC) strategies, with full TradingView integration, real-time market data, and comprehensive risk management.

## Features

Magnus provides 8 comprehensive navigation features for managing your options trading:

### 1. Dashboard - Performance Overview
- Real-time portfolio metrics (balance, buying power, premium collected)
- Balance forecast timeline by expiration date
- Individual position forecasts with theta decay analysis
- Best/expected/worst case scenario projections
- Comprehensive trade history with P&L tracking

### 2. Opportunities - Find New Trades
- Scan for cash-secured put opportunities
- Covered call finder for existing positions
- Score-based ranking with AI recommendations
- Risk assessment for each opportunity

### 3. Positions - Active Trade Management
- Real-time P&L tracking for all positions
- Theta decay profit forecasting (daily projections)
- AI trade analysis with actionable recommendations
- Live option price monitoring
- Auto-refresh capabilities

### 4. Premium Scanner - Advanced Screening
- Multi-expiration scanning (7d, 14d, 30d, 45d DTE)
- Volatility filters (IV levels)
- Return calculations (monthly & annualized)
- Volume screening for liquidity
- Customizable filters (price, premium %, DTE)

### 5. TradingView Watchlists - Sync & Analyze
- Automatic watchlist synchronization from TradingView
- Auto-refresh session management (handles expiration)
- Complete premium analysis for watchlist symbols
- Import/export capabilities
- Support for multiple watchlists

### 6. Database Scan - PostgreSQL Scanner
- Bulk stock addition to database
- Comprehensive premium scanning across all stored symbols
- Analytics by sector and price range
- Historical data management

### 7. Earnings Calendar - Event Tracking
- Upcoming earnings date tracking
- Automatic risk warnings for positions near earnings
- Integration with position management
- Real-time earnings data sync

### 8. Settings - Configuration
- Robinhood account integration
- Strategy parameters (max price, DTE preferences, risk limits)
- Alert settings and notifications
- TradingView API configuration
- Database connection management

## Screenshots

### Dashboard Overview
![Dashboard](./docs/screenshots/dashboard.png)
*Main dashboard showing portfolio metrics, balance forecasts, and trade history*

### Premium Scanner
![Premium Scanner](./docs/screenshots/premium-scanner.png)
*Advanced scanner with multi-expiration filtering and return calculations*

### Position Analysis
![Positions](./docs/screenshots/positions.png)
*Real-time position tracking with AI recommendations and theta decay forecasts*

### TradingView Integration
![TradingView Watchlists](./docs/screenshots/tradingview-watchlists.png)
*Automatic watchlist synchronization and premium analysis*

## Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 14 or higher
- Redis server
- Chrome/Chromium (for TradingView automation)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/WheelStrategy.git
cd WheelStrategy
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up PostgreSQL Database

```bash
# Windows
setup_postgres.bat

# Linux/Mac
./setup_postgres.sh
```

Or manually create the database:

```sql
CREATE DATABASE magnus;
```

Then run the schema setup scripts in `sql/` directory.

### Step 5: Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=magnus
DB_USER=postgres
DB_PASSWORD=your_password

# Robinhood Configuration (Optional)
ROBINHOOD_USERNAME=your@email.com
ROBINHOOD_PASSWORD=your_password
ROBINHOOD_MFA_SECRET=your_mfa_secret

# TradingView Configuration (Optional)
TRADINGVIEW_USERNAME=your@email.com
TRADINGVIEW_PASSWORD=your_password
TRADINGVIEW_SESSION_ID=auto_generated

# API Keys (Optional - for enhanced data)
FINNHUB_API_KEY=your_key
POLYGON_API_KEY=your_key

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Step 6: Start Redis Server

```bash
# Windows (if installed via Chocolatey or MSI)
redis-server

# Linux
sudo systemctl start redis

# Mac
brew services start redis
```

## Quick Start

### Start the Dashboard

```bash
# Windows
start_dashboard.bat

# Linux/Mac
streamlit run dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

### Sync TradingView Watchlists (Optional)

```bash
python src/tradingview_api_sync.py
```

This syncs all your TradingView watchlists automatically. If your session expires, it will open a browser for you to re-authenticate.

### Connect to Robinhood (Optional)

1. Navigate to the Settings page in the dashboard
2. Enter your Robinhood credentials
3. Click "Connect to Robinhood"
4. Your positions and account data will sync automatically

## Configuration

### Environment Variables

All configuration is managed through environment variables in the `.env` file:

#### Required Variables

- `DB_HOST` - PostgreSQL hostname (default: localhost)
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name (default: magnus)
- `DB_USER` - Database user
- `DB_PASSWORD` - Database password

#### Optional Variables

- `ROBINHOOD_USERNAME` - Robinhood account email
- `ROBINHOOD_PASSWORD` - Robinhood account password
- `ROBINHOOD_MFA_SECRET` - Two-factor authentication secret (TOTP)
- `TRADINGVIEW_USERNAME` - TradingView account email
- `TRADINGVIEW_PASSWORD` - TradingView account password
- `TRADINGVIEW_SESSION_ID` - Auto-generated session token
- `FINNHUB_API_KEY` - Finnhub API key for enhanced market data
- `POLYGON_API_KEY` - Polygon.io API key for options data

### Security Best Practices

1. **Never commit `.env` files** - They are already in `.gitignore`
2. **Use strong passwords** - Especially for database and brokerage accounts
3. **Enable MFA** - Two-factor authentication for Robinhood
4. **Rotate API keys** - Regularly update API keys and tokens
5. **Limit database access** - Use restricted database users for the application
6. **Review logs** - Monitor `logs/` directory for suspicious activity

## Usage Examples

### Find Premium Opportunities

```python
from src.premium_scanner import PremiumScanner
from src.tradingview_db_manager import TradingViewDBManager

# Get symbols from your TradingView watchlist
manager = TradingViewDBManager()
symbols = manager.get_watchlist_symbols('NVDA')

# Scan for premiums
scanner = PremiumScanner()
opportunities = scanner.scan_premiums(
    symbols=symbols,
    max_price=50,
    min_premium_pct=1.0,
    dte=30
)

# Display top 10 opportunities
for opp in opportunities[:10]:
    print(f"{opp['symbol']}: {opp['premium_pct']:.2f}% premium")
```

### Analyze Position with AI

```python
from src.ai_trade_analyzer import AITradeAnalyzer

analyzer = AITradeAnalyzer()

# Analyze a CSP position
analysis = analyzer.analyze_csp(
    symbol='NVDA',
    strike=135,
    expiration='2025-11-15',
    premium=250,
    current_value=50,
    days_to_expiry=21
)

print(analysis['recommendation']['action'])
print(analysis['recommendation']['reason'])
```

### Access Robinhood Data

```python
from src.robinhood_integration import RobinhoodIntegration

rh = RobinhoodIntegration()
rh.login()

# Get account balance
balance = rh.get_account_balance()
print(f"Buying power: ${balance['buying_power']}")

# Get current positions
positions = rh.get_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['quantity']} shares")
```

## Documentation

Detailed feature documentation is available in the `features/` directory:

- [Dashboard](./features/dashboard/README.md) - Portfolio overview and trade history
- [Opportunities](./features/opportunities/README.md) - Finding new trades
- [Positions](./features/positions/README.md) - Managing active positions
- [Premium Scanner](./features/premium_scanner/README.md) - Advanced screening tools
- [TradingView Watchlists](./features/tradingview_watchlists/README.md) - Watchlist integration
- [Database Scan](./features/database_scan/README.md) - Database management
- [Earnings Calendar](./features/earnings_calendar/README.md) - Earnings tracking
- [Settings](./features/settings/README.md) - Configuration and preferences

For a complete feature index, see [features/INDEX.md](./features/INDEX.md)

### Additional Documentation

- [NO_DUMMY_DATA_POLICY.md](./NO_DUMMY_DATA_POLICY.md) - Critical development policy
- [POSTGRES_SETUP.md](./POSTGRES_SETUP.md) - Database setup guide
- [TRADINGVIEW_SYNC_README.md](./TRADINGVIEW_SYNC_README.md) - TradingView integration
- [DATABASE_SCHEMA.md](./DATABASE_SCHEMA.md) - Complete schema documentation
- [README_OPTIONS_GUIDE.md](./README_OPTIONS_GUIDE.md) - Options trading education

## Contributing

We welcome contributions to Magnus! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the NO_DUMMY_DATA_POLICY (no fake/test data)
4. Write tests for new features
5. Ensure code passes linting (`black`, `flake8`, `mypy`)
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Disclaimer

**IMPORTANT: Please read carefully before using Magnus**

### Trading Risk Disclosure

Magnus Trading Platform is provided for **educational and informational purposes only**.

- **Not Financial Advice**: Nothing in this software constitutes financial, investment, legal, or tax advice.
- **Trade at Your Own Risk**: Options trading involves substantial risk of loss. You could lose all invested capital.
- **No Guarantees**: Past performance does not guarantee future results. All trading involves risk.
- **Do Your Research**: Always conduct your own research and due diligence before making any trades.
- **Paper Trading First**: Test the platform with paper trading before risking real capital.
- **Professional Advice**: Consult with a licensed financial advisor before making investment decisions.

### Software Disclaimer

- **No Warranty**: This software is provided "as is" without warranty of any kind.
- **No Liability**: The developers are not liable for any losses incurred through use of this software.
- **Data Accuracy**: While we strive for accuracy, data may be delayed, incomplete, or incorrect.
- **Broker Integration**: Robinhood integration is unofficial and may break without notice.
- **Beta Software**: This platform is under active development and may contain bugs.

### Security Warning

- **Credential Storage**: Your brokerage credentials are stored locally. Keep your system secure.
- **API Limitations**: Brokerage APIs have rate limits. Excessive use may result in temporary blocks.
- **Two-Factor Authentication**: Always use MFA for brokerage accounts.
- **Regular Updates**: Keep the software updated for security patches.

### Options Trading Risks

- **Leverage Risk**: Options involve leverage, which can magnify both gains and losses.
- **Assignment Risk**: Short options can be assigned at any time, requiring capital deployment.
- **Time Decay**: Options lose value over time (theta decay).
- **Volatility Risk**: Option values are sensitive to implied volatility changes.
- **Liquidity Risk**: Some options have low volume and wide bid-ask spreads.
- **Early Assignment**: American-style options can be exercised early.
- **Earnings Risk**: Positions held through earnings face elevated risk.

### Responsible Use

By using Magnus Trading Platform, you acknowledge that:

1. You understand the risks associated with options trading
2. You are solely responsible for your trading decisions
3. You will not hold the developers liable for any losses
4. You will comply with all applicable laws and regulations
5. You will use the software ethically and responsibly
6. You accept all risks associated with automated trading systems

**If you do not agree with these terms, do not use this software.**

## Support

### Getting Help

For issues, questions, or suggestions:

1. Check the [Documentation](./features/INDEX.md)
2. Review [Troubleshooting Guide](./docs/TROUBLESHOOTING.md)
3. Search existing [GitHub Issues](https://github.com/yourusername/WheelStrategy/issues)
4. Create a new issue with detailed information

### Reporting Bugs

When reporting bugs, include:

- Operating system and version
- Python version
- Error messages and stack traces
- Steps to reproduce
- Expected vs actual behavior

### Community

- GitHub Discussions: Share strategies and ask questions
- Discord Server: Real-time chat and support (link TBD)
- Documentation: Comprehensive guides in `features/` directory

## Roadmap

### Planned Features

- [ ] Multi-broker support (TD Ameritrade, Interactive Brokers)
- [ ] Mobile app (iOS/Android)
- [ ] Advanced backtesting engine
- [ ] Social trading features (share strategies)
- [ ] Machine learning price prediction
- [ ] Real-time profit/loss notifications
- [ ] Tax reporting and P&L exports
- [ ] Portfolio optimization algorithms

### Recent Updates

See [CHANGELOG.md](./CHANGELOG.md) for version history and updates.

## Acknowledgments

Built with:

- [Streamlit](https://streamlit.io/) - Web framework
- [PostgreSQL](https://www.postgresql.org/) - Database
- [Redis](https://redis.io/) - Caching layer
- [yfinance](https://github.com/ranaroussi/yfinance) - Market data
- [robin-stocks](https://github.com/jmfernandes/robin_stocks) - Robinhood API
- [Plotly](https://plotly.com/) - Data visualization

Special thanks to the open-source community for making this project possible.

---

**Built for options traders, by options traders.**

Current Status: **Fully Operational**
- 8 features implemented
- Real-time data integration
- AI-powered analysis
- Production-ready

**Start trading smarter with Magnus!**
