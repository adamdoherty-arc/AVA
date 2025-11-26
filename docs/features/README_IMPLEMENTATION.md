# Wheel Strategy Trading System - Implementation Guide

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Run the system
python src/main.py --watchlist F INTC T PLTR SOFI --capital 50000

# Launch dashboard (in another terminal)
streamlit run dashboard.py
```

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Web Dashboard (Streamlit)              │
└─────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────┐
│                    Main Application                      │
│                    (src/main.py)                        │
└─────────────────────────────────────────────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
┌──────────┐         ┌──────────────┐       ┌─────────────┐
│ Market   │         │ Wheel        │       │ Risk        │
│ Data     │         │ Strategy     │       │ Management  │
│ Agent    │         │ Agent        │       │ Agent       │
└──────────┘         └──────────────┘       └─────────────┘
     │                       │                       │
┌─────────────────────────────────────────────────────────┐
│               Redis Cache & Message Queue                │
└─────────────────────────────────────────────────────────┘
     │                                               │
┌──────────┐                                 ┌─────────────┐
│PostgreSQL│                                 │ Alert Agent │
│   +      │                                 │  (Multi-    │
│TimescaleDB                                 │  Channel)   │
└──────────┘                                 └─────────────┘
```

## Key Components

### 1. Market Data Agent
- **File**: `src/agents/market_data_agent.py`
- **Functions**:
  - Scans for stocks under $50
  - Monitors price movements
  - Integrates TradingView signals
  - Filters by volume and liquidity

### 2. Wheel Strategy Agent
- **File**: `src/agents/wheel_strategy_agent.py`
- **Functions**:
  - Finds optimal CSP opportunities
  - Identifies CC opportunities
  - Calculates Greeks
  - Manages wheel cycles

### 3. Risk Management Agent
- **File**: `src/agents/risk_management_agent.py`
- **Functions**:
  - Position sizing (Kelly Criterion)
  - Portfolio risk assessment
  - Sector allocation monitoring
  - Trade validation

### 4. Alert Agent
- **File**: `src/agents/alert_agent.py`
- **Channels**: Console, Discord, Email, Telegram
- **Alert Types**:
  - New opportunities
  - Price movements (>3%)
  - Assignment warnings
  - Risk alerts

## Configuration

Edit `config.json` for your needs:

```json
{
  "watchlist": ["F", "INTC", "T", "PLTR", "SOFI"],
  "strategy_params": {
    "min_premium_yield": 0.01,  // 1% monthly minimum
    "target_delta": 0.30,        // 30 delta
    "min_dte": 21,               // Days to expiration
    "max_dte": 45
  },
  "risk_limits": {
    "max_position_size_pct": 0.05,  // 5% max
    "max_sector_exposure": 0.30     // 30% max
  }
}
```

## Dashboard Features

### Pages:
1. **Dashboard**: Portfolio overview, performance charts
2. **Opportunities**: Live scanning, ranked by score
3. **Positions**: Active positions management
4. **Risk Analysis**: Portfolio risk metrics, VaR
5. **Settings**: Strategy parameters, alerts

### Key Metrics:
- Portfolio value and P&L
- Cash available vs deployed
- Active positions by strategy
- Monthly income tracking
- Risk score (0-100)
- Sector allocation

## Database Schema

Core tables (PostgreSQL + TimescaleDB):
- `stocks`: Symbol metadata
- `stock_prices`: Time-series price data
- `options_chains`: Options data
- `positions`: Active positions
- `wheel_cycles`: Wheel strategy cycles
- `strategy_signals`: Trading opportunities
- `risk_metrics`: Portfolio risk tracking

## API Integrations

### Current:
- **yfinance**: Stock and options data (free)
- **tradingview-ta**: Technical analysis

### Planned:
- **Polygon.io**: Professional market data
- **Tradier**: Options chains
- **Alpaca**: Trading execution

## Alert Setup

### Discord
```json
"discord_webhook": "https://discord.com/api/webhooks/YOUR_WEBHOOK"
```

### Email (Gmail)
```json
"email": {
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "username": "your-email@gmail.com",
  "password": "app-specific-password"
}
```

## Performance Optimization

### Caching Strategy
- Redis for real-time prices (5min TTL)
- Options chains (1hr TTL)
- Opportunity scores (15min TTL)

### Scaling
- Horizontal: Add worker nodes
- Vertical: Increase Redis memory
- Database: Read replicas for queries

## Risk Management Rules

1. **Position Sizing**:
   - Max 5% per position
   - Adjust for volatility
   - Kelly Criterion based

2. **Sector Limits**:
   - Max 30% per sector
   - Tech sector max 40%
   - Minimum 3 sectors

3. **Portfolio Risk**:
   - Max 20% total risk
   - VaR monitoring
   - Correlation checks

## Trading Workflow

1. **Opportunity Discovery**:
   - System scans every 5 minutes
   - Filters by price, volume, IV
   - Ranks by opportunity score

2. **Validation**:
   - Risk agent validates trade
   - Checks position sizing
   - Verifies sector allocation

3. **Alert**:
   - High score opportunities alerted
   - Details sent to configured channels
   - Dashboard updates in real-time

4. **Execution** (Manual for now):
   - Review opportunity
   - Place trade with broker
   - System tracks position

## Monitoring

### Logs
- Location: `logs/wheel_strategy.log`
- Rotation: Daily
- Levels: DEBUG, INFO, WARNING, ERROR

### Metrics
- Portfolio value
- Win rate
- Average premium
- Assignment frequency
- Risk score

## Troubleshooting

### No Opportunities Found
```bash
# Check market hours
python -c "import yfinance; print(yfinance.Ticker('SPY').history(period='1d'))"

# Test with lower criteria
python src/main.py --max-price 100
```

### Redis Connection Error
```bash
# Check Redis is running
redis-cli ping

# Start Redis
redis-server
```

### Dashboard Not Loading
```bash
# Check Streamlit installation
streamlit --version

# Run with verbose output
streamlit run dashboard.py --logger.level=debug
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

## Production Deployment

### Docker
```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Monitor
docker-compose logs -f
```

### Environment Variables
```bash
export DATABASE_URL=postgresql://user:pass@localhost/wheel_strategy
export REDIS_URL=redis://localhost:6379
export DISCORD_WEBHOOK=your_webhook_url
```

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review configuration in `config.json`
3. Ensure all services are running (Redis, PostgreSQL)
4. Create an issue on GitHub

## License

MIT - See LICENSE file