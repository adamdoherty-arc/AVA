# API Specifications and Integration Requirements

## External API Integrations

### 1. TradingView Integration

**Challenges**: TradingView doesn't provide a public API for watchlist data. Alternative approaches:

#### Option A: Web Scraping (Not Recommended)
- Use Selenium/Playwright to scrape watchlist data
- High maintenance, potential legal issues
- Rate limiting and anti-bot detection

#### Option B: Manual CSV Export/Import
- Users export watchlist as CSV from TradingView
- System imports and monitors symbols
- Most practical approach initially

#### Option C: Alternative Data Providers
**Recommended**: Use professional market data APIs:

```python
# Alpha Vantage (Free tier available)
BASE_URL = "https://www.alphavantage.co/query"
API_KEY = "your_api_key"

async def get_stock_data(symbol: str):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    # Implementation details...

# Polygon.io (Professional)
async def get_polygon_data(symbol: str):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2023-01-01/2023-12-31"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    # Implementation details...
```

### 2. Options Data Providers

#### Option A: Polygon.io Options API
```python
async def get_options_chain(symbol: str, expiry: str = None):
    url = f"https://api.polygon.io/v3/reference/options/contracts"
    params = {
        "underlying_ticker": symbol,
        "expiration_date": expiry,
        "limit": 1000
    }
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}

async def get_options_quotes(option_symbol: str):
    url = f"https://api.polygon.io/v3/quotes/{option_symbol}"
    # Real-time options quotes
```

#### Option B: Tradier API
```python
class TradierClient:
    BASE_URL = "https://api.tradier.com/v1"

    async def get_options_chain(self, symbol: str):
        endpoint = f"/markets/options/chains"
        params = {"symbol": symbol, "greeks": "true"}
        headers = {"Authorization": f"Bearer {self.token}"}

    async def get_options_quotes(self, symbols: list):
        endpoint = "/markets/quotes"
        params = {"symbols": ",".join(symbols), "greeks": "true"}
```

#### Option C: Interactive Brokers TWS API
```python
from ib_insync import IB, Stock, Option

class IBDataProvider:
    def __init__(self):
        self.ib = IB()

    async def connect(self):
        await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)

    async def get_options_chain(self, symbol: str):
        stock = Stock(symbol, 'SMART', 'USD')
        chains = await self.ib.reqSecDefOptParamsAsync(stock)
        # Get full options chain with Greeks
```

## Core API Endpoints

### Authentication Service

```python
# POST /auth/login
{
    "email": "user@example.com",
    "password": "secure_password"
}
# Response: {"access_token": "jwt_token", "refresh_token": "refresh_jwt"}

# POST /auth/refresh
{
    "refresh_token": "refresh_jwt"
}
# Response: {"access_token": "new_jwt_token"}

# POST /auth/logout
# Headers: Authorization: Bearer jwt_token
```

### Market Data Service

```python
# GET /api/v1/stocks
# Query params: ?symbol=AAPL&active=true&price_under=50
{
    "data": [
        {
            "id": "uuid",
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "current_price": 175.50,
            "change_percent": 1.25,
            "volume": 50000000,
            "market_cap": 2800000000000,
            "sector": "Technology",
            "is_optionable": true
        }
    ],
    "pagination": {"page": 1, "limit": 50, "total": 150}
}

# GET /api/v1/stocks/{symbol}/prices
# Query params: ?period=1d&interval=1m
{
    "symbol": "AAPL",
    "prices": [
        {
            "timestamp": "2023-10-23T09:30:00Z",
            "open": 175.00,
            "high": 175.75,
            "low": 174.50,
            "close": 175.25,
            "volume": 1000000
        }
    ]
}

# POST /api/v1/watchlists
{
    "name": "Wheel Strategy Candidates",
    "description": "Stocks under $50 for wheel strategy",
    "symbols": ["AAPL", "MSFT", "TSLA"]
}

# GET /api/v1/watchlists/{id}/stocks
{
    "watchlist": {
        "id": "uuid",
        "name": "Wheel Strategy Candidates",
        "stocks": [
            {
                "symbol": "AAPL",
                "current_price": 175.50,
                "target_entry": 170.00,
                "target_premium_yield": 0.02
            }
        ]
    }
}
```

### Options Service

```python
# GET /api/v1/options/chains/{symbol}
# Query params: ?expiry=2023-11-17&option_type=PUT
{
    "symbol": "AAPL",
    "expiry": "2023-11-17",
    "options": [
        {
            "strike": 170.00,
            "option_type": "PUT",
            "bid": 2.50,
            "ask": 2.65,
            "last": 2.55,
            "volume": 1500,
            "open_interest": 5000,
            "implied_volatility": 0.25,
            "delta": -0.30,
            "theta": -0.05,
            "gamma": 0.02,
            "vega": 0.15
        }
    ]
}

# GET /api/v1/options/opportunities
# Query params: ?strategy=csp&min_yield=0.01&max_dte=45
{
    "opportunities": [
        {
            "symbol": "AAPL",
            "strategy": "cash_secured_put",
            "strike": 170.00,
            "expiry": "2023-11-17",
            "premium": 2.55,
            "yield": 0.015,
            "probability_profit": 0.70,
            "days_to_expiry": 25,
            "capital_required": 17000,
            "risk_reward_ratio": 2.3
        }
    ]
}

# POST /api/v1/options/analyze
{
    "symbol": "AAPL",
    "strategy": "wheel",
    "position_size": 100,
    "target_yield": 0.02
}
# Response: Analysis with recommendations
```

### Position Management Service

```python
# GET /api/v1/positions
# Query params: ?status=open&strategy=wheel
{
    "positions": [
        {
            "id": "uuid",
            "symbol": "AAPL",
            "position_type": "put",
            "strategy": "wheel",
            "quantity": 1,
            "strike": 170.00,
            "expiry": "2023-11-17",
            "entry_price": 2.55,
            "current_price": 2.30,
            "unrealized_pnl": 25.00,
            "status": "open",
            "dte": 25,
            "wheel_cycle_id": "uuid"
        }
    ]
}

# POST /api/v1/positions
{
    "symbol": "AAPL",
    "position_type": "put",
    "strategy": "cash_secured_put",
    "strike": 170.00,
    "expiry": "2023-11-17",
    "quantity": 1,
    "entry_price": 2.55
}

# PUT /api/v1/positions/{id}/close
{
    "close_price": 0.50,
    "close_reason": "target_profit"
}

# GET /api/v1/wheel-cycles
{
    "cycles": [
        {
            "id": "uuid",
            "symbol": "AAPL",
            "cycle_number": 1,
            "status": "active",
            "total_premium": 510.00,
            "total_pnl": 125.00,
            "positions": [
                {
                    "position_type": "put",
                    "strike": 170.00,
                    "premium": 255.00,
                    "status": "closed"
                }
            ]
        }
    ]
}
```

### Risk Management Service

```python
# GET /api/v1/risk/portfolio
{
    "portfolio_value": 100000,
    "buying_power": 25000,
    "total_delta": -150.5,
    "total_theta": 25.75,
    "portfolio_beta": 1.15,
    "var_1_day": -2500,
    "max_position_size": 5000,
    "sector_exposure": {
        "Technology": 0.35,
        "Financial": 0.20,
        "Healthcare": 0.15
    }
}

# POST /api/v1/risk/check-position
{
    "symbol": "AAPL",
    "position_type": "put",
    "quantity": 5,
    "strike": 170.00,
    "premium": 2.55
}
# Response: Risk assessment and approval

# GET /api/v1/risk/limits
{
    "max_position_size_pct": 0.05,
    "max_sector_exposure": 0.25,
    "max_single_stock_exposure": 0.10,
    "max_portfolio_delta": 1000,
    "max_daily_loss": 0.02
}
```

### Alert Service

```python
# POST /api/v1/alerts
{
    "symbol": "AAPL",
    "alert_type": "price_below",
    "threshold": 170.00,
    "notification_methods": ["email", "webhook"]
}

# GET /api/v1/alerts
{
    "alerts": [
        {
            "id": "uuid",
            "symbol": "AAPL",
            "alert_type": "price_below",
            "threshold": 170.00,
            "is_active": true,
            "last_triggered": "2023-10-22T15:30:00Z"
        }
    ]
}

# WebSocket endpoint for real-time alerts
# ws://api.domain.com/ws/alerts
{
    "type": "price_alert",
    "symbol": "AAPL",
    "current_price": 169.50,
    "threshold": 170.00,
    "alert_type": "price_below",
    "timestamp": "2023-10-23T10:15:00Z"
}
```

## Error Handling Standards

```python
# Standard error response format
{
    "error": {
        "code": "INVALID_SYMBOL",
        "message": "Symbol 'INVALID' not found",
        "details": {
            "symbol": "INVALID",
            "suggestions": ["AAPL", "MSFT"]
        },
        "timestamp": "2023-10-23T10:15:00Z",
        "request_id": "req_123456"
    }
}

# HTTP Status Codes
# 400 - Bad Request (validation errors)
# 401 - Unauthorized (invalid/expired token)
# 403 - Forbidden (insufficient permissions)
# 404 - Not Found (resource doesn't exist)
# 409 - Conflict (duplicate resource)
# 422 - Unprocessable Entity (business logic error)
# 429 - Too Many Requests (rate limiting)
# 500 - Internal Server Error
# 503 - Service Unavailable (external API down)
```

## Rate Limiting

```python
# Rate limiting headers
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1634567890

# Rate limiting by endpoint
# /api/v1/stocks: 100 requests/minute
# /api/v1/options: 50 requests/minute
# /api/v1/positions: 200 requests/minute
# WebSocket connections: 5 concurrent/user
```

## Real-time Data Flow

```python
# WebSocket endpoints
# ws://api.domain.com/ws/prices/{symbol}
# ws://api.domain.com/ws/positions
# ws://api.domain.com/ws/alerts

# Real-time price updates
{
    "type": "price_update",
    "symbol": "AAPL",
    "price": 175.25,
    "change": 1.25,
    "change_percent": 0.71,
    "volume": 1000000,
    "timestamp": "2023-10-23T10:15:00Z"
}

# Position updates
{
    "type": "position_update",
    "position_id": "uuid",
    "unrealized_pnl": 125.00,
    "current_price": 2.30,
    "timestamp": "2023-10-23T10:15:00Z"
}
```

## Data Provider Costs and Recommendations

### Recommended Tier Strategy

**Development/Testing:**
- Alpha Vantage (Free tier: 5 calls/minute, 500 calls/day)
- IEX Cloud (Free tier: 50K calls/month)

**Production (Small Scale):**
- Polygon.io Starter ($99/month): Real-time + options
- Tradier Developer ($25/month): Real-time + options + trading

**Production (Large Scale):**
- Polygon.io Professional ($399/month): Full market data
- Interactive Brokers API: Professional data feeds

### Implementation Priority

1. **Phase 1**: Basic stock data (Alpha Vantage/IEX)
2. **Phase 2**: Options data (Polygon.io/Tradier)
3. **Phase 3**: Real-time streaming (WebSocket feeds)
4. **Phase 4**: Advanced analytics and Greeks

This API design provides a solid foundation for building the options wheel strategy system with proper separation of concerns, comprehensive error handling, and scalable data integration patterns.