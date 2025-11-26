# Implementation Guide

## Service Implementation Examples

### 1. Market Service (FastAPI)

```python
# market_service/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
from datetime import datetime, timedelta

from .database import get_db
from .models import Stock, StockPrice, Watchlist
from .schemas import StockResponse, PriceResponse
from .data_providers import PolygonProvider, AlphaVantageProvider

app = FastAPI(title="Market Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data provider factory
def get_data_provider():
    return PolygonProvider(api_key=settings.POLYGON_API_KEY)

@app.get("/stocks", response_model=List[StockResponse])
async def get_stocks(
    symbol: Optional[str] = None,
    price_under: Optional[float] = None,
    active: bool = True,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(Stock).filter(Stock.is_active == active)

    if symbol:
        query = query.filter(Stock.symbol.ilike(f"%{symbol}%"))

    if price_under:
        # Join with latest price
        subquery = db.query(StockPrice.stock_id, StockPrice.close_price)\
                     .filter(StockPrice.time >= datetime.now() - timedelta(days=1))\
                     .order_by(StockPrice.time.desc())\
                     .subquery()

        query = query.join(subquery, Stock.id == subquery.c.stock_id)\
                    .filter(subquery.c.close_price <= price_under)

    stocks = query.offset(offset).limit(limit).all()
    return stocks

@app.get("/stocks/{symbol}/prices")
async def get_stock_prices(
    symbol: str,
    period: str = "1d",
    interval: str = "1m",
    db: Session = Depends(get_db)
):
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    # Calculate time range based on period
    end_time = datetime.now()
    if period == "1d":
        start_time = end_time - timedelta(days=1)
    elif period == "1w":
        start_time = end_time - timedelta(weeks=1)
    elif period == "1m":
        start_time = end_time - timedelta(days=30)
    else:
        start_time = end_time - timedelta(days=1)

    prices = db.query(StockPrice)\
               .filter(StockPrice.stock_id == stock.id)\
               .filter(StockPrice.time >= start_time)\
               .order_by(StockPrice.time)\
               .all()

    return {
        "symbol": symbol,
        "prices": [
            {
                "timestamp": price.time,
                "open": price.open_price,
                "high": price.high_price,
                "low": price.low_price,
                "close": price.close_price,
                "volume": price.volume
            }
            for price in prices
        ]
    }

# Background task for data ingestion
@app.on_event("startup")
async def start_data_ingestion():
    asyncio.create_task(ingest_market_data())

async def ingest_market_data():
    """Background task to continuously ingest market data"""
    provider = get_data_provider()
    db = next(get_db())

    while True:
        try:
            # Get all active stocks
            stocks = db.query(Stock).filter(Stock.is_active == True).all()

            for stock in stocks:
                # Fetch latest price data
                price_data = await provider.get_current_price(stock.symbol)

                # Store in database
                new_price = StockPrice(
                    time=datetime.now(),
                    stock_id=stock.id,
                    open_price=price_data['open'],
                    high_price=price_data['high'],
                    low_price=price_data['low'],
                    close_price=price_data['close'],
                    volume=price_data['volume']
                )

                db.add(new_price)
                db.commit()

                # Publish price update to message queue
                await publish_price_update(stock.symbol, price_data)

            # Wait before next iteration (60 seconds for demo)
            await asyncio.sleep(60)

        except Exception as e:
            print(f"Error in data ingestion: {e}")
            await asyncio.sleep(10)

# Data providers
class PolygonProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

    async def get_current_price(self, symbol: str):
        import aiohttp

        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                data = await response.json()

                if data['resultsCount'] > 0:
                    result = data['results'][0]
                    return {
                        'open': result['o'],
                        'high': result['h'],
                        'low': result['l'],
                        'close': result['c'],
                        'volume': result['v']
                    }
                else:
                    raise Exception(f"No data for symbol {symbol}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 2. Options Service

```python
# options_service/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
from datetime import datetime, timedelta
import math

from .database import get_db
from .models import Stock, OptionsChain, StrategySignal
from .schemas import OptionsChainResponse, OpportunityResponse
from .calculations import BlackScholes, Greeks

app = FastAPI(title="Options Service", version="1.0.0")

@app.get("/options/chains/{symbol}")
async def get_options_chain(
    symbol: str,
    expiry: Optional[str] = None,
    option_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    query = db.query(OptionsChain).filter(OptionsChain.stock_id == stock.id)

    if expiry:
        query = query.filter(OptionsChain.expiration_date == expiry)

    if option_type:
        query = query.filter(OptionsChain.option_type == option_type.upper())

    options = query.order_by(OptionsChain.strike_price).all()

    return {
        "symbol": symbol,
        "options": [
            {
                "strike": option.strike_price,
                "option_type": option.option_type,
                "expiry": option.expiration_date,
                "bid": option.bid_price,
                "ask": option.ask_price,
                "last": option.last_price,
                "volume": option.volume,
                "open_interest": option.open_interest,
                "implied_volatility": option.implied_volatility,
                "delta": option.delta,
                "theta": option.theta,
                "gamma": option.gamma,
                "vega": option.vega
            }
            for option in options
        ]
    }

@app.get("/options/opportunities")
async def get_opportunities(
    strategy: str = "csp",
    min_yield: float = 0.01,
    max_dte: int = 45,
    min_dte: int = 15,
    db: Session = Depends(get_db)
):
    """Find options trading opportunities based on strategy"""

    opportunities = []

    if strategy.lower() == "csp":
        opportunities = find_csp_opportunities(db, min_yield, max_dte, min_dte)
    elif strategy.lower() == "cc":
        opportunities = find_cc_opportunities(db, min_yield, max_dte, min_dte)
    elif strategy.lower() == "wheel":
        opportunities = find_wheel_opportunities(db, min_yield, max_dte, min_dte)

    return {"opportunities": opportunities}

def find_csp_opportunities(db: Session, min_yield: float, max_dte: int, min_dte: int):
    """Find cash-secured put opportunities"""

    current_date = datetime.now().date()
    expiry_start = current_date + timedelta(days=min_dte)
    expiry_end = current_date + timedelta(days=max_dte)

    # Query puts with good premium yield
    opportunities = db.query(
        OptionsChain, Stock
    ).join(
        Stock, OptionsChain.stock_id == Stock.id
    ).filter(
        OptionsChain.option_type == 'PUT',
        OptionsChain.expiration_date >= expiry_start,
        OptionsChain.expiration_date <= expiry_end,
        OptionsChain.delta >= -0.35,  # Not too deep ITM
        OptionsChain.delta <= -0.15,  # Not too far OTM
        OptionsChain.bid_price > 0,
        Stock.is_active == True
    ).all()

    results = []
    for option, stock in opportunities:
        # Calculate premium yield
        capital_required = option.strike_price * 100
        premium = option.bid_price * 100
        premium_yield = premium / capital_required

        if premium_yield >= min_yield:
            # Calculate additional metrics
            dte = (option.expiration_date - current_date).days
            annualized_yield = premium_yield * (365 / dte)

            # Probability of profit (approximation)
            prob_profit = 0.5 + option.delta  # Simplified calculation

            results.append({
                "symbol": stock.symbol,
                "strategy": "cash_secured_put",
                "strike": float(option.strike_price),
                "expiry": option.expiration_date.isoformat(),
                "premium": float(option.bid_price),
                "yield": premium_yield,
                "annualized_yield": annualized_yield,
                "probability_profit": abs(prob_profit),
                "days_to_expiry": dte,
                "capital_required": capital_required,
                "delta": float(option.delta),
                "theta": float(option.theta),
                "implied_volatility": float(option.implied_volatility)
            })

    # Sort by yield descending
    results.sort(key=lambda x: x['yield'], reverse=True)
    return results[:20]  # Return top 20

@app.post("/options/analyze")
async def analyze_strategy(
    symbol: str,
    strategy: str,
    position_size: int,
    target_yield: float,
    db: Session = Depends(get_db)
):
    """Analyze a specific options strategy"""

    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    # Get current stock price (latest price from stock_prices table)
    current_price_query = db.query(StockPrice)\
                           .filter(StockPrice.stock_id == stock.id)\
                           .order_by(StockPrice.time.desc())\
                           .first()

    if not current_price_query:
        raise HTTPException(status_code=404, detail="No price data available")

    current_price = current_price_query.close_price

    analysis = {
        "symbol": symbol,
        "current_price": float(current_price),
        "strategy": strategy,
        "position_size": position_size,
        "analysis": {}
    }

    if strategy.lower() == "wheel":
        analysis["analysis"] = analyze_wheel_strategy(
            db, stock, current_price, position_size, target_yield
        )

    return analysis

def analyze_wheel_strategy(db: Session, stock, current_price: float,
                          position_size: int, target_yield: float):
    """Analyze wheel strategy for a specific stock"""

    # Find optimal CSP strike
    csp_options = db.query(OptionsChain).filter(
        OptionsChain.stock_id == stock.id,
        OptionsChain.option_type == 'PUT',
        OptionsChain.strike_price <= current_price * 0.95,  # 5% OTM or more
        OptionsChain.expiration_date >= datetime.now().date() + timedelta(days=15),
        OptionsChain.expiration_date <= datetime.now().date() + timedelta(days=45)
    ).order_by(OptionsChain.strike_price.desc()).all()

    # Find optimal CC strike (if assigned)
    cc_options = db.query(OptionsChain).filter(
        OptionsChain.stock_id == stock.id,
        OptionsChain.option_type == 'CALL',
        OptionsChain.strike_price >= current_price * 1.05,  # 5% OTM or more
        OptionsChain.expiration_date >= datetime.now().date() + timedelta(days=15),
        OptionsChain.expiration_date <= datetime.now().date() + timedelta(days=45)
    ).order_by(OptionsChain.strike_price.asc()).all()

    recommendations = {
        "csp_recommendations": [],
        "cc_recommendations": [],
        "wheel_analysis": {}
    }

    # Analyze CSP opportunities
    for option in csp_options[:5]:  # Top 5 CSP options
        capital_required = option.strike_price * 100 * position_size
        premium = option.bid_price * 100 * position_size
        premium_yield = premium / capital_required
        dte = (option.expiration_date - datetime.now().date()).days

        recommendations["csp_recommendations"].append({
            "strike": float(option.strike_price),
            "expiry": option.expiration_date.isoformat(),
            "premium": float(premium),
            "yield": premium_yield,
            "annualized_yield": premium_yield * (365 / dte),
            "capital_required": capital_required,
            "probability_profit": abs(0.5 + option.delta),
            "delta": float(option.delta)
        })

    # Analyze CC opportunities (assuming assignment)
    for option in cc_options[:5]:  # Top 5 CC options
        premium = option.bid_price * 100 * position_size
        stock_cost = current_price * 100 * position_size
        potential_gain = (option.strike_price - current_price) * 100 * position_size
        total_return = premium + potential_gain
        total_yield = total_return / stock_cost
        dte = (option.expiration_date - datetime.now().date()).days

        recommendations["cc_recommendations"].append({
            "strike": float(option.strike_price),
            "expiry": option.expiration_date.isoformat(),
            "premium": float(premium),
            "capital_gain": float(potential_gain),
            "total_return": float(total_return),
            "yield": total_yield,
            "annualized_yield": total_yield * (365 / dte),
            "delta": float(option.delta)
        })

    return recommendations

# Background task for strategy signal generation
@app.on_event("startup")
async def start_strategy_scanner():
    asyncio.create_task(scan_opportunities())

async def scan_opportunities():
    """Background task to scan for trading opportunities"""
    db = next(get_db())

    while True:
        try:
            # Scan for high-probability CSP opportunities
            csp_opportunities = find_csp_opportunities(db, 0.015, 45, 15)

            for opp in csp_opportunities[:10]:  # Top 10
                # Create strategy signal
                signal = StrategySignal(
                    stock_id=db.query(Stock).filter(Stock.symbol == opp['symbol']).first().id,
                    signal_type='csp_opportunity',
                    strategy='wheel',
                    strike_price=opp['strike'],
                    expiration_date=datetime.fromisoformat(opp['expiry']).date(),
                    premium_yield=opp['yield'],
                    probability_profit=opp['probability_profit'],
                    confidence_score=min(opp['yield'] * 50, 1.0),  # Simple scoring
                    reasoning=f"High premium yield {opp['yield']:.2%} with {opp['probability_profit']:.1%} profit probability"
                )

                db.merge(signal)  # Use merge to avoid duplicates

            db.commit()

            # Wait 15 minutes before next scan
            await asyncio.sleep(900)

        except Exception as e:
            print(f"Error in strategy scanning: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### 3. Risk Management Service

```python
# risk_service/main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import numpy as np
from datetime import datetime, timedelta

from .database import get_db
from .models import Position, RiskMetrics, TradingAccount, Stock
from .schemas import RiskAssessmentResponse, PortfolioRiskResponse

app = FastAPI(title="Risk Management Service", version="1.0.0")

@app.get("/risk/portfolio/{user_id}")
async def get_portfolio_risk(
    user_id: str,
    account_id: str = None,
    db: Session = Depends(get_db)
) -> PortfolioRiskResponse:
    """Get comprehensive portfolio risk metrics"""

    # Get all open positions for user
    query = db.query(Position).filter(
        Position.user_id == user_id,
        Position.status == 'open'
    )

    if account_id:
        query = query.filter(Position.account_id == account_id)

    positions = query.all()

    if not positions:
        return PortfolioRiskResponse(
            portfolio_value=0,
            total_delta=0,
            total_theta=0,
            var_1_day=0,
            max_position_size=0,
            sector_exposure={}
        )

    # Calculate portfolio metrics
    portfolio_value = sum(pos.quantity * pos.current_price for pos in positions)
    total_delta = calculate_portfolio_delta(positions)
    total_theta = calculate_portfolio_theta(positions)
    var_1_day = calculate_var(positions, 1)
    var_30_day = calculate_var(positions, 30)

    # Sector exposure analysis
    sector_exposure = calculate_sector_exposure(db, positions)

    # Position concentration
    position_sizes = {}
    for pos in positions:
        stock = db.query(Stock).filter(Stock.id == pos.stock_id).first()
        symbol = stock.symbol
        position_value = pos.quantity * pos.current_price
        position_sizes[symbol] = position_value / portfolio_value

    return {
        "portfolio_value": portfolio_value,
        "total_delta": total_delta,
        "total_theta": total_theta,
        "total_vega": sum(getattr(pos, 'vega', 0) for pos in positions),
        "var_1_day": var_1_day,
        "var_30_day": var_30_day,
        "max_position_size": max(position_sizes.values()) if position_sizes else 0,
        "sector_exposure": sector_exposure,
        "position_concentration": position_sizes,
        "buying_power_used": calculate_buying_power_used(positions),
        "margin_requirement": calculate_margin_requirement(positions)
    }

@app.post("/risk/check-position")
async def check_position_risk(
    user_id: str,
    symbol: str,
    position_type: str,
    quantity: int,
    strike_price: float = None,
    premium: float = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Check if a new position meets risk criteria"""

    # Get current portfolio
    current_positions = db.query(Position).filter(
        Position.user_id == user_id,
        Position.status == 'open'
    ).all()

    # Get user risk limits
    user_limits = get_user_risk_limits(db, user_id)

    # Calculate current portfolio metrics
    current_portfolio_value = sum(pos.quantity * pos.current_price for pos in current_positions)

    # Calculate proposed position value
    if position_type.lower() == 'stock':
        position_value = quantity * strike_price  # strike_price is stock price here
    else:  # options
        position_value = quantity * 100 * premium  # Premium for options
        if position_type.lower() == 'put':
            # CSP requires cash collateral
            position_value = max(position_value, quantity * 100 * strike_price)

    # Risk checks
    risk_checks = {
        "approved": True,
        "warnings": [],
        "violations": [],
        "risk_metrics": {}
    }

    # Position size check
    position_size_pct = position_value / (current_portfolio_value + position_value)
    max_position_size = user_limits.get('max_position_size_pct', 0.05)

    if position_size_pct > max_position_size:
        risk_checks["violations"].append(
            f"Position size {position_size_pct:.2%} exceeds limit {max_position_size:.2%}"
        )
        risk_checks["approved"] = False
    elif position_size_pct > max_position_size * 0.8:
        risk_checks["warnings"].append(
            f"Position size {position_size_pct:.2%} approaching limit {max_position_size:.2%}"
        )

    # Sector concentration check
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock:
        sector_exposure = calculate_sector_exposure(db, current_positions)
        current_sector_exposure = sector_exposure.get(stock.sector, 0)
        max_sector_exposure = user_limits.get('max_sector_exposure', 0.25)

        new_sector_exposure = (
            current_sector_exposure * current_portfolio_value + position_value
        ) / (current_portfolio_value + position_value)

        if new_sector_exposure > max_sector_exposure:
            risk_checks["violations"].append(
                f"Sector exposure {new_sector_exposure:.2%} exceeds limit {max_sector_exposure:.2%}"
            )
            risk_checks["approved"] = False

    # Single stock concentration check
    current_stock_positions = [
        pos for pos in current_positions
        if pos.stock_id == stock.id if stock else False
    ]
    current_stock_value = sum(pos.quantity * pos.current_price for pos in current_stock_positions)
    new_stock_exposure = (current_stock_value + position_value) / (current_portfolio_value + position_value)
    max_single_stock = user_limits.get('max_single_stock_exposure', 0.10)

    if new_stock_exposure > max_single_stock:
        risk_checks["violations"].append(
            f"Single stock exposure {new_stock_exposure:.2%} exceeds limit {max_single_stock:.2%}"
        )
        risk_checks["approved"] = False

    # Delta exposure check (for options)
    if position_type.lower() in ['put', 'call']:
        current_delta = calculate_portfolio_delta(current_positions)
        # Estimate new position delta (simplified)
        if position_type.lower() == 'put':
            estimated_delta = quantity * -30  # Approximate delta for CSP
        else:
            estimated_delta = quantity * 50   # Approximate delta for CC

        new_total_delta = abs(current_delta + estimated_delta)
        max_delta = user_limits.get('max_portfolio_delta', 1000)

        if new_total_delta > max_delta:
            risk_checks["violations"].append(
                f"Portfolio delta {new_total_delta} exceeds limit {max_delta}"
            )
            risk_checks["approved"] = False

    risk_checks["risk_metrics"] = {
        "position_size_pct": position_size_pct,
        "sector_exposure": new_sector_exposure if stock else 0,
        "single_stock_exposure": new_stock_exposure,
        "estimated_portfolio_delta": current_delta + (estimated_delta if 'estimated_delta' in locals() else 0)
    }

    return risk_checks

def calculate_portfolio_delta(positions):
    """Calculate total portfolio delta"""
    total_delta = 0
    for pos in positions:
        if pos.position_type in ['put', 'call']:
            # Get delta from options_chain or estimate
            delta = getattr(pos, 'delta', 0)
            if pos.position_type == 'put':
                total_delta += pos.quantity * delta * 100
            else:  # call
                total_delta += pos.quantity * delta * 100
        else:  # stock
            total_delta += pos.quantity  # Stock delta = 1 per share

    return total_delta

def calculate_portfolio_theta(positions):
    """Calculate total portfolio theta (time decay)"""
    total_theta = 0
    for pos in positions:
        if pos.position_type in ['put', 'call']:
            theta = getattr(pos, 'theta', 0)
            total_theta += pos.quantity * theta * 100

    return total_theta

def calculate_var(positions, days: int, confidence: float = 0.05):
    """Calculate Value at Risk using Monte Carlo simulation"""
    # Simplified VaR calculation
    # In production, use proper Monte Carlo with correlation matrices

    portfolio_value = sum(pos.quantity * pos.current_price for pos in positions)

    # Estimate portfolio volatility (simplified)
    avg_volatility = 0.20  # 20% annual volatility assumption
    daily_volatility = avg_volatility / np.sqrt(252)
    period_volatility = daily_volatility * np.sqrt(days)

    # 5% VaR (95% confidence)
    var = portfolio_value * period_volatility * 1.645  # 95% confidence z-score

    return var

def calculate_sector_exposure(db: Session, positions):
    """Calculate exposure by sector"""
    sector_values = {}
    total_value = 0

    for pos in positions:
        stock = db.query(Stock).filter(Stock.id == pos.stock_id).first()
        if stock and stock.sector:
            position_value = pos.quantity * pos.current_price
            sector_values[stock.sector] = sector_values.get(stock.sector, 0) + position_value
            total_value += position_value

    # Convert to percentages
    if total_value > 0:
        return {sector: value / total_value for sector, value in sector_values.items()}
    else:
        return {}

def calculate_buying_power_used(positions):
    """Calculate buying power used by positions"""
    buying_power_used = 0

    for pos in positions:
        if pos.position_type == 'stock':
            buying_power_used += pos.quantity * pos.current_price
        elif pos.position_type == 'put':
            # CSP requires cash collateral
            buying_power_used += pos.quantity * 100 * pos.strike_price
        # Covered calls don't require additional buying power if stock is owned

    return buying_power_used

def calculate_margin_requirement(positions):
    """Calculate margin requirement for positions"""
    # Simplified margin calculation
    # In production, use broker-specific margin requirements

    margin_requirement = 0

    for pos in positions:
        if pos.position_type == 'stock':
            # 50% margin for stocks
            margin_requirement += pos.quantity * pos.current_price * 0.5
        elif pos.position_type == 'put':
            # CSP margin = strike price - premium received
            margin_requirement += pos.quantity * 100 * (pos.strike_price - pos.opening_premium)

    return margin_requirement

def get_user_risk_limits(db: Session, user_id: str):
    """Get user-specific risk limits"""
    # In production, fetch from user preferences or risk profile
    return {
        'max_position_size_pct': 0.05,     # 5% max per position
        'max_sector_exposure': 0.25,        # 25% max per sector
        'max_single_stock_exposure': 0.10,  # 10% max per stock
        'max_portfolio_delta': 1000,        # Max portfolio delta
        'max_daily_loss': 0.02              # 2% max daily loss
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
```

## Deployment Configuration

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Database
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: wheel_strategy
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    ports:
      - "5432:5432"
    networks:
      - wheel-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - wheel-network

  # Message Queue
  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: your_secure_password
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - wheel-network

  # API Gateway
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - market-service
      - options-service
      - risk-service
    networks:
      - wheel-network

  # Services
  market-service:
    build: ./market_service
    environment:
      DATABASE_URL: postgresql://postgres:your_secure_password@postgres:5432/wheel_strategy
      REDIS_URL: redis://redis:6379
      POLYGON_API_KEY: ${POLYGON_API_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - wheel-network

  options-service:
    build: ./options_service
    environment:
      DATABASE_URL: postgresql://postgres:your_secure_password@postgres:5432/wheel_strategy
      REDIS_URL: redis://redis:6379
      TRADIER_API_KEY: ${TRADIER_API_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - wheel-network

  risk-service:
    build: ./risk_service
    environment:
      DATABASE_URL: postgresql://postgres:your_secure_password@postgres:5432/wheel_strategy
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - wheel-network

  # Frontend
  web-dashboard:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      REACT_APP_API_URL: http://localhost
    depends_on:
      - nginx
    networks:
      - wheel-network

  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - wheel-network

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    networks:
      - wheel-network

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
  prometheus_data:
  grafana_data:

networks:
  wheel-network:
    driver: bridge
```

### Environment Configuration

```bash
# .env
# Database
DATABASE_URL=postgresql://postgres:your_secure_password@localhost:5432/wheel_strategy

# External APIs
POLYGON_API_KEY=your_polygon_api_key
TRADIER_API_KEY=your_tradier_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Redis
REDIS_URL=redis://localhost:6379

# JWT Secret
JWT_SECRET=your_jwt_secret_key

# Environment
ENVIRONMENT=development
DEBUG=true

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100

# Monitoring
SENTRY_DSN=your_sentry_dsn
```

## Scaling Strategies

### Horizontal Scaling Approach

1. **Service Replication**: Scale services independently based on load
2. **Database Sharding**: Partition by user_id or symbol
3. **Caching Strategy**: Multi-layer caching (Redis, CDN, application)
4. **Load Balancing**: Round-robin with health checks

### Performance Monitoring

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
api_requests = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_duration = Histogram('api_request_duration_seconds', 'API request duration')
active_positions = Gauge('active_positions_total', 'Total active positions')
portfolio_value = Gauge('portfolio_value_total', 'Total portfolio value', ['user_id'])

# Middleware for FastAPI
@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    api_requests.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    api_duration.observe(duration)

    return response
```

This implementation guide provides a solid foundation for building the options wheel strategy trading system with proper architecture, error handling, monitoring, and scalability considerations.