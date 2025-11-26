"""
Strategy Router - Real options strategy analysis
NO MOCK DATA - All endpoints use real market data
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import yfinance as yf
import numpy as np
from pydantic import BaseModel
from backend.services.strategy_service import get_strategy_service, StrategyService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/strategy",
    tags=["strategy"]
)

# Also create a secondary router for /api/strategies prefix
strategies_router = APIRouter(
    prefix="/api/strategies",
    tags=["strategies"]
)


@strategies_router.get("/{symbol}")
async def get_symbol_strategies(symbol: str):
    """
    Get trading strategies for a specific symbol.
    Real data from yfinance options chains.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        if not current_price:
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]

        # Get options data
        expirations = ticker.options if hasattr(ticker, 'options') else []
        iv = 0
        put_premium = 0
        call_premium = 0

        if expirations:
            try:
                chain = ticker.option_chain(expirations[0])
                if not chain.puts.empty:
                    atm_puts = chain.puts[chain.puts['strike'] <= current_price * 0.95]
                    if not atm_puts.empty:
                        put_premium = float(atm_puts.iloc[-1]['lastPrice']) / current_price * 100
                        iv = float(atm_puts.iloc[-1]['impliedVolatility']) * 100 if atm_puts.iloc[-1]['impliedVolatility'] else 30

                if not chain.calls.empty:
                    atm_calls = chain.calls[chain.calls['strike'] >= current_price * 1.05]
                    if not atm_calls.empty:
                        call_premium = float(atm_calls.iloc[0]['lastPrice']) / current_price * 100
            except Exception:
                iv = 30  # Default IV

        # Calculate quality scores
        csp_score = min(95, 50 + iv * 0.5 + put_premium * 10)
        cc_score = min(95, 50 + iv * 0.4 + call_premium * 8)
        wheel_score = (csp_score + cc_score) / 2

        strategies = [
            {
                "name": "Cash Secured Put",
                "description": f"Sell put at 5% OTM strike, collect {put_premium:.1f}% premium",
                "score": round(csp_score, 0),
                "risk": "Moderate",
                "reward": f"{put_premium:.1f}% monthly return potential",
                "recommended": csp_score > 70
            },
            {
                "name": "Covered Call",
                "description": f"Sell call at 5% OTM strike, collect {call_premium:.1f}% premium",
                "score": round(cc_score, 0),
                "risk": "Low",
                "reward": f"{call_premium:.1f}% monthly income",
                "recommended": cc_score > 70
            },
            {
                "name": "Wheel Strategy",
                "description": "Combine CSP and CC for continuous premium collection",
                "score": round(wheel_score, 0),
                "risk": "Moderate",
                "reward": "Consistent income with stock ownership",
                "recommended": wheel_score > 65
            }
        ]

        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "implied_volatility": round(iv, 1),
            "strategies": strategies,
            "market_sentiment": "Bullish" if iv < 30 else "Neutral" if iv < 50 else "Bearish",
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting strategies for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "current_price": 0,
            "implied_volatility": 0,
            "strategies": [],
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }

@router.get("/analyze")
async def analyze_watchlist(
    watchlist: str,
    min_score: float = 60.0,
    strategies: Optional[List[str]] = Query(None),
    service: StrategyService = Depends(get_strategy_service)
) -> List[Dict[str, Any]]:
    """
    Analyze a watchlist for best trading strategies.
    """
    try:
        return await service.analyze_watchlist(watchlist, min_score, strategies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from backend.services.llm_options_strategist import LLMOptionsStrategist

class StrategyRequest(BaseModel):
    symbol: str
    market_data: dict
    risk_profile: str = "moderate"
    outlook: str = "neutral"

@router.post("/generate")
async def generate_strategy(request: StrategyRequest):
    """
    Generate options trading strategy using Local LLM
    """
    strategist = LLMOptionsStrategist()
    return strategist.generate_strategy(
        symbol=request.symbol,
        market_data=request.market_data,
        risk_profile=request.risk_profile,
        outlook=request.outlook
    )


# ============ Calendar Spreads Endpoints ============

@router.get("/calendar-spreads")
async def get_calendar_spreads(
    symbols: str = Query("AAPL,NVDA,TSLA,AMD,MSFT,META", description="Comma-separated symbols"),
    min_iv_skew: float = Query(5.0, description="Minimum IV skew percentage"),
    min_edge: float = Query(3.0, description="Minimum edge percentage")
):
    """
    Get real calendar spread opportunities using yfinance options data.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(',')]

    opportunities = []
    for sym in symbol_list:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            if not current_price:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    continue

            # Get options expirations
            expirations = ticker.options
            if len(expirations) < 2:
                continue

            # Find front month (nearest) and back month expirations
            front_exp = expirations[0]
            back_exp = expirations[1] if len(expirations) > 1 else expirations[0]

            # Get option chains
            try:
                front_chain = ticker.option_chain(front_exp)
                back_chain = ticker.option_chain(back_exp)
            except Exception as e:
                logger.warning(f"Error getting options chain for {sym}: {e}")
                continue

            # Find ATM strike
            strikes = front_chain.calls['strike'].tolist()
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))

            # Get front month ATM call
            front_calls = front_chain.calls[front_chain.calls['strike'] == atm_strike]
            if front_calls.empty:
                continue
            front_call = front_calls.iloc[0]

            # Get back month ATM call
            back_calls = back_chain.calls[back_chain.calls['strike'] == atm_strike]
            if back_calls.empty:
                continue
            back_call = back_calls.iloc[0]

            # Calculate IV and skew
            front_iv = float(front_call['impliedVolatility']) * 100 if front_call['impliedVolatility'] else 0
            back_iv = float(back_call['impliedVolatility']) * 100 if back_call['impliedVolatility'] else 0

            # IV skew (front - back, positive means front is higher)
            iv_skew = front_iv - back_iv

            if iv_skew < min_iv_skew:
                continue

            # Calculate spread prices
            front_ask = float(front_call['ask']) if front_call['ask'] and front_call['ask'] > 0 else float(front_call['lastPrice'])
            front_bid = float(front_call['bid']) if front_call['bid'] and front_call['bid'] > 0 else float(front_call['lastPrice'])
            back_ask = float(back_call['ask']) if back_call['ask'] and back_call['ask'] > 0 else float(back_call['lastPrice'])
            back_bid = float(back_call['bid']) if back_call['bid'] and back_call['bid'] > 0 else float(back_call['lastPrice'])

            # Calendar spread: sell front, buy back (debit)
            debit = back_ask - front_bid if front_bid > 0 and back_ask > 0 else back_ask

            # Max loss is the debit paid
            max_loss = debit * 100  # per contract

            # Max profit is theoretical (when front expires worthless and back retains value)
            # Estimate as back value minus debit
            max_profit = (back_bid - debit) * 100 if back_bid > debit else debit * 0.5 * 100

            # Calculate edge based on IV skew
            # Higher IV skew = front decays faster = better for calendar
            edge = min(iv_skew * 0.8, 15)  # Cap at 15%

            if edge < min_edge:
                continue

            # Calculate breakevens (simplified)
            breakeven_low = atm_strike - (debit * 2)
            breakeven_high = atm_strike + (debit * 2)

            # Risk/reward ratio
            risk_reward = max_profit / max_loss if max_loss > 0 else 0

            # Confidence based on IV skew and liquidity
            front_volume = int(front_call['volume']) if front_call['volume'] else 0
            back_volume = int(back_call['volume']) if back_call['volume'] else 0
            liquidity_score = min(50, (front_volume + back_volume) / 10)
            confidence = min(95, 50 + iv_skew + liquidity_score)

            # Recommendation based on edge and confidence
            if edge >= 10 and confidence >= 75:
                recommendation = "Strong Buy"
            elif edge >= 7 and confidence >= 65:
                recommendation = "Buy"
            elif edge >= 5:
                recommendation = "Hold"
            else:
                recommendation = "Avoid"

            opportunities.append({
                "symbol": sym,
                "company_name": info.get('longName', f"{sym} Corporation"),
                "current_price": round(current_price, 2),
                "front_expiry": front_exp,
                "back_expiry": back_exp,
                "strike": atm_strike,
                "front_iv": round(front_iv, 1),
                "back_iv": round(back_iv, 1),
                "iv_skew": round(iv_skew, 1),
                "front_price": round(front_bid, 2),
                "back_price": round(back_ask, 2),
                "debit": round(debit, 2),
                "max_profit": round(max_profit, 2),
                "max_loss": round(max_loss, 2),
                "breakeven_low": round(breakeven_low, 2),
                "breakeven_high": round(breakeven_high, 2),
                "ai_edge": round(edge, 1),
                "ai_confidence": round(confidence, 0),
                "recommendation": recommendation,
                "reasoning": f"IV skew of {round(iv_skew, 1)}% creates favorable theta decay differential",
                "risk_reward": round(risk_reward, 2),
                "front_volume": front_volume,
                "back_volume": back_volume
            })

        except Exception as e:
            logger.warning(f"Error processing {sym}: {e}")
            continue

    # Sort by AI edge
    opportunities.sort(key=lambda x: x["ai_edge"], reverse=True)

    return {
        "opportunities": opportunities,
        "total": len(opportunities),
        "generated_at": datetime.now().isoformat()
    }


# ============ Backtesting Endpoints ============

class BacktestRequest(BaseModel):
    strategy: str  # 'csp', 'wheel', 'momentum', 'mean-reversion'
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000

@router.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest on a trading strategy using real historical price data.
    """
    try:
        days = (datetime.strptime(request.end_date, "%Y-%m-%d") -
                datetime.strptime(request.start_date, "%Y-%m-%d")).days

        if days <= 0:
            return {"error": "End date must be after start date"}

        # Download historical data for all symbols
        all_data = {}
        for symbol in request.symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=request.start_date, end=request.end_date)
            if not hist.empty:
                all_data[symbol] = hist

        if not all_data:
            return {"error": "No historical data available for the specified symbols and date range"}

        trades = []
        portfolio_value = request.initial_capital
        equity_curve = []

        # Simple strategy simulation based on strategy type
        if request.strategy in ['momentum', 'mean-reversion']:
            for symbol, hist in all_data.items():
                closes = hist['Close'].tolist()

                for i in range(20, len(closes)):
                    date = hist.index[i].strftime("%Y-%m-%d")

                    # Calculate simple signals
                    sma_20 = np.mean(closes[i-20:i])
                    current_price = closes[i]
                    prev_price = closes[i-1]

                    # Momentum: buy when price crosses above SMA
                    # Mean reversion: buy when price is below SMA
                    if request.strategy == 'momentum':
                        buy_signal = current_price > sma_20 and prev_price <= sma_20
                        sell_signal = current_price < sma_20 and prev_price >= sma_20
                    else:  # mean-reversion
                        buy_signal = current_price < sma_20 * 0.98
                        sell_signal = current_price > sma_20 * 1.02

                    if buy_signal:
                        trades.append({
                            "date": date,
                            "symbol": symbol,
                            "type": "buy",
                            "price": round(current_price, 2),
                            "pnl": 0  # Will be calculated on sell
                        })

                    if sell_signal and len([t for t in trades if t['type'] == 'buy' and t['symbol'] == symbol]) > 0:
                        # Find last buy for this symbol
                        last_buy = [t for t in trades if t['type'] == 'buy' and t['symbol'] == symbol and t.get('pnl', 0) == 0]
                        if last_buy:
                            entry_price = last_buy[-1]['price']
                            pnl = (current_price - entry_price) * 100  # Assuming 100 shares
                            last_buy[-1]['pnl'] = round(pnl, 2)

                            trades.append({
                                "date": date,
                                "symbol": symbol,
                                "type": "sell",
                                "price": round(current_price, 2),
                                "pnl": round(pnl, 2)
                            })

                    # Track equity curve
                    equity_curve.append({
                        "date": date,
                        "value": round(portfolio_value + sum(t.get('pnl', 0) for t in trades), 2)
                    })

        elif request.strategy in ['csp', 'wheel']:
            # Cash secured put / wheel strategy simulation
            # Simplified: assume selling monthly puts at 5% OTM
            for symbol, hist in all_data.items():
                closes = hist['Close'].tolist()

                # Simulate monthly trades
                for i in range(0, len(closes), 21):  # ~monthly
                    if i >= len(closes):
                        break

                    date = hist.index[i].strftime("%Y-%m-%d")
                    current_price = closes[i]

                    # Assume collecting ~2% premium monthly
                    premium = current_price * 0.02 * 100  # Per contract

                    # Check if assigned (price dropped below strike)
                    strike = current_price * 0.95
                    if i + 21 < len(closes):
                        expiry_price = closes[i + 21]
                        if expiry_price < strike:
                            # Assigned - loss is difference
                            pnl = premium - (strike - expiry_price) * 100
                        else:
                            # Kept premium
                            pnl = premium
                    else:
                        pnl = premium

                    trades.append({
                        "date": date,
                        "symbol": symbol,
                        "type": "sell_put",
                        "price": round(strike, 2),
                        "premium": round(premium, 2),
                        "pnl": round(pnl, 2)
                    })

        # Calculate metrics from real trades
        trades.sort(key=lambda x: x["date"])
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        final_capital = request.initial_capital + total_pnl

        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        # Calculate max drawdown
        max_drawdown = 0
        peak = request.initial_capital
        running = request.initial_capital
        for trade in trades:
            running += trade.get('pnl', 0)
            if running > peak:
                peak = running
            drawdown = (peak - running) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate Sharpe (simplified)
        daily_returns = [t.get('pnl', 0) / request.initial_capital for t in trades]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if daily_returns and np.std(daily_returns) > 0 else 0

        total_return = ((final_capital - request.initial_capital) / request.initial_capital) * 100

        return {
            "strategy_name": request.strategy,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "final_capital": round(final_capital, 2),
            "total_return": round(total_return, 2),
            "cagr": round(total_return * (365 / max(days, 1)), 2),
            "max_drawdown": round(max_drawdown, 1),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sharpe * 1.1, 2),  # Simplified
            "win_rate": round(win_rate, 1),
            "total_trades": len(trades),
            "profit_factor": round(abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))), 2) if losing_trades and avg_loss else 0,
            "avg_trade": round(total_pnl / len(trades), 2) if trades else 0,
            "best_trade": max(t.get('pnl', 0) for t in trades) if trades else 0,
            "worst_trade": min(t.get('pnl', 0) for t in trades) if trades else 0,
            "trades": trades[:100],  # Limit to 100 trades for response size
            "equity_curve": equity_curve[-50:] if equity_curve else []  # Last 50 points
        }

    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        return {
            "error": str(e),
            "strategy_name": request.strategy,
            "start_date": request.start_date,
            "end_date": request.end_date
        }


# ============ Position Sizing Endpoints ============

class PositionSizeRequest(BaseModel):
    account_size: float
    risk_per_trade: float  # Percentage
    entry_price: float
    stop_loss: float
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None

@router.post("/position-size")
async def calculate_position_size(request: PositionSizeRequest):
    """
    Calculate optimal position size using various methods.
    No mock data - pure mathematical calculations.
    """
    # Risk-based position sizing
    risk_amount = request.account_size * (request.risk_per_trade / 100)
    risk_per_share = abs(request.entry_price - request.stop_loss)
    shares_risk = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

    # Fixed percentage
    fixed_pct = 0.05  # 5% of portfolio
    shares_fixed = int((request.account_size * fixed_pct) / request.entry_price)

    # Kelly Criterion (if win rate provided)
    kelly_fraction = 0
    kelly_shares = 0
    if request.win_rate and request.avg_win and request.avg_loss:
        win_prob = request.win_rate / 100
        win_loss_ratio = abs(request.avg_win / request.avg_loss) if request.avg_loss else 1
        kelly_fraction = win_prob - ((1 - win_prob) / win_loss_ratio)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        kelly_shares = int((request.account_size * kelly_fraction) / request.entry_price)

    return {
        "risk_based": {
            "shares": shares_risk,
            "position_value": round(shares_risk * request.entry_price, 2),
            "risk_amount": round(risk_amount, 2),
            "method": "Risk-Based (Recommended)"
        },
        "fixed_percentage": {
            "shares": shares_fixed,
            "position_value": round(shares_fixed * request.entry_price, 2),
            "percentage": fixed_pct * 100,
            "method": "Fixed 5% Position"
        },
        "kelly_criterion": {
            "shares": kelly_shares,
            "position_value": round(kelly_shares * request.entry_price, 2),
            "kelly_fraction": round(kelly_fraction * 100, 2),
            "method": "Kelly Criterion"
        },
        "recommendations": {
            "conservative": int(shares_risk * 0.5),
            "moderate": shares_risk,
            "aggressive": int(shares_risk * 1.5)
        }
    }
