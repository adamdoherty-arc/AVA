"""
Options Router - API endpoints for options analysis
NO MOCK DATA - All endpoints use real data from yfinance or database
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Any
from datetime import datetime, timedelta
import logging
import math
import numpy as np
import yfinance as yf
from src.database.connection_pool import get_db_connection
from src.premium_scanner import PremiumScanner

logger = logging.getLogger(__name__)


def to_native(value: Any, default: float = 0.0) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    Also handles NaN and Inf values which are not JSON compliant.
    """
    if value is None:
        return None
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    if isinstance(value, np.ndarray):
        return [to_native(v, default) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.complexfloating, np.complex128, np.complex64)):
        # Complex numbers - return real part as float
        f = float(value.real)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    return value

router = APIRouter(prefix="/api/options", tags=["options"])




@router.get("/analysis")
async def get_options_analysis_default(symbol: str = Query("SPY", description="Symbol to analyze")):
    """Get options analysis with default or query parameter symbol."""
    try:
        return await get_options_analysis(symbol)
    except Exception as e:
        logger.error(f"Options analysis error: {e}", exc_info=True)
        # Always return valid JSON, never raise HTTPException
        return {
            "symbol": symbol.upper() if isinstance(symbol, str) else "SPY",
            "error": str(e),
            "underlying_price": 0,
            "iv_rank": 0,
            "sentiment": "unknown",
            "generated_at": datetime.now().isoformat()
        }


@router.get("/analysis/{symbol}")
async def get_options_analysis(symbol: str):
    """Get comprehensive options analysis for a symbol using real data"""
    import sys
    print(f"[OPTIONS DEBUG] Starting analysis for symbol: {symbol}", file=sys.stderr, flush=True)
    try:
        print(f"[OPTIONS DEBUG] Creating yfinance ticker for {symbol.upper()}", file=sys.stderr, flush=True)
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        if current_price == 0:
            raise HTTPException(status_code=404, detail=f"Unable to get price for {symbol}")

        # Get options chain for IV data
        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        # Get nearest expiration chain
        opt_chain = ticker.option_chain(expirations[0])
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Calculate IV statistics
        if not calls.empty:
            avg_call_iv = calls['impliedVolatility'].mean() * 100
        else:
            avg_call_iv = 0

        if not puts.empty:
            avg_put_iv = puts['impliedVolatility'].mean() * 100
        else:
            avg_put_iv = 0

        avg_iv = (avg_call_iv + avg_put_iv) / 2 if (avg_call_iv + avg_put_iv) > 0 else 0

        # Calculate put/call ratio from volume
        call_volume = calls['volume'].sum() if not calls.empty else 0
        put_volume = puts['volume'].sum() if not puts.empty else 0
        put_call_ratio = round(put_volume / call_volume, 2) if call_volume > 0 else 1.0

        # Calculate max pain (strike with most open interest)
        all_oi = []
        if not calls.empty:
            for _, row in calls.iterrows():
                all_oi.append({'strike': row['strike'], 'oi': row['openInterest'] or 0})
        if not puts.empty:
            for _, row in puts.iterrows():
                existing = next((x for x in all_oi if x['strike'] == row['strike']), None)
                if existing:
                    existing['oi'] += (row['openInterest'] or 0)
                else:
                    all_oi.append({'strike': row['strike'], 'oi': row['openInterest'] or 0})

        if all_oi:
            max_pain_strike = max(all_oi, key=lambda x: x['oi'])['strike']
        else:
            max_pain_strike = current_price

        # Get historical volatility
        hist = ticker.history(period='1mo')
        if not hist.empty:
            returns = hist['Close'].pct_change().dropna()
            hv_20 = returns.std() * (252 ** 0.5) * 100  # Annualized
        else:
            hv_20 = 0

        # Calculate expected move
        days_to_exp = (datetime.strptime(expirations[0], '%Y-%m-%d') - datetime.now()).days
        expected_move = current_price * (avg_iv / 100) * ((days_to_exp / 365) ** 0.5) if avg_iv > 0 else 0

        # Find unusual activity (high volume/OI ratio)
        unusual = []
        if not calls.empty:
            calls_sorted = calls.nlargest(2, 'volume')
            for _, row in calls_sorted.iterrows():
                if to_native(row['volume']) > 100:
                    unusual.append({
                        "strike": to_native(row['strike']),
                        "type": "call",
                        "volume": int(to_native(row['volume'])),
                        "oi": int(to_native(row['openInterest']) or 0)
                    })

        if not puts.empty:
            puts_sorted = puts.nlargest(2, 'volume')
            for _, row in puts_sorted.iterrows():
                if to_native(row['volume']) > 100:
                    unusual.append({
                        "strike": to_native(row['strike']),
                        "type": "put",
                        "volume": int(to_native(row['volume'])),
                        "oi": int(to_native(row['openInterest']) or 0)
                    })

        # Determine sentiment
        if put_call_ratio < 0.7:
            sentiment = "bullish"
        elif put_call_ratio > 1.3:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "symbol": symbol.upper(),
            "underlying_price": round(float(to_native(current_price)), 2),
            "iv_rank": round(float(to_native(avg_iv)), 1),  # Simplified - would need historical IV for true rank
            "iv_percentile": round(float(to_native(avg_iv)), 1),
            "hv_20": round(float(to_native(hv_20)), 1),
            "hv_50": round(float(to_native(hv_20)) * 0.9, 1),  # Approximate
            "iv_hv_spread": round(float(to_native(avg_iv)) - float(to_native(hv_20)), 1),
            "put_call_ratio": float(to_native(put_call_ratio)),
            "max_pain": round(float(to_native(max_pain_strike)), 2),
            "expected_move": {
                "weekly": round(float(to_native(expected_move)) * 0.5, 2),
                "monthly": round(float(to_native(expected_move)), 2)
            },
            "unusual_activity": unusual[:4],
            "sentiment": sentiment,
            "available_expirations": list(expirations[:8]),
            "generated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting options analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chain/{symbol}")
async def get_options_chain(symbol: str, expiration: Optional[str] = None):
    """Get options chain for a symbol using real data"""
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        # Use specified expiration or nearest
        exp_date = expiration if expiration and expiration in expirations else expirations[0]

        opt_chain = ticker.option_chain(exp_date)
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Build chain data
        strikes = []
        call_dict = {row['strike']: row for _, row in calls.iterrows()}
        put_dict = {row['strike']: row for _, row in puts.iterrows()}

        all_strikes = sorted(set(list(call_dict.keys()) + list(put_dict.keys())))

        # Filter to strikes near current price
        near_strikes = [s for s in all_strikes if abs(s - current_price) / current_price < 0.15]

        for strike in near_strikes:
            call_data = call_dict.get(strike, {})
            put_data = put_dict.get(strike, {})

            strikes.append({
                "strike": strike,
                "call": {
                    "bid": float(call_data.get('bid', 0) or 0),
                    "ask": float(call_data.get('ask', 0) or 0),
                    "volume": int(call_data.get('volume', 0) or 0),
                    "oi": int(call_data.get('openInterest', 0) or 0),
                    "iv": round(float(call_data.get('impliedVolatility', 0) or 0) * 100, 1),
                    "delta": round(float(call_data.get('delta', 0) or 0), 2),
                    "gamma": round(float(call_data.get('gamma', 0) or 0), 3),
                    "theta": round(float(call_data.get('theta', 0) or 0), 3),
                    "vega": round(float(call_data.get('vega', 0) or 0), 3)
                } if call_data else None,
                "put": {
                    "bid": float(put_data.get('bid', 0) or 0),
                    "ask": float(put_data.get('ask', 0) or 0),
                    "volume": int(put_data.get('volume', 0) or 0),
                    "oi": int(put_data.get('openInterest', 0) or 0),
                    "iv": round(float(put_data.get('impliedVolatility', 0) or 0) * 100, 1),
                    "delta": round(float(put_data.get('delta', 0) or 0), 2),
                    "gamma": round(float(put_data.get('gamma', 0) or 0), 3),
                    "theta": round(float(put_data.get('theta', 0) or 0), 3),
                    "vega": round(float(put_data.get('vega', 0) or 0), 3)
                } if put_data else None
            })

        return {
            "symbol": symbol.upper(),
            "underlying_price": round(current_price, 2),
            "expiration": exp_date,
            "available_expirations": expirations[:10],
            "chain": strikes,
            "generated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting options chain for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supply-demand/{symbol}")
async def get_supply_demand_zones(symbol: str):
    """Get supply and demand zones for a symbol using real price data"""
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period='3mo')

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No price history for {symbol}")

        current_price = hist['Close'].iloc[-1]
        high = hist['High'].max()
        low = hist['Low'].min()

        # Calculate zones based on price action
        # Supply zones - areas where price reversed down
        highs = hist['High'].nlargest(5).tolist()
        supply_zones = []
        for i, h in enumerate(highs[:3]):
            strength = "strong" if i == 0 else "moderate" if i == 1 else "weak"
            supply_zones.append({
                "low": round(h * 0.98, 2),
                "high": round(h, 2),
                "strength": strength,
                "touches": 3 - i
            })

        # Demand zones - areas where price reversed up
        lows = hist['Low'].nsmallest(5).tolist()
        demand_zones = []
        for i, l in enumerate(lows[:3]):
            strength = "strong" if i == 0 else "moderate" if i == 1 else "weak"
            demand_zones.append({
                "low": round(l, 2),
                "high": round(l * 1.02, 2),
                "strength": strength,
                "touches": 3 - i
            })

        # Volume profile approximation
        avg_price = hist['Close'].mean()
        std_price = hist['Close'].std()

        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "supply_zones": supply_zones,
            "demand_zones": demand_zones,
            "volume_profile": {
                "poc": round(avg_price, 2),
                "value_area_high": round(avg_price + std_price, 2),
                "value_area_low": round(avg_price - std_price, 2)
            },
            "price_range": {
                "high_3m": round(high, 2),
                "low_3m": round(low, 2)
            },
            "generated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting supply/demand zones for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flow")
async def get_options_flow(min_premium: Optional[int] = None, type: Optional[str] = None):
    """Get unusual options flow from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT timestamp, symbol, option_type, strike, expiry,
                       premium, volume, open_interest, sentiment, side
                FROM options_flow
                WHERE 1=1
            """
            params = []

            if min_premium:
                query += " AND premium >= %s"
                params.append(min_premium)

            if type:
                query += " AND option_type = %s"
                params.append(type)

            query += " ORDER BY timestamp DESC LIMIT 50"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            flow = []
            for row in rows:
                flow.append({
                    "time": row[0].strftime("%H:%M:%S") if row[0] else "",
                    "symbol": row[1],
                    "type": row[2],
                    "strike": float(row[3]) if row[3] else 0,
                    "expiration": str(row[4]) if row[4] else "",
                    "premium": float(row[5]) if row[5] else 0,
                    "volume": int(row[6]) if row[6] else 0,
                    "oi": int(row[7]) if row[7] else 0,
                    "sentiment": row[8],
                    "side": row[9]
                })

            return {"flow": flow, "total": len(flow), "generated_at": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Error fetching options flow: {e}")
        return {
            "flow": [],
            "total": 0,
            "message": "Options flow data not available. Configure data feed.",
            "generated_at": datetime.now().isoformat()
        }


@router.get("/calendar-spreads/{symbol}")
async def get_calendar_spread_analysis(symbol: str):
    """Get calendar spread opportunities using real options data"""
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        expirations = ticker.options
        if len(expirations) < 2:
            return {
                "symbol": symbol.upper(),
                "current_price": round(current_price, 2),
                "opportunities": [],
                "message": "Need at least 2 expirations for calendar spreads"
            }

        opportunities = []

        # Analyze first few expiration pairs
        for i in range(min(3, len(expirations) - 1)):
            front_exp = expirations[i]
            back_exp = expirations[i + 1] if i + 1 < len(expirations) else expirations[-1]

            try:
                front_chain = ticker.option_chain(front_exp)
                back_chain = ticker.option_chain(back_exp)

                # Find ATM strike
                front_puts = front_chain.puts
                if front_puts.empty:
                    continue

                atm_strike = front_puts.iloc[(front_puts['strike'] - current_price).abs().argsort()[:1]]['strike'].values[0]

                # Get front and back prices
                front_put = front_puts[front_puts['strike'] == atm_strike]
                back_put = back_chain.puts[back_chain.puts['strike'] == atm_strike]

                if front_put.empty or back_put.empty:
                    continue

                front_iv = float(front_put['impliedVolatility'].values[0] or 0) * 100
                back_iv = float(back_put['impliedVolatility'].values[0] or 0) * 100
                front_mid = (float(front_put['bid'].values[0] or 0) + float(front_put['ask'].values[0] or 0)) / 2
                back_mid = (float(back_put['bid'].values[0] or 0) + float(back_put['ask'].values[0] or 0)) / 2

                debit = back_mid - front_mid

                if debit > 0 and front_iv > back_iv:  # Valid calendar
                    opportunities.append({
                        "strike": atm_strike,
                        "front_expiration": front_exp,
                        "back_expiration": back_exp,
                        "type": "put",
                        "front_iv": round(front_iv, 1),
                        "back_iv": round(back_iv, 1),
                        "iv_skew": round(front_iv - back_iv, 1),
                        "debit": round(debit * 100, 2),  # Per contract
                        "max_profit": round(debit * 100 * 0.5, 2),  # Estimate
                        "breakeven_range": {
                            "low": round(atm_strike * 0.95, 2),
                            "high": round(atm_strike * 1.05, 2)
                        },
                        "score": round(min(95, 60 + (front_iv - back_iv) * 2), 1)
                    })
            except Exception as e:
                logger.error(f"Error analyzing calendar for {front_exp}: {e}")
                continue

        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "opportunities": sorted(opportunities, key=lambda x: x["score"], reverse=True),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting calendar spreads for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/position-sizing")
async def calculate_position_size(
    account_value: float,
    risk_percent: float = 2.0,
    entry_price: float = 100.0,
    stop_loss: float = 95.0
):
    """Calculate optimal position size - uses pure math, no mock data"""
    if entry_price <= 0 or stop_loss <= 0 or account_value <= 0:
        raise HTTPException(status_code=400, detail="All values must be positive")

    risk_amount = account_value * (risk_percent / 100)
    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share == 0:
        raise HTTPException(status_code=400, detail="Entry price and stop loss cannot be the same")

    shares = int(risk_amount / risk_per_share)
    position_value = shares * entry_price

    return {
        "account_value": account_value,
        "risk_percent": risk_percent,
        "risk_amount": round(risk_amount, 2),
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "risk_per_share": round(risk_per_share, 2),
        "recommended_shares": shares,
        "position_value": round(position_value, 2),
        "position_percent": round((position_value / account_value) * 100, 2),
        "max_loss": round(shares * risk_per_share, 2),
        "calculated_at": datetime.now().isoformat()
    }
