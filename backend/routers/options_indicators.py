"""
Options Indicators API Router
IVR, IVP, Expected Move, Greeks, Strategy Recommendations
NO MOCK DATA - All endpoints use real market data from yfinance
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime, timedelta
import logging
import yfinance as yf
import pandas as pd
import numpy as np

# Import existing Options Indicators
from src.options_indicators import OptionsIndicators
from backend.infrastructure.errors import safe_internal_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/options-indicators", tags=["options-indicators"])


def get_iv_history(symbol: str, days: int = 252) -> pd.Series:
    """
    Get IV history approximation from historical volatility
    Note: True IV requires options data; this uses HV as proxy
    """
    ticker = yf.Ticker(symbol.upper())
    hist = ticker.history(period="1y")

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data available for {symbol}")

    # Calculate rolling 20-day HV as IV proxy
    returns = hist['Close'].pct_change()
    hv = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized

    return hv.dropna()


@router.get("/ivr/{symbol}")
async def get_iv_rank(symbol: str):
    """
    Get Implied Volatility Rank (IVR) for a symbol

    IVR = (Current IV - Min IV) / (Max IV - Min IV) × 100

    Interpretation:
    - IVR > 50: High IV - Good for selling premium
    - IVR < 50: Low IV - Good for buying premium
    - IVR > 80: Very high - Excellent for credit strategies
    - IVR < 20: Very low - Excellent for debit strategies
    """
    try:
        ticker = yf.Ticker(symbol.upper())

        # Get options data for current IV
        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        # Get ATM options IV
        opt_chain = ticker.option_chain(expirations[0])
        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        if current_price == 0:
            raise HTTPException(status_code=404, detail=f"Unable to get price for {symbol}")

        # Find ATM strike
        calls = opt_chain.calls
        if calls.empty:
            raise HTTPException(status_code=404, detail=f"No call options for {symbol}")

        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        current_iv = float(atm_call['impliedVolatility'].values[0]) if len(atm_call) > 0 else 0.25

        # Get IV history (using HV as proxy)
        iv_history = get_iv_history(symbol)

        # Calculate IVR
        options_calc = OptionsIndicators()
        ivr_result = options_calc.implied_volatility_rank(current_iv, iv_history, lookback=252)

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'timestamp': datetime.now().isoformat(),

            'ivr': {
                'value': round(ivr_result['ivr'], 1),
                'current_iv': round(ivr_result['current_iv'] * 100, 1),
                'iv_min_52w': round(ivr_result['iv_min'] * 100, 1),
                'iv_max_52w': round(ivr_result['iv_max'] * 100, 1),
                'interpretation': ivr_result['interpretation'],
                'strategy': ivr_result['strategy'],
                'recommendation': ivr_result['recommendation']
            },

            'trading_guidance': {
                'sell_premium': ivr_result['ivr'] > 50,
                'buy_premium': ivr_result['ivr'] < 50,
                'optimal_strategies': get_ivr_strategies(ivr_result['ivr'])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting IVR for {symbol}: {e}")
        safe_internal_error(e, "get IV rank")


def get_ivr_strategies(ivr: float) -> list:
    """Get optimal strategies based on IVR"""
    if ivr > 80:
        return ['Iron Condor', 'Short Strangle', 'Credit Spreads', 'Covered Calls']
    elif ivr > 50:
        return ['Bull Put Spread', 'Bear Call Spread', 'Cash-Secured Puts']
    elif ivr < 20:
        return ['Long Straddle', 'Long Strangle', 'Debit Spreads', 'Long Calls/Puts']
    else:
        return ['Bull Call Spread', 'Bear Put Spread', 'Calendar Spreads']


@router.get("/ivp/{symbol}")
async def get_iv_percentile(symbol: str):
    """
    Get Implied Volatility Percentile (IVP) for a symbol

    IVP = Percentage of days where IV was below current IV

    More accurate than IVR for trading decisions
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        opt_chain = ticker.option_chain(expirations[0])
        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        calls = opt_chain.calls
        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        current_iv = float(atm_call['impliedVolatility'].values[0]) if len(atm_call) > 0 else 0.25

        iv_history = get_iv_history(symbol)

        options_calc = OptionsIndicators()
        ivp_result = options_calc.implied_volatility_percentile(current_iv, iv_history, lookback=252)

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'timestamp': datetime.now().isoformat(),

            'ivp': {
                'value': round(ivp_result['ivp'], 1),
                'current_iv': round(ivp_result['current_iv'] * 100, 1),
                'interpretation': ivp_result['interpretation'],
                'recommendation': ivp_result['recommendation']
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting IVP for {symbol}: {e}")
        safe_internal_error(e, "get IV percentile")


@router.get("/expected-move/{symbol}")
async def get_expected_move(symbol: str, dte: Optional[int] = None):
    """
    Get Expected Move for a symbol

    Expected Move = Price × IV × sqrt(DTE / 365)

    Shows the expected price range for a given timeframe:
    - 1 Standard Deviation (68% probability)
    - 2 Standard Deviations (95% probability)
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        opt_chain = ticker.option_chain(expirations[0])
        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        calls = opt_chain.calls
        puts = opt_chain.puts

        # Calculate average IV from ATM options
        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]

        call_iv = float(atm_call['impliedVolatility'].values[0]) if len(atm_call) > 0 else 0.25
        put_iv = float(atm_put['impliedVolatility'].values[0]) if len(atm_put) > 0 else 0.25
        avg_iv = (call_iv + put_iv) / 2

        # Calculate DTE if not provided
        if dte is None:
            exp_date = datetime.strptime(expirations[0], '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days
            dte = max(1, dte)

        options_calc = OptionsIndicators()

        # 1 Standard Deviation (68%)
        em_1sd = options_calc.expected_move(current_price, avg_iv, dte, confidence=0.68)

        # 2 Standard Deviations (95%)
        em_2sd = options_calc.expected_move(current_price, avg_iv, dte, confidence=0.95)

        # Weekly expected move
        em_weekly = options_calc.expected_move(current_price, avg_iv, 7, confidence=0.68)

        # Monthly expected move
        em_monthly = options_calc.expected_move(current_price, avg_iv, 30, confidence=0.68)

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'iv': round(avg_iv * 100, 1),
            'timestamp': datetime.now().isoformat(),

            'expected_move': {
                'dte': dte,
                '1_std_dev': {
                    'move': round(em_1sd['expected_move'], 2),
                    'move_pct': round(em_1sd['move_pct'], 2),
                    'upper_bound': round(em_1sd['upper_bound'], 2),
                    'lower_bound': round(em_1sd['lower_bound'], 2),
                    'probability': 68
                },
                '2_std_dev': {
                    'move': round(em_2sd['expected_move'], 2),
                    'move_pct': round(em_2sd['move_pct'], 2),
                    'upper_bound': round(em_2sd['upper_bound'], 2),
                    'lower_bound': round(em_2sd['lower_bound'], 2),
                    'probability': 95
                }
            },

            'timeframes': {
                'weekly': {
                    'move': round(em_weekly['expected_move'], 2),
                    'move_pct': round(em_weekly['move_pct'], 2),
                    'range': f"${round(em_weekly['lower_bound'], 2)} - ${round(em_weekly['upper_bound'], 2)}"
                },
                'monthly': {
                    'move': round(em_monthly['expected_move'], 2),
                    'move_pct': round(em_monthly['move_pct'], 2),
                    'range': f"${round(em_monthly['lower_bound'], 2)} - ${round(em_monthly['upper_bound'], 2)}"
                }
            },

            'trading_guidance': {
                'strike_selection': {
                    'safe_put_strike': round(em_1sd['lower_bound'] * 0.95, 2),
                    'safe_call_strike': round(em_1sd['upper_bound'] * 1.05, 2),
                    'aggressive_put_strike': round(em_1sd['lower_bound'], 2),
                    'aggressive_call_strike': round(em_1sd['upper_bound'], 2)
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Expected Move for {symbol}: {e}")
        safe_internal_error(e, "get expected move")


@router.get("/greeks/{symbol}")
async def get_option_greeks(
    symbol: str,
    strike: Optional[float] = None,
    expiration: Optional[str] = None,
    option_type: str = "call"
):
    """
    Get Greeks for a specific option

    Greeks:
    - Delta: Rate of change vs stock price
    - Gamma: Rate of change of delta
    - Theta: Time decay per day
    - Vega: Sensitivity to IV changes
    - Rho: Sensitivity to interest rates
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        # Use provided expiration or nearest
        exp_date = expiration if expiration and expiration in expirations else expirations[0]
        opt_chain = ticker.option_chain(exp_date)

        # Get appropriate chain
        chain = opt_chain.calls if option_type.lower() == 'call' else opt_chain.puts

        # Find strike (use provided or ATM)
        if strike:
            option = chain[chain['strike'] == strike]
        else:
            option = chain.iloc[(chain['strike'] - current_price).abs().argsort()[:1]]
            strike = float(option['strike'].values[0])

        if option.empty:
            raise HTTPException(status_code=404, detail=f"No option found for strike {strike}")

        # Get option data
        iv = float(option['impliedVolatility'].values[0]) if len(option) > 0 else 0.25

        # Calculate DTE
        dte = (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days
        dte = max(1, dte)

        # Calculate Greeks
        options_calc = OptionsIndicators()
        greeks = options_calc.calculate_greeks(
            spot=current_price,
            strike=strike,
            rate=0.05,  # Assume 5% risk-free rate
            dte=dte,
            iv=iv,
            option_type=option_type
        )

        # Option price from chain
        bid = float(option['bid'].values[0] or 0)
        ask = float(option['ask'].values[0] or 0)
        mid = (bid + ask) / 2

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'timestamp': datetime.now().isoformat(),

            'option': {
                'strike': strike,
                'expiration': exp_date,
                'type': option_type,
                'dte': dte,
                'bid': round(bid, 2),
                'ask': round(ask, 2),
                'mid': round(mid, 2),
                'iv': round(iv * 100, 1)
            },

            'greeks': {
                'delta': round(greeks['delta'], 4) if greeks.get('delta') else None,
                'gamma': round(greeks['gamma'], 6) if greeks.get('gamma') else None,
                'theta': round(greeks['theta'], 4) if greeks.get('theta') else None,
                'vega': round(greeks['vega'], 4) if greeks.get('vega') else None,
                'rho': round(greeks['rho'], 4) if greeks.get('rho') else None,
                'theoretical_price': round(greeks['price'], 2) if greeks.get('price') else None
            },

            'interpretation': {
                'delta': greeks.get('delta_interpretation', ''),
                'gamma': greeks.get('gamma_interpretation', ''),
                'theta': greeks.get('theta_interpretation', ''),
                'vega': greeks.get('vega_interpretation', '')
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Greeks for {symbol}: {e}")
        safe_internal_error(e, "get option greeks")


@router.get("/put-call-ratio/{symbol}")
async def get_put_call_ratio(symbol: str):
    """
    Get Put/Call Ratio analysis

    PCR > 1.0: More puts than calls (bearish sentiment)
    PCR < 1.0: More calls than puts (bullish sentiment)
    PCR > 1.5: Extremely bearish (contrarian bullish)
    PCR < 0.5: Extremely bullish (contrarian bearish)
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        # Aggregate volume and OI across expirations
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0

        for exp in expirations[:4]:  # First 4 expirations
            try:
                chain = ticker.option_chain(exp)
                total_call_volume += chain.calls['volume'].sum() if not chain.calls.empty else 0
                total_put_volume += chain.puts['volume'].sum() if not chain.puts.empty else 0
                total_call_oi += chain.calls['openInterest'].sum() if not chain.calls.empty else 0
                total_put_oi += chain.puts['openInterest'].sum() if not chain.puts.empty else 0
            except Exception:
                continue

        options_calc = OptionsIndicators()

        # Volume-based PCR
        volume_pcr = options_calc.put_call_ratio(total_put_volume, total_call_volume)

        # OI-based PCR
        oi_pcr = options_calc.put_call_ratio(total_put_oi, total_call_oi)

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'timestamp': datetime.now().isoformat(),

            'volume_pcr': {
                'value': volume_pcr['pcr'],
                'put_volume': int(total_put_volume),
                'call_volume': int(total_call_volume),
                'sentiment': volume_pcr['sentiment'],
                'interpretation': volume_pcr['interpretation']
            },

            'oi_pcr': {
                'value': oi_pcr['pcr'],
                'put_oi': int(total_put_oi),
                'call_oi': int(total_call_oi),
                'sentiment': oi_pcr['sentiment'],
                'interpretation': oi_pcr['interpretation']
            },

            'contrarian_view': volume_pcr['contrarian_view'],

            'trading_guidance': {
                'sentiment_score': get_sentiment_score(volume_pcr['pcr']),
                'recommendation': get_pcr_recommendation(volume_pcr['pcr'])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting PCR for {symbol}: {e}")
        safe_internal_error(e, "get put/call ratio")


def get_sentiment_score(pcr: float) -> int:
    """Convert PCR to sentiment score (0-100, 100 = bullish)"""
    if pcr is None:
        return 50
    if pcr > 1.5:
        return 20  # Very bearish
    elif pcr > 1.0:
        return 35  # Bearish
    elif pcr > 0.7:
        return 50  # Neutral
    elif pcr > 0.5:
        return 65  # Bullish
    else:
        return 80  # Very bullish


def get_pcr_recommendation(pcr: float) -> str:
    """Get trading recommendation from PCR"""
    if pcr is None:
        return "No PCR data available"
    if pcr > 1.5:
        return "Extreme bearish sentiment - Consider contrarian bullish plays"
    elif pcr > 1.0:
        return "Bearish sentiment - Caution on bullish positions"
    elif pcr < 0.5:
        return "Extreme bullish sentiment - Consider contrarian bearish plays"
    elif pcr < 0.7:
        return "Bullish sentiment - Momentum favors bulls"
    else:
        return "Neutral sentiment - No strong directional bias"


@router.get("/strategy-recommendation/{symbol}")
async def get_strategy_recommendation(symbol: str, trend: str = "auto"):
    """
    Get options strategy recommendations based on IV and trend

    Factors considered:
    - IV Rank (high/low)
    - Market trend (bullish/bearish/neutral)
    - Expected move
    - Put/Call ratio
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        # Get IV data
        opt_chain = ticker.option_chain(expirations[0])
        calls = opt_chain.calls
        puts = opt_chain.puts

        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        call_iv = float(atm_call['impliedVolatility'].values[0]) if len(atm_call) > 0 else 0.25

        iv_history = get_iv_history(symbol)
        options_calc = OptionsIndicators()
        ivr_result = options_calc.implied_volatility_rank(call_iv, iv_history)

        # Auto-detect trend from price action
        if trend == "auto":
            hist = ticker.history(period="1mo")
            if not hist.empty:
                sma_20 = hist['Close'].tail(20).mean()
                if current_price > sma_20 * 1.02:
                    detected_trend = "BULLISH"
                elif current_price < sma_20 * 0.98:
                    detected_trend = "BEARISH"
                else:
                    detected_trend = "NEUTRAL"
            else:
                detected_trend = "NEUTRAL"
        else:
            detected_trend = trend.upper()

        # Calculate expected move
        dte = (datetime.strptime(expirations[0], '%Y-%m-%d') - datetime.now()).days
        dte = max(1, dte)
        em = options_calc.expected_move(current_price, call_iv, dte)

        # Get strategy recommendation
        recommendation = options_calc.option_strategy_recommendation(
            ivr=ivr_result['ivr'],
            trend=detected_trend,
            expected_move=em
        )

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'timestamp': datetime.now().isoformat(),

            'market_conditions': {
                'ivr': round(ivr_result['ivr'], 1),
                'iv_interpretation': ivr_result['interpretation'],
                'trend': detected_trend,
                'expected_move_pct': round(em['move_pct'], 2)
            },

            'strategies': [
                {
                    'name': s['strategy'],
                    'reason': s['reason'],
                    'profit_potential': s['profit_potential'],
                    'risk': s['risk'],
                    'ideal_for': s['ideal_for']
                } for s in recommendation['strategies']
            ],

            'top_recommendation': recommendation['top_recommendation'],

            'general_guidance': {
                'premium_strategy': 'SELL' if ivr_result['ivr'] > 50 else 'BUY',
                'direction': detected_trend,
                'risk_level': 'DEFINED' if ivr_result['ivr'] > 70 or ivr_result['ivr'] < 30 else 'MODERATE'
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy recommendation for {symbol}: {e}")
        safe_internal_error(e, "get strategy recommendation")


@router.get("/comprehensive/{symbol}")
async def get_comprehensive_options_analysis(symbol: str):
    """
    Get comprehensive options analysis including IVR, IVP, Expected Move, PCR, and recommendations
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        expirations = ticker.options
        if not expirations:
            raise HTTPException(status_code=404, detail=f"No options available for {symbol}")

        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        opt_chain = ticker.option_chain(expirations[0])
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Get ATM IV
        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
        call_iv = float(atm_call['impliedVolatility'].values[0]) if len(atm_call) > 0 else 0.25
        put_iv = float(atm_put['impliedVolatility'].values[0]) if len(atm_put) > 0 else 0.25
        avg_iv = (call_iv + put_iv) / 2

        # Get IV history
        iv_history = get_iv_history(symbol)

        options_calc = OptionsIndicators()

        # IVR
        ivr = options_calc.implied_volatility_rank(avg_iv, iv_history)

        # IVP
        ivp = options_calc.implied_volatility_percentile(avg_iv, iv_history)

        # Expected Move
        dte = (datetime.strptime(expirations[0], '%Y-%m-%d') - datetime.now()).days
        dte = max(1, dte)
        em = options_calc.expected_move(current_price, avg_iv, dte)

        # PCR
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        pcr = options_calc.put_call_ratio(total_put_vol, total_call_vol)

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'timestamp': datetime.now().isoformat(),

            'iv_analysis': {
                'current_iv': round(avg_iv * 100, 1),
                'ivr': round(ivr['ivr'], 1),
                'ivp': round(ivp['ivp'], 1),
                'interpretation': ivr['interpretation'],
                'strategy_bias': ivr['strategy']
            },

            'expected_move': {
                'dte': dte,
                'move_dollars': round(em['expected_move'], 2),
                'move_pct': round(em['move_pct'], 2),
                'upper': round(em['upper_bound'], 2),
                'lower': round(em['lower_bound'], 2)
            },

            'sentiment': {
                'pcr': round(pcr['pcr'], 2) if pcr['pcr'] else None,
                'sentiment': pcr['sentiment'],
                'interpretation': pcr['interpretation']
            },

            'available_expirations': expirations[:8]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comprehensive analysis for {symbol}: {e}")
        safe_internal_error(e, "get comprehensive options analysis")
