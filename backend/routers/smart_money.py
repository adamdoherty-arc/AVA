"""
Smart Money Concepts (ICT) API Router
Order Blocks, Fair Value Gaps, BOS/CHoCH, Liquidity Pools
NO MOCK DATA - All endpoints use real market data from yfinance
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime
import logging
import yfinance as yf
import pandas as pd

# Import existing Smart Money Indicators
from src.smart_money_indicators import SmartMoneyIndicators

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/smart-money", tags=["smart-money"])


def get_ohlcv_data(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance"""
    ticker = yf.Ticker(symbol.upper())
    hist = ticker.history(period=period, interval=interval)

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data available for {symbol}")

    # Normalize column names
    hist.columns = [col.lower() for col in hist.columns]
    return hist


@router.get("/{symbol}")
async def get_smart_money_analysis(
    symbol: str,
    timeframe: str = "1D",
    swing_window: int = 5
):
    """
    Get comprehensive Smart Money Concepts analysis for a symbol

    Returns:
    - Order Blocks (bullish/bearish institutional entry zones)
    - Fair Value Gaps (price imbalances)
    - Market Structure (BOS/CHoCH, trend direction)
    - Liquidity Pools (stop loss clusters)
    """
    try:
        # Map timeframe to yfinance parameters
        tf_map = {
            "1H": ("5d", "1h"),
            "4H": ("1mo", "1h"),
            "1D": ("3mo", "1d"),
            "1W": ("1y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("3mo", "1d"))

        df = get_ohlcv_data(symbol, period, interval)
        current_price = float(df['close'].iloc[-1])

        # Initialize Smart Money Indicators
        smc = SmartMoneyIndicators(swing_window=swing_window)

        # Get all SMC indicators
        indicators = smc.get_all_smc_indicators(df)

        # Process Order Blocks
        order_blocks = []
        for ob in indicators.get('order_blocks', [])[-10:]:  # Last 10
            # Calculate midpoint if not present
            top = float(ob.get('top', 0))
            bottom = float(ob.get('bottom', 0))
            midpoint = float(ob.get('midpoint', (top + bottom) / 2 if top and bottom else 0))

            # Calculate distance from current price
            distance_pct = ((midpoint - current_price) / current_price) * 100 if midpoint else 0

            order_blocks.append({
                'type': ob.get('type', 'UNKNOWN'),
                'top': top,
                'bottom': bottom,
                'midpoint': midpoint,
                'strength': int(ob.get('strength', 50)),
                'mitigated': bool(ob.get('mitigated', False)),
                'distance_pct': round(distance_pct, 2),
                'zone': 'support' if ob.get('type') == 'BULLISH_OB' else 'resistance'
            })

        # Process Fair Value Gaps
        fair_value_gaps = []
        for fvg in indicators.get('fair_value_gaps', [])[-10:]:
            top = float(fvg.get('top', 0))
            bottom = float(fvg.get('bottom', 0))
            midpoint = float(fvg.get('midpoint', (top + bottom) / 2 if top and bottom else 0))
            distance_pct = ((midpoint - current_price) / current_price) * 100 if midpoint else 0

            fair_value_gaps.append({
                'type': fvg.get('type', 'UNKNOWN'),
                'top': top,
                'bottom': bottom,
                'midpoint': midpoint,
                'gap_pct': round(float(fvg.get('gap_pct', 0)), 2),
                'filled': bool(fvg.get('filled', False)),
                'fill_percentage': round(float(fvg.get('fill_percentage', 0)), 1),
                'distance_pct': round(distance_pct, 2)
            })

        # Process Market Structure
        market_structure = indicators.get('market_structure', {'bos': [], 'choch': [], 'current_trend': 'NEUTRAL'})

        # Get recent structure breaks
        recent_bos = market_structure.get('bos', [])[-5:] if market_structure.get('bos') else []
        recent_choch = market_structure.get('choch', [])[-3:] if market_structure.get('choch') else []

        # Process Liquidity Pools
        liquidity_pools = []
        for pool in indicators.get('liquidity_pools', [])[:10]:
            price = float(pool.get('price', 0))
            distance_pct = ((price - current_price) / current_price) * 100 if price else 0

            liquidity_pools.append({
                'type': pool.get('type', 'UNKNOWN'),
                'price': price,
                'touches': int(pool.get('touches', 0)),
                'strength': int(pool.get('strength', 0)),
                'swept': bool(pool.get('swept', False)),
                'distance_pct': round(distance_pct, 2)
            })

        # Find nearest support/resistance from SMC
        bullish_obs = [ob for ob in order_blocks if ob['type'] == 'BULLISH_OB' and ob['midpoint'] < current_price]
        bearish_obs = [ob for ob in order_blocks if ob['type'] == 'BEARISH_OB' and ob['midpoint'] > current_price]

        bullish_fvgs = [fvg for fvg in fair_value_gaps if fvg['type'] == 'BULLISH_FVG' and not fvg['filled']]
        bearish_fvgs = [fvg for fvg in fair_value_gaps if fvg['type'] == 'BEARISH_FVG' and not fvg['filled']]

        # Nearest support (highest bullish OB below price)
        nearest_support = max(bullish_obs, key=lambda x: x['midpoint']) if bullish_obs else None

        # Nearest resistance (lowest bearish OB above price)
        nearest_resistance = min(bearish_obs, key=lambda x: x['midpoint']) if bearish_obs else None

        # Generate trading signals
        signals = []

        # Check if price is at an order block
        for ob in order_blocks:
            if abs(ob['distance_pct']) < 1.0 and not ob['mitigated']:
                signal_type = 'BUY' if ob['type'] == 'BULLISH_OB' else 'SELL'
                signals.append({
                    'type': signal_type,
                    'indicator': 'ORDER_BLOCK',
                    'price': ob['midpoint'],
                    'strength': ob['strength'],
                    'description': f"Price at {ob['type'].replace('_', ' ')} zone"
                })

        # Check if price is near unfilled FVG
        for fvg in fair_value_gaps:
            if abs(fvg['distance_pct']) < 2.0 and not fvg['filled']:
                signal_type = 'BUY' if fvg['type'] == 'BULLISH_FVG' else 'SELL'
                signals.append({
                    'type': signal_type,
                    'indicator': 'FAIR_VALUE_GAP',
                    'price': fvg['midpoint'],
                    'strength': 70,
                    'description': f"Unfilled {fvg['type'].replace('_', ' ')} nearby"
                })

        # Check for recent CHoCH (trend reversal)
        if recent_choch:
            latest_choch = recent_choch[-1]
            signals.append({
                'type': 'BUY' if latest_choch['direction'] == 'BULLISH' else 'SELL',
                'indicator': 'CHOCH',
                'price': latest_choch['price'],
                'strength': 85,
                'description': f"Change of Character - {latest_choch['direction']} reversal"
            })

        # Overall bias based on market structure
        current_trend = market_structure['current_trend']
        if current_trend == 'BULLISH':
            overall_bias = 'BULLISH'
            bias_description = 'Higher highs and higher lows - Uptrend'
        elif current_trend == 'BEARISH':
            overall_bias = 'BEARISH'
            bias_description = 'Lower highs and lower lows - Downtrend'
        else:
            overall_bias = 'NEUTRAL'
            bias_description = 'No clear market structure'

        return {
            'symbol': symbol.upper(),
            'timeframe': timeframe,
            'current_price': round(current_price, 2),
            'timestamp': datetime.now().isoformat(),

            'market_structure': {
                'current_trend': current_trend,
                'bias': overall_bias,
                'description': bias_description,
                'recent_bos': [
                    {
                        'direction': b['direction'],
                        'price': b['price'],
                        'type': 'Break of Structure'
                    } for b in recent_bos
                ],
                'recent_choch': [
                    {
                        'direction': c['direction'],
                        'price': c['price'],
                        'type': 'Change of Character'
                    } for c in recent_choch
                ]
            },

            'order_blocks': order_blocks,
            'fair_value_gaps': fair_value_gaps,
            'liquidity_pools': liquidity_pools,

            'key_levels': {
                'nearest_support': {
                    'price': nearest_support['midpoint'] if nearest_support else None,
                    'type': 'BULLISH_ORDER_BLOCK',
                    'strength': nearest_support['strength'] if nearest_support else 0
                } if nearest_support else None,
                'nearest_resistance': {
                    'price': nearest_resistance['midpoint'] if nearest_resistance else None,
                    'type': 'BEARISH_ORDER_BLOCK',
                    'strength': nearest_resistance['strength'] if nearest_resistance else 0
                } if nearest_resistance else None
            },

            'signals': signals,

            'summary': {
                'total_order_blocks': len(order_blocks),
                'bullish_obs': len([ob for ob in order_blocks if ob['type'] == 'BULLISH_OB']),
                'bearish_obs': len([ob for ob in order_blocks if ob['type'] == 'BEARISH_OB']),
                'unfilled_fvgs': len([fvg for fvg in fair_value_gaps if not fvg['filled']]),
                'liquidity_pools': len(liquidity_pools)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Smart Money analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/order-blocks/{symbol}")
async def get_order_blocks(symbol: str, timeframe: str = "1D"):
    """Get Order Blocks only for a symbol"""
    try:
        tf_map = {
            "1H": ("5d", "1h"),
            "4H": ("1mo", "1h"),
            "1D": ("3mo", "1d"),
            "1W": ("1y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("3mo", "1d"))

        df = get_ohlcv_data(symbol, period, interval)
        current_price = float(df['close'].iloc[-1])

        smc = SmartMoneyIndicators()
        order_blocks = smc.detect_order_blocks(df)

        # Format response
        result = []
        for ob in order_blocks:
            distance_pct = ((ob['midpoint'] - current_price) / current_price) * 100
            result.append({
                'type': ob['type'],
                'top': float(ob['top']),
                'bottom': float(ob['bottom']),
                'midpoint': float(ob['midpoint']),
                'strength': int(ob['strength']),
                'mitigated': bool(ob['mitigated']),
                'distance_pct': round(distance_pct, 2)
            })

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'order_blocks': result,
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Order Blocks for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fvg/{symbol}")
async def get_fair_value_gaps(symbol: str, timeframe: str = "1D"):
    """Get Fair Value Gaps only for a symbol"""
    try:
        tf_map = {
            "1H": ("5d", "1h"),
            "4H": ("1mo", "1h"),
            "1D": ("3mo", "1d"),
            "1W": ("1y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("3mo", "1d"))

        df = get_ohlcv_data(symbol, period, interval)
        current_price = float(df['close'].iloc[-1])

        smc = SmartMoneyIndicators()
        fvgs = smc.detect_fair_value_gaps(df)

        result = []
        for fvg in fvgs:
            distance_pct = ((fvg['midpoint'] - current_price) / current_price) * 100
            result.append({
                'type': fvg['type'],
                'top': float(fvg['top']),
                'bottom': float(fvg['bottom']),
                'gap_pct': round(float(fvg['gap_pct']), 2),
                'filled': bool(fvg['filled']),
                'fill_percentage': round(float(fvg['fill_percentage']), 1),
                'distance_pct': round(distance_pct, 2)
            })

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'fair_value_gaps': result,
            'unfilled_count': len([f for f in result if not f['filled']]),
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting FVGs for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liquidity/{symbol}")
async def get_liquidity_pools(symbol: str, timeframe: str = "1D"):
    """Get Liquidity Pools for a symbol"""
    try:
        tf_map = {
            "1H": ("5d", "1h"),
            "4H": ("1mo", "1h"),
            "1D": ("3mo", "1d"),
            "1W": ("1y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("3mo", "1d"))

        df = get_ohlcv_data(symbol, period, interval)
        current_price = float(df['close'].iloc[-1])

        smc = SmartMoneyIndicators()
        pools = smc.detect_liquidity_pools(df)

        result = []
        for pool in pools:
            distance_pct = ((pool['price'] - current_price) / current_price) * 100
            result.append({
                'type': pool['type'],
                'price': float(pool['price']),
                'touches': int(pool['touches']),
                'strength': int(pool['strength']),
                'swept': bool(pool['swept']),
                'distance_pct': round(distance_pct, 2)
            })

        # Separate buy-side and sell-side
        buy_side = [p for p in result if p['type'] == 'BUY_SIDE_LIQUIDITY']
        sell_side = [p for p in result if p['type'] == 'SELL_SIDE_LIQUIDITY']

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'buy_side_liquidity': buy_side,
            'sell_side_liquidity': sell_side,
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting liquidity pools for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/structure/{symbol}")
async def get_market_structure(symbol: str, timeframe: str = "1D"):
    """Get Market Structure (BOS/CHoCH) for a symbol"""
    try:
        tf_map = {
            "1H": ("5d", "1h"),
            "4H": ("1mo", "1h"),
            "1D": ("3mo", "1d"),
            "1W": ("1y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("3mo", "1d"))

        df = get_ohlcv_data(symbol, period, interval)
        current_price = float(df['close'].iloc[-1])

        smc = SmartMoneyIndicators()
        structure = smc.detect_market_structure(df)

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'current_trend': structure['current_trend'],
            'break_of_structure': [
                {
                    'direction': b['direction'],
                    'price': b['price'],
                    'type': 'BOS'
                } for b in structure['bos'][-10:]
            ],
            'change_of_character': [
                {
                    'direction': c['direction'],
                    'price': c['price'],
                    'type': 'CHOCH'
                } for c in structure['choch'][-5:]
            ],
            'swing_highs': [
                {'price': h['price'], 'index': h['index']}
                for h in structure['swing_highs'][-10:]
            ],
            'swing_lows': [
                {'price': l['price'], 'index': l['index']}
                for l in structure['swing_lows'][-10:]
            ],
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market structure for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
