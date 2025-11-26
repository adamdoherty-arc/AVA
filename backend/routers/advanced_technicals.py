"""
Advanced Technical Indicators API Router
Volume Profile, Ichimoku Cloud, Fibonacci, CVD, Order Flow
NO MOCK DATA - All endpoints use real market data from yfinance
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime
import logging
import yfinance as yf
import pandas as pd
import numpy as np

# Import existing indicator modules
from src.advanced_technical_indicators import VolumeProfileCalculator, OrderFlowAnalyzer
from src.standard_indicators import StandardIndicators
from src.momentum_indicators import MomentumIndicators

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/advanced-technicals", tags=["advanced-technicals"])


def get_ohlcv_data(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance"""
    ticker = yf.Ticker(symbol.upper())
    hist = ticker.history(period=period, interval=interval)

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data available for {symbol}")

    hist.columns = [col.lower() for col in hist.columns]
    return hist


@router.get("/volume-profile/{symbol}")
async def get_volume_profile(symbol: str, period: str = "3M", bins: int = 50):
    """
    Get Volume Profile analysis with POC, VAH, VAL, HVN, LVN

    - POC (Point of Control): Price with highest traded volume
    - VAH (Value Area High): Top of 70% volume range
    - VAL (Value Area Low): Bottom of 70% volume range
    - HVN (High Volume Nodes): Price levels with high volume
    - LVN (Low Volume Nodes): Price levels with low volume (fast breakout areas)
    """
    try:
        period_map = {
            "1W": "5d",
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y"
        }
        yf_period = period_map.get(period, "3mo")

        df = get_ohlcv_data(symbol, yf_period, "1d")
        current_price = float(df['close'].iloc[-1])

        # Calculate Volume Profile
        vp_calc = VolumeProfileCalculator(value_area_pct=0.70)
        vp = vp_calc.calculate_volume_profile(df, price_bins=bins)

        # Get trading signals
        signals = vp_calc.get_trading_signals(current_price, vp)

        # Format HVN and LVN with distances
        hvn_with_distance = []
        for price in vp['high_volume_nodes'][:10]:
            distance_pct = ((price - current_price) / current_price) * 100
            hvn_with_distance.append({
                'price': round(price, 2),
                'distance_pct': round(distance_pct, 2),
                'type': 'support' if price < current_price else 'resistance'
            })

        lvn_with_distance = []
        for price in vp['low_volume_nodes'][:10]:
            distance_pct = ((price - current_price) / current_price) * 100
            lvn_with_distance.append({
                'price': round(price, 2),
                'distance_pct': round(distance_pct, 2),
                'note': 'Fast move area - low liquidity'
            })

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'period': period,
            'timestamp': datetime.now().isoformat(),

            'volume_profile': {
                'poc': {
                    'price': round(vp['poc']['price'], 2),
                    'volume': int(vp['poc']['volume']),
                    'pct_of_total': round(vp['poc']['pct_of_total'], 1)
                },
                'vah': round(vp['vah'], 2),
                'val': round(vp['val'], 2),
                'value_area_width': round(vp['value_area_width'], 2),
                'value_area_width_pct': round(vp['value_area_width_pct'], 2)
            },

            'high_volume_nodes': hvn_with_distance,
            'low_volume_nodes': lvn_with_distance,

            'signals': {
                'position': signals['position'],
                'bias': signals['bias'],
                'setup_quality': signals['setup_quality'],
                'recommendation': signals['recommendation'],
                'distance_from_poc_pct': round(signals['distance_from_poc_pct'], 2),
                'near_hvn': signals['near_hvn'],
                'near_lvn': signals['near_lvn']
            },

            # Distribution data for visualization
            'distribution': {
                'price_levels': [round(p, 2) for p in vp['price_levels']],
                'volume_at_price': [int(v) for v in vp['volume_at_price']]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Volume Profile for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cvd/{symbol}")
async def get_cvd_analysis(symbol: str, period: str = "1M"):
    """
    Get Cumulative Volume Delta (CVD) analysis

    CVD measures buying vs selling pressure:
    - Rising CVD = Accumulation (buying pressure)
    - Falling CVD = Distribution (selling pressure)
    - CVD divergence from price = Potential reversal
    """
    try:
        period_map = {
            "1W": "5d",
            "1M": "1mo",
            "3M": "3mo"
        }
        yf_period = period_map.get(period, "1mo")

        df = get_ohlcv_data(symbol, yf_period, "1d")
        current_price = float(df['close'].iloc[-1])

        # Calculate CVD
        of_analyzer = OrderFlowAnalyzer()
        cvd = of_analyzer.calculate_cvd(df)
        df['cvd'] = cvd

        # Find divergences
        try:
            divergences = of_analyzer.find_cvd_divergences(df, lookback=10)
        except Exception:
            divergences = []

        # Calculate CVD trend
        cvd_current = float(cvd.iloc[-1])
        cvd_5d_ago = float(cvd.iloc[-6]) if len(cvd) >= 6 else cvd_current
        cvd_10d_ago = float(cvd.iloc[-11]) if len(cvd) >= 11 else cvd_current

        cvd_trend_5d = 'RISING' if cvd_current > cvd_5d_ago else 'FALLING'
        cvd_trend_10d = 'RISING' if cvd_current > cvd_10d_ago else 'FALLING'

        # Build CVD chart data
        cvd_data = []
        for i, (idx, row) in enumerate(df.iterrows()):
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
            cvd_data.append({
                'date': date_str,
                'price': round(row['close'], 2),
                'cvd': int(cvd.iloc[i]),
                'volume': int(row['volume'])
            })

        # Interpretation
        if cvd_trend_5d == 'RISING' and df['close'].iloc[-1] > df['close'].iloc[-6]:
            interpretation = 'BULLISH_CONFIRMED'
            description = 'Price and CVD both rising - Strong buying pressure'
        elif cvd_trend_5d == 'FALLING' and df['close'].iloc[-1] < df['close'].iloc[-6]:
            interpretation = 'BEARISH_CONFIRMED'
            description = 'Price and CVD both falling - Strong selling pressure'
        elif cvd_trend_5d == 'RISING' and df['close'].iloc[-1] < df['close'].iloc[-6]:
            interpretation = 'BULLISH_DIVERGENCE'
            description = 'CVD rising while price falling - Potential bullish reversal'
        elif cvd_trend_5d == 'FALLING' and df['close'].iloc[-1] > df['close'].iloc[-6]:
            interpretation = 'BEARISH_DIVERGENCE'
            description = 'CVD falling while price rising - Potential bearish reversal'
        else:
            interpretation = 'NEUTRAL'
            description = 'No clear CVD signal'

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'period': period,
            'timestamp': datetime.now().isoformat(),

            'cvd': {
                'current': int(cvd_current),
                'change_5d': int(cvd_current - cvd_5d_ago),
                'change_10d': int(cvd_current - cvd_10d_ago),
                'trend_5d': cvd_trend_5d,
                'trend_10d': cvd_trend_10d
            },

            'interpretation': interpretation,
            'description': description,

            'divergences': [
                {
                    'type': d['type'],
                    'price': d['price'],
                    'signal': d['signal'],
                    'strength': d['strength']
                } for d in divergences
            ] if divergences else [],

            'chart_data': cvd_data[-30:]  # Last 30 days
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting CVD for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ichimoku/{symbol}")
async def get_ichimoku_cloud(symbol: str, period: str = "3M"):
    """
    Get Ichimoku Cloud analysis

    Components:
    - Tenkan-sen (Conversion Line): 9-period midpoint
    - Kijun-sen (Base Line): 26-period midpoint
    - Senkou Span A (Leading Span A): Midpoint of Tenkan/Kijun, plotted 26 periods ahead
    - Senkou Span B (Leading Span B): 52-period midpoint, plotted 26 periods ahead
    - Chikou Span (Lagging Span): Close plotted 26 periods back

    Signals:
    - TK Cross: Tenkan crosses Kijun (trend signal)
    - Cloud position: Price above/below/in cloud
    - Cloud color: Green (bullish) or Red (bearish)
    """
    try:
        period_map = {
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y"
        }
        yf_period = period_map.get(period, "3mo")

        df = get_ohlcv_data(symbol, yf_period, "1d")
        current_price = float(df['close'].iloc[-1])

        # Calculate Ichimoku
        std_indicators = StandardIndicators()
        ichimoku = std_indicators.ichimoku(df)

        # Get signal
        signal = std_indicators.ichimoku_signal(current_price, ichimoku)

        # Build cloud data for visualization
        cloud_data = []
        for i in range(len(df)):
            idx = df.index[i]
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]

            cloud_data.append({
                'date': date_str,
                'close': round(df['close'].iloc[i], 2),
                'tenkan': round(float(ichimoku['tenkan'].iloc[i]), 2) if pd.notna(ichimoku['tenkan'].iloc[i]) else None,
                'kijun': round(float(ichimoku['kijun'].iloc[i]), 2) if pd.notna(ichimoku['kijun'].iloc[i]) else None,
                'senkou_a': round(float(ichimoku['senkou_a'].iloc[i]), 2) if pd.notna(ichimoku['senkou_a'].iloc[i]) else None,
                'senkou_b': round(float(ichimoku['senkou_b'].iloc[i]), 2) if pd.notna(ichimoku['senkou_b'].iloc[i]) else None
            })

        # Current values
        tenkan_current = float(ichimoku['tenkan'].iloc[-1]) if pd.notna(ichimoku['tenkan'].iloc[-1]) else 0
        kijun_current = float(ichimoku['kijun'].iloc[-1]) if pd.notna(ichimoku['kijun'].iloc[-1]) else 0
        senkou_a_current = float(ichimoku['senkou_a'].iloc[-1]) if pd.notna(ichimoku['senkou_a'].iloc[-1]) else 0
        senkou_b_current = float(ichimoku['senkou_b'].iloc[-1]) if pd.notna(ichimoku['senkou_b'].iloc[-1]) else 0

        # Cloud color
        cloud_color = 'green' if senkou_a_current > senkou_b_current else 'red'

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'period': period,
            'timestamp': datetime.now().isoformat(),

            'ichimoku': {
                'tenkan': round(tenkan_current, 2),
                'kijun': round(kijun_current, 2),
                'senkou_a': round(senkou_a_current, 2),
                'senkou_b': round(senkou_b_current, 2),
                'cloud_top': round(max(senkou_a_current, senkou_b_current), 2),
                'cloud_bottom': round(min(senkou_a_current, senkou_b_current), 2),
                'cloud_color': cloud_color
            },

            'signal': {
                'overall': signal['signal'],
                'strength': signal['strength'],
                'cloud_position': signal['cloud_position'],
                'cloud_bias': signal['cloud_bias'],
                'tk_bullish_cross': signal['tk_bullish_cross'],
                'tk_bearish_cross': signal['tk_bearish_cross'],
                'recommendation': signal['recommendation']
            },

            'interpretation': {
                'trend': 'BULLISH' if signal['cloud_position'] == 'ABOVE_CLOUD' else 'BEARISH' if signal['cloud_position'] == 'BELOW_CLOUD' else 'CONSOLIDATING',
                'tk_relationship': 'BULLISH' if tenkan_current > kijun_current else 'BEARISH',
                'cloud_future': 'BULLISH' if cloud_color == 'green' else 'BEARISH'
            },

            'chart_data': cloud_data[-60:]  # Last 60 days
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Ichimoku for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fibonacci/{symbol}")
async def get_fibonacci_levels(symbol: str, period: str = "3M", trend: str = "auto"):
    """
    Get Fibonacci retracement and extension levels

    Retracement Levels:
    - 23.6%, 38.2%, 50%, 61.8% (Golden Ratio), 78.6%

    Extension Levels:
    - 127.2%, 161.8%, 261.8%, 423.6%

    Golden Zone: 50% - 61.8% (high probability reversal zone)
    """
    try:
        period_map = {
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y"
        }
        yf_period = period_map.get(period, "3mo")

        df = get_ohlcv_data(symbol, yf_period, "1d")
        current_price = float(df['close'].iloc[-1])

        # Find swing high and low
        swing_high = float(df['high'].max())
        swing_low = float(df['low'].min())

        # Auto-detect trend
        if trend == "auto":
            # If current price is closer to high, assume downtrend (retracing from high)
            # If closer to low, assume uptrend (retracing from low)
            mid_range = (swing_high + swing_low) / 2
            detected_trend = "BULLISH" if current_price > mid_range else "BEARISH"
        else:
            detected_trend = trend.upper()

        # Calculate Fibonacci levels
        momentum = MomentumIndicators()
        fib_levels = momentum.calculate_fibonacci_levels(swing_high, swing_low, detected_trend)

        # Check if at a Fibonacci level
        at_level = momentum.is_at_fibonacci_level(current_price, fib_levels, tolerance_pct=1.5)

        # Build levels list with distances
        levels = []
        level_names = {
            'level_0': '0% (Start)',
            'level_236': '23.6%',
            'level_382': '38.2%',
            'level_50': '50%',
            'level_618': '61.8% (Golden)',
            'level_786': '78.6%',
            'level_100': '100% (End)'
        }

        for key, name in level_names.items():
            price = fib_levels[key]
            distance_pct = ((price - current_price) / current_price) * 100
            levels.append({
                'level': name,
                'price': round(price, 2),
                'distance_pct': round(distance_pct, 2),
                'is_golden_zone': key in ['level_50', 'level_618']
            })

        # Extension levels
        diff = swing_high - swing_low
        extensions = [
            {'level': '127.2%', 'price': round(swing_low + diff * 1.272, 2) if detected_trend == 'BULLISH' else round(swing_high - diff * 1.272, 2)},
            {'level': '161.8%', 'price': round(swing_low + diff * 1.618, 2) if detected_trend == 'BULLISH' else round(swing_high - diff * 1.618, 2)},
            {'level': '261.8%', 'price': round(swing_low + diff * 2.618, 2) if detected_trend == 'BULLISH' else round(swing_high - diff * 2.618, 2)},
        ]

        # Determine signal
        if at_level.get('at_fib_level'):
            level_name = at_level['level']
            if 'level_618' in level_name or 'level_50' in level_name:
                signal = 'GOLDEN_ZONE'
                recommendation = f'Price at {level_name} - High probability reversal zone'
            else:
                signal = 'AT_FIB_LEVEL'
                recommendation = f'Price at {level_name} - Watch for reaction'
        else:
            signal = 'BETWEEN_LEVELS'
            recommendation = 'Price between Fibonacci levels'

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'period': period,
            'timestamp': datetime.now().isoformat(),

            'trend': detected_trend,
            'swing_high': round(swing_high, 2),
            'swing_low': round(swing_low, 2),

            'retracement_levels': levels,
            'extension_levels': extensions,

            'golden_zone': {
                'top': round(fib_levels['level_618'], 2),
                'bottom': round(fib_levels['level_50'], 2),
                'description': '50% - 61.8% zone with highest probability of reversal'
            },

            'signal': {
                'status': signal,
                'at_level': at_level,
                'recommendation': recommendation
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Fibonacci for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all-indicators/{symbol}")
async def get_all_standard_indicators(symbol: str, period: str = "3M"):
    """
    Get all standard technical indicators at once

    Includes: Bollinger Bands, Stochastic, OBV, VWAP, MFI, ADX, CCI
    """
    try:
        period_map = {
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y"
        }
        yf_period = period_map.get(period, "3mo")

        df = get_ohlcv_data(symbol, yf_period, "1d")
        current_price = float(df['close'].iloc[-1])

        std_indicators = StandardIndicators()
        all_indicators = std_indicators.get_all_indicators(df, current_price)

        return {
            'symbol': symbol.upper(),
            'current_price': round(current_price, 2),
            'period': period,
            'timestamp': datetime.now().isoformat(),

            'bollinger_bands': all_indicators['bollinger']['signal'],
            'stochastic': all_indicators['stochastic']['signal'],
            'obv': all_indicators['obv']['signal'],
            'vwap': all_indicators['vwap']['signal'],
            'mfi': all_indicators['mfi']['signal'],
            'adx': all_indicators['adx']['signal'],
            'cci': all_indicators['cci']['signal'],

            'summary': {
                'bullish_signals': len([k for k, v in all_indicators.items()
                                       if v.get('signal', {}).get('signal', '').upper() in ['BUY', 'BULLISH', 'STRONG_BUY', 'BULLISH_CONFIRMED']]),
                'bearish_signals': len([k for k, v in all_indicators.items()
                                       if v.get('signal', {}).get('signal', '').upper() in ['SELL', 'BEARISH', 'STRONG_SELL', 'BEARISH_CONFIRMED']]),
                'neutral_signals': len([k for k, v in all_indicators.items()
                                       if v.get('signal', {}).get('signal', '').upper() == 'NEUTRAL'])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
