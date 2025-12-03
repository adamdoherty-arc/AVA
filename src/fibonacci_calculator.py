"""
Fibonacci Calculator - Advanced Technical Analysis
===================================================

Calculates Fibonacci retracement, extension, and advanced levels for trading.

Features:
- Fibonacci Retracement (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Fibonacci Extension (127.2%, 161.8%, 261.8%)
- Golden Zone identification (50%-61.8%)
- Fibonacci Fan levels
- Fibonacci Confluence zones
- Auto swing high/low detection

Based on 2025 best practices from:
- Mind Math Money Fibonacci Trading Course
- TradingView Fibonacci strategies
- Professional trader methodologies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


class FibonacciCalculator:
    """
    Advanced Fibonacci calculator for trading analysis

    Calculates:
    1. Fibonacci Retracement levels
    2. Fibonacci Extension levels
    3. Golden Zone (50%-61.8%)
    4. Fibonacci Fan angles
    5. Confluence zones (multiple Fib levels overlapping)
    """

    # Standard Fibonacci ratios
    RETRACEMENT_LEVELS = {
        '0%': 0.0,
        '23.6%': 0.236,
        '38.2%': 0.382,
        '50%': 0.5,
        '61.8%': 0.618,
        '78.6%': 0.786,
        '100%': 1.0
    }

    EXTENSION_LEVELS = {
        '0%': 0.0,
        '61.8%': 0.618,
        '100%': 1.0,
        '127.2%': 1.272,
        '161.8%': 1.618,
        '200%': 2.0,
        '261.8%': 2.618,
        '423.6%': 4.236
    }

    # Golden Zone (highest probability reversal area)
    GOLDEN_ZONE = (0.5, 0.618)

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def calculate_retracement(
        self,
        swing_high: float,
        swing_low: float,
        direction: str = 'up'
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels

        Args:
            swing_high: Highest price point
            swing_low: Lowest price point
            direction: 'up' for uptrend retracement, 'down' for downtrend

        Returns:
            Dictionary of level names to prices
        """
        price_range = swing_high - swing_low

        levels = {}

        if direction == 'up':
            # Uptrend: Calculate from swing_low (100%) to swing_high (0%)
            for name, ratio in self.RETRACEMENT_LEVELS.items():
                levels[name] = swing_high - (price_range * ratio)
        else:
            # Downtrend: Calculate from swing_high (100%) to swing_low (0%)
            for name, ratio in self.RETRACEMENT_LEVELS.items():
                levels[name] = swing_low + (price_range * ratio)

        return levels

    def calculate_extension(
        self,
        swing_high: float,
        swing_low: float,
        retracement_point: float,
        direction: str = 'up'
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci extension levels (profit targets)

        Args:
            swing_high: Initial swing high
            swing_low: Initial swing low
            retracement_point: Where price retraced to before continuation
            direction: 'up' for uptrend extension, 'down' for downtrend

        Returns:
            Dictionary of extension level names to prices
        """
        initial_move = abs(swing_high - swing_low)

        levels = {}

        if direction == 'up':
            # Uptrend extensions: Project upward from retracement point
            for name, ratio in self.EXTENSION_LEVELS.items():
                levels[name] = retracement_point + (initial_move * ratio)
        else:
            # Downtrend extensions: Project downward from retracement point
            for name, ratio in self.EXTENSION_LEVELS.items():
                levels[name] = retracement_point - (initial_move * ratio)

        return levels

    def identify_golden_zone(
        self,
        swing_high: float,
        swing_low: float,
        direction: str = 'up'
    ) -> Tuple[float, float]:
        """
        Identify the Golden Zone (50%-61.8% retracement)

        This is the highest probability reversal zone

        Returns:
            Tuple of (golden_zone_top, golden_zone_bottom)
        """
        retracements = self.calculate_retracement(swing_high, swing_low, direction)

        golden_top = retracements['50%']
        golden_bottom = retracements['61.8%']

        return (max(golden_top, golden_bottom), min(golden_top, golden_bottom))

    def auto_detect_swings(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        prominence_pct: float = 0.02
    ) -> List[Dict]:
        """
        Auto-detect swing highs and lows from price data

        Args:
            df: DataFrame with OHLC data (lowercase columns)
            lookback: Number of candles to look back
            prominence_pct: Minimum prominence as % of price (default 2%)

        Returns:
            List of swing points with Fibonacci levels
        """
        if 'high' not in df.columns or 'low' not in df.columns:
            raise ValueError("DataFrame must have 'high' and 'low' columns")

        swings = []

        # Find swing highs
        highs = df['high'].values
        avg_price = df['close'].mean()
        prominence = avg_price * prominence_pct

        high_peaks, _ = find_peaks(highs, distance=lookback, prominence=prominence)

        # Find swing lows
        lows = df['low'].values
        low_peaks, _ = find_peaks(-lows, distance=lookback, prominence=prominence)

        # Process swing patterns (high -> low -> high for uptrend retracement)
        for i in range(len(high_peaks) - 1):
            high_idx = high_peaks[i]
            swing_high = highs[high_idx]

            # Find lows between this high and next high
            next_high_idx = high_peaks[i + 1] if i + 1 < len(high_peaks) else len(df)
            lows_between = [idx for idx in low_peaks if high_idx < idx < next_high_idx]

            if lows_between:
                low_idx = lows_between[0]
                swing_low = lows[low_idx]

                # Calculate Fibonacci levels for this swing
                retracements = self.calculate_retracement(swing_high, swing_low, direction='up')
                golden_zone = self.identify_golden_zone(swing_high, swing_low, direction='up')

                swing = {
                    'type': 'UPTREND_RETRACEMENT',
                    'swing_high': float(swing_high),
                    'swing_low': float(swing_low),
                    'high_date': df.index[high_idx],
                    'low_date': df.index[low_idx],
                    'retracement_levels': retracements,
                    'golden_zone': {
                        'top': golden_zone[0],
                        'bottom': golden_zone[1]
                    },
                    'price_range': float(swing_high - swing_low),
                    'range_pct': float((swing_high - swing_low) / swing_low * 100)
                }

                swings.append(swing)

        # Process swing patterns (low -> high -> low for downtrend retracement)
        for i in range(len(low_peaks) - 1):
            low_idx = low_peaks[i]
            swing_low = lows[low_idx]

            # Find highs between this low and next low
            next_low_idx = low_peaks[i + 1] if i + 1 < len(low_peaks) else len(df)
            highs_between = [idx for idx in high_peaks if low_idx < idx < next_low_idx]

            if highs_between:
                high_idx = highs_between[0]
                swing_high = highs[high_idx]

                # Calculate Fibonacci levels for this swing
                retracements = self.calculate_retracement(swing_high, swing_low, direction='down')
                golden_zone = self.identify_golden_zone(swing_high, swing_low, direction='down')

                swing = {
                    'type': 'DOWNTREND_RETRACEMENT',
                    'swing_high': float(swing_high),
                    'swing_low': float(swing_low),
                    'low_date': df.index[low_idx],
                    'high_date': df.index[high_idx],
                    'retracement_levels': retracements,
                    'golden_zone': {
                        'top': golden_zone[0],
                        'bottom': golden_zone[1]
                    },
                    'price_range': float(swing_high - swing_low),
                    'range_pct': float((swing_high - swing_low) / swing_low * 100)
                }

                swings.append(swing)

        return swings

    def find_fibonacci_confluence(
        self,
        swings: List[Dict],
        tolerance_pct: float = 0.5
    ) -> List[Dict]:
        """
        Find confluence zones where multiple Fibonacci levels overlap

        Confluence zones have higher probability of support/resistance

        Args:
            swings: List of swing dictionaries from auto_detect_swings()
            tolerance_pct: Price tolerance for confluence (default 0.5%)

        Returns:
            List of confluence zones with overlapping levels
        """
        all_levels = []

        # Collect all Fibonacci levels from all swings
        for swing in swings:
            for level_name, price in swing['retracement_levels'].items():
                all_levels.append({
                    'price': price,
                    'level': level_name,
                    'swing_type': swing['type'],
                    'swing_range': swing['price_range']
                })

        # Find clusters of levels (confluence)
        confluences = []

        for i, level1 in enumerate(all_levels):
            cluster = [level1]
            cluster_price = level1['price']

            # Find other levels within tolerance
            for j, level2 in enumerate(all_levels):
                if i != j:
                    price_diff_pct = abs(level2['price'] - cluster_price) / cluster_price * 100

                    if price_diff_pct <= tolerance_pct:
                        cluster.append(level2)

            # Only consider clusters with 2+ levels
            if len(cluster) >= 2:
                avg_price = sum(l['price'] for l in cluster) / len(cluster)

                confluence = {
                    'price': avg_price,
                    'level_count': len(cluster),
                    'levels': cluster,
                    'strength': len(cluster),  # More levels = stronger confluence
                    'price_min': min(l['price'] for l in cluster),
                    'price_max': max(l['price'] for l in cluster),
                    'zone_width_pct': (max(l['price'] for l in cluster) - min(l['price'] for l in cluster)) / avg_price * 100
                }

                # Avoid duplicates
                if not any(abs(c['price'] - confluence['price']) / confluence['price'] * 100 < 0.1
                          for c in confluences):
                    confluences.append(confluence)

        # Sort by strength (most levels = strongest)
        confluences.sort(key=lambda x: x['strength'], reverse=True)

        return confluences

    def get_current_position_relative_to_fibonacci(
        self,
        current_price: float,
        swing_high: float,
        swing_low: float,
        direction: str = 'up'
    ) -> Dict:
        """
        Determine where current price is relative to Fibonacci levels

        Returns:
            Dictionary with position info and trading recommendations
        """
        retracements = self.calculate_retracement(swing_high, swing_low, direction)
        golden_zone = self.identify_golden_zone(swing_high, swing_low, direction)

        # Find nearest Fibonacci level
        nearest_level = None
        min_distance = float('inf')

        for level_name, level_price in retracements.items():
            distance = abs(current_price - level_price)
            distance_pct = (distance / current_price) * 100

            if distance < min_distance:
                min_distance = distance
                nearest_level = {
                    'name': level_name,
                    'price': level_price,
                    'distance_pct': distance_pct
                }

        # Check if in Golden Zone
        in_golden_zone = golden_zone[1] <= current_price <= golden_zone[0]

        # Generate recommendation
        if in_golden_zone:
            if direction == 'up':
                recommendation = "🔥 IN GOLDEN ZONE - High probability BUY area (watch for reversal confirmation)"
                setup_quality = "EXCELLENT"
            else:
                recommendation = "🔥 IN GOLDEN ZONE - High probability SELL area (watch for reversal confirmation)"
                setup_quality = "EXCELLENT"
        elif nearest_level['distance_pct'] < 1.0:
            recommendation = f"✅ Near {nearest_level['name']} Fib level - Watch for support/resistance"
            setup_quality = "GOOD"
        else:
            recommendation = "⚠️ Between Fibonacci levels - Wait for price to reach key level"
            setup_quality = "FAIR"

        return {
            'current_price': current_price,
            'nearest_level': nearest_level,
            'in_golden_zone': in_golden_zone,
            'golden_zone_top': golden_zone[0],
            'golden_zone_bottom': golden_zone[1],
            'recommendation': recommendation,
            'setup_quality': setup_quality,
            'all_levels': retracements
        }


if __name__ == "__main__":
    # Test Fibonacci calculator
    import yfinance as yf

    print("=" * 80)
    print("FIBONACCI CALCULATOR TEST")
    print("=" * 80)

    calc = FibonacciCalculator()

    # Test manual calculation
    print("\n1. Manual Fibonacci Retracement (Uptrend):")
    print("-" * 80)
    swing_high = 150.0
    swing_low = 100.0

    retracements = calc.calculate_retracement(swing_high, swing_low, direction='up')
    print(f"Swing: ${swing_low:.2f} -> ${swing_high:.2f}")
    print("\nRetracement Levels:")
    for level, price in retracements.items():
        print(f"  {level:>6} : ${price:.2f}")

    golden_zone = calc.identify_golden_zone(swing_high, swing_low, direction='up')
    print(f"\nGolden Zone: ${golden_zone[1]:.2f} - ${golden_zone[0]:.2f}")

    # Test with real data
    print("\n\n2. Auto-Detect Swings from Real Data (AAPL):")
    print("-" * 80)

    ticker = yf.Ticker('AAPL')
    df = ticker.history(period='6mo', interval='1d')
    df.columns = [col.lower() for col in df.columns]

    swings = calc.auto_detect_swings(df, lookback=10)

    print(f"Found {len(swings)} swing patterns\n")

    for i, swing in enumerate(swings[:3], 1):
        print(f"Swing {i} ({swing['type']}):")
        print(f"  Range: ${swing['swing_low']:.2f} -> ${swing['swing_high']:.2f} ({swing['range_pct']:.1f}%)")
        print(f"  Golden Zone: ${swing['golden_zone']['bottom']:.2f} - ${swing['golden_zone']['top']:.2f}")
        print()

    # Test confluence
    if len(swings) > 0:
        print("\n3. Fibonacci Confluence Zones:")
        print("-" * 80)

        confluences = calc.find_fibonacci_confluence(swings, tolerance_pct=1.0)

        print(f"Found {len(confluences)} confluence zones\n")

        for i, conf in enumerate(confluences[:5], 1):
            print(f"Confluence {i}:")
            print(f"  Price: ${conf['price']:.2f}")
            print(f"  Strength: {conf['strength']} overlapping levels")
            print(f"  Zone: ${conf['price_min']:.2f} - ${conf['price_max']:.2f}")
            print()

    # Test current position
    current_price = df['close'].iloc[-1]

    if len(swings) > 0:
        print("\n4. Current Price Position:")
        print("-" * 80)

        latest_swing = swings[-1]
        position = calc.get_current_position_relative_to_fibonacci(
            current_price,
            latest_swing['swing_high'],
            latest_swing['swing_low'],
            'up' if latest_swing['type'] == 'UPTREND_RETRACEMENT' else 'down'
        )

        print(f"Current Price: ${position['current_price']:.2f}")
        print(f"Nearest Level: {position['nearest_level']['name']} @ ${position['nearest_level']['price']:.2f}")
        print(f"Distance: {position['nearest_level']['distance_pct']:.2f}%")
        print(f"In Golden Zone: {position['in_golden_zone']}")
        print(f"\n{position['recommendation']}")

    print("\n" + "=" * 80)
    print("✅ Fibonacci Calculator Test Complete")
    print("=" * 80)
