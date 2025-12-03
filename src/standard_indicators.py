"""
Standard Technical Indicators
Bollinger Bands, Stochastic, OBV, VWAP, MFI, Ichimoku, ADX, CCI
Using pandas-ta for implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StandardIndicators:
    """
    Standard technical indicators using pandas-ta

    Includes:
    - Bollinger Bands (volatility)
    - Stochastic Oscillator (momentum)
    - OBV (On-Balance Volume)
    - VWAP (Volume Weighted Average Price)
    - MFI (Money Flow Index)
    - Ichimoku Cloud (trend)
    - ADX (Average Directional Index)
    - CCI (Commodity Channel Index)
    """

    def __init__(self) -> None:
        try:
            import pandas_ta as ta
            self.ta = ta
        except ImportError:
            raise ImportError("pandas_ta is required. Install with: pip install pandas-ta")

    def bollinger_bands(
        self,
        df: pd.DataFrame,
        length: int = 20,
        std: float = 2.0
    ) -> Dict:
        """
        Calculate Bollinger Bands

        Args:
            df: DataFrame with 'close' column
            length: Period for moving average
            std: Standard deviation multiplier

        Returns:
            Dictionary with upper, middle, lower bands and bandwidth
        """
        bbands = self.ta.bbands(df['close'], length=length, std=std)

        # Dynamically find column names (pandas-ta may format floats differently)
        upper_col = [col for col in bbands.columns if col.startswith('BBU_')][0]
        middle_col = [col for col in bbands.columns if col.startswith('BBM_')][0]
        lower_col = [col for col in bbands.columns if col.startswith('BBL_')][0]

        bandwidth = (bbands[upper_col] - bbands[lower_col]) / bbands[middle_col]

        return {
            'upper': bbands[upper_col],
            'middle': bbands[middle_col],
            'lower': bbands[lower_col],
            'bandwidth': bandwidth,
            'percent_b': (df['close'] - bbands[lower_col]) / (bbands[upper_col] - bbands[lower_col])
        }

    def bollinger_signal(
        self,
        current_price: float,
        bbands: Dict
    ) -> Dict:
        """
        Generate Bollinger Bands trading signal

        Returns signal based on price position relative to bands
        """
        upper = bbands['upper'].iloc[-1]
        middle = bbands['middle'].iloc[-1]
        lower = bbands['lower'].iloc[-1]
        bandwidth = bbands['bandwidth'].iloc[-1]
        percent_b = bbands['percent_b'].iloc[-1]

        # Volatility state
        bandwidth_percentile = bbands['bandwidth'].rank(pct=True).iloc[-1]

        if bandwidth_percentile < 0.2:
            volatility_state = 'SQUEEZE'
            vol_recommendation = 'Low volatility - expect breakout. Consider long options (straddles/strangles)'
        elif bandwidth_percentile > 0.8:
            volatility_state = 'EXPANSION'
            vol_recommendation = 'High volatility - consider selling premium (iron condors/credit spreads)'
        else:
            volatility_state = 'NORMAL'
            vol_recommendation = 'Normal volatility'

        # Price position
        if current_price > upper:
            position = 'ABOVE_UPPER'
            signal = 'OVERBOUGHT'
            recommendation = 'Price above upper band - watch for reversal or continuation'
        elif current_price < lower:
            position = 'BELOW_LOWER'
            signal = 'OVERSOLD'
            recommendation = 'Price below lower band - watch for bounce or breakdown'
        elif current_price > middle:
            position = 'UPPER_HALF'
            signal = 'BULLISH'
            recommendation = 'Price in upper half - bullish bias'
        elif current_price < middle:
            position = 'LOWER_HALF'
            signal = 'BEARISH'
            recommendation = 'Price in lower half - bearish bias'
        else:
            position = 'AT_MIDDLE'
            signal = 'NEUTRAL'
            recommendation = 'Price at middle band - no clear bias'

        return {
            'position': position,
            'signal': signal,
            'volatility_state': volatility_state,
            'bandwidth': float(bandwidth),
            'percent_b': float(percent_b),
            'recommendation': recommendation,
            'volatility_recommendation': vol_recommendation
        }

    def stochastic(
        self,
        df: pd.DataFrame,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3
    ) -> Dict:
        """
        Calculate Stochastic Oscillator

        Args:
            df: DataFrame with 'high', 'low', 'close'
            k: %K period
            d: %D period (signal line)
            smooth_k: Smoothing for %K

        Returns:
            Dictionary with %K and %D values
        """
        stoch = self.ta.stoch(
            df['high'],
            df['low'],
            df['close'],
            k=k,
            d=d,
            smooth_k=smooth_k
        )

        k_col = f'STOCHk_{k}_{d}_{smooth_k}'
        d_col = f'STOCHd_{k}_{d}_{smooth_k}'

        return {
            'k': stoch[k_col],
            'd': stoch[d_col]
        }

    def stochastic_signal(self, stoch: Dict) -> Dict:
        """Generate Stochastic trading signal"""
        k_current = stoch['k'].iloc[-1]
        d_current = stoch['d'].iloc[-1]
        k_prev = stoch['k'].iloc[-2]
        d_prev = stoch['d'].iloc[-2]

        # Overbought/Oversold
        if k_current < 20 and d_current < 20:
            zone = 'OVERSOLD'
            strength = 'STRONG'
        elif k_current < 30 and d_current < 30:
            zone = 'OVERSOLD'
            strength = 'MODERATE'
        elif k_current > 80 and d_current > 80:
            zone = 'OVERBOUGHT'
            strength = 'STRONG'
        elif k_current > 70 and d_current > 70:
            zone = 'OVERBOUGHT'
            strength = 'MODERATE'
        else:
            zone = 'NEUTRAL'
            strength = 'NEUTRAL'

        # Crossover detection
        bullish_cross = (k_current > d_current) and (k_prev <= d_prev)
        bearish_cross = (k_current < d_current) and (k_prev >= d_prev)

        if bullish_cross and zone == 'OVERSOLD':
            signal = 'STRONG_BUY'
            recommendation = 'Bullish crossover in oversold zone - strong buy signal'
        elif bullish_cross:
            signal = 'BUY'
            recommendation = 'Bullish crossover - buy signal'
        elif bearish_cross and zone == 'OVERBOUGHT':
            signal = 'STRONG_SELL'
            recommendation = 'Bearish crossover in overbought zone - strong sell signal'
        elif bearish_cross:
            signal = 'SELL'
            recommendation = 'Bearish crossover - sell signal'
        elif zone == 'OVERSOLD':
            signal = 'BUY'
            recommendation = 'Oversold - potential bounce'
        elif zone == 'OVERBOUGHT':
            signal = 'SELL'
            recommendation = 'Overbought - potential reversal'
        else:
            signal = 'NEUTRAL'
            recommendation = 'No clear signal'

        return {
            'signal': signal,
            'zone': zone,
            'strength': strength,
            'k': float(k_current),
            'd': float(d_current),
            'bullish_cross': bool(bullish_cross),
            'bearish_cross': bool(bearish_cross),
            'recommendation': recommendation
        }

    def obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        return self.ta.obv(df['close'], df['volume'])

    def obv_signal(self, obv: pd.Series, price: pd.Series) -> Dict:
        """Generate OBV trading signal"""
        obv_current = obv.iloc[-1]
        obv_prev = obv.iloc[-20]  # 20 periods ago

        price_current = price.iloc[-1]
        price_prev = price.iloc[-20]

        obv_rising = obv_current > obv_prev
        price_rising = price_current > price_prev

        # Determine trend
        if obv_rising:
            trend = "RISING"
        else:
            trend = "FALLING"

        # Trend confirmation
        if obv_rising and price_rising:
            signal = 'BULLISH_CONFIRMED'
            recommendation = 'Price and volume confirm uptrend - strong bullish'
        elif not obv_rising and not price_rising:
            signal = 'BEARISH_CONFIRMED'
            recommendation = 'Price and volume confirm downtrend - strong bearish'
        elif obv_rising and not price_rising:
            signal = 'BULLISH_DIVERGENCE'
            recommendation = 'OBV rising while price falling - potential reversal up'
        elif not obv_rising and price_rising:
            signal = 'BEARISH_DIVERGENCE'
            recommendation = 'OBV falling while price rising - potential reversal down'
        else:
            signal = 'NEUTRAL'
            recommendation = 'No clear OBV signal'

        return {
            'signal': signal,
            'trend': trend,
            'obv': float(obv_current),
            'obv_change': float(obv_current - obv_prev),
            'recommendation': recommendation
        }

    def vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price)

        Note: VWAP typically resets daily. For intraday data, group by date.
        """
        return self.ta.vwap(df['high'], df['low'], df['close'], df['volume'])

    def vwap_signal(self, current_price: float, vwap: pd.Series) -> Dict:
        """Generate VWAP trading signal"""
        vwap_current = vwap.iloc[-1]

        distance_pct = ((current_price - vwap_current) / vwap_current) * 100

        if current_price > vwap_current:
            position = 'ABOVE_VWAP'
            signal = 'BULLISH'
            recommendation = f'Price {abs(distance_pct):.2f}% above VWAP - bullish bias'
        elif current_price < vwap_current:
            position = 'BELOW_VWAP'
            signal = 'BEARISH'
            recommendation = f'Price {abs(distance_pct):.2f}% below VWAP - bearish bias'
        else:
            position = 'AT_VWAP'
            signal = 'NEUTRAL'
            recommendation = 'Price at VWAP - fair value'

        return {
            'signal': signal,
            'position': position,
            'vwap': float(vwap_current),
            'distance_pct': float(distance_pct),
            'recommendation': recommendation
        }

    def mfi(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Calculate Money Flow Index (volume-weighted RSI)"""
        return self.ta.mfi(
            df['high'],
            df['low'],
            df['close'],
            df['volume'],
            length=length
        )

    def mfi_signal(self, mfi: pd.Series) -> Dict:
        """Generate MFI trading signal"""
        mfi_current = mfi.iloc[-1]

        if mfi_current < 20:
            zone = 'OVERSOLD'
            signal = 'STRONG_BUY'
            recommendation = 'MFI < 20 - strong oversold, potential bounce'
        elif mfi_current < 30:
            zone = 'OVERSOLD'
            signal = 'BUY'
            recommendation = 'MFI < 30 - oversold territory'
        elif mfi_current > 80:
            zone = 'OVERBOUGHT'
            signal = 'STRONG_SELL'
            recommendation = 'MFI > 80 - strong overbought, potential reversal'
        elif mfi_current > 70:
            zone = 'OVERBOUGHT'
            signal = 'SELL'
            recommendation = 'MFI > 70 - overbought territory'
        else:
            zone = 'NEUTRAL'
            signal = 'NEUTRAL'
            recommendation = 'MFI in neutral zone'

        return {
            'signal': signal,
            'zone': zone,
            'value': float(mfi_current),
            'mfi': float(mfi_current),
            'recommendation': recommendation
        }

    def ichimoku(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Ichimoku Cloud

        Returns all Ichimoku components
        """
        ich = self.ta.ichimoku(df['high'], df['low'], df['close'])[0]

        return {
            'conversion': ich['ITS_9'],      # Conversion line (Tenkan-sen)
            'base': ich['IKS_26'],           # Base line (Kijun-sen)
            'span_a': ich['ISA_9'],          # Leading span A (Senkou Span A)
            'span_b': ich['ISB_26'],         # Leading span B (Senkou Span B)
            'lagging': ich['ICS_26'],        # Lagging span (Chikou Span)
            # Also keep original names for compatibility
            'tenkan': ich['ITS_9'],
            'kijun': ich['IKS_26'],
            'senkou_a': ich['ISA_9'],
            'senkou_b': ich['ISB_26'],
            'chikou': ich['ICS_26']
        }

    def ichimoku_signal(
        self,
        current_price: float,
        ichimoku: Dict
    ) -> Dict:
        """Generate Ichimoku trading signal"""
        tenkan = ichimoku['conversion'].iloc[-1]
        kijun = ichimoku['base'].iloc[-1]
        senkou_a = ichimoku['span_a'].iloc[-1]
        senkou_b = ichimoku['span_b'].iloc[-1]

        # TK Cross
        tenkan_prev = ichimoku['conversion'].iloc[-2]
        kijun_prev = ichimoku['base'].iloc[-2]

        tk_bullish_cross = (tenkan > kijun) and (tenkan_prev <= kijun_prev)
        tk_bearish_cross = (tenkan < kijun) and (tenkan_prev >= kijun_prev)

        # Cloud position
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        if current_price > cloud_top:
            cloud_position = 'ABOVE_CLOUD'
            cloud_bias = 'BULLISH'
        elif current_price < cloud_bottom:
            cloud_position = 'BELOW_CLOUD'
            cloud_bias = 'BEARISH'
        else:
            cloud_position = 'IN_CLOUD'
            cloud_bias = 'NEUTRAL'

        # Cloud thickness (support/resistance strength)
        cloud_thickness = abs(senkou_a - senkou_b)
        cloud_thick = cloud_thickness > (current_price * 0.02)  # > 2% of price

        # Overall signal
        if tk_bullish_cross and cloud_position == 'ABOVE_CLOUD':
            signal = 'STRONG_BULLISH'
            strength = 'STRONG'
            recommendation = 'TK bullish cross above cloud - strong buy'
        elif tk_bullish_cross:
            signal = 'BULLISH'
            strength = 'MODERATE'
            recommendation = 'TK bullish cross - buy signal'
        elif tk_bearish_cross and cloud_position == 'BELOW_CLOUD':
            signal = 'STRONG_BEARISH'
            strength = 'STRONG'
            recommendation = 'TK bearish cross below cloud - strong sell'
        elif tk_bearish_cross:
            signal = 'BEARISH'
            strength = 'MODERATE'
            recommendation = 'TK bearish cross - sell signal'
        elif cloud_position == 'ABOVE_CLOUD':
            signal = 'BULLISH'
            strength = 'WEAK'
            recommendation = 'Price above cloud - bullish bias'
        elif cloud_position == 'BELOW_CLOUD':
            signal = 'BEARISH'
            strength = 'WEAK'
            recommendation = 'Price below cloud - bearish bias'
        else:
            signal = 'NEUTRAL'
            strength = 'NEUTRAL'
            recommendation = 'Price in cloud - consolidation'

        return {
            'signal': signal,
            'strength': strength,
            'cloud_position': cloud_position,
            'cloud_bias': cloud_bias,
            'cloud_thick': bool(cloud_thick),
            'tk_bullish_cross': bool(tk_bullish_cross),
            'tk_bearish_cross': bool(tk_bearish_cross),
            'recommendation': recommendation
        }

    def adx(self, df: pd.DataFrame, length: int = 14) -> Dict:
        """
        Calculate ADX (Average Directional Index)

        Returns ADX, +DI, -DI
        """
        adx_result = self.ta.adx(
            df['high'],
            df['low'],
            df['close'],
            length=length
        )

        return {
            'adx': adx_result[f'ADX_{length}'],
            'di_plus': adx_result[f'DMP_{length}'],
            'di_minus': adx_result[f'DMN_{length}']
        }

    def adx_signal(self, adx: Dict) -> Dict:
        """Generate ADX trading signal"""
        adx_value = adx['adx'].iloc[-1]
        di_plus = adx['di_plus'].iloc[-1]
        di_minus = adx['di_minus'].iloc[-1]

        # Trend strength
        if adx_value < 20:
            trend_strength = 'WEAK'
            strength_description = 'Weak or no trend'
        elif adx_value < 40:
            trend_strength = 'MODERATE'
            strength_description = 'Moderate trend'
        else:
            trend_strength = 'STRONG'
            strength_description = 'Strong trend'

        # Direction
        if di_plus > di_minus:
            direction = 'BULLISH'
            if adx_value > 40:
                signal = 'STRONG_BULLISH'
                recommendation = 'Strong uptrend - consider directional bullish plays'
            elif adx_value > 20:
                signal = 'BULLISH'
                recommendation = 'Moderate uptrend - bullish bias'
            else:
                signal = 'NEUTRAL'
                recommendation = 'Weak trend - avoid directional plays'
        elif di_minus > di_plus:
            direction = 'BEARISH'
            if adx_value > 40:
                signal = 'STRONG_BEARISH'
                recommendation = 'Strong downtrend - consider directional bearish plays'
            elif adx_value > 20:
                signal = 'BEARISH'
                recommendation = 'Moderate downtrend - bearish bias'
            else:
                signal = 'NEUTRAL'
                recommendation = 'Weak trend - avoid directional plays'
        else:
            direction = 'NEUTRAL'
            signal = 'NEUTRAL'
            recommendation = 'No clear directional bias'

        # ADX < 20: Consider non-directional strategies (iron condors)
        if adx_value < 20:
            options_strategy = 'Iron condors, butterflies (range-bound strategies)'
        else:
            options_strategy = 'Directional spreads, long options'

        return {
            'signal': signal,
            'direction': direction,
            'trend_strength': trend_strength,
            'adx': float(adx_value),
            'adx_value': float(adx_value),
            'di_plus': float(di_plus),
            'di_minus': float(di_minus),
            'recommendation': recommendation,
            'options_strategy': options_strategy
        }

    def cci(self, df: pd.DataFrame, length: int = 20) -> pd.Series:
        """Calculate CCI (Commodity Channel Index)"""
        return self.ta.cci(df['high'], df['low'], df['close'], length=length)

    def cci_signal(self, cci: pd.Series) -> Dict:
        """Generate CCI trading signal"""
        cci_current = cci.iloc[-1]
        cci_prev = cci.iloc[-2]

        # Zone detection
        if cci_current < -200:
            zone = 'EXTREME_OVERSOLD'
            signal = 'STRONG_BUY'
            recommendation = 'CCI < -200 - extreme oversold, strong reversal potential'
        elif cci_current < -100:
            zone = 'OVERSOLD'
            signal = 'BUY'
            recommendation = 'CCI < -100 - oversold, watch for reversal'
        elif cci_current > 200:
            zone = 'EXTREME_OVERBOUGHT'
            signal = 'STRONG_SELL'
            recommendation = 'CCI > 200 - extreme overbought, strong reversal potential'
        elif cci_current > 100:
            zone = 'OVERBOUGHT'
            signal = 'SELL'
            recommendation = 'CCI > 100 - overbought, watch for reversal'
        else:
            zone = 'NEUTRAL'
            signal = 'NEUTRAL'
            recommendation = 'CCI in neutral zone'

        # Zero line cross
        zero_cross_bullish = (cci_current > 0) and (cci_prev <= 0)
        zero_cross_bearish = (cci_current < 0) and (cci_prev >= 0)

        if zero_cross_bullish:
            signal = 'BUY'
            recommendation = 'CCI crossed above zero - bullish signal'
        elif zero_cross_bearish:
            signal = 'SELL'
            recommendation = 'CCI crossed below zero - bearish signal'

        return {
            'signal': signal,
            'zone': zone,
            'cci': float(cci_current),
            'zero_cross_bullish': bool(zero_cross_bullish),
            'zero_cross_bearish': bool(zero_cross_bearish),
            'recommendation': recommendation
        }

    def get_all_indicators(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Calculate all standard indicators at once

        Returns comprehensive dictionary with all indicators and signals
        """
        # Calculate all indicators
        bbands = self.bollinger_bands(df)
        stoch = self.stochastic(df)
        obv_series = self.obv(df)
        vwap_series = self.vwap(df)
        mfi_series = self.mfi(df)
        ichimoku_data = self.ichimoku(df)
        adx_data = self.adx(df)
        cci_series = self.cci(df)

        # Generate signals
        return {
            'bollinger': {
                'data': bbands,
                'signal': self.bollinger_signal(current_price, bbands)
            },
            'stochastic': {
                'data': stoch,
                'signal': self.stochastic_signal(stoch)
            },
            'obv': {
                'data': obv_series,
                'signal': self.obv_signal(obv_series, df['close'])
            },
            'vwap': {
                'data': vwap_series,
                'signal': self.vwap_signal(current_price, vwap_series)
            },
            'mfi': {
                'data': mfi_series,
                'signal': self.mfi_signal(mfi_series)
            },
            'ichimoku': {
                'data': ichimoku_data,
                'signal': self.ichimoku_signal(current_price, ichimoku_data)
            },
            'adx': {
                'data': adx_data,
                'signal': self.adx_signal(adx_data)
            },
            'cci': {
                'data': cci_series,
                'signal': self.cci_signal(cci_series)
            }
        }


if __name__ == "__main__":
    # Test
    import yfinance as yf
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("STANDARD TECHNICAL INDICATORS TEST")
    print("=" * 80)

    # Fetch data
    ticker = yf.Ticker('AAPL')
    df = ticker.history(period='3mo', interval='1d')
    df.columns = [col.lower() for col in df.columns]

    current_price = df['close'].iloc[-1]

    # Initialize
    indicators = StandardIndicators()

    # Get all indicators
    results = indicators.get_all_indicators(df, current_price)

    print(f"\nCurrent Price: ${current_price:.2f}\n")

    # Display results
    print("Bollinger Bands:")
    bb_sig = results['bollinger']['signal']
    print(f"  Signal: {bb_sig['signal']}")
    print(f"  Position: {bb_sig['position']}")
    print(f"  Volatility: {bb_sig['volatility_state']}")
    print(f"  {bb_sig['recommendation']}")

    print("\nStochastic:")
    stoch_sig = results['stochastic']['signal']
    print(f"  Signal: {stoch_sig['signal']}")
    print(f"  Zone: {stoch_sig['zone']}")
    print(f"  %K: {stoch_sig['k']:.1f}, %D: {stoch_sig['d']:.1f}")
    print(f"  {stoch_sig['recommendation']}")

    print("\nOBV:")
    obv_sig = results['obv']['signal']
    print(f"  Signal: {obv_sig['signal']}")
    print(f"  {obv_sig['recommendation']}")

    print("\nVWAP:")
    vwap_sig = results['vwap']['signal']
    print(f"  Signal: {vwap_sig['signal']}")
    print(f"  Position: {vwap_sig['position']}")
    print(f"  {vwap_sig['recommendation']}")

    print("\nMFI:")
    mfi_sig = results['mfi']['signal']
    print(f"  Signal: {mfi_sig['signal']}")
    print(f"  MFI: {mfi_sig['mfi']:.1f}")
    print(f"  {mfi_sig['recommendation']}")

    print("\nIchimoku:")
    ich_sig = results['ichimoku']['signal']
    print(f"  Signal: {ich_sig['signal']}")
    print(f"  Cloud Position: {ich_sig['cloud_position']}")
    print(f"  {ich_sig['recommendation']}")

    print("\nADX:")
    adx_sig = results['adx']['signal']
    print(f"  Signal: {adx_sig['signal']}")
    print(f"  Trend Strength: {adx_sig['trend_strength']}")
    print(f"  ADX: {adx_sig['adx']:.1f}")
    print(f"  {adx_sig['recommendation']}")
    print(f"  Options Strategy: {adx_sig['options_strategy']}")

    print("\nCCI:")
    cci_sig = results['cci']['signal']
    print(f"  Signal: {cci_sig['signal']}")
    print(f"  Zone: {cci_sig['zone']}")
    print(f"  CCI: {cci_sig['cci']:.1f}")
    print(f"  {cci_sig['recommendation']}")

    print("\n" + "=" * 80)
    print("Standard Indicators Test Complete")
    print("=" * 80)
