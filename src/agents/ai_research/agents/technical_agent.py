"""
Technical Analyst Agent
Analyzes price action, indicators, chart patterns
"""

import logging
from typing import List, Dict
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agents.ai_research.models import TechnicalAnalysis, TrendDirection, SignalType

logger = logging.getLogger(__name__)


class TechnicalAgent:
    """
    Technical analysis specialist

    Analyzes:
    - Price trends and momentum
    - Technical indicators (RSI, MACD, moving averages)
    - Support/resistance levels
    - Volume patterns
    - Chart patterns
    """

    def __init__(self) -> None:
        self.api_calls = 0

    async def analyze(self, symbol: str) -> TechnicalAnalysis:
        """
        Perform technical analysis

        Args:
            symbol: Stock ticker symbol

        Returns:
            TechnicalAnalysis object
        """
        self.api_calls = 0
        logger.info(f"Starting technical analysis for {symbol}")

        try:
            # Fetch historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo", interval="1d")
            self.api_calls += 1

            if hist.empty:
                raise ValueError(f"No historical data for {symbol}")

            # Calculate indicators
            rsi = self._calculate_rsi(hist['Close'])
            macd_signal = self._calculate_macd_signal(hist['Close'])
            trend = self._determine_trend(hist['Close'])
            moving_averages = self._calculate_moving_averages(hist['Close'])
            bollinger_bands = self._calculate_bollinger_bands(hist['Close'])
            support_levels = self._find_support_levels(hist['Low'])
            resistance_levels = self._find_resistance_levels(hist['High'])
            volume_analysis = self._analyze_volume(hist['Volume'])
            chart_patterns = self._identify_chart_patterns(hist)

            # Generate recommendation
            recommendation = self._generate_recommendation(
                trend, rsi, macd_signal, moving_averages
            )

            # Calculate score
            score = self._calculate_score(
                trend, rsi, macd_signal, moving_averages, volume_analysis
            )

            return TechnicalAnalysis(
                score=score,
                trend=trend,
                rsi=rsi,
                macd_signal=macd_signal,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                moving_averages=moving_averages,
                bollinger_bands=bollinger_bands,
                volume_analysis=volume_analysis,
                chart_patterns=chart_patterns,
                recommendation=recommendation
            )

        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {str(e)}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

    def _calculate_macd_signal(self, prices: pd.Series) -> SignalType:
        """Calculate MACD and determine signal"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()

            # Check if MACD crossed signal line
            if macd.iloc[-1] > signal.iloc[-1]:
                if macd.iloc[-2] <= signal.iloc[-2]:
                    return SignalType.BULLISH  # Bullish crossover
                return SignalType.BULLISH
            elif macd.iloc[-1] < signal.iloc[-1]:
                if macd.iloc[-2] >= signal.iloc[-2]:
                    return SignalType.BEARISH  # Bearish crossover
                return SignalType.BEARISH
            else:
                return SignalType.NEUTRAL

        except:
            return SignalType.NEUTRAL

    def _determine_trend(self, prices: pd.Series) -> TrendDirection:
        """Determine price trend"""
        try:
            # Use 50-day and 200-day moving averages
            ma50 = prices.rolling(window=50).mean().iloc[-1]
            ma200 = prices.rolling(window=200).mean().iloc[-1]
            current_price = prices.iloc[-1]

            # Also check recent momentum
            returns_20d = (prices.iloc[-1] / prices.iloc[-20] - 1)

            if ma50 > ma200 and current_price > ma50 and returns_20d > 0.05:
                return TrendDirection.UPTREND
            elif ma50 < ma200 and current_price < ma50 and returns_20d < -0.05:
                return TrendDirection.DOWNTREND
            else:
                return TrendDirection.SIDEWAYS

        except:
            return TrendDirection.SIDEWAYS

    def _calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate key moving averages"""
        try:
            return {
                'MA20': float(prices.rolling(window=20).mean().iloc[-1]),
                'MA50': float(prices.rolling(window=50).mean().iloc[-1]),
                'MA200': float(prices.rolling(window=200).mean().iloc[-1])
            }
        except:
            return {'MA20': 0.0, 'MA50': 0.0, 'MA200': 0.0}

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()

            upper = sma + (std * 2)
            lower = sma - (std * 2)

            return {
                'upper': float(upper.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower.iloc[-1])
            }
        except:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}

    def _find_support_levels(self, lows: pd.Series, window: int = 20) -> List[float]:
        """Find support levels"""
        try:
            # Use rolling min to find local lows
            recent_lows = lows.tail(60)
            rolling_min = recent_lows.rolling(window=window, center=True).min()

            # Find levels where price bounced
            supports = []
            for i in range(window, len(recent_lows) - window):
                if recent_lows.iloc[i] == rolling_min.iloc[i]:
                    supports.append(float(recent_lows.iloc[i]))

            # Return top 3 unique levels
            supports = sorted(list(set(supports)))[-3:]
            return supports

        except:
            return []

    def _find_resistance_levels(self, highs: pd.Series, window: int = 20) -> List[float]:
        """Find resistance levels"""
        try:
            # Use rolling max to find local highs
            recent_highs = highs.tail(60)
            rolling_max = recent_highs.rolling(window=window, center=True).max()

            # Find levels where price was rejected
            resistances = []
            for i in range(window, len(recent_highs) - window):
                if recent_highs.iloc[i] == rolling_max.iloc[i]:
                    resistances.append(float(recent_highs.iloc[i]))

            # Return top 3 unique levels
            resistances = sorted(list(set(resistances)))[-3:]
            return resistances

        except:
            return []

    def _analyze_volume(self, volume: pd.Series) -> str:
        """Analyze volume trends"""
        try:
            avg_volume = volume.tail(50).mean()
            recent_volume = volume.tail(5).mean()

            ratio = recent_volume / avg_volume

            if ratio > 1.5:
                return "Above average volume (increasing interest)"
            elif ratio > 1.2:
                return "Slightly elevated volume"
            elif ratio < 0.7:
                return "Below average volume (decreasing interest)"
            else:
                return "Normal volume levels"

        except:
            return "Unable to analyze volume"

    def _identify_chart_patterns(self, hist: pd.DataFrame) -> List[str]:
        """Identify basic chart patterns"""
        patterns = []

        try:
            closes = hist['Close']

            # Check for breakouts
            ma20 = closes.rolling(window=20).mean()
            if closes.iloc[-1] > ma20.iloc[-1] * 1.05:
                patterns.append("Breakout above 20-day MA")
            elif closes.iloc[-1] < ma20.iloc[-1] * 0.95:
                patterns.append("Breakdown below 20-day MA")

            # Check for golden/death cross
            ma50 = closes.rolling(window=50).mean()
            ma200 = closes.rolling(window=200).mean()

            if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-5] <= ma200.iloc[-5]:
                patterns.append("Golden Cross (bullish)")
            elif ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-5] >= ma200.iloc[-5]:
                patterns.append("Death Cross (bearish)")

            # Volume spikes
            if len(hist) > 20:
                avg_vol = hist['Volume'].tail(20).mean()
                if hist['Volume'].iloc[-1] > avg_vol * 2:
                    patterns.append("Volume spike")

            if not patterns:
                patterns.append("No significant patterns detected")

        except:
            patterns.append("Unable to detect patterns")

        return patterns[:5]  # Max 5 patterns

    def _generate_recommendation(
        self,
        trend: TrendDirection,
        rsi: float,
        macd_signal: SignalType,
        moving_averages: Dict[str, float]
    ) -> str:
        """Generate technical recommendation"""
        bullish_signals = 0
        bearish_signals = 0

        # Trend
        if trend == TrendDirection.UPTREND:
            bullish_signals += 2
        elif trend == TrendDirection.DOWNTREND:
            bearish_signals += 2

        # RSI
        if rsi < 30:
            bullish_signals += 1  # Oversold
        elif rsi > 70:
            bearish_signals += 1  # Overbought

        # MACD
        if macd_signal == SignalType.BULLISH:
            bullish_signals += 1
        elif macd_signal == SignalType.BEARISH:
            bearish_signals += 1

        # Moving averages
        if moving_averages['MA50'] > moving_averages['MA200']:
            bullish_signals += 1

        # Generate recommendation
        if bullish_signals >= bearish_signals + 2:
            return "Strong technical setup - consider buying"
        elif bullish_signals > bearish_signals:
            return "Moderately bullish technical setup"
        elif bearish_signals >= bullish_signals + 2:
            return "Weak technical setup - consider selling"
        elif bearish_signals > bullish_signals:
            return "Moderately bearish technical setup"
        else:
            return "Mixed technical signals - neutral outlook"

    def _calculate_score(
        self,
        trend: TrendDirection,
        rsi: float,
        macd_signal: SignalType,
        moving_averages: Dict[str, float],
        volume_analysis: str
    ) -> int:
        """Calculate technical score 0-100"""
        score = 50  # Start neutral

        # Trend (+/- 20 points)
        if trend == TrendDirection.UPTREND:
            score += 20
        elif trend == TrendDirection.DOWNTREND:
            score -= 20

        # RSI (+/- 15 points)
        if 40 <= rsi <= 60:
            score += 10  # Neutral RSI is good
        elif rsi < 30:
            score += 15  # Oversold (potential buy)
        elif rsi > 70:
            score -= 15  # Overbought (potential sell)
        elif 30 <= rsi <= 40:
            score += 5
        elif 60 <= rsi <= 70:
            score -= 5

        # MACD (+/- 15 points)
        if macd_signal == SignalType.BULLISH:
            score += 15
        elif macd_signal == SignalType.BEARISH:
            score -= 15

        # Moving average alignment (+/- 10 points)
        if moving_averages.get('MA50', 0) > moving_averages.get('MA200', 0):
            score += 10
        elif moving_averages.get('MA50', 0) < moving_averages.get('MA200', 0):
            score -= 10

        # Volume (+/- 5 points)
        if "Above average" in volume_analysis:
            score += 5
        elif "Below average" in volume_analysis:
            score -= 5

        # Clamp to 0-100
        return max(0, min(100, score))

    def get_api_call_count(self) -> int:
        """Get number of API calls made"""
        return self.api_calls
