"""
Technical Analyst Agent
Uses yfinance for price data and pandas_ta for technical indicators
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger

try:
    import pandas_ta as ta
except ImportError:
    logger.warning("pandas_ta not installed, will use custom indicator calculations")
    ta = None

from .models import TechnicalAnalysis, TrendDirection, SignalType


class TechnicalAgent:
    """
    Specialist agent for technical analysis using yfinance and pandas_ta.

    Features:
    - RSI calculation and interpretation
    - MACD signal detection
    - Bollinger Bands
    - Multiple moving averages
    - Support/Resistance levels
    - Volume analysis
    - Chart pattern recognition
    """

    def __init__(self) -> None:
        """Initialize the Technical Agent."""
        self.lookback_days = 200  # Fetch 200 days for MA200
        logger.info("TechnicalAgent initialized")

    async def analyze(self, symbol: str) -> TechnicalAnalysis:
        """
        Perform comprehensive technical analysis on a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            TechnicalAnalysis object with all indicators

        Raises:
            ValueError: If symbol is invalid
            RuntimeError: If data fetching fails
        """
        symbol = symbol.upper().strip()
        logger.info(f"Starting technical analysis for {symbol}")

        try:
            # Fetch historical data
            df = await self._fetch_price_data(symbol)

            if df is None or df.empty:
                logger.error(f"No price data available for {symbol}")
                return self._create_fallback_analysis(symbol)

            # Calculate all indicators
            indicators = self._calculate_indicators(df)

            # Determine trend
            trend = self._determine_trend(indicators)

            # Find support/resistance
            support, resistance = self._find_support_resistance(df)

            # Analyze volume
            volume_analysis = self._analyze_volume(df)

            # Detect chart patterns
            patterns = self._detect_patterns(df, indicators)

            # Generate recommendation
            recommendation = self._generate_recommendation(indicators, trend)

            # Calculate overall score
            score = self._calculate_score(indicators, trend)

            analysis = TechnicalAnalysis(
                score=score,
                trend=trend,
                rsi=indicators["rsi"],
                macd_signal=indicators["macd_signal"],
                support_levels=support,
                resistance_levels=resistance,
                moving_averages=indicators["moving_averages"],
                bollinger_bands=indicators["bollinger_bands"],
                volume_analysis=volume_analysis,
                chart_patterns=patterns,
                recommendation=recommendation
            )

            logger.info(f"Technical analysis completed for {symbol} with score {score}")
            return analysis

        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            return self._create_fallback_analysis(symbol)

    async def _fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data from yfinance.

        Args:
            symbol: Stock ticker

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Run yfinance download in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)

            # Fetch data for lookback period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            df = await loop.run_in_executor(
                None,
                lambda: ticker.history(start=start_date, end=end_date)
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            logger.debug(f"Fetched {len(df)} days of data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """
        Calculate all technical indicators.

        Args:
            df: Price DataFrame with OHLCV data

        Returns:
            Dictionary of indicator values
        """
        indicators = {}

        try:
            # Current price
            current_price = float(df['Close'].iloc[-1])

            # RSI (14-period)
            indicators["rsi"] = self._calculate_rsi(df['Close'], period=14)

            # MACD
            macd_data = self._calculate_macd(df['Close'])
            indicators["macd"] = macd_data["macd"]
            indicators["macd_signal_line"] = macd_data["signal"]
            indicators["macd_histogram"] = macd_data["histogram"]
            indicators["macd_signal"] = self._interpret_macd(macd_data)

            # Moving Averages
            indicators["moving_averages"] = {
                "MA20": float(df['Close'].rolling(window=20).mean().iloc[-1]),
                "MA50": float(df['Close'].rolling(window=50).mean().iloc[-1]),
                "MA200": float(df['Close'].rolling(window=200).mean().iloc[-1]) if len(df) >= 200 else current_price,
                "current_price": current_price
            }

            # Bollinger Bands (20-period, 2 std dev)
            bb = self._calculate_bollinger_bands(df['Close'])
            indicators["bollinger_bands"] = bb

            # Volume indicators
            indicators["avg_volume_20d"] = float(df['Volume'].rolling(window=20).mean().iloc[-1])
            indicators["current_volume"] = float(df['Volume'].iloc[-1])

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Return minimal indicators
            return {
                "rsi": 50.0,
                "macd_signal": SignalType.NEUTRAL,
                "moving_averages": {"MA20": 0, "MA50": 0, "MA200": 0, "current_price": 0},
                "bollinger_bands": {"upper": 0, "middle": 0, "lower": 0}
            }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Series of closing prices
            period: RSI period (default 14)

        Returns:
            RSI value (0-100)
        """
        if ta is not None:
            try:
                rsi_series = ta.rsi(prices, length=period)
                return float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
            except Exception:
                pass

        # Manual RSI calculation
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return 50.0

    def _calculate_macd(self, prices: pd.Series) -> dict:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of closing prices

        Returns:
            Dictionary with macd, signal, and histogram values
        """
        if ta is not None:
            try:
                macd_df = ta.macd(prices, fast=12, slow=26, signal=9)
                return {
                    "macd": float(macd_df[f'MACD_12_26_9'].iloc[-1]),
                    "signal": float(macd_df[f'MACDs_12_26_9'].iloc[-1]),
                    "histogram": float(macd_df[f'MACDh_12_26_9'].iloc[-1])
                }
            except Exception:
                pass

        # Manual MACD calculation
        try:
            ema_12 = prices.ewm(span=12, adjust=False).mean()
            ema_26 = prices.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal

            return {
                "macd": float(macd.iloc[-1]),
                "signal": float(signal.iloc[-1]),
                "histogram": float(histogram.iloc[-1])
            }

        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    def _interpret_macd(self, macd_data: dict) -> SignalType:
        """Interpret MACD signal."""
        histogram = macd_data["histogram"]

        if histogram > 0.5:
            return SignalType.BULLISH
        elif histogram < -0.5:
            return SignalType.BEARISH
        else:
            return SignalType.NEUTRAL

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of closing prices
            period: Period for moving average
            std_dev: Number of standard deviations

        Returns:
            Dictionary with upper, middle, lower bands
        """
        if ta is not None:
            try:
                bb = ta.bbands(prices, length=period, std=std_dev)
                return {
                    "upper": float(bb[f'BBU_{period}_{std_dev}.0'].iloc[-1]),
                    "middle": float(bb[f'BBM_{period}_{std_dev}.0'].iloc[-1]),
                    "lower": float(bb[f'BBL_{period}_{std_dev}.0'].iloc[-1])
                }
            except Exception:
                pass

        # Manual calculation
        try:
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)

            return {
                "upper": float(upper.iloc[-1]),
                "middle": float(middle.iloc[-1]),
                "lower": float(lower.iloc[-1])
            }

        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
            current = float(prices.iloc[-1])
            return {"upper": current * 1.1, "middle": current, "lower": current * 0.9}

    def _determine_trend(self, indicators: dict) -> TrendDirection:
        """
        Determine overall trend based on multiple indicators.

        Args:
            indicators: Dictionary of calculated indicators

        Returns:
            TrendDirection enum
        """
        ma = indicators["moving_averages"]
        current = ma["current_price"]

        # Check if price is above/below key moving averages
        above_ma20 = current > ma["MA20"]
        above_ma50 = current > ma["MA50"]
        above_ma200 = current > ma["MA200"] if ma["MA200"] > 0 else True

        # MA alignment (bullish when short-term > long-term)
        ma_aligned_bullish = ma["MA20"] > ma["MA50"] > ma["MA200"] if ma["MA200"] > 0 else ma["MA20"] > ma["MA50"]
        ma_aligned_bearish = ma["MA20"] < ma["MA50"] < ma["MA200"] if ma["MA200"] > 0 else ma["MA20"] < ma["MA50"]

        # RSI trend
        rsi = indicators["rsi"]
        rsi_bullish = rsi > 50 and rsi < 70
        rsi_bearish = rsi < 50 and rsi > 30

        # MACD trend
        macd_bullish = indicators["macd_signal"] == SignalType.BULLISH
        macd_bearish = indicators["macd_signal"] == SignalType.BEARISH

        # Score the trend
        bullish_score = sum([above_ma20, above_ma50, above_ma200, ma_aligned_bullish, rsi_bullish, macd_bullish])
        bearish_score = sum([not above_ma20, not above_ma50, not above_ma200, ma_aligned_bearish, rsi_bearish, macd_bearish])

        if bullish_score >= 4:
            return TrendDirection.UPTREND
        elif bearish_score >= 4:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS

    def _find_support_resistance(self, df: pd.DataFrame, num_levels: int = 3) -> tuple[List[float], List[float]]:
        """
        Find support and resistance levels using pivot points.

        Args:
            df: Price DataFrame
            num_levels: Number of levels to return

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        try:
            # Use last 50 days for S/R calculation
            recent_df = df.tail(50)

            # Find local minima (support) and maxima (resistance)
            highs = recent_df['High'].values
            lows = recent_df['Low'].values

            # Simple pivot point method
            support_candidates = []
            resistance_candidates = []

            for i in range(2, len(lows) - 2):
                # Support: local minimum
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    support_candidates.append(lows[i])

                # Resistance: local maximum
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    resistance_candidates.append(highs[i])

            # Get current price
            current_price = float(df['Close'].iloc[-1])

            # Filter and sort support levels (below current price)
            support = sorted([s for s in support_candidates if s < current_price], reverse=True)[:num_levels]

            # Filter and sort resistance levels (above current price)
            resistance = sorted([r for r in resistance_candidates if r > current_price])[:num_levels]

            # If not enough levels found, use percentages
            if len(support) < num_levels:
                support.extend([current_price * (1 - 0.05 * i) for i in range(1, num_levels - len(support) + 1)])

            if len(resistance) < num_levels:
                resistance.extend([current_price * (1 + 0.05 * i) for i in range(1, num_levels - len(resistance) + 1)])

            return support[:num_levels], resistance[:num_levels]

        except Exception as e:
            logger.warning(f"Error finding support/resistance: {e}")
            current = float(df['Close'].iloc[-1])
            return (
                [current * 0.95, current * 0.90, current * 0.85],
                [current * 1.05, current * 1.10, current * 1.15]
            )

    def _analyze_volume(self, df: pd.DataFrame) -> str:
        """
        Analyze volume patterns.

        Args:
            df: Price DataFrame

        Returns:
            Volume analysis description
        """
        try:
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            if volume_ratio > 2.0:
                return "Exceptionally high volume - strong interest"
            elif volume_ratio > 1.5:
                return "Above average volume - increasing interest"
            elif volume_ratio > 0.8:
                return "Normal volume - typical trading activity"
            else:
                return "Below average volume - low interest"

        except Exception as e:
            logger.warning(f"Volume analysis failed: {e}")
            return "Volume data unavailable"

    def _detect_patterns(self, df: pd.DataFrame, indicators: dict) -> List[str]:
        """
        Detect common chart patterns.

        Args:
            df: Price DataFrame
            indicators: Calculated indicators

        Returns:
            List of detected patterns
        """
        patterns = []

        try:
            current_price = float(df['Close'].iloc[-1])
            bb = indicators["bollinger_bands"]

            # Bollinger Band squeeze
            band_width = (bb["upper"] - bb["lower"]) / bb["middle"] if bb["middle"] > 0 else 0
            if band_width < 0.05:
                patterns.append("Bollinger Band Squeeze - potential breakout coming")

            # Price near bands
            if current_price > bb["upper"]:
                patterns.append("Price above upper Bollinger Band - potentially overbought")
            elif current_price < bb["lower"]:
                patterns.append("Price below lower Bollinger Band - potentially oversold")

            # Golden Cross / Death Cross
            ma = indicators["moving_averages"]
            if ma["MA50"] > ma["MA200"] and ma["MA50"] > 0 and ma["MA200"] > 0:
                patterns.append("Golden Cross - bullish long-term signal")
            elif ma["MA50"] < ma["MA200"] and ma["MA50"] > 0 and ma["MA200"] > 0:
                patterns.append("Death Cross - bearish long-term signal")

            # RSI divergence detection would require more complex analysis
            rsi = indicators["rsi"]
            if rsi > 70:
                patterns.append("RSI overbought - potential reversal")
            elif rsi < 30:
                patterns.append("RSI oversold - potential bounce")

            if not patterns:
                patterns.append("No significant patterns detected")

        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            patterns.append("Pattern analysis unavailable")

        return patterns

    def _generate_recommendation(self, indicators: dict, trend: TrendDirection) -> str:
        """Generate trading recommendation based on technical analysis."""
        rsi = indicators["rsi"]
        macd_signal = indicators["macd_signal"]
        ma = indicators["moving_averages"]

        if trend == TrendDirection.UPTREND and rsi < 70:
            return "Technical indicators suggest continuation of uptrend. Consider buying on pullbacks to support."
        elif trend == TrendDirection.DOWNTREND and rsi > 30:
            return "Downtrend in progress. Wait for trend reversal signals before entering long positions."
        elif rsi > 70:
            return "Overbought conditions. Consider taking profits or waiting for pullback."
        elif rsi < 30:
            return "Oversold conditions. Watch for reversal signals for potential entry."
        elif macd_signal == SignalType.BULLISH:
            return "MACD showing bullish momentum. Monitor for entry opportunities."
        elif macd_signal == SignalType.BEARISH:
            return "MACD showing bearish momentum. Exercise caution on long positions."
        else:
            return "Mixed signals. Wait for clearer trend confirmation before taking positions."

    def _calculate_score(self, indicators: dict, trend: TrendDirection) -> int:
        """
        Calculate overall technical score (0-100).

        Scoring methodology:
        - Trend strength (30 points)
        - RSI position (25 points)
        - MACD signal (25 points)
        - MA alignment (20 points)
        """
        score = 0

        # Trend score (30 points)
        if trend == TrendDirection.UPTREND:
            score += 30
        elif trend == TrendDirection.SIDEWAYS:
            score += 15
        else:
            score += 0

        # RSI score (25 points) - prefer 40-60 range
        rsi = indicators["rsi"]
        if 40 <= rsi <= 60:
            score += 25
        elif 30 <= rsi <= 70:
            score += 20
        elif rsi < 30:
            score += 15  # Oversold can be opportunity
        else:
            score += 10  # Overbought is risky

        # MACD score (25 points)
        if indicators["macd_signal"] == SignalType.BULLISH:
            score += 25
        elif indicators["macd_signal"] == SignalType.NEUTRAL:
            score += 15
        else:
            score += 5

        # MA alignment score (20 points)
        ma = indicators["moving_averages"]
        current = ma["current_price"]

        ma_score = 0
        if current > ma["MA20"]:
            ma_score += 7
        if current > ma["MA50"]:
            ma_score += 7
        if ma["MA200"] > 0 and current > ma["MA200"]:
            ma_score += 6

        score += ma_score

        return min(score, 100)

    def _create_fallback_analysis(self, symbol: str) -> TechnicalAnalysis:
        """Create fallback analysis when data fetching fails."""
        logger.warning(f"Creating fallback technical analysis for {symbol}")

        return TechnicalAnalysis(
            score=50,
            trend=TrendDirection.SIDEWAYS,
            rsi=50.0,
            macd_signal=SignalType.NEUTRAL,
            support_levels=[0.0, 0.0, 0.0],
            resistance_levels=[0.0, 0.0, 0.0],
            moving_averages={"MA20": 0, "MA50": 0, "MA200": 0, "current_price": 0},
            bollinger_bands={"upper": 0, "middle": 0, "lower": 0},
            volume_analysis="Data unavailable",
            chart_patterns=["Unable to fetch price data"],
            recommendation="Insufficient data for technical recommendation"
        )
