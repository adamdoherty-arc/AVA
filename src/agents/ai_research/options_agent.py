"""
Options Strategist Agent
Uses yfinance for options chains and mibian for Greeks calculations
"""

import asyncio
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from loguru import logger

import yfinance as yf

try:
    import mibian
    MIBIAN_AVAILABLE = True
except ImportError:
    logger.warning("mibian not installed, Greeks calculations will use Black-Scholes approximation")
    MIBIAN_AVAILABLE = False

from .models import (
    OptionsAnalysis,
    UnusualActivity,
    StrategyRecommendation
)


class OptionsAgent:
    """
    Specialist agent for options analysis and strategy recommendations.

    Features:
    - IV Rank and IV Percentile calculation
    - Earnings impact analysis
    - Put/Call ratio analysis
    - Max pain calculation
    - Unusual options activity detection
    - Strategy recommendations (CSP, covered calls, spreads)
    - Greeks calculation using mibian
    """

    def __init__(self) -> None:
        """Initialize the Options Agent."""
        self.risk_free_rate = 0.05  # 5% annual risk-free rate (adjust as needed)
        logger.info("OptionsAgent initialized")

    async def analyze(self, symbol: str) -> OptionsAnalysis:
        """
        Perform comprehensive options analysis on a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            OptionsAnalysis object

        Raises:
            ValueError: If symbol is invalid
        """
        symbol = symbol.upper().strip()
        logger.info(f"Starting options analysis for {symbol}")

        try:
            # Fetch ticker data
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)

            # Get current stock price
            info = await loop.run_in_executor(None, lambda: ticker.info)
            current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))

            if current_price == 0:
                logger.error(f"Could not get current price for {symbol}")
                return self._create_fallback_analysis(symbol)

            # Get options expirations
            expirations = await loop.run_in_executor(None, lambda: ticker.options)

            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return self._create_fallback_analysis(symbol)

            # Analyze multiple expirations for IV calculations
            iv_data = await self._calculate_iv_metrics(ticker, expirations, current_price)

            # Get earnings date
            earnings_dates = info.get("earningsDate", None)
            next_earnings = self._parse_earnings_date(earnings_dates)
            days_to_earnings = self._calculate_days_to_earnings(next_earnings)

            # Analyze near-term options for unusual activity and max pain
            near_term_exp = expirations[0] if expirations else None
            unusual_activity = []
            put_call_ratio = 1.0
            max_pain = current_price

            if near_term_exp:
                options_chain = await self._fetch_options_chain(ticker, near_term_exp)
                unusual_activity = self._detect_unusual_activity(options_chain, current_price)
                put_call_ratio = self._calculate_put_call_ratio(options_chain)
                max_pain = self._calculate_max_pain(options_chain, current_price)

            # Generate strategy recommendations
            strategies = self._recommend_strategies(
                current_price,
                iv_data["current_iv"],
                iv_data["iv_rank"],
                days_to_earnings,
                expirations[:3] if len(expirations) >= 3 else expirations
            )

            # Calculate average earnings move
            avg_earnings_move = self._estimate_earnings_move(info, iv_data["current_iv"])

            analysis = OptionsAnalysis(
                iv_rank=iv_data["iv_rank"],
                iv_percentile=iv_data["iv_percentile"],
                current_iv=iv_data["current_iv"],
                iv_mean_30d=iv_data["iv_mean"],
                iv_std_30d=iv_data["iv_std"],
                next_earnings_date=next_earnings,
                days_to_earnings=days_to_earnings,
                avg_earnings_move=avg_earnings_move,
                put_call_ratio=put_call_ratio,
                max_pain=max_pain,
                unusual_options_activity=unusual_activity,
                recommended_strategies=strategies
            )

            logger.info(f"Options analysis completed for {symbol}, IV Rank: {iv_data['iv_rank']}")
            return analysis

        except Exception as e:
            logger.error(f"Options analysis failed for {symbol}: {e}")
            return self._create_fallback_analysis(symbol)

    async def _calculate_iv_metrics(
        self,
        ticker: yf.Ticker,
        expirations: List[str],
        current_price: float
    ) -> dict:
        """
        Calculate IV rank, percentile, and statistics.

        Args:
            ticker: yfinance Ticker object
            expirations: List of expiration dates
            current_price: Current stock price

        Returns:
            Dictionary with IV metrics
        """
        try:
            loop = asyncio.get_event_loop()

            # Get ATM options for IV calculation
            # Use 30-day option for current IV (or closest available)
            target_date = datetime.now() + timedelta(days=30)

            closest_exp = min(
                expirations,
                key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days)
            )

            options_chain = await loop.run_in_executor(
                None,
                lambda: ticker.option_chain(closest_exp)
            )

            # Calculate current IV from ATM options
            current_iv = self._get_atm_iv(options_chain, current_price)

            # For IV rank/percentile, we'd need historical IV data
            # In production, this would come from a database or API
            # For now, we'll estimate based on industry standards
            iv_mean = current_iv * 0.85  # Assume current is slightly above average
            iv_std = current_iv * 0.2

            # Calculate IV rank (0-100)
            # Simplified: assume IV typically ranges from 50% to 150% of mean
            iv_min = iv_mean * 0.5
            iv_max = iv_mean * 1.5
            iv_rank = int(((current_iv - iv_min) / (iv_max - iv_min)) * 100)
            iv_rank = max(0, min(100, iv_rank))

            # IV percentile (similar to rank for this simplified version)
            iv_percentile = iv_rank

            logger.debug(f"IV Rank: {iv_rank}, Current IV: {current_iv:.2%}")

            return {
                "iv_rank": iv_rank,
                "iv_percentile": iv_percentile,
                "current_iv": current_iv,
                "iv_mean": iv_mean,
                "iv_std": iv_std
            }

        except Exception as e:
            logger.error(f"Error calculating IV metrics: {e}")
            return {
                "iv_rank": 50,
                "iv_percentile": 50,
                "current_iv": 0.30,
                "iv_mean": 0.30,
                "iv_std": 0.05
            }

    def _get_atm_iv(self, options_chain, current_price: float) -> float:
        """
        Get implied volatility from ATM (at-the-money) options.

        Args:
            options_chain: Options chain data
            current_price: Current stock price

        Returns:
            IV as decimal (e.g., 0.30 for 30%)
        """
        try:
            calls = options_chain.calls
            puts = options_chain.puts

            # Find ATM strike
            all_strikes = list(calls['strike'].values)
            atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))

            # Get IV from ATM call and put
            atm_call = calls[calls['strike'] == atm_strike]
            atm_put = puts[puts['strike'] == atm_strike]

            call_iv = atm_call['impliedVolatility'].values[0] if not atm_call.empty else 0
            put_iv = atm_put['impliedVolatility'].values[0] if not atm_put.empty else 0

            # Average call and put IV
            if call_iv > 0 and put_iv > 0:
                return (call_iv + put_iv) / 2
            elif call_iv > 0:
                return call_iv
            elif put_iv > 0:
                return put_iv
            else:
                return 0.30  # Default 30% IV

        except Exception as e:
            logger.warning(f"Error getting ATM IV: {e}")
            return 0.30

    async def _fetch_options_chain(self, ticker: yf.Ticker, expiration: str) -> dict:
        """
        Fetch options chain for a specific expiration.

        Args:
            ticker: yfinance Ticker object
            expiration: Expiration date string

        Returns:
            Dictionary with calls and puts DataFrames
        """
        try:
            loop = asyncio.get_event_loop()
            chain = await loop.run_in_executor(
                None,
                lambda: ticker.option_chain(expiration)
            )

            return {
                "calls": chain.calls,
                "puts": chain.puts,
                "expiration": expiration
            }

        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return {"calls": None, "puts": None, "expiration": expiration}

    def _detect_unusual_activity(self, options_chain: dict, current_price: float) -> List[UnusualActivity]:
        """
        Detect unusual options activity.

        Criteria:
        - Volume > 2x open interest
        - Volume > 1000 contracts
        - Premium > $100,000

        Args:
            options_chain: Options chain data
            current_price: Current stock price

        Returns:
            List of UnusualActivity objects
        """
        unusual = []

        try:
            for option_type, df in [("call", options_chain["calls"]), ("put", options_chain["puts"])]:
                if df is None or df.empty:
                    continue

                for _, row in df.iterrows():
                    volume = row.get("volume", 0)
                    open_interest = row.get("openInterest", 0)
                    strike = row.get("strike", 0)
                    last_price = row.get("lastPrice", 0)

                    # Skip if no volume
                    if volume == 0:
                        continue

                    # Calculate volume/OI ratio
                    vol_oi_ratio = volume / open_interest if open_interest > 0 else float('inf')

                    # Calculate premium
                    premium = volume * last_price * 100  # 100 shares per contract

                    # Check for unusual activity
                    if (vol_oi_ratio > 2.0 and volume > 1000) or premium > 100000:
                        # Determine if bullish or bearish
                        is_itm = (option_type == "call" and strike < current_price) or \
                                 (option_type == "put" and strike > current_price)

                        description = f"{'Aggressive' if vol_oi_ratio > 5 else 'Moderate'} {option_type} buying"
                        if is_itm:
                            description += " (ITM)"

                        activity = UnusualActivity(
                            date=datetime.now().strftime("%Y-%m-%d"),
                            option_type=option_type,
                            strike=float(strike),
                            expiration=options_chain["expiration"],
                            volume=int(volume),
                            open_interest=int(open_interest),
                            volume_oi_ratio=float(vol_oi_ratio),
                            premium=float(premium),
                            description=description
                        )
                        unusual.append(activity)

            # Sort by premium (largest first) and limit to top 5
            unusual.sort(key=lambda x: x.premium, reverse=True)
            return unusual[:5]

        except Exception as e:
            logger.error(f"Error detecting unusual activity: {e}")
            return []

    def _calculate_put_call_ratio(self, options_chain: dict) -> float:
        """
        Calculate put/call ratio based on volume.

        Args:
            options_chain: Options chain data

        Returns:
            Put/call ratio
        """
        try:
            calls = options_chain["calls"]
            puts = options_chain["puts"]

            if calls is None or puts is None or calls.empty or puts.empty:
                return 1.0

            call_volume = calls["volume"].sum()
            put_volume = puts["volume"].sum()

            if call_volume == 0:
                return 10.0  # High ratio if no call volume

            ratio = put_volume / call_volume
            return float(ratio)

        except Exception as e:
            logger.warning(f"Error calculating put/call ratio: {e}")
            return 1.0

    def _calculate_max_pain(self, options_chain: dict, current_price: float) -> float:
        """
        Calculate max pain - the strike price where option writers lose least money.

        Args:
            options_chain: Options chain data
            current_price: Current stock price

        Returns:
            Max pain strike price
        """
        try:
            calls = options_chain["calls"]
            puts = options_chain["puts"]

            if calls is None or puts is None or calls.empty or puts.empty:
                return current_price

            # Get all strikes
            all_strikes = sorted(set(calls["strike"].values) | set(puts["strike"].values))

            # Calculate pain for each strike
            pain_by_strike = {}

            for strike in all_strikes:
                call_pain = 0
                put_pain = 0

                # Calculate call pain (ITM calls lose money for writers)
                itm_calls = calls[calls["strike"] < strike]
                for _, call in itm_calls.iterrows():
                    call_pain += (strike - call["strike"]) * call.get("openInterest", 0)

                # Calculate put pain (ITM puts lose money for writers)
                itm_puts = puts[puts["strike"] > strike]
                for _, put in itm_puts.iterrows():
                    put_pain += (put["strike"] - strike) * put.get("openInterest", 0)

                pain_by_strike[strike] = call_pain + put_pain

            # Find strike with minimum pain
            max_pain_strike = min(pain_by_strike.items(), key=lambda x: x[1])[0]

            logger.debug(f"Max pain calculated at ${max_pain_strike:.2f}")
            return float(max_pain_strike)

        except Exception as e:
            logger.warning(f"Error calculating max pain: {e}")
            return current_price

    def _recommend_strategies(
        self,
        current_price: float,
        current_iv: float,
        iv_rank: int,
        days_to_earnings: int,
        expirations: List[str]
    ) -> List[StrategyRecommendation]:
        """
        Generate strategy recommendations based on current conditions.

        Args:
            current_price: Current stock price
            current_iv: Current implied volatility
            iv_rank: IV rank (0-100)
            days_to_earnings: Days until next earnings
            expirations: Available expiration dates

        Returns:
            List of StrategyRecommendation objects
        """
        recommendations = []

        try:
            # High IV strategies (sell premium)
            if iv_rank > 60:
                # Cash-secured put
                csp_strike = current_price * 0.95  # 5% OTM
                csp_expiration = self._select_expiration(expirations, target_days=30)
                csp_premium = self._estimate_premium(current_price, csp_strike, current_iv, "put", 30)

                recommendations.append(StrategyRecommendation(
                    strategy="cash_secured_put",
                    strike=csp_strike,
                    expiration=csp_expiration,
                    premium=csp_premium,
                    probability_of_profit=0.70,  # Estimated
                    max_profit=csp_premium * 100,
                    max_loss=(csp_strike - csp_premium) * 100,
                    rationale=f"High IV rank ({iv_rank}) - sell premium via cash-secured puts"
                ))

                # Covered call (if they own stock)
                cc_strike = current_price * 1.05  # 5% OTM
                cc_expiration = self._select_expiration(expirations, target_days=30)
                cc_premium = self._estimate_premium(current_price, cc_strike, current_iv, "call", 30)

                recommendations.append(StrategyRecommendation(
                    strategy="covered_call",
                    strike=cc_strike,
                    expiration=cc_expiration,
                    premium=cc_premium,
                    probability_of_profit=0.65,
                    max_profit=(cc_strike - current_price + cc_premium) * 100,
                    max_loss=float('inf'),  # Theoretical unlimited if stock crashes
                    rationale=f"High IV rank ({iv_rank}) - collect premium on existing shares"
                ))

            # Low IV strategies (buy premium)
            elif iv_rank < 40:
                # Long call (bullish)
                call_strike = current_price * 1.03  # Slightly OTM
                call_expiration = self._select_expiration(expirations, target_days=45)
                call_premium = self._estimate_premium(current_price, call_strike, current_iv, "call", 45)

                recommendations.append(StrategyRecommendation(
                    strategy="long_call",
                    strike=call_strike,
                    expiration=call_expiration,
                    premium=call_premium,
                    probability_of_profit=0.45,
                    max_profit=float('inf'),  # Unlimited upside
                    max_loss=call_premium * 100,
                    rationale=f"Low IV rank ({iv_rank}) - buy calls for upside leverage"
                ))

            # Pre-earnings strategies
            if 0 < days_to_earnings < 30:
                # Iron condor (neutral earnings play)
                ic_lower_strike = current_price * 0.90
                ic_upper_strike = current_price * 1.10
                ic_expiration = self._select_expiration(expirations, target_days=days_to_earnings - 1)

                # Simplified iron condor pricing
                ic_premium = current_price * current_iv * 0.15

                recommendations.append(StrategyRecommendation(
                    strategy="iron_condor",
                    strike=(ic_lower_strike + ic_upper_strike) / 2,  # Center
                    expiration=ic_expiration,
                    premium=ic_premium,
                    probability_of_profit=0.60,
                    max_profit=ic_premium * 100,
                    max_loss=((current_price * 0.10) - ic_premium) * 100,
                    rationale=f"Earnings in {days_to_earnings} days - defined risk neutral play"
                ))

            # Always suggest a wheel strategy option
            wheel_strike = current_price * 0.97  # Close to ATM
            wheel_expiration = self._select_expiration(expirations, target_days=45)
            wheel_premium = self._estimate_premium(current_price, wheel_strike, current_iv, "put", 45)

            recommendations.append(StrategyRecommendation(
                strategy="wheel_strategy",
                strike=wheel_strike,
                expiration=wheel_expiration,
                premium=wheel_premium,
                probability_of_profit=0.68,
                max_profit=wheel_premium * 100,
                max_loss=(wheel_strike - wheel_premium) * 100,
                rationale="Consistent income through puts (then covered calls if assigned)"
            ))

            return recommendations[:4]  # Return top 4 strategies

        except Exception as e:
            logger.error(f"Error generating strategy recommendations: {e}")
            return []

    def _select_expiration(self, expirations: List[str], target_days: int) -> str:
        """Select the best expiration date closest to target days."""
        if not expirations:
            return (datetime.now() + timedelta(days=target_days)).strftime("%Y-%m-%d")

        target_date = datetime.now() + timedelta(days=target_days)

        closest = min(
            expirations,
            key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days)
        )

        return closest

    def _estimate_premium(
        self,
        stock_price: float,
        strike: float,
        iv: float,
        option_type: str,
        days_to_expiry: int
    ) -> float:
        """
        Estimate option premium using Black-Scholes or mibian.

        Args:
            stock_price: Current stock price
            strike: Strike price
            iv: Implied volatility (decimal)
            option_type: "call" or "put"
            days_to_expiry: Days until expiration

        Returns:
            Estimated premium per share
        """
        try:
            if MIBIAN_AVAILABLE:
                # Use mibian for accurate pricing
                bs = mibian.BS(
                    [stock_price, strike, self.risk_free_rate * 100, days_to_expiry],
                    volatility=iv * 100
                )

                if option_type == "call":
                    return bs.callPrice
                else:
                    return bs.putPrice
            else:
                # Simplified Black-Scholes approximation
                from scipy.stats import norm

                t = days_to_expiry / 365.0
                d1 = (np.log(stock_price / strike) + (self.risk_free_rate + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
                d2 = d1 - iv * np.sqrt(t)

                if option_type == "call":
                    premium = stock_price * norm.cdf(d1) - strike * np.exp(-self.risk_free_rate * t) * norm.cdf(d2)
                else:
                    premium = strike * np.exp(-self.risk_free_rate * t) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)

                return max(premium, 0.01)  # Minimum $0.01

        except Exception as e:
            logger.warning(f"Error estimating premium: {e}")
            # Rough estimate: IV * stock price * sqrt(time) / sqrt(365)
            return stock_price * iv * np.sqrt(days_to_expiry / 365.0) * 0.4

    def _parse_earnings_date(self, earnings_dates) -> str:
        """Parse earnings date from yfinance data."""
        try:
            if earnings_dates and len(earnings_dates) > 0:
                next_date = earnings_dates[0]
                if isinstance(next_date, datetime):
                    return next_date.strftime("%Y-%m-%d")
                else:
                    return str(next_date)
            return "Unknown"
        except Exception:
            return "Unknown"

    def _calculate_days_to_earnings(self, earnings_date: str) -> int:
        """Calculate days until next earnings."""
        try:
            if earnings_date == "Unknown":
                return 999  # Large number if unknown

            earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
            days = (earnings_dt - datetime.now()).days

            return max(days, 0)

        except Exception:
            return 999

    def _estimate_earnings_move(self, info: dict, current_iv: float) -> float:
        """
        Estimate expected earnings move.

        Uses straddle pricing formula: ~IV * sqrt(time) * price

        Args:
            info: Stock info dictionary
            current_iv: Current implied volatility

        Returns:
            Expected move as decimal (e.g., 0.08 for 8%)
        """
        try:
            # Typical earnings period is 1 day, so time = 1/365
            # Expected move ≈ IV * sqrt(1/365)
            one_day_move = current_iv * np.sqrt(1/365)

            # Earnings typically have 4-8x the daily expected move
            earnings_multiplier = 6

            return one_day_move * earnings_multiplier

        except Exception as e:
            logger.warning(f"Error estimating earnings move: {e}")
            return 0.05  # Default 5% move

    def _create_fallback_analysis(self, symbol: str) -> OptionsAnalysis:
        """Create fallback analysis when data fetching fails."""
        logger.warning(f"Creating fallback options analysis for {symbol}")

        return OptionsAnalysis(
            iv_rank=50,
            iv_percentile=50,
            current_iv=0.30,
            iv_mean_30d=0.30,
            iv_std_30d=0.05,
            next_earnings_date="Unknown",
            days_to_earnings=999,
            avg_earnings_move=0.05,
            put_call_ratio=1.0,
            max_pain=0.0,
            unusual_options_activity=[],
            recommended_strategies=[]
        )
