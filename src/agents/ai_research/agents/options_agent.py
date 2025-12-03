"""
Options Strategist Agent
Analyzes IV, options flow, and recommends strategies
"""

import logging
from typing import List
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agents.ai_research.models import (
    OptionsAnalysis,
    UnusualActivity,
    StrategyRecommendation
)

logger = logging.getLogger(__name__)


class OptionsAgent:
    """
    Options analysis specialist

    Analyzes:
    - Implied volatility (IV rank, IV percentile)
    - Put/call ratios and max pain
    - Unusual options activity
    - Options strategies for wheel traders
    - Earnings-related volatility
    """

    def __init__(self) -> None:
        self.api_calls = 0

    async def analyze(self, symbol: str) -> OptionsAnalysis:
        """
        Perform options analysis

        Args:
            symbol: Stock ticker symbol

        Returns:
            OptionsAnalysis object
        """
        self.api_calls = 0
        logger.info(f"Starting options analysis for {symbol}")

        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            self.api_calls += 1

            # Get options data
            options_dates = ticker.options
            self.api_calls += 1

            if not options_dates:
                raise ValueError(f"No options data available for {symbol}")

            # Analyze IV
            current_iv = self._get_current_iv(ticker, options_dates)
            iv_history = self._get_iv_history(ticker)
            iv_rank, iv_percentile = self._calculate_iv_rank(current_iv, iv_history)
            iv_mean = iv_history.mean() if len(iv_history) > 0 else current_iv
            iv_std = iv_history.std() if len(iv_history) > 0 else 0.0

            # Earnings analysis
            next_earnings_date = self._get_next_earnings_date(info)
            days_to_earnings = self._calculate_days_to_earnings(next_earnings_date)
            avg_earnings_move = self._estimate_earnings_move(ticker, current_iv)

            # Options metrics
            put_call_ratio = self._calculate_put_call_ratio(ticker, options_dates[0] if options_dates else None)
            max_pain = self._calculate_max_pain(ticker, options_dates[0] if options_dates else None)

            # Unusual activity
            unusual_activity = self._detect_unusual_activity(ticker, options_dates[:3] if len(options_dates) >= 3 else options_dates)

            # Strategy recommendations
            current_price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
            strategies = self._recommend_strategies(
                symbol, current_price, current_iv, iv_rank,
                days_to_earnings, put_call_ratio, ticker, options_dates
            )

            return OptionsAnalysis(
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                current_iv=current_iv,
                iv_mean_30d=iv_mean,
                iv_std_30d=iv_std,
                next_earnings_date=next_earnings_date,
                days_to_earnings=days_to_earnings,
                avg_earnings_move=avg_earnings_move,
                put_call_ratio=put_call_ratio,
                max_pain=max_pain,
                unusual_options_activity=unusual_activity,
                recommended_strategies=strategies
            )

        except Exception as e:
            logger.error(f"Options analysis failed for {symbol}: {str(e)}")
            raise

    def _get_current_iv(self, ticker, options_dates: list) -> float:
        """Get current implied volatility"""
        try:
            if not options_dates:
                return 0.0

            # Get ATM options from nearest expiration
            nearest_date = options_dates[0]
            chain = ticker.option_chain(nearest_date)
            self.api_calls += 1

            calls = chain.calls
            if calls.empty:
                return 0.0

            # Get ATM call IV
            atm_calls = calls[calls['inTheMoney'] is False].head(3)
            if not atm_calls.empty and 'impliedVolatility' in atm_calls.columns:
                return float(atm_calls['impliedVolatility'].mean())

            return 0.0

        except Exception as e:
            logger.warning(f"Failed to get current IV: {str(e)}")
            return 0.0

    def _get_iv_history(self, ticker) -> pd.Series:
        """Get historical IV (30 days)"""
        try:
            # Use historical volatility as proxy
            hist = ticker.history(period="1mo")
            self.api_calls += 1

            if hist.empty:
                return pd.Series([])

            # Calculate historical volatility
            returns = hist['Close'].pct_change().dropna()
            # Annualized volatility
            hv = returns.std() * np.sqrt(252)

            # Return series of HV (simplified - in production, use actual IV history)
            return pd.Series([hv] * 30)

        except:
            return pd.Series([])

    def _calculate_iv_rank(self, current_iv: float, iv_history: pd.Series) -> tuple[int, int]:
        """Calculate IV rank and IV percentile"""
        try:
            if len(iv_history) == 0 or current_iv == 0:
                return 50, 50

            iv_min = iv_history.min()
            iv_max = iv_history.max()

            # IV Rank: where current IV sits in 52-week range
            if iv_max == iv_min:
                iv_rank = 50
            else:
                iv_rank = int(((current_iv - iv_min) / (iv_max - iv_min)) * 100)

            # IV Percentile: % of days IV was below current
            iv_percentile = int((iv_history < current_iv).sum() / len(iv_history) * 100)

            return iv_rank, iv_percentile

        except:
            return 50, 50

    def _get_next_earnings_date(self, info: dict) -> str:
        """Get next earnings date"""
        try:
            earnings_date = info.get('earningsDate')
            if earnings_date:
                if isinstance(earnings_date, list) and len(earnings_date) > 0:
                    return earnings_date[0].strftime('%Y-%m-%d')
                elif hasattr(earnings_date, 'strftime'):
                    return earnings_date.strftime('%Y-%m-%d')
            return "Unknown"
        except:
            return "Unknown"

    def _calculate_days_to_earnings(self, earnings_date: str) -> int:
        """Calculate days until next earnings"""
        try:
            if earnings_date == "Unknown":
                return 999

            earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
            days = (earnings_dt - datetime.now()).days

            return max(0, days)

        except:
            return 999

    def _estimate_earnings_move(self, ticker, current_iv: float) -> float:
        """Estimate expected earnings move based on IV"""
        try:
            # Simplified calculation: expected move = stock price * IV * sqrt(days/365)
            # For earnings (typically 1-day event), this gives rough estimate
            info = ticker.info
            price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)

            if price == 0 or current_iv == 0:
                return 0.0

            # Expected 1-day move
            expected_move_pct = current_iv * np.sqrt(1/365)

            return expected_move_pct

        except:
            return 0.0

    def _calculate_put_call_ratio(self, ticker, expiration_date: str) -> float:
        """Calculate put/call ratio"""
        try:
            if not expiration_date:
                return 1.0

            chain = ticker.option_chain(expiration_date)
            self.api_calls += 1

            puts = chain.puts
            calls = chain.calls

            if puts.empty or calls.empty:
                return 1.0

            put_volume = puts['volume'].sum()
            call_volume = calls['volume'].sum()

            if call_volume == 0:
                return 1.0

            return put_volume / call_volume

        except:
            return 1.0

    def _calculate_max_pain(self, ticker, expiration_date: str) -> float:
        """Calculate max pain strike"""
        try:
            if not expiration_date:
                return 0.0

            chain = ticker.option_chain(expiration_date)
            self.api_calls += 1

            puts = chain.puts[['strike', 'openInterest']]
            calls = chain.calls[['strike', 'openInterest']]

            if puts.empty or calls.empty:
                return 0.0

            # Get all unique strikes
            all_strikes = sorted(set(puts['strike'].tolist() + calls['strike'].tolist()))

            # Calculate total loss at each strike
            min_loss = float('inf')
            max_pain_strike = 0.0

            for strike in all_strikes:
                # Loss for call holders
                call_loss = calls[calls['strike'] < strike].apply(
                    lambda row: (strike - row['strike']) * row['openInterest'] * 100, axis=1
                ).sum()

                # Loss for put holders
                put_loss = puts[puts['strike'] > strike].apply(
                    lambda row: (row['strike'] - strike) * row['openInterest'] * 100, axis=1
                ).sum()

                total_loss = call_loss + put_loss

                if total_loss < min_loss:
                    min_loss = total_loss
                    max_pain_strike = strike

            return max_pain_strike

        except Exception as e:
            logger.warning(f"Failed to calculate max pain: {str(e)}")
            return 0.0

    def _detect_unusual_activity(self, ticker, expiration_dates: list) -> List[UnusualActivity]:
        """Detect unusual options activity"""
        unusual_activities = []

        try:
            for exp_date in expiration_dates[:3]:  # Check first 3 expirations
                try:
                    chain = ticker.option_chain(exp_date)
                    self.api_calls += 1

                    # Check calls
                    for _, row in chain.calls.iterrows():
                        volume = row.get('volume', 0)
                        oi = row.get('openInterest', 1)

                        if volume > 0 and oi > 0:
                            vol_oi_ratio = volume / oi

                            # Flag if volume > 2x open interest
                            if vol_oi_ratio > 2.0 and volume > 100:
                                unusual_activities.append(UnusualActivity(
                                    date=datetime.now().strftime('%Y-%m-%d'),
                                    option_type='call',
                                    strike=float(row.get('strike', 0)),
                                    expiration=exp_date,
                                    volume=int(volume),
                                    open_interest=int(oi),
                                    volume_oi_ratio=vol_oi_ratio,
                                    premium=float(row.get('lastPrice', 0)),
                                    description=f"High call volume at ${row.get('strike', 0)} strike"
                                ))

                    # Check puts
                    for _, row in chain.puts.iterrows():
                        volume = row.get('volume', 0)
                        oi = row.get('openInterest', 1)

                        if volume > 0 and oi > 0:
                            vol_oi_ratio = volume / oi

                            if vol_oi_ratio > 2.0 and volume > 100:
                                unusual_activities.append(UnusualActivity(
                                    date=datetime.now().strftime('%Y-%m-%d'),
                                    option_type='put',
                                    strike=float(row.get('strike', 0)),
                                    expiration=exp_date,
                                    volume=int(volume),
                                    open_interest=int(oi),
                                    volume_oi_ratio=vol_oi_ratio,
                                    premium=float(row.get('lastPrice', 0)),
                                    description=f"High put volume at ${row.get('strike', 0)} strike"
                                ))

                except Exception as e:
                    logger.warning(f"Failed to check expiration {exp_date}: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to detect unusual activity: {str(e)}")

        # Return top 5 by volume
        return sorted(unusual_activities, key=lambda x: x.volume, reverse=True)[:5]

    def _recommend_strategies(
        self,
        symbol: str,
        current_price: float,
        current_iv: float,
        iv_rank: int,
        days_to_earnings: int,
        put_call_ratio: float,
        ticker,
        options_dates: list
    ) -> List[StrategyRecommendation]:
        """Recommend options strategies for wheel traders"""
        strategies = []

        try:
            if not options_dates or current_price == 0:
                return strategies

            # Get 30-45 DTE options
            target_dte = 35
            best_expiration = None
            min_dte_diff = float('inf')

            for exp_date in options_dates:
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                dte = (exp_dt - datetime.now()).days

                if abs(dte - target_dte) < min_dte_diff:
                    min_dte_diff = abs(dte - target_dte)
                    best_expiration = exp_date

            if not best_expiration:
                return strategies

            chain = ticker.option_chain(best_expiration)
            self.api_calls += 1

            exp_dt = datetime.strptime(best_expiration, '%Y-%m-%d')
            actual_dte = (exp_dt - datetime.now()).days

            # Cash-Secured Put (sell puts)
            if iv_rank > 50:  # High IV - good for selling
                puts = chain.puts

                # Find ~0.30 delta put (OTM)
                target_strike = current_price * 0.95  # ~5% OTM
                put_options = puts[
                    (puts['strike'] < current_price) &
                    (puts['strike'] >= target_strike - 5) &
                    (puts['strike'] <= target_strike + 5)
                ]

                if not put_options.empty:
                    best_put = put_options.iloc[0]
                    premium = float(best_put.get('lastPrice', 0)) * 100

                    if premium > 0:
                        strategies.append(StrategyRecommendation(
                            strategy='cash_secured_put',
                            strike=float(best_put.get('strike', 0)),
                            expiration=best_expiration,
                            premium=premium,
                            probability_of_profit=0.70,  # Approximate for 0.30 delta
                            max_profit=premium,
                            max_loss=float(best_put.get('strike', 0)) * 100 - premium,
                            rationale=f"High IV rank ({iv_rank}) makes selling puts attractive. {actual_dte} DTE provides good theta decay."
                        ))

            # Covered Call (sell calls on existing stock)
            if iv_rank > 40:
                calls = chain.calls

                # Find ~0.30 delta call (OTM)
                target_strike = current_price * 1.05  # ~5% OTM
                call_options = calls[
                    (calls['strike'] > current_price) &
                    (calls['strike'] >= target_strike - 5) &
                    (calls['strike'] <= target_strike + 5)
                ]

                if not call_options.empty:
                    best_call = call_options.iloc[0]
                    premium = float(best_call.get('lastPrice', 0)) * 100

                    if premium > 0:
                        strategies.append(StrategyRecommendation(
                            strategy='covered_call',
                            strike=float(best_call.get('strike', 0)),
                            expiration=best_expiration,
                            premium=premium,
                            probability_of_profit=0.70,
                            max_profit=premium + (float(best_call.get('strike', 0)) - current_price) * 100,
                            max_loss=current_price * 100 - premium,
                            rationale=f"Elevated IV ({iv_rank} rank) good for income generation. Strike provides {((float(best_call.get('strike', 0)) / current_price - 1) * 100):.1f}% upside."
                        ))

            # Add earnings-based recommendation
            if days_to_earnings < 14 and days_to_earnings > 0:
                strategies.append(StrategyRecommendation(
                    strategy='wait_for_earnings',
                    strike=0.0,
                    expiration=best_expiration,
                    premium=0.0,
                    probability_of_profit=0.0,
                    max_profit=0.0,
                    max_loss=0.0,
                    rationale=f"Earnings in {days_to_earnings} days. Consider waiting for post-earnings IV crush before selling premium."
                ))

        except Exception as e:
            logger.warning(f"Failed to recommend strategies: {str(e)}")

        return strategies[:3]  # Return top 3 strategies

    def get_api_call_count(self) -> int:
        """Get number of API calls made"""
        return self.api_calls
