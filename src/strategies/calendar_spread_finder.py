"""
Calendar Spread Finder
Finds and evaluates calendar spread opportunities
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
from scipy.stats import norm

from .calendar_spread_models import CalendarSpreadOpportunity, OptionContract


class CalendarSpreadFinder:
    """Find and evaluate calendar spread opportunities"""

    def __init__(self) -> None:
        self.min_volume = 10  # Minimum daily volume
        self.min_oi = 50  # Minimum open interest
        self.strike_range_pct = 0.15  # ±15% from current price

    def find_opportunities(self, symbol: str,
                          near_dte_range: tuple = (30, 45),
                          far_dte_range: tuple = (60, 90),
                          option_type: str = 'call') -> List[CalendarSpreadOpportunity]:
        """
        Find all calendar spread opportunities for a symbol

        Args:
            symbol: Stock ticker
            near_dte_range: (min_dte, max_dte) for near-term leg
            far_dte_range: (min_dte, max_dte) for far-term leg
            option_type: 'call' or 'put'

        Returns:
            List of opportunities sorted by score
        """
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            if hist.empty:
                return []

            stock_price = hist['Close'].iloc[-1]

            # Get options chains
            near_options = self._get_options_in_dte_range(ticker, stock_price, near_dte_range, option_type)
            far_options = self._get_options_in_dte_range(ticker, stock_price, far_dte_range, option_type)

            if not near_options or not far_options:
                return []

            # Find matching strikes
            opportunities = []

            # Get strikes within range of current price
            min_strike = stock_price * (1 - self.strike_range_pct)
            max_strike = stock_price * (1 + self.strike_range_pct)

            # Group options by strike
            near_by_strike = {opt.strike: opt for opt in near_options
                            if min_strike <= opt.strike <= max_strike}
            far_by_strike = {opt.strike: opt for opt in far_options
                           if min_strike <= opt.strike <= max_strike}

            # Find matching strikes
            common_strikes = set(near_by_strike.keys()) & set(far_by_strike.keys())

            for strike in common_strikes:
                near_opt = near_by_strike[strike]
                far_opt = far_by_strike[strike]

                # Calculate spread metrics
                opportunity = self._evaluate_calendar_spread(
                    symbol, stock_price, near_opt, far_opt
                )

                if opportunity and opportunity.opportunity_score > 30:  # Minimum score threshold
                    opportunities.append(opportunity)

            # Sort by opportunity score
            opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)

            # Assign ranks
            for idx, opp in enumerate(opportunities, 1):
                opp.rank = idx

            return opportunities

        except Exception as e:
            print(f"Error finding opportunities for {symbol}: {e}")
            return []

    def _get_options_in_dte_range(self, ticker, stock_price: float,
                                  dte_range: tuple, option_type: str) -> List[OptionContract]:
        """Get options contracts within DTE range"""
        options_list = []

        try:
            expirations = ticker.options
            today = datetime.now().date()

            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                dte = (exp_date - today).days

                if dte_range[0] <= dte <= dte_range[1]:
                    # Get options chain for this expiration
                    chain = ticker.option_chain(exp_str)

                    # Select calls or puts
                    if option_type == 'call':
                        df = chain.calls
                    else:
                        df = chain.puts

                    # Filter by liquidity
                    df = df[(df['volume'] >= self.min_volume) &
                           (df['openInterest'] >= self.min_oi)]

                    # Convert to OptionContract objects
                    for _, row in df.iterrows():
                        try:
                            # Calculate Greeks if not available
                            if 'delta' not in row or pd.isna(row.get('delta')):
                                # Simple delta approximation for calls
                                if option_type == 'call':
                                    moneyness = stock_price / float(row['strike'])
                                    delta = 0.5 if abs(moneyness - 1.0) < 0.01 else (0.6 if moneyness > 1.0 else 0.4)
                                else:
                                    delta = -0.5
                            else:
                                delta = float(row['delta'])

                            # Simple theta approximation if not available
                            if 'theta' not in row or pd.isna(row.get('theta')):
                                theta = -float(row['ask']) / dte if dte > 0 else -0.01
                            else:
                                theta = float(row['theta'])

                            options_list.append(OptionContract(
                                symbol=ticker.ticker,
                                strike=float(row['strike']),
                                expiration=exp_date,
                                option_type=option_type,
                                premium=(float(row['bid']) + float(row['ask'])) / 2,
                                bid=float(row['bid']),
                                ask=float(row['ask']),
                                volume=int(row['volume']) if pd.notna(row['volume']) else 0,
                                open_interest=int(row['openInterest']) if pd.notna(row['openInterest']) else 0,
                                implied_volatility=float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else 0.3,
                                delta=delta,
                                theta=theta,
                                gamma=float(row.get('gamma', 0.001)) if pd.notna(row.get('gamma')) else 0.001,
                                vega=float(row.get('vega', 0.01)) if pd.notna(row.get('vega')) else 0.01,
                                dte=dte
                            ))
                        except Exception as e:
                            continue

        except Exception as e:
            print(f"Error getting options chain: {e}")

        return options_list

    def _evaluate_calendar_spread(self, symbol: str, stock_price: float,
                                  near_opt: OptionContract,
                                  far_opt: OptionContract) -> Optional[CalendarSpreadOpportunity]:
        """Evaluate a specific calendar spread"""
        try:
            # Net debit (cost to open)
            net_debit = (far_opt.premium - near_opt.premium) * 100  # Convert to dollars per spread

            if net_debit <= 0:
                return None  # Not a valid calendar spread (should be net debit)

            # Calculate max profit
            # Max profit occurs when near-term expires worthless and far-term retains max value
            # Simplified: far_premium at near expiration - net_debit
            max_profit = self._calculate_max_profit(near_opt, far_opt, stock_price)

            # Calculate probability of profit using Monte Carlo or simplified model
            prob_profit = self._calculate_probability_profit(near_opt, far_opt, stock_price)

            # Calculate breakeven range
            breakeven_lower, breakeven_upper = self._calculate_breakeven_range(
                near_opt.strike, net_debit / 100, far_opt.premium
            )

            # Calculate liquidity score
            liquidity_score = min(100, (min(near_opt.volume, far_opt.volume) / 100) * 100)

            # Calculate opportunity score (0-100)
            opportunity_score = self._calculate_opportunity_score(
                profit_potential=max_profit / net_debit if net_debit > 0 else 0,
                probability=prob_profit,
                theta_advantage=abs(near_opt.theta) - abs(far_opt.theta),
                liquidity=liquidity_score,
                iv_differential=abs(near_opt.implied_volatility - far_opt.implied_volatility)
            )

            return CalendarSpreadOpportunity(
                symbol=symbol,
                stock_price=stock_price,
                near_strike=near_opt.strike,
                near_expiration=near_opt.expiration,
                near_dte=near_opt.dte,
                near_premium=near_opt.premium * 100,
                near_iv=near_opt.implied_volatility * 100,
                near_theta=near_opt.theta,
                near_volume=near_opt.volume,
                far_strike=far_opt.strike,
                far_expiration=far_opt.expiration,
                far_dte=far_opt.dte,
                far_premium=far_opt.premium * 100,
                far_iv=far_opt.implied_volatility * 100,
                far_theta=far_opt.theta,
                far_volume=far_opt.volume,
                net_debit=net_debit,
                max_profit=max_profit,
                max_loss=net_debit,
                profit_potential=max_profit / net_debit if net_debit > 0 else 0,
                probability_profit=prob_profit,
                breakeven_lower=breakeven_lower,
                breakeven_upper=breakeven_upper,
                net_theta=abs(near_opt.theta) - abs(far_opt.theta),
                net_vega=far_opt.vega - near_opt.vega,
                liquidity_score=liquidity_score,
                iv_differential=abs(near_opt.implied_volatility - far_opt.implied_volatility) * 100,
                opportunity_score=opportunity_score
            )

        except Exception as e:
            print(f"Error evaluating spread: {e}")
            return None

    def _calculate_max_profit(self, near_opt: OptionContract, far_opt: OptionContract, stock_price: float) -> float:
        """Calculate maximum profit potential"""
        # More sophisticated calculation using time decay
        # Estimate far option value at near expiration
        days_between = (far_opt.expiration - near_opt.expiration).days

        # Time decay factor - option loses value as time passes
        # But still has time value remaining
        time_decay_factor = np.sqrt(days_between / far_opt.dte)

        # Estimate far option value at near expiration
        # Assumes optimal scenario where stock is at strike
        far_value_at_near_exp = far_opt.premium * time_decay_factor * 100

        # Near option expires worthless (best case)
        near_value_at_exp = 0

        # Net debit paid
        net_debit = (far_opt.premium - near_opt.premium) * 100

        # Max profit = value of far option - net debit paid
        max_profit = far_value_at_near_exp - net_debit

        return max(0, max_profit)

    def _calculate_probability_profit(self, near_opt: OptionContract, far_opt: OptionContract, stock_price: float) -> float:
        """Calculate probability of profit using more sophisticated model"""
        strike = near_opt.strike

        # Use average IV for calculations
        avg_iv = (near_opt.implied_volatility + far_opt.implied_volatility) / 2

        # Calculate expected move based on IV
        days_to_near_exp = near_opt.dte
        expected_move = stock_price * avg_iv * np.sqrt(days_to_near_exp / 365)

        # Calendar spreads profit when stock stays near strike
        # Define profit zone as ±0.5 standard deviations
        profit_range = expected_move * 0.5

        lower_profit = strike - profit_range
        upper_profit = strike + profit_range

        # Calculate probability using normal distribution
        if expected_move > 0:
            z_lower = (lower_profit - stock_price) / expected_move
            z_upper = (upper_profit - stock_price) / expected_move

            # Probability that stock is within profit range
            prob_in_range = norm.cdf(z_upper) - norm.cdf(z_lower)

            # Adjust for IV differential (higher differential = higher probability)
            iv_adjustment = 1 + (abs(near_opt.implied_volatility - far_opt.implied_volatility) * 0.5)
            prob_in_range *= iv_adjustment
        else:
            prob_in_range = 0.5  # Default if no volatility data

        # Convert to percentage
        return max(10, min(90, prob_in_range * 100))

    def _calculate_breakeven_range(self, strike: float, net_debit_per_share: float, far_premium: float) -> tuple:
        """Calculate breakeven price range"""
        # More accurate breakeven calculation
        # Breakevens occur where the far option value equals the net debit paid

        # Estimate breakeven range based on net debit
        # Wider range for higher net debit
        breakeven_width = net_debit_per_share * 1.5

        breakeven_lower = strike - breakeven_width
        breakeven_upper = strike + breakeven_width

        return (breakeven_lower, breakeven_upper)

    def _calculate_opportunity_score(self, profit_potential: float, probability: float,
                                    theta_advantage: float, liquidity: float, iv_differential: float) -> float:
        """
        Calculate composite opportunity score (0-100)

        Weights:
        - Profit Potential: 35%
        - Probability: 30%
        - Theta Advantage: 20%
        - Liquidity: 10%
        - IV Differential: 5%
        """
        # Normalize profit potential (target 1.5x = 100)
        profit_score = min(100, (profit_potential / 1.5) * 100) * 0.35

        # Probability score
        prob_score = probability * 0.30

        # Theta advantage score (target 0.05 = 100)
        theta_score = min(100, abs(theta_advantage) / 0.05 * 100) * 0.20

        # Liquidity score
        liq_score = liquidity * 0.10

        # IV differential score (higher is better, target 10% = 100)
        iv_score = min(100, iv_differential / 10 * 100) * 0.05

        total_score = profit_score + prob_score + theta_score + liq_score + iv_score

        return min(100, max(0, total_score))