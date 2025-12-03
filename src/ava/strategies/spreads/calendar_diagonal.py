"""
Calendar and Diagonal Spread Strategies
========================================

Time-based spread strategies that profit from differential theta decay
between near-term and far-term options.

Strategies:
1. Calendar Spread (Horizontal Spread):
   - Same strike, different expirations
   - Sell near-term, buy far-term
   - Profits from accelerated near-term theta decay

2. Diagonal Spread (Diagonal Calendar):
   - Different strikes AND expirations
   - Combines time spread with directional bias
   - More flexible than pure calendar

3. Double Calendar:
   - Two calendar spreads (put + call)
   - Profits from range-bound movement
   - Similar to Iron Condor but with time element

Author: AVA Trading Platform
Created: 2025-11-28
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from scipy import stats

from src.ava.strategies.base import (
    OptionsStrategy,
    StrategyType,
    OptionType,
    OptionSide,
    OptionLeg,
    StrategySetup,
    EntrySignal,
    ExitSignal,
    SignalStrength,
    ExitReason,
    MarketContext,
    OptionsChain,
    register_strategy
)

logger = logging.getLogger(__name__)


# =============================================================================
# CALENDAR SPREAD
# =============================================================================

@register_strategy
class CalendarSpreadStrategy(OptionsStrategy):
    """
    Calendar Spread (Time Spread / Horizontal Spread)

    Structure:
    - Sell near-term option (front month)
    - Buy far-term option (back month)
    - Same strike price

    Characteristics:
    - Debit spread (costs money to enter)
    - Limited risk = net debit paid
    - Max profit when underlying at strike at front expiration
    - Benefits from near-term theta decay faster than back-month
    - Vega positive (benefits from IV increase)

    Best Conditions:
    - Low/moderate IV (want IV to rise)
    - Expecting price to stay near strike
    - DTE: Front 20-30, Back 45-60
    """

    name: str = "Calendar Spread"
    description: str = "Time spread profiting from differential theta decay"
    strategy_type: StrategyType = StrategyType.SPREAD

    # Calendar parameters
    option_type_preference: OptionType = OptionType.CALL  # or PUT
    front_month_dte_min: int = 14
    front_month_dte_max: int = 35
    back_month_dte_min: int = 40
    back_month_dte_max: int = 75
    min_dte_spread: int = 21  # Minimum days between expirations

    # Entry criteria
    min_iv_rank: float = 0.0
    max_iv_rank: float = 60.0  # Want room for IV to rise
    target_delta: float = 0.50  # ATM for max theta difference

    # Risk parameters
    profit_target_pct: float = 0.30   # 30% of max profit
    stop_loss_pct: float = 0.50       # 50% of debit
    front_dte_close: int = 5          # Close before front expiration

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Calendar Spread opportunities"""
        opportunities = []

        # Prefer lower IV (room to expand)
        if context.iv_rank > self.max_iv_rank:
            logger.debug(f"IV rank {context.iv_rank:.1f} too high for calendar")
            return opportunities

        # Need multiple expirations
        if len(chain.expirations) < 2:
            return opportunities

        # Find valid expiration pairs
        for front_exp in chain.expirations:
            front_dte = (front_exp - date.today()).days
            if not (self.front_month_dte_min <= front_dte <= self.front_month_dte_max):
                continue

            for back_exp in chain.expirations:
                back_dte = (back_exp - date.today()).days
                if not (self.back_month_dte_min <= back_dte <= self.back_month_dte_max):
                    continue

                dte_spread = back_dte - front_dte
                if dte_spread < self.min_dte_spread:
                    continue

                # Find ATM strike
                atm_strike = chain.get_atm_strike(front_exp)

                # Get option data
                front_calls, front_puts = chain.get_chain_for_expiration(front_exp)
                back_calls, back_puts = chain.get_chain_for_expiration(back_exp)

                if self.option_type_preference == OptionType.CALL:
                    front_chain = front_calls
                    back_chain = back_calls
                else:
                    front_chain = front_puts
                    back_chain = back_puts

                if front_chain.empty or back_chain.empty:
                    continue

                front_data = front_chain[front_chain['strike'] == atm_strike]
                back_data = back_chain[back_chain['strike'] == atm_strike]

                if front_data.empty or back_data.empty:
                    continue

                front = front_data.iloc[0]
                back = back_data.iloc[0]

                # Calendar = sell front, buy back
                net_debit = back.get('ask', 0) - front.get('bid', 0)
                if net_debit <= 0:
                    continue  # Should be debit

                # Build legs
                legs = [
                    OptionLeg(
                        option_type=self.option_type_preference,
                        side=OptionSide.SELL,
                        strike=atm_strike,
                        expiration=front_exp,
                        quantity=1,
                        bid=front.get('bid', 0),
                        ask=front.get('ask', 0),
                        delta=front.get('delta', 0.5),
                        theta=front.get('theta', 0),
                        vega=front.get('vega', 0)
                    ),
                    OptionLeg(
                        option_type=self.option_type_preference,
                        side=OptionSide.BUY,
                        strike=atm_strike,
                        expiration=back_exp,
                        quantity=1,
                        bid=back.get('bid', 0),
                        ask=back.get('ask', 0),
                        delta=back.get('delta', 0.5),
                        theta=back.get('theta', 0),
                        vega=back.get('vega', 0)
                    )
                ]

                # Max loss = net debit
                max_loss = net_debit * 100

                # Max profit is complex - estimate based on theta difference
                theta_diff = abs(front.get('theta', 0)) - abs(back.get('theta', 0))
                estimated_max_profit = theta_diff * front_dte * 100 if theta_diff > 0 else net_debit * 0.5 * 100

                net_greeks = self.aggregate_greeks(legs)

                # Probability estimation - price staying near strike
                pop = self._calculate_calendar_pop(
                    context.current_price, atm_strike,
                    context.implied_volatility, front_dte
                )

                setup = StrategySetup(
                    symbol=chain.symbol,
                    strategy_name=self.name,
                    legs=legs,
                    max_profit=estimated_max_profit,
                    max_loss=max_loss,
                    breakeven_prices=[atm_strike - net_debit, atm_strike + net_debit],
                    probability_of_profit=pop,
                    net_delta=net_greeks['delta'],
                    net_theta=net_greeks['theta'],
                    net_vega=net_greeks['vega'],
                    underlying_price=context.current_price,
                    iv_rank=context.iv_rank,
                    notes=f"Front: {front_dte}DTE, Back: {back_dte}DTE, Strike: ${atm_strike}"
                )

                self.score_setup(setup, context)
                opportunities.append(setup)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def _calculate_calendar_pop(
        self,
        underlying: float,
        strike: float,
        iv: float,
        dte: int
    ) -> float:
        """Estimate probability of profit for calendar spread"""
        if iv <= 0 or dte <= 0:
            return 0.5

        # Calendar profits when price is near strike
        # Estimate profit zone as +/- 1 standard deviation
        t = dte / 365
        std_dev = underlying * iv * np.sqrt(t)

        lower = strike - std_dev * 0.5
        upper = strike + std_dev * 0.5

        d2_lower = (np.log(underlying / lower) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))
        d2_upper = (np.log(underlying / upper) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))

        pop = stats.norm.cdf(d2_lower) - stats.norm.cdf(d2_upper)
        return max(0, min(1, abs(pop)))

    def entry_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> EntrySignal:
        """Entry conditions for Calendar Spread"""
        reasons = []
        warnings = []

        if context.iv_rank > self.max_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too high"]
            )

        if context.iv_rank < 30:
            reasons.append(f"Low IV with room to expand: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        elif context.iv_rank < 50:
            reasons.append(f"Moderate IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Positive vega is good
        if setup.net_vega > 0:
            reasons.append(f"Positive vega: {setup.net_vega:.2f}")

        # Should have positive theta from front month decay
        if setup.net_theta > 0:
            reasons.append(f"Positive theta: ${setup.net_theta:.2f}/day")
        else:
            warnings.append(f"Negative net theta: ${setup.net_theta:.2f}")

        # Check earnings - calendars can profit from IV expansion
        if context.days_to_earnings and 7 <= context.days_to_earnings <= 30:
            reasons.append(f"Earnings catalyst in {context.days_to_earnings} days - IV may rise")

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings
        )

    def exit_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext,
        entry_price: float,
        current_price: float
    ) -> ExitSignal:
        """Exit conditions for Calendar Spread"""

        # For debit spread: current value - entry cost
        pnl = current_price - entry_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        # Profit target
        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close calendar - profit target reached",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close calendar - stop loss",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Close before front expiration
        front_leg = min(setup.legs, key=lambda x: x.expiration)
        if front_leg.days_to_expiration <= self.front_dte_close:
            if pnl_pct > 0:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DTE_THRESHOLD,
                    urgency=SignalStrength.MODERATE,
                    suggested_action="Close profitable calendar or roll front",
                    current_pnl=pnl * 100,
                    current_pnl_pct=pnl_pct
                )
            else:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DTE_THRESHOLD,
                    urgency=SignalStrength.STRONG,
                    suggested_action="Close or roll - front expiration approaching",
                    current_pnl=pnl * 100,
                    current_pnl_pct=pnl_pct
                )

        return ExitSignal(
            should_exit=False,
            reason=ExitReason.MANUAL,
            urgency=SignalStrength.NONE,
            current_pnl=pnl * 100,
            current_pnl_pct=pnl_pct
        )

    def calculate_position_size(
        self,
        setup: StrategySetup,
        account_value: float,
        current_positions: List[StrategySetup]
    ) -> int:
        """Position size for Calendar Spread"""
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0

        max_risk = account_value * 0.02
        return max(int(max_risk / max_loss), 0)


# =============================================================================
# DIAGONAL SPREAD
# =============================================================================

@register_strategy
class DiagonalSpreadStrategy(OptionsStrategy):
    """
    Diagonal Spread (Diagonal Calendar)

    Structure:
    - Sell near-term OTM option
    - Buy far-term ITM/ATM option
    - Different strikes AND expirations

    Characteristics:
    - Combines calendar spread + directional bias
    - The long option (LEAPS) finances short-term premium sales
    - Can be bullish (calls) or bearish (puts)

    Example (Bullish Diagonal / Poor Man's Covered Call):
    - Buy 90-day ITM call (delta ~0.70)
    - Sell 30-day OTM call (delta ~0.30)
    """

    name: str = "Diagonal Spread"
    description: str = "Time spread with directional bias"
    strategy_type: StrategyType = StrategyType.SPREAD

    # Diagonal parameters
    option_type_preference: OptionType = OptionType.CALL
    long_delta: float = 0.70          # ITM long option
    short_delta: float = 0.30         # OTM short option

    front_month_dte_min: int = 21
    front_month_dte_max: int = 45
    back_month_dte_min: int = 60
    back_month_dte_max: int = 120

    min_iv_rank: float = 20.0
    max_iv_rank: float = 70.0

    profit_target_pct: float = 0.25
    stop_loss_pct: float = 0.50

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Diagonal Spread opportunities"""
        opportunities = []

        if not (self.min_iv_rank <= context.iv_rank <= self.max_iv_rank):
            return opportunities

        if len(chain.expirations) < 2:
            return opportunities

        for front_exp in chain.expirations:
            front_dte = (front_exp - date.today()).days
            if not (self.front_month_dte_min <= front_dte <= self.front_month_dte_max):
                continue

            for back_exp in chain.expirations:
                back_dte = (back_exp - date.today()).days
                if not (self.back_month_dte_min <= back_dte <= self.back_month_dte_max):
                    continue

                if back_dte <= front_dte + 30:
                    continue

                # Find strikes by delta
                short_strike = chain.get_strike_by_delta(
                    front_exp, self.short_delta, self.option_type_preference
                )
                long_strike = chain.get_strike_by_delta(
                    back_exp, self.long_delta, self.option_type_preference
                )

                if not short_strike or not long_strike:
                    continue

                # For diagonal, long strike should be different from short
                # Call diagonal: long ITM (lower strike), sell OTM (higher strike)
                if self.option_type_preference == OptionType.CALL:
                    if long_strike >= short_strike:
                        continue  # Want ITM long
                else:
                    if long_strike <= short_strike:
                        continue  # Want ITM long put (higher strike)

                # Get option data
                front_calls, front_puts = chain.get_chain_for_expiration(front_exp)
                back_calls, back_puts = chain.get_chain_for_expiration(back_exp)

                if self.option_type_preference == OptionType.CALL:
                    front_chain = front_calls
                    back_chain = back_calls
                else:
                    front_chain = front_puts
                    back_chain = back_puts

                if front_chain.empty or back_chain.empty:
                    continue

                short_data = front_chain[front_chain['strike'] == short_strike]
                long_data = back_chain[back_chain['strike'] == long_strike]

                if short_data.empty or long_data.empty:
                    continue

                short_opt = short_data.iloc[0]
                long_opt = long_data.iloc[0]

                net_debit = long_opt.get('ask', 0) - short_opt.get('bid', 0)
                if net_debit <= 0:
                    continue

                legs = [
                    OptionLeg(
                        option_type=self.option_type_preference,
                        side=OptionSide.SELL,
                        strike=short_strike,
                        expiration=front_exp,
                        quantity=1,
                        bid=short_opt.get('bid', 0),
                        ask=short_opt.get('ask', 0),
                        delta=short_opt.get('delta', self.short_delta),
                        theta=short_opt.get('theta', 0),
                        vega=short_opt.get('vega', 0)
                    ),
                    OptionLeg(
                        option_type=self.option_type_preference,
                        side=OptionSide.BUY,
                        strike=long_strike,
                        expiration=back_exp,
                        quantity=1,
                        bid=long_opt.get('bid', 0),
                        ask=long_opt.get('ask', 0),
                        delta=long_opt.get('delta', self.long_delta),
                        theta=long_opt.get('theta', 0),
                        vega=long_opt.get('vega', 0)
                    )
                ]

                # Calculate risk/reward
                max_loss = net_debit * 100

                # Max profit is when short expires worthless and long appreciates
                strike_diff = abs(short_strike - long_strike)
                estimated_max_profit = (short_opt.get('bid', 0) + strike_diff) * 100

                net_greeks = self.aggregate_greeks(legs)

                pop = 0.55  # Estimate

                setup = StrategySetup(
                    symbol=chain.symbol,
                    strategy_name=self.name,
                    legs=legs,
                    max_profit=estimated_max_profit,
                    max_loss=max_loss,
                    breakeven_prices=[long_strike + net_debit if self.option_type_preference == OptionType.CALL else long_strike - net_debit],
                    probability_of_profit=pop,
                    net_delta=net_greeks['delta'],
                    net_theta=net_greeks['theta'],
                    net_vega=net_greeks['vega'],
                    underlying_price=context.current_price,
                    iv_rank=context.iv_rank,
                    notes=f"Long ${long_strike} {back_dte}DTE, Short ${short_strike} {front_dte}DTE"
                )

                self.score_setup(setup, context)
                opportunities.append(setup)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def entry_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> EntrySignal:
        """Entry conditions for Diagonal Spread"""
        reasons = []
        warnings = []

        if not (self.min_iv_rank <= context.iv_rank <= self.max_iv_rank):
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} outside range"]
            )

        reasons.append(f"IV rank {context.iv_rank:.1f} in target range")

        # Net delta shows directional bias
        if abs(setup.net_delta) > 0.20:
            direction = "bullish" if setup.net_delta > 0 else "bearish"
            reasons.append(f"Directional bias: {direction} (delta {setup.net_delta:.2f})")
        else:
            warnings.append("Low net delta - limited directional exposure")

        # Positive theta good for premium sellers
        if setup.net_theta > 0:
            reasons.append(f"Positive theta: ${setup.net_theta:.2f}/day")

        strength = SignalStrength.MODERATE

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings
        )

    def exit_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext,
        entry_price: float,
        current_price: float
    ) -> ExitSignal:
        """Exit conditions for Diagonal Spread"""

        pnl = current_price - entry_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close or roll front leg",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close diagonal",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Check front leg expiration
        front_leg = min(setup.legs, key=lambda x: x.expiration)
        if front_leg.days_to_expiration <= 5:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DTE_THRESHOLD,
                urgency=SignalStrength.MODERATE,
                suggested_action="Roll front leg to next expiration",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        return ExitSignal(
            should_exit=False,
            reason=ExitReason.MANUAL,
            urgency=SignalStrength.NONE,
            current_pnl=pnl * 100,
            current_pnl_pct=pnl_pct
        )

    def calculate_position_size(
        self,
        setup: StrategySetup,
        account_value: float,
        current_positions: List[StrategySetup]
    ) -> int:
        """Position size for Diagonal Spread"""
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0

        max_risk = account_value * 0.03  # 3% max risk
        return max(int(max_risk / max_loss), 0)


# =============================================================================
# DOUBLE CALENDAR
# =============================================================================

@register_strategy
class DoubleCalendarStrategy(OptionsStrategy):
    """
    Double Calendar Spread

    Structure:
    - Calendar spread on puts (below current price)
    - Calendar spread on calls (above current price)

    Characteristics:
    - Profits from range-bound movement
    - Similar profile to Iron Condor but with time element
    - Benefits from IV increase (vega positive)
    - Two profit peaks at each strike
    """

    name: str = "Double Calendar"
    description: str = "Two calendar spreads for range-bound profit"
    strategy_type: StrategyType = StrategyType.NEUTRAL

    put_strike_delta: float = 0.30
    call_strike_delta: float = 0.30

    front_month_dte_min: int = 14
    front_month_dte_max: int = 30
    back_month_dte_min: int = 40
    back_month_dte_max: int = 60

    min_iv_rank: float = 0.0
    max_iv_rank: float = 50.0

    profit_target_pct: float = 0.25
    stop_loss_pct: float = 0.50

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Double Calendar opportunities"""
        opportunities = []

        if context.iv_rank > self.max_iv_rank:
            return opportunities

        if len(chain.expirations) < 2:
            return opportunities

        for front_exp in chain.expirations:
            front_dte = (front_exp - date.today()).days
            if not (self.front_month_dte_min <= front_dte <= self.front_month_dte_max):
                continue

            for back_exp in chain.expirations:
                back_dte = (back_exp - date.today()).days
                if not (self.back_month_dte_min <= back_dte <= self.back_month_dte_max):
                    continue

                if back_dte <= front_dte + 14:
                    continue

                # Find OTM strikes
                put_strike = chain.get_strike_by_delta(front_exp, self.put_strike_delta, OptionType.PUT)
                call_strike = chain.get_strike_by_delta(front_exp, self.call_strike_delta, OptionType.CALL)

                if not put_strike or not call_strike:
                    continue

                # Get option data for all 4 legs
                front_calls, front_puts = chain.get_chain_for_expiration(front_exp)
                back_calls, back_puts = chain.get_chain_for_expiration(back_exp)

                front_put = front_puts[front_puts['strike'] == put_strike]
                back_put = back_puts[back_puts['strike'] == put_strike]
                front_call = front_calls[front_calls['strike'] == call_strike]
                back_call = back_calls[back_calls['strike'] == call_strike]

                if any(df.empty for df in [front_put, back_put, front_call, back_call]):
                    continue

                fp = front_put.iloc[0]
                bp = back_put.iloc[0]
                fc = front_call.iloc[0]
                bc = back_call.iloc[0]

                # Calculate net debit
                put_calendar_debit = bp.get('ask', 0) - fp.get('bid', 0)
                call_calendar_debit = bc.get('ask', 0) - fc.get('bid', 0)
                total_debit = put_calendar_debit + call_calendar_debit

                if total_debit <= 0:
                    continue

                legs = [
                    # Put calendar
                    OptionLeg(
                        option_type=OptionType.PUT,
                        side=OptionSide.SELL,
                        strike=put_strike,
                        expiration=front_exp,
                        quantity=1,
                        bid=fp.get('bid', 0),
                        ask=fp.get('ask', 0),
                        delta=fp.get('delta', -self.put_strike_delta),
                        theta=fp.get('theta', 0)
                    ),
                    OptionLeg(
                        option_type=OptionType.PUT,
                        side=OptionSide.BUY,
                        strike=put_strike,
                        expiration=back_exp,
                        quantity=1,
                        bid=bp.get('bid', 0),
                        ask=bp.get('ask', 0),
                        delta=bp.get('delta', -self.put_strike_delta),
                        theta=bp.get('theta', 0)
                    ),
                    # Call calendar
                    OptionLeg(
                        option_type=OptionType.CALL,
                        side=OptionSide.SELL,
                        strike=call_strike,
                        expiration=front_exp,
                        quantity=1,
                        bid=fc.get('bid', 0),
                        ask=fc.get('ask', 0),
                        delta=fc.get('delta', self.call_strike_delta),
                        theta=fc.get('theta', 0)
                    ),
                    OptionLeg(
                        option_type=OptionType.CALL,
                        side=OptionSide.BUY,
                        strike=call_strike,
                        expiration=back_exp,
                        quantity=1,
                        bid=bc.get('bid', 0),
                        ask=bc.get('ask', 0),
                        delta=bc.get('delta', self.call_strike_delta),
                        theta=bc.get('theta', 0)
                    )
                ]

                max_loss = total_debit * 100
                estimated_max_profit = max_loss * 0.5  # Estimate

                net_greeks = self.aggregate_greeks(legs)

                setup = StrategySetup(
                    symbol=chain.symbol,
                    strategy_name=self.name,
                    legs=legs,
                    max_profit=estimated_max_profit,
                    max_loss=max_loss,
                    breakeven_prices=[put_strike, call_strike],
                    probability_of_profit=0.50,
                    net_delta=net_greeks['delta'],
                    net_theta=net_greeks['theta'],
                    net_vega=net_greeks['vega'],
                    underlying_price=context.current_price,
                    iv_rank=context.iv_rank,
                    notes=f"Put ${put_strike}, Call ${call_strike}"
                )

                self.score_setup(setup, context)
                opportunities.append(setup)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def entry_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> EntrySignal:
        """Entry conditions for Double Calendar"""
        reasons = []
        warnings = []

        if context.iv_rank > self.max_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too high"]
            )

        if context.iv_rank < 30:
            reasons.append(f"Low IV with expansion potential: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        else:
            reasons.append(f"Moderate IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE

        if abs(setup.net_delta) < 0.10:
            reasons.append(f"Delta neutral: {setup.net_delta:.3f}")
        else:
            warnings.append(f"Net delta bias: {setup.net_delta:.3f}")

        if setup.net_vega > 0:
            reasons.append(f"Positive vega: {setup.net_vega:.2f}")

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings
        )

    def exit_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext,
        entry_price: float,
        current_price: float
    ) -> ExitSignal:
        """Exit conditions for Double Calendar"""

        pnl = current_price - entry_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close double calendar",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close position",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        front_legs = [leg for leg in setup.legs if leg.is_short]
        min_front_dte = min(leg.days_to_expiration for leg in front_legs)

        if min_front_dte <= 5:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DTE_THRESHOLD,
                urgency=SignalStrength.MODERATE,
                suggested_action="Close or roll front legs",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        return ExitSignal(
            should_exit=False,
            reason=ExitReason.MANUAL,
            urgency=SignalStrength.NONE,
            current_pnl=pnl * 100,
            current_pnl_pct=pnl_pct
        )

    def calculate_position_size(
        self,
        setup: StrategySetup,
        account_value: float,
        current_positions: List[StrategySetup]
    ) -> int:
        """Position size for Double Calendar"""
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0

        max_risk = account_value * 0.02
        return max(int(max_risk / max_loss), 0)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Calendar & Diagonal Strategies ===\n")

    print("1. Calendar Spread Strategy")
    calendar = CalendarSpreadStrategy()
    print(f"   Name: {calendar.name}")
    print(f"   Front DTE: {calendar.front_month_dte_min}-{calendar.front_month_dte_max}")
    print(f"   Back DTE: {calendar.back_month_dte_min}-{calendar.back_month_dte_max}")

    print("\n2. Diagonal Spread Strategy")
    diagonal = DiagonalSpreadStrategy()
    print(f"   Name: {diagonal.name}")
    print(f"   Long delta: {diagonal.long_delta}")
    print(f"   Short delta: {diagonal.short_delta}")

    print("\n3. Double Calendar Strategy")
    double_cal = DoubleCalendarStrategy()
    print(f"   Name: {double_cal.name}")
    print(f"   Put strike delta: {double_cal.put_strike_delta}")
    print(f"   Call strike delta: {double_cal.call_strike_delta}")

    print("\n✅ Calendar & Diagonal strategies ready!")
