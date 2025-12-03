"""
Straddle and Strangle Strategies
================================

Volatility strategies that profit from large price moves (long) or
no movement (short).

Strategies:
1. Long Straddle: Buy ATM call + ATM put (debit, unlimited profit potential)
2. Short Straddle: Sell ATM call + ATM put (credit, limited profit, high risk)
3. Long Strangle: Buy OTM call + OTM put (cheaper debit, needs bigger move)
4. Short Strangle: Sell OTM call + OTM put (wider profit zone, lower credit)

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
# BASE STRADDLE/STRANGLE CLASS
# =============================================================================

class VolatilityStrategy(OptionsStrategy):
    """Base class for straddle/strangle strategies"""

    # Common parameters
    target_delta: float = 0.50      # ATM = 0.50, OTM = lower
    min_dte: int = 14
    max_dte: int = 60
    optimal_dte: int = 30

    # Whether we're buying or selling volatility
    is_long: bool = True

    def _get_atm_strike(self, chain: OptionsChain, expiration: date) -> float:
        """Get ATM strike for expiration"""
        return chain.get_atm_strike(expiration)

    def _get_otm_strikes(
        self,
        chain: OptionsChain,
        expiration: date,
        target_delta: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get OTM put and call strikes by delta"""
        put_strike = chain.get_strike_by_delta(expiration, target_delta, OptionType.PUT)
        call_strike = chain.get_strike_by_delta(expiration, target_delta, OptionType.CALL)
        return put_strike, call_strike

    def _calculate_straddle_pop(
        self,
        underlying: float,
        strike: float,
        premium: float,
        iv: float,
        dte: int,
        is_long: bool
    ) -> float:
        """Calculate POP for straddle"""
        if iv <= 0 or dte <= 0:
            return 0.5

        t = dte / 365

        # Breakeven points
        lower_be = strike - premium
        upper_be = strike + premium

        # For long straddle: P(S < lower_be OR S > upper_be)
        # For short straddle: P(lower_be < S < upper_be)
        d2_lower = (np.log(underlying / lower_be) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))
        d2_upper = (np.log(underlying / upper_be) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))

        prob_above_lower = stats.norm.cdf(d2_lower)
        prob_above_upper = stats.norm.cdf(d2_upper)

        if is_long:
            # Need price to move outside breakevens
            pop = (1 - prob_above_lower) + prob_above_upper
        else:
            # Need price to stay within breakevens
            pop = prob_above_lower - prob_above_upper

        return max(0, min(1, pop))

    def _calculate_strangle_pop(
        self,
        underlying: float,
        put_strike: float,
        call_strike: float,
        premium: float,
        iv: float,
        dte: int,
        is_long: bool
    ) -> float:
        """Calculate POP for strangle"""
        if iv <= 0 or dte <= 0:
            return 0.5

        t = dte / 365

        # Breakeven points
        lower_be = put_strike - premium
        upper_be = call_strike + premium

        d2_lower = (np.log(underlying / lower_be) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))
        d2_upper = (np.log(underlying / upper_be) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))

        prob_above_lower = stats.norm.cdf(d2_lower)
        prob_above_upper = stats.norm.cdf(d2_upper)

        if is_long:
            pop = (1 - prob_above_lower) + prob_above_upper
        else:
            pop = prob_above_lower - prob_above_upper

        return max(0, min(1, pop))


# =============================================================================
# LONG STRADDLE
# =============================================================================

@register_strategy
class LongStraddleStrategy(VolatilityStrategy):
    """
    Long Straddle Strategy

    Structure:
    - Buy ATM Call
    - Buy ATM Put

    Characteristics:
    - Unlimited profit potential in either direction
    - Max loss = total premium paid
    - Profits from large price moves or IV expansion
    - Best before earnings/events with expected volatility

    Entry Criteria:
    - Low IV rank (< 30) for cheap premiums
    - Expected catalyst (earnings, FDA, etc.)
    - DTE 14-45 days
    """

    name: str = "Long Straddle"
    description: str = "Buy ATM call and put to profit from large moves"
    strategy_type: StrategyType = StrategyType.VOLATILITY
    is_long: bool = True

    # Entry criteria - want low IV to buy cheap
    min_iv_rank: float = 0.0
    max_iv_rank: float = 40.0
    min_dte: int = 14
    max_dte: int = 45

    # Exit parameters
    profit_target_pct: float = 0.50   # Take profit at 50% gain
    stop_loss_pct: float = 0.50       # Stop at 50% loss
    dte_close_threshold: int = 7      # Close with 7 DTE left

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Long Straddle opportunities"""
        opportunities = []

        # Want low IV for buying premium
        if context.iv_rank > self.max_iv_rank:
            logger.debug(f"IV rank {context.iv_rank:.1f} too high for long straddle")
            return opportunities

        # Prefer when earnings/catalyst approaching
        catalyst_bonus = 0
        if context.days_to_earnings and 7 <= context.days_to_earnings <= 30:
            catalyst_bonus = 20  # Bonus score for catalyst

        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            calls, puts = chain.get_chain_for_expiration(expiration)
            if calls.empty or puts.empty:
                continue

            # Get ATM strike
            atm_strike = self._get_atm_strike(chain, expiration)

            # Get option data
            call_data = calls[calls['strike'] == atm_strike]
            put_data = puts[puts['strike'] == atm_strike]

            if call_data.empty or put_data.empty:
                continue

            call = call_data.iloc[0]
            put = put_data.iloc[0]

            # Calculate total premium
            total_premium = call.get('ask', 0) + put.get('ask', 0)
            if total_premium <= 0:
                continue

            # Build legs
            legs = [
                OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.BUY,
                    strike=atm_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=call.get('bid', 0),
                    ask=call.get('ask', 0),
                    delta=call.get('delta', 0.5),
                    theta=call.get('theta', 0),
                    vega=call.get('vega', 0)
                ),
                OptionLeg(
                    option_type=OptionType.PUT,
                    side=OptionSide.BUY,
                    strike=atm_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=put.get('bid', 0),
                    ask=put.get('ask', 0),
                    delta=put.get('delta', -0.5),
                    theta=put.get('theta', 0),
                    vega=put.get('vega', 0)
                )
            ]

            # Calculate metrics
            max_loss = total_premium * 100
            dte = (expiration - date.today()).days

            pop = self._calculate_straddle_pop(
                context.current_price, atm_strike, total_premium,
                context.implied_volatility, dte, is_long=True
            )

            net_greeks = self.aggregate_greeks(legs)

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=float('inf'),
                max_loss=max_loss,
                breakeven_prices=[atm_strike - total_premium, atm_strike + total_premium],
                probability_of_profit=pop,
                net_delta=net_greeks['delta'],
                net_gamma=net_greeks['gamma'],
                net_theta=net_greeks['theta'],
                net_vega=net_greeks['vega'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"ATM ${atm_strike}, Premium ${total_premium:.2f}"
            )

            self.score_setup(setup, context)
            setup.score += catalyst_bonus

            opportunities.append(setup)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def entry_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> EntrySignal:
        """Check entry conditions for Long Straddle"""
        reasons = []
        warnings = []

        # IV rank check - want low IV
        if context.iv_rank > self.max_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too high (max {self.max_iv_rank})"]
            )

        if context.iv_rank < 20:
            reasons.append(f"Excellent low IV: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        elif context.iv_rank < 30:
            reasons.append(f"Good IV level: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE
        else:
            reasons.append(f"Acceptable IV: {context.iv_rank:.1f}")
            strength = SignalStrength.WEAK

        # Catalyst check
        if context.days_to_earnings and context.days_to_earnings <= 30:
            reasons.append(f"Earnings catalyst in {context.days_to_earnings} days")
            strength = SignalStrength.STRONG

        # High vega is good for long vol
        if setup.net_vega > 0.5:
            reasons.append(f"High vega exposure: {setup.net_vega:.2f}")

        # Theta decay warning
        if setup.net_theta < -0.5:
            warnings.append(f"High theta decay: {setup.net_theta:.2f}/day")

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
        """Check exit conditions for Long Straddle"""

        # P&L = current value - entry cost (debit)
        pnl = current_price - entry_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        # Profit target
        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close position - profit target reached",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close position - stop loss hit",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # DTE threshold - theta decay accelerates
        if setup.days_to_expiration <= self.dte_close_threshold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DTE_THRESHOLD,
                urgency=SignalStrength.MODERATE,
                suggested_action="Close or roll - theta decay accelerating",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # After earnings - IV usually crushes
        if context.days_to_earnings and context.days_to_earnings < 0:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.EARNINGS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close post-earnings - IV crush",
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
        """Calculate position size for Long Straddle"""
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0

        # Max 2% risk per trade
        max_risk = account_value * self.max_loss_pct
        contracts = int(max_risk / max_loss)

        return max(contracts, 0)


# =============================================================================
# SHORT STRADDLE
# =============================================================================

@register_strategy
class ShortStraddleStrategy(VolatilityStrategy):
    """
    Short Straddle Strategy

    Structure:
    - Sell ATM Call
    - Sell ATM Put

    Characteristics:
    - Max profit = total premium received
    - Unlimited risk in either direction
    - Profits from no movement and theta decay
    - Best in high IV environments when expecting range-bound

    WARNING: Undefined risk - use with caution!
    """

    name: str = "Short Straddle"
    description: str = "Sell ATM call and put to collect premium"
    strategy_type: StrategyType = StrategyType.INCOME
    is_long: bool = False

    # Entry criteria - want high IV to sell rich
    min_iv_rank: float = 50.0
    max_iv_rank: float = 100.0
    min_dte: int = 30
    max_dte: int = 60

    # Tighter risk management due to undefined risk
    profit_target_pct: float = 0.25   # Take profit at 25%
    stop_loss_pct: float = 1.0        # Stop at 100% of credit
    dte_close_threshold: int = 14

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Short Straddle opportunities"""
        opportunities = []

        # Want high IV for selling premium
        if context.iv_rank < self.min_iv_rank:
            logger.debug(f"IV rank {context.iv_rank:.1f} too low for short straddle")
            return opportunities

        # Avoid earnings - undefined risk is too dangerous
        if context.days_to_earnings and context.days_to_earnings < 14:
            logger.debug("Earnings too close for short straddle")
            return opportunities

        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            calls, puts = chain.get_chain_for_expiration(expiration)
            if calls.empty or puts.empty:
                continue

            atm_strike = self._get_atm_strike(chain, expiration)

            call_data = calls[calls['strike'] == atm_strike]
            put_data = puts[puts['strike'] == atm_strike]

            if call_data.empty or put_data.empty:
                continue

            call = call_data.iloc[0]
            put = put_data.iloc[0]

            # Credit received
            total_credit = call.get('bid', 0) + put.get('bid', 0)
            if total_credit <= 0:
                continue

            legs = [
                OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.SELL,
                    strike=atm_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=call.get('bid', 0),
                    ask=call.get('ask', 0),
                    delta=call.get('delta', 0.5),
                    theta=call.get('theta', 0),
                    vega=call.get('vega', 0)
                ),
                OptionLeg(
                    option_type=OptionType.PUT,
                    side=OptionSide.SELL,
                    strike=atm_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=put.get('bid', 0),
                    ask=put.get('ask', 0),
                    delta=put.get('delta', -0.5),
                    theta=put.get('theta', 0),
                    vega=put.get('vega', 0)
                )
            ]

            dte = (expiration - date.today()).days
            pop = self._calculate_straddle_pop(
                context.current_price, atm_strike, total_credit,
                context.implied_volatility, dte, is_long=False
            )

            net_greeks = self.aggregate_greeks(legs)

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=total_credit * 100,
                max_loss=float('inf'),  # UNDEFINED RISK
                breakeven_prices=[atm_strike - total_credit, atm_strike + total_credit],
                probability_of_profit=pop,
                net_delta=net_greeks['delta'],
                net_theta=net_greeks['theta'],
                net_vega=net_greeks['vega'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"WARNING: Undefined risk! Credit ${total_credit:.2f}"
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
        """Check entry conditions for Short Straddle"""
        reasons = []
        warnings = ["WARNING: This strategy has UNDEFINED RISK"]

        if context.iv_rank < self.min_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too low"]
            )

        if context.iv_rank >= 70:
            reasons.append(f"Excellent high IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE  # Never strong due to risk
        else:
            reasons.append(f"Good IV level: {context.iv_rank:.1f}")
            strength = SignalStrength.WEAK

        # Positive theta is critical
        if setup.net_theta > 0:
            reasons.append(f"Positive theta: ${setup.net_theta:.2f}/day")
        else:
            warnings.append("Negative theta - unusual")

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
        """Exit conditions for Short Straddle - more conservative"""

        pnl = entry_price - current_price  # Credit - debit to close
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        # Earlier profit taking due to risk
        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - take profits on undefined risk position",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Tighter stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="CLOSE IMMEDIATELY - loss exceeds threshold",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Price movement warning
        atm_strike = setup.legs[0].strike
        price_move = abs(context.current_price - atm_strike) / atm_strike
        if price_move > 0.05:  # 5% move from ATM
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - price moving significantly",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if setup.days_to_expiration <= self.dte_close_threshold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DTE_THRESHOLD,
                urgency=SignalStrength.MODERATE,
                suggested_action="Close - gamma risk increasing",
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
        """Very conservative position sizing for undefined risk"""
        # Use notional value for sizing since max loss is undefined
        atm_strike = setup.legs[0].strike
        notional_per_contract = atm_strike * 100

        # Max 5% notional exposure
        max_notional = account_value * 0.05
        contracts = int(max_notional / notional_per_contract)

        # Cap at 2 contracts regardless
        return min(contracts, 2)


# =============================================================================
# LONG STRANGLE
# =============================================================================

@register_strategy
class LongStrangleStrategy(VolatilityStrategy):
    """
    Long Strangle Strategy

    Structure:
    - Buy OTM Call
    - Buy OTM Put

    Characteristics:
    - Cheaper than straddle (OTM options)
    - Needs larger move to profit
    - Unlimited profit potential
    - Lower max loss than straddle
    """

    name: str = "Long Strangle"
    description: str = "Buy OTM call and put for large moves"
    strategy_type: StrategyType = StrategyType.VOLATILITY
    is_long: bool = True

    target_delta: float = 0.25  # OTM strikes
    min_iv_rank: float = 0.0
    max_iv_rank: float = 35.0
    min_dte: int = 21
    max_dte: int = 60

    profit_target_pct: float = 1.0    # 100% gain target
    stop_loss_pct: float = 0.50
    dte_close_threshold: int = 10

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Long Strangle opportunities"""
        opportunities = []

        if context.iv_rank > self.max_iv_rank:
            return opportunities

        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            put_strike, call_strike = self._get_otm_strikes(
                chain, expiration, self.target_delta
            )

            if not put_strike or not call_strike:
                continue

            calls, puts = chain.get_chain_for_expiration(expiration)

            call_data = calls[calls['strike'] == call_strike]
            put_data = puts[puts['strike'] == put_strike]

            if call_data.empty or put_data.empty:
                continue

            call = call_data.iloc[0]
            put = put_data.iloc[0]

            total_premium = call.get('ask', 0) + put.get('ask', 0)
            if total_premium <= 0:
                continue

            legs = [
                OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.BUY,
                    strike=call_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=call.get('bid', 0),
                    ask=call.get('ask', 0),
                    delta=call.get('delta', self.target_delta),
                    theta=call.get('theta', 0),
                    vega=call.get('vega', 0)
                ),
                OptionLeg(
                    option_type=OptionType.PUT,
                    side=OptionSide.BUY,
                    strike=put_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=put.get('bid', 0),
                    ask=put.get('ask', 0),
                    delta=put.get('delta', -self.target_delta),
                    theta=put.get('theta', 0),
                    vega=put.get('vega', 0)
                )
            ]

            dte = (expiration - date.today()).days
            pop = self._calculate_strangle_pop(
                context.current_price, put_strike, call_strike,
                total_premium, context.implied_volatility, dte, is_long=True
            )

            net_greeks = self.aggregate_greeks(legs)

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=float('inf'),
                max_loss=total_premium * 100,
                breakeven_prices=[put_strike - total_premium, call_strike + total_premium],
                probability_of_profit=pop,
                net_delta=net_greeks['delta'],
                net_theta=net_greeks['theta'],
                net_vega=net_greeks['vega'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"Strikes: ${put_strike}P/${call_strike}C, Cost: ${total_premium:.2f}"
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
        """Entry conditions for Long Strangle"""
        reasons = []
        warnings = []

        if context.iv_rank > self.max_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too high"]
            )

        if context.iv_rank < 20:
            reasons.append(f"Very cheap IV: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        elif context.iv_rank < 30:
            reasons.append(f"Low IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        if context.days_to_earnings and context.days_to_earnings <= 30:
            reasons.append(f"Earnings catalyst: {context.days_to_earnings} days")
            strength = SignalStrength.STRONG

        if setup.net_theta < -0.3:
            warnings.append(f"Theta decay: ${setup.net_theta:.2f}/day")

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
        """Exit conditions for Long Strangle"""

        pnl = current_price - entry_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - profit target reached",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - stop loss hit",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if setup.days_to_expiration <= self.dte_close_threshold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DTE_THRESHOLD,
                urgency=SignalStrength.MODERATE,
                suggested_action="Close - theta accelerating",
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
        """Position sizing for Long Strangle"""
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0

        max_risk = account_value * 0.02
        return max(int(max_risk / max_loss), 0)


# =============================================================================
# SHORT STRANGLE
# =============================================================================

@register_strategy
class ShortStrangleStrategy(VolatilityStrategy):
    """
    Short Strangle Strategy

    Structure:
    - Sell OTM Call
    - Sell OTM Put

    Characteristics:
    - Wider profit zone than straddle
    - Lower credit than straddle
    - UNDEFINED RISK on both sides
    - Best for high IV, range-bound expectations
    """

    name: str = "Short Strangle"
    description: str = "Sell OTM call and put for premium"
    strategy_type: StrategyType = StrategyType.INCOME
    is_long: bool = False

    target_delta: float = 0.16  # ~1 standard deviation
    min_iv_rank: float = 50.0
    max_iv_rank: float = 100.0
    min_dte: int = 30
    max_dte: int = 60

    profit_target_pct: float = 0.50   # 50% of max profit
    stop_loss_pct: float = 2.0        # 200% of credit
    dte_close_threshold: int = 14

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Short Strangle opportunities"""
        opportunities = []

        if context.iv_rank < self.min_iv_rank:
            return opportunities

        # Avoid earnings
        if context.days_to_earnings and context.days_to_earnings < 14:
            return opportunities

        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            put_strike, call_strike = self._get_otm_strikes(
                chain, expiration, self.target_delta
            )

            if not put_strike or not call_strike:
                continue

            calls, puts = chain.get_chain_for_expiration(expiration)

            call_data = calls[calls['strike'] == call_strike]
            put_data = puts[puts['strike'] == put_strike]

            if call_data.empty or put_data.empty:
                continue

            call = call_data.iloc[0]
            put = put_data.iloc[0]

            total_credit = call.get('bid', 0) + put.get('bid', 0)
            if total_credit <= 0:
                continue

            legs = [
                OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.SELL,
                    strike=call_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=call.get('bid', 0),
                    ask=call.get('ask', 0),
                    delta=call.get('delta', self.target_delta),
                    theta=call.get('theta', 0),
                    vega=call.get('vega', 0)
                ),
                OptionLeg(
                    option_type=OptionType.PUT,
                    side=OptionSide.SELL,
                    strike=put_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=put.get('bid', 0),
                    ask=put.get('ask', 0),
                    delta=put.get('delta', -self.target_delta),
                    theta=put.get('theta', 0),
                    vega=put.get('vega', 0)
                )
            ]

            dte = (expiration - date.today()).days
            pop = self._calculate_strangle_pop(
                context.current_price, put_strike, call_strike,
                total_credit, context.implied_volatility, dte, is_long=False
            )

            net_greeks = self.aggregate_greeks(legs)

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=total_credit * 100,
                max_loss=float('inf'),
                breakeven_prices=[put_strike - total_credit, call_strike + total_credit],
                probability_of_profit=pop,
                net_delta=net_greeks['delta'],
                net_theta=net_greeks['theta'],
                net_vega=net_greeks['vega'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"WARNING: Undefined risk! ${put_strike}P/${call_strike}C"
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
        """Entry conditions for Short Strangle"""
        reasons = []
        warnings = ["WARNING: UNDEFINED RISK on both sides"]

        if context.iv_rank < self.min_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too low"]
            )

        if context.iv_rank >= 70:
            reasons.append(f"High IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE
        else:
            reasons.append(f"Acceptable IV: {context.iv_rank:.1f}")
            strength = SignalStrength.WEAK

        if setup.probability_of_profit >= 0.80:
            reasons.append(f"High POP: {setup.probability_of_profit:.1%}")
        elif setup.probability_of_profit >= 0.70:
            reasons.append(f"Good POP: {setup.probability_of_profit:.1%}")

        if setup.net_theta > 0:
            reasons.append(f"Positive theta: ${setup.net_theta:.2f}/day")

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
        """Exit conditions for Short Strangle"""

        pnl = entry_price - current_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - profit target on undefined risk",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="CLOSE IMMEDIATELY",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Check if approaching short strikes
        put_strike = min(leg.strike for leg in setup.legs if leg.is_put)
        call_strike = max(leg.strike for leg in setup.legs if leg.is_call)

        if context.current_price <= put_strike * 1.02:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - approaching put strike",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if context.current_price >= call_strike * 0.98:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - approaching call strike",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if setup.days_to_expiration <= self.dte_close_threshold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DTE_THRESHOLD,
                urgency=SignalStrength.MODERATE,
                suggested_action="Close - gamma risk",
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
        """Conservative position sizing for undefined risk"""
        put_strike = min(leg.strike for leg in setup.legs if leg.is_put)
        notional = put_strike * 100

        max_notional = account_value * 0.05
        contracts = int(max_notional / notional)

        return min(contracts, 3)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Straddle/Strangle Strategies ===\n")

    # Test Long Straddle
    print("1. Long Straddle Strategy")
    long_straddle = LongStraddleStrategy()
    print(f"   Name: {long_straddle.name}")
    print(f"   IV Range: {long_straddle.min_iv_rank}-{long_straddle.max_iv_rank}")
    print(f"   Type: {long_straddle.strategy_type.value}")

    # Test Short Straddle
    print("\n2. Short Straddle Strategy")
    short_straddle = ShortStraddleStrategy()
    print(f"   Name: {short_straddle.name}")
    print(f"   IV Range: {short_straddle.min_iv_rank}-{short_straddle.max_iv_rank}")

    # Test Long Strangle
    print("\n3. Long Strangle Strategy")
    long_strangle = LongStrangleStrategy()
    print(f"   Name: {long_strangle.name}")
    print(f"   Target Delta: {long_strangle.target_delta}")

    # Test Short Strangle
    print("\n4. Short Strangle Strategy")
    short_strangle = ShortStrangleStrategy()
    print(f"   Name: {short_strangle.name}")
    print(f"   Target Delta: {short_strangle.target_delta}")

    # Test POP calculation
    print("\n5. Testing Probability Calculations")
    pop = long_straddle._calculate_straddle_pop(
        underlying=100, strike=100, premium=5,
        iv=0.30, dte=30, is_long=True
    )
    print(f"   Long Straddle POP: {pop:.1%}")

    pop = short_straddle._calculate_straddle_pop(
        underlying=100, strike=100, premium=5,
        iv=0.30, dte=30, is_long=False
    )
    print(f"   Short Straddle POP: {pop:.1%}")

    print("\n✅ Straddle/Strangle strategies ready!")
