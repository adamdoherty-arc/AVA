"""
Vertical Spread Strategies
==========================

Defined risk directional strategies using options at different strikes
but same expiration.

Strategies:
1. Bull Put Spread (Credit): Bullish, sell put, buy lower put
2. Bear Call Spread (Credit): Bearish, sell call, buy higher call
3. Bull Call Spread (Debit): Bullish, buy call, sell higher call
4. Bear Put Spread (Debit): Bearish, buy put, sell lower put

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
# BASE VERTICAL SPREAD CLASS
# =============================================================================

class VerticalSpreadStrategy(OptionsStrategy):
    """Base class for vertical spread strategies"""

    # Parameters
    short_delta: float = 0.30
    spread_width: float = 5.0      # Points between strikes
    spread_width_pct: float = 0.02  # Or as % of underlying

    min_dte: int = 21
    max_dte: int = 60
    optimal_dte: int = 45

    min_credit_pct: float = 0.25   # Min credit as % of width (for credit spreads)

    profit_target_pct: float = 0.50
    stop_loss_pct: float = 2.0
    dte_close_threshold: int = 7

    def _calculate_spread_width(self, underlying_price: float) -> float:
        """Calculate appropriate spread width"""
        width = max(self.spread_width, underlying_price * self.spread_width_pct)
        # Round to standard increments
        if underlying_price > 100:
            return round(width / 5) * 5
        else:
            return round(width / 2.5) * 2.5

    def _calculate_pop(
        self,
        underlying: float,
        short_strike: float,
        iv: float,
        dte: int,
        is_put: bool
    ) -> float:
        """Calculate probability of profit for vertical spread"""
        if iv <= 0 or dte <= 0:
            return 0.5

        t = dte / 365
        d2 = (np.log(underlying / short_strike) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))

        if is_put:
            # Bull put: want price to stay above short put
            return stats.norm.cdf(d2)
        else:
            # Bear call: want price to stay below short call
            return 1 - stats.norm.cdf(d2)


# =============================================================================
# BULL PUT SPREAD (Credit)
# =============================================================================

@register_strategy
class BullPutSpreadStrategy(VerticalSpreadStrategy):
    """
    Bull Put Spread (Put Credit Spread)

    Structure:
    - Sell OTM put (higher strike)
    - Buy further OTM put (lower strike)

    Characteristics:
    - CREDIT spread (receive premium)
    - Bullish/neutral bias
    - Max profit = credit received (price stays above short strike)
    - Max loss = width - credit (price below long strike)
    - Defined risk

    Best Conditions:
    - Moderately bullish outlook
    - High IV rank (50+) for rich premium
    - Support level at or below short strike
    """

    name: str = "Bull Put Spread"
    description: str = "Bullish credit spread selling puts"
    strategy_type: StrategyType = StrategyType.INCOME

    short_delta: float = 0.30
    min_iv_rank: float = 40.0
    max_iv_rank: float = 100.0

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Bull Put Spread opportunities"""
        opportunities = []

        if context.iv_rank < self.min_iv_rank:
            return opportunities

        # Avoid earnings for credit spreads
        if context.days_to_earnings and context.days_to_earnings < 14:
            return opportunities

        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            _, puts = chain.get_chain_for_expiration(expiration)
            if puts.empty:
                continue

            # Find short put strike
            short_strike = chain.get_strike_by_delta(expiration, self.short_delta, OptionType.PUT)
            if not short_strike:
                continue

            # Calculate spread width
            width = self._calculate_spread_width(context.current_price)
            long_strike = short_strike - width

            # Get option data
            short_put = puts[puts['strike'] == short_strike]
            long_put = puts[puts['strike'] == long_strike]

            if short_put.empty or long_put.empty:
                # Find closest available strikes
                available_strikes = puts['strike'].unique()
                long_strike = float(available_strikes[np.argmin(np.abs(available_strikes - long_strike))])
                long_put = puts[puts['strike'] == long_strike]
                if long_put.empty:
                    continue

            sp = short_put.iloc[0]
            lp = long_put.iloc[0]

            # Calculate credit
            credit = sp.get('bid', 0) - lp.get('ask', 0)
            if credit <= 0:
                continue

            actual_width = short_strike - long_strike
            credit_pct = credit / actual_width
            if credit_pct < self.min_credit_pct:
                continue

            legs = [
                OptionLeg(
                    option_type=OptionType.PUT,
                    side=OptionSide.SELL,
                    strike=short_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=sp.get('bid', 0),
                    ask=sp.get('ask', 0),
                    delta=sp.get('delta', -self.short_delta),
                    theta=sp.get('theta', 0),
                    vega=sp.get('vega', 0)
                ),
                OptionLeg(
                    option_type=OptionType.PUT,
                    side=OptionSide.BUY,
                    strike=long_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=lp.get('bid', 0),
                    ask=lp.get('ask', 0),
                    delta=lp.get('delta', 0),
                    theta=lp.get('theta', 0),
                    vega=lp.get('vega', 0)
                )
            ]

            max_profit = credit * 100
            max_loss = (actual_width - credit) * 100

            dte = (expiration - date.today()).days
            pop = self._calculate_pop(context.current_price, short_strike, context.implied_volatility, dte, is_put=True)

            net_greeks = self.aggregate_greeks(legs)

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_prices=[short_strike - credit],
                probability_of_profit=pop,
                net_delta=net_greeks['delta'],
                net_theta=net_greeks['theta'],
                net_vega=net_greeks['vega'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"${long_strike}/${short_strike}P, Credit ${credit:.2f} ({credit_pct:.1%})"
            )

            self.score_setup(setup, context)
            opportunities.append(setup)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def entry_conditions(self, setup: StrategySetup, context: MarketContext) -> EntrySignal:
        """Entry conditions for Bull Put Spread"""
        reasons = []
        warnings = []

        if context.iv_rank < self.min_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too low"]
            )

        if context.iv_rank >= 60:
            reasons.append(f"High IV: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        elif context.iv_rank >= 50:
            reasons.append(f"Good IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        if setup.probability_of_profit >= 0.70:
            reasons.append(f"High POP: {setup.probability_of_profit:.1%}")
        elif setup.probability_of_profit >= 0.60:
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
        """Exit conditions for Bull Put Spread"""

        # Credit spread: P&L = credit received - debit to close
        pnl = entry_price - current_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close spread - profit target",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close spread - stop loss",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Check if approaching short strike
        short_strike = max(leg.strike for leg in setup.legs if leg.is_put and leg.is_short)
        if context.current_price <= short_strike * 1.02:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - price near short strike",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if setup.days_to_expiration <= self.dte_close_threshold:
            if pnl_pct > 0:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DTE_THRESHOLD,
                    urgency=SignalStrength.MODERATE,
                    suggested_action="Close profitable spread near expiration",
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
        """Position size for Bull Put Spread"""
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0

        max_risk = account_value * self.max_loss_pct
        return max(int(max_risk / max_loss), 0)


# =============================================================================
# BEAR CALL SPREAD (Credit)
# =============================================================================

@register_strategy
class BearCallSpreadStrategy(VerticalSpreadStrategy):
    """
    Bear Call Spread (Call Credit Spread)

    Structure:
    - Sell OTM call (lower strike)
    - Buy further OTM call (higher strike)

    Characteristics:
    - CREDIT spread
    - Bearish/neutral bias
    - Max profit = credit (price stays below short strike)
    - Max loss = width - credit
    """

    name: str = "Bear Call Spread"
    description: str = "Bearish credit spread selling calls"
    strategy_type: StrategyType = StrategyType.INCOME

    short_delta: float = 0.30
    min_iv_rank: float = 40.0
    max_iv_rank: float = 100.0

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Bear Call Spread opportunities"""
        opportunities = []

        if context.iv_rank < self.min_iv_rank:
            return opportunities

        if context.days_to_earnings and context.days_to_earnings < 14:
            return opportunities

        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            calls, _ = chain.get_chain_for_expiration(expiration)
            if calls.empty:
                continue

            short_strike = chain.get_strike_by_delta(expiration, self.short_delta, OptionType.CALL)
            if not short_strike:
                continue

            width = self._calculate_spread_width(context.current_price)
            long_strike = short_strike + width

            short_call = calls[calls['strike'] == short_strike]
            long_call = calls[calls['strike'] == long_strike]

            if short_call.empty or long_call.empty:
                available_strikes = calls['strike'].unique()
                long_strike = float(available_strikes[np.argmin(np.abs(available_strikes - long_strike))])
                long_call = calls[calls['strike'] == long_strike]
                if long_call.empty:
                    continue

            sc = short_call.iloc[0]
            lc = long_call.iloc[0]

            credit = sc.get('bid', 0) - lc.get('ask', 0)
            if credit <= 0:
                continue

            actual_width = long_strike - short_strike
            credit_pct = credit / actual_width
            if credit_pct < self.min_credit_pct:
                continue

            legs = [
                OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.SELL,
                    strike=short_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=sc.get('bid', 0),
                    ask=sc.get('ask', 0),
                    delta=sc.get('delta', self.short_delta),
                    theta=sc.get('theta', 0)
                ),
                OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.BUY,
                    strike=long_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=lc.get('bid', 0),
                    ask=lc.get('ask', 0),
                    delta=lc.get('delta', 0),
                    theta=lc.get('theta', 0)
                )
            ]

            max_profit = credit * 100
            max_loss = (actual_width - credit) * 100

            dte = (expiration - date.today()).days
            pop = self._calculate_pop(context.current_price, short_strike, context.implied_volatility, dte, is_put=False)

            net_greeks = self.aggregate_greeks(legs)

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_prices=[short_strike + credit],
                probability_of_profit=pop,
                net_delta=net_greeks['delta'],
                net_theta=net_greeks['theta'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"${short_strike}/${long_strike}C, Credit ${credit:.2f}"
            )

            self.score_setup(setup, context)
            opportunities.append(setup)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def entry_conditions(self, setup: StrategySetup, context: MarketContext) -> EntrySignal:
        """Entry conditions for Bear Call Spread"""
        reasons = []
        warnings = []

        if context.iv_rank < self.min_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too low"]
            )

        if context.iv_rank >= 60:
            reasons.append(f"High IV: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        else:
            reasons.append(f"Good IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE

        if setup.probability_of_profit >= 0.70:
            reasons.append(f"High POP: {setup.probability_of_profit:.1%}")

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings
        )

    def exit_conditions(self, setup: StrategySetup, context: MarketContext, entry_price: float, current_price: float) -> ExitSignal:
        """Exit conditions for Bear Call Spread"""

        pnl = entry_price - current_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close spread",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close spread",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        short_strike = min(leg.strike for leg in setup.legs if leg.is_call and leg.is_short)
        if context.current_price >= short_strike * 0.98:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - price approaching short strike",
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

    def calculate_position_size(self, setup: StrategySetup, account_value: float, current_positions: List[StrategySetup]) -> int:
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0
        max_risk = account_value * self.max_loss_pct
        return max(int(max_risk / max_loss), 0)


# =============================================================================
# BULL CALL SPREAD (Debit)
# =============================================================================

@register_strategy
class BullCallSpreadStrategy(VerticalSpreadStrategy):
    """
    Bull Call Spread (Call Debit Spread)

    Structure:
    - Buy ATM/ITM call (lower strike)
    - Sell OTM call (higher strike)

    Characteristics:
    - DEBIT spread (pay premium)
    - Bullish bias
    - Max profit = width - debit
    - Max loss = debit paid
    - Better for lower IV environments
    """

    name: str = "Bull Call Spread"
    description: str = "Bullish debit spread buying calls"
    strategy_type: StrategyType = StrategyType.DIRECTIONAL

    long_delta: float = 0.60
    short_delta: float = 0.30
    min_iv_rank: float = 0.0
    max_iv_rank: float = 50.0  # Prefer lower IV

    profit_target_pct: float = 0.50
    stop_loss_pct: float = 0.50

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Bull Call Spread opportunities"""
        opportunities = []

        if context.iv_rank > self.max_iv_rank:
            return opportunities

        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            calls, _ = chain.get_chain_for_expiration(expiration)
            if calls.empty:
                continue

            long_strike = chain.get_strike_by_delta(expiration, self.long_delta, OptionType.CALL)
            short_strike = chain.get_strike_by_delta(expiration, self.short_delta, OptionType.CALL)

            if not long_strike or not short_strike or long_strike >= short_strike:
                continue

            long_call = calls[calls['strike'] == long_strike]
            short_call = calls[calls['strike'] == short_strike]

            if long_call.empty or short_call.empty:
                continue

            lc = long_call.iloc[0]
            sc = short_call.iloc[0]

            # Debit = buy long - sell short
            debit = lc.get('ask', 0) - sc.get('bid', 0)
            if debit <= 0:
                continue

            width = short_strike - long_strike

            legs = [
                OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.BUY,
                    strike=long_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=lc.get('bid', 0),
                    ask=lc.get('ask', 0),
                    delta=lc.get('delta', self.long_delta),
                    theta=lc.get('theta', 0)
                ),
                OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.SELL,
                    strike=short_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=sc.get('bid', 0),
                    ask=sc.get('ask', 0),
                    delta=sc.get('delta', self.short_delta),
                    theta=sc.get('theta', 0)
                )
            ]

            max_profit = (width - debit) * 100
            max_loss = debit * 100

            net_greeks = self.aggregate_greeks(legs)

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_prices=[long_strike + debit],
                probability_of_profit=0.45,  # Estimate
                net_delta=net_greeks['delta'],
                net_theta=net_greeks['theta'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"${long_strike}/${short_strike}C, Debit ${debit:.2f}"
            )

            self.score_setup(setup, context)
            opportunities.append(setup)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def entry_conditions(self, setup: StrategySetup, context: MarketContext) -> EntrySignal:
        reasons = []
        warnings = []

        if context.iv_rank > self.max_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too high for debit spread"]
            )

        if context.iv_rank < 30:
            reasons.append(f"Low IV favorable for debit spreads: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        else:
            reasons.append(f"Moderate IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE

        if setup.net_delta > 0.20:
            reasons.append(f"Positive delta exposure: {setup.net_delta:.2f}")

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings
        )

    def exit_conditions(self, setup: StrategySetup, context: MarketContext, entry_price: float, current_price: float) -> ExitSignal:
        # Debit spread: P&L = current value - entry cost
        pnl = current_price - entry_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close spread",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close spread",
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

    def calculate_position_size(self, setup: StrategySetup, account_value: float, current_positions: List[StrategySetup]) -> int:
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0
        max_risk = account_value * self.max_loss_pct
        return max(int(max_risk / max_loss), 0)


# =============================================================================
# BEAR PUT SPREAD (Debit)
# =============================================================================

@register_strategy
class BearPutSpreadStrategy(VerticalSpreadStrategy):
    """
    Bear Put Spread (Put Debit Spread)

    Structure:
    - Buy ATM/ITM put (higher strike)
    - Sell OTM put (lower strike)

    Characteristics:
    - DEBIT spread
    - Bearish bias
    - Max profit = width - debit
    - Max loss = debit paid
    """

    name: str = "Bear Put Spread"
    description: str = "Bearish debit spread buying puts"
    strategy_type: StrategyType = StrategyType.DIRECTIONAL

    long_delta: float = 0.60
    short_delta: float = 0.30
    min_iv_rank: float = 0.0
    max_iv_rank: float = 50.0

    profit_target_pct: float = 0.50
    stop_loss_pct: float = 0.50

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find Bear Put Spread opportunities"""
        opportunities = []

        if context.iv_rank > self.max_iv_rank:
            return opportunities

        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            _, puts = chain.get_chain_for_expiration(expiration)
            if puts.empty:
                continue

            long_strike = chain.get_strike_by_delta(expiration, self.long_delta, OptionType.PUT)
            short_strike = chain.get_strike_by_delta(expiration, self.short_delta, OptionType.PUT)

            if not long_strike or not short_strike or long_strike <= short_strike:
                continue

            long_put = puts[puts['strike'] == long_strike]
            short_put = puts[puts['strike'] == short_strike]

            if long_put.empty or short_put.empty:
                continue

            lp = long_put.iloc[0]
            sp = short_put.iloc[0]

            debit = lp.get('ask', 0) - sp.get('bid', 0)
            if debit <= 0:
                continue

            width = long_strike - short_strike

            legs = [
                OptionLeg(
                    option_type=OptionType.PUT,
                    side=OptionSide.BUY,
                    strike=long_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=lp.get('bid', 0),
                    ask=lp.get('ask', 0),
                    delta=lp.get('delta', -self.long_delta),
                    theta=lp.get('theta', 0)
                ),
                OptionLeg(
                    option_type=OptionType.PUT,
                    side=OptionSide.SELL,
                    strike=short_strike,
                    expiration=expiration,
                    quantity=1,
                    bid=sp.get('bid', 0),
                    ask=sp.get('ask', 0),
                    delta=sp.get('delta', -self.short_delta),
                    theta=sp.get('theta', 0)
                )
            ]

            max_profit = (width - debit) * 100
            max_loss = debit * 100

            net_greeks = self.aggregate_greeks(legs)

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_prices=[long_strike - debit],
                probability_of_profit=0.45,
                net_delta=net_greeks['delta'],
                net_theta=net_greeks['theta'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"${short_strike}/${long_strike}P, Debit ${debit:.2f}"
            )

            self.score_setup(setup, context)
            opportunities.append(setup)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def entry_conditions(self, setup: StrategySetup, context: MarketContext) -> EntrySignal:
        reasons = []
        warnings = []

        if context.iv_rank > self.max_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too high"]
            )

        if context.iv_rank < 30:
            reasons.append(f"Low IV: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        else:
            strength = SignalStrength.MODERATE

        if setup.net_delta < -0.20:
            reasons.append(f"Negative delta: {setup.net_delta:.2f}")

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings
        )

    def exit_conditions(self, setup: StrategySetup, context: MarketContext, entry_price: float, current_price: float) -> ExitSignal:
        pnl = current_price - entry_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close spread",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close spread",
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

    def calculate_position_size(self, setup: StrategySetup, account_value: float, current_positions: List[StrategySetup]) -> int:
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0
        max_risk = account_value * self.max_loss_pct
        return max(int(max_risk / max_loss), 0)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Vertical Spread Strategies ===\n")

    print("1. Bull Put Spread (Credit)")
    bps = BullPutSpreadStrategy()
    print(f"   Name: {bps.name}")
    print(f"   Type: {bps.strategy_type.value}")
    print(f"   Short Delta: {bps.short_delta}")

    print("\n2. Bear Call Spread (Credit)")
    bcs = BearCallSpreadStrategy()
    print(f"   Name: {bcs.name}")
    print(f"   Short Delta: {bcs.short_delta}")

    print("\n3. Bull Call Spread (Debit)")
    bcs = BullCallSpreadStrategy()
    print(f"   Name: {bcs.name}")
    print(f"   Long Delta: {bcs.long_delta}")

    print("\n4. Bear Put Spread (Debit)")
    bps = BearPutSpreadStrategy()
    print(f"   Name: {bps.name}")
    print(f"   Long Delta: {bps.long_delta}")

    print("\n✅ Vertical spread strategies ready!")
