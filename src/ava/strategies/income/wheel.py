"""
Wheel Strategy Implementation
=============================

The Wheel Strategy - Automated premium harvesting

Cycle:
1. Sell cash-secured puts on stocks you want to own
2. If assigned, own the stock at a discount (strike - premium)
3. Sell covered calls on the stock
4. If called away, restart with puts
5. Repeat, collecting premium throughout

AI-Enhanced Features:
- Dynamic strike selection based on IV rank
- Optimal DTE selection based on theta decay curve
- Smart roll decisions using LLM analysis
- Earnings avoidance
- Position sizing based on account size

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.ava.strategies.base import (
    OptionsStrategy,
    StrategyType,
    StrategySetup,
    OptionLeg,
    OptionType,
    OptionSide,
    EntrySignal,
    ExitSignal,
    SignalStrength,
    ExitReason,
    MarketContext,
    OptionsChain,
    register_strategy
)

logger = logging.getLogger(__name__)


class WheelPhase:
    """Current phase of the wheel strategy"""
    CASH = "cash"           # Have cash, looking to sell puts
    PUT_OPEN = "put_open"   # Short put position open
    ASSIGNED = "assigned"   # Put was assigned, own stock
    CALL_OPEN = "call_open" # Short call position open
    CALLED_AWAY = "called_away"  # Stock was called away


@dataclass
class WheelPosition:
    """Tracks a wheel strategy position"""
    symbol: str
    phase: str
    cycle_number: int = 1

    # Cash secured put phase
    put_strike: Optional[float] = None
    put_expiration: Optional[date] = None
    put_premium: float = 0.0

    # Stock ownership phase
    shares: int = 0
    cost_basis: float = 0.0  # Per share
    assignment_price: float = 0.0

    # Covered call phase
    call_strike: Optional[float] = None
    call_expiration: Optional[date] = None
    call_premium: float = 0.0

    # Totals
    total_premium_collected: float = 0.0
    realized_pnl: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'phase': self.phase,
            'cycle_number': self.cycle_number,
            'put_strike': self.put_strike,
            'put_premium': self.put_premium,
            'shares': self.shares,
            'cost_basis': self.cost_basis,
            'call_strike': self.call_strike,
            'call_premium': self.call_premium,
            'total_premium_collected': self.total_premium_collected
        }


@register_strategy
class WheelStrategy(OptionsStrategy):
    """
    The Wheel Strategy - Premium harvesting through CSPs and covered calls

    Parameters:
        target_delta: Target delta for options (default 0.30)
        min_premium_yield: Minimum annualized premium yield (default 12%)
        min_dte: Minimum days to expiration (default 21)
        max_dte: Maximum days to expiration (default 45)
        roll_dte_threshold: DTE at which to consider rolling (default 7)
        roll_profit_threshold: Profit % at which to roll early (default 50%)
        avoid_earnings: Avoid trades with earnings within DTE (default True)
        min_iv_rank: Minimum IV rank to enter (default 30)
    """

    name = "Wheel Strategy"
    description = "Sell cash-secured puts, get assigned, sell covered calls, repeat"
    strategy_type = StrategyType.INCOME

    # Strategy-specific parameters
    target_delta: float = 0.30          # ~70% POP
    min_premium_yield: float = 0.12     # 12% annualized
    min_dte: int = 21
    max_dte: int = 45
    optimal_dte: int = 30
    roll_dte_threshold: int = 7
    roll_profit_threshold: float = 0.50  # Roll at 50% profit
    avoid_earnings: bool = True
    min_iv_rank: float = 30.0
    max_iv_rank: float = 100.0

    # Covered call parameters
    call_strike_above_cost: float = 0.02  # Sell calls 2% above cost basis minimum

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_wheels: Dict[str, WheelPosition] = {}

    # =========================================================================
    # CORE STRATEGY METHODS
    # =========================================================================

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """
        Find wheel opportunities (CSPs or covered calls).

        For CSP phase: Find puts with target delta and good premium
        For CC phase: Find calls above cost basis with good premium
        """
        opportunities = []

        # Check if we have an existing wheel position
        wheel_pos = self.active_wheels.get(context.symbol)

        if wheel_pos and wheel_pos.phase == WheelPhase.ASSIGNED:
            # Need covered calls
            opportunities.extend(
                self._find_covered_call_opportunities(chain, context, wheel_pos)
            )
        else:
            # Default: Find CSP opportunities
            opportunities.extend(
                self._find_csp_opportunities(chain, context)
            )

        # Score and sort opportunities
        for opp in opportunities:
            self.score_setup(opp, context)

        opportunities.sort(key=lambda x: x.score, reverse=True)

        return opportunities

    def _find_csp_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find cash-secured put opportunities"""
        opportunities = []

        if chain.puts.empty:
            return opportunities

        # Filter valid expirations
        valid_expirations = [
            exp for exp in chain.expirations
            if self.filter_by_dte(exp)
        ]

        for expiration in valid_expirations:
            _, puts = chain.get_chain_for_expiration(expiration)

            if puts.empty:
                continue

            # Check earnings
            if self.avoid_earnings and context.days_to_earnings:
                dte = (expiration - date.today()).days
                if context.days_to_earnings <= dte:
                    continue  # Skip - earnings before expiration

            # Find put near target delta
            target_strike = chain.get_strike_by_delta(
                expiration, self.target_delta, OptionType.PUT
            )

            if target_strike is None:
                continue

            # Get the option data
            put_row = puts[puts['strike'] == target_strike]

            if put_row.empty:
                continue

            put_data = put_row.iloc[0]

            # Check minimum premium yield
            premium = (put_data.get('bid', 0) + put_data.get('ask', 0)) / 2
            dte = (expiration - date.today()).days

            if premium <= 0 or dte <= 0:
                continue

            annualized_yield = (premium / target_strike) * (365 / dte)

            if annualized_yield < self.min_premium_yield:
                continue

            # Create option leg
            leg = OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.SELL,
                strike=target_strike,
                expiration=expiration,
                quantity=1,
                bid=float(put_data.get('bid', 0)),
                ask=float(put_data.get('ask', 0)),
                last_price=float(put_data.get('last', premium)),
                implied_volatility=float(put_data.get('iv', context.implied_volatility)),
                delta=float(put_data.get('delta', -self.target_delta)),
                gamma=float(put_data.get('gamma', 0)),
                theta=float(put_data.get('theta', 0)),
                vega=float(put_data.get('vega', 0))
            )

            # Calculate risk/reward
            max_profit = premium * 100
            max_loss = (target_strike - premium) * 100  # Assignment risk
            breakeven = target_strike - premium

            # Create setup
            setup = StrategySetup(
                symbol=context.symbol,
                strategy_name="Cash Secured Put (Wheel)",
                legs=[leg],
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_prices=[breakeven],
                probability_of_profit=1 - abs(leg.delta),  # Approximation
                net_delta=leg.delta,
                net_gamma=leg.gamma,
                net_theta=leg.theta,
                net_vega=leg.vega,
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"Annualized yield: {annualized_yield:.1%}"
            )

            opportunities.append(setup)

        return opportunities

    def _find_covered_call_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext,
        wheel_pos: WheelPosition
    ) -> List[StrategySetup]:
        """Find covered call opportunities for assigned shares"""
        opportunities = []

        if chain.calls.empty:
            return opportunities

        # Minimum strike = cost basis + buffer
        min_strike = wheel_pos.cost_basis * (1 + self.call_strike_above_cost)

        # Filter valid expirations
        valid_expirations = [
            exp for exp in chain.expirations
            if self.filter_by_dte(exp)
        ]

        for expiration in valid_expirations:
            calls, _ = chain.get_chain_for_expiration(expiration)

            if calls.empty:
                continue

            # Find strikes above cost basis
            valid_calls = calls[calls['strike'] >= min_strike]

            if valid_calls.empty:
                continue

            # Find call near target delta
            for _, call_data in valid_calls.iterrows():
                delta = abs(float(call_data.get('delta', 0)))

                # Check if delta is reasonable
                if delta < 0.15 or delta > 0.45:
                    continue

                strike = float(call_data['strike'])
                premium = (call_data.get('bid', 0) + call_data.get('ask', 0)) / 2

                if premium <= 0:
                    continue

                leg = OptionLeg(
                    option_type=OptionType.CALL,
                    side=OptionSide.SELL,
                    strike=strike,
                    expiration=expiration,
                    quantity=1,
                    bid=float(call_data.get('bid', 0)),
                    ask=float(call_data.get('ask', 0)),
                    last_price=float(call_data.get('last', premium)),
                    implied_volatility=float(call_data.get('iv', context.implied_volatility)),
                    delta=float(call_data.get('delta', 0)),
                    gamma=float(call_data.get('gamma', 0)),
                    theta=float(call_data.get('theta', 0)),
                    vega=float(call_data.get('vega', 0))
                )

                # Calculate risk/reward for covered call
                max_profit = (strike - wheel_pos.cost_basis + premium) * 100
                breakeven = wheel_pos.cost_basis - premium

                setup = StrategySetup(
                    symbol=context.symbol,
                    strategy_name="Covered Call (Wheel)",
                    legs=[leg],
                    max_profit=max_profit,
                    max_loss=wheel_pos.cost_basis * 100,  # Stock goes to 0
                    breakeven_prices=[breakeven],
                    probability_of_profit=1 - abs(leg.delta),
                    net_delta=leg.delta,
                    net_gamma=leg.gamma,
                    net_theta=leg.theta,
                    net_vega=leg.vega,
                    underlying_price=context.current_price,
                    iv_rank=context.iv_rank,
                    notes=f"Cost basis: ${wheel_pos.cost_basis:.2f}"
                )

                opportunities.append(setup)

        return opportunities

    def entry_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> EntrySignal:
        """Check if entry conditions are met for a wheel trade"""
        reasons = []
        warnings = []
        should_enter = True
        strength = SignalStrength.MODERATE

        # Check IV rank
        if context.iv_rank < self.min_iv_rank:
            warnings.append(f"IV rank ({context.iv_rank:.0f}) below minimum ({self.min_iv_rank})")
            strength = SignalStrength.WEAK
        elif context.iv_rank > 50:
            reasons.append(f"Elevated IV rank ({context.iv_rank:.0f}) - good for selling premium")
            if context.iv_rank > 70:
                strength = SignalStrength.STRONG

        # Check earnings
        if self.avoid_earnings and context.days_to_earnings:
            dte = setup.days_to_expiration
            if context.days_to_earnings <= dte:
                warnings.append(f"Earnings in {context.days_to_earnings} days (before expiration)")
                should_enter = False
            elif context.days_to_earnings <= dte + 7:
                warnings.append(f"Earnings shortly after expiration ({context.days_to_earnings} days)")

        # Check probability of profit
        if setup.probability_of_profit < 0.60:
            warnings.append(f"POP ({setup.probability_of_profit:.0%}) below 60%")
            if setup.probability_of_profit < 0.50:
                should_enter = False
        else:
            reasons.append(f"Good POP ({setup.probability_of_profit:.0%})")

        # Check DTE
        dte = setup.days_to_expiration
        if dte < self.min_dte:
            warnings.append(f"DTE ({dte}) below minimum ({self.min_dte})")
            should_enter = False
        elif self.min_dte <= dte <= self.optimal_dte:
            reasons.append(f"Optimal DTE ({dte} days)")

        # Check theta
        if setup.net_theta > 0:
            reasons.append(f"Positive theta (${setup.net_theta:.2f}/day)")
        else:
            warnings.append("Negative theta position")

        # Check setup validity
        is_valid, errors = self.validate_setup(setup)
        if not is_valid:
            warnings.extend(errors)
            should_enter = False

        return EntrySignal(
            should_enter=should_enter,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings,
            suggested_position_size=self.calculate_position_size(setup, 100000, [])
        )

    def exit_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext,
        entry_price: float,
        current_price: float
    ) -> ExitSignal:
        """Check if exit conditions are met"""

        # Calculate P&L
        if entry_price != 0:
            pnl_pct = (entry_price - current_price) / entry_price
        else:
            pnl_pct = 0

        pnl = (entry_price - current_price) * 100  # For short position

        notes = []

        # Check profit target
        if pnl_pct >= self.roll_profit_threshold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close position - profit target reached",
                current_pnl=pnl,
                current_pnl_pct=pnl_pct,
                notes=[f"Profit of {pnl_pct:.0%} exceeds {self.roll_profit_threshold:.0%} target"]
            )

        # Check stop loss (for short premium, loss is when price increases)
        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close position - stop loss triggered",
                current_pnl=pnl,
                current_pnl_pct=pnl_pct,
                notes=[f"Loss of {abs(pnl_pct):.0%} exceeds {self.stop_loss_pct:.0%} stop"]
            )

        # Check DTE
        dte = setup.days_to_expiration
        if dte <= self.roll_dte_threshold:
            if pnl_pct > 0:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DTE_THRESHOLD,
                    urgency=SignalStrength.MODERATE,
                    suggested_action="Close or roll - approaching expiration with profit",
                    current_pnl=pnl,
                    current_pnl_pct=pnl_pct,
                    notes=[f"Only {dte} DTE remaining", f"Current profit: {pnl_pct:.0%}"]
                )
            else:
                notes.append(f"Low DTE ({dte}) with unrealized loss")

        # Check delta breach (position going ITM)
        if setup.legs:
            leg = setup.legs[0]
            if abs(leg.delta) > 0.70:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DELTA_BREACH,
                    urgency=SignalStrength.MODERATE,
                    suggested_action="Consider rolling - high delta indicates ITM risk",
                    current_pnl=pnl,
                    current_pnl_pct=pnl_pct,
                    notes=[f"Delta ({leg.delta:.2f}) indicates high ITM probability"]
                )

        # Check assignment risk
        if dte <= 1 and setup.legs:
            leg = setup.legs[0]
            if leg.option_type == OptionType.PUT:
                if context.current_price < leg.strike:
                    return ExitSignal(
                        should_exit=False,  # Let it assign for wheel
                        reason=ExitReason.ASSIGNMENT_RISK,
                        urgency=SignalStrength.MODERATE,
                        suggested_action="Prepare for assignment - stock is ITM",
                        current_pnl=pnl,
                        current_pnl_pct=pnl_pct,
                        notes=["Put is ITM - will likely be assigned", "Prepare to sell covered calls"]
                    )

        # Check upcoming earnings
        if self.avoid_earnings and context.days_to_earnings:
            if context.days_to_earnings <= dte and context.days_to_earnings <= 7:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.EARNINGS,
                    urgency=SignalStrength.STRONG,
                    suggested_action="Close before earnings",
                    current_pnl=pnl,
                    current_pnl_pct=pnl_pct,
                    notes=[f"Earnings in {context.days_to_earnings} days"]
                )

        # No exit signal
        return ExitSignal(
            should_exit=False,
            reason=ExitReason.MANUAL,
            urgency=SignalStrength.NONE,
            current_pnl=pnl,
            current_pnl_pct=pnl_pct,
            notes=notes if notes else ["Position within normal parameters"]
        )

    def calculate_position_size(
        self,
        setup: StrategySetup,
        account_value: float,
        current_positions: List[StrategySetup]
    ) -> int:
        """
        Calculate position size for wheel trade.

        For CSP: Based on cash required to secure the put
        For CC: Based on shares owned
        """
        if not setup.legs:
            return 0

        leg = setup.legs[0]

        if leg.option_type == OptionType.PUT:
            # Cash secured put - need strike * 100 per contract
            cash_per_contract = leg.strike * 100

            # Max position based on account percentage
            max_by_account = account_value * self.max_position_size_pct

            # Max position based on risk (max loss)
            max_by_risk = account_value * self.max_loss_pct
            if setup.max_loss > 0:
                max_contracts_by_risk = max_by_risk / setup.max_loss
            else:
                max_contracts_by_risk = float('inf')

            # Calculate contracts
            max_contracts = min(
                max_by_account / cash_per_contract,
                max_contracts_by_risk,
                10  # Hard cap
            )

            return max(1, int(max_contracts))

        else:
            # Covered call - one contract per 100 shares
            # This depends on shares owned
            return 1  # Default to 1, adjust based on actual shares

    # =========================================================================
    # WHEEL-SPECIFIC METHODS
    # =========================================================================

    def start_wheel(self, symbol: str) -> WheelPosition:
        """Start a new wheel cycle"""
        wheel = WheelPosition(
            symbol=symbol,
            phase=WheelPhase.CASH,
            cycle_number=1
        )
        self.active_wheels[symbol] = wheel
        logger.info(f"Started new wheel cycle for {symbol}")
        return wheel

    def record_put_sale(
        self,
        symbol: str,
        strike: float,
        expiration: date,
        premium: float
    ):
        """Record a CSP sale"""
        wheel = self.active_wheels.get(symbol)
        if not wheel:
            wheel = self.start_wheel(symbol)

        wheel.phase = WheelPhase.PUT_OPEN
        wheel.put_strike = strike
        wheel.put_expiration = expiration
        wheel.put_premium = premium
        wheel.total_premium_collected += premium

        logger.info(f"Wheel {symbol}: Sold {strike} put for ${premium:.2f}")

    def record_assignment(self, symbol: str, assignment_price: float, shares: int = 100):
        """Record put assignment"""
        wheel = self.active_wheels.get(symbol)
        if not wheel:
            logger.warning(f"No wheel position found for {symbol}")
            return

        wheel.phase = WheelPhase.ASSIGNED
        wheel.shares = shares
        wheel.assignment_price = assignment_price
        wheel.cost_basis = assignment_price - wheel.put_premium

        logger.info(f"Wheel {symbol}: Assigned at ${assignment_price:.2f}, cost basis ${wheel.cost_basis:.2f}")

    def record_call_sale(
        self,
        symbol: str,
        strike: float,
        expiration: date,
        premium: float
    ):
        """Record a covered call sale"""
        wheel = self.active_wheels.get(symbol)
        if not wheel:
            logger.warning(f"No wheel position found for {symbol}")
            return

        wheel.phase = WheelPhase.CALL_OPEN
        wheel.call_strike = strike
        wheel.call_expiration = expiration
        wheel.call_premium = premium
        wheel.total_premium_collected += premium

        logger.info(f"Wheel {symbol}: Sold {strike} call for ${premium:.2f}")

    def record_call_assignment(self, symbol: str, sale_price: float):
        """Record covered call assignment (shares called away)"""
        wheel = self.active_wheels.get(symbol)
        if not wheel:
            logger.warning(f"No wheel position found for {symbol}")
            return

        # Calculate P&L
        stock_pnl = (sale_price - wheel.cost_basis) * wheel.shares
        total_premium = wheel.total_premium_collected
        total_pnl = stock_pnl + total_premium

        wheel.phase = WheelPhase.CALLED_AWAY
        wheel.realized_pnl = total_pnl
        wheel.shares = 0

        logger.info(f"Wheel {symbol} Cycle {wheel.cycle_number} complete!")
        logger.info(f"  Stock P&L: ${stock_pnl:.2f}")
        logger.info(f"  Total premium: ${total_premium:.2f}")
        logger.info(f"  Total P&L: ${total_pnl:.2f}")

        # Start new cycle
        wheel.cycle_number += 1
        wheel.phase = WheelPhase.CASH
        wheel.put_strike = None
        wheel.put_premium = 0
        wheel.call_strike = None
        wheel.call_premium = 0
        wheel.cost_basis = 0

    def get_wheel_status(self, symbol: str) -> Optional[Dict]:
        """Get current status of wheel position"""
        wheel = self.active_wheels.get(symbol)
        if wheel:
            return wheel.to_dict()
        return None

    def score_setup(self, setup: StrategySetup, context: MarketContext) -> float:
        """
        Custom scoring for wheel strategy.

        Weights:
        - Premium yield (25%)
        - IV rank (20%)
        - Probability of profit (20%)
        - DTE optimization (15%)
        - Theta (10%)
        - Liquidity (10%)
        """
        score = 0.0
        components = {}

        # Premium yield (25%)
        if setup.legs:
            leg = setup.legs[0]
            premium = leg.mid_price
            strike = leg.strike
            dte = setup.days_to_expiration

            if dte > 0 and strike > 0:
                annualized_yield = (premium / strike) * (365 / dte)
                yield_score = min(annualized_yield / 0.20, 1.0) * 25  # Cap at 20% yield
                components['premium_yield'] = yield_score
                score += yield_score

        # IV rank (20%)
        if context.iv_rank >= self.min_iv_rank:
            iv_score = min(context.iv_rank / 100, 1.0) * 20
            components['iv_rank'] = iv_score
            score += iv_score

        # Probability of profit (20%)
        pop_score = setup.probability_of_profit * 20
        components['pop'] = pop_score
        score += pop_score

        # DTE optimization (15%) - prefer ~30 DTE
        dte = setup.days_to_expiration
        dte_diff = abs(dte - self.optimal_dte)
        dte_score = max(0, (30 - dte_diff) / 30) * 15
        components['dte'] = dte_score
        score += dte_score

        # Theta (10%)
        if setup.net_theta > 0:
            theta_score = min(setup.net_theta / 5, 1.0) * 10
            components['theta'] = theta_score
            score += theta_score

        # Liquidity (10%)
        if setup.legs:
            leg = setup.legs[0]
            if leg.mid_price > 0:
                spread = (leg.ask - leg.bid) / leg.mid_price
                liquidity_score = max(0, (0.10 - spread) / 0.10) * 10
                components['liquidity'] = liquidity_score
                score += liquidity_score

        setup.score = round(score, 2)
        setup.score_components = components

        return score


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Wheel Strategy ===\n")

    # Create strategy instance
    wheel = WheelStrategy(
        target_delta=0.30,
        min_dte=21,
        max_dte=45,
        min_iv_rank=30
    )

    print(f"Strategy: {wheel.name}")
    print(f"Type: {wheel.strategy_type.value}")
    print(f"Target delta: {wheel.target_delta}")
    print(f"DTE range: {wheel.min_dte}-{wheel.max_dte}")
    print(f"Min IV rank: {wheel.min_iv_rank}")

    # Test wheel tracking
    print("\n--- Testing Wheel Cycle Tracking ---")

    wheel.start_wheel('NVDA')
    wheel.record_put_sale('NVDA', 500.0, date.today() + timedelta(days=30), 8.50)

    status = wheel.get_wheel_status('NVDA')
    print(f"After CSP sale: {status}")

    wheel.record_assignment('NVDA', 500.0)
    status = wheel.get_wheel_status('NVDA')
    print(f"After assignment: {status}")

    wheel.record_call_sale('NVDA', 520.0, date.today() + timedelta(days=30), 6.00)
    status = wheel.get_wheel_status('NVDA')
    print(f"After CC sale: {status}")

    wheel.record_call_assignment('NVDA', 520.0)
    status = wheel.get_wheel_status('NVDA')
    print(f"After called away: {status}")

    print("\n✅ Wheel Strategy ready!")
