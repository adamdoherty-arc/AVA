"""
Iron Condor Strategy
====================

A market-neutral options strategy that profits from low volatility and time decay.

Structure:
- Sell OTM Put (short put leg)
- Buy further OTM Put (long put wing - protection)
- Sell OTM Call (short call leg)
- Buy further OTM Call (long call wing - protection)

Characteristics:
- Limited risk, limited reward
- Profits when underlying stays within a range
- Benefits from theta decay and IV contraction
- Ideal for high IV rank environments (>50)
- Best DTE: 30-60 days

Profit Zone: Between short strikes
Max Profit: Net credit received
Max Loss: Width of spread - net credit

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


@dataclass
class IronCondorPosition:
    """Tracks an active Iron Condor position"""
    symbol: str
    entry_date: date
    expiration: date

    # Strikes
    long_put_strike: float
    short_put_strike: float
    short_call_strike: float
    long_call_strike: float

    # Premiums
    entry_credit: float  # Total credit received
    current_value: float = 0.0

    # Greeks at entry
    entry_delta: float = 0.0
    entry_theta: float = 0.0
    entry_vega: float = 0.0

    # Status
    quantity: int = 1
    is_closed: bool = False
    close_date: Optional[date] = None
    realized_pnl: float = 0.0

    @property
    def put_spread_width(self) -> float:
        return self.short_put_strike - self.long_put_strike

    @property
    def call_spread_width(self) -> float:
        return self.long_call_strike - self.short_call_strike

    @property
    def wing_width(self) -> float:
        """Width of each spread (assuming symmetric)"""
        return self.put_spread_width

    @property
    def profit_zone_width(self) -> float:
        """Width between short strikes"""
        return self.short_call_strike - self.short_put_strike

    @property
    def max_profit(self) -> float:
        return self.entry_credit * 100 * self.quantity

    @property
    def max_loss(self) -> float:
        return (self.wing_width * 100 - self.entry_credit * 100) * self.quantity

    @property
    def days_to_expiration(self) -> int:
        return (self.expiration - date.today()).days

    @property
    def unrealized_pnl(self) -> float:
        if self.is_closed:
            return 0.0
        return (self.entry_credit - self.current_value) * 100 * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_credit == 0:
            return 0.0
        return self.unrealized_pnl / self.max_profit


@register_strategy
class IronCondorStrategy(OptionsStrategy):
    """
    Iron Condor strategy implementation.

    Entry Criteria:
    - IV Rank > 50 (elevated volatility)
    - DTE between 30-60 days
    - No earnings within 14 days
    - Liquid options chain

    Strike Selection:
    - Short strikes: ~0.20-0.30 delta (configurable)
    - Wing width: 5-20 points depending on underlying price
    - Symmetric or asymmetric based on market bias

    Exit Criteria:
    - Profit target: 50% of max profit (configurable)
    - Stop loss: 100-200% of credit received
    - DTE < 14 days (roll or close)
    - Short strike breached
    """

    name: str = "Iron Condor"
    description: str = "Market-neutral strategy profiting from range-bound movement"
    strategy_type: StrategyType = StrategyType.NEUTRAL

    # Iron Condor specific parameters
    target_short_delta: float = 0.20  # Delta for short strikes
    min_wing_width: float = 5.0       # Minimum wing width
    max_wing_width: float = 20.0      # Maximum wing width
    wing_width_pct: float = 0.02      # Wing width as % of underlying (2%)

    # Entry criteria
    min_iv_rank: float = 40.0         # Prefer elevated IV
    max_iv_rank: float = 100.0
    min_dte: int = 30
    max_dte: int = 60
    optimal_dte: int = 45

    # Risk management
    profit_target_pct: float = 0.50   # Close at 50% profit
    stop_loss_pct: float = 2.0        # Stop at 200% of credit
    adjustment_threshold: float = 0.30  # Adjust if delta exceeds
    min_credit_pct: float = 0.20      # Min credit as % of wing width

    # Exit thresholds
    dte_close_threshold: int = 14     # Close if DTE < this
    short_strike_buffer: float = 0.02  # Close if price within 2% of short strike

    # Position tracking
    positions: Dict[str, List[IronCondorPosition]] = field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.positions = {}

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """
        Find valid Iron Condor setups in the options chain.

        Scans for optimal strike combinations based on:
        - Target delta for short strikes
        - Appropriate wing widths
        - Credit received vs risk
        """
        opportunities = []

        # Filter by IV rank
        if not self.filter_by_iv_rank(context):
            logger.debug(f"IV rank {context.iv_rank:.1f} outside range [{self.min_iv_rank}, {self.max_iv_rank}]")
            return opportunities

        # Check for earnings
        if not context.days_to_earnings_safe:
            logger.debug(f"Earnings in {context.days_to_earnings} days - skipping")
            return opportunities

        # Scan each expiration
        for expiration in chain.expirations:
            if not self.filter_by_dte(expiration):
                continue

            setups = self._find_condors_for_expiration(chain, context, expiration)
            opportunities.extend(setups)

        # Sort by score
        for setup in opportunities:
            self.score_setup(setup, context)

        opportunities.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Found {len(opportunities)} Iron Condor opportunities for {chain.symbol}")
        return opportunities

    def _find_condors_for_expiration(
        self,
        chain: OptionsChain,
        context: MarketContext,
        expiration: date
    ) -> List[StrategySetup]:
        """Find Iron Condor setups for a specific expiration"""
        setups = []

        calls, puts = chain.get_chain_for_expiration(expiration)
        if calls.empty or puts.empty:
            return setups

        underlying = context.current_price

        # Calculate wing width based on underlying price
        wing_width = max(
            self.min_wing_width,
            min(underlying * self.wing_width_pct, self.max_wing_width)
        )
        # Round to nearest standard strike increment
        if underlying > 100:
            wing_width = round(wing_width / 5) * 5
        else:
            wing_width = round(wing_width / 2.5) * 2.5
        wing_width = max(wing_width, 5.0)

        # Find short put strike (~target delta)
        short_put_strike = chain.get_strike_by_delta(
            expiration, self.target_short_delta, OptionType.PUT
        )
        if short_put_strike is None:
            return setups

        # Find short call strike (~target delta)
        short_call_strike = chain.get_strike_by_delta(
            expiration, self.target_short_delta, OptionType.CALL
        )
        if short_call_strike is None:
            return setups

        # Calculate wing strikes
        long_put_strike = short_put_strike - wing_width
        long_call_strike = short_call_strike + wing_width

        # Validate strikes exist in chain
        available_put_strikes = puts['strike'].unique()
        available_call_strikes = calls['strike'].unique()

        # Find closest available strikes
        long_put_strike = self._find_closest_strike(long_put_strike, available_put_strikes)
        long_call_strike = self._find_closest_strike(long_call_strike, available_call_strikes)

        if long_put_strike is None or long_call_strike is None:
            return setups

        # Get option data for each leg
        short_put = self._get_option_data(puts, short_put_strike)
        long_put = self._get_option_data(puts, long_put_strike)
        short_call = self._get_option_data(calls, short_call_strike)
        long_call = self._get_option_data(calls, long_call_strike)

        if not all([short_put, long_put, short_call, long_call]):
            return setups

        # Build the legs
        legs = [
            OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.BUY,
                strike=long_put_strike,
                expiration=expiration,
                quantity=1,
                bid=long_put.get('bid', 0),
                ask=long_put.get('ask', 0),
                last_price=long_put.get('last', 0),
                implied_volatility=long_put.get('iv', 0),
                delta=long_put.get('delta', 0),
                gamma=long_put.get('gamma', 0),
                theta=long_put.get('theta', 0),
                vega=long_put.get('vega', 0)
            ),
            OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.SELL,
                strike=short_put_strike,
                expiration=expiration,
                quantity=1,
                bid=short_put.get('bid', 0),
                ask=short_put.get('ask', 0),
                last_price=short_put.get('last', 0),
                implied_volatility=short_put.get('iv', 0),
                delta=short_put.get('delta', 0),
                gamma=short_put.get('gamma', 0),
                theta=short_put.get('theta', 0),
                vega=short_put.get('vega', 0)
            ),
            OptionLeg(
                option_type=OptionType.CALL,
                side=OptionSide.SELL,
                strike=short_call_strike,
                expiration=expiration,
                quantity=1,
                bid=short_call.get('bid', 0),
                ask=short_call.get('ask', 0),
                last_price=short_call.get('last', 0),
                implied_volatility=short_call.get('iv', 0),
                delta=short_call.get('delta', 0),
                gamma=short_call.get('gamma', 0),
                theta=short_call.get('theta', 0),
                vega=short_call.get('vega', 0)
            ),
            OptionLeg(
                option_type=OptionType.CALL,
                side=OptionSide.BUY,
                strike=long_call_strike,
                expiration=expiration,
                quantity=1,
                bid=long_call.get('bid', 0),
                ask=long_call.get('ask', 0),
                last_price=long_call.get('last', 0),
                implied_volatility=long_call.get('iv', 0),
                delta=long_call.get('delta', 0),
                gamma=long_call.get('gamma', 0),
                theta=long_call.get('theta', 0),
                vega=long_call.get('vega', 0)
            )
        ]

        # Calculate net credit
        net_credit = (
            short_put['bid'] - long_put['ask'] +
            short_call['bid'] - long_call['ask']
        )

        if net_credit <= 0:
            logger.debug("No net credit available - skipping")
            return setups

        # Calculate max profit/loss
        put_width = short_put_strike - long_put_strike
        call_width = long_call_strike - short_call_strike
        max_width = max(put_width, call_width)

        max_profit = net_credit * 100
        max_loss = (max_width - net_credit) * 100

        # Check minimum credit threshold
        credit_pct = net_credit / max_width
        if credit_pct < self.min_credit_pct:
            logger.debug(f"Credit {credit_pct:.1%} below minimum {self.min_credit_pct:.1%}")
            return setups

        # Calculate probability of profit
        pop = self._calculate_pop(
            underlying,
            short_put_strike,
            short_call_strike,
            context.implied_volatility,
            (expiration - date.today()).days
        )

        # Calculate aggregate Greeks
        net_greeks = self.aggregate_greeks(legs)

        # Create setup
        setup = StrategySetup(
            symbol=chain.symbol,
            strategy_name=self.name,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_prices=[short_put_strike - net_credit, short_call_strike + net_credit],
            probability_of_profit=pop,
            probability_of_max_profit=self._calculate_prob_max_profit(
                underlying, short_put_strike, short_call_strike,
                context.implied_volatility, (expiration - date.today()).days
            ),
            probability_of_max_loss=1 - pop,
            net_delta=net_greeks['delta'],
            net_gamma=net_greeks['gamma'],
            net_theta=net_greeks['theta'],
            net_vega=net_greeks['vega'],
            underlying_price=underlying,
            iv_rank=context.iv_rank,
            iv_percentile=context.iv_percentile,
            notes=f"Wing width: ${max_width:.0f}, Credit: ${net_credit:.2f} ({credit_pct:.1%})"
        )

        setups.append(setup)

        # Try variations with different wing widths
        for width_multiplier in [0.5, 1.5, 2.0]:
            alt_width = wing_width * width_multiplier
            if self.min_wing_width <= alt_width <= self.max_wing_width:
                alt_setups = self._create_variant_condor(
                    chain, context, expiration,
                    short_put_strike, short_call_strike,
                    alt_width, puts, calls
                )
                setups.extend(alt_setups)

        return setups

    def _create_variant_condor(
        self,
        chain: OptionsChain,
        context: MarketContext,
        expiration: date,
        short_put_strike: float,
        short_call_strike: float,
        wing_width: float,
        puts: pd.DataFrame,
        calls: pd.DataFrame
    ) -> List[StrategySetup]:
        """Create an Iron Condor variant with different wing width"""
        setups = []

        available_put_strikes = puts['strike'].unique()
        available_call_strikes = calls['strike'].unique()

        long_put_strike = self._find_closest_strike(short_put_strike - wing_width, available_put_strikes)
        long_call_strike = self._find_closest_strike(short_call_strike + wing_width, available_call_strikes)

        if long_put_strike is None or long_call_strike is None:
            return setups

        # Get option data
        short_put = self._get_option_data(puts, short_put_strike)
        long_put = self._get_option_data(puts, long_put_strike)
        short_call = self._get_option_data(calls, short_call_strike)
        long_call = self._get_option_data(calls, long_call_strike)

        if not all([short_put, long_put, short_call, long_call]):
            return setups

        # Calculate net credit
        net_credit = (
            short_put['bid'] - long_put['ask'] +
            short_call['bid'] - long_call['ask']
        )

        if net_credit <= 0:
            return setups

        # Build legs
        legs = [
            OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.BUY,
                strike=long_put_strike,
                expiration=expiration,
                quantity=1,
                bid=long_put.get('bid', 0),
                ask=long_put.get('ask', 0),
                delta=long_put.get('delta', 0),
                theta=long_put.get('theta', 0)
            ),
            OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.SELL,
                strike=short_put_strike,
                expiration=expiration,
                quantity=1,
                bid=short_put.get('bid', 0),
                ask=short_put.get('ask', 0),
                delta=short_put.get('delta', 0),
                theta=short_put.get('theta', 0)
            ),
            OptionLeg(
                option_type=OptionType.CALL,
                side=OptionSide.SELL,
                strike=short_call_strike,
                expiration=expiration,
                quantity=1,
                bid=short_call.get('bid', 0),
                ask=short_call.get('ask', 0),
                delta=short_call.get('delta', 0),
                theta=short_call.get('theta', 0)
            ),
            OptionLeg(
                option_type=OptionType.CALL,
                side=OptionSide.BUY,
                strike=long_call_strike,
                expiration=expiration,
                quantity=1,
                bid=long_call.get('bid', 0),
                ask=long_call.get('ask', 0),
                delta=long_call.get('delta', 0),
                theta=long_call.get('theta', 0)
            )
        ]

        # Calculate metrics
        put_width = short_put_strike - long_put_strike
        call_width = long_call_strike - short_call_strike
        max_width = max(put_width, call_width)

        max_profit = net_credit * 100
        max_loss = (max_width - net_credit) * 100

        pop = self._calculate_pop(
            context.current_price,
            short_put_strike,
            short_call_strike,
            context.implied_volatility,
            (expiration - date.today()).days
        )

        net_greeks = self.aggregate_greeks(legs)

        setup = StrategySetup(
            symbol=chain.symbol,
            strategy_name=self.name,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_prices=[short_put_strike - net_credit, short_call_strike + net_credit],
            probability_of_profit=pop,
            net_delta=net_greeks['delta'],
            net_theta=net_greeks['theta'],
            underlying_price=context.current_price,
            iv_rank=context.iv_rank,
            notes=f"Variant: ${max_width:.0f} wings"
        )

        setups.append(setup)
        return setups

    def entry_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> EntrySignal:
        """Check if entry conditions are met for an Iron Condor"""
        reasons = []
        warnings = []

        # Check IV rank
        if context.iv_rank < self.min_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} below minimum {self.min_iv_rank}"]
            )
        if context.iv_rank >= 60:
            reasons.append(f"Excellent IV rank: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        elif context.iv_rank >= 50:
            reasons.append(f"Good IV rank: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE
        else:
            reasons.append(f"Acceptable IV rank: {context.iv_rank:.1f}")
            strength = SignalStrength.WEAK

        # Check DTE
        dte = setup.days_to_expiration
        if dte < self.min_dte:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"DTE {dte} below minimum {self.min_dte}"]
            )
        if abs(dte - self.optimal_dte) <= 10:
            reasons.append(f"Optimal DTE: {dte}")
        else:
            warnings.append(f"DTE {dte} not optimal (target: {self.optimal_dte})")

        # Check earnings
        if context.days_to_earnings is not None and context.days_to_earnings < 14:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"Earnings in {context.days_to_earnings} days - avoid"]
            )

        # Check probability of profit
        if setup.probability_of_profit >= 0.70:
            reasons.append(f"High POP: {setup.probability_of_profit:.1%}")
        elif setup.probability_of_profit >= 0.60:
            reasons.append(f"Good POP: {setup.probability_of_profit:.1%}")
        else:
            warnings.append(f"Lower POP: {setup.probability_of_profit:.1%}")

        # Check risk/reward
        if setup.risk_reward_ratio <= 2.0:
            reasons.append(f"Favorable risk/reward: {setup.risk_reward_ratio:.2f}:1")
        else:
            warnings.append(f"High risk/reward: {setup.risk_reward_ratio:.2f}:1")

        # Check delta neutrality
        if abs(setup.net_delta) < 0.10:
            reasons.append(f"Delta neutral: {setup.net_delta:.3f}")
        else:
            warnings.append(f"Net delta: {setup.net_delta:.3f}")

        # Validate setup
        is_valid, errors = self.validate_setup(setup)
        if not is_valid:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=errors
            )

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings,
            suggested_position_size=1
        )

    def exit_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext,
        entry_price: float,
        current_price: float
    ) -> ExitSignal:
        """Check if exit conditions are met for an open Iron Condor"""

        # Calculate P&L
        pnl = entry_price - current_price  # Credit received - current debit to close
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        # 1. Profit target
        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close position - profit target reached",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct,
                notes=[f"Target {self.profit_target_pct:.0%} reached at {pnl_pct:.1%}"]
            )

        # 2. Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close position - stop loss triggered",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct,
                notes=[f"Loss {pnl_pct:.1%} exceeds stop {-self.stop_loss_pct:.0%}"]
            )

        # 3. DTE threshold
        dte = setup.days_to_expiration
        if dte <= self.dte_close_threshold:
            if pnl_pct > 0:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DTE_THRESHOLD,
                    urgency=SignalStrength.MODERATE,
                    suggested_action="Close profitable position or roll",
                    current_pnl=pnl * 100,
                    current_pnl_pct=pnl_pct,
                    notes=[f"DTE {dte} below threshold, position profitable"]
                )
            else:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DTE_THRESHOLD,
                    urgency=SignalStrength.STRONG,
                    suggested_action="Close losing position - gamma risk increasing",
                    current_pnl=pnl * 100,
                    current_pnl_pct=pnl_pct,
                    notes=[f"DTE {dte} below threshold, close to avoid assignment"]
                )

        # 4. Short strike breach
        price = context.current_price
        short_put_strike = min(leg.strike for leg in setup.legs if leg.is_put and leg.is_short)
        short_call_strike = max(leg.strike for leg in setup.legs if leg.is_call and leg.is_short)

        put_breach = price <= short_put_strike * (1 + self.short_strike_buffer)
        call_breach = price >= short_call_strike * (1 - self.short_strike_buffer)

        if put_breach:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="Close or roll put spread - strike breached",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct,
                notes=[f"Price ${price:.2f} near short put ${short_put_strike:.2f}"]
            )

        if call_breach:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="Close or roll call spread - strike breached",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct,
                notes=[f"Price ${price:.2f} near short call ${short_call_strike:.2f}"]
            )

        # 5. Check for upcoming earnings
        if context.days_to_earnings is not None and context.days_to_earnings < setup.days_to_expiration:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.EARNINGS,
                urgency=SignalStrength.MODERATE,
                suggested_action="Close before earnings - IV crush risk",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct,
                notes=[f"Earnings in {context.days_to_earnings} days"]
            )

        # No exit signal
        return ExitSignal(
            should_exit=False,
            reason=ExitReason.MANUAL,
            urgency=SignalStrength.NONE,
            current_pnl=pnl * 100,
            current_pnl_pct=pnl_pct,
            notes=[f"Position OK - P&L: {pnl_pct:.1%}, DTE: {dte}"]
        )

    def calculate_position_size(
        self,
        setup: StrategySetup,
        account_value: float,
        current_positions: List[StrategySetup]
    ) -> int:
        """Calculate optimal position size for Iron Condor"""

        # Max loss per contract
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0

        # Max risk per trade (2% of account)
        max_risk = account_value * self.max_loss_pct

        # Calculate contracts based on risk
        contracts = int(max_risk / max_loss)

        # Cap at 5% of portfolio
        max_allocation = account_value * self.max_position_size_pct
        contracts = min(contracts, int(max_allocation / max_loss))

        # Check existing positions for same symbol
        symbol_exposure = sum(
            pos.max_loss for pos in current_positions
            if pos.symbol == setup.symbol
        )

        # Limit total symbol exposure to 10%
        max_symbol_exposure = account_value * 0.10
        remaining_capacity = max_symbol_exposure - symbol_exposure

        if remaining_capacity <= 0:
            return 0

        contracts = min(contracts, int(remaining_capacity / max_loss))

        return max(contracts, 0)

    def score_setup(self, setup: StrategySetup, context: MarketContext) -> float:
        """Custom scoring for Iron Condor setups"""
        score = 0.0
        components = {}

        # POP (30%) - Iron Condors need high POP
        pop_score = setup.probability_of_profit * 100 * 0.30
        components['pop'] = pop_score
        score += pop_score

        # Risk/Reward (20%) - target 1:2 or better
        rr = setup.risk_reward_ratio
        if rr <= 1.5:
            rr_score = 20
        elif rr <= 2.5:
            rr_score = 15
        elif rr <= 3.5:
            rr_score = 10
        else:
            rr_score = 5
        components['risk_reward'] = rr_score
        score += rr_score

        # IV Rank (20%) - prefer high IV
        if context.iv_rank >= 70:
            iv_score = 20
        elif context.iv_rank >= 60:
            iv_score = 17
        elif context.iv_rank >= 50:
            iv_score = 14
        elif context.iv_rank >= 40:
            iv_score = 10
        else:
            iv_score = 5
        components['iv_rank'] = iv_score
        score += iv_score

        # DTE (15%) - prefer 40-50 days
        dte = setup.days_to_expiration
        dte_optimal = abs(dte - self.optimal_dte)
        if dte_optimal <= 5:
            dte_score = 15
        elif dte_optimal <= 10:
            dte_score = 12
        elif dte_optimal <= 15:
            dte_score = 9
        else:
            dte_score = 5
        components['dte'] = dte_score
        score += dte_score

        # Delta neutrality (10%) - prefer low net delta
        delta_score = max(0, (0.15 - abs(setup.net_delta)) / 0.15) * 10
        components['delta'] = delta_score
        score += delta_score

        # Profit zone width (5%) - wider is better
        if len(setup.breakeven_prices) >= 2:
            profit_zone = setup.breakeven_prices[1] - setup.breakeven_prices[0]
            zone_pct = profit_zone / context.current_price
            zone_score = min(zone_pct * 100, 5)
            components['profit_zone'] = zone_score
            score += zone_score

        setup.score = round(score, 2)
        setup.score_components = components

        return score

    def adjust_position(
        self,
        setup: StrategySetup,
        context: MarketContext,
        current_pnl: float
    ) -> Optional[StrategySetup]:
        """
        Recommend position adjustment for an Iron Condor.

        Adjustments:
        - Roll untested side closer for more credit
        - Roll threatened side out/away for defense
        - Convert to Iron Butterfly if price centered
        """
        price = context.current_price

        # Find short strikes
        short_put = min(leg.strike for leg in setup.legs if leg.is_put and leg.is_short)
        short_call = max(leg.strike for leg in setup.legs if leg.is_call and leg.is_short)

        # Calculate distance to short strikes
        put_distance_pct = (price - short_put) / price
        call_distance_pct = (short_call - price) / price

        # If price is very close to one side, suggest rolling
        if put_distance_pct < 0.03:  # Within 3% of short put
            return self._suggest_roll_adjustment(
                setup, context, 'put', 'defensive'
            )

        if call_distance_pct < 0.03:  # Within 3% of short call
            return self._suggest_roll_adjustment(
                setup, context, 'call', 'defensive'
            )

        # If price is centered and we have profit, suggest rolling untested side
        center = (short_put + short_call) / 2
        if abs(price - center) / price < 0.02:  # Within 2% of center
            if current_pnl > setup.max_profit * 0.3:  # At 30%+ profit
                # Roll both sides closer for more credit
                return self._suggest_roll_adjustment(
                    setup, context, 'both', 'aggressive'
                )

        return None

    def _suggest_roll_adjustment(
        self,
        setup: StrategySetup,
        context: MarketContext,
        side: str,
        adjustment_type: str
    ) -> Optional[StrategySetup]:
        """Create a suggested adjustment setup"""
        # This would create a new setup with adjusted strikes
        # For now, return None and log the suggestion
        logger.info(f"Adjustment suggested: Roll {side} side ({adjustment_type})")
        return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _find_closest_strike(self, target: float, strikes: np.ndarray) -> Optional[float]:
        """Find the closest available strike to target"""
        if len(strikes) == 0:
            return None
        idx = np.argmin(np.abs(strikes - target))
        return float(strikes[idx])

    def _get_option_data(self, chain: pd.DataFrame, strike: float) -> Optional[Dict]:
        """Get option data for a specific strike"""
        row = chain[chain['strike'] == strike]
        if row.empty:
            return None
        row = row.iloc[0]
        return {
            'strike': strike,
            'bid': row.get('bid', 0),
            'ask': row.get('ask', 0),
            'last': row.get('last', row.get('last_price', 0)),
            'iv': row.get('iv', row.get('implied_volatility', 0)),
            'delta': row.get('delta', 0),
            'gamma': row.get('gamma', 0),
            'theta': row.get('theta', 0),
            'vega': row.get('vega', 0)
        }

    def _calculate_pop(
        self,
        underlying: float,
        short_put: float,
        short_call: float,
        iv: float,
        dte: int
    ) -> float:
        """
        Calculate probability of profit using Black-Scholes distribution.

        POP = P(short_put < S_T < short_call) at expiration
        """
        if iv <= 0 or dte <= 0:
            return 0.5

        t = dte / 365

        # Calculate d2 for each boundary
        d2_put = (np.log(underlying / short_put) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))
        d2_call = (np.log(underlying / short_call) + (-0.5 * iv**2) * t) / (iv * np.sqrt(t))

        # P(S_T > short_put) - P(S_T > short_call)
        prob_above_put = stats.norm.cdf(d2_put)
        prob_above_call = stats.norm.cdf(d2_call)

        pop = prob_above_put - prob_above_call

        return max(0, min(1, pop))

    def _calculate_prob_max_profit(
        self,
        underlying: float,
        short_put: float,
        short_call: float,
        iv: float,
        dte: int
    ) -> float:
        """Calculate probability of achieving max profit (expiring between short strikes)"""
        # Max profit requires price to stay between short strikes at expiration
        return self._calculate_pop(underlying, short_put, short_call, iv, dte)

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def open_position(
        self,
        symbol: str,
        expiration: date,
        long_put_strike: float,
        short_put_strike: float,
        short_call_strike: float,
        long_call_strike: float,
        credit_received: float,
        quantity: int = 1
    ) -> IronCondorPosition:
        """Record a new Iron Condor position"""
        position = IronCondorPosition(
            symbol=symbol,
            entry_date=date.today(),
            expiration=expiration,
            long_put_strike=long_put_strike,
            short_put_strike=short_put_strike,
            short_call_strike=short_call_strike,
            long_call_strike=long_call_strike,
            entry_credit=credit_received,
            quantity=quantity
        )

        if symbol not in self.positions:
            self.positions[symbol] = []
        self.positions[symbol].append(position)

        logger.info(
            f"Opened Iron Condor on {symbol}: "
            f"{long_put_strike}P/{short_put_strike}P/{short_call_strike}C/{long_call_strike}C "
            f"for ${credit_received:.2f} credit x{quantity}"
        )

        return position

    def close_position(
        self,
        symbol: str,
        position_index: int,
        debit_paid: float
    ) -> float:
        """Close an Iron Condor position and calculate P&L"""
        if symbol not in self.positions or position_index >= len(self.positions[symbol]):
            raise ValueError(f"Position not found: {symbol}[{position_index}]")

        position = self.positions[symbol][position_index]
        position.is_closed = True
        position.close_date = date.today()

        # P&L = credit received - debit paid to close
        position.realized_pnl = (position.entry_credit - debit_paid) * 100 * position.quantity

        logger.info(
            f"Closed Iron Condor on {symbol}: "
            f"P&L ${position.realized_pnl:.2f} "
            f"({position.realized_pnl / position.max_profit * 100:.1f}% of max)"
        )

        return position.realized_pnl

    def get_open_positions(self, symbol: Optional[str] = None) -> List[IronCondorPosition]:
        """Get all open Iron Condor positions"""
        if symbol:
            return [p for p in self.positions.get(symbol, []) if not p.is_closed]

        all_positions = []
        for positions in self.positions.values():
            all_positions.extend([p for p in positions if not p.is_closed])
        return all_positions


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Iron Condor Strategy ===\n")

    # Create strategy instance
    strategy = IronCondorStrategy(
        target_short_delta=0.20,
        min_iv_rank=50.0,
        profit_target_pct=0.50
    )

    print(f"Strategy: {strategy.name}")
    print(f"Type: {strategy.strategy_type.value}")
    print(f"Min IV Rank: {strategy.min_iv_rank}")
    print(f"DTE Range: {strategy.min_dte}-{strategy.max_dte}")

    # Test position tracking
    print("\n2. Testing Position Tracking")
    position = strategy.open_position(
        symbol='SPY',
        expiration=date.today() + timedelta(days=45),
        long_put_strike=545,
        short_put_strike=550,
        short_call_strike=570,
        long_call_strike=575,
        credit_received=1.25,
        quantity=2
    )

    print(f"   Position opened: {position.symbol}")
    print(f"   Max profit: ${position.max_profit:.2f}")
    print(f"   Max loss: ${position.max_loss:.2f}")
    print(f"   Profit zone: ${position.short_put_strike} - ${position.short_call_strike}")
    print(f"   DTE: {position.days_to_expiration}")

    # Test POP calculation
    print("\n3. Testing Probability Calculations")
    pop = strategy._calculate_pop(
        underlying=560,
        short_put=550,
        short_call=570,
        iv=0.20,
        dte=45
    )
    print(f"   POP: {pop:.1%}")

    print("\n✅ Iron Condor strategy ready!")
