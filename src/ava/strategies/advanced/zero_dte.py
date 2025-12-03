"""
0DTE (Zero Days to Expiration) Strategies
=========================================

Strategies designed for same-day expiration options trading.

Key Characteristics:
- Maximum theta decay (options expire worthless by EOD)
- High gamma risk (rapid delta changes)
- Requires active management
- Best for liquid underlyings (SPY, QQQ, major indices)

Strategies:
1. 0DTE Iron Condor: Neutral strategy, sell OTM puts and calls
2. 0DTE Credit Spreads: Directional credit spreads for rapid decay
3. Gamma Scalping: Delta-neutral strategy profiting from gamma

WARNING: 0DTE trading is HIGH RISK and requires:
- Real-time monitoring
- Strict position sizing
- Fast execution capabilities
- Understanding of gamma risk

Author: AVA Trading Platform
Created: 2025-11-28
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
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
# 0DTE BASE CLASS
# =============================================================================

class ZeroDTEStrategy(OptionsStrategy):
    """Base class for 0DTE strategies with common utilities"""

    # 0DTE specific parameters
    supported_symbols: List[str] = ['SPY', 'QQQ', 'SPX', 'IWM', 'DIA', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'AMZN']
    min_dte: int = 0
    max_dte: int = 0  # 0DTE only

    # Time-based parameters
    entry_start_time: time = time(9, 45)   # Don't enter at open
    entry_end_time: time = time(14, 30)    # Don't enter too late
    force_close_time: time = time(15, 45)  # Force close before close

    # Tighter risk management for 0DTE
    profit_target_pct: float = 0.30        # Take profits faster
    stop_loss_pct: float = 1.0             # Tighter stops
    max_loss_per_trade_pct: float = 0.005  # 0.5% max per trade

    def _is_trading_time(self) -> Tuple[bool, str]:
        """Check if current time is appropriate for 0DTE trading"""
        now = datetime.now().time()

        if now < self.entry_start_time:
            return False, f"Too early - wait until {self.entry_start_time.strftime('%H:%M')}"

        if now > self.entry_end_time:
            return False, f"Too late for new entries - after {self.entry_end_time.strftime('%H:%M')}"

        return True, "Within trading window"

    def _should_force_close(self) -> bool:
        """Check if position should be force closed"""
        return datetime.now().time() >= self.force_close_time

    def _is_supported_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported for 0DTE"""
        return symbol.upper() in self.supported_symbols

    def _get_0dte_expiration(self, chain: OptionsChain) -> Optional[date]:
        """Get today's expiration from chain"""
        today = date.today()
        if today in chain.expirations:
            return today
        return None

    def _calculate_expected_move(
        self,
        underlying: float,
        iv: float,
        hours_remaining: float
    ) -> float:
        """Calculate expected move for remaining trading time"""
        if hours_remaining <= 0:
            return 0

        # Convert to years
        t = hours_remaining / (252 * 6.5)  # Trading hours per year
        return underlying * iv * np.sqrt(t)


# =============================================================================
# 0DTE IRON CONDOR
# =============================================================================

@register_strategy
class ZeroDTEIronCondorStrategy(ZeroDTEStrategy):
    """
    0DTE Iron Condor Strategy

    Structure:
    - Sell OTM put spread
    - Sell OTM call spread
    - Same day expiration

    Key Differences from Standard Iron Condor:
    - Much wider strikes (further OTM due to high gamma)
    - Tighter profit targets (30-50% vs 50%)
    - Strict time-based management
    - Must close by 3:45 PM

    Best Conditions:
    - Low VIX environment (< 20)
    - No major events (FOMC, earnings)
    - High liquidity underlying (SPY, QQQ)
    """

    name: str = "0DTE Iron Condor"
    description: str = "Same-day expiration iron condor for rapid theta decay"
    strategy_type: StrategyType = StrategyType.ADVANCED

    # Strike selection - wider than normal IC
    short_put_delta: float = 0.10   # ~90% POP
    short_call_delta: float = 0.10
    wing_width: float = 5.0         # $5 wings for SPY

    # Entry criteria
    max_vix: float = 25.0           # Avoid high volatility
    min_credit_pct: float = 0.15    # Min 15% of width

    # Risk management
    profit_target_pct: float = 0.40
    stop_loss_pct: float = 1.5      # 150% of credit
    delta_adjustment_threshold: float = 0.30

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find 0DTE Iron Condor opportunities"""
        opportunities = []

        # Check if symbol supported
        if not self._is_supported_symbol(chain.symbol):
            logger.debug(f"{chain.symbol} not supported for 0DTE")
            return opportunities

        # Check trading time
        can_trade, reason = self._is_trading_time()
        if not can_trade:
            logger.debug(reason)
            return opportunities

        # Get 0DTE expiration
        expiration = self._get_0dte_expiration(chain)
        if not expiration:
            logger.debug("No 0DTE expiration available")
            return opportunities

        # Check VIX
        if context.vix > self.max_vix:
            logger.debug(f"VIX {context.vix:.1f} too high for 0DTE")
            return opportunities

        calls, puts = chain.get_chain_for_expiration(expiration)
        if calls.empty or puts.empty:
            return opportunities

        # Find short strikes by delta
        short_put_strike = chain.get_strike_by_delta(expiration, self.short_put_delta, OptionType.PUT)
        short_call_strike = chain.get_strike_by_delta(expiration, self.short_call_delta, OptionType.CALL)

        if not short_put_strike or not short_call_strike:
            return opportunities

        # Calculate wing strikes
        long_put_strike = short_put_strike - self.wing_width
        long_call_strike = short_call_strike + self.wing_width

        # Get option data
        sp_data = puts[puts['strike'] == short_put_strike]
        lp_data = puts[puts['strike'] == long_put_strike]
        sc_data = calls[calls['strike'] == short_call_strike]
        lc_data = calls[calls['strike'] == long_call_strike]

        # Find closest available strikes if exact not found
        if lp_data.empty:
            available = puts['strike'].unique()
            closest = available[np.argmin(np.abs(available - long_put_strike))]
            lp_data = puts[puts['strike'] == closest]
            long_put_strike = closest

        if lc_data.empty:
            available = calls['strike'].unique()
            closest = available[np.argmin(np.abs(available - long_call_strike))]
            lc_data = calls[calls['strike'] == closest]
            long_call_strike = closest

        if any(df.empty for df in [sp_data, lp_data, sc_data, lc_data]):
            return opportunities

        sp = sp_data.iloc[0]
        lp = lp_data.iloc[0]
        sc = sc_data.iloc[0]
        lc = lc_data.iloc[0]

        # Calculate credit
        put_spread_credit = sp.get('bid', 0) - lp.get('ask', 0)
        call_spread_credit = sc.get('bid', 0) - lc.get('ask', 0)
        total_credit = put_spread_credit + call_spread_credit

        if total_credit <= 0:
            return opportunities

        # Calculate width and check minimum credit
        put_width = short_put_strike - long_put_strike
        call_width = long_call_strike - short_call_strike
        max_width = max(put_width, call_width)

        credit_pct = total_credit / max_width
        if credit_pct < self.min_credit_pct:
            return opportunities

        # Build legs
        legs = [
            OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.BUY,
                strike=long_put_strike,
                expiration=expiration,
                quantity=1,
                bid=lp.get('bid', 0),
                ask=lp.get('ask', 0),
                delta=lp.get('delta', 0)
            ),
            OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.SELL,
                strike=short_put_strike,
                expiration=expiration,
                quantity=1,
                bid=sp.get('bid', 0),
                ask=sp.get('ask', 0),
                delta=sp.get('delta', -self.short_put_delta)
            ),
            OptionLeg(
                option_type=OptionType.CALL,
                side=OptionSide.SELL,
                strike=short_call_strike,
                expiration=expiration,
                quantity=1,
                bid=sc.get('bid', 0),
                ask=sc.get('ask', 0),
                delta=sc.get('delta', self.short_call_delta)
            ),
            OptionLeg(
                option_type=OptionType.CALL,
                side=OptionSide.BUY,
                strike=long_call_strike,
                expiration=expiration,
                quantity=1,
                bid=lc.get('bid', 0),
                ask=lc.get('ask', 0),
                delta=lc.get('delta', 0)
            )
        ]

        max_profit = total_credit * 100
        max_loss = (max_width - total_credit) * 100

        # Calculate hours remaining
        now = datetime.now()
        market_close = datetime.combine(date.today(), time(16, 0))
        hours_remaining = (market_close - now).seconds / 3600

        # Expected move
        expected_move = self._calculate_expected_move(
            context.current_price,
            context.implied_volatility,
            hours_remaining
        )

        # Probability of profit (simplified)
        profit_zone_width = short_call_strike - short_put_strike
        pop = min(0.90, profit_zone_width / (2 * expected_move)) if expected_move > 0 else 0.70

        net_greeks = self.aggregate_greeks(legs)

        setup = StrategySetup(
            symbol=chain.symbol,
            strategy_name=self.name,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_prices=[short_put_strike - total_credit, short_call_strike + total_credit],
            probability_of_profit=pop,
            net_delta=net_greeks['delta'],
            net_theta=net_greeks['theta'],
            net_gamma=net_greeks['gamma'],
            underlying_price=context.current_price,
            iv_rank=context.iv_rank,
            notes=f"0DTE: ${short_put_strike}P/${short_call_strike}C, Credit ${total_credit:.2f}, {hours_remaining:.1f}h remaining"
        )

        self.score_setup(setup, context)
        opportunities.append(setup)

        return opportunities

    def entry_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> EntrySignal:
        """Entry conditions for 0DTE Iron Condor"""
        reasons = []
        warnings = ["WARNING: 0DTE HIGH RISK - Requires active monitoring"]

        # Check trading time
        can_trade, time_reason = self._is_trading_time()
        if not can_trade:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[time_reason]
            )

        # Check VIX
        if context.vix > self.max_vix:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"VIX {context.vix:.1f} too high (max {self.max_vix})"]
            )

        if context.vix < 15:
            reasons.append(f"Low VIX environment: {context.vix:.1f}")
            strength = SignalStrength.MODERATE
        else:
            reasons.append(f"VIX acceptable: {context.vix:.1f}")
            strength = SignalStrength.WEAK

        if setup.probability_of_profit >= 0.80:
            reasons.append(f"High POP: {setup.probability_of_profit:.1%}")
        elif setup.probability_of_profit >= 0.70:
            reasons.append(f"Good POP: {setup.probability_of_profit:.1%}")

        # Delta neutrality important for 0DTE
        if abs(setup.net_delta) < 0.05:
            reasons.append(f"Delta neutral: {setup.net_delta:.3f}")
        else:
            warnings.append(f"Directional bias: delta {setup.net_delta:.3f}")

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
        """Exit conditions for 0DTE Iron Condor"""

        pnl = entry_price - current_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        # Force close near market close
        if self._should_force_close():
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.EXPIRATION,
                urgency=SignalStrength.STRONG,
                suggested_action="FORCE CLOSE - Market close approaching",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct,
                notes=["Must close 0DTE before expiration"]
            )

        # Profit target
        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close - profit target on 0DTE",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="CLOSE NOW - stop loss on 0DTE",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        # Short strike breach
        short_put = max(leg.strike for leg in setup.legs if leg.is_put and leg.is_short)
        short_call = min(leg.strike for leg in setup.legs if leg.is_call and leg.is_short)

        if context.current_price <= short_put:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="CLOSE - Price breached short put",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if context.current_price >= short_call:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.DELTA_BREACH,
                urgency=SignalStrength.STRONG,
                suggested_action="CLOSE - Price breached short call",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        return ExitSignal(
            should_exit=False,
            reason=ExitReason.MANUAL,
            urgency=SignalStrength.NONE,
            current_pnl=pnl * 100,
            current_pnl_pct=pnl_pct,
            notes=[f"P&L: {pnl_pct:.1%}, Monitoring for {self.force_close_time.strftime('%H:%M')} cutoff"]
        )

    def calculate_position_size(
        self,
        setup: StrategySetup,
        account_value: float,
        current_positions: List[StrategySetup]
    ) -> int:
        """Very conservative position sizing for 0DTE"""
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0

        # Much smaller position for 0DTE - 0.5% max risk
        max_risk = account_value * self.max_loss_per_trade_pct
        contracts = int(max_risk / max_loss)

        # Cap at 5 contracts for 0DTE
        return min(contracts, 5)


# =============================================================================
# 0DTE CREDIT SPREAD
# =============================================================================

@register_strategy
class ZeroDTECreditSpreadStrategy(ZeroDTEStrategy):
    """
    0DTE Credit Spread Strategy

    Structure:
    - Single credit spread (bull put OR bear call)
    - Directional bias based on market outlook

    Use Cases:
    - Bullish: Sell put spread below support
    - Bearish: Sell call spread above resistance
    - Rapid theta decay throughout the day
    """

    name: str = "0DTE Credit Spread"
    description: str = "Same-day directional credit spread"
    strategy_type: StrategyType = StrategyType.ADVANCED

    short_delta: float = 0.15
    spread_width: float = 5.0

    min_credit_pct: float = 0.20
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 1.0

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find 0DTE credit spread opportunities"""
        opportunities = []

        if not self._is_supported_symbol(chain.symbol):
            return opportunities

        can_trade, reason = self._is_trading_time()
        if not can_trade:
            return opportunities

        expiration = self._get_0dte_expiration(chain)
        if not expiration:
            return opportunities

        calls, puts = chain.get_chain_for_expiration(expiration)
        if calls.empty or puts.empty:
            return opportunities

        # Find both bull put and bear call spreads
        # Bull Put Spread
        bull_put = self._create_bull_put_spread(
            chain, context, expiration, puts
        )
        if bull_put:
            opportunities.append(bull_put)

        # Bear Call Spread
        bear_call = self._create_bear_call_spread(
            chain, context, expiration, calls
        )
        if bear_call:
            opportunities.append(bear_call)

        for setup in opportunities:
            self.score_setup(setup, context)

        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

    def _create_bull_put_spread(
        self,
        chain: OptionsChain,
        context: MarketContext,
        expiration: date,
        puts: pd.DataFrame
    ) -> Optional[StrategySetup]:
        """Create a 0DTE bull put spread"""
        short_strike = chain.get_strike_by_delta(expiration, self.short_delta, OptionType.PUT)
        if not short_strike:
            return None

        long_strike = short_strike - self.spread_width

        sp_data = puts[puts['strike'] == short_strike]
        lp_data = puts[puts['strike'] == long_strike]

        if sp_data.empty:
            return None

        if lp_data.empty:
            available = puts['strike'].unique()
            closest = available[np.argmin(np.abs(available - long_strike))]
            lp_data = puts[puts['strike'] == closest]
            long_strike = closest

        if lp_data.empty:
            return None

        sp = sp_data.iloc[0]
        lp = lp_data.iloc[0]

        credit = sp.get('bid', 0) - lp.get('ask', 0)
        if credit <= 0:
            return None

        width = short_strike - long_strike
        if credit / width < self.min_credit_pct:
            return None

        legs = [
            OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.BUY,
                strike=long_strike,
                expiration=expiration,
                quantity=1,
                bid=lp.get('bid', 0),
                ask=lp.get('ask', 0)
            ),
            OptionLeg(
                option_type=OptionType.PUT,
                side=OptionSide.SELL,
                strike=short_strike,
                expiration=expiration,
                quantity=1,
                bid=sp.get('bid', 0),
                ask=sp.get('ask', 0)
            )
        ]

        return StrategySetup(
            symbol=chain.symbol,
            strategy_name=f"{self.name} (Bull Put)",
            legs=legs,
            max_profit=credit * 100,
            max_loss=(width - credit) * 100,
            breakeven_prices=[short_strike - credit],
            probability_of_profit=0.75,
            net_delta=0.15,
            underlying_price=context.current_price,
            iv_rank=context.iv_rank,
            notes=f"0DTE Bull Put: ${long_strike}/${short_strike}P"
        )

    def _create_bear_call_spread(
        self,
        chain: OptionsChain,
        context: MarketContext,
        expiration: date,
        calls: pd.DataFrame
    ) -> Optional[StrategySetup]:
        """Create a 0DTE bear call spread"""
        short_strike = chain.get_strike_by_delta(expiration, self.short_delta, OptionType.CALL)
        if not short_strike:
            return None

        long_strike = short_strike + self.spread_width

        sc_data = calls[calls['strike'] == short_strike]
        lc_data = calls[calls['strike'] == long_strike]

        if sc_data.empty:
            return None

        if lc_data.empty:
            available = calls['strike'].unique()
            closest = available[np.argmin(np.abs(available - long_strike))]
            lc_data = calls[calls['strike'] == closest]
            long_strike = closest

        if lc_data.empty:
            return None

        sc = sc_data.iloc[0]
        lc = lc_data.iloc[0]

        credit = sc.get('bid', 0) - lc.get('ask', 0)
        if credit <= 0:
            return None

        width = long_strike - short_strike
        if credit / width < self.min_credit_pct:
            return None

        legs = [
            OptionLeg(
                option_type=OptionType.CALL,
                side=OptionSide.SELL,
                strike=short_strike,
                expiration=expiration,
                quantity=1,
                bid=sc.get('bid', 0),
                ask=sc.get('ask', 0)
            ),
            OptionLeg(
                option_type=OptionType.CALL,
                side=OptionSide.BUY,
                strike=long_strike,
                expiration=expiration,
                quantity=1,
                bid=lc.get('bid', 0),
                ask=lc.get('ask', 0)
            )
        ]

        return StrategySetup(
            symbol=chain.symbol,
            strategy_name=f"{self.name} (Bear Call)",
            legs=legs,
            max_profit=credit * 100,
            max_loss=(width - credit) * 100,
            breakeven_prices=[short_strike + credit],
            probability_of_profit=0.75,
            net_delta=-0.15,
            underlying_price=context.current_price,
            iv_rank=context.iv_rank,
            notes=f"0DTE Bear Call: ${short_strike}/${long_strike}C"
        )

    def entry_conditions(self, setup: StrategySetup, context: MarketContext) -> EntrySignal:
        reasons = []
        warnings = ["WARNING: 0DTE HIGH RISK"]

        can_trade, time_reason = self._is_trading_time()
        if not can_trade:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[time_reason]
            )

        if setup.probability_of_profit >= 0.75:
            reasons.append(f"High POP: {setup.probability_of_profit:.1%}")
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings
        )

    def exit_conditions(self, setup: StrategySetup, context: MarketContext, entry_price: float, current_price: float) -> ExitSignal:
        pnl = entry_price - current_price
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if self._should_force_close():
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.EXPIRATION,
                urgency=SignalStrength.STRONG,
                suggested_action="FORCE CLOSE",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

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
        max_risk = account_value * self.max_loss_per_trade_pct
        return min(int(max_risk / max_loss), 10)


# =============================================================================
# GAMMA SCALPING
# =============================================================================

@register_strategy
class GammaScalpingStrategy(ZeroDTEStrategy):
    """
    Gamma Scalping Strategy

    Structure:
    - Long ATM straddle (long gamma position)
    - Delta hedge with underlying to capture gamma profits

    How it Works:
    - Buy ATM straddle (long gamma)
    - As price moves, delta changes
    - Sell shares when price rises (delta becomes positive)
    - Buy shares when price falls (delta becomes negative)
    - Net result: Profit from price oscillation

    Requirements:
    - High intraday volatility
    - Active delta hedging capability
    - Real-time position monitoring
    """

    name: str = "Gamma Scalping"
    description: str = "Long gamma strategy with delta hedging"
    strategy_type: StrategyType = StrategyType.ADVANCED

    # Hedge parameters
    hedge_delta_threshold: float = 0.10  # Hedge when delta exceeds
    min_hedge_interval_seconds: int = 60  # Minimum time between hedges

    # Entry parameters
    min_iv_rank: float = 0.0
    max_iv_rank: float = 40.0  # Want cheap options

    profit_target_pct: float = 0.30
    stop_loss_pct: float = 0.40

    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """Find gamma scalping opportunities"""
        opportunities = []

        if not self._is_supported_symbol(chain.symbol):
            return opportunities

        can_trade, reason = self._is_trading_time()
        if not can_trade:
            return opportunities

        # For gamma scalping, can use 0-3 DTE
        valid_expirations = [
            exp for exp in chain.expirations
            if 0 <= (exp - date.today()).days <= 3
        ]

        if not valid_expirations:
            return opportunities

        # Want low IV to buy cheap gamma
        if context.iv_rank > self.max_iv_rank:
            return opportunities

        for expiration in valid_expirations:
            calls, puts = chain.get_chain_for_expiration(expiration)
            if calls.empty or puts.empty:
                continue

            atm_strike = chain.get_atm_strike(expiration)

            call_data = calls[calls['strike'] == atm_strike]
            put_data = puts[puts['strike'] == atm_strike]

            if call_data.empty or put_data.empty:
                continue

            call = call_data.iloc[0]
            put = put_data.iloc[0]

            # Cost of straddle
            straddle_cost = call.get('ask', 0) + put.get('ask', 0)
            if straddle_cost <= 0:
                continue

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
                    gamma=call.get('gamma', 0),
                    theta=call.get('theta', 0)
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
                    gamma=put.get('gamma', 0),
                    theta=put.get('theta', 0)
                )
            ]

            net_greeks = self.aggregate_greeks(legs)
            dte = (expiration - date.today()).days

            # High gamma is key
            if net_greeks['gamma'] < 0.05:
                continue

            max_loss = straddle_cost * 100

            setup = StrategySetup(
                symbol=chain.symbol,
                strategy_name=self.name,
                legs=legs,
                max_profit=float('inf'),  # Unlimited with hedging
                max_loss=max_loss,
                breakeven_prices=[atm_strike - straddle_cost, atm_strike + straddle_cost],
                probability_of_profit=0.40,  # Depends on volatility
                net_delta=net_greeks['delta'],
                net_gamma=net_greeks['gamma'],
                net_theta=net_greeks['theta'],
                underlying_price=context.current_price,
                iv_rank=context.iv_rank,
                notes=f"Gamma: {net_greeks['gamma']:.4f}, Theta: ${net_greeks['theta']:.2f}/day"
            )

            self.score_setup(setup, context)
            opportunities.append(setup)

        opportunities.sort(key=lambda x: x.net_gamma, reverse=True)
        return opportunities

    def entry_conditions(self, setup: StrategySetup, context: MarketContext) -> EntrySignal:
        reasons = []
        warnings = ["REQUIRES: Real-time delta hedging capability"]

        if context.iv_rank > self.max_iv_rank:
            return EntrySignal(
                should_enter=False,
                strength=SignalStrength.NONE,
                reasons=[f"IV rank {context.iv_rank:.1f} too high - options expensive"]
            )

        if context.iv_rank < 20:
            reasons.append(f"Low IV - cheap gamma: {context.iv_rank:.1f}")
            strength = SignalStrength.STRONG
        else:
            reasons.append(f"Moderate IV: {context.iv_rank:.1f}")
            strength = SignalStrength.MODERATE

        if setup.net_gamma > 0.10:
            reasons.append(f"High gamma exposure: {setup.net_gamma:.4f}")
        elif setup.net_gamma > 0.05:
            reasons.append(f"Good gamma: {setup.net_gamma:.4f}")

        warnings.append(f"Daily theta cost: ${abs(setup.net_theta):.2f}")

        return EntrySignal(
            should_enter=True,
            strength=strength,
            setup=setup,
            reasons=reasons,
            warnings=warnings
        )

    def exit_conditions(self, setup: StrategySetup, context: MarketContext, entry_price: float, current_price: float) -> ExitSignal:
        pnl = current_price - entry_price  # Debit strategy
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        if self._should_force_close() and setup.days_to_expiration == 0:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.EXPIRATION,
                urgency=SignalStrength.STRONG,
                suggested_action="Close gamma position",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct >= self.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=SignalStrength.STRONG,
                suggested_action="Close with profit",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=SignalStrength.STRONG,
                suggested_action="Close to limit loss",
                current_pnl=pnl * 100,
                current_pnl_pct=pnl_pct
            )

        return ExitSignal(
            should_exit=False,
            reason=ExitReason.MANUAL,
            urgency=SignalStrength.NONE,
            current_pnl=pnl * 100,
            current_pnl_pct=pnl_pct,
            notes=[f"Delta: {setup.net_delta:.3f} - hedge if |delta| > {self.hedge_delta_threshold}"]
        )

    def calculate_position_size(self, setup: StrategySetup, account_value: float, current_positions: List[StrategySetup]) -> int:
        max_loss = setup.max_loss
        if max_loss <= 0:
            return 0
        # Conservative for gamma scalping
        max_risk = account_value * 0.01
        return min(int(max_risk / max_loss), 5)

    def calculate_hedge_shares(
        self,
        net_delta: float,
        contracts: int
    ) -> int:
        """Calculate shares needed to delta hedge"""
        # Each contract = 100 shares
        total_delta = net_delta * contracts * 100

        # Hedge to neutralize
        hedge_shares = -int(round(total_delta))
        return hedge_shares


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing 0DTE Strategies ===\n")

    print("1. 0DTE Iron Condor")
    ic = ZeroDTEIronCondorStrategy()
    print(f"   Name: {ic.name}")
    print(f"   Short delta: {ic.short_put_delta}")
    print(f"   Wing width: ${ic.wing_width}")
    print(f"   Profit target: {ic.profit_target_pct:.0%}")
    print(f"   Force close: {ic.force_close_time}")

    print("\n2. 0DTE Credit Spread")
    cs = ZeroDTECreditSpreadStrategy()
    print(f"   Name: {cs.name}")
    print(f"   Short delta: {cs.short_delta}")
    print(f"   Spread width: ${cs.spread_width}")

    print("\n3. Gamma Scalping")
    gs = GammaScalpingStrategy()
    print(f"   Name: {gs.name}")
    print(f"   Hedge threshold: {gs.hedge_delta_threshold}")
    print(f"   IV range: {gs.min_iv_rank}-{gs.max_iv_rank}")

    # Test time functions
    print("\n4. Testing Time Functions")
    can_trade, reason = ic._is_trading_time()
    print(f"   Can trade now: {can_trade}")
    print(f"   Reason: {reason}")
    print(f"   Should force close: {ic._should_force_close()}")

    # Test hedge calculation
    print("\n5. Testing Hedge Calculation")
    hedge_shares = gs.calculate_hedge_shares(0.15, 10)
    print(f"   Delta 0.15 x 10 contracts = {hedge_shares} shares to hedge")

    print("\n✅ 0DTE strategies ready!")
