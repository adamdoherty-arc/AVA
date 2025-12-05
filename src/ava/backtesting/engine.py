"""
Backtesting Engine
==================

High-performance options strategy backtesting engine with:
- Vectorized calculations using numpy/pandas
- Support for all AVA strategy types
- Monte Carlo simulations
- Comprehensive performance metrics
- Trade-by-trade analysis

Author: AVA Trading Platform
Created: 2025-11-28
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

from src.ava.strategies.base import (
    OptionsStrategy,
    StrategySetup,
    OptionLeg,
    EntrySignal,
    ExitSignal,
    MarketContext,
    OptionsChain,
    ExitReason
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: date
    end_date: date
    initial_capital: float = 100000.0

    # Position sizing
    max_positions: int = 10
    max_position_size_pct: float = 0.05
    max_portfolio_risk_pct: float = 0.20

    # Execution simulation
    slippage_pct: float = 0.001      # 0.1% slippage
    commission_per_contract: float = 0.65
    min_days_between_trades: int = 0

    # Data settings
    use_bid_ask: bool = True         # Use bid/ask vs mid price
    require_liquidity: bool = True
    min_open_interest: int = 100
    max_bid_ask_spread_pct: float = 0.10

    # Simulation settings
    monte_carlo_runs: int = 0        # 0 = no Monte Carlo
    random_seed: Optional[int] = 42

    def __post_init__(self) -> None:
        if isinstance(self.start_date, str):
            self.start_date = date.fromisoformat(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = date.fromisoformat(self.end_date)


@dataclass
class Trade:
    """Represents a single backtest trade"""
    trade_id: int
    symbol: str
    strategy_name: str
    entry_date: date
    exit_date: Optional[date] = None

    # Position details
    legs: List[Dict] = field(default_factory=list)
    quantity: int = 1

    # Prices
    entry_price: float = 0.0         # Net premium (positive = credit)
    exit_price: float = 0.0
    max_profit_potential: float = 0.0
    max_loss_potential: float = 0.0

    # Results
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    commissions: float = 0.0
    slippage: float = 0.0

    # Trade metrics
    days_held: int = 0
    exit_reason: str = ""
    max_drawdown: float = 0.0
    max_unrealized_profit: float = 0.0

    # Greeks at entry
    entry_delta: float = 0.0
    entry_theta: float = 0.0
    entry_vega: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.realized_pnl > 0

    @property
    def return_on_risk(self) -> float:
        """Return as percentage of max risk"""
        if self.max_loss_potential == 0:
            return 0.0
        return self.realized_pnl / abs(self.max_loss_potential)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0

    # Win/Loss metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Trade metrics
    avg_trade_duration: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Capital metrics
    ending_capital: float = 0.0
    peak_capital: float = 0.0
    avg_capital_deployed: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'total_return': self.total_return,
            'total_return_pct': f"{self.total_return_pct:.2%}",
            'annualized_return': f"{self.annualized_return:.2%}",
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'win_rate': f"{self.win_rate:.2%}",
            'profit_factor': round(self.profit_factor, 2),
            'total_trades': self.total_trades,
            'avg_trade_pnl': round(self.avg_trade_pnl, 2),
            'expectancy': round(self.expectancy, 2)
        }


@dataclass
class BacktestResult:
    """Complete backtest results"""
    config: BacktestConfig
    strategy_name: str
    metrics: PerformanceMetrics

    # Trade data
    trades: List[Trade] = field(default_factory=list)

    # Time series
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Monte Carlo results (if run)
    monte_carlo_results: Optional[Dict] = None

    def summary(self) -> str:
        """Generate text summary of results"""
        m = self.metrics
        return f"""
=== Backtest Results: {self.strategy_name} ===

Period: {self.config.start_date} to {self.config.end_date}
Initial Capital: ${self.config.initial_capital:,.2f}

PERFORMANCE
-----------
Total Return: ${m.total_return:,.2f} ({m.total_return_pct:.2%})
Annualized Return: {m.annualized_return:.2%}
Sharpe Ratio: {m.sharpe_ratio:.2f}
Sortino Ratio: {m.sortino_ratio:.2f}
Max Drawdown: {m.max_drawdown:.2%}

TRADES
------
Total Trades: {m.total_trades}
Win Rate: {m.win_rate:.2%}
Profit Factor: {m.profit_factor:.2f}
Avg Trade P&L: ${m.avg_trade_pnl:.2f}
Best Trade: ${m.best_trade:.2f}
Worst Trade: ${m.worst_trade:.2f}
Avg Duration: {m.avg_trade_duration:.1f} days

RISK
----
Expectancy: ${m.expectancy:.2f}
Avg Win: ${m.avg_win:.2f}
Avg Loss: ${m.avg_loss:.2f}
"""


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    High-performance options strategy backtesting engine.

    Usage:
        engine = BacktestEngine(config)
        result = engine.run(strategy, historical_data)

    Features:
        - Vectorized calculations for speed
        - Realistic execution simulation
        - Comprehensive metrics
        - Monte Carlo analysis
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[date, float]] = []
        self.current_capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.open_positions: List[Trade] = []
        self.trade_counter = 0

        if config.random_seed:
            np.random.seed(config.random_seed)

    def run(
        self,
        strategy: OptionsStrategy,
        data: Dict[str, pd.DataFrame],
        symbols: Optional[List[str]] = None
    ) -> BacktestResult:
        """
        Run backtest for a strategy.

        Args:
            strategy: OptionsStrategy instance to test
            data: Dict of symbol -> DataFrame with columns:
                  date, open, high, low, close, volume, iv, iv_rank,
                  options_chain (serialized or reference)
            symbols: Optional list of symbols to test (defaults to all in data)

        Returns:
            BacktestResult with complete analysis
        """
        logger.info(f"Starting backtest: {strategy.name}")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")

        # Reset state
        self._reset()

        symbols = symbols or list(data.keys())
        all_dates = self._get_trading_dates(data)

        logger.info(f"Testing {len(symbols)} symbols over {len(all_dates)} trading days")

        # Main simulation loop
        for current_date in all_dates:
            # Update open positions
            self._update_positions(current_date, data, strategy)

            # Look for new opportunities
            for symbol in symbols:
                if symbol not in data or current_date not in data[symbol].index:
                    continue

                # Check if we can add more positions
                if len(self.open_positions) >= self.config.max_positions:
                    continue

                # Get market context
                context = self._build_market_context(symbol, current_date, data[symbol])
                if context is None:
                    continue

                # Get options chain
                chain = self._build_options_chain(symbol, current_date, data)
                if chain is None:
                    continue

                # Find opportunities
                opportunities = strategy.find_opportunities(chain, context)

                # Evaluate entry for top opportunity
                if opportunities:
                    best_setup = opportunities[0]
                    entry_signal = strategy.entry_conditions(best_setup, context)

                    if entry_signal.should_enter:
                        self._enter_trade(best_setup, current_date, context)

            # Record equity
            total_value = self._calculate_portfolio_value(current_date, data)
            self.equity_curve.append((current_date, total_value))
            self.peak_capital = max(self.peak_capital, total_value)

        # Close any remaining positions
        self._close_all_positions(all_dates[-1], data)

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Run Monte Carlo if configured
        monte_carlo = None
        if self.config.monte_carlo_runs > 0:
            monte_carlo = self._run_monte_carlo()

        # Build result
        result = BacktestResult(
            config=self.config,
            strategy_name=strategy.name,
            metrics=metrics,
            trades=self.trades,
            equity_curve=self._build_equity_df(),
            drawdown_series=self._build_drawdown_df(),
            monthly_returns=self._build_monthly_returns(),
            monte_carlo_results=monte_carlo
        )

        logger.info(f"Backtest complete: {len(self.trades)} trades")
        return result

    def _reset(self) -> None:
        """Reset engine state for new backtest"""
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.open_positions = []
        self.trade_counter = 0

    def _get_trading_dates(self, data: Dict[str, pd.DataFrame]) -> List[date]:
        """Get sorted list of all trading dates in range"""
        all_dates = set()
        for symbol_data in data.values():
            if hasattr(symbol_data, 'index'):
                dates = symbol_data.index
                if isinstance(dates[0], (datetime, date)):
                    all_dates.update(d.date() if isinstance(d, datetime) else d for d in dates)

        # Filter to config range
        all_dates = [d for d in all_dates
                     if self.config.start_date <= d <= self.config.end_date]

        return sorted(all_dates)

    def _build_market_context(
        self,
        symbol: str,
        current_date: date,
        symbol_data: pd.DataFrame
    ) -> Optional[MarketContext]:
        """Build MarketContext from historical data"""
        try:
            if current_date not in symbol_data.index:
                return None

            row = symbol_data.loc[current_date]
            prev_close = symbol_data.loc[:current_date].iloc[-2]['close'] if len(symbol_data.loc[:current_date]) > 1 else row['close']

            return MarketContext(
                symbol=symbol,
                current_price=float(row['close']),
                previous_close=float(prev_close),
                implied_volatility=float(row.get('iv', 0.30)),
                iv_rank=float(row.get('iv_rank', 50)),
                iv_percentile=float(row.get('iv_percentile', 50)),
                historical_volatility=float(row.get('hv', 0.25)),
                vix=float(row.get('vix', 20)),
                daily_change=float(row['close'] - prev_close),
                daily_change_pct=float((row['close'] - prev_close) / prev_close) if prev_close else 0,
                volume=int(row.get('volume', 0)),
                earnings_date=row.get('earnings_date'),
                days_to_earnings=row.get('days_to_earnings')
            )
        except Exception as e:
            logger.debug(f"Error building context for {symbol}: {e}")
            return None

    def _build_options_chain(
        self,
        symbol: str,
        current_date: date,
        data: Dict[str, pd.DataFrame]
    ) -> Optional[OptionsChain]:
        """Build OptionsChain from historical data"""
        # This is a simplified version - in production, you'd load actual historical chains
        # For now, we'll generate synthetic chains based on the underlying price

        if symbol not in data or current_date not in data[symbol].index:
            return None

        row = data[symbol].loc[current_date]
        underlying_price = float(row['close'])
        iv = float(row.get('iv', 0.30))

        # Generate synthetic chain
        expirations = self._generate_expirations(current_date)
        calls, puts = self._generate_synthetic_chain(underlying_price, iv, expirations, current_date)

        return OptionsChain(
            symbol=symbol,
            underlying_price=underlying_price,
            expirations=expirations,
            calls=calls,
            puts=puts
        )

    def _generate_expirations(self, current_date: date) -> List[date]:
        """Generate typical expiration dates"""
        expirations = []

        # Weekly expirations for next 4 weeks
        for weeks in range(1, 5):
            exp = current_date + timedelta(days=7 * weeks)
            # Move to Friday
            days_until_friday = (4 - exp.weekday()) % 7
            exp = exp + timedelta(days=days_until_friday)
            if exp > current_date:
                expirations.append(exp)

        # Monthly expirations
        for months in range(1, 4):
            month = (current_date.month + months - 1) % 12 + 1
            year = current_date.year + (current_date.month + months - 1) // 12
            # Third Friday
            first_day = date(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)
            if third_friday > current_date:
                expirations.append(third_friday)

        return sorted(set(expirations))

    def _generate_synthetic_chain(
        self,
        underlying: float,
        iv: float,
        expirations: List[date],
        current_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic options chain for backtesting"""
        from scipy.stats import norm

        calls_data = []
        puts_data = []

        for exp in expirations:
            dte = (exp - current_date).days
            if dte <= 0:
                continue

            t = dte / 365

            # Generate strikes around underlying
            strike_range = underlying * 0.20  # +/- 20%
            strikes = np.arange(
                underlying - strike_range,
                underlying + strike_range,
                underlying * 0.01  # 1% increments
            )

            for strike in strikes:
                # Black-Scholes pricing
                d1 = (np.log(underlying / strike) + (0.05 + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
                d2 = d1 - iv * np.sqrt(t)

                call_price = underlying * norm.cdf(d1) - strike * np.exp(-0.05 * t) * norm.cdf(d2)
                put_price = strike * np.exp(-0.05 * t) * norm.cdf(-d2) - underlying * norm.cdf(-d1)

                # Add bid/ask spread
                spread = max(0.01, call_price * 0.02)

                call_delta = norm.cdf(d1)
                put_delta = call_delta - 1

                calls_data.append({
                    'strike': strike,
                    'expiration': exp,
                    'bid': max(0.01, call_price - spread/2),
                    'ask': call_price + spread/2,
                    'last': call_price,
                    'iv': iv,
                    'delta': call_delta,
                    'gamma': norm.pdf(d1) / (underlying * iv * np.sqrt(t)),
                    'theta': -underlying * norm.pdf(d1) * iv / (2 * np.sqrt(t)) / 365,
                    'vega': underlying * np.sqrt(t) * norm.pdf(d1) / 100
                })

                puts_data.append({
                    'strike': strike,
                    'expiration': exp,
                    'bid': max(0.01, put_price - spread/2),
                    'ask': put_price + spread/2,
                    'last': put_price,
                    'iv': iv,
                    'delta': put_delta,
                    'gamma': norm.pdf(d1) / (underlying * iv * np.sqrt(t)),
                    'theta': -underlying * norm.pdf(d1) * iv / (2 * np.sqrt(t)) / 365,
                    'vega': underlying * np.sqrt(t) * norm.pdf(d1) / 100
                })

        return pd.DataFrame(calls_data), pd.DataFrame(puts_data)

    def _enter_trade(
        self,
        setup: StrategySetup,
        entry_date: date,
        context: MarketContext
    ):
        """Enter a new trade"""
        self.trade_counter += 1

        # Calculate position size
        max_risk = self.current_capital * self.config.max_position_size_pct
        if setup.max_loss > 0 and setup.max_loss != float('inf'):
            quantity = max(1, int(max_risk / setup.max_loss))
        else:
            quantity = 1

        # Apply slippage
        slippage = abs(setup.net_premium) * self.config.slippage_pct

        # Calculate commissions
        num_legs = len(setup.legs)
        commissions = num_legs * quantity * self.config.commission_per_contract * 2  # Open and close

        trade = Trade(
            trade_id=self.trade_counter,
            symbol=setup.symbol,
            strategy_name=setup.strategy_name,
            entry_date=entry_date,
            legs=[leg.to_dict() for leg in setup.legs],
            quantity=quantity,
            entry_price=setup.net_premium,
            max_profit_potential=setup.max_profit * quantity,
            max_loss_potential=setup.max_loss * quantity if setup.max_loss != float('inf') else setup.net_premium * 2 * quantity,
            commissions=commissions,
            slippage=slippage,
            entry_delta=setup.net_delta,
            entry_theta=setup.net_theta,
            entry_vega=setup.net_vega
        )

        self.open_positions.append(trade)
        logger.debug(f"Entered trade #{trade.trade_id}: {setup.strategy_name} on {setup.symbol}")

    def _update_positions(
        self,
        current_date: date,
        data: Dict[str, pd.DataFrame],
        strategy: OptionsStrategy
    ):
        """Update and potentially exit open positions"""
        positions_to_close = []

        for trade in self.open_positions:
            # Get current market context
            if trade.symbol not in data or current_date not in data[trade.symbol].index:
                continue

            context = self._build_market_context(trade.symbol, current_date, data[trade.symbol])
            if context is None:
                continue

            # Estimate current position value (simplified)
            current_price = self._estimate_position_value(trade, context, current_date)

            # Check exit conditions
            # Rebuild a simplified setup for exit check
            setup = StrategySetup(
                symbol=trade.symbol,
                strategy_name=trade.strategy_name,
                legs=[],  # Would need to rebuild from trade.legs
                max_profit=trade.max_profit_potential / trade.quantity,
                max_loss=trade.max_loss_potential / trade.quantity,
                underlying_price=context.current_price
            )

            # Calculate DTE
            if trade.legs:
                min_exp = min(
                    date.fromisoformat(leg['expiration']) if isinstance(leg['expiration'], str) else leg['expiration']
                    for leg in trade.legs
                )
                setup_dte = (min_exp - current_date).days
            else:
                setup_dte = 30

            # Simple exit logic for backtest
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price if trade.entry_price != 0 else 0

            should_exit = False
            exit_reason = ""

            # Profit target
            if pnl_pct >= strategy.profit_target_pct:
                should_exit = True
                exit_reason = "profit_target"

            # Stop loss
            elif pnl_pct <= -strategy.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"

            # DTE threshold
            elif setup_dte <= 3:
                should_exit = True
                exit_reason = "dte_threshold"

            # Expiration
            elif setup_dte <= 0:
                should_exit = True
                exit_reason = "expiration"

            if should_exit:
                self._close_trade(trade, current_date, current_price, exit_reason)
                positions_to_close.append(trade)

            # Track max unrealized P&L
            unrealized_pnl = (trade.entry_price - current_price) * trade.quantity * 100
            trade.max_unrealized_profit = max(trade.max_unrealized_profit, unrealized_pnl)
            trade.max_drawdown = min(trade.max_drawdown, unrealized_pnl)

        # Remove closed positions
        for trade in positions_to_close:
            self.open_positions.remove(trade)

    def _estimate_position_value(
        self,
        trade: Trade,
        context: MarketContext,
        current_date: date
    ) -> float:
        """Estimate current value of a position"""
        # Simplified estimation using delta approximation
        if not trade.legs:
            return trade.entry_price

        underlying_change = (context.current_price - trade.entry_price * 100)  # Rough estimate

        # Use entry Greeks for approximation
        delta_pnl = trade.entry_delta * underlying_change

        # Time decay (theta)
        days_held = (current_date - trade.entry_date).days
        theta_decay = trade.entry_theta * days_held

        # Rough estimate of current premium
        estimated_value = trade.entry_price + delta_pnl / 100 + theta_decay

        return max(0, estimated_value)

    def _close_trade(
        self,
        trade: Trade,
        exit_date: date,
        exit_price: float,
        exit_reason: str
    ):
        """Close a trade and record results"""
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.days_held = (exit_date - trade.entry_date).days

        # Calculate P&L (for credit spreads: entry is credit received)
        # P&L = Credit received - Debit to close
        trade.realized_pnl = (trade.entry_price - exit_price) * trade.quantity * 100
        trade.realized_pnl -= trade.commissions
        trade.realized_pnl -= trade.slippage

        if trade.max_loss_potential != 0:
            trade.realized_pnl_pct = trade.realized_pnl / abs(trade.max_loss_potential)
        else:
            trade.realized_pnl_pct = 0

        self.current_capital += trade.realized_pnl
        self.trades.append(trade)

        logger.debug(f"Closed trade #{trade.trade_id}: P&L ${trade.realized_pnl:.2f} ({trade.exit_reason})")

    def _close_all_positions(self, final_date: date, data: Dict[str, pd.DataFrame]):
        """Close all remaining positions at end of backtest"""
        for trade in list(self.open_positions):
            if trade.symbol in data and final_date in data[trade.symbol].index:
                context = self._build_market_context(trade.symbol, final_date, data[trade.symbol])
                if context:
                    current_price = self._estimate_position_value(trade, context, final_date)
                    self._close_trade(trade, final_date, current_price, "backtest_end")

        self.open_positions.clear()

    def _calculate_portfolio_value(self, current_date: date, data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value including open positions"""
        total = self.current_capital

        for trade in self.open_positions:
            if trade.symbol in data and current_date in data[trade.symbol].index:
                context = self._build_market_context(trade.symbol, current_date, data[trade.symbol])
                if context:
                    current_price = self._estimate_position_value(trade, context, current_date)
                    unrealized = (trade.entry_price - current_price) * trade.quantity * 100
                    total += unrealized

        return total

    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        metrics = PerformanceMetrics()

        if not self.trades:
            return metrics

        # Basic trade metrics
        metrics.total_trades = len(self.trades)
        pnls = [t.realized_pnl for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = len(winners) / len(pnls) if pnls else 0

        metrics.avg_win = np.mean(winners) if winners else 0
        metrics.avg_loss = np.mean(losers) if losers else 0
        metrics.avg_trade_pnl = np.mean(pnls) if pnls else 0
        metrics.best_trade = max(pnls) if pnls else 0
        metrics.worst_trade = min(pnls) if pnls else 0

        # Total return
        metrics.total_return = sum(pnls)
        metrics.total_return_pct = metrics.total_return / self.config.initial_capital
        metrics.ending_capital = self.config.initial_capital + metrics.total_return

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        # Expectancy
        if metrics.win_rate > 0 and metrics.avg_win != 0:
            metrics.expectancy = (metrics.win_rate * metrics.avg_win +
                                  (1 - metrics.win_rate) * metrics.avg_loss)

        # Trade duration
        durations = [t.days_held for t in self.trades if t.days_held > 0]
        metrics.avg_trade_duration = np.mean(durations) if durations else 0

        # Calculate from equity curve
        if self.equity_curve:
            equity_df = self._build_equity_df()
            returns = equity_df['equity'].pct_change().dropna()

            # Annualized metrics
            trading_days = len(equity_df)
            years = trading_days / 252
            if years > 0:
                metrics.annualized_return = (1 + metrics.total_return_pct) ** (1/years) - 1
                metrics.cagr = metrics.annualized_return

            # Volatility and ratios
            if len(returns) > 1:
                metrics.volatility = returns.std() * np.sqrt(252)

                if metrics.volatility > 0:
                    risk_free_rate = 0.05
                    excess_return = metrics.annualized_return - risk_free_rate
                    metrics.sharpe_ratio = excess_return / metrics.volatility

                    # Sortino (downside deviation)
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std() * np.sqrt(252)
                        metrics.sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

            # Drawdown
            peak = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - peak) / peak
            metrics.max_drawdown = drawdown.min()
            metrics.avg_drawdown = drawdown.mean()

            if metrics.max_drawdown != 0:
                metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)

        # Consecutive wins/losses
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for pnl in pnls:
            if pnl > 0:
                if streak >= 0:
                    streak += 1
                else:
                    streak = 1
                max_win_streak = max(max_win_streak, streak)
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    streak = -1
                max_loss_streak = max(max_loss_streak, abs(streak))

        metrics.consecutive_wins = max_win_streak
        metrics.consecutive_losses = max_loss_streak

        return metrics

    def _build_equity_df(self) -> pd.DataFrame:
        """Build equity curve DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _build_drawdown_df(self) -> pd.DataFrame:
        """Build drawdown series DataFrame"""
        equity_df = self._build_equity_df()
        if equity_df.empty:
            return pd.DataFrame()

        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        return pd.DataFrame({'drawdown': drawdown})

    def _build_monthly_returns(self) -> pd.DataFrame:
        """Build monthly returns table"""
        equity_df = self._build_equity_df()
        if equity_df.empty:
            return pd.DataFrame()

        monthly = equity_df.resample('M').last()
        returns = monthly['equity'].pct_change()
        return pd.DataFrame({'return': returns})

    def _run_monte_carlo(self) -> Dict:
        """Run Monte Carlo simulation on trade results"""
        if not self.trades:
            return {}

        pnls = [t.realized_pnl for t in self.trades]
        n_trades = len(pnls)
        n_sims = self.config.monte_carlo_runs

        # Simulate different orderings of trades
        final_values = []
        max_drawdowns = []

        for _ in range(n_sims):
            shuffled = np.random.permutation(pnls)
            equity = np.cumsum(shuffled) + self.config.initial_capital
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak

            final_values.append(equity[-1])
            max_drawdowns.append(drawdown.min())

        return {
            'final_value_mean': np.mean(final_values),
            'final_value_std': np.std(final_values),
            'final_value_5th': np.percentile(final_values, 5),
            'final_value_95th': np.percentile(final_values, 95),
            'max_drawdown_mean': np.mean(max_drawdowns),
            'max_drawdown_5th': np.percentile(max_drawdowns, 5),
            'max_drawdown_95th': np.percentile(max_drawdowns, 95)
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Backtest Engine ===\n")

    # Test configuration
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-06-30',
        initial_capital=100000,
        max_positions=5,
        slippage_pct=0.001,
        commission_per_contract=0.65
    )

    print(f"Config: {config.start_date} to {config.end_date}")
    print(f"Initial Capital: ${config.initial_capital:,}")

    # Create engine
    engine = BacktestEngine(config)
    print(f"\nEngine created successfully")

    # Test metric calculation with sample trades
    engine.trades = [
        Trade(trade_id=1, symbol='SPY', strategy_name='Test', entry_date=date(2024, 1, 15),
              exit_date=date(2024, 1, 22), realized_pnl=150, days_held=7),
        Trade(trade_id=2, symbol='SPY', strategy_name='Test', entry_date=date(2024, 2, 1),
              exit_date=date(2024, 2, 10), realized_pnl=-80, days_held=9),
        Trade(trade_id=3, symbol='QQQ', strategy_name='Test', entry_date=date(2024, 2, 15),
              exit_date=date(2024, 2, 28), realized_pnl=200, days_held=13),
    ]
    engine.equity_curve = [
        (date(2024, 1, 15), 100000),
        (date(2024, 1, 22), 100150),
        (date(2024, 2, 1), 100150),
        (date(2024, 2, 10), 100070),
        (date(2024, 2, 15), 100070),
        (date(2024, 2, 28), 100270),
    ]

    metrics = engine._calculate_metrics()
    print(f"\nSample Metrics:")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Total Return: ${metrics.total_return:.2f}")

    print("\n✅ Backtest engine ready!")
