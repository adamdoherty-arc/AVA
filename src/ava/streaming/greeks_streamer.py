"""
Real-Time Greeks Streamer
=========================

Live Greeks calculation and streaming for option positions.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import math

logger = logging.getLogger(__name__)


@dataclass
class LiveGreeks:
    """Real-time Greeks for an option position"""
    symbol: str
    underlying_symbol: str
    underlying_price: float
    strike: float
    expiration: date
    option_type: str  # "call" or "put"
    days_to_expiration: int

    # First-order Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    # Second-order Greeks
    vanna: float = 0
    volga: float = 0
    charm: float = 0

    # Position Greeks (multiplied by quantity)
    position_delta: float = 0
    position_gamma: float = 0
    position_theta: float = 0
    position_vega: float = 0

    # Other metrics
    implied_volatility: float = 0
    theoretical_value: float = 0
    intrinsic_value: float = 0
    extrinsic_value: float = 0
    probability_itm: float = 0
    probability_otm: float = 0

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "underlying_symbol": self.underlying_symbol,
            "underlying_price": self.underlying_price,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "option_type": self.option_type,
            "days_to_expiration": self.days_to_expiration,
            "delta": round(self.delta, 4),
            "gamma": round(self.gamma, 4),
            "theta": round(self.theta, 4),
            "vega": round(self.vega, 4),
            "rho": round(self.rho, 4),
            "vanna": round(self.vanna, 4),
            "volga": round(self.volga, 4),
            "charm": round(self.charm, 4),
            "position_delta": round(self.position_delta, 2),
            "position_gamma": round(self.position_gamma, 4),
            "position_theta": round(self.position_theta, 2),
            "position_vega": round(self.position_vega, 2),
            "implied_volatility": round(self.implied_volatility, 4),
            "theoretical_value": round(self.theoretical_value, 2),
            "intrinsic_value": round(self.intrinsic_value, 2),
            "extrinsic_value": round(self.extrinsic_value, 2),
            "probability_itm": round(self.probability_itm, 4),
            "probability_otm": round(self.probability_otm, 4),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PortfolioGreeksSummary:
    """Aggregated Greeks for entire portfolio"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float

    # Dollar-weighted metrics
    beta_weighted_delta: float = 0  # SPY beta-weighted
    notional_exposure: float = 0

    # Risk metrics
    max_daily_theta_decay: float = 0
    gamma_risk_1pct: float = 0  # P&L impact of 1% move
    vega_risk_1pt: float = 0   # P&L impact of 1pt IV change

    # Counts
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "total_delta": round(self.total_delta, 2),
            "total_gamma": round(self.total_gamma, 4),
            "total_theta": round(self.total_theta, 2),
            "total_vega": round(self.total_vega, 2),
            "total_rho": round(self.total_rho, 2),
            "beta_weighted_delta": round(self.beta_weighted_delta, 2),
            "notional_exposure": round(self.notional_exposure, 0),
            "max_daily_theta_decay": round(self.max_daily_theta_decay, 2),
            "gamma_risk_1pct": round(self.gamma_risk_1pct, 2),
            "vega_risk_1pt": round(self.vega_risk_1pt, 2),
            "total_positions": self.total_positions,
            "long_positions": self.long_positions,
            "short_positions": self.short_positions,
            "timestamp": self.timestamp.isoformat()
        }


class GreeksCalculator:
    """
    Black-Scholes Greeks calculator with higher-order Greeks.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def calculate(
        self,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,  # In years
        volatility: float,
        option_type: str,
        quantity: int = 1
    ) -> Dict[str, float]:
        """Calculate all Greeks for an option"""

        if time_to_expiry <= 0:
            time_to_expiry = 1 / 365  # Minimum 1 day

        S = underlying_price
        K = strike
        T = time_to_expiry
        r = self.risk_free_rate
        sigma = volatility

        # Calculate d1 and d2
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        # Standard normal CDF and PDF
        n_d1 = self._norm_cdf(d1)
        n_d2 = self._norm_cdf(d2)
        n_d1_neg = self._norm_cdf(-d1)
        n_d2_neg = self._norm_cdf(-d2)
        phi_d1 = self._norm_pdf(d1)

        # Theoretical value
        if option_type.lower() == 'call':
            price = S * n_d1 - K * math.exp(-r * T) * n_d2
            delta = n_d1
            prob_itm = n_d2
        else:  # put
            price = K * math.exp(-r * T) * n_d2_neg - S * n_d1_neg
            delta = n_d1 - 1
            prob_itm = n_d2_neg

        # First-order Greeks
        gamma = phi_d1 / (S * sigma * math.sqrt(T))
        theta_raw = (-(S * phi_d1 * sigma) / (2 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * (n_d2 if option_type.lower() == 'call' else n_d2_neg))
        theta = theta_raw / 365  # Per day
        vega = S * phi_d1 * math.sqrt(T) / 100  # Per 1% IV
        rho = K * T * math.exp(-r * T) * (n_d2 if option_type.lower() == 'call' else -n_d2_neg) / 100

        # Second-order Greeks
        vanna = -phi_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma
        charm = -phi_d1 * (2 * r * T - d2 * sigma * math.sqrt(T)) / (2 * T * sigma * math.sqrt(T))

        # Intrinsic/extrinsic value
        if option_type.lower() == 'call':
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        extrinsic = price - intrinsic

        # Per-contract values (multiply by 100 for standard contracts)
        contract_multiplier = 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta * contract_multiplier,  # Per contract per day
            'vega': vega * contract_multiplier,
            'rho': rho * contract_multiplier,
            'vanna': vanna,
            'volga': volga,
            'charm': charm,
            'theoretical_value': price,
            'intrinsic_value': intrinsic,
            'extrinsic_value': extrinsic,
            'probability_itm': prob_itm,
            'probability_otm': 1 - prob_itm,
            # Position Greeks
            'position_delta': delta * quantity * contract_multiplier,
            'position_gamma': gamma * quantity * contract_multiplier,
            'position_theta': theta * contract_multiplier * quantity,
            'position_vega': vega * contract_multiplier * quantity
        }

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF"""
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


class GreeksStreamer:
    """
    Real-time Greeks streaming service.

    Usage:
        streamer = GreeksStreamer()
        streamer.on_greeks_update(lambda g: print(f"{g.symbol}: Delta={g.delta}"))

        # Add positions
        await streamer.add_position(position_data)

        # Start streaming
        await streamer.start()
    """

    def __init__(
        self,
        update_interval: float = 5.0,
        price_provider: Optional[Any] = None,
        risk_free_rate: float = 0.05
    ):
        self.update_interval = update_interval
        self.price_provider = price_provider
        self.calculator = GreeksCalculator(risk_free_rate)

        # Tracked positions
        self.positions: Dict[str, Dict] = {}

        # Callbacks
        self.greeks_callbacks: List[Callable[[LiveGreeks], None]] = []
        self.portfolio_callbacks: List[Callable[[PortfolioGreeksSummary], None]] = []

        # State
        self.running = False
        self.task: Optional[asyncio.Task] = None

        # Cache
        self._last_greeks: Dict[str, LiveGreeks] = {}
        self._underlying_prices: Dict[str, float] = {}

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def add_position(self, position: Dict):
        """Add or update a position to track"""
        position_id = position.get('id') or position.get('symbol')
        self.positions[position_id] = position
        logger.debug(f"Added position: {position_id}")

    def remove_position(self, position_id: str):
        """Remove a position from tracking"""
        if position_id in self.positions:
            del self.positions[position_id]
            if position_id in self._last_greeks:
                del self._last_greeks[position_id]

    def clear_positions(self):
        """Clear all tracked positions"""
        self.positions.clear()
        self._last_greeks.clear()

    def set_positions(self, positions: List[Dict]):
        """Set all positions at once"""
        self.positions = {
            p.get('id') or p.get('symbol'): p
            for p in positions
        }

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_greeks_update(self, callback: Callable[[LiveGreeks], None]):
        """Register callback for individual position Greeks updates"""
        self.greeks_callbacks.append(callback)

    def on_portfolio_update(self, callback: Callable[[PortfolioGreeksSummary], None]):
        """Register callback for portfolio Greeks updates"""
        self.portfolio_callbacks.append(callback)

    # =========================================================================
    # STREAMING
    # =========================================================================

    async def start(self):
        """Start Greeks streaming"""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._streaming_loop())
        logger.info("Greeks streamer started")

    async def stop(self):
        """Stop Greeks streaming"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Greeks streamer stopped")

    async def _streaming_loop(self):
        """Main streaming loop"""
        while self.running:
            try:
                all_greeks = []

                for position_id, position in self.positions.items():
                    try:
                        greeks = await self._calculate_position_greeks(position)
                        if greeks:
                            all_greeks.append(greeks)
                            self._last_greeks[position_id] = greeks

                            # Notify callbacks
                            for callback in self.greeks_callbacks:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(greeks)
                                    else:
                                        callback(greeks)
                                except Exception as e:
                                    logger.error(f"Greeks callback error: {e}")

                    except Exception as e:
                        logger.error(f"Error calculating Greeks for {position_id}: {e}")

                # Calculate and broadcast portfolio summary
                if all_greeks:
                    summary = self._calculate_portfolio_summary(all_greeks)
                    for callback in self.portfolio_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(summary)
                            else:
                                callback(summary)
                        except Exception as e:
                            logger.error(f"Portfolio callback error: {e}")

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Greeks streaming loop error: {e}")
                await asyncio.sleep(1)

    async def _calculate_position_greeks(self, position: Dict) -> Optional[LiveGreeks]:
        """Calculate Greeks for a single position"""

        # Extract position data
        symbol = position.get('symbol', '')
        underlying_symbol = position.get('underlying_symbol') or self._extract_underlying(symbol)
        strike = position.get('strike', 0)
        expiration = position.get('expiration')
        option_type = position.get('option_type', 'call')
        quantity = position.get('quantity', 1)
        iv = position.get('implied_volatility', 0.30)

        # Parse expiration
        if isinstance(expiration, str):
            try:
                expiration = date.fromisoformat(expiration)
            except ValueError:
                expiration = date.today()
        elif not expiration:
            expiration = date.today()

        # Calculate days to expiration
        dte = max(1, (expiration - date.today()).days)
        time_to_expiry = dte / 365

        # Get underlying price
        underlying_price = await self._get_underlying_price(underlying_symbol)
        if underlying_price <= 0:
            underlying_price = position.get('underlying_price', 100)

        # Calculate Greeks
        greeks_data = self.calculator.calculate(
            underlying_price=underlying_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=iv,
            option_type=option_type,
            quantity=quantity
        )

        return LiveGreeks(
            symbol=symbol,
            underlying_symbol=underlying_symbol,
            underlying_price=underlying_price,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            days_to_expiration=dte,
            delta=greeks_data['delta'],
            gamma=greeks_data['gamma'],
            theta=greeks_data['theta'],
            vega=greeks_data['vega'],
            rho=greeks_data['rho'],
            vanna=greeks_data['vanna'],
            volga=greeks_data['volga'],
            charm=greeks_data['charm'],
            position_delta=greeks_data['position_delta'],
            position_gamma=greeks_data['position_gamma'],
            position_theta=greeks_data['position_theta'],
            position_vega=greeks_data['position_vega'],
            implied_volatility=iv,
            theoretical_value=greeks_data['theoretical_value'],
            intrinsic_value=greeks_data['intrinsic_value'],
            extrinsic_value=greeks_data['extrinsic_value'],
            probability_itm=greeks_data['probability_itm'],
            probability_otm=greeks_data['probability_otm'],
            timestamp=datetime.now()
        )

    async def _get_underlying_price(self, symbol: str) -> float:
        """Get current underlying price"""
        # Check cache first
        if symbol in self._underlying_prices:
            return self._underlying_prices[symbol]

        # Try price provider
        if self.price_provider:
            try:
                if hasattr(self.price_provider, 'get_quote'):
                    quote = await self.price_provider.get_quote(symbol)
                    if quote:
                        self._underlying_prices[symbol] = quote.price
                        return quote.price
            except Exception as e:
                logger.debug(f"Error getting price for {symbol}: {e}")

        return 0

    def _extract_underlying(self, option_symbol: str) -> str:
        """Extract underlying symbol from option symbol"""
        # Simple extraction - take letters before numbers
        import re
        match = re.match(r'^([A-Z]+)', option_symbol.upper())
        return match.group(1) if match else option_symbol

    def _calculate_portfolio_summary(self, all_greeks: List[LiveGreeks]) -> PortfolioGreeksSummary:
        """Calculate aggregated portfolio Greeks"""

        total_delta = sum(g.position_delta for g in all_greeks)
        total_gamma = sum(g.position_gamma for g in all_greeks)
        total_theta = sum(g.position_theta for g in all_greeks)
        total_vega = sum(g.position_vega for g in all_greeks)
        total_rho = sum(g.rho for g in all_greeks)

        long_count = sum(1 for g in all_greeks if g.position_delta > 0)
        short_count = len(all_greeks) - long_count

        # Calculate notional exposure
        notional = sum(abs(g.position_delta * g.underlying_price) for g in all_greeks)

        # Gamma risk for 1% move
        gamma_risk = sum(
            0.5 * g.position_gamma * (g.underlying_price * 0.01) ** 2
            for g in all_greeks
        )

        # Vega risk for 1pt IV change
        vega_risk = total_vega

        return PortfolioGreeksSummary(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            total_rho=total_rho,
            beta_weighted_delta=total_delta,  # Would need beta for proper weighting
            notional_exposure=notional,
            max_daily_theta_decay=total_theta,
            gamma_risk_1pct=gamma_risk,
            vega_risk_1pt=vega_risk,
            total_positions=len(all_greeks),
            long_positions=long_count,
            short_positions=short_count,
            timestamp=datetime.now()
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def calculate_greeks(self, position: Dict) -> Dict[str, float]:
        """Calculate Greeks for a position (synchronous)"""

        strike = position.get('strike', 0)
        expiration = position.get('expiration')
        option_type = position.get('option_type', 'call')
        quantity = position.get('quantity', 1)
        iv = position.get('implied_volatility', 0.30)
        underlying_price = position.get('underlying_price', 100)

        if isinstance(expiration, str):
            try:
                expiration = date.fromisoformat(expiration)
            except ValueError:
                expiration = date.today()
        elif not expiration:
            expiration = date.today()

        dte = max(1, (expiration - date.today()).days)

        return self.calculator.calculate(
            underlying_price=underlying_price,
            strike=strike,
            time_to_expiry=dte / 365,
            volatility=iv,
            option_type=option_type,
            quantity=quantity
        )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing Greeks Streamer ===\n")

    async def test_greeks():
        streamer = GreeksStreamer(update_interval=2.0)

        # Add callback
        def on_greeks(greeks: LiveGreeks):
            print(f"  {greeks.symbol}:")
            print(f"    Delta: {greeks.delta:.4f} | Position Delta: {greeks.position_delta:.2f}")
            print(f"    Gamma: {greeks.gamma:.4f} | Theta: ${greeks.position_theta:.2f}/day")
            print(f"    Vega: ${greeks.position_vega:.2f} | IV: {greeks.implied_volatility:.1%}")

        def on_portfolio(summary: PortfolioGreeksSummary):
            print(f"\n  Portfolio Summary:")
            print(f"    Total Delta: {summary.total_delta:.2f}")
            print(f"    Total Theta: ${summary.total_theta:.2f}/day")
            print(f"    Total Vega: ${summary.total_vega:.2f}")

        streamer.on_greeks_update(on_greeks)
        streamer.on_portfolio_update(on_portfolio)

        # Add test positions
        streamer.add_position({
            'id': '1',
            'symbol': 'AAPL 230C 12/20',
            'underlying_symbol': 'AAPL',
            'underlying_price': 232.50,
            'strike': 230,
            'expiration': '2024-12-20',
            'option_type': 'call',
            'quantity': 5,
            'implied_volatility': 0.28
        })

        streamer.add_position({
            'id': '2',
            'symbol': 'SPY 580P 12/15',
            'underlying_symbol': 'SPY',
            'underlying_price': 585.00,
            'strike': 580,
            'expiration': '2024-12-15',
            'option_type': 'put',
            'quantity': -2,
            'implied_volatility': 0.15
        })

        # Start streaming
        await streamer.start()

        print("Streaming Greeks (2 updates)...\n")
        await asyncio.sleep(5)

        # Stop
        await streamer.stop()

        print("\n✅ Greeks streamer test complete!")

    asyncio.run(test_greeks())
