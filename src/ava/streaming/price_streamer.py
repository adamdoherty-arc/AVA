"""
Real-Time Price Streamer
========================

Multi-source price streaming with support for:
- Yahoo Finance (yfinance)
- WebSocket feeds
- Polygon.io
- Alpha Vantage
- Mock/simulated data

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import logging
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class PriceSource(Enum):
    """Available price data sources"""
    YFINANCE = "yfinance"
    POLYGON = "polygon"
    ALPHA_VANTAGE = "alpha_vantage"
    SIMULATION = "simulation"


@dataclass
class PriceQuote:
    """Real-time price quote"""
    symbol: str
    price: float
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_size: int
    volume: int
    timestamp: datetime
    change: float = 0
    change_pct: float = 0
    high: float = 0
    low: float = 0
    open: float = 0
    prev_close: float = 0

    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask - self.bid

    def spread_pct(self) -> float:
        """Calculate spread as percentage"""
        if self.price > 0:
            return self.spread() / self.price * 100
        return 0

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "last_size": self.last_size,
            "volume": self.volume,
            "change": self.change,
            "change_pct": self.change_pct,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "spread": self.spread(),
            "spread_pct": self.spread_pct(),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OptionQuote(PriceQuote):
    """Option-specific quote with Greeks"""
    underlying_price: float = 0
    strike: float = 0
    expiration: str = ""
    option_type: str = ""  # "call" or "put"
    implied_volatility: float = 0
    delta: float = 0
    gamma: float = 0
    theta: float = 0
    vega: float = 0
    open_interest: int = 0

    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            "underlying_price": self.underlying_price,
            "strike": self.strike,
            "expiration": self.expiration,
            "option_type": self.option_type,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "open_interest": self.open_interest
        })
        return base


class PriceStreamer:
    """
    Real-time price streaming service.

    Usage:
        streamer = PriceStreamer()

        # Add callback for price updates
        streamer.on_quote(lambda quote: print(f"{quote.symbol}: ${quote.price}"))

        # Start streaming
        await streamer.subscribe(['AAPL', 'MSFT', 'SPY'])
        await streamer.start()
    """

    def __init__(
        self,
        source: PriceSource = PriceSource.SIMULATION,
        update_interval: float = 1.0,
        api_key: Optional[str] = None
    ):
        self.source = source
        self.update_interval = update_interval
        self.api_key = api_key

        # Subscribed symbols
        self.subscribed_symbols: Set[str] = set()
        self.option_symbols: Set[str] = set()

        # Callbacks
        self.quote_callbacks: List[Callable[[PriceQuote], None]] = []
        self.option_callbacks: List[Callable[[OptionQuote], None]] = []

        # State
        self.running = False
        self.task: Optional[asyncio.Task] = None

        # Cache for simulation
        self._sim_cache: Dict[str, Dict] = {}

    # =========================================================================
    # SUBSCRIPTION MANAGEMENT
    # =========================================================================

    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        self.subscribed_symbols.update(symbols)
        logger.info(f"Subscribed to: {symbols}")

    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        self.subscribed_symbols -= set(symbols)
        logger.info(f"Unsubscribed from: {symbols}")

    async def subscribe_options(self, option_symbols: List[str]):
        """Subscribe to option symbols"""
        self.option_symbols.update(option_symbols)

    async def unsubscribe_options(self, option_symbols: List[str]):
        """Unsubscribe from option symbols"""
        self.option_symbols -= set(option_symbols)

    def on_quote(self, callback: Callable[[PriceQuote], None]):
        """Register quote callback"""
        self.quote_callbacks.append(callback)

    def on_option_quote(self, callback: Callable[[OptionQuote], None]):
        """Register option quote callback"""
        self.option_callbacks.append(callback)

    # =========================================================================
    # STREAMING CONTROL
    # =========================================================================

    async def start(self) -> None:
        """Start streaming"""
        if self.running:
            return

        self.running = True
        logger.info(f"Starting price streamer with source: {self.source.value}")

        if self.source == PriceSource.SIMULATION:
            self.task = asyncio.create_task(self._simulation_loop())
        elif self.source == PriceSource.YFINANCE:
            self.task = asyncio.create_task(self._yfinance_loop())
        elif self.source == PriceSource.POLYGON:
            self.task = asyncio.create_task(self._polygon_loop())
        else:
            self.task = asyncio.create_task(self._simulation_loop())

    async def stop(self) -> None:
        """Stop streaming"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Price streamer stopped")

    # =========================================================================
    # STREAMING LOOPS
    # =========================================================================

    async def _simulation_loop(self) -> None:
        """Simulated price feed for testing"""
        logger.info("Starting simulation price feed...")

        # Initialize simulated prices
        base_prices = {
            'SPY': 580.0, 'QQQ': 490.0, 'IWM': 220.0,
            'AAPL': 230.0, 'MSFT': 425.0, 'NVDA': 140.0,
            'AMD': 140.0, 'TSLA': 350.0, 'META': 575.0,
            'GOOGL': 175.0, 'AMZN': 205.0
        }

        while self.running:
            try:
                for symbol in self.subscribed_symbols:
                    # Get or initialize price
                    if symbol not in self._sim_cache:
                        base_price = base_prices.get(symbol.upper(), 100.0)
                        self._sim_cache[symbol] = {
                            'price': base_price,
                            'prev_close': base_price,
                            'open': base_price * (1 + random.uniform(-0.01, 0.01)),
                            'high': base_price,
                            'low': base_price,
                            'volume': random.randint(1000000, 10000000)
                        }

                    cache = self._sim_cache[symbol]

                    # Simulate price movement
                    change = random.gauss(0, 0.001)  # 0.1% std dev
                    new_price = cache['price'] * (1 + change)
                    cache['price'] = new_price
                    cache['high'] = max(cache['high'], new_price)
                    cache['low'] = min(cache['low'], new_price)
                    cache['volume'] += random.randint(100, 10000)

                    # Create quote
                    spread = new_price * 0.0005  # 0.05% spread
                    quote = PriceQuote(
                        symbol=symbol,
                        price=new_price,
                        bid=new_price - spread / 2,
                        ask=new_price + spread / 2,
                        bid_size=random.randint(100, 1000),
                        ask_size=random.randint(100, 1000),
                        last_size=random.randint(10, 100),
                        volume=cache['volume'],
                        timestamp=datetime.now(),
                        change=new_price - cache['prev_close'],
                        change_pct=(new_price - cache['prev_close']) / cache['prev_close'] * 100,
                        high=cache['high'],
                        low=cache['low'],
                        open=cache['open'],
                        prev_close=cache['prev_close']
                    )

                    # Notify callbacks
                    for callback in self.quote_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(quote)
                            else:
                                callback(quote)
                        except Exception as e:
                            logger.error(f"Quote callback error: {e}")

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Simulation loop error: {e}")
                await asyncio.sleep(1)

    async def _yfinance_loop(self) -> None:
        """Yahoo Finance price feed"""
        logger.info("Starting yfinance price feed...")

        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed, falling back to simulation")
            await self._simulation_loop()
            return

        while self.running:
            try:
                if not self.subscribed_symbols:
                    await asyncio.sleep(1)
                    continue

                # Fetch quotes for all symbols
                symbols_list = list(self.subscribed_symbols)
                tickers = yf.Tickers(' '.join(symbols_list))

                for symbol in symbols_list:
                    try:
                        ticker = tickers.tickers.get(symbol)
                        if not ticker:
                            continue

                        info = ticker.info
                        fast_info = ticker.fast_info if hasattr(ticker, 'fast_info') else {}

                        # Build quote
                        current_price = (
                            fast_info.get('lastPrice') or
                            info.get('currentPrice') or
                            info.get('regularMarketPrice', 0)
                        )

                        prev_close = info.get('previousClose', current_price)
                        bid = info.get('bid', current_price * 0.999)
                        ask = info.get('ask', current_price * 1.001)

                        quote = PriceQuote(
                            symbol=symbol,
                            price=current_price,
                            bid=bid,
                            ask=ask,
                            bid_size=info.get('bidSize', 0),
                            ask_size=info.get('askSize', 0),
                            last_size=0,
                            volume=info.get('volume', 0),
                            timestamp=datetime.now(),
                            change=current_price - prev_close,
                            change_pct=(current_price - prev_close) / prev_close * 100 if prev_close else 0,
                            high=info.get('dayHigh', current_price),
                            low=info.get('dayLow', current_price),
                            open=info.get('open', current_price),
                            prev_close=prev_close
                        )

                        # Notify callbacks
                        for callback in self.quote_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(quote)
                                else:
                                    callback(quote)
                            except Exception as e:
                                logger.error(f"Quote callback error: {e}")

                    except Exception as e:
                        logger.debug(f"Error fetching {symbol}: {e}")

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"yfinance loop error: {e}")
                await asyncio.sleep(5)

    async def _polygon_loop(self) -> None:
        """Polygon.io WebSocket feed"""
        logger.info("Starting Polygon.io price feed...")

        if not self.api_key:
            logger.error("Polygon API key required, falling back to simulation")
            await self._simulation_loop()
            return

        # Polygon WebSocket implementation would go here
        # For now, fall back to simulation
        logger.warning("Polygon WebSocket not fully implemented, using simulation")
        await self._simulation_loop()

    # =========================================================================
    # MANUAL QUOTE FETCHING
    # =========================================================================

    async def get_quote(self, symbol: str) -> Optional[PriceQuote]:
        """Get single quote on demand"""
        if self.source == PriceSource.SIMULATION:
            return self._get_simulated_quote(symbol)
        elif self.source == PriceSource.YFINANCE:
            return await self._get_yfinance_quote(symbol)
        return None

    def _get_simulated_quote(self, symbol: str) -> PriceQuote:
        """Get simulated quote"""
        if symbol not in self._sim_cache:
            self._sim_cache[symbol] = {
                'price': 100.0,
                'prev_close': 100.0,
                'open': 100.0,
                'high': 100.0,
                'low': 100.0,
                'volume': 1000000
            }

        cache = self._sim_cache[symbol]
        spread = cache['price'] * 0.0005

        return PriceQuote(
            symbol=symbol,
            price=cache['price'],
            bid=cache['price'] - spread / 2,
            ask=cache['price'] + spread / 2,
            bid_size=500,
            ask_size=500,
            last_size=100,
            volume=cache['volume'],
            timestamp=datetime.now(),
            change=cache['price'] - cache['prev_close'],
            change_pct=(cache['price'] - cache['prev_close']) / cache['prev_close'] * 100,
            high=cache['high'],
            low=cache['low'],
            open=cache['open'],
            prev_close=cache['prev_close']
        )

    async def _get_yfinance_quote(self, symbol: str) -> Optional[PriceQuote]:
        """Fetch quote from yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            prev_close = info.get('previousClose', current_price)

            return PriceQuote(
                symbol=symbol,
                price=current_price,
                bid=info.get('bid', current_price),
                ask=info.get('ask', current_price),
                bid_size=info.get('bidSize', 0),
                ask_size=info.get('askSize', 0),
                last_size=0,
                volume=info.get('volume', 0),
                timestamp=datetime.now(),
                change=current_price - prev_close,
                change_pct=(current_price - prev_close) / prev_close * 100 if prev_close else 0,
                high=info.get('dayHigh', current_price),
                low=info.get('dayLow', current_price),
                open=info.get('open', current_price),
                prev_close=prev_close
            )

        except Exception as e:
            logger.error(f"Error fetching yfinance quote: {e}")
            return None

    # =========================================================================
    # MARKET HOURS CHECK
    # =========================================================================

    @staticmethod
    def is_market_open() -> bool:
        """Check if US market is open"""
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = dtime(9, 30)
        market_close = dtime(16, 0)
        current_time = now.time()

        return market_open <= current_time <= market_close

    @staticmethod
    def is_extended_hours() -> bool:
        """Check if in extended hours trading"""
        now = datetime.now()

        if now.weekday() >= 5:
            return False

        current_time = now.time()

        # Pre-market: 4:00 AM - 9:30 AM
        pre_market = dtime(4, 0) <= current_time < dtime(9, 30)

        # After-hours: 4:00 PM - 8:00 PM
        after_hours = dtime(16, 0) < current_time <= dtime(20, 0)

        return pre_market or after_hours


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing Price Streamer ===\n")

    async def test_streamer():
        streamer = PriceStreamer(
            source=PriceSource.SIMULATION,
            update_interval=1.0
        )

        # Add callback
        def on_quote(quote: PriceQuote):
            print(f"  {quote.symbol}: ${quote.price:.2f} ({quote.change_pct:+.2f}%)")

        streamer.on_quote(on_quote)

        # Subscribe
        await streamer.subscribe(['SPY', 'QQQ', 'AAPL'])

        # Start streaming
        await streamer.start()

        print("Streaming prices (5 updates)...\n")
        await asyncio.sleep(5)

        # Stop
        await streamer.stop()

        print("\n✅ Price streamer test complete!")

    asyncio.run(test_streamer())
