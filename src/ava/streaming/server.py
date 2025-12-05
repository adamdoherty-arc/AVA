"""
Real-Time Streaming Server
==========================

WebSocket-based streaming server for:
- Real-time positions updates
- Live Greeks calculations
- Price streaming
- Alert notifications
- Trade signals

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import weakref

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams"""
    POSITIONS = "positions"
    GREEKS = "greeks"
    PRICES = "prices"
    ALERTS = "alerts"
    SIGNALS = "signals"
    PORTFOLIO = "portfolio"
    MARKET = "market"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    TRADE = "trade"


@dataclass
class StreamMessage:
    """Message structure for streaming data"""
    stream_type: StreamType
    timestamp: datetime
    data: Dict[str, Any]
    symbol: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({
            "type": self.stream_type.value,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "data": self.data
        })


@dataclass
class Alert:
    """Trading alert"""
    level: AlertLevel
    title: str
    message: str
    symbol: Optional[str] = None
    action_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "symbol": self.symbol,
            "action_required": self.action_required,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Subscription:
    """Client subscription to a stream"""
    stream_types: Set[StreamType]
    symbols: Set[str]  # Empty means all symbols
    filters: Dict[str, Any] = field(default_factory=dict)


class StreamingClient:
    """Represents a connected streaming client"""

    def __init__(
        self,
        client_id: str,
        send_callback: Callable[[str], Any],
        subscription: Optional[Subscription] = None
    ):
        self.client_id = client_id
        self.send_callback = send_callback
        self.subscription = subscription or Subscription(
            stream_types=set(),
            symbols=set()
        )
        self.connected_at = datetime.now()
        self.last_message_at = datetime.now()
        self.message_count = 0

    async def send(self, message: StreamMessage) -> bool:
        """Send message to client if subscribed"""
        # Check if client is subscribed to this stream type
        if message.stream_type not in self.subscription.stream_types:
            return False

        # Check symbol filter
        if self.subscription.symbols and message.symbol:
            if message.symbol not in self.subscription.symbols:
                return False

        try:
            await self.send_callback(message.to_json())
            self.last_message_at = datetime.now()
            self.message_count += 1
            return True
        except Exception as e:
            logger.error(f"Error sending to client {self.client_id}: {e}")
            return False

    def subscribe(self, stream_type: StreamType, symbols: Optional[List[str]] = None):
        """Add subscription"""
        self.subscription.stream_types.add(stream_type)
        if symbols:
            self.subscription.symbols.update(symbols)

    def unsubscribe(self, stream_type: StreamType, symbols: Optional[List[str]] = None):
        """Remove subscription"""
        if symbols:
            self.subscription.symbols -= set(symbols)
        else:
            self.subscription.stream_types.discard(stream_type)


class StreamingServer:
    """
    Real-time streaming server for trading data.

    Usage:
        server = StreamingServer()

        # Connect data providers
        server.set_position_provider(robinhood_client)
        server.set_price_provider(price_feed)

        # Start streaming
        await server.start()

        # Broadcast updates
        await server.broadcast_position_update(positions)
    """

    def __init__(
        self,
        update_interval: float = 1.0,
        greeks_interval: float = 5.0,
        portfolio_interval: float = 10.0
    ):
        self.update_interval = update_interval
        self.greeks_interval = greeks_interval
        self.portfolio_interval = portfolio_interval

        # Connected clients
        self.clients: Dict[str, StreamingClient] = {}

        # Data providers (set externally)
        self.position_provider = None
        self.price_provider = None
        self.greeks_provider = None
        self.portfolio_provider = None

        # Alert queue
        self.alert_queue: asyncio.Queue = asyncio.Queue()

        # State
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Last known state for delta updates
        self._last_positions: Dict = {}
        self._last_prices: Dict[str, float] = {}
        self._last_greeks: Dict = {}

        # Alert rules
        self.alert_rules: List[Callable] = []

    def set_position_provider(self, provider):
        """Set position data provider"""
        self.position_provider = provider

    def set_price_provider(self, provider):
        """Set price data provider"""
        self.price_provider = provider

    def set_greeks_provider(self, provider):
        """Set Greeks calculation provider"""
        self.greeks_provider = provider

    def set_portfolio_provider(self, provider):
        """Set portfolio analytics provider"""
        self.portfolio_provider = provider

    def add_alert_rule(self, rule: Callable[[Dict], Optional[Alert]]):
        """Add alert rule function"""
        self.alert_rules.append(rule)

    # =========================================================================
    # CLIENT MANAGEMENT
    # =========================================================================

    def add_client(
        self,
        client_id: str,
        send_callback: Callable[[str], Any]
    ) -> StreamingClient:
        """Add a new streaming client"""
        client = StreamingClient(client_id, send_callback)
        self.clients[client_id] = client
        logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
        return client

    def remove_client(self, client_id: str):
        """Remove a disconnected client"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Client {client_id} disconnected. Total clients: {len(self.clients)}")

    def get_client(self, client_id: str) -> Optional[StreamingClient]:
        """Get client by ID"""
        return self.clients.get(client_id)

    # =========================================================================
    # BROADCASTING
    # =========================================================================

    async def broadcast(self, message: StreamMessage):
        """Broadcast message to all subscribed clients"""
        tasks = [client.send(message) for client in self.clients.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_positions(self, positions: List[Dict]):
        """Broadcast position updates"""
        message = StreamMessage(
            stream_type=StreamType.POSITIONS,
            timestamp=datetime.now(),
            data={"positions": positions}
        )
        await self.broadcast(message)

        # Check alert rules
        for rule in self.alert_rules:
            alert = rule({"positions": positions})
            if alert:
                await self.send_alert(alert)

    async def broadcast_greeks(self, symbol: str, greeks: Dict):
        """Broadcast Greeks update for a symbol"""
        message = StreamMessage(
            stream_type=StreamType.GREEKS,
            timestamp=datetime.now(),
            symbol=symbol,
            data=greeks
        )
        await self.broadcast(message)

    async def broadcast_price(self, symbol: str, price: float, change: float = 0, change_pct: float = 0):
        """Broadcast price update"""
        message = StreamMessage(
            stream_type=StreamType.PRICES,
            timestamp=datetime.now(),
            symbol=symbol,
            data={
                "price": price,
                "change": change,
                "change_pct": change_pct
            }
        )
        await self.broadcast(message)

    async def broadcast_portfolio(self, portfolio: Dict):
        """Broadcast portfolio summary"""
        message = StreamMessage(
            stream_type=StreamType.PORTFOLIO,
            timestamp=datetime.now(),
            data=portfolio
        )
        await self.broadcast(message)

    async def broadcast_signal(self, signal: Dict):
        """Broadcast trading signal"""
        message = StreamMessage(
            stream_type=StreamType.SIGNALS,
            timestamp=datetime.now(),
            symbol=signal.get("symbol"),
            data=signal
        )
        await self.broadcast(message)

    async def send_alert(self, alert: Alert):
        """Send alert to all clients"""
        message = StreamMessage(
            stream_type=StreamType.ALERTS,
            timestamp=datetime.now(),
            symbol=alert.symbol,
            data=alert.to_dict()
        )
        await self.broadcast(message)

    # =========================================================================
    # STREAMING LOOPS
    # =========================================================================

    async def start(self) -> None:
        """Start all streaming loops"""
        if self.running:
            return

        self.running = True
        logger.info("Starting streaming server...")

        # Start streaming loops
        self.tasks = [
            asyncio.create_task(self._position_stream_loop()),
            asyncio.create_task(self._greeks_stream_loop()),
            asyncio.create_task(self._portfolio_stream_loop()),
            asyncio.create_task(self._alert_processor_loop())
        ]

        logger.info("Streaming server started")

    async def stop(self) -> None:
        """Stop all streaming loops"""
        self.running = False

        for task in self.tasks:
            task.cancel()

        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        self.tasks = []
        logger.info("Streaming server stopped")

    async def _position_stream_loop(self) -> None:
        """Stream position updates"""
        while self.running:
            try:
                if self.position_provider:
                    positions = await self._get_positions()

                    # Check for changes
                    if self._has_position_changes(positions):
                        await self.broadcast_positions(positions)
                        self._last_positions = {p.get('id'): p for p in positions}

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position stream error: {e}")

            await asyncio.sleep(self.update_interval)

    async def _greeks_stream_loop(self) -> None:
        """Stream Greeks updates"""
        while self.running:
            try:
                if self.greeks_provider and self.position_provider:
                    positions = await self._get_positions()

                    for position in positions:
                        symbol = position.get('symbol')
                        if symbol:
                            greeks = await self._calculate_greeks(position)
                            if greeks:
                                await self.broadcast_greeks(symbol, greeks)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Greeks stream error: {e}")

            await asyncio.sleep(self.greeks_interval)

    async def _portfolio_stream_loop(self) -> None:
        """Stream portfolio updates"""
        while self.running:
            try:
                if self.portfolio_provider:
                    portfolio = await self._get_portfolio_summary()
                    await self.broadcast_portfolio(portfolio)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Portfolio stream error: {e}")

            await asyncio.sleep(self.portfolio_interval)

    async def _alert_processor_loop(self) -> None:
        """Process queued alerts"""
        while self.running:
            try:
                # Process with timeout to allow cancellation
                try:
                    alert = await asyncio.wait_for(
                        self.alert_queue.get(),
                        timeout=1.0
                    )
                    await self.send_alert(alert)
                except asyncio.TimeoutError:
                    pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert processor error: {e}")

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    async def _get_positions(self) -> List[Dict]:
        """Get current positions from provider"""
        if hasattr(self.position_provider, 'get_positions'):
            if asyncio.iscoroutinefunction(self.position_provider.get_positions):
                return await self.position_provider.get_positions()
            return self.position_provider.get_positions()
        return []

    async def _calculate_greeks(self, position: Dict) -> Optional[Dict]:
        """Calculate Greeks for a position"""
        if hasattr(self.greeks_provider, 'calculate_greeks'):
            if asyncio.iscoroutinefunction(self.greeks_provider.calculate_greeks):
                return await self.greeks_provider.calculate_greeks(position)
            return self.greeks_provider.calculate_greeks(position)
        return None

    async def _get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from provider"""
        if hasattr(self.portfolio_provider, 'get_summary'):
            if asyncio.iscoroutinefunction(self.portfolio_provider.get_summary):
                return await self.portfolio_provider.get_summary()
            return self.portfolio_provider.get_summary()
        return {}

    def _has_position_changes(self, positions: List[Dict]) -> bool:
        """Check if positions have changed"""
        current = {p.get('id'): p for p in positions}

        # Different number of positions
        if len(current) != len(self._last_positions):
            return True

        # Check for changes in each position
        for pos_id, pos in current.items():
            if pos_id not in self._last_positions:
                return True

            last = self._last_positions[pos_id]
            # Check key fields
            for key in ['quantity', 'average_cost', 'current_price', 'market_value']:
                if pos.get(key) != last.get(key):
                    return True

        return False

    # =========================================================================
    # ALERT RULES
    # =========================================================================

    def queue_alert(self, alert: Alert):
        """Queue an alert for sending"""
        self.alert_queue.put_nowait(alert)

    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        symbol: Optional[str] = None,
        **kwargs
    ):
        """Create and queue an alert"""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            symbol=symbol,
            **kwargs
        )
        self.queue_alert(alert)


# =============================================================================
# BUILT-IN ALERT RULES
# =============================================================================

def profit_target_alert_rule(threshold: float = 0.50) -> Callable:
    """Alert when position hits profit target"""
    def rule(data: Dict) -> Optional[Alert]:
        positions = data.get('positions', [])
        for pos in positions:
            pnl_pct = pos.get('unrealized_pnl_pct', 0)
            if pnl_pct >= threshold:
                return Alert(
                    level=AlertLevel.TRADE,
                    title="Profit Target Hit",
                    message=f"{pos['symbol']} has reached {pnl_pct:.0%} profit",
                    symbol=pos['symbol'],
                    action_required=True,
                    metadata={"pnl_pct": pnl_pct}
                )
        return None
    return rule


def stop_loss_alert_rule(threshold: float = -0.50) -> Callable:
    """Alert when position hits stop loss"""
    def rule(data: Dict) -> Optional[Alert]:
        positions = data.get('positions', [])
        for pos in positions:
            pnl_pct = pos.get('unrealized_pnl_pct', 0)
            if pnl_pct <= threshold:
                return Alert(
                    level=AlertLevel.CRITICAL,
                    title="Stop Loss Triggered",
                    message=f"{pos['symbol']} has reached {pnl_pct:.0%} loss",
                    symbol=pos['symbol'],
                    action_required=True,
                    metadata={"pnl_pct": pnl_pct}
                )
        return None
    return rule


def expiration_alert_rule(days_warning: int = 7) -> Callable:
    """Alert when option is approaching expiration"""
    def rule(data: Dict) -> Optional[Alert]:
        positions = data.get('positions', [])
        for pos in positions:
            dte = pos.get('days_to_expiration', 999)
            if dte <= days_warning and dte > 0:
                return Alert(
                    level=AlertLevel.WARNING,
                    title="Expiration Approaching",
                    message=f"{pos['symbol']} expires in {dte} days",
                    symbol=pos['symbol'],
                    action_required=True,
                    metadata={"dte": dte}
                )
        return None
    return rule


def delta_threshold_alert_rule(max_delta: float = 0.70) -> Callable:
    """Alert when option delta exceeds threshold"""
    def rule(data: Dict) -> Optional[Alert]:
        positions = data.get('positions', [])
        for pos in positions:
            delta = abs(pos.get('delta', 0))
            if delta >= max_delta:
                return Alert(
                    level=AlertLevel.WARNING,
                    title="High Delta Warning",
                    message=f"{pos['symbol']} delta is {delta:.2f}",
                    symbol=pos['symbol'],
                    action_required=False,
                    metadata={"delta": delta}
                )
        return None
    return rule


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing Streaming Server ===\n")

    # Create mock data provider
    class MockPositionProvider:
        def __init__(self) -> None:
            self.positions = [
                {
                    "id": "1",
                    "symbol": "AAPL 220C 12/20",
                    "quantity": 2,
                    "average_cost": 5.50,
                    "current_price": 6.25,
                    "market_value": 1250,
                    "unrealized_pnl": 150,
                    "unrealized_pnl_pct": 0.136,
                    "days_to_expiration": 22,
                    "delta": 0.45
                },
                {
                    "id": "2",
                    "symbol": "SPY 580P 12/15",
                    "quantity": -1,
                    "average_cost": 3.20,
                    "current_price": 2.80,
                    "market_value": -280,
                    "unrealized_pnl": 40,
                    "unrealized_pnl_pct": 0.125,
                    "days_to_expiration": 17,
                    "delta": -0.25
                }
            ]

        def get_positions(self) -> None:
            return self.positions

    async def mock_send(message: str):
        data = json.loads(message)
        print(f"  [{data['type']}] {data.get('symbol', 'portfolio')}: {data['data']}")

    async def test_streaming():
        server = StreamingServer(
            update_interval=2.0,
            greeks_interval=5.0,
            portfolio_interval=10.0
        )

        # Set up provider
        provider = MockPositionProvider()
        server.set_position_provider(provider)

        # Add alert rules
        server.add_alert_rule(profit_target_alert_rule(0.50))
        server.add_alert_rule(stop_loss_alert_rule(-0.50))
        server.add_alert_rule(expiration_alert_rule(7))

        # Add test client
        client = server.add_client("test-client", mock_send)
        client.subscribe(StreamType.POSITIONS)
        client.subscribe(StreamType.ALERTS)

        print("1. Starting server...")
        await server.start()

        print("2. Simulating position updates...")
        await asyncio.sleep(3)

        print("3. Triggering manual alert...")
        server.create_alert(
            AlertLevel.INFO,
            "Test Alert",
            "This is a test alert message",
            symbol="AAPL"
        )
        await asyncio.sleep(2)

        print("4. Stopping server...")
        await server.stop()

        print("\n✅ Streaming server test complete!")

    asyncio.run(test_streaming())
