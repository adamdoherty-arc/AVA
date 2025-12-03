"""
AVA Real-Time Streaming Module
==============================

Real-time streaming services for:
- Position updates
- Live Greeks calculations
- Price streaming
- Trading alerts and signals
- Portfolio analytics

Usage:
    from src.ava.streaming import (
        StreamingServer,
        WebSocketManager,
        PriceStreamer,
        GreeksStreamer
    )

    # Create streaming server
    server = StreamingServer()

    # Set up providers
    server.set_position_provider(robinhood_client)
    server.set_price_provider(price_feed)

    # Start streaming
    await server.start()

Author: AVA Trading Platform
Created: 2025-11-28
"""

from .server import (
    StreamingServer,
    StreamingClient,
    StreamMessage,
    StreamType,
    Alert,
    AlertLevel,
    Subscription,
    profit_target_alert_rule,
    stop_loss_alert_rule,
    expiration_alert_rule,
    delta_threshold_alert_rule
)

from .websocket_handler import (
    WebSocketManager,
    create_streaming_router
)

from .price_streamer import (
    PriceStreamer,
    PriceQuote,
    OptionQuote,
    PriceSource
)

from .greeks_streamer import (
    GreeksStreamer,
    GreeksCalculator,
    LiveGreeks,
    PortfolioGreeksSummary
)


__all__ = [
    # Server
    'StreamingServer',
    'StreamingClient',
    'StreamMessage',
    'StreamType',
    'Alert',
    'AlertLevel',
    'Subscription',

    # Alert rules
    'profit_target_alert_rule',
    'stop_loss_alert_rule',
    'expiration_alert_rule',
    'delta_threshold_alert_rule',

    # WebSocket
    'WebSocketManager',
    'create_streaming_router',

    # Price streaming
    'PriceStreamer',
    'PriceQuote',
    'OptionQuote',
    'PriceSource',

    # Greeks streaming
    'GreeksStreamer',
    'GreeksCalculator',
    'LiveGreeks',
    'PortfolioGreeksSummary'
]
