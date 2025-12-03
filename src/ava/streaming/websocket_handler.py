"""
WebSocket Handler for Real-Time Streaming
==========================================

FastAPI WebSocket integration for the streaming server.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

from .server import (
    StreamingServer,
    StreamingClient,
    StreamType,
    profit_target_alert_rule,
    stop_loss_alert_rule,
    expiration_alert_rule,
    delta_threshold_alert_rule
)

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and integrates with StreamingServer.

    Usage with FastAPI:
        app = FastAPI()
        ws_manager = WebSocketManager()

        @app.on_event("startup")
        async def startup():
            await ws_manager.start()

        @app.websocket("/ws/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await ws_manager.handle_connection(websocket)
    """

    def __init__(
        self,
        position_provider=None,
        price_provider=None,
        greeks_provider=None,
        portfolio_provider=None
    ):
        self.streaming_server = StreamingServer()
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_count = 0

        # Set providers
        if position_provider:
            self.streaming_server.set_position_provider(position_provider)
        if price_provider:
            self.streaming_server.set_price_provider(price_provider)
        if greeks_provider:
            self.streaming_server.set_greeks_provider(greeks_provider)
        if portfolio_provider:
            self.streaming_server.set_portfolio_provider(portfolio_provider)

        # Add default alert rules
        self.streaming_server.add_alert_rule(profit_target_alert_rule(0.50))
        self.streaming_server.add_alert_rule(stop_loss_alert_rule(-0.50))
        self.streaming_server.add_alert_rule(expiration_alert_rule(7))
        self.streaming_server.add_alert_rule(delta_threshold_alert_rule(0.70))

    def set_position_provider(self, provider):
        """Set position data provider"""
        self.streaming_server.set_position_provider(provider)

    def set_price_provider(self, provider):
        """Set price data provider"""
        self.streaming_server.set_price_provider(provider)

    def set_greeks_provider(self, provider):
        """Set Greeks calculation provider"""
        self.streaming_server.set_greeks_provider(provider)

    def set_portfolio_provider(self, provider):
        """Set portfolio analytics provider"""
        self.streaming_server.set_portfolio_provider(provider)

    async def start(self):
        """Start the streaming server"""
        await self.streaming_server.start()

    async def stop(self):
        """Stop the streaming server"""
        await self.streaming_server.stop()

    async def handle_connection(
        self,
        websocket: WebSocket,
        token: Optional[str] = None
    ):
        """
        Handle a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket connection
            token: Optional authentication token
        """
        # Accept connection
        await websocket.accept()

        # Generate client ID
        self.connection_count += 1
        client_id = f"ws-{self.connection_count}-{id(websocket)}"

        logger.info(f"WebSocket connection accepted: {client_id}")

        # Create send callback
        async def send_callback(message: str):
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)

        # Register with streaming server
        client = self.streaming_server.add_client(client_id, send_callback)
        self.active_connections[client_id] = websocket

        try:
            # Send welcome message
            await websocket.send_json({
                "type": "connected",
                "client_id": client_id,
                "message": "Connected to AVA Streaming Server",
                "timestamp": datetime.now().isoformat()
            })

            # Handle incoming messages
            while True:
                try:
                    message = await websocket.receive_json()
                    await self._handle_client_message(client, message)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {client_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
        finally:
            # Cleanup
            self.streaming_server.remove_client(client_id)
            del self.active_connections[client_id]

    async def _handle_client_message(
        self,
        client: StreamingClient,
        message: Dict
    ):
        """Handle incoming message from client"""
        action = message.get("action")

        if action == "subscribe":
            # Subscribe to streams
            streams = message.get("streams", [])
            symbols = message.get("symbols", [])

            for stream_name in streams:
                try:
                    stream_type = StreamType(stream_name)
                    client.subscribe(stream_type, symbols if symbols else None)
                    logger.debug(f"Client {client.client_id} subscribed to {stream_name}")
                except ValueError:
                    pass

            await client.send_callback(json.dumps({
                "type": "subscribed",
                "streams": list(s.value for s in client.subscription.stream_types),
                "symbols": list(client.subscription.symbols)
            }))

        elif action == "unsubscribe":
            # Unsubscribe from streams
            streams = message.get("streams", [])
            symbols = message.get("symbols", [])

            for stream_name in streams:
                try:
                    stream_type = StreamType(stream_name)
                    client.unsubscribe(stream_type, symbols if symbols else None)
                except ValueError:
                    pass

            await client.send_callback(json.dumps({
                "type": "unsubscribed",
                "streams": list(s.value for s in client.subscription.stream_types)
            }))

        elif action == "ping":
            # Health check
            await client.send_callback(json.dumps({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }))

        elif action == "request_snapshot":
            # Request current state snapshot
            await self._send_snapshot(client)

    async def _send_snapshot(self, client: StreamingClient):
        """Send current state snapshot to client"""
        snapshot = {
            "type": "snapshot",
            "timestamp": datetime.now().isoformat(),
            "data": {}
        }

        # Get positions if provider available
        if self.streaming_server.position_provider:
            try:
                positions = await self.streaming_server._get_positions()
                snapshot["data"]["positions"] = positions
            except Exception as e:
                logger.error(f"Error getting positions snapshot: {e}")

        # Get portfolio if provider available
        if self.streaming_server.portfolio_provider:
            try:
                portfolio = await self.streaming_server._get_portfolio_summary()
                snapshot["data"]["portfolio"] = portfolio
            except Exception as e:
                logger.error(f"Error getting portfolio snapshot: {e}")

        await client.send_callback(json.dumps(snapshot))

    # =========================================================================
    # MANUAL TRIGGERS (for external use)
    # =========================================================================

    async def trigger_position_update(self, positions: List[Dict]):
        """Manually trigger position update broadcast"""
        await self.streaming_server.broadcast_positions(positions)

    async def trigger_price_update(
        self,
        symbol: str,
        price: float,
        change: float = 0,
        change_pct: float = 0
    ):
        """Manually trigger price update broadcast"""
        await self.streaming_server.broadcast_price(symbol, price, change, change_pct)

    async def trigger_signal(self, signal: Dict):
        """Manually trigger trading signal broadcast"""
        await self.streaming_server.broadcast_signal(signal)

    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "total_connections": self.connection_count,
            "active_connections": len(self.active_connections),
            "clients": [
                {
                    "id": client_id,
                    "subscriptions": list(
                        s.value for s in
                        self.streaming_server.clients.get(client_id, StreamingClient("", lambda x: None)).subscription.stream_types
                    )
                }
                for client_id in self.active_connections.keys()
            ]
        }


# =============================================================================
# FASTAPI INTEGRATION EXAMPLE
# =============================================================================

def create_streaming_router(ws_manager: WebSocketManager):
    """
    Create FastAPI router with WebSocket endpoint.

    Usage:
        from fastapi import FastAPI
        from ava.streaming import WebSocketManager, create_streaming_router

        app = FastAPI()
        ws_manager = WebSocketManager()

        app.include_router(create_streaming_router(ws_manager))
    """
    from fastapi import APIRouter

    router = APIRouter(prefix="/stream", tags=["streaming"])

    @router.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        token: str = Query(None)
    ):
        await ws_manager.handle_connection(websocket, token)

    @router.get("/stats")
    async def get_stats():
        return ws_manager.get_connection_stats()

    return router


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    print("\n=== Testing WebSocket Handler ===\n")

    # Create mock provider
    class MockProvider:
        def get_positions(self):
            return [
                {
                    "id": "1",
                    "symbol": "AAPL 220C",
                    "quantity": 2,
                    "unrealized_pnl_pct": 0.15
                }
            ]

        def get_summary(self):
            return {
                "total_value": 125000,
                "daily_pnl": 450,
                "options_value": 15000
            }

    # Create app
    app = FastAPI(title="AVA Streaming Test")

    # Create WebSocket manager
    ws_manager = WebSocketManager()
    ws_manager.set_position_provider(MockProvider())
    ws_manager.set_portfolio_provider(MockProvider())

    # Add router
    app.include_router(create_streaming_router(ws_manager))

    @app.on_event("startup")
    async def startup():
        await ws_manager.start()

    @app.on_event("shutdown")
    async def shutdown():
        await ws_manager.stop()

    @app.get("/")
    async def root():
        return {
            "message": "AVA Streaming Server",
            "websocket_url": "ws://localhost:8000/stream/ws",
            "docs": "/docs"
        }

    print("Starting test server...")
    print("WebSocket URL: ws://localhost:8000/stream/ws")
    print("API docs: http://localhost:8000/docs")
    print("\nTo test, connect via WebSocket and send:")
    print('  {"action": "subscribe", "streams": ["positions", "portfolio"]}')

    uvicorn.run(app, host="0.0.0.0", port=8000)
