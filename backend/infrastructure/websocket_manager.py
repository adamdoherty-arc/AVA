"""
WebSocket Manager for Real-Time Updates

Provides:
- Connection management
- Room-based broadcasting
- Heartbeat monitoring
- Automatic reconnection handling
"""

import asyncio
import logging
import json
from typing import Dict, Set, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""
    websocket: WebSocket
    user_id: Optional[str]
    rooms: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    message_count: int = 0


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.

    Features:
    - Connection tracking
    - Room-based messaging (e.g., "positions", "alerts")
    - Broadcast to all or specific users
    - Heartbeat monitoring
    - Graceful disconnection handling

    Usage:
        manager = WebSocketManager()

        # In endpoint
        @router.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            await manager.connect(websocket, user_id)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle incoming messages
            except WebSocketDisconnect:
                manager.disconnect(websocket)

        # Broadcast updates
        await manager.broadcast_to_room("positions", {"type": "update", "data": positions})
    """

    def __init__(
        self,
        heartbeat_interval: int = 30,
        heartbeat_timeout: int = 60
    ):
        self._connections: Dict[WebSocket, ConnectionInfo] = {}
        self._rooms: Dict[str, Set[WebSocket]] = {}
        self._user_connections: Dict[str, Set[WebSocket]] = {}
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        rooms: Optional[Set[str]] = None
    ) -> ConnectionInfo:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            user_id: Optional user identifier
            rooms: Initial rooms to join

        Returns:
            ConnectionInfo for the new connection
        """
        await websocket.accept()

        async with self._lock:
            conn_info = ConnectionInfo(
                websocket=websocket,
                user_id=user_id,
                rooms=rooms or set()
            )
            self._connections[websocket] = conn_info

            # Track by user
            if user_id:
                if user_id not in self._user_connections:
                    self._user_connections[user_id] = set()
                self._user_connections[user_id].add(websocket)

            # Join initial rooms
            for room in conn_info.rooms:
                if room not in self._rooms:
                    self._rooms[room] = set()
                self._rooms[room].add(websocket)

            logger.info(
                f"WebSocket connected: user={user_id}, "
                f"rooms={rooms}, total={len(self._connections)}"
            )

            return conn_info

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        async with self._lock:
            conn_info = self._connections.pop(websocket, None)

            if conn_info:
                # Remove from user tracking
                if conn_info.user_id:
                    user_conns = self._user_connections.get(conn_info.user_id, set())
                    user_conns.discard(websocket)
                    if not user_conns:
                        self._user_connections.pop(conn_info.user_id, None)

                # Remove from rooms
                for room in conn_info.rooms:
                    room_conns = self._rooms.get(room, set())
                    room_conns.discard(websocket)
                    if not room_conns:
                        self._rooms.pop(room, None)

                logger.info(
                    f"WebSocket disconnected: user={conn_info.user_id}, "
                    f"total={len(self._connections)}"
                )

    async def join_room(self, websocket: WebSocket, room: str):
        """Add a connection to a room."""
        async with self._lock:
            conn_info = self._connections.get(websocket)
            if conn_info:
                conn_info.rooms.add(room)
                if room not in self._rooms:
                    self._rooms[room] = set()
                self._rooms[room].add(websocket)

    async def leave_room(self, websocket: WebSocket, room: str):
        """Remove a connection from a room."""
        async with self._lock:
            conn_info = self._connections.get(websocket)
            if conn_info:
                conn_info.rooms.discard(room)
                room_conns = self._rooms.get(room, set())
                room_conns.discard(websocket)

    async def send_personal(self, websocket: WebSocket, message: Any):
        """Send a message to a specific connection."""
        try:
            if isinstance(message, dict):
                await websocket.send_json(message)
            else:
                await websocket.send_text(str(message))

            conn_info = self._connections.get(websocket)
            if conn_info:
                conn_info.message_count += 1

        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)

    async def send_to_user(self, user_id: str, message: Any):
        """Send a message to all connections for a user."""
        connections = self._user_connections.get(user_id, set()).copy()

        for websocket in connections:
            await self.send_personal(websocket, message)

    async def broadcast_to_room(self, room: str, message: Any):
        """Broadcast a message to all connections in a room."""
        connections = self._rooms.get(room, set()).copy()

        for websocket in connections:
            await self.send_personal(websocket, message)

        logger.debug(f"Broadcast to room '{room}': {len(connections)} connections")

    async def broadcast_all(self, message: Any):
        """Broadcast a message to all connections."""
        connections = list(self._connections.keys())

        for websocket in connections:
            await self.send_personal(websocket, message)

        logger.debug(f"Broadcast to all: {len(connections)} connections")

    async def handle_heartbeat(self, websocket: WebSocket):
        """Update heartbeat timestamp for a connection."""
        conn_info = self._connections.get(websocket)
        if conn_info:
            conn_info.last_heartbeat = datetime.now()

    async def start_heartbeat_monitor(self):
        """Start background task to monitor connection health."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        """Background loop to check connection health."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)

            now = datetime.now()
            stale_connections = []

            async with self._lock:
                for websocket, conn_info in self._connections.items():
                    elapsed = (now - conn_info.last_heartbeat).total_seconds()
                    if elapsed > self._heartbeat_timeout:
                        stale_connections.append(websocket)

            # Disconnect stale connections
            for websocket in stale_connections:
                logger.warning(f"Disconnecting stale WebSocket (no heartbeat)")
                try:
                    await websocket.close()
                except Exception as e:
                    logger.debug(f"Error closing stale websocket: {e}")
                await self.disconnect(websocket)

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            "total_connections": len(self._connections),
            "unique_users": len(self._user_connections),
            "rooms": {
                room: len(conns)
                for room, conns in self._rooms.items()
            },
            "total_messages_sent": sum(
                c.message_count for c in self._connections.values()
            )
        }


# =============================================================================
# Position Update Broadcaster
# =============================================================================

class PositionUpdateBroadcaster:
    """
    Specialized broadcaster for portfolio position updates.

    Automatically sends updates to subscribed clients at configurable intervals.
    """

    def __init__(
        self,
        ws_manager: WebSocketManager,
        update_interval: int = 5
    ):
        self._ws_manager = ws_manager
        self._update_interval = update_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._position_fetcher: Optional[Callable] = None

    def set_position_fetcher(self, fetcher: Callable):
        """Set the function to call to get positions."""
        self._position_fetcher = fetcher

    async def start(self):
        """Start the update broadcaster."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._broadcast_loop())
            logger.info(f"Position broadcaster started (interval={self._update_interval}s)")

    async def stop(self):
        """Stop the update broadcaster."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Position broadcaster stopped")

    async def _broadcast_loop(self):
        """Background loop to broadcast position updates."""
        while self._running:
            try:
                # Only broadcast if there are subscribers
                room_size = len(self._ws_manager._rooms.get("positions", set()))

                if room_size > 0 and self._position_fetcher:
                    positions = await self._position_fetcher()

                    await self._ws_manager.broadcast_to_room("positions", {
                        "type": "positions_update",
                        "data": positions,
                        "timestamp": datetime.now().isoformat()
                    })

                    logger.debug(f"Broadcast positions to {room_size} clients")

            except Exception as e:
                logger.error(f"Error in position broadcast: {e}")

            await asyncio.sleep(self._update_interval)


# =============================================================================
# Singleton Instance
# =============================================================================

_ws_manager: Optional[WebSocketManager] = None
_position_broadcaster: Optional[PositionUpdateBroadcaster] = None


def get_ws_manager() -> WebSocketManager:
    """Get the WebSocket manager singleton."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


def get_position_broadcaster() -> PositionUpdateBroadcaster:
    """Get the position broadcaster singleton."""
    global _position_broadcaster, _ws_manager
    if _position_broadcaster is None:
        _ws_manager = get_ws_manager()
        _position_broadcaster = PositionUpdateBroadcaster(_ws_manager)
    return _position_broadcaster
