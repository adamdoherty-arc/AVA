"""
Sports WebSocket Broadcaster
Real-time push updates for live games, odds, and AI predictions
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
from fastapi import WebSocket, WebSocketDisconnect
import threading

logger = logging.getLogger(__name__)


class SportsBroadcastChannel(str, Enum):
    """Available broadcast channels for sports data"""
    LIVE_GAMES = "live_games"           # Live game scores and status
    ODDS_UPDATES = "odds_updates"       # Real-time odds movement
    PREDICTIONS = "predictions"         # AI prediction updates
    ALERTS = "alerts"                   # Important notifications
    GAME_DETAIL = "game_detail"         # Specific game deep data


@dataclass
class SportsUpdate:
    """Represents a sports data update message"""
    channel: str
    sport: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    update_type: str = "update"  # 'update', 'initial', 'score_change', 'odds_move'
    game_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class ConnectionManager:
    """Manages WebSocket connections with channel subscriptions"""

    def __init__(self):
        # Active connections: client_id -> websocket
        self.connections: Dict[str, WebSocket] = {}

        # Channel subscriptions: channel -> set of client_ids
        self.channel_subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # Sport subscriptions: sport -> set of client_ids (for filtering by sport)
        self.sport_subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # Game-specific subscriptions: game_id -> set of client_ids
        self.game_subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # Connection metadata
        self.connection_metadata: Dict[str, Dict] = {}

        # Lock for thread-safe operations
        self.lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "total_connections": 0,
            "messages_sent": 0,
            "errors": 0
        }

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Accept and register a new WebSocket connection"""
        try:
            await websocket.accept()

            async with self.lock:
                self.connections[client_id] = websocket
                self.connection_metadata[client_id] = {
                    "connected_at": datetime.now().isoformat(),
                    "channels": [],
                    "sports": [],
                    "games": []
                }
                self.stats["total_connections"] += 1

            logger.info(f"Sports WebSocket client {client_id} connected")
            return True

        except Exception as e:
            logger.error(f"Failed to accept WebSocket: {e}")
            return False

    async def disconnect(self, client_id: str):
        """Remove a client connection and all subscriptions"""
        async with self.lock:
            # Remove from all channel subscriptions
            for channel_subs in self.channel_subscriptions.values():
                channel_subs.discard(client_id)

            # Remove from sport subscriptions
            for sport_subs in self.sport_subscriptions.values():
                sport_subs.discard(client_id)

            # Remove from game subscriptions
            for game_subs in self.game_subscriptions.values():
                game_subs.discard(client_id)

            # Remove connection
            self.connections.pop(client_id, None)
            self.connection_metadata.pop(client_id, None)

        logger.info(f"Sports WebSocket client {client_id} disconnected")

    async def subscribe_channel(self, client_id: str, channel: str):
        """Subscribe client to a broadcast channel"""
        async with self.lock:
            self.channel_subscriptions[channel].add(client_id)
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["channels"].append(channel)

        logger.debug(f"Client {client_id} subscribed to channel: {channel}")

    async def subscribe_sport(self, client_id: str, sport: str):
        """Subscribe client to updates for a specific sport"""
        async with self.lock:
            self.sport_subscriptions[sport.upper()].add(client_id)
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["sports"].append(sport)

        logger.debug(f"Client {client_id} subscribed to sport: {sport}")

    async def subscribe_game(self, client_id: str, game_id: str):
        """Subscribe client to updates for a specific game"""
        async with self.lock:
            self.game_subscriptions[game_id].add(client_id)
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["games"].append(game_id)

        logger.debug(f"Client {client_id} subscribed to game: {game_id}")

    async def broadcast_to_channel(self, channel: str, update: SportsUpdate):
        """Broadcast update to all subscribers of a channel"""
        async with self.lock:
            subscribers = list(self.channel_subscriptions.get(channel, set()))

        if not subscribers:
            return

        message = update.to_json()
        disconnected = []

        for client_id in subscribers:
            websocket = self.connections.get(client_id)
            if websocket:
                try:
                    await websocket.send_text(message)
                    self.stats["messages_sent"] += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {client_id}: {e}")
                    disconnected.append(client_id)
                    self.stats["errors"] += 1

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

    async def broadcast_to_sport(self, sport: str, update: SportsUpdate):
        """Broadcast update to all subscribers of a sport"""
        async with self.lock:
            subscribers = list(self.sport_subscriptions.get(sport.upper(), set()))

        if not subscribers:
            return

        message = update.to_json()
        disconnected = []

        for client_id in subscribers:
            websocket = self.connections.get(client_id)
            if websocket:
                try:
                    await websocket.send_text(message)
                    self.stats["messages_sent"] += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {client_id}: {e}")
                    disconnected.append(client_id)

        for client_id in disconnected:
            await self.disconnect(client_id)

    async def broadcast_game_update(self, game_id: str, update: SportsUpdate):
        """Broadcast update to subscribers watching a specific game"""
        async with self.lock:
            subscribers = list(self.game_subscriptions.get(game_id, set()))

        if not subscribers:
            return

        message = update.to_json()
        disconnected = []

        for client_id in subscribers:
            websocket = self.connections.get(client_id)
            if websocket:
                try:
                    await websocket.send_text(message)
                    self.stats["messages_sent"] += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {client_id}: {e}")
                    disconnected.append(client_id)

        for client_id in disconnected:
            await self.disconnect(client_id)

    async def send_personal(self, client_id: str, message: Dict):
        """Send a message to a specific client"""
        websocket = self.connections.get(client_id)
        if websocket:
            try:
                await websocket.send_text(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"Failed to send personal message: {e}")
                return False
        return False

    def get_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            **self.stats,
            "active_connections": len(self.connections),
            "channels": {ch: len(subs) for ch, subs in self.channel_subscriptions.items()},
            "sports": {sp: len(subs) for sp, subs in self.sport_subscriptions.items()},
            "watched_games": len(self.game_subscriptions)
        }


class SportsBroadcaster:
    """
    Main sports broadcaster service.
    Handles real-time updates for live games, odds, and predictions.
    """

    def __init__(self):
        self.manager = ConnectionManager()
        self._previous_scores: Dict[str, Dict] = {}  # Track score changes
        self._previous_odds: Dict[str, Dict] = {}     # Track odds changes

    async def handle_websocket(self, websocket: WebSocket, client_id: str):
        """
        Main WebSocket handler for sports data.

        Expects messages like:
        {"action": "subscribe", "channel": "live_games"}
        {"action": "subscribe_sport", "sport": "NFL"}
        {"action": "subscribe_game", "game_id": "12345"}
        {"action": "unsubscribe", "channel": "live_games"}
        {"action": "ping"}
        """
        connected = await self.manager.connect(websocket, client_id)
        if not connected:
            return

        try:
            # Send welcome message with available channels
            await self.manager.send_personal(client_id, {
                "type": "connected",
                "message": "Connected to AVA Sports WebSocket",
                "available_channels": [ch.value for ch in SportsBroadcastChannel],
                "timestamp": datetime.now().isoformat()
            })

            # Listen for subscription requests
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    action = message.get("action")

                    if action == "subscribe":
                        channel = message.get("channel")
                        if channel:
                            await self.manager.subscribe_channel(client_id, channel)
                            await self.manager.send_personal(client_id, {
                                "type": "subscribed",
                                "channel": channel
                            })

                    elif action == "subscribe_sport":
                        sport = message.get("sport")
                        if sport:
                            await self.manager.subscribe_sport(client_id, sport)
                            await self.manager.send_personal(client_id, {
                                "type": "subscribed_sport",
                                "sport": sport
                            })

                    elif action == "subscribe_game":
                        game_id = message.get("game_id")
                        if game_id:
                            await self.manager.subscribe_game(client_id, game_id)
                            await self.manager.send_personal(client_id, {
                                "type": "subscribed_game",
                                "game_id": game_id
                            })

                    elif action == "ping":
                        await self.manager.send_personal(client_id, {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        })

                    elif action == "get_stats":
                        stats = self.manager.get_stats()
                        await self.manager.send_personal(client_id, {
                            "type": "stats",
                            "data": stats
                        })

                except json.JSONDecodeError:
                    await self.manager.send_personal(client_id, {
                        "type": "error",
                        "message": "Invalid JSON"
                    })

        except WebSocketDisconnect:
            await self.manager.disconnect(client_id)

        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
            await self.manager.disconnect(client_id)

    async def broadcast_live_games(self, games: List[Dict], sport: str):
        """
        Broadcast live game updates.
        Detects score changes and sends appropriate update types.
        """
        for game in games:
            game_id = game.get("game_id") or game.get("id")
            if not game_id:
                continue

            # Detect score changes
            prev = self._previous_scores.get(game_id, {})
            current_home = game.get("home_score", 0)
            current_away = game.get("away_score", 0)

            update_type = "update"
            if prev:
                if prev.get("home_score") != current_home or prev.get("away_score") != current_away:
                    update_type = "score_change"

            # Store current scores
            self._previous_scores[game_id] = {
                "home_score": current_home,
                "away_score": current_away
            }

            # Create update
            update = SportsUpdate(
                channel=SportsBroadcastChannel.LIVE_GAMES.value,
                sport=sport.upper(),
                data=game,
                update_type=update_type,
                game_id=game_id
            )

            # Broadcast to channel, sport, and game-specific subscribers
            await self.manager.broadcast_to_channel(SportsBroadcastChannel.LIVE_GAMES.value, update)
            await self.manager.broadcast_to_sport(sport, update)
            await self.manager.broadcast_game_update(game_id, update)

    async def broadcast_odds_update(self, game_id: str, sport: str, odds: Dict):
        """
        Broadcast odds update for a game.
        Detects significant odds movement.
        """
        prev = self._previous_odds.get(game_id, {})

        # Detect significant movement (> 5 cents / 5%)
        update_type = "update"
        if prev:
            home_move = abs(odds.get("home_prob", 0) - prev.get("home_prob", 0))
            if home_move > 0.05:  # 5% movement
                update_type = "odds_move"
                odds["movement"] = {
                    "home_change": odds.get("home_prob", 0) - prev.get("home_prob", 0),
                    "away_change": odds.get("away_prob", 0) - prev.get("away_prob", 0)
                }

        self._previous_odds[game_id] = {
            "home_prob": odds.get("home_prob"),
            "away_prob": odds.get("away_prob")
        }

        update = SportsUpdate(
            channel=SportsBroadcastChannel.ODDS_UPDATES.value,
            sport=sport.upper(),
            data=odds,
            update_type=update_type,
            game_id=game_id
        )

        await self.manager.broadcast_to_channel(SportsBroadcastChannel.ODDS_UPDATES.value, update)
        await self.manager.broadcast_game_update(game_id, update)

    async def broadcast_prediction(self, game_id: str, sport: str, prediction: Dict):
        """Broadcast AI prediction update"""
        update = SportsUpdate(
            channel=SportsBroadcastChannel.PREDICTIONS.value,
            sport=sport.upper(),
            data=prediction,
            update_type="prediction",
            game_id=game_id
        )

        await self.manager.broadcast_to_channel(SportsBroadcastChannel.PREDICTIONS.value, update)
        await self.manager.broadcast_to_sport(sport, update)
        await self.manager.broadcast_game_update(game_id, update)

    async def broadcast_alert(self, alert: Dict, sport: Optional[str] = None):
        """Broadcast an alert/notification"""
        update = SportsUpdate(
            channel=SportsBroadcastChannel.ALERTS.value,
            sport=sport.upper() if sport else "ALL",
            data=alert,
            update_type="alert"
        )

        await self.manager.broadcast_to_channel(SportsBroadcastChannel.ALERTS.value, update)
        if sport:
            await self.manager.broadcast_to_sport(sport, update)

    def get_connection_stats(self) -> Dict:
        """Get current connection statistics"""
        return self.manager.get_stats()


# Global singleton instance
_broadcaster: Optional[SportsBroadcaster] = None


def get_sports_broadcaster() -> SportsBroadcaster:
    """Get the global sports broadcaster instance"""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = SportsBroadcaster()
    return _broadcaster


# Helper for pushing updates from sync code (non-async context)
def push_live_games_sync(games: List[Dict], sport: str):
    """Push live game updates from synchronous code"""
    broadcaster = get_sports_broadcaster()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(broadcaster.broadcast_live_games(games, sport))
        else:
            loop.run_until_complete(broadcaster.broadcast_live_games(games, sport))
    except RuntimeError:
        # No event loop, create one
        asyncio.run(broadcaster.broadcast_live_games(games, sport))


def push_odds_sync(game_id: str, sport: str, odds: Dict):
    """Push odds update from synchronous code"""
    broadcaster = get_sports_broadcaster()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(broadcaster.broadcast_odds_update(game_id, sport, odds))
        else:
            loop.run_until_complete(broadcaster.broadcast_odds_update(game_id, sport, odds))
    except RuntimeError:
        asyncio.run(broadcaster.broadcast_odds_update(game_id, sport, odds))
