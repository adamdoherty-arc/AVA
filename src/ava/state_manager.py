"""
AVA State Manager
=================

Manages conversation state for multi-step workflows using Redis-backed FSM.

Features:
- Finite State Machine (FSM) for conversation flows
- User session data with TTL
- Workflow context storage
- Timeout handling for idle conversations
- State transition history

States:
- IDLE: No active conversation
- AWAITING_TICKER: Waiting for ticker symbol
- AWAITING_STRATEGY: Waiting for strategy selection
- AWAITING_CONFIRMATION: Waiting for yes/no confirmation
- PROCESSING: Bot is processing request
- ERROR: Error state

Usage:
    state_manager = StateManager(redis_url="redis://localhost:6379")
    await state_manager.set_state(chat_id, ConversationState.AWAITING_TICKER)
    current = await state_manager.get_state(chat_id)
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Enumeration of conversation states"""
    IDLE = "idle"

    # Stock analysis workflow
    AWAITING_TICKER = "awaiting_ticker"
    AWAITING_STRATEGY_TYPE = "awaiting_strategy_type"
    AWAITING_CONFIRMATION = "awaiting_confirmation"

    # Trade execution workflow
    REVIEWING_TRADE = "reviewing_trade"
    AWAITING_TRADE_APPROVAL = "awaiting_trade_approval"
    EXECUTING_TRADE = "executing_trade"
    TRADE_EXECUTED = "trade_executed"

    # Portfolio workflow
    SHOWING_POSITIONS = "showing_positions"
    POSITION_DETAIL = "position_detail"

    # General states
    PROCESSING = "processing"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class StateContext:
    """Context data for a conversation state"""
    workflow: Optional[str] = None  # e.g., "stock_analysis", "trade_execution"
    step: int = 0
    data: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = {}


@dataclass
class StateRecord:
    """Complete state record for a chat"""
    current_state: ConversationState
    previous_state: Optional[ConversationState] = None
    context: StateContext = None
    history: List[str] = None
    created_at: str = None
    updated_at: str = None
    expires_at: str = None

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = StateContext()
        if self.history is None:
            self.history = []
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow().isoformat()
        if self.expires_at is None:
            self.expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat()


class StateManager:
    """
    Manages conversation state using Redis

    Redis Keys:
        ava:state:{chat_id} → StateRecord (JSON)
        ava:session:{chat_id} → Session data (JSON)
        ava:state:timeouts → Sorted set of (chat_id, timestamp)
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize state manager

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis: Optional[redis.Redis] = None

        # Key prefixes
        self.STATE_PREFIX = "ava:state"
        self.SESSION_PREFIX = "ava:session"
        self.TIMEOUT_KEY = "ava:state:timeouts"

        # Default TTLs
        self.STATE_TTL = 3600  # 1 hour
        self.SESSION_TTL = 3600  # 1 hour

        # State timeouts (minutes)
        self.STATE_TIMEOUTS = {
            ConversationState.AWAITING_TICKER: 5,
            ConversationState.AWAITING_STRATEGY_TYPE: 3,
            ConversationState.AWAITING_CONFIRMATION: 10,
            ConversationState.REVIEWING_TRADE: 10,
            ConversationState.AWAITING_TRADE_APPROVAL: 5,
            ConversationState.EXECUTING_TRADE: 2,
            ConversationState.PROCESSING: 5,
        }

    async def connect(self) -> None:
        """Connect to Redis"""
        if not self.redis:
            self.redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10
            )
            await self.redis.ping()
            logger.info(f"✅ StateManager connected to Redis: {self.redis_url}")

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.info("✅ StateManager disconnected from Redis")

    def _state_key(self, chat_id: int) -> str:
        """Get Redis key for state"""
        return f"{self.STATE_PREFIX}:{chat_id}"

    def _session_key(self, chat_id: int) -> str:
        """Get Redis key for session"""
        return f"{self.SESSION_PREFIX}:{chat_id}"

    async def get_state(self, chat_id: int) -> ConversationState:
        """
        Get current conversation state

        Args:
            chat_id: Telegram chat ID

        Returns:
            Current conversation state (IDLE if not found)
        """
        if not self.redis:
            await self.connect()

        try:
            key = self._state_key(chat_id)
            data = await self.redis.get(key)

            if not data:
                return ConversationState.IDLE

            record = json.loads(data)
            return ConversationState(record["current_state"])

        except Exception as e:
            logger.error(f"Error getting state for {chat_id}: {e}")
            return ConversationState.IDLE

    async def set_state(
        self,
        chat_id: int,
        state: ConversationState,
        context: Optional[StateContext] = None
    ) -> bool:
        """
        Set conversation state

        Args:
            chat_id: Telegram chat ID
            state: New conversation state
            context: Optional context data

        Returns:
            True if successful
        """
        if not self.redis:
            await self.connect()

        try:
            # Get current state record
            key = self._state_key(chat_id)
            existing_data = await self.redis.get(key)

            if existing_data:
                record_dict = json.loads(existing_data)
                record = StateRecord(
                    current_state=ConversationState(record_dict["current_state"]),
                    previous_state=ConversationState(record_dict.get("previous_state")) if record_dict.get("previous_state") else None,
                    context=StateContext(**record_dict.get("context", {})),
                    history=record_dict.get("history", []),
                    created_at=record_dict.get("created_at"),
                    updated_at=datetime.utcnow().isoformat(),
                    expires_at=record_dict.get("expires_at")
                )
                record.previous_state = record.current_state
            else:
                record = StateRecord(current_state=state, context=context or StateContext())

            # Update state
            record.current_state = state
            if context:
                record.context = context
            record.updated_at = datetime.utcnow().isoformat()

            # Add to history
            record.history.append(f"{state.value}@{record.updated_at}")

            # Keep only last 10 history entries
            if len(record.history) > 10:
                record.history = record.history[-10:]

            # Calculate expiration
            timeout_minutes = self.STATE_TIMEOUTS.get(state, 60)
            expires_at = datetime.utcnow() + timedelta(minutes=timeout_minutes)
            record.expires_at = expires_at.isoformat()

            # Serialize to JSON
            record_dict = {
                "current_state": record.current_state.value,
                "previous_state": record.previous_state.value if record.previous_state else None,
                "context": asdict(record.context),
                "history": record.history,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "expires_at": record.expires_at
            }

            # Save to Redis
            await self.redis.setex(
                key,
                self.STATE_TTL,
                json.dumps(record_dict)
            )

            # Add to timeout sorted set
            await self.redis.zadd(
                self.TIMEOUT_KEY,
                {str(chat_id): expires_at.timestamp()}
            )

            logger.info(f"State updated for {chat_id}: {state.value}")
            return True

        except Exception as e:
            logger.error(f"Error setting state for {chat_id}: {e}")
            return False

    async def get_context(self, chat_id: int) -> StateContext:
        """
        Get conversation context

        Args:
            chat_id: Telegram chat ID

        Returns:
            State context
        """
        if not self.redis:
            await self.connect()

        try:
            key = self._state_key(chat_id)
            data = await self.redis.get(key)

            if not data:
                return StateContext()

            record = json.loads(data)
            context_dict = record.get("context", {})
            return StateContext(**context_dict)

        except Exception as e:
            logger.error(f"Error getting context for {chat_id}: {e}")
            return StateContext()

    async def set_context_data(self, chat_id: int, key: str, value: Any) -> bool:
        """
        Set a value in the context data

        Args:
            chat_id: Telegram chat ID
            key: Context data key
            value: Value to store

        Returns:
            True if successful
        """
        context = await self.get_context(chat_id)
        context.data[key] = value

        current_state = await self.get_state(chat_id)
        return await self.set_state(chat_id, current_state, context)

    async def get_context_data(self, chat_id: int, key: str, default: Any = None) -> Any:
        """
        Get a value from the context data

        Args:
            chat_id: Telegram chat ID
            key: Context data key
            default: Default value if not found

        Returns:
            Context data value
        """
        context = await self.get_context(chat_id)
        return context.data.get(key, default)

    async def reset_state(self, chat_id: int) -> bool:
        """
        Reset conversation to IDLE state

        Args:
            chat_id: Telegram chat ID

        Returns:
            True if successful
        """
        return await self.set_state(chat_id, ConversationState.IDLE, StateContext())

    async def set_session_data(self, chat_id: int, data: Dict[str, Any]) -> bool:
        """
        Set session data (user preferences, temp data, etc.)

        Args:
            chat_id: Telegram chat ID
            data: Session data dictionary

        Returns:
            True if successful
        """
        if not self.redis:
            await self.connect()

        try:
            key = self._session_key(chat_id)
            await self.redis.setex(
                key,
                self.SESSION_TTL,
                json.dumps(data)
            )
            return True

        except Exception as e:
            logger.error(f"Error setting session for {chat_id}: {e}")
            return False

    async def get_session_data(self, chat_id: int) -> Dict[str, Any]:
        """
        Get session data

        Args:
            chat_id: Telegram chat ID

        Returns:
            Session data dictionary
        """
        if not self.redis:
            await self.connect()

        try:
            key = self._session_key(chat_id)
            data = await self.redis.get(key)

            if not data:
                return {}

            return json.loads(data)

        except Exception as e:
            logger.error(f"Error getting session for {chat_id}: {e}")
            return {}

    async def get_expired_conversations(self) -> List[int]:
        """
        Get list of expired conversation chat IDs

        Returns:
            List of chat IDs with expired conversations
        """
        if not self.redis:
            await self.connect()

        try:
            now = datetime.utcnow().timestamp()

            # Get chat IDs from sorted set where score (timestamp) < now
            expired = await self.redis.zrangebyscore(
                self.TIMEOUT_KEY,
                min=0,
                max=now
            )

            return [int(chat_id) for chat_id in expired]

        except Exception as e:
            logger.error(f"Error getting expired conversations: {e}")
            return []

    async def cleanup_expired(self) -> int:
        """
        Clean up expired conversations (set to TIMEOUT state)

        Returns:
            Number of conversations cleaned up
        """
        if not self.redis:
            await self.connect()

        try:
            expired_chat_ids = await self.get_expired_conversations()

            for chat_id in expired_chat_ids:
                await self.set_state(chat_id, ConversationState.TIMEOUT)
                await self.redis.zrem(self.TIMEOUT_KEY, str(chat_id))
                logger.info(f"Conversation timed out for chat {chat_id}")

            return len(expired_chat_ids)

        except Exception as e:
            logger.error(f"Error cleaning up expired conversations: {e}")
            return 0


# Convenience functions
async def get_state(chat_id: int) -> ConversationState:
    """Get current state for a chat"""
    manager = StateManager()
    return await manager.get_state(chat_id)


async def set_state(chat_id: int, state: ConversationState, context: Optional[StateContext] = None):
    """Set state for a chat"""
    manager = StateManager()
    return await manager.set_state(chat_id, state, context)


async def reset_state(chat_id: int):
    """Reset chat to IDLE state"""
    manager = StateManager()
    return await manager.reset_state(chat_id)


# Test function
async def test_state_manager():
    """Test state manager functionality"""
    print("Testing StateManager...")

    manager = StateManager()
    await manager.connect()

    test_chat_id = 123456789

    # Test state transitions
    print(f"\n1. Initial state: {await manager.get_state(test_chat_id)}")

    await manager.set_state(test_chat_id, ConversationState.AWAITING_TICKER)
    print(f"2. After setting AWAITING_TICKER: {await manager.get_state(test_chat_id)}")

    await manager.set_context_data(test_chat_id, "ticker", "AAPL")
    context = await manager.get_context(test_chat_id)
    print(f"3. Context data: {context.data}")

    await manager.set_state(test_chat_id, ConversationState.AWAITING_STRATEGY_TYPE)
    print(f"4. After setting AWAITING_STRATEGY: {await manager.get_state(test_chat_id)}")

    await manager.reset_state(test_chat_id)
    print(f"5. After reset: {await manager.get_state(test_chat_id)}")

    await manager.disconnect()
    print("\n✅ StateManager test complete")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_state_manager())
