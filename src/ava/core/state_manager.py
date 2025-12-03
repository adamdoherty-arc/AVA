"""
Unified State Management for AVA
Manages conversation state across all platforms
"""

import threading
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from .models import ConversationState
import logging

logger = logging.getLogger(__name__)


class AVAStateManager:
    """Unified state manager for AVA conversations"""

    def __init__(self, ttl_hours: int = 24):
        """
        Initialize state manager

        Args:
            ttl_hours: Time to live for conversation states (default 24 hours)
        """
        self._states: Dict[str, ConversationState] = {}
        self._lock = threading.RLock()
        self.ttl = timedelta(hours=ttl_hours)
        logger.info(f"AVAStateManager initialized with {ttl_hours}h TTL")

    def get_state(self, user_id: str, platform: str = "web") -> ConversationState:
        """
        Get or create conversation state

        Args:
            user_id: User identifier
            platform: Platform (web, telegram, api)

        Returns:
            ConversationState object
        """
        state_key = f"{platform}:{user_id}"

        with self._lock:
            # Check if state exists and is valid
            if state_key in self._states:
                state = self._states[state_key]
                # Check if expired
                if datetime.now() - state.updated_at < self.ttl:
                    return state
                else:
                    # Expired, remove it
                    del self._states[state_key]

            # Create new state
            state = ConversationState(
                user_id=user_id,
                context={"platform": platform}
            )
            self._states[state_key] = state
            logger.debug(f"Created new state for {state_key}")
            return state

    def update_state(
        self,
        user_id: str,
        platform: str = "web",
        messages: Optional[List[Dict]] = None,
        context: Optional[Dict] = None,
        tools_used: Optional[List[str]] = None,
        rag_results: Optional[Dict] = None,
        preferences: Optional[Dict] = None
    ):
        """
        Update conversation state

        Args:
            user_id: User identifier
            platform: Platform identifier
            messages: New messages to add
            context: Context to update
            tools_used: Tools used in conversation
            rag_results: RAG query results
            preferences: User preferences
        """
        state_key = f"{platform}:{user_id}"

        with self._lock:
            state = self.get_state(user_id, platform)

            if messages:
                state.messages.extend(messages)

            if context:
                state.context.update(context)

            if tools_used:
                state.tools_used.extend(tools_used)

            if rag_results:
                state.rag_results = rag_results

            if preferences:
                state.preferences.update(preferences)

            state.updated_at = datetime.now()
            self._states[state_key] = state

    def clear_state(self, user_id: str, platform: str = "web"):
        """Clear conversation state"""
        state_key = f"{platform}:{user_id}"

        with self._lock:
            if state_key in self._states:
                del self._states[state_key]
                logger.debug(f"Cleared state for {state_key}")

    def get_recent_messages(self, user_id: str, platform: str = "web", limit: int = 10) -> List[Dict]:
        """Get recent messages from conversation"""
        state = self.get_state(user_id, platform)
        return state.messages[-limit:]

    def cleanup_expired(self) -> None:
        """Remove expired conversation states"""
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, state in self._states.items()
                if now - state.updated_at >= self.ttl
            ]

            for key in expired_keys:
                del self._states[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired states")

    def get_stats(self) -> Dict:
        """Get state manager statistics"""
        with self._lock:
            self.cleanup_expired()
            return {
                "total_states": len(self._states),
                "ttl_hours": self.ttl.total_seconds() / 3600,
                "platforms": set(key.split(":")[0] for key in self._states.keys())
            }

