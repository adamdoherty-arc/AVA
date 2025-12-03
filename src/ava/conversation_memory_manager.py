"""
AVA Conversation Memory Manager
================================

Tracks all conversations, actions performed, and unanswered questions.
Automatically creates Legion tasks for improvement opportunities.

Features:
- Session tracking with context retention
- Action history (what AVA did)
- Unanswered questions database
- Automatic Legion task creation
- Memory/recall functionality
- Performance analytics

Author: Magnus Trading Platform
Created: 2025-11-11
"""

import logging
import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """Manages conversation memory, recall, and unanswered questions"""

    def __init__(self) -> None:
        """Initialize memory manager with database connection"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

        # Ensure schema exists
        self._initialize_schema()

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _initialize_schema(self) -> None:
        """Initialize database schema if not exists"""
        try:
            schema_file = os.path.join(
                os.path.dirname(__file__),
                'conversation_memory_schema.sql'
            )

            if os.path.exists(schema_file):
                conn = self.get_connection()
                with conn.cursor() as cur:
                    with open(schema_file, 'r') as f:
                        cur.execute(f.read())
                conn.commit()
                conn.close()
                logger.info("[OK] Conversation memory schema initialized")
            else:
                logger.warning(f"Schema file not found: {schema_file}")

        except Exception as e:
            # Ignore "already exists" errors for triggers
            error_str = str(e).lower()
            if 'already exists' in error_str and 'trigger' in error_str:
                logger.debug(f"Schema already initialized (trigger exists)")
            else:
                logger.error(f"Error initializing schema: {e}")

    # =================================================================
    # CONVERSATION MANAGEMENT
    # =================================================================

    def start_conversation(self, user_id: str, platform: str = 'web') -> int:
        """
        Start a new conversation session

        Args:
            user_id: Unique user identifier (Telegram ID, session ID, etc.)
            platform: Platform type ('telegram', 'web', 'api')

        Returns:
            conversation_id
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ava_conversations (user_id, platform)
                    VALUES (%s, %s)
                    RETURNING conversation_id
                """, (user_id, platform))

                conversation_id = cur.fetchone()[0]
                conn.commit()
                conn.close()

                logger.info(f"Started conversation {conversation_id} for user {user_id} on {platform}")
                return conversation_id

        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            return None

    def end_conversation(self, conversation_id: int, satisfaction_rating: Optional[int] = None):
        """End a conversation session"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE ava_conversations
                    SET ended_at = NOW(),
                        user_satisfaction_rating = %s,
                        total_duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))
                    WHERE conversation_id = %s
                """, (satisfaction_rating, conversation_id))

                conn.commit()
                conn.close()

                logger.info(f"Ended conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Error ending conversation: {e}")

    def get_active_conversation(self, user_id: str, platform: str = 'web') -> Optional[int]:
        """Get active conversation for user or create new one"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                # Find recent active conversation (within last 30 minutes)
                cur.execute("""
                    SELECT conversation_id
                    FROM ava_conversations
                    WHERE user_id = %s
                      AND platform = %s
                      AND ended_at IS NULL
                      AND started_at > NOW() - INTERVAL '30 minutes'
                    ORDER BY started_at DESC
                    LIMIT 1
                """, (user_id, platform))

                result = cur.fetchone()
                conn.close()

                if result:
                    return result[0]
                else:
                    # Create new conversation
                    return self.start_conversation(user_id, platform)

        except Exception as e:
            logger.error(f"Error getting active conversation: {e}")
            return self.start_conversation(user_id, platform)

    # =================================================================
    # MESSAGE TRACKING
    # =================================================================

    def log_message(
        self,
        conversation_id: int,
        user_message: str,
        ava_response: str,
        intent_detected: str,
        confidence_score: float,
        action_performed: Optional[str] = None,
        action_success: Optional[bool] = None,
        action_duration_ms: Optional[int] = None,
        action_metadata: Optional[Dict] = None,
        model_used: Optional[str] = None,
        provider: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None
    ) -> int:
        """
        Log a message exchange

        Returns:
            message_id
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ava_messages (
                        conversation_id, message_type, user_message, ava_response,
                        intent_detected, confidence_score,
                        action_performed, action_success, action_duration_ms, action_metadata,
                        model_used, provider, tokens_used, cost_usd
                    ) VALUES (
                        %s, 'ava_response', %s, %s,
                        %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    RETURNING message_id
                """, (
                    conversation_id, user_message, ava_response,
                    intent_detected, confidence_score,
                    action_performed, action_success, action_duration_ms,
                    Json(action_metadata) if action_metadata else None,
                    model_used, provider, tokens_used, cost_usd
                ))

                message_id = cur.fetchone()[0]

                # Update message count
                cur.execute("""
                    UPDATE ava_conversations
                    SET message_count = message_count + 1
                    WHERE conversation_id = %s
                """, (conversation_id,))

                conn.commit()
                conn.close()

                return message_id

        except Exception as e:
            logger.error(f"Error logging message: {e}")
            return None

    # =================================================================
    # ACTION TRACKING
    # =================================================================

    def log_action(
        self,
        conversation_id: int,
        message_id: Optional[int],
        action_type: str,
        action_target: str,
        parameters: Dict,
        result_summary: str,
        result_data: Optional[Dict] = None,
        result_count: Optional[int] = None,
        execution_time_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        platform: Optional[str] = None
    ) -> int:
        """
        Log an action performed by AVA

        Args:
            action_type: Type of action ('analyzed_watchlist', 'ranked_strategies', etc.)
            action_target: What was acted upon ('NVDA', 'Tech Watchlist', etc.)
            parameters: Input parameters used
            result_summary: Human-readable summary
            result_data: Structured results

        Returns:
            action_id
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ava_action_history (
                        conversation_id, message_id,
                        action_type, action_target,
                        parameters, result_summary, result_data, result_count,
                        execution_time_ms, success, error_message,
                        user_id, platform
                    ) VALUES (
                        %s, %s,
                        %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    )
                    RETURNING action_id
                """, (
                    conversation_id, message_id,
                    action_type, action_target,
                    Json(parameters), result_summary, Json(result_data) if result_data else None, result_count,
                    execution_time_ms, success, error_message,
                    user_id, platform
                ))

                action_id = cur.fetchone()[0]
                conn.commit()
                conn.close()

                logger.info(f"Logged action {action_id}: {action_type} on {action_target}")
                return action_id

        except Exception as e:
            logger.error(f"Error logging action: {e}")
            return None

    def get_recent_actions(self, conversation_id: int, limit: int = 5) -> List[Dict]:
        """Get recent actions for memory/context"""
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        action_type,
                        action_target,
                        result_summary,
                        executed_at
                    FROM ava_action_history
                    WHERE conversation_id = %s
                    ORDER BY executed_at DESC
                    LIMIT %s
                """, (conversation_id, limit))

                actions = cur.fetchall()
                conn.close()

                return [dict(action) for action in actions]

        except Exception as e:
            logger.error(f"Error getting recent actions: {e}")
            return []

    # =================================================================
    # UNANSWERED QUESTIONS
    # =================================================================

    def record_unanswered_question(
        self,
        user_question: str,
        intent_detected: Optional[str],
        confidence_score: float,
        failure_reason: str,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        conversation_id: Optional[int] = None,
        message_id: Optional[int] = None
    ) -> int:
        """
        Record an unanswered question
        Auto-deduplicates and tracks frequency
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                # Use the database function for deduplication
                cur.execute("""
                    SELECT record_unanswered_question(
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    user_question, intent_detected, confidence_score,
                    failure_reason, error_message,
                    user_id, platform,
                    conversation_id, message_id
                ))

                question_id = cur.fetchone()[0]
                conn.commit()
                conn.close()

                logger.warning(f"Recorded unanswered question {question_id}: {user_question} ({failure_reason})")
                return question_id

        except Exception as e:
            logger.error(f"Error recording unanswered question: {e}")
            return None

    def get_unanswered_questions_needing_tasks(self, min_occurrences: int = 3) -> List[Dict]:
        """Get unanswered questions that need Legion tasks created"""
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM ava_questions_needing_tasks
                    WHERE occurrence_count >= %s
                    ORDER BY occurrence_count DESC
                    LIMIT 20
                """, (min_occurrences,))

                questions = cur.fetchall()
                conn.close()

                return [dict(q) for q in questions]

        except Exception as e:
            logger.error(f"Error getting questions needing tasks: {e}")
            return []

    # =================================================================
    # CONTEXT / MEMORY
    # =================================================================

    def set_context(
        self,
        conversation_id: int,
        key: str,
        value: str,
        value_json: Optional[Dict] = None,
        context_type: str = 'session_state',
        expires_at: Optional[datetime] = None
    ):
        """Set conversation context for memory/recall"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ava_conversation_context (
                        conversation_id, key, value, value_json, context_type, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (conversation_id, key)
                    DO UPDATE SET
                        value = EXCLUDED.value,
                        value_json = EXCLUDED.value_json,
                        updated_at = NOW()
                """, (conversation_id, key, value, Json(value_json) if value_json else None, context_type, expires_at))

                conn.commit()
                conn.close()

        except Exception as e:
            logger.error(f"Error setting context: {e}")

    def get_context(self, conversation_id: int, key: str) -> Optional[Dict]:
        """Get conversation context"""
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT key, value, value_json, context_type, created_at, updated_at
                    FROM ava_conversation_context
                    WHERE conversation_id = %s AND key = %s
                      AND (expires_at IS NULL OR expires_at > NOW())
                """, (conversation_id, key))

                result = cur.fetchone()
                conn.close()

                return dict(result) if result else None

        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return None

    def get_all_context(self, conversation_id: int) -> Dict:
        """Get all context for conversation"""
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT key, value, value_json
                    FROM ava_conversation_context
                    WHERE conversation_id = %s
                      AND (expires_at IS NULL OR expires_at > NOW())
                """, (conversation_id,))

                results = cur.fetchall()
                conn.close()

                # Build context dict
                context = {}
                for row in results:
                    context[row['key']] = row['value_json'] if row['value_json'] else row['value']

                return context

        except Exception as e:
            logger.error(f"Error getting all context: {e}")
            return {}

    # =================================================================
    # USER PREFERENCES
    # =================================================================

    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences (persistent across sessions)"""
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM ava_user_preferences
                    WHERE user_id = %s
                """, (user_id,))

                result = cur.fetchone()
                conn.close()

                return dict(result) if result else {}

        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}

    def set_user_preference(self, user_id: str, preference_key: str, preference_value: Any):
        """Set a user preference"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                # First ensure user exists
                cur.execute("""
                    INSERT INTO ava_user_preferences (user_id)
                    VALUES (%s)
                    ON CONFLICT (user_id) DO NOTHING
                """, (user_id,))

                # Update preferences_json
                cur.execute("""
                    UPDATE ava_user_preferences
                    SET preferences_json = COALESCE(preferences_json, '{}'::jsonb) || %s::jsonb,
                        updated_at = NOW()
                    WHERE user_id = %s
                """, (json.dumps({preference_key: preference_value}), user_id))

                conn.commit()
                conn.close()

        except Exception as e:
            logger.error(f"Error setting user preference: {e}")

    # =================================================================
    # ANALYTICS
    # =================================================================

    def get_unanswered_questions_summary(self) -> List[Dict]:
        """Get summary of unanswered questions by failure reason"""
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM ava_unanswered_questions_summary")

                results = cur.fetchall()
                conn.close()

                return [dict(r) for r in results]

        except Exception as e:
            logger.error(f"Error getting unanswered summary: {e}")
            return []

    def get_performance_metrics(self, days: int = 7) -> List[Dict]:
        """Get AVA performance metrics"""
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM ava_performance_metrics
                    WHERE date > CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY date DESC
                """, (days,))

                results = cur.fetchall()
                conn.close()

                return [dict(r) for r in results]

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return []


# Example usage
if __name__ == "__main__":
    manager = ConversationMemoryManager()

    # Start conversation
    conv_id = manager.start_conversation(user_id="test_user_123", platform="web")
    print(f"Started conversation: {conv_id}")

    # Log a message
    msg_id = manager.log_message(
        conversation_id=conv_id,
        user_message="Analyze the NVDA watchlist",
        ava_response="I found 10 high-quality CSP opportunities...",
        intent_detected="WATCHLIST_ANALYSIS",
        confidence_score=0.95,
        action_performed="analyzed_watchlist",
        action_success=True,
        action_duration_ms=2500,
        action_metadata={"watchlist": "NVDA", "strategies_found": 10}
    )
    print(f"Logged message: {msg_id}")

    # Log an action
    action_id = manager.log_action(
        conversation_id=conv_id,
        message_id=msg_id,
        action_type="analyzed_watchlist",
        action_target="NVDA",
        parameters={"min_score": 60, "strategies": ["CSP", "CC"]},
        result_summary="Found 10 opportunities above score 60",
        result_count=10,
        execution_time_ms=2500,
        success=True
    )
    print(f"Logged action: {action_id}")

    # Set context
    manager.set_context(
        conversation_id=conv_id,
        key="last_watchlist_analyzed",
        value="NVDA",
        context_type="recent_action"
    )

    # Get context
    context = manager.get_context(conv_id, "last_watchlist_analyzed")
    print(f"Context: {context}")

    # End conversation
    manager.end_conversation(conv_id, satisfaction_rating=5)
    print("Conversation ended")
