"""
Automation Control Service
==========================

Centralized service for managing automation state (enable/disable)
with Redis caching for fast lookups and PostgreSQL for persistence.

Features:
- Fast Redis-based O(1) state lookups
- PostgreSQL persistence that survives restarts
- Execution tracking (start/complete)
- Running task revocation support
- Comprehensive statistics and history

Author: AVA Trading Platform
Created: 2025-11-28
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from contextlib import contextmanager

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status values for automation executions."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REVOKED = "revoked"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class AutomationControlService:
    """
    Manages enable/disable state for all automations.

    Uses Redis for fast O(1) lookups and PostgreSQL for persistence.
    Thread-safe and designed for concurrent access from multiple workers.
    """

    # Redis key patterns
    REDIS_STATE_PREFIX = "automation:enabled:"
    REDIS_RUNNING_PREFIX = "automation:running:"
    REDIS_CACHE_TTL = 3600  # 1 hour cache TTL

    def __init__(
        self,
        redis_url: Optional[str] = None,
        db_config: Optional[Dict] = None
    ):
        """
        Initialize the control service.

        Args:
            redis_url: Redis connection URL (default: from CELERY_BROKER_URL)
            db_config: PostgreSQL connection config dict
        """
        # Redis connection
        redis_url = redis_url or os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            self._redis_available = True
            logger.info("AutomationControlService: Redis connected")
        except Exception as e:
            logger.warning(f"AutomationControlService: Redis not available ({e}), using DB only")
            self.redis = None
            self._redis_available = False

        # Database configuration
        self.db_config = db_config or {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "trading"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "")
        }

        # Connection pool for database
        try:
            self._db_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                **self.db_config
            )
            logger.info("AutomationControlService: Database pool initialized")
        except Exception as e:
            logger.error(f"AutomationControlService: Database pool failed ({e})")
            self._db_pool = None

        # Sync states to Redis on startup
        if self._redis_available:
            self._sync_states_to_redis()

    @contextmanager
    def _get_db_connection(self):
        """Get a database connection from the pool."""
        if not self._db_pool:
            raise Exception("Database pool not initialized")

        conn = self._db_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self._db_pool.putconn(conn)

    def _sync_states_to_redis(self) -> None:
        """Load all automation states from PostgreSQL into Redis cache."""
        if not self._redis_available:
            return

        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT name, is_enabled FROM automations")
                    rows = cur.fetchall()

                    pipe = self.redis.pipeline()
                    for name, is_enabled in rows:
                        key = f"{self.REDIS_STATE_PREFIX}{name}"
                        pipe.setex(key, self.REDIS_CACHE_TTL, "1" if is_enabled else "0")
                    pipe.execute()

            logger.info(f"AutomationControlService: Synced {len(rows)} automation states to Redis")

        except Exception as e:
            logger.error(f"AutomationControlService: Failed to sync states to Redis: {e}")

    def is_enabled(self, automation_name: str) -> bool:
        """
        Check if an automation is enabled.

        Uses Redis cache for O(1) lookup performance.
        Falls back to database if cache miss.

        Args:
            automation_name: The automation identifier (e.g., 'sync-kalshi-markets')

        Returns:
            True if enabled, False if disabled or not found
        """
        # Try Redis cache first
        if self._redis_available:
            try:
                key = f"{self.REDIS_STATE_PREFIX}{automation_name}"
                cached = self.redis.get(key)

                if cached is not None:
                    return cached == "1"
            except Exception as e:
                logger.warning(f"Redis lookup failed for {automation_name}: {e}")

        # Cache miss or Redis unavailable - check database
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT is_enabled FROM automations WHERE name = %s",
                        (automation_name,)
                    )
                    row = cur.fetchone()

                    if row:
                        is_enabled = row[0]

                        # Update Redis cache
                        if self._redis_available:
                            key = f"{self.REDIS_STATE_PREFIX}{automation_name}"
                            self.redis.setex(key, self.REDIS_CACHE_TTL, "1" if is_enabled else "0")

                        return is_enabled

                    # Not found - default to enabled (unknown tasks run by default)
                    logger.warning(f"Automation '{automation_name}' not found in database, defaulting to enabled")
                    return True

        except Exception as e:
            logger.error(f"Database lookup failed for {automation_name}: {e}")
            return True  # Fail open - don't break tasks on errors

    def set_enabled(
        self,
        automation_name: str,
        enabled: bool,
        revoke_running: bool = False,
        changed_by: str = "api",
        reason: Optional[str] = None
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Enable or disable an automation.

        Args:
            automation_name: The automation identifier
            enabled: True to enable, False to disable
            revoke_running: If disabling, also revoke currently running tasks
            changed_by: Who made this change (for audit log)
            reason: Optional reason for the change

        Returns:
            Tuple of (success, list of revoked task IDs if any)
        """
        revoked_tasks = []

        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Get current state
                    cur.execute(
                        "SELECT id, is_enabled FROM automations WHERE name = %s",
                        (automation_name,)
                    )
                    row = cur.fetchone()

                    if not row:
                        logger.warning(f"Automation not found: {automation_name}")
                        return False, None

                    automation_id, previous_state = row

                    # Skip if state unchanged
                    if previous_state == enabled:
                        logger.info(f"Automation '{automation_name}' already {'enabled' if enabled else 'disabled'}")
                        return True, []

                    # Update state
                    cur.execute("""
                        UPDATE automations
                        SET is_enabled = %s,
                            enabled_updated_at = NOW(),
                            enabled_updated_by = %s,
                            updated_at = NOW()
                        WHERE id = %s
                    """, (enabled, changed_by, automation_id))

                    # Log state change
                    cur.execute("""
                        INSERT INTO automation_state_log
                        (automation_id, previous_state, new_state, changed_by, reason)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (automation_id, previous_state, enabled, changed_by, reason))

                    log_id = cur.fetchone()[0]

                    # Handle running tasks if disabling
                    if not enabled and revoke_running:
                        revoked_tasks = self._revoke_running_tasks(automation_name)

                        if revoked_tasks:
                            cur.execute("""
                                UPDATE automation_state_log
                                SET affected_task_ids = %s
                                WHERE id = %s
                            """, (revoked_tasks, log_id))

                    conn.commit()

                    # Update Redis cache
                    if self._redis_available:
                        key = f"{self.REDIS_STATE_PREFIX}{automation_name}"
                        self.redis.setex(key, self.REDIS_CACHE_TTL, "1" if enabled else "0")

                    logger.info(
                        f"Automation '{automation_name}' {'enabled' if enabled else 'disabled'} "
                        f"by {changed_by}. Revoked tasks: {revoked_tasks}"
                    )

                    return True, revoked_tasks

        except Exception as e:
            logger.error(f"Error setting enabled state for {automation_name}: {e}")
            return False, None

    def _revoke_running_tasks(self, automation_name: str) -> List[str]:
        """
        Revoke any currently running Celery tasks for this automation.

        Returns list of revoked task IDs.
        """
        revoked = []

        if not self._redis_available:
            return revoked

        try:
            # Get running task IDs from Redis tracking
            running_key = f"{self.REDIS_RUNNING_PREFIX}{automation_name}"
            task_ids = self.redis.smembers(running_key)

            for task_id in task_ids:
                try:
                    # Import Celery app for revocation
                    from src.services.celery_app import app as celery_app
                    celery_app.control.revoke(task_id, terminate=True, signal='SIGTERM')
                    revoked.append(task_id)

                    # Mark as revoked in execution history
                    self._update_execution_status(task_id, ExecutionStatus.REVOKED)

                except Exception as e:
                    logger.warning(f"Failed to revoke task {task_id}: {e}")

            # Clear running tasks set
            if revoked:
                self.redis.delete(running_key)

        except Exception as e:
            logger.error(f"Error revoking running tasks for {automation_name}: {e}")

        return revoked

    def register_task_start(
        self,
        automation_name: str,
        celery_task_id: str,
        triggered_by: str = "scheduler"
    ) -> int:
        """
        Register that a task has started executing.

        Args:
            automation_name: The automation identifier
            celery_task_id: The Celery task UUID
            triggered_by: What triggered this execution

        Returns:
            Execution log ID (-1 on failure)
        """
        # Track in Redis for fast lookup
        if self._redis_available:
            try:
                running_key = f"{self.REDIS_RUNNING_PREFIX}{automation_name}"
                self.redis.sadd(running_key, celery_task_id)
                self.redis.expire(running_key, 3600)  # 1 hour TTL
            except Exception as e:
                logger.warning(f"Failed to track running task in Redis: {e}")

        # Record in database
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO automation_executions
                        (automation_id, celery_task_id, status, triggered_by, worker_hostname)
                        SELECT id, %s, 'running', %s, %s
                        FROM automations WHERE name = %s
                        RETURNING id
                    """, (
                        celery_task_id,
                        triggered_by,
                        os.getenv('HOSTNAME', 'unknown'),
                        automation_name
                    ))
                    result = cur.fetchone()
                    return result[0] if result else -1

        except Exception as e:
            logger.error(f"Error registering task start: {e}")
            return -1

    def register_task_complete(
        self,
        automation_name: str,
        celery_task_id: str,
        status: ExecutionStatus,
        result: Optional[Any] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
        records_processed: Optional[int] = None
    ) -> bool:
        """
        Register that a task has completed.

        Args:
            automation_name: The automation identifier
            celery_task_id: The Celery task UUID
            status: Final execution status
            result: Task result/return value
            error_message: Error message if failed
            error_traceback: Full traceback if failed
            records_processed: Number of records processed

        Returns:
            True if successfully recorded
        """
        # Remove from running set in Redis
        if self._redis_available:
            try:
                running_key = f"{self.REDIS_RUNNING_PREFIX}{automation_name}"
                self.redis.srem(running_key, celery_task_id)
            except Exception as e:
                logger.warning(f"Failed to remove task from Redis running set: {e}")

        # Update database
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE automation_executions
                        SET completed_at = NOW(),
                            duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at)),
                            status = %s,
                            result = %s,
                            error_message = %s,
                            error_traceback = %s,
                            records_processed = %s
                        WHERE celery_task_id = %s
                    """, (
                        status.value,
                        json.dumps(result) if result else None,
                        error_message,
                        error_traceback,
                        records_processed,
                        celery_task_id
                    ))
                    return True

        except Exception as e:
            logger.error(f"Error registering task complete: {e}")
            return False

    def _update_execution_status(self, celery_task_id: str, status: ExecutionStatus) -> None:
        """Update execution status by Celery task ID."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE automation_executions
                        SET status = %s, completed_at = NOW()
                        WHERE celery_task_id = %s
                    """, (status.value, celery_task_id))
        except Exception as e:
            logger.error(f"Error updating execution status: {e}")

    def get_all_automations(self, category: Optional[str] = None) -> List[Dict]:
        """
        Get all registered automations with their current state.

        Args:
            category: Optional filter by category

        Returns:
            List of automation dictionaries
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = "SELECT * FROM v_automation_status"
                    params = []

                    if category:
                        query += " WHERE category = %s"
                        params.append(category)

                    query += " ORDER BY category, name"

                    cur.execute(query, params)
                    return [dict(row) for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Error getting automations: {e}")
            return []

    def get_automation(self, automation_name: str) -> Optional[Dict]:
        """Get a single automation by name."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM v_automation_status WHERE name = %s",
                        (automation_name,)
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None

        except Exception as e:
            logger.error(f"Error getting automation {automation_name}: {e}")
            return None

    def get_execution_history(
        self,
        automation_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        since: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get execution history with optional filtering.

        Args:
            automation_name: Filter by specific automation
            status: Filter by status
            limit: Max records to return
            offset: Pagination offset
            since: Only executions after this time

        Returns:
            List of execution records
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT
                            e.*,
                            a.name as automation_name,
                            a.display_name,
                            a.category
                        FROM automation_executions e
                        JOIN automations a ON e.automation_id = a.id
                        WHERE 1=1
                    """
                    params = []

                    if automation_name:
                        query += " AND a.name = %s"
                        params.append(automation_name)

                    if status:
                        query += " AND e.status = %s"
                        params.append(status)

                    if since:
                        query += " AND e.started_at >= %s"
                        params.append(since)

                    query += " ORDER BY e.started_at DESC LIMIT %s OFFSET %s"
                    params.extend([limit, offset])

                    cur.execute(query, params)
                    return [dict(row) for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            return []

    def get_dashboard_stats(self, hours: int = 24) -> Dict:
        """
        Get dashboard statistics.

        Args:
            hours: Time window for statistics

        Returns:
            Statistics dictionary
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get automation counts
                    cur.execute("""
                        SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE is_enabled = true) as enabled,
                            COUNT(*) FILTER (WHERE is_enabled = false) as disabled
                        FROM automations
                    """)
                    automation_counts = dict(cur.fetchone())

                    # Get execution stats for time window
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_executions,
                            COUNT(*) FILTER (WHERE status = 'success') as successful,
                            COUNT(*) FILTER (WHERE status = 'failed') as failed,
                            COUNT(*) FILTER (WHERE status = 'skipped') as skipped,
                            COUNT(*) FILTER (WHERE status = 'running') as running,
                            ROUND(AVG(duration_seconds)::NUMERIC, 2) as avg_duration
                        FROM automation_executions
                        WHERE started_at > NOW() - INTERVAL '%s hours'
                    """ % hours)
                    execution_stats = dict(cur.fetchone())

                    # Calculate success rate
                    completed = (execution_stats['successful'] or 0) + (execution_stats['failed'] or 0)
                    success_rate = None
                    if completed > 0:
                        success_rate = round((execution_stats['successful'] / completed) * 100, 1)

                    # Get recent failures
                    cur.execute("""
                        SELECT
                            a.name,
                            a.display_name,
                            e.started_at,
                            e.error_message
                        FROM automation_executions e
                        JOIN automations a ON e.automation_id = a.id
                        WHERE e.status = 'failed'
                        AND e.started_at > NOW() - INTERVAL '%s hours'
                        ORDER BY e.started_at DESC
                        LIMIT 5
                    """ % hours)
                    recent_failures = [dict(row) for row in cur.fetchall()]

                    # Get category breakdown
                    cur.execute("""
                        SELECT
                            category,
                            COUNT(*) as count,
                            COUNT(*) FILTER (WHERE is_enabled) as enabled_count
                        FROM automations
                        GROUP BY category
                        ORDER BY category
                    """)
                    categories = [dict(row) for row in cur.fetchall()]

                    return {
                        "automations": automation_counts,
                        "executions": {
                            **execution_stats,
                            "success_rate": success_rate
                        },
                        "recent_failures": recent_failures,
                        "categories": categories,
                        "time_window_hours": hours
                    }

        except Exception as e:
            logger.error(f"Error getting dashboard stats: {e}")
            return {}

    def get_categories(self) -> List[str]:
        """Get all unique automation categories."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT category FROM automations ORDER BY category
                    """)
                    return [row[0] for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def bulk_set_enabled(
        self,
        automation_names: List[str],
        enabled: bool,
        changed_by: str = "api",
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enable or disable multiple automations at once.

        Args:
            automation_names: List of automation names
            enabled: True to enable, False to disable
            changed_by: Who made this change
            reason: Optional reason

        Returns:
            Summary of updates
        """
        results = {
            "updated": [],
            "failed": [],
            "unchanged": []
        }

        for name in automation_names:
            success, _ = self.set_enabled(
                automation_name=name,
                enabled=enabled,
                revoke_running=False,
                changed_by=changed_by,
                reason=reason
            )

            if success:
                results["updated"].append(name)
            else:
                results["failed"].append(name)

        return results

    def trigger_automation(self, automation_name: str) -> Optional[str]:
        """
        Manually trigger an automation.

        Args:
            automation_name: The automation to trigger

        Returns:
            Celery task ID if triggered, None on failure
        """
        try:
            # Get automation details
            automation = self.get_automation(automation_name)
            if not automation:
                logger.error(f"Automation not found: {automation_name}")
                return None

            if not automation['is_enabled']:
                logger.warning(f"Cannot trigger disabled automation: {automation_name}")
                return None

            celery_task_name = automation.get('celery_task_name')
            if not celery_task_name:
                logger.error(f"No Celery task for automation: {automation_name}")
                return None

            # Import and trigger the task
            from src.services.celery_app import app as celery_app
            result = celery_app.send_task(celery_task_name)

            logger.info(f"Triggered automation '{automation_name}' with task ID: {result.id}")
            return result.id

        except Exception as e:
            logger.error(f"Error triggering automation {automation_name}: {e}")
            return None


# Global singleton instance
_control_service: Optional[AutomationControlService] = None


@lru_cache(maxsize=1)
def get_automation_control_service() -> AutomationControlService:
    """Get the global automation control service instance."""
    global _control_service
    if _control_service is None:
        _control_service = AutomationControlService()
    return _control_service


def reset_automation_control_service() -> None:
    """Reset the global instance (for testing)."""
    global _control_service
    _control_service = None
    get_automation_control_service.cache_clear()
