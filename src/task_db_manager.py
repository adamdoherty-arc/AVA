"""
Task Management Database Manager
=================================

Manages CRUD operations for the Task Management System tables:
- development_tasks: Core task tracking
- task_execution_log: Execution audit trail
- task_verification: QA and user feedback
- task_files: File change tracking

Usage:
    from src.task_db_manager import TaskDBManager

    task_mgr = TaskDBManager()

    # Create a new task
    task_id = task_mgr.create_task(
        title="Fix CSP opportunities display",
        description="Update dashboard to show CSP opportunities correctly",
        task_type="bug_fix",
        priority="high",
        assigned_agent="backend-architect",
        feature_area="dashboard"
    )

    # Update task status
    task_mgr.update_task_status(task_id, "in_progress")

    # Log execution activity
    task_mgr.log_execution(
        task_id=task_id,
        agent_name="backend-architect",
        action_type="progress_update",
        message="Fixed database query, updating UI components",
        files_modified=["dashboard.py", "src/db_manager.py"]
    )

    # Complete task
    task_mgr.complete_task(task_id)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


class TaskDBManager:
    """Manages Task Management System data in PostgreSQL database"""

    def __init__(self) -> None:
        """Initialize database connection configuration"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD')
        }

    def get_connection(self) -> None:
        """Create and return a database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    # ========================================================================
    # TASK CRUD OPERATIONS
    # ========================================================================

    def create_task(
        self,
        title: str,
        description: str = None,
        task_type: str = "feature",
        priority: str = "medium",
        assigned_agent: str = None,
        feature_area: str = None,
        estimated_duration_minutes: int = None,
        dependencies: List[int] = None,
        parent_task_id: int = None,
        tags: List[str] = None,
        created_by: str = "user"
    ) -> int:
        """
        Create a new development task

        Args:
            title: Task title (required)
            description: Detailed task description
            task_type: Type of task (bug_fix, feature, enhancement, qa, refactor, documentation, investigation)
            priority: Priority level (low, medium, high, critical)
            assigned_agent: Agent to handle the task
            feature_area: System area (comprehensive_strategy, dashboard, xtrades, kalshi, ava, options_flow)
            estimated_duration_minutes: Estimated time to complete
            dependencies: List of task IDs this task depends on
            parent_task_id: Parent task ID if this is a subtask
            tags: List of tags for categorization
            created_by: Who created the task

        Returns:
            int: New task ID
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO development_tasks (
                        title, description, task_type, priority, assigned_agent,
                        feature_area, estimated_duration_minutes, dependencies,
                        parent_task_id, tags, created_by
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id
                """, (
                    title, description, task_type, priority, assigned_agent,
                    feature_area, estimated_duration_minutes, dependencies,
                    parent_task_id, tags, created_by
                ))

                task_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Created task {task_id}: {title}")
                return task_id

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error creating task: {e}")
            raise
        finally:
            conn.close()

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get task details by ID

        Args:
            task_id: Task ID

        Returns:
            Dict with task details or None if not found
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM development_tasks
                    WHERE id = %s
                """, (task_id,))

                task = cursor.fetchone()
                return dict(task) if task else None

        except psycopg2.Error as e:
            logger.error(f"Error fetching task {task_id}: {e}")
            raise
        finally:
            conn.close()

    def update_task_status(
        self,
        task_id: int,
        status: str,
        blocked_reason: str = None
    ) -> bool:
        """
        Update task status

        Args:
            task_id: Task ID
            status: New status (pending, in_progress, completed, failed, blocked, cancelled)
            blocked_reason: Reason if status is 'blocked'

        Returns:
            bool: True if updated successfully
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE development_tasks
                    SET status = %s,
                        blocked_reason = %s
                    WHERE id = %s
                """, (status, blocked_reason, task_id))

                conn.commit()
                logger.info(f"Updated task {task_id} status to {status}")
                return cursor.rowcount > 0

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error updating task {task_id} status: {e}")
            raise
        finally:
            conn.close()

    def complete_task(self, task_id: int) -> bool:
        """
        Mark task as completed (convenience method)

        Args:
            task_id: Task ID

        Returns:
            bool: True if updated successfully
        """
        return self.update_task_status(task_id, "completed")

    def get_active_tasks(
        self,
        assigned_agent: str = None,
        feature_area: str = None,
        priority: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get all active tasks (pending, in_progress, blocked)

        Args:
            assigned_agent: Filter by assigned agent (optional)
            feature_area: Filter by feature area (optional)
            priority: Filter by priority (optional)

        Returns:
            List of task dictionaries
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT * FROM v_active_tasks
                    WHERE 1=1
                """
                params = []

                if assigned_agent:
                    query += " AND assigned_agent = %s"
                    params.append(assigned_agent)

                if feature_area:
                    query += " AND feature_area = %s"
                    params.append(feature_area)

                if priority:
                    query += " AND priority = %s"
                    params.append(priority)

                query += " ORDER BY priority, created_at"

                cursor.execute(query, params)
                tasks = cursor.fetchall()

                return [dict(task) for task in tasks]

        except psycopg2.Error as e:
            logger.error(f"Error fetching active tasks: {e}")
            raise
        finally:
            conn.close()

    def get_tasks_by_status(
        self,
        status: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get tasks by status

        Args:
            status: Task status to filter by
            limit: Maximum number of tasks to return

        Returns:
            List of task dictionaries
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM development_tasks
                    WHERE status = %s
                    ORDER BY
                        CASE priority
                            WHEN 'critical' THEN 1
                            WHEN 'high' THEN 2
                            WHEN 'medium' THEN 3
                            WHEN 'low' THEN 4
                        END,
                        created_at DESC
                    LIMIT %s
                """, (status, limit))

                tasks = cursor.fetchall()
                return [dict(task) for task in tasks]

        except psycopg2.Error as e:
            logger.error(f"Error fetching tasks by status: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # EXECUTION LOGGING
    # ========================================================================

    def log_execution(
        self,
        task_id: int,
        agent_name: str,
        action_type: str,
        message: str = None,
        files_modified: List[str] = None,
        error_details: str = None,
        duration_seconds: int = None
    ) -> int:
        """
        Log a task execution action

        Args:
            task_id: Task ID
            agent_name: Name of agent performing action
            action_type: Type of action (started, progress_update, completed, failed, etc.)
            message: Description of action
            files_modified: List of file paths modified
            error_details: Error message if action failed
            duration_seconds: Duration of action

        Returns:
            int: Log entry ID
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO task_execution_log (
                        task_id, agent_name, action_type, message,
                        files_modified, error_details, duration_seconds
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id
                """, (
                    task_id, agent_name, action_type, message,
                    files_modified, error_details, duration_seconds
                ))

                log_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Logged {action_type} for task {task_id} by {agent_name}")
                return log_id

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error logging execution: {e}")
            raise
        finally:
            conn.close()

    def get_execution_log(
        self,
        task_id: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get execution log for a task

        Args:
            task_id: Task ID
            limit: Maximum number of log entries

        Returns:
            List of log entry dictionaries
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM task_execution_log
                    WHERE task_id = %s
                    ORDER BY execution_timestamp DESC
                    LIMIT %s
                """, (task_id, limit))

                logs = cursor.fetchall()
                return [dict(log) for log in logs]

        except psycopg2.Error as e:
            logger.error(f"Error fetching execution log: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # VERIFICATION & FEEDBACK
    # ========================================================================

    def add_verification(
        self,
        task_id: int,
        verified_by: str,
        passed: bool,
        verification_notes: str = None,
        test_results: Dict[str, Any] = None,
        user_feedback: str = None,
        user_comments: str = None
    ) -> int:
        """
        Add verification result for a task

        Args:
            task_id: Task ID
            verified_by: Who performed verification (qa_agent, user, automated_test)
            passed: True if verification passed
            verification_notes: Detailed findings
            test_results: Structured test results (stored as JSONB)
            user_feedback: User decision (approved, rejected, work_again, needs_changes)
            user_comments: User's comments

        Returns:
            int: Verification ID
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO task_verification (
                        task_id, verified_by, passed, verification_notes,
                        test_results, user_feedback, user_comments
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id
                """, (
                    task_id, verified_by, passed, verification_notes,
                    psycopg2.extras.Json(test_results) if test_results else None,
                    user_feedback, user_comments
                ))

                verification_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Added verification for task {task_id}: passed={passed}")
                return verification_id

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error adding verification: {e}")
            raise
        finally:
            conn.close()

    def get_verifications(self, task_id: int) -> List[Dict[str, Any]]:
        """
        Get all verifications for a task

        Args:
            task_id: Task ID

        Returns:
            List of verification dictionaries
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM task_verification
                    WHERE task_id = %s
                    ORDER BY verification_timestamp DESC
                """, (task_id,))

                verifications = cursor.fetchall()
                return [dict(v) for v in verifications]

        except psycopg2.Error as e:
            logger.error(f"Error fetching verifications: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # FILE TRACKING
    # ========================================================================

    def track_file_change(
        self,
        task_id: int,
        file_path: str,
        change_type: str,
        lines_added: int = 0,
        lines_removed: int = 0,
        file_size_bytes: int = None,
        git_commit_hash: str = None
    ) -> int:
        """
        Track a file change for a task

        Args:
            task_id: Task ID
            file_path: Absolute file path
            change_type: Type of change (created, modified, deleted, renamed)
            lines_added: Number of lines added
            lines_removed: Number of lines removed
            file_size_bytes: File size after modification
            git_commit_hash: Git commit SHA

        Returns:
            int: File tracking ID
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO task_files (
                        task_id, file_path, change_type, lines_added,
                        lines_removed, file_size_bytes, git_commit_hash
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id
                """, (
                    task_id, file_path, change_type, lines_added,
                    lines_removed, file_size_bytes, git_commit_hash
                ))

                file_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Tracked file {change_type}: {file_path} for task {task_id}")
                return file_id

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error tracking file change: {e}")
            raise
        finally:
            conn.close()

    def get_task_files(self, task_id: int) -> List[Dict[str, Any]]:
        """
        Get all file changes for a task

        Args:
            task_id: Task ID

        Returns:
            List of file change dictionaries
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM task_files
                    WHERE task_id = %s
                    ORDER BY modified_at DESC
                """, (task_id,))

                files = cursor.fetchall()
                return [dict(f) for f in files]

        except psycopg2.Error as e:
            logger.error(f"Error fetching task files: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # ANALYTICS & REPORTING
    # ========================================================================

    def get_feature_progress(self) -> List[Dict[str, Any]]:
        """
        Get progress statistics for all feature areas

        Returns:
            List of feature area statistics
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM v_feature_progress")
                features = cursor.fetchall()
                return [dict(f) for f in features]

        except psycopg2.Error as e:
            logger.error(f"Error fetching feature progress: {e}")
            raise
        finally:
            conn.close()

    def get_agent_workload(self) -> List[Dict[str, Any]]:
        """
        Get workload statistics for all agents

        Returns:
            List of agent workload statistics
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM v_agent_workload")
                agents = cursor.fetchall()
                return [dict(a) for a in agents]

        except psycopg2.Error as e:
            logger.error(f"Error fetching agent workload: {e}")
            raise
        finally:
            conn.close()

    def get_task_metrics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get performance metrics for completed tasks

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of task metrics
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM v_task_metrics
                    LIMIT %s
                """, (limit,))

                metrics = cursor.fetchall()
                return [dict(m) for m in metrics]

        except psycopg2.Error as e:
            logger.error(f"Error fetching task metrics: {e}")
            raise
        finally:
            conn.close()

    def check_dependencies(self, task_id: int) -> bool:
        """
        Check if all dependencies for a task are completed

        Args:
            task_id: Task ID

        Returns:
            bool: True if all dependencies are completed
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT check_task_dependencies(%s)
                """, (task_id,))

                result = cursor.fetchone()[0]
                return result

        except psycopg2.Error as e:
            logger.error(f"Error checking dependencies: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def search_tasks(
        self,
        search_term: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Full-text search tasks by title and description

        Args:
            search_term: Search query
            limit: Maximum number of results

        Returns:
            List of matching tasks
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT *,
                        ts_rank(
                            to_tsvector('english', title || ' ' || COALESCE(description, '')),
                            plainto_tsquery('english', %s)
                        ) AS rank
                    FROM development_tasks
                    WHERE to_tsvector('english', title || ' ' || COALESCE(description, ''))
                        @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, (search_term, search_term, limit))

                tasks = cursor.fetchall()
                return [dict(task) for task in tasks]

        except psycopg2.Error as e:
            logger.error(f"Error searching tasks: {e}")
            raise
        finally:
            conn.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Testing Task Management Database Manager...\n")

    try:
        # Initialize manager
        task_mgr = TaskDBManager()

        # Test 1: Create a task
        print("Test 1: Creating a task...")
        task_id = task_mgr.create_task(
            title="Optimize database queries for dashboard",
            description="Analyze and optimize slow queries in positions page",
            task_type="enhancement",
            priority="high",
            assigned_agent="database-optimizer",
            feature_area="dashboard",
            estimated_duration_minutes=120,
            tags=["performance", "database", "dashboard"]
        )
        print(f"   Created task ID: {task_id}")

        # Test 2: Update task status
        print("\nTest 2: Starting task...")
        task_mgr.update_task_status(task_id, "in_progress")

        # Test 3: Log execution
        print("\nTest 3: Logging execution activity...")
        task_mgr.log_execution(
            task_id=task_id,
            agent_name="database-optimizer",
            action_type="progress_update",
            message="Analyzed EXPLAIN plans, identified missing indexes",
            files_modified=["src/db_manager.py", "dashboard.py"]
        )

        # Test 4: Track file changes
        print("\nTest 4: Tracking file changes...")
        task_mgr.track_file_change(
            task_id=task_id,
            file_path="C:/Code/WheelStrategy/src/db_manager.py",
            change_type="modified",
            lines_added=25,
            lines_removed=10
        )

        # Test 5: Get active tasks
        print("\nTest 5: Fetching active tasks...")
        active_tasks = task_mgr.get_active_tasks()
        print(f"   Found {len(active_tasks)} active tasks")

        # Test 6: Complete task
        print("\nTest 6: Completing task...")
        task_mgr.complete_task(task_id)

        # Test 7: Add verification
        print("\nTest 7: Adding verification...")
        task_mgr.add_verification(
            task_id=task_id,
            verified_by="qa_agent",
            passed=True,
            verification_notes="All queries optimized, performance improved by 40%",
            test_results={
                "queries_tested": 5,
                "avg_speedup": "40%",
                "all_tests_passed": True
            },
            user_feedback="approved"
        )

        # Test 8: Get task details
        print("\nTest 8: Getting task details...")
        task = task_mgr.get_task(task_id)
        print(f"   Task: {task['title']}")
        print(f"   Status: {task['status']}")
        print(f"   Duration: {task['actual_duration_minutes']} minutes")

        # Test 9: Get feature progress
        print("\nTest 9: Getting feature progress...")
        progress = task_mgr.get_feature_progress()
        for feature in progress:
            print(f"   {feature['feature_area']}: {feature['completion_percentage']}% complete")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
