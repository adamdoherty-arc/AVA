"""
Task Manager - Database interface for development task management
Handles all database operations for the Enhancement & Task Manager system
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv()


class TaskManager:
    """Manages development tasks with PostgreSQL database backend"""

    def __init__(self) -> None:
        """Initialize task manager with database connection parameters"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        self.conn = None
        self.cursor = None

    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def is_connected(self) -> bool:
        """Check if database connection is active"""
        if not self.conn:
            return False
        try:
            self.conn.isolation_level
            return True
        except:
            return False

    # ========================================================================
    # TASK CRUD OPERATIONS
    # ========================================================================

    def create_task(
        self,
        title: str,
        description: str,
        task_type: str,
        priority: str,
        assigned_agent: str,
        feature_area: str,
        estimated_duration_minutes: Optional[int] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[int]] = None,
        parent_task_id: Optional[int] = None
    ) -> Optional[int]:
        """Create a new development task"""
        try:
            query = """
                INSERT INTO development_tasks (
                    title, description, task_type, priority, assigned_agent,
                    feature_area, estimated_duration_minutes, tags, dependencies,
                    parent_task_id, created_by
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNS id;
            """

            self.cursor.execute(query, (
                title, description, task_type, priority, assigned_agent,
                feature_area, estimated_duration_minutes, tags, dependencies,
                parent_task_id, 'user'
            ))

            result = self.cursor.fetchone()
            task_id = result['id'] if result else None

            self.conn.commit()
            return task_id

        except Exception as e:
            print(f"Error creating task: {e}")
            self.conn.rollback()
            return None

    def fetch_tasks(
        self,
        status: Optional[List[str]] = None,
        priority: Optional[List[str]] = None,
        feature_area: Optional[str] = None,
        assigned_agent: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Fetch tasks with optional filters"""
        try:
            query = """
                SELECT
                    id, title, description, task_type, priority, status,
                    assigned_agent, feature_area, created_at, started_at,
                    updated_at, completed_at, estimated_duration_minutes,
                    actual_duration_minutes, dependencies, parent_task_id,
                    tags, blocked_reason, created_by
                FROM development_tasks
                WHERE 1=1
            """

            params = []

            if status:
                query += f" AND status = ANY(%s)"
                params.append(status)

            if priority:
                query += f" AND priority = ANY(%s)"
                params.append(priority)

            if feature_area:
                query += f" AND feature_area = %s"
                params.append(feature_area)

            if assigned_agent:
                query += f" AND assigned_agent = %s"
                params.append(assigned_agent)

            query += """
                ORDER BY
                    CASE priority
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                    END,
                    created_at DESC
                LIMIT %s
            """
            params.append(limit)

            self.cursor.execute(query, tuple(params))
            tasks = self.cursor.fetchall()

            return [dict(task) for task in tasks]

        except Exception as e:
            print(f"Error fetching tasks: {e}")
            return []

    def get_task_by_id(self, task_id: int) -> Optional[Dict]:
        """Fetch a single task by ID"""
        try:
            query = """
                SELECT
                    id, title, description, task_type, priority, status,
                    assigned_agent, feature_area, created_at, started_at,
                    updated_at, completed_at, estimated_duration_minutes,
                    actual_duration_minutes, dependencies, parent_task_id,
                    tags, blocked_reason, created_by
                FROM development_tasks
                WHERE id = %s
            """

            self.cursor.execute(query, (task_id,))
            task = self.cursor.fetchone()

            return dict(task) if task else None

        except Exception as e:
            print(f"Error fetching task {task_id}: {e}")
            return None

    def update_task_status(
        self,
        task_id: int,
        new_status: str,
        message: Optional[str] = None
    ) -> bool:
        """Update task status"""
        try:
            query = """
                UPDATE development_tasks
                SET status = %s,
                    updated_at = NOW()
                WHERE id = %s
            """

            self.cursor.execute(query, (new_status, task_id))
            self.conn.commit()

            # Log status change
            if message:
                self.log_execution(
                    task_id=task_id,
                    agent_name='user',
                    action_type='progress_update',
                    message=message
                )

            return True

        except Exception as e:
            print(f"Error updating task status: {e}")
            self.conn.rollback()
            return False

    def update_task(
        self,
        task_id: int,
        **kwargs
    ) -> bool:
        """Update task fields"""
        try:
            # Build dynamic update query
            update_fields = []
            params = []

            for field, value in kwargs.items():
                update_fields.append(f"{field} = %s")
                params.append(value)

            if not update_fields:
                return False

            update_fields.append("updated_at = NOW()")
            params.append(task_id)

            query = f"""
                UPDATE development_tasks
                SET {', '.join(update_fields)}
                WHERE id = %s
            """

            self.cursor.execute(query, tuple(params))
            self.conn.commit()

            return True

        except Exception as e:
            print(f"Error updating task: {e}")
            self.conn.rollback()
            return False

    def delete_task(self, task_id: int) -> bool:
        """Delete a task (use with caution - cascades to logs, files, verification)"""
        try:
            query = "DELETE FROM development_tasks WHERE id = %s"
            self.cursor.execute(query, (task_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting task: {e}")
            self.conn.rollback()
            return False

    # ========================================================================
    # TASK EXECUTION LOGGING
    # ========================================================================

    def log_execution(
        self,
        task_id: int,
        agent_name: str,
        action_type: str,
        message: str,
        files_modified: Optional[List[str]] = None,
        error_details: Optional[str] = None,
        duration_seconds: Optional[int] = None
    ) -> bool:
        """Log task execution activity"""
        try:
            query = """
                INSERT INTO task_execution_log (
                    task_id, agent_name, action_type, message,
                    files_modified, error_details, duration_seconds
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            self.cursor.execute(query, (
                task_id, agent_name, action_type, message,
                files_modified, error_details, duration_seconds
            ))

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error logging execution: {e}")
            self.conn.rollback()
            return False

    def fetch_task_logs(
        self,
        task_id: int,
        limit: int = 50
    ) -> List[Dict]:
        """Fetch execution logs for a task"""
        try:
            query = """
                SELECT
                    id, task_id, execution_timestamp, agent_name,
                    action_type, message, files_modified, error_details,
                    duration_seconds
                FROM task_execution_log
                WHERE task_id = %s
                ORDER BY execution_timestamp DESC
                LIMIT %s
            """

            self.cursor.execute(query, (task_id, limit))
            logs = self.cursor.fetchall()

            return [dict(log) for log in logs]

        except Exception as e:
            print(f"Error fetching task logs: {e}")
            return []

    # ========================================================================
    # TASK VERIFICATION
    # ========================================================================

    def create_verification(
        self,
        task_id: int,
        verified_by: str,
        passed: bool,
        verification_notes: str,
        test_results: Optional[Dict] = None
    ) -> bool:
        """Create task verification record"""
        try:
            query = """
                INSERT INTO task_verification (
                    task_id, verified_by, passed, verification_notes, test_results
                )
                VALUES (%s, %s, %s, %s, %s)
            """

            test_results_json = json.dumps(test_results) if test_results else None

            self.cursor.execute(query, (
                task_id, verified_by, passed, verification_notes, test_results_json
            ))

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error creating verification: {e}")
            self.conn.rollback()
            return False

    def fetch_task_verification(self, task_id: int) -> Optional[Dict]:
        """Fetch latest verification for a task"""
        try:
            query = """
                SELECT
                    id, task_id, verified_by, verification_timestamp,
                    passed, verification_notes, test_results,
                    user_feedback, user_comments
                FROM task_verification
                WHERE task_id = %s
                ORDER BY verification_timestamp DESC
                LIMIT 1
            """

            self.cursor.execute(query, (task_id,))
            verification = self.cursor.fetchone()

            return dict(verification) if verification else None

        except Exception as e:
            print(f"Error fetching verification: {e}")
            return None

    def save_user_feedback(
        self,
        task_id: int,
        user_feedback: str,
        user_comments: Optional[str] = None
    ) -> bool:
        """Save user feedback on task verification"""
        try:
            query = """
                UPDATE task_verification
                SET user_feedback = %s,
                    user_comments = %s
                WHERE task_id = %s
                AND id = (
                    SELECT id FROM task_verification
                    WHERE task_id = %s
                    ORDER BY verification_timestamp DESC
                    LIMIT 1
                )
            """

            self.cursor.execute(query, (
                user_feedback, user_comments, task_id, task_id
            ))

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error saving user feedback: {e}")
            self.conn.rollback()
            return False

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
        file_size_bytes: Optional[int] = None,
        git_commit_hash: Optional[str] = None
    ) -> bool:
        """Track file changes for a task"""
        try:
            query = """
                INSERT INTO task_files (
                    task_id, file_path, change_type, lines_added,
                    lines_removed, file_size_bytes, git_commit_hash
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            self.cursor.execute(query, (
                task_id, file_path, change_type, lines_added,
                lines_removed, file_size_bytes, git_commit_hash
            ))

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error tracking file change: {e}")
            self.conn.rollback()
            return False

    def fetch_task_files(self, task_id: int) -> List[Dict]:
        """Fetch file changes for a task"""
        try:
            query = """
                SELECT
                    id, task_id, file_path, change_type,
                    lines_added, lines_removed, file_size_bytes,
                    git_commit_hash, modified_at
                FROM task_files
                WHERE task_id = %s
                ORDER BY modified_at DESC
            """

            self.cursor.execute(query, (task_id,))
            files = self.cursor.fetchall()

            return [dict(file) for file in files]

        except Exception as e:
            print(f"Error fetching task files: {e}")
            return []

    # ========================================================================
    # ANALYTICS & METRICS
    # ========================================================================

    def get_overall_metrics(self) -> Dict:
        """Get overall task system metrics"""
        try:
            query = """
                SELECT
                    COUNT(*) AS total_tasks,
                    COUNT(*) FILTER (WHERE status = 'completed') AS completed_tasks,
                    COUNT(*) FILTER (WHERE status = 'in_progress') AS in_progress_tasks,
                    COUNT(*) FILTER (WHERE status = 'pending') AS pending_tasks,
                    COUNT(*) FILTER (WHERE status = 'blocked') AS blocked_tasks,
                    COUNT(*) FILTER (WHERE status = 'failed') AS failed_tasks
                FROM development_tasks
                WHERE status != 'cancelled'
            """

            self.cursor.execute(query)
            metrics = self.cursor.fetchone()

            return dict(metrics) if metrics else {}

        except Exception as e:
            print(f"Error fetching overall metrics: {e}")
            return {}

    def get_status_distribution(self) -> Dict[str, int]:
        """Get task count by status"""
        try:
            query = """
                SELECT status, COUNT(*) AS count
                FROM development_tasks
                WHERE status != 'cancelled'
                GROUP BY status
                ORDER BY count DESC
            """

            self.cursor.execute(query)
            results = self.cursor.fetchall()

            return {row['status']: row['count'] for row in results}

        except Exception as e:
            print(f"Error fetching status distribution: {e}")
            return {}

    def get_priority_distribution(self) -> Dict[str, int]:
        """Get task count by priority"""
        try:
            query = """
                SELECT priority, COUNT(*) AS count
                FROM development_tasks
                WHERE status NOT IN ('cancelled', 'completed')
                GROUP BY priority
                ORDER BY
                    CASE priority
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                    END
            """

            self.cursor.execute(query)
            results = self.cursor.fetchall()

            return {row['priority']: row['count'] for row in results}

        except Exception as e:
            print(f"Error fetching priority distribution: {e}")
            return {}

    def get_feature_progress(self) -> List[Dict]:
        """Get progress by feature area"""
        try:
            query = """
                SELECT * FROM v_feature_progress
                ORDER BY completion_percentage ASC, total_tasks DESC
            """

            self.cursor.execute(query)
            results = self.cursor.fetchall()

            return [dict(row) for row in results]

        except Exception as e:
            print(f"Error fetching feature progress: {e}")
            return []

    def get_agent_workload(self) -> List[Dict]:
        """Get workload distribution by agent"""
        try:
            query = """
                SELECT * FROM v_agent_workload
                ORDER BY active_tasks DESC, total_assigned_tasks DESC
            """

            self.cursor.execute(query)
            results = self.cursor.fetchall()

            return [dict(row) for row in results]

        except Exception as e:
            print(f"Error fetching agent workload: {e}")
            return []

    def get_feature_areas(self) -> List[str]:
        """Get list of unique feature areas"""
        try:
            query = """
                SELECT DISTINCT feature_area
                FROM development_tasks
                WHERE feature_area IS NOT NULL
                ORDER BY feature_area
            """

            self.cursor.execute(query)
            results = self.cursor.fetchall()

            return [row['feature_area'] for row in results]

        except Exception as e:
            print(f"Error fetching feature areas: {e}")
            return []

    # ========================================================================
    # SYSTEM UTILITIES
    # ========================================================================

    def get_table_count(self) -> int:
        """Get count of task management tables"""
        try:
            query = """
                SELECT COUNT(*) AS count
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN (
                    'development_tasks', 'task_execution_log',
                    'task_verification', 'task_files'
                )
            """

            self.cursor.execute(query)
            result = self.cursor.fetchone()

            return result['count'] if result else 0

        except Exception as e:
            print(f"Error fetching table count: {e}")
            return 0

    def get_last_activity_time(self) -> Optional[datetime]:
        """Get timestamp of last task activity"""
        try:
            query = """
                SELECT MAX(updated_at) AS last_activity
                FROM development_tasks
            """

            self.cursor.execute(query)
            result = self.cursor.fetchone()

            return result['last_activity'] if result else None

        except Exception as e:
            print(f"Error fetching last activity time: {e}")
            return None

    def clean_old_logs(self, days: int = 30) -> int:
        """Clean execution logs older than specified days"""
        try:
            query = """
                DELETE FROM task_execution_log
                WHERE execution_timestamp < NOW() - INTERVAL '%s days'
            """

            self.cursor.execute(query, (days,))
            deleted_count = self.cursor.rowcount

            self.conn.commit()
            return deleted_count

        except Exception as e:
            print(f"Error cleaning old logs: {e}")
            self.conn.rollback()
            return -1

    def refresh_analytics_views(self) -> bool:
        """Refresh materialized views for analytics (if any)"""
        try:
            # This would refresh any materialized views if we had them
            # For now, just return success since we're using regular views
            return True

        except Exception as e:
            print(f"Error refreshing analytics: {e}")
            return False

    def verify_schema(self) -> bool:
        """Verify that all required tables exist"""
        try:
            required_tables = [
                'development_tasks',
                'task_execution_log',
                'task_verification',
                'task_files'
            ]

            query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = ANY(%s)
            """

            self.cursor.execute(query, (required_tables,))
            existing_tables = [row['table_name'] for row in self.cursor.fetchall()]

            return len(existing_tables) == len(required_tables)

        except Exception as e:
            print(f"Error verifying schema: {e}")
            return False

    # ========================================================================
    # TASK DEPENDENCIES
    # ========================================================================

    def check_dependencies_met(self, task_id: int) -> bool:
        """Check if all task dependencies are completed"""
        try:
            query = "SELECT check_task_dependencies(%s) AS met"
            self.cursor.execute(query, (task_id,))
            result = self.cursor.fetchone()

            return result['met'] if result else False

        except Exception as e:
            print(f"Error checking dependencies: {e}")
            return False

    def get_task_dependencies(self, task_id: int) -> List[Dict]:
        """Get detailed info about task dependencies"""
        try:
            # First get the dependency IDs
            query = """
                SELECT dependencies
                FROM development_tasks
                WHERE id = %s
            """

            self.cursor.execute(query, (task_id,))
            result = self.cursor.fetchone()

            if not result or not result['dependencies']:
                return []

            dependency_ids = result['dependencies']

            # Now get full details of dependency tasks
            query = """
                SELECT id, title, status, priority, completed_at
                FROM development_tasks
                WHERE id = ANY(%s)
            """

            self.cursor.execute(query, (dependency_ids,))
            dependencies = self.cursor.fetchall()

            return [dict(dep) for dep in dependencies]

        except Exception as e:
            print(f"Error fetching task dependencies: {e}")
            return []


# Convenience function for one-off operations
def execute_task_query(query: str, params: tuple = None) -> Any:
    """Execute a custom query (use with caution)"""
    manager = TaskManager()
    if not manager.connect():
        return None

    try:
        manager.cursor.execute(query, params)
        if query.strip().upper().startswith('SELECT'):
            results = manager.cursor.fetchall()
            return [dict(row) for row in results]
        else:
            manager.conn.commit()
            return manager.cursor.rowcount
    except Exception as e:
        print(f"Query execution error: {e}")
        manager.conn.rollback()
        return None
    finally:
        manager.disconnect()
