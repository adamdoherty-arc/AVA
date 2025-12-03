"""
QA Issues Database Manager
==========================

Manages CRUD operations for the QA Issues Tracking System tables:
- qa_runs: QA cycle execution tracking
- qa_check_results: Individual check results per run
- qa_issues: Issue tracking with deduplication
- qa_issue_occurrences: Issue-to-run linking
- qa_fixes: Fix tracking
- qa_health_history: Health score trending
- qa_hot_spots: File hot spot tracking

Usage:
    from src.qa_issues_db_manager import QAIssuesDBManager

    qa_mgr = QAIssuesDBManager()

    # Start a QA run
    run_db_id = qa_mgr.start_run(run_id="qa_20251126_120000")

    # Log a check result
    qa_mgr.log_check_result(
        run_id=run_db_id,
        module_name="api_endpoints",
        check_name="server_available",
        status="passed",
        message="API server responding at http://localhost:8002"
    )

    # Log an issue
    issue_id = qa_mgr.log_issue(
        run_id=run_db_id,
        module_name="api_endpoints",
        check_name="endpoint_responses",
        title="Endpoint /api/options/analysis returning 500",
        severity="high",
        category="api"
    )

    # Complete the run
    qa_mgr.complete_run(run_db_id, health_score=85.5)
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


class QAIssuesDBManager:
    """Manages QA Issues Tracking System data in PostgreSQL database"""

    def __init__(self) -> None:
        """Initialize database connection configuration"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD')
        }
        self._connection = None

    def get_connection(self):
        """Create and return a database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _calculate_issue_hash(self, module_name: str, check_name: str, title: str) -> str:
        """Calculate SHA256 hash for issue deduplication"""
        content = f"{module_name}::{check_name}::{title}"
        return hashlib.sha256(content.encode()).hexdigest()

    # ========================================================================
    # QA RUNS OPERATIONS
    # ========================================================================

    def start_run(
        self,
        run_id: str,
        triggered_by: str = "scheduler"
    ) -> int:
        """
        Start a new QA run

        Args:
            run_id: Unique identifier for this run (e.g., "qa_20251126_120000")
            triggered_by: What triggered this run ('scheduler', 'manual', 'hook')

        Returns:
            int: Database ID of the new run
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO qa_runs (run_id, triggered_by, status)
                    VALUES (%s, %s, 'running')
                    RETURNING id
                """, (run_id, triggered_by))

                db_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Started QA run {run_id} (DB ID: {db_id})")
                return db_id

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error starting QA run: {e}")
            raise
        finally:
            conn.close()

    def complete_run(
        self,
        run_db_id: int,
        health_score: float,
        total_checks: int = 0,
        passed_checks: int = 0,
        failed_checks: int = 0,
        warned_checks: int = 0,
        skipped_checks: int = 0,
        critical_issues: int = 0,
        high_issues: int = 0,
        medium_issues: int = 0,
        low_issues: int = 0,
        auto_fixes_attempted: int = 0,
        auto_fixes_succeeded: int = 0,
        status: str = "completed",
        error_message: str = None
    ) -> bool:
        """
        Complete a QA run with final metrics

        Args:
            run_db_id: Database ID of the run
            health_score: Overall health score (0-100)
            ... (other metrics)

        Returns:
            bool: True if updated successfully
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE qa_runs SET
                        completed_at = NOW(),
                        duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INTEGER,
                        health_score = %s,
                        total_checks = %s,
                        passed_checks = %s,
                        failed_checks = %s,
                        warned_checks = %s,
                        skipped_checks = %s,
                        critical_issues = %s,
                        high_issues = %s,
                        medium_issues = %s,
                        low_issues = %s,
                        auto_fixes_attempted = %s,
                        auto_fixes_succeeded = %s,
                        status = %s,
                        error_message = %s
                    WHERE id = %s
                """, (
                    health_score, total_checks, passed_checks, failed_checks,
                    warned_checks, skipped_checks, critical_issues, high_issues,
                    medium_issues, low_issues, auto_fixes_attempted, auto_fixes_succeeded,
                    status, error_message, run_db_id
                ))

                conn.commit()
                logger.info(f"Completed QA run {run_db_id} with health score {health_score}")
                return cursor.rowcount > 0

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error completing QA run: {e}")
            raise
        finally:
            conn.close()

    def get_run(self, run_db_id: int) -> Optional[Dict[str, Any]]:
        """Get QA run details by database ID"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM qa_runs WHERE id = %s
                """, (run_db_id,))

                run = cursor.fetchone()
                return dict(run) if run else None

        except psycopg2.Error as e:
            logger.error(f"Error fetching QA run: {e}")
            raise
        finally:
            conn.close()

    def get_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent QA runs"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM v_qa_recent_runs LIMIT %s
                """, (limit,))

                runs = cursor.fetchall()
                return [dict(r) for r in runs]

        except psycopg2.Error as e:
            logger.error(f"Error fetching recent runs: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # CHECK RESULTS OPERATIONS
    # ========================================================================

    def log_check_result(
        self,
        run_id: int,
        module_name: str,
        check_name: str,
        status: str,
        message: str = None,
        details: Dict[str, Any] = None,
        duration_ms: int = None,
        auto_fixable: bool = False,
        fix_attempted: bool = False,
        fix_succeeded: bool = False,
        fix_message: str = None
    ) -> int:
        """
        Log a check result for a QA run

        Args:
            run_id: Database ID of the QA run
            module_name: Name of the check module (e.g., 'api_endpoints')
            check_name: Name of the specific check (e.g., 'server_available')
            status: Check status ('passed', 'failed', 'warned', 'skipped', 'error')
            message: Result message
            details: Structured details (stored as JSONB)
            duration_ms: Check duration in milliseconds
            auto_fixable: Whether this issue can be auto-fixed
            fix_attempted: Whether a fix was attempted
            fix_succeeded: Whether the fix succeeded
            fix_message: Fix attempt message

        Returns:
            int: Check result ID
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO qa_check_results (
                        run_id, module_name, check_name, status, message,
                        details, duration_ms, auto_fixable, fix_attempted,
                        fix_succeeded, fix_message
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id
                """, (
                    run_id, module_name, check_name, status, message,
                    Json(details) if details else None, duration_ms,
                    auto_fixable, fix_attempted, fix_succeeded, fix_message
                ))

                result_id = cursor.fetchone()[0]
                conn.commit()

                logger.debug(f"Logged check result: {module_name}.{check_name} = {status}")
                return result_id

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error logging check result: {e}")
            raise
        finally:
            conn.close()

    def get_check_results(
        self,
        run_id: int,
        status_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Get all check results for a QA run"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT * FROM qa_check_results
                    WHERE run_id = %s
                """
                params = [run_id]

                if status_filter:
                    query += " AND status = %s"
                    params.append(status_filter)

                query += " ORDER BY started_at"

                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(r) for r in results]

        except psycopg2.Error as e:
            logger.error(f"Error fetching check results: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # ISSUES OPERATIONS
    # ========================================================================

    def log_issue(
        self,
        run_id: int,
        module_name: str,
        check_name: str,
        title: str,
        severity: str,
        description: str = None,
        category: str = None,
        files_affected: List[str] = None,
        primary_file: str = None,
        details: Dict[str, Any] = None,
        tags: List[str] = None,
        check_result_id: int = None
    ) -> int:
        """
        Log an issue (with automatic deduplication)

        Args:
            run_id: Database ID of the QA run
            module_name: Check module that found the issue
            check_name: Specific check that found the issue
            title: Issue title (used for deduplication)
            severity: Issue severity ('critical', 'high', 'medium', 'low')
            description: Detailed description
            category: Issue category (e.g., 'api', 'import', 'security')
            files_affected: List of affected file paths
            primary_file: Main file affected
            details: Additional structured details
            tags: Issue tags for filtering
            check_result_id: ID of the check result that found this issue

        Returns:
            int: Issue ID (either new or existing)
        """
        conn = self.get_connection()
        try:
            issue_hash = self._calculate_issue_hash(module_name, check_name, title)

            with conn.cursor() as cursor:
                # Use upsert to handle deduplication
                cursor.execute("""
                    INSERT INTO qa_issues (
                        issue_hash, module_name, check_name, title, description,
                        severity, category, first_seen_run_id, last_seen_run_id,
                        files_affected, primary_file, details, tags
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (issue_hash) DO UPDATE SET
                        last_seen_at = NOW(),
                        last_seen_run_id = EXCLUDED.last_seen_run_id,
                        occurrence_count = qa_issues.occurrence_count + 1,
                        status = CASE
                            WHEN qa_issues.status = 'fixed' THEN 'open'
                            ELSE qa_issues.status
                        END
                    RETURNING id, (xmax = 0) AS is_new
                """, (
                    issue_hash, module_name, check_name, title, description,
                    severity, category, run_id, run_id,
                    files_affected, primary_file,
                    Json(details) if details else None, tags
                ))

                result = cursor.fetchone()
                issue_id = result[0]
                is_new = result[1]

                # Link issue to this run
                cursor.execute("""
                    INSERT INTO qa_issue_occurrences (issue_id, run_id, check_result_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (issue_id, run_id) DO NOTHING
                """, (issue_id, run_id, check_result_id))

                # Update hot spots for affected files
                if files_affected:
                    for file_path in files_affected:
                        cursor.execute("""
                            SELECT update_hot_spot(%s, %s, %s, %s)
                        """, (file_path, severity, module_name, category))

                conn.commit()

                if is_new:
                    logger.info(f"New issue logged: {title} ({severity})")
                else:
                    logger.info(f"Existing issue updated: {title}")

                return issue_id

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error logging issue: {e}")
            raise
        finally:
            conn.close()

    def get_issue(self, issue_id: int) -> Optional[Dict[str, Any]]:
        """Get issue details by ID"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM qa_issues WHERE id = %s
                """, (issue_id,))

                issue = cursor.fetchone()
                return dict(issue) if issue else None

        except psycopg2.Error as e:
            logger.error(f"Error fetching issue: {e}")
            raise
        finally:
            conn.close()

    def get_open_issues(
        self,
        severity_filter: str = None,
        category_filter: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all open issues"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = "SELECT * FROM v_qa_open_issues WHERE 1=1"
                params = []

                if severity_filter:
                    query += " AND severity = %s"
                    params.append(severity_filter)

                if category_filter:
                    query += " AND category = %s"
                    params.append(category_filter)

                query += " LIMIT %s"
                params.append(limit)

                cursor.execute(query, params)
                issues = cursor.fetchall()
                return [dict(i) for i in issues]

        except psycopg2.Error as e:
            logger.error(f"Error fetching open issues: {e}")
            raise
        finally:
            conn.close()

    def get_issues(
        self,
        status_filter: str = None,
        severity_filter: str = None,
        category_filter: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get issues with flexible filtering

        Args:
            status_filter: Filter by status ('open', 'fixing', 'fixed', 'ignored', 'wont_fix')
            severity_filter: Filter by severity ('critical', 'high', 'medium', 'low')
            category_filter: Filter by category
            limit: Maximum number of results

        Returns:
            List of issue dictionaries
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT
                        id, issue_hash, module_name, check_name, title, description,
                        severity, category, status, first_seen_at, last_seen_at,
                        occurrence_count, primary_file, files_affected, tags,
                        resolved_at, resolved_by, resolution_notes
                    FROM qa_issues
                    WHERE 1=1
                """
                params = []

                if status_filter:
                    query += " AND status = %s"
                    params.append(status_filter)

                if severity_filter:
                    query += " AND severity = %s"
                    params.append(severity_filter)

                if category_filter:
                    query += " AND category = %s"
                    params.append(category_filter)

                # Order by severity priority, then occurrence count
                query += """
                    ORDER BY
                        CASE severity
                            WHEN 'critical' THEN 1
                            WHEN 'high' THEN 2
                            WHEN 'medium' THEN 3
                            WHEN 'low' THEN 4
                        END,
                        occurrence_count DESC
                    LIMIT %s
                """
                params.append(limit)

                cursor.execute(query, params)
                issues = cursor.fetchall()
                return [dict(i) for i in issues]

        except psycopg2.Error as e:
            logger.error(f"Error fetching issues: {e}")
            raise
        finally:
            conn.close()

    def update_issue_status(
        self,
        issue_id: int,
        status: str,
        resolved_by: str = None,
        resolution_notes: str = None
    ) -> bool:
        """
        Update issue status

        Args:
            issue_id: Issue ID
            status: New status ('open', 'fixing', 'fixed', 'ignored', 'wont_fix')
            resolved_by: Who resolved (if fixing/fixed)
            resolution_notes: Notes about the resolution

        Returns:
            bool: True if updated successfully
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE qa_issues SET
                        status = %s,
                        resolved_at = CASE WHEN %s IN ('fixed', 'ignored', 'wont_fix') THEN NOW() ELSE resolved_at END,
                        resolved_by = COALESCE(%s, resolved_by),
                        resolution_notes = COALESCE(%s, resolution_notes)
                    WHERE id = %s
                """, (status, status, resolved_by, resolution_notes, issue_id))

                conn.commit()
                logger.info(f"Updated issue {issue_id} status to {status}")
                return cursor.rowcount > 0

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error updating issue status: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # FIXES OPERATIONS
    # ========================================================================

    def log_fix_attempt(
        self,
        issue_id: int,
        fix_type: str,
        success: bool,
        run_id: int = None,
        fixer_name: str = None,
        message: str = None,
        error_details: str = None,
        files_modified: List[str] = None,
        lines_added: int = 0,
        lines_removed: int = 0,
        git_commit_hash: str = None,
        details: Dict[str, Any] = None
    ) -> int:
        """
        Log a fix attempt for an issue

        Args:
            issue_id: Issue ID
            fix_type: Type of fix ('auto', 'manual', 'agent')
            success: Whether the fix succeeded
            run_id: QA run ID (if part of auto-fix)
            fixer_name: Name of fixer (agent name, 'auto_fix', 'user')
            message: Fix message
            error_details: Error details if failed
            files_modified: List of modified files
            lines_added: Lines added
            lines_removed: Lines removed
            git_commit_hash: Git commit SHA
            details: Additional details

        Returns:
            int: Fix record ID
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO qa_fixes (
                        issue_id, run_id, fix_type, fixer_name, success, message,
                        error_details, files_modified, lines_added, lines_removed,
                        git_commit_hash, details, completed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        CASE WHEN %s THEN NOW() ELSE NULL END
                    )
                    RETURNING id
                """, (
                    issue_id, run_id, fix_type, fixer_name, success, message,
                    error_details, files_modified, lines_added, lines_removed,
                    git_commit_hash, Json(details) if details else None, success
                ))

                fix_id = cursor.fetchone()[0]

                # Update issue status if fix succeeded
                if success:
                    cursor.execute("""
                        UPDATE qa_issues SET
                            status = 'fixed',
                            resolved_at = NOW(),
                            resolved_by = %s,
                            resolution_notes = %s
                        WHERE id = %s
                    """, (fixer_name or fix_type, message, issue_id))

                conn.commit()
                logger.info(f"Logged fix attempt for issue {issue_id}: success={success}")
                return fix_id

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error logging fix attempt: {e}")
            raise
        finally:
            conn.close()

    def get_fix_history(self, issue_id: int) -> List[Dict[str, Any]]:
        """Get fix history for an issue"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM qa_fixes
                    WHERE issue_id = %s
                    ORDER BY attempted_at DESC
                """, (issue_id,))

                fixes = cursor.fetchall()
                return [dict(f) for f in fixes]

        except psycopg2.Error as e:
            logger.error(f"Error fetching fix history: {e}")
            raise
        finally:
            conn.close()

    def get_issue_occurrences(self, issue_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get occurrence history for an issue"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        o.id,
                        o.run_id,
                        o.occurred_at as detected_at,
                        cr.details::TEXT as context
                    FROM qa_issue_occurrences o
                    LEFT JOIN qa_check_results cr ON cr.id = o.check_result_id
                    WHERE o.issue_id = %s
                    ORDER BY o.occurred_at DESC
                    LIMIT %s
                """, (issue_id, limit))

                occurrences = cursor.fetchall()
                return [dict(o) for o in occurrences]

        except psycopg2.Error as e:
            logger.error(f"Error fetching issue occurrences: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # HEALTH & ANALYTICS OPERATIONS
    # ========================================================================

    def get_health_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get health score history"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM qa_health_history
                    WHERE recorded_at > NOW() - INTERVAL '%s days'
                    ORDER BY recorded_at DESC
                """, (days,))

                history = cursor.fetchall()
                return [dict(h) for h in history]

        except psycopg2.Error as e:
            logger.error(f"Error fetching health history: {e}")
            raise
        finally:
            conn.close()

    def get_health_trend(self, hours: int = 168) -> List[Dict[str, Any]]:
        """
        Get hourly health score trend

        Args:
            hours: Number of hours to look back (default 168 = 7 days)

        Returns:
            List of hourly health trend data
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        DATE_TRUNC('hour', recorded_at) AS timestamp,
                        DATE_TRUNC('hour', recorded_at) AS hour,
                        AVG(health_score)::NUMERIC(5,2) AS avg_score,
                        MIN(health_score) AS min_score,
                        MAX(health_score) AS max_score,
                        AVG(total_open_issues)::INTEGER AS avg_open_issues
                    FROM qa_health_history
                    WHERE recorded_at > NOW() - INTERVAL '%s hours'
                    GROUP BY DATE_TRUNC('hour', recorded_at)
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (hours, hours))
                trend = cursor.fetchall()
                return [dict(t) for t in trend]

        except psycopg2.Error as e:
            logger.error(f"Error fetching health trend: {e}")
            raise
        finally:
            conn.close()

    def get_hot_spots(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top hot spots (files with frequent issues)"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM v_qa_top_hot_spots LIMIT %s
                """, (limit,))

                hot_spots = cursor.fetchall()
                return [dict(h) for h in hot_spots]

        except psycopg2.Error as e:
            logger.error(f"Error fetching hot spots: {e}")
            raise
        finally:
            conn.close()

    def get_issue_trends_by_category(self) -> List[Dict[str, Any]]:
        """Get issue trends grouped by category"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM v_qa_issue_trends")
                trends = cursor.fetchall()
                return [dict(t) for t in trends]

        except psycopg2.Error as e:
            logger.error(f"Error fetching issue trends: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # SUMMARY OPERATIONS
    # ========================================================================

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard summary

        Returns:
            Dict with all dashboard metrics matching frontend expectations
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get latest run
                cursor.execute("""
                    SELECT * FROM qa_runs
                    ORDER BY started_at DESC LIMIT 1
                """)
                latest_run = cursor.fetchone()

                # Get issues count by severity (all statuses)
                cursor.execute("""
                    SELECT
                        severity,
                        COUNT(*) as count
                    FROM qa_issues
                    GROUP BY severity
                """)
                by_severity = {r['severity']: r['count'] for r in cursor.fetchall()}

                # Get issues count by status
                cursor.execute("""
                    SELECT
                        status,
                        COUNT(*) as count
                    FROM qa_issues
                    GROUP BY status
                """)
                by_status = {r['status']: r['count'] for r in cursor.fetchall()}

                # Get issue counts
                cursor.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status IN ('open', 'fixing')) as open_issues,
                        COUNT(*) FILTER (WHERE status = 'fixed') as fixed_issues,
                        COUNT(*) FILTER (WHERE status IN ('ignored', 'wont_fix')) as dismissed_issues,
                        COUNT(*) as total_issues
                    FROM qa_issues
                """)
                issue_counts = cursor.fetchone()

                # Get recent runs (for the runs list)
                cursor.execute("""
                    SELECT * FROM v_qa_recent_runs LIMIT 10
                """)
                recent_runs = [dict(r) for r in cursor.fetchall()]

                # Get top open issues
                cursor.execute("""
                    SELECT * FROM v_qa_open_issues LIMIT 10
                """)
                top_issues = [dict(i) for i in cursor.fetchall()]

                # Get health trend
                cursor.execute("""
                    SELECT * FROM v_qa_health_trend LIMIT 48
                """)
                health_trend = [dict(h) for h in cursor.fetchall()]

                return {
                    "latest_run": dict(latest_run) if latest_run else None,
                    "total_issues": issue_counts['total_issues'] if issue_counts else 0,
                    "open_issues": issue_counts['open_issues'] if issue_counts else 0,
                    "fixed_issues": issue_counts['fixed_issues'] if issue_counts else 0,
                    "dismissed_issues": issue_counts['dismissed_issues'] if issue_counts else 0,
                    "by_severity": by_severity,
                    "by_status": by_status,
                    "recent_runs": recent_runs,
                    "top_issues": top_issues,
                    "health_trend": health_trend,
                    "timestamp": datetime.now().isoformat()
                }

        except psycopg2.Error as e:
            logger.error(f"Error fetching dashboard summary: {e}")
            raise
        finally:
            conn.close()

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def initialize_schema(self) -> bool:
        """
        Initialize the QA issues schema (run migrations)

        Returns:
            bool: True if successful
        """
        schema_path = os.path.join(
            os.path.dirname(__file__),
            'qa_issues_schema.sql'
        )

        if not os.path.exists(schema_path):
            logger.error(f"Schema file not found: {schema_path}")
            return False

        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()

                cursor.execute(schema_sql)
                conn.commit()

                logger.info("QA issues schema initialized successfully")
                return True

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error initializing schema: {e}")
            raise
        finally:
            conn.close()

    def check_schema_exists(self) -> bool:
        """Check if the QA issues tables exist"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('qa_runs', 'qa_check_results', 'qa_issues', 'qa_fixes')
                """)

                count = cursor.fetchone()[0]
                return count == 4

        except psycopg2.Error as e:
            logger.error(f"Error checking schema: {e}")
            return False
        finally:
            conn.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Testing QA Issues Database Manager...\n")

    try:
        # Initialize manager
        qa_mgr = QAIssuesDBManager()

        # Check if schema exists
        print("Test 1: Checking schema...")
        if not qa_mgr.check_schema_exists():
            print("   Schema not found, initializing...")
            qa_mgr.initialize_schema()
        print("   Schema OK")

        # Test 2: Start a QA run
        print("\nTest 2: Starting a QA run...")
        run_id = f"qa_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_db_id = qa_mgr.start_run(run_id=run_id, triggered_by="manual")
        print(f"   Created run: {run_id} (DB ID: {run_db_id})")

        # Test 3: Log check results
        print("\nTest 3: Logging check results...")
        qa_mgr.log_check_result(
            run_id=run_db_id,
            module_name="api_endpoints",
            check_name="server_available",
            status="passed",
            message="API server responding at http://localhost:8002",
            duration_ms=150
        )
        qa_mgr.log_check_result(
            run_id=run_db_id,
            module_name="api_endpoints",
            check_name="endpoint_responses",
            status="failed",
            message="2 of 10 endpoints failing",
            details={"failing": ["/api/options/analysis", "/api/earnings/calendar"]},
            auto_fixable=False
        )
        print("   Check results logged")

        # Test 4: Log an issue
        print("\nTest 4: Logging an issue...")
        issue_id = qa_mgr.log_issue(
            run_id=run_db_id,
            module_name="api_endpoints",
            check_name="endpoint_responses",
            title="Endpoint /api/options/analysis returning 500",
            severity="high",
            category="api",
            description="The options analysis endpoint is returning HTTP 500 errors",
            files_affected=["backend/routers/options.py"],
            primary_file="backend/routers/options.py"
        )
        print(f"   Issue logged: ID {issue_id}")

        # Test 5: Complete the run
        print("\nTest 5: Completing the run...")
        qa_mgr.complete_run(
            run_db_id=run_db_id,
            health_score=85.5,
            total_checks=10,
            passed_checks=8,
            failed_checks=2,
            high_issues=1
        )
        print("   Run completed")

        # Test 6: Get dashboard summary
        print("\nTest 6: Getting dashboard summary...")
        summary = qa_mgr.get_dashboard_summary()
        print(f"   Latest health score: {summary.get('latest_run', {}).get('health_score')}")
        print(f"   Open issues: {summary.get('issue_stats', {}).get('open_count', 0)}")

        # Test 7: Get open issues
        print("\nTest 7: Getting open issues...")
        open_issues = qa_mgr.get_open_issues(limit=5)
        print(f"   Found {len(open_issues)} open issues")
        for issue in open_issues:
            print(f"   - [{issue['severity']}] {issue['title']}")

        print("\n All tests passed!")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
