"""
Multi-Agent QA Service
======================

Orchestrates the multi-agent QA review process with sign-offs and issue tracking.

Workflow:
1. Task completed → Trigger QA review
2. Assign required agents for review
3. Each agent reviews using their RAG expertise
4. Agents create QA tasks for issues found
5. All issues must be resolved before sign-off
6. Once all agents sign off → Mark task truly complete
7. Historical record preserved (never deleted)

Features:
- Multi-agent sign-off requirements
- RAG-assisted expert reviews
- QA issue tracking
- Historical audit trail
- Legion integration
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
import logging
import json

from .agent_rag_expertise import get_expertise_registry, AgentRAGExpertise

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentQAService:
    """
    Orchestrates multi-agent QA review process.

    Ensures all tasks go through proper QA review before being marked complete.
    """

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        self.expertise_registry = get_expertise_registry()

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def trigger_qa_review(self, task_id: int) -> Dict[str, Any]:
        """
        Trigger QA review for a completed task.

        This creates sign-off requests for all required agents.

        Args:
            task_id: Task to review

        Returns:
            Dictionary with review status
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get task details
            cur.execute("""
                SELECT id, title, task_type, priority, feature_area, status
                FROM development_tasks
                WHERE id = %s;
            """, (task_id,))

            task = cur.fetchone()

            if not task:
                return {"error": "Task not found", "task_id": task_id}

            if task['status'] != 'completed':
                return {"error": "Task must be completed before QA review", "task_id": task_id, "status": task['status']}

            # Get required agents for this task type
            cur.execute("""
                SELECT required_agents, minimum_required_sign_offs, requires_unanimous
                FROM qa_sign_off_requirements
                WHERE task_type = %s
                AND (feature_area IS NULL OR feature_area = %s)
                AND (priority IS NULL OR priority = %s)
                AND is_active = true
                ORDER BY
                    CASE WHEN feature_area IS NOT NULL THEN 1 ELSE 2 END,
                    CASE WHEN priority IS NOT NULL THEN 1 ELSE 2 END
                LIMIT 1;
            """, (task['task_type'], task['feature_area'], task['priority']))

            requirements = cur.fetchone()

            if not requirements or not requirements['required_agents']:
                logger.info(f"No QA requirements for task {task_id}, marking as complete")
                return {
                    "status": "no_qa_required",
                    "task_id": task_id,
                    "message": "No QA review required for this task type"
                }

            required_agents = requirements['required_agents']

            # Create sign-off records for each required agent
            sign_offs_created = []

            for agent_name in required_agents:
                cur.execute("""
                    INSERT INTO qa_agent_sign_offs (
                        task_id,
                        agent_name,
                        sign_off_status,
                        created_at
                    )
                    VALUES (%s, %s, 'pending', NOW())
                    RETURNING id;
                """, (task_id, agent_name))

                sign_off_id = cur.fetchone()['id']
                sign_offs_created.append({
                    "agent_name": agent_name,
                    "sign_off_id": sign_off_id
                })

            conn.commit()

            logger.info(f"Created {len(sign_offs_created)} sign-off requests for task {task_id}")

            return {
                "status": "qa_review_triggered",
                "task_id": task_id,
                "task_title": task['title'],
                "required_agents": required_agents,
                "minimum_sign_offs": requirements['minimum_required_sign_offs'],
                "requires_unanimous": requirements['requires_unanimous'],
                "sign_offs_created": sign_offs_created
            }

        except Exception as e:
            conn.rollback()
            logger.error(f"Error triggering QA review: {e}")
            return {"error": str(e), "task_id": task_id}

        finally:
            cur.close()
            conn.close()

    def perform_agent_review(
        self,
        sign_off_id: int,
        agent_name: str,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Perform review as a specific agent.

        Uses agent's RAG expertise to assist with review.

        Args:
            sign_off_id: Sign-off record ID
            agent_name: Agent performing review
            use_rag: Whether to use RAG expertise

        Returns:
            Review results
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get sign-off and task details
            cur.execute("""
                SELECT
                    s.id, s.task_id, s.agent_name, s.sign_off_status,
                    t.title, t.description, t.task_type, t.feature_area
                FROM qa_agent_sign_offs s
                JOIN development_tasks t ON s.task_id = t.id
                WHERE s.id = %s AND s.agent_name = %s;
            """, (sign_off_id, agent_name))

            review = cur.fetchone()

            if not review:
                return {"error": "Sign-off not found or agent mismatch"}

            if review['sign_off_status'] != 'pending':
                return {"error": f"Review already {review['sign_off_status']}"}

            # Start review
            cur.execute("""
                UPDATE qa_agent_sign_offs
                SET review_started_at = NOW()
                WHERE id = %s;
            """, (sign_off_id,))

            conn.commit()

            # Get agent expertise
            agent_expertise = None
            rag_context = None
            rag_sources = []

            if use_rag:
                agent_expertise = self.expertise_registry.get_agent_expertise(agent_name)

                if agent_expertise:
                    # Get relevant expertise for this task
                    rag_context = agent_expertise.get_review_context(
                        f"{review['title']}\n\n{review['description']}"
                    )

                    # Get checklist
                    checklist = agent_expertise.get_checklist_for_review(
                        review['task_type'],
                        review['feature_area']
                    )

                    # Search for relevant patterns
                    relevant = agent_expertise.search_expertise(
                        f"{review['task_type']} {review['feature_area']} {review['title']}"
                    )

                    rag_sources = [r.document.title for r in relevant]

            return {
                "status": "review_in_progress",
                "sign_off_id": sign_off_id,
                "task_id": review['task_id'],
                "task_title": review['title'],
                "agent_name": agent_name,
                "rag_context": rag_context if use_rag else None,
                "rag_sources": rag_sources if use_rag else [],
                "checklist": checklist if use_rag and agent_expertise else [],
                "message": "Review started. Use complete_agent_review() when done."
            }

        except Exception as e:
            conn.rollback()
            logger.error(f"Error performing review: {e}")
            return {"error": str(e)}

        finally:
            cur.close()
            conn.close()

    def complete_agent_review(
        self,
        sign_off_id: int,
        agent_name: str,
        approved: bool,
        review_notes: str,
        issues_found: List[Dict[str, Any]] = None,
        confidence_score: float = 1.0
    ) -> Dict[str, Any]:
        """
        Complete an agent's review with their decision.

        Args:
            sign_off_id: Sign-off record ID
            agent_name: Agent completing review
            approved: Whether agent approves
            review_notes: Detailed review notes
            issues_found: List of QA issues (if any)
            confidence_score: Agent's confidence (0.0 to 1.0)

        Returns:
            Completion status
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get review details
            cur.execute("""
                SELECT task_id, review_started_at
                FROM qa_agent_sign_offs
                WHERE id = %s AND agent_name = %s;
            """, (sign_off_id, agent_name))

            review = cur.fetchone()

            if not review:
                return {"error": "Sign-off not found"}

            task_id = review['task_id']

            # Calculate review duration
            if review['review_started_at']:
                # Make sure both datetimes have timezone info
                started_at = review['review_started_at']
                if started_at.tzinfo is None:
                    # Database value is naive, use naive datetime
                    duration = (datetime.now() - started_at).total_seconds() / 60
                else:
                    # Database value is aware, use aware datetime
                    from datetime import timezone
                    duration = (datetime.now(timezone.utc) - started_at).total_seconds() / 60
            else:
                duration = None

            # Determine status
            if approved:
                status = 'approved'
            else:
                status = 'rejected' if issues_found else 'needs_changes'

            # Determine severity
            severity = None
            if issues_found:
                severities = [issue.get('severity', 'low') for issue in issues_found]
                if 'critical' in severities:
                    severity = 'critical'
                elif 'high' in severities:
                    severity = 'high'
                else:
                    severity = 'medium'

            # Update sign-off
            cur.execute("""
                UPDATE qa_agent_sign_offs
                SET
                    sign_off_status = %s,
                    review_completed_at = NOW(),
                    review_duration_minutes = %s,
                    review_notes = %s,
                    issues_found = %s,
                    severity_level = %s,
                    confidence_score = %s,
                    is_final = true
                WHERE id = %s;
            """, (
                status,
                int(duration) if duration else None,
                review_notes,
                len(issues_found) if issues_found else 0,
                severity,
                confidence_score,
                sign_off_id
            ))

            # Create QA tasks for issues found
            qa_task_ids = []
            if issues_found:
                for issue in issues_found:
                    cur.execute("""
                        SELECT create_qa_task_from_signoff(%s, %s, %s, %s, %s, %s, %s);
                    """, (
                        task_id,
                        sign_off_id,
                        issue.get('title', 'Issue found during review'),
                        issue.get('description', ''),
                        issue.get('issue_type', 'code_quality'),
                        issue.get('severity', 'medium'),
                        agent_name
                    ))

                    qa_task_id = cur.fetchone()[0]
                    qa_task_ids.append(qa_task_id)

            conn.commit()

            # Check if all sign-offs complete
            all_complete = self.check_all_sign_offs_complete(task_id)

            logger.info(f"Agent {agent_name} completed review for sign-off {sign_off_id}: {status}")

            return {
                "status": "review_completed",
                "sign_off_id": sign_off_id,
                "task_id": task_id,
                "agent_name": agent_name,
                "approved": approved,
                "issues_found": len(issues_found) if issues_found else 0,
                "qa_task_ids": qa_task_ids,
                "all_sign_offs_complete": all_complete,
                "can_mark_task_complete": all_complete
            }

        except Exception as e:
            conn.rollback()
            logger.error(f"Error completing review: {e}")
            return {"error": str(e)}

        finally:
            cur.close()
            conn.close()

    def check_all_sign_offs_complete(self, task_id: int) -> bool:
        """Check if all required sign-offs are complete for a task"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT check_qa_sign_offs_complete(%s);", (task_id,))
            result = cur.fetchone()
            return result[0] if result else False

        finally:
            cur.close()
            conn.close()

    def get_pending_reviews(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending reviews for an agent"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT * FROM v_pending_qa_reviews
                WHERE agent_name = %s
                LIMIT %s;
            """, (agent_name, limit))

            return cur.fetchall()

        finally:
            cur.close()
            conn.close()

    def get_qa_tasks_for_parent(self, parent_task_id: int) -> List[Dict[str, Any]]:
        """Get all QA tasks for a parent task"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT *
                FROM qa_tasks
                WHERE parent_task_id = %s
                AND is_deleted = false
                ORDER BY severity, reported_at;
            """, (parent_task_id,))

            return cur.fetchall()

        finally:
            cur.close()
            conn.close()

    def mark_qa_task_complete(self, qa_task_id: int, resolution_notes: str, verified_by: str) -> Dict[str, Any]:
        """Mark a QA task as fixed and verified"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                UPDATE qa_tasks
                SET
                    status = 'verified',
                    fixed_at = NOW(),
                    verified_at = NOW(),
                    resolution_notes = %s,
                    verified_by_agent = %s
                WHERE id = %s
                RETURNING parent_task_id;
            """, (resolution_notes, verified_by, qa_task_id))

            result = cur.fetchone()

            if not result:
                return {"error": "QA task not found"}

            parent_task_id = result['parent_task_id']

            conn.commit()

            # Check if parent task can now be marked truly complete
            cur.execute("""
                SELECT COUNT(*)
                FROM qa_tasks
                WHERE parent_task_id = %s
                AND status NOT IN ('verified', 'closed', 'wont_fix')
                AND is_deleted = false;
            """, (parent_task_id,))

            open_issues = cur.fetchone()[0]

            all_clear = (open_issues == 0)

            return {
                "status": "qa_task_verified",
                "qa_task_id": qa_task_id,
                "parent_task_id": parent_task_id,
                "remaining_open_issues": open_issues,
                "all_qa_issues_resolved": all_clear
            }

        except Exception as e:
            conn.rollback()
            logger.error(f"Error marking QA task complete: {e}")
            return {"error": str(e)}

        finally:
            cur.close()
            conn.close()

    def finalize_task_completion(self, task_id: int) -> Dict[str, Any]:
        """
        Finalize task completion after all QA sign-offs and issues resolved.

        This is when the task is TRULY complete.

        Args:
            task_id: Task to finalize

        Returns:
            Finalization status
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Check all sign-offs complete
            all_sign_offs_done = self.check_all_sign_offs_complete(task_id)

            if not all_sign_offs_done:
                return {
                    "error": "Not all required sign-offs complete",
                    "task_id": task_id,
                    "can_finalize": False
                }

            # Check all QA issues resolved
            cur.execute("""
                SELECT COUNT(*)
                FROM qa_tasks
                WHERE parent_task_id = %s
                AND status NOT IN ('verified', 'closed', 'wont_fix')
                AND is_deleted = false;
            """, (task_id,))

            open_issues = cur.fetchone()[0]

            if open_issues > 0:
                return {
                    "error": f"{open_issues} QA issues still open",
                    "task_id": task_id,
                    "open_issues": open_issues,
                    "can_finalize": False
                }

            # All checks passed - mark task as truly complete
            cur.execute("""
                UPDATE development_tasks
                SET
                    status = 'qa_approved',
                    updated_at = NOW()
                WHERE id = %s;
            """, (task_id,))

            conn.commit()

            logger.info(f"Task {task_id} finalized - all QA complete!")

            return {
                "status": "task_finalized",
                "task_id": task_id,
                "message": "Task approved by all QA agents and all issues resolved",
                "can_deploy": True
            }

        except Exception as e:
            conn.rollback()
            logger.error(f"Error finalizing task: {e}")
            return {"error": str(e)}

        finally:
            cur.close()
            conn.close()

    def get_task_qa_status(self, task_id: int) -> Dict[str, Any]:
        """Get comprehensive QA status for a task using the database view"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT *
                FROM v_task_qa_status
                WHERE task_id = %s;
            """, (task_id,))

            result = cur.fetchone()

            if not result:
                return {
                    "error": "Task not found or no QA review triggered",
                    "task_id": task_id
                }

            return dict(result)

        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    # Test the service
    print("Testing Multi-Agent QA Service\n")

    qa_service = MultiAgentQAService()

    print("✅ Multi-Agent QA Service initialized")
    print("\nExample workflow:")
    print("1. Task completed → qa_service.trigger_qa_review(task_id)")
    print("2. Agents review → qa_service.perform_agent_review(sign_off_id, agent_name)")
    print("3. Complete review → qa_service.complete_agent_review(...)")
    print("4. Fix QA issues → qa_service.mark_qa_task_complete(...)")
    print("5. Finalize → qa_service.finalize_task_completion(task_id)")
    print("\nTask is NEVER deleted, only marked 'qa_approved' when truly complete")
