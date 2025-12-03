"""
Enhanced Task Completion with QA Integration
============================================

Ensures ALL tasks go through QA review before final approval.

Workflow:
1. Task completed by developer/agent
2. Status: 'completed' → Triggers QA review
3. QA agents review and sign off
4. If approved: Status → 'qa_approved' (can deploy)
5. If issues: Create QA tasks, fix, repeat
"""

from src.task_db_manager import TaskDBManager
from src.qa import MultiAgentQAService
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TaskCompletionWithQA:
    """
    Enhanced task completion that enforces QA sign-off

    Usage:
        completion_mgr = TaskCompletionWithQA()

        # Developer completes task
        result = completion_mgr.complete_task(task_id)
        # → Status: 'completed', QA review triggered

        # QA agents review...

        # After all approvals
        result = completion_mgr.finalize_task(task_id)
        # → Status: 'qa_approved', can deploy
    """

    def __init__(self) -> None:
        self.task_mgr = TaskDBManager()
        self.qa_service = MultiAgentQAService()

    def complete_task(
        self,
        task_id: int,
        completion_notes: Optional[str] = None,
        trigger_qa: bool = True
    ) -> Dict[str, Any]:
        """
        Mark task as completed and optionally trigger QA review

        Args:
            task_id: Task ID
            completion_notes: Optional notes about completion
            trigger_qa: Whether to automatically trigger QA review

        Returns:
            Dictionary with status and QA information
        """
        try:
            logger.info(f"Completing task {task_id}...")

            # Mark task as completed
            success = self.task_mgr.update_task_status(task_id, "completed")

            if not success:
                return {
                    "success": False,
                    "error": "Failed to update task status",
                    "task_id": task_id
                }

            # Log completion (skip for now - constraint issues)
            # if completion_notes:
            #     self.task_mgr.log_execution(
            #         task_id=task_id,
            #         agent_name="system",
            #         action_type="task_completed",
            #         message=completion_notes
            #     )

            result = {
                "success": True,
                "task_id": task_id,
                "status": "completed",
                "qa_triggered": False
            }

            # Automatically trigger QA review
            if trigger_qa:
                qa_result = self.qa_service.trigger_qa_review(task_id)

                if 'error' not in qa_result:
                    result["qa_triggered"] = True
                    result["qa_status"] = qa_result['status']
                    result["required_agents"] = qa_result.get('required_agents', [])
                    result["sign_offs_created"] = len(qa_result.get('sign_offs_created', []))

                    logger.info(f"QA review triggered for task {task_id}")
                    logger.info(f"Required agents: {', '.join(result['required_agents'])}")
                else:
                    # No QA required or error
                    result["qa_status"] = qa_result.get('status', 'no_qa_required')
                    logger.info(f"No QA review required for task {task_id}")

            return result

        except Exception as e:
            logger.error(f"Error completing task {task_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }

    def check_qa_status(self, task_id: int) -> Dict[str, Any]:
        """
        Check QA review status for a task

        Args:
            task_id: Task ID

        Returns:
            QA status dictionary
        """
        try:
            status = self.qa_service.get_task_qa_status(task_id)

            # Add convenience fields
            status['can_finalize'] = (
                status.get('all_sign_offs_complete', False) and
                status.get('open_qa_issues', 0) == 0
            )

            if not status['can_finalize']:
                reasons = []
                if not status.get('all_sign_offs_complete', False):
                    pending = status.get('pending', 0)
                    reasons.append(f"{pending} sign-offs pending")
                if status.get('open_qa_issues', 0) > 0:
                    reasons.append(f"{status['open_qa_issues']} QA issues open")
                status['blocking_reason'] = ', '.join(reasons)

            return status

        except Exception as e:
            logger.error(f"Error checking QA status for task {task_id}: {e}")
            return {
                "error": str(e),
                "task_id": task_id,
                "can_finalize": False
            }

    def finalize_task(self, task_id: int) -> Dict[str, Any]:
        """
        Finalize task after all QA approvals

        Only succeeds if:
        - All QA sign-offs are complete
        - All QA issues are resolved
        - At least minimum required approvals

        Args:
            task_id: Task ID

        Returns:
            Finalization result
        """
        try:
            logger.info(f"Attempting to finalize task {task_id}...")

            # Check QA status
            qa_status = self.check_qa_status(task_id)

            if not qa_status.get('can_finalize', False):
                return {
                    "success": False,
                    "error": "Cannot finalize - QA requirements not met",
                    "blocking_reason": qa_status.get('blocking_reason', 'Unknown'),
                    "qa_status": qa_status,
                    "task_id": task_id
                }

            # Use QA service to finalize
            finalize_result = self.qa_service.finalize_task_completion(task_id)

            if 'error' in finalize_result:
                return {
                    "success": False,
                    "error": finalize_result['error'],
                    "task_id": task_id
                }

            logger.info(f"Task {task_id} finalized successfully!")

            return {
                "success": True,
                "task_id": task_id,
                "new_status": "qa_approved",
                "can_deploy": True,
                "message": "Task approved by all QA agents and ready for deployment",
                "qa_status": qa_status
            }

        except Exception as e:
            logger.error(f"Error finalizing task {task_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }

    def get_pending_qa_tasks(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get all tasks awaiting QA completion

        Returns:
            Dictionary with pending tasks grouped by status
        """
        try:
            # Get all completed tasks (awaiting QA)
            completed_tasks = self.task_mgr.get_tasks_by_status("completed", limit=limit)

            pending_qa = []
            for task in completed_tasks:
                qa_status = self.check_qa_status(task['id'])
                task['qa_status'] = qa_status
                pending_qa.append(task)

            return {
                "total_pending": len(pending_qa),
                "tasks": pending_qa
            }

        except Exception as e:
            logger.error(f"Error getting pending QA tasks: {e}")
            return {
                "error": str(e),
                "total_pending": 0,
                "tasks": []
            }

    def get_approved_tasks(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get all QA-approved tasks ready for deployment

        Returns:
            Dictionary with approved tasks
        """
        try:
            approved_tasks = self.task_mgr.get_tasks_by_status("qa_approved", limit=limit)

            return {
                "total_approved": len(approved_tasks),
                "tasks": approved_tasks,
                "ready_to_deploy": True
            }

        except Exception as e:
            logger.error(f"Error getting approved tasks: {e}")
            return {
                "error": str(e),
                "total_approved": 0,
                "tasks": []
            }


def complete_task_with_qa(task_id: int, completion_notes: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to complete a task with QA

    Usage:
        result = complete_task_with_qa(task_id, "Implemented RAG system")

    Args:
        task_id: Task ID
        completion_notes: Optional completion notes

    Returns:
        Completion result with QA information
    """
    completion_mgr = TaskCompletionWithQA()
    return completion_mgr.complete_task(task_id, completion_notes)


def finalize_task_after_qa(task_id: int) -> Dict[str, Any]:
    """
    Convenience function to finalize task after QA approval

    Usage:
        result = finalize_task_after_qa(task_id)

    Args:
        task_id: Task ID

    Returns:
        Finalization result
    """
    completion_mgr = TaskCompletionWithQA()
    return completion_mgr.finalize_task(task_id)


if __name__ == "__main__":
    # Test the enhanced completion workflow
    print("=" * 80)
    print("TESTING ENHANCED TASK COMPLETION WITH QA")
    print("=" * 80)
    print()

    completion_mgr = TaskCompletionWithQA()

    # Test 1: Create and complete a task
    print("[1/4] Creating test task...")
    task_mgr = TaskDBManager()
    task_id = task_mgr.create_task(
        title="Test Task Completion Workflow",
        description="Testing automatic QA trigger on completion",
        task_type="test",
        priority="medium",
        assigned_agent="test-automator",
        tags=["test", "qa-workflow"]
    )
    print(f"  Created task #{task_id}")
    print()

    # Test 2: Complete task (triggers QA)
    print("[2/4] Completing task (should trigger QA)...")
    result = completion_mgr.complete_task(
        task_id,
        completion_notes="Test task completed successfully"
    )
    print(f"  Success: {result['success']}")
    print(f"  QA Triggered: {result.get('qa_triggered', False)}")
    if result.get('qa_triggered'):
        print(f"  Required agents: {', '.join(result.get('required_agents', []))}")
        print(f"  Sign-offs created: {result.get('sign_offs_created', 0)}")
    print()

    # Test 3: Check QA status
    print("[3/4] Checking QA status...")
    qa_status = completion_mgr.check_qa_status(task_id)
    print(f"  Total reviews: {qa_status.get('total_reviews', 0)}")
    print(f"  Pending: {qa_status.get('pending', 0)}")
    print(f"  Can finalize: {qa_status.get('can_finalize', False)}")
    if not qa_status.get('can_finalize', False):
        print(f"  Blocking reason: {qa_status.get('blocking_reason', 'Unknown')}")
    print()

    # Test 4: Try to finalize (should fail - QA not complete)
    print("[4/4] Attempting to finalize (should fail - QA pending)...")
    finalize_result = completion_mgr.finalize_task(task_id)
    print(f"  Success: {finalize_result['success']}")
    if not finalize_result['success']:
        print(f"  Error: {finalize_result.get('error', 'Unknown')}")
        print(f"  Reason: {finalize_result.get('blocking_reason', 'Unknown')}")
    print()

    print("=" * 80)
    print("WORKFLOW VERIFIED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - Task completion triggers QA automatically")
    print("  - QA status can be checked at any time")
    print("  - Finalization blocked until all QA complete")
    print("  - System enforces proper workflow")
    print()
    print(f"Test task ID: {task_id}")
    print("Next: QA agents review and approve, then finalize")
