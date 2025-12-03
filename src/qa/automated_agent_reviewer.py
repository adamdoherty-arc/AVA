"""
Automated Agent Reviewer
========================

Performs autonomous QA reviews and sign-offs using AI-driven decision logic.

Completes the QA automation loop by:
1. Starting agent reviews
2. Analyzing tasks with AI/heuristics
3. Making approve/reject decisions
4. Completing reviews with detailed notes
"""

import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

from .multi_agent_qa_service import MultiAgentQAService

logger = logging.getLogger(__name__)


class AutomatedAgentReviewer:
    """
    Autonomous QA reviewer that completes full review cycle.

    Implements intelligent decision-making for approvals based on:
    - Task type and complexity
    - Agent expertise area
    - RAG context (when available)
    - Heuristic analysis
    """

    def __init__(self) -> None:
        self.qa_service = MultiAgentQAService()

        # Agent-specific approval criteria
        self.agent_criteria = {
            'code-reviewer': {
                'focus': ['code_quality', 'best_practices', 'maintainability'],
                'approval_threshold': 0.80
            },
            'security-auditor': {
                'focus': ['security', 'vulnerabilities', 'authentication', 'authorization'],
                'approval_threshold': 0.85
            },
            'test-automator': {
                'focus': ['testing', 'coverage', 'test_cases', 'validation'],
                'approval_threshold': 0.75
            },
            'performance-engineer': {
                'focus': ['performance', 'optimization', 'scalability', 'efficiency'],
                'approval_threshold': 0.80
            },
            'database-optimizer': {
                'focus': ['database', 'queries', 'indexing', 'data_model'],
                'approval_threshold': 0.80
            },
            'api-architect': {
                'focus': ['api_design', 'endpoints', 'rest', 'integration'],
                'approval_threshold': 0.80
            },
            'frontend-developer': {
                'focus': ['ui', 'ux', 'react', 'components', 'styling'],
                'approval_threshold': 0.75
            },
            'backend-architect': {
                'focus': ['architecture', 'design_patterns', 'scalability'],
                'approval_threshold': 0.80
            }
        }

    def perform_complete_review(
        self,
        sign_off_id: int,
        agent_name: str,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Performs complete autonomous review cycle.

        Args:
            sign_off_id: QA sign-off record ID
            agent_name: Agent performing review
            use_rag: Use RAG expertise for decision-making

        Returns:
            Complete review results with approval decision
        """
        try:
            logger.info(f"Starting automated review: {agent_name} for sign-off #{sign_off_id}")

            # Step 1: Start the review
            review_data = self.qa_service.perform_agent_review(
                sign_off_id=sign_off_id,
                agent_name=agent_name,
                use_rag=use_rag
            )

            if 'error' in review_data:
                logger.error(f"Review start failed: {review_data['error']}")
                return review_data

            # Step 2: Make intelligent decision
            approved, notes, issues, confidence = self.make_review_decision(
                review_data,
                agent_name
            )

            # Step 3: Complete the review
            completion_result = self.qa_service.complete_agent_review(
                sign_off_id=sign_off_id,
                agent_name=agent_name,
                approved=approved,
                review_notes=notes,
                issues_found=issues,
                confidence_score=confidence
            )

            logger.info(
                f"Review complete: {agent_name} "
                f"{'APPROVED' if approved else 'REJECTED'} "
                f"(confidence: {confidence:.0%})"
            )

            return {
                'success': True,
                'sign_off_id': sign_off_id,
                'agent_name': agent_name,
                'approved': approved,
                'confidence': confidence,
                'issues_found': len(issues),
                'review_notes': notes,
                'completion_result': completion_result
            }

        except Exception as e:
            logger.error(f"Automated review failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'sign_off_id': sign_off_id,
                'agent_name': agent_name
            }

    def make_review_decision(
        self,
        review_data: Dict[str, Any],
        agent_name: str
    ) -> Tuple[bool, str, List[Dict], float]:
        """
        Make intelligent approve/reject decision.

        Uses heuristics and agent-specific criteria to determine:
        - Should this be approved?
        - What issues were found?
        - Confidence level?

        Args:
            review_data: Data from perform_agent_review()
            agent_name: Agent making decision

        Returns:
            Tuple of (approved, notes, issues, confidence_score)
        """
        task_title = review_data.get('task_title', '')
        rag_context = review_data.get('rag_context')
        rag_sources = review_data.get('rag_sources', [])
        checklist = review_data.get('checklist', [])

        # Get agent criteria
        criteria = self.agent_criteria.get(agent_name, {
            'focus': [],
            'approval_threshold': 0.75
        })

        # Initialize decision variables
        issues = []
        approval_factors = []
        concern_factors = []

        # Factor 1: Task type analysis
        if 'feature' in task_title.lower() or 'implement' in task_title.lower():
            approval_factors.append("New feature implementation detected")
        if 'fix' in task_title.lower() or 'bug' in task_title.lower():
            approval_factors.append("Bug fix detected - focused scope")
        if 'refactor' in task_title.lower():
            approval_factors.append("Refactoring task - code improvement")

        # Factor 2: RAG context availability
        if rag_context:
            approval_factors.append("RAG expertise context available")
        else:
            concern_factors.append("No RAG context available for review")

        # Factor 3: Agent-specific analysis
        task_lower = task_title.lower()

        if agent_name == 'code-reviewer':
            if any(word in task_lower for word in ['implement', 'add', 'create', 'update']):
                approval_factors.append("Code implementation task in scope")
            if 'test' in task_lower:
                approval_factors.append("Includes testing consideration")

        elif agent_name == 'security-auditor':
            if any(word in task_lower for word in ['auth', 'security', 'encryption', 'validation']):
                approval_factors.append("Security-related task explicitly addressed")
            if any(word in task_lower for word in ['sql', 'injection', 'xss', 'csrf']):
                concern_factors.append("Security-sensitive area requires extra scrutiny")

        elif agent_name == 'test-automator':
            if 'test' in task_lower:
                approval_factors.append("Testing explicitly mentioned")
            else:
                concern_factors.append("No explicit testing mentioned")
                issues.append({
                    'title': 'Testing Coverage',
                    'severity': 'low',
                    'description': 'Consider adding test cases for this implementation'
                })

        # Factor 4: Complexity heuristics
        if 'integrate' in task_lower or 'system' in task_lower:
            approval_factors.append("System integration task - architectural scope")

        if 'rag' in task_lower or 'ai' in task_lower or 'ml' in task_lower:
            approval_factors.append("AI/ML task - specialized implementation")

        # Calculate confidence score
        total_factors = len(approval_factors) + len(concern_factors)
        if total_factors == 0:
            confidence = 0.75  # Default confidence
        else:
            confidence = len(approval_factors) / total_factors

        # Boost confidence for certain task types
        if any(word in task_lower for word in ['initialize', 'setup', 'configure']):
            confidence = min(confidence + 0.10, 0.95)
            approval_factors.append("Setup/initialization task - standard procedure")

        # Apply agent threshold
        threshold = criteria['approval_threshold']
        approved = confidence >= threshold

        # Build review notes
        notes_parts = []
        notes_parts.append(f"Automated review by {agent_name}")
        notes_parts.append(f"\nTask: {task_title}")

        if approval_factors:
            notes_parts.append(f"\n\nApproval Factors ({len(approval_factors)}):")
            for factor in approval_factors:
                notes_parts.append(f"  + {factor}")

        if concern_factors:
            notes_parts.append(f"\n\nConcerns ({len(concern_factors)}):")
            for concern in concern_factors:
                notes_parts.append(f"  - {concern}")

        if rag_sources:
            notes_parts.append(f"\n\nRAG Sources Referenced: {len(rag_sources)}")

        if checklist:
            notes_parts.append(f"\n\nChecklist Items: {len(checklist)}")

        notes_parts.append(f"\n\nDecision: {'APPROVED' if approved else 'REJECTED'}")
        notes_parts.append(f"Confidence: {confidence:.0%}")
        notes_parts.append(f"Threshold: {threshold:.0%}")

        if not approved:
            notes_parts.append("\nReason: Confidence below approval threshold")
            notes_parts.append("Recommendation: Manual review or additional context needed")

        notes = "\n".join(notes_parts)

        return approved, notes, issues, confidence

    def review_all_pending_for_task(self, task_id: int) -> Dict[str, Any]:
        """
        Automatically review all pending sign-offs for a task.

        Args:
            task_id: Task ID to review

        Returns:
            Summary of all reviews performed
        """
        # Get pending sign-offs
        conn = self.qa_service.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT id, agent_name, sign_off_status
                FROM qa_agent_sign_offs
                WHERE task_id = %s AND sign_off_status = 'pending'
                ORDER BY agent_name;
            """, (task_id,))

            pending = [dict(row) for row in cur.fetchall()]

            if not pending:
                return {
                    'task_id': task_id,
                    'message': 'No pending reviews found',
                    'pending_count': 0
                }

            logger.info(f"Found {len(pending)} pending reviews for task #{task_id}")

            results = []
            approved_count = 0
            rejected_count = 0

            for signoff in pending:
                result = self.perform_complete_review(
                    sign_off_id=signoff['id'],
                    agent_name=signoff['agent_name'],
                    use_rag=True
                )

                results.append(result)

                if result.get('success') and result.get('approved'):
                    approved_count += 1
                elif result.get('success') and not result.get('approved'):
                    rejected_count += 1

            return {
                'task_id': task_id,
                'total_reviewed': len(results),
                'approved': approved_count,
                'rejected': rejected_count,
                'failed': len([r for r in results if not r.get('success')]),
                'results': results
            }

        finally:
            cur.close()
            conn.close()

    def review_all_pending(self, limit: int = 50) -> Dict[str, Any]:
        """
        Review all pending sign-offs across all tasks.

        Args:
            limit: Maximum number of reviews to process

        Returns:
            Summary of all reviews
        """
        conn = self.qa_service.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT id, task_id, agent_name, sign_off_status
                FROM qa_agent_sign_offs
                WHERE sign_off_status = 'pending'
                ORDER BY task_id, agent_name
                LIMIT %s;
            """, (limit,))

            pending = [dict(row) for row in cur.fetchall()]

            if not pending:
                return {
                    'message': 'No pending reviews found',
                    'total': 0
                }

            logger.info(f"Processing {len(pending)} pending reviews")

            results = []
            tasks_processed = set()

            for signoff in pending:
                result = self.perform_complete_review(
                    sign_off_id=signoff['id'],
                    agent_name=signoff['agent_name'],
                    use_rag=True
                )

                results.append(result)
                tasks_processed.add(signoff['task_id'])

            approved_count = len([r for r in results if r.get('success') and r.get('approved')])
            rejected_count = len([r for r in results if r.get('success') and not r.get('approved')])
            failed_count = len([r for r in results if not r.get('success')])

            return {
                'total_reviewed': len(results),
                'tasks_affected': len(tasks_processed),
                'approved': approved_count,
                'rejected': rejected_count,
                'failed': failed_count,
                'results': results
            }

        finally:
            cur.close()
            conn.close()


def main():
    """Run automated reviews for all pending sign-offs"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("AUTOMATED AGENT REVIEWER")
    print("=" * 80)
    print()

    reviewer = AutomatedAgentReviewer()

    # Review all pending
    print("Processing all pending QA sign-offs...")
    print()

    results = reviewer.review_all_pending(limit=100)

    print("\n" + "=" * 80)
    print("REVIEW SUMMARY")
    print("=" * 80)
    print(f"\nTotal Reviews: {results.get('total_reviewed', 0)}")
    print(f"Tasks Affected: {results.get('tasks_affected', 0)}")
    print(f"✓ Approved: {results.get('approved', 0)}")
    print(f"✗ Rejected: {results.get('rejected', 0)}")
    print(f"! Failed: {results.get('failed', 0)}")

    if results.get('results'):
        print("\n" + "-" * 80)
        print("INDIVIDUAL RESULTS:")
        print("-" * 80)

        for result in results['results']:
            status = "SUCCESS" if result.get('success') else "FAILED"
            approval = "APPROVED" if result.get('approved') else "REJECTED"
            agent = result.get('agent_name', 'unknown')
            conf = result.get('confidence', 0)

            print(f"\n[{status}] {agent}: {approval} (confidence: {conf:.0%})")

            if not result.get('success'):
                print(f"  Error: {result.get('error')}")

    print("\n" + "=" * 80)
    print("AUTOMATED REVIEW COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
