"""
Legion Task Creator for AVA Feedback
=====================================

Automatically creates Legion tasks from high-frequency unanswered questions.
Ensures continuous improvement of Magnus by tracking what AVA can't handle.

Features:
- Monitors unanswered questions
- Creates Legion tasks when threshold reached
- Syncs status back to AVA memory
- Prevents duplicate task creation
- Prioritizes by frequency and recency

Author: Magnus Trading Platform
Created: 2025-11-11
"""

import logging
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ava.conversation_memory_manager import ConversationMemoryManager
from src.legion.legion_task_sync_service import LegionTaskSyncService
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegionTaskCreator:
    """Creates Legion tasks from AVA feedback and unanswered questions"""

    def __init__(self) -> None:
        """Initialize task creator with memory and Legion sync"""
        self.memory = ConversationMemoryManager()
        self.legion_sync = LegionTaskSyncService()

        # Thresholds for task creation
        self.thresholds = {
            'critical': 10,  # 10+ occurrences = immediate task
            'high': 5,       # 5-9 occurrences = high priority
            'medium': 3      # 3-4 occurrences = medium priority
        }

    def scan_and_create_tasks(self, dry_run: bool = False) -> List[Dict]:
        """
        Scan unanswered questions and create Legion tasks as needed

        Args:
            dry_run: If True, only report what would be created

        Returns:
            List of created tasks
        """
        logger.info("=" * 80)
        logger.info("SCANNING UNANSWERED QUESTIONS FOR LEGION TASK CREATION")
        logger.info("=" * 80)

        # Get questions that need tasks
        questions_needing_tasks = self.memory.get_unanswered_questions_needing_tasks(
            min_occurrences=self.thresholds['medium']
        )

        if not questions_needing_tasks:
            logger.info("No unanswered questions need tasks yet")
            return []

        logger.info(f"Found {len(questions_needing_tasks)} questions that need Legion tasks")

        created_tasks = []

        for question in questions_needing_tasks:
            try:
                # Determine priority
                priority = self._determine_priority(question['occurrence_count'])

                # Generate task details
                task_title, task_description = self._generate_task_details(question)

                logger.info(f"\n[Question {question['question_id']}]")
                logger.info(f"Question: {question['user_question']}")
                logger.info(f"Occurrences: {question['occurrence_count']}")
                logger.info(f"Priority: {priority}")
                logger.info(f"Task Title: {task_title}")

                if dry_run:
                    logger.info("[DRY RUN] Would create Legion task")
                    created_tasks.append({
                        'question_id': question['question_id'],
                        'task_title': task_title,
                        'priority': priority,
                        'dry_run': True
                    })
                    continue

                # Create Legion task
                legion_task_id = self._create_legion_task(
                    title=task_title,
                    description=task_description,
                    priority=priority,
                    task_type='enhancement',
                    feature_area='AVA'
                )

                if legion_task_id:
                    # Mark question as having task created
                    self._mark_task_created(
                        question_id=question['question_id'],
                        legion_task_id=legion_task_id,
                        task_title=task_title,
                        priority=priority,
                        occurrence_count=question['occurrence_count']
                    )

                    logger.info(f"[OK] Created Legion task #{legion_task_id}")

                    created_tasks.append({
                        'question_id': question['question_id'],
                        'legion_task_id': legion_task_id,
                        'task_title': task_title,
                        'priority': priority
                    })

            except Exception as e:
                logger.error(f"Error creating task for question {question['question_id']}: {e}")
                continue

        logger.info(f"\n{'=' * 80}")
        logger.info(f"SUMMARY: Created {len(created_tasks)} Legion tasks")
        logger.info(f"{'=' * 80}\n")

        return created_tasks

    def _determine_priority(self, occurrence_count: int) -> str:
        """Determine task priority based on occurrence frequency"""
        if occurrence_count >= self.thresholds['critical']:
            return 'critical'
        elif occurrence_count >= self.thresholds['high']:
            return 'high'
        elif occurrence_count >= self.thresholds['medium']:
            return 'medium'
        else:
            return 'low'

    def _generate_task_details(self, question: Dict) -> tuple:
        """
        Generate task title and description from question

        Returns:
            (task_title, task_description)
        """
        user_question = question['user_question']
        failure_reason = question['failure_reason']
        occurrence_count = question['occurrence_count']

        # Generate task title
        task_title = f"AVA Enhancement: {user_question[:80]}"

        # Generate detailed description
        task_description = f"""# AVA Unanswered Question - {occurrence_count} Occurrences

## User Question
**"{user_question}"**

## Failure Details
- **Reason:** {failure_reason}
- **Times Asked:** {occurrence_count}
- **Last Occurred:** {question['last_occurred_at']}
- **Question ID:** {question['question_id']}

## Context
Users have asked this question {occurrence_count} time(s), but AVA was unable to provide a satisfactory answer.

## Required Actions

### Investigation Phase
1. Review why AVA couldn't answer this
2. Determine if it's:
   - Missing data/integration
   - Unsupported feature
   - Low confidence in existing logic
   - Technical limitation

### Implementation Phase
Based on failure reason (`{failure_reason}`):

**If "no_data":**
- Add necessary data source integration
- Update data sync schedules
- Test data availability

**If "unsupported_feature":**
- Design and implement the feature
- Add to AVA's capabilities
- Update knowledge base

**If "low_confidence":**
- Improve NLP intent detection
- Add more training examples
- Refine entity extraction

**If "error":**
- Fix the underlying bug
- Add error handling
- Improve logging

### Testing Phase
1. Test with original user question
2. Verify AVA can now answer correctly
3. Update conversation memory
4. Mark question as resolved

### QA Requirements (Legion Multi-Agent QA)
This task must be reviewed by Legion's QA agents:
- **code-reviewer**: Code quality and maintainability
- **security-auditor**: No security vulnerabilities
- **test-automator**: Adequate test coverage

**All agents must approve before task completion.**

## Success Criteria
- [x] AVA can successfully answer: "{user_question}"
- [x] Response has confidence >= 0.80
- [x] Action completes without errors
- [x] User satisfaction improves
- [x] Question marked as resolved in ava_unanswered_questions table
- [x] All Legion QA agents approve

## Related Tables
- `ava_unanswered_questions` (question_id: {question['question_id']})
- `ava_legion_task_log` (will track this task)

## Metrics to Track
- Reduction in occurrence_count for this question
- Increase in AVA confidence scores for similar queries
- User satisfaction ratings after fix

---

**Auto-created by AVA Feedback System**
**Creation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return task_title, task_description

    def _create_legion_task(
        self,
        title: str,
        description: str,
        priority: str,
        task_type: str,
        feature_area: str
    ) -> Optional[int]:
        """
        Create task in Legion system

        Returns:
            legion_task_id or None if failed
        """
        try:
            # Use Legion sync service to create task
            # This will create task in Legion database
            task_data = {
                'title': title,
                'description': description,
                'task_type': task_type,
                'priority': priority,
                'status': 'pending',
                'assigned_agent': 'auto',  # Will be assigned by Legion
                'feature_area': feature_area,
                'source': 'AVA_FEEDBACK',
                'created_by': 'AVA_AUTO',
                'project_name': 'Magnus'
            }

            # For now, create directly in Magnus development_tasks table
            # Later, this will sync to Legion
            import psycopg2
            from psycopg2.extras import Json

            conn = self.memory.get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO development_tasks (
                        title, description, task_type, priority, status,
                        assigned_agent, feature_area, source,
                        metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s
                    )
                    RETURNING task_id
                """, (
                    task_data['title'],
                    task_data['description'],
                    task_data['task_type'],
                    task_data['priority'],
                    task_data['status'],
                    task_data['assigned_agent'],
                    task_data['feature_area'],
                    'AVA_FEEDBACK',
                    Json({'auto_created': True, 'from_ava_feedback': True})
                ))

                task_id = cur.fetchone()[0]
                conn.commit()
                conn.close()

                return task_id

        except Exception as e:
            logger.error(f"Error creating Legion task: {e}")
            return None

    def _mark_task_created(
        self,
        question_id: int,
        legion_task_id: int,
        task_title: str,
        priority: str,
        occurrence_count: int
    ):
        """Mark question as having Legion task created"""
        try:
            conn = self.memory.get_connection()
            with conn.cursor() as cur:
                # Update unanswered_questions table
                cur.execute("""
                    UPDATE ava_unanswered_questions
                    SET legion_task_created = TRUE,
                        legion_task_id = %s,
                        task_created_at = NOW(),
                        updated_at = NOW()
                    WHERE question_id = %s
                """, (legion_task_id, question_id))

                # Log in task creation log
                cur.execute("""
                    INSERT INTO ava_legion_task_log (
                        question_id, legion_task_id, task_title, task_priority,
                        creation_reason, occurrence_count, task_status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    question_id, legion_task_id, task_title, priority,
                    'High-frequency unanswered question', occurrence_count,
                    'pending'
                ))

                conn.commit()
                conn.close()

        except Exception as e:
            logger.error(f"Error marking task created: {e}")

    def check_and_resolve_questions(self) -> List[Dict]:
        """
        Check if any questions can now be answered (tasks completed)
        Mark them as resolved
        """
        try:
            conn = self.memory.get_connection()
            with conn.cursor() as cur:
                # Find questions with completed tasks
                cur.execute("""
                    SELECT
                        uq.question_id,
                        uq.user_question,
                        uq.legion_task_id,
                        dt.status
                    FROM ava_unanswered_questions uq
                    JOIN development_tasks dt ON uq.legion_task_id = dt.task_id
                    WHERE uq.legion_task_created = TRUE
                      AND NOT uq.resolved
                      AND dt.status = 'completed'
                """)

                completed_questions = cur.fetchall()

                resolved_count = 0
                for row in completed_questions:
                    question_id, user_question, task_id, status = row

                    # Mark as resolved
                    cur.execute("""
                        UPDATE ava_unanswered_questions
                        SET resolved = TRUE,
                            resolved_at = NOW(),
                            resolution_notes = 'Resolved via Legion task #' || %s,
                            updated_at = NOW()
                        WHERE question_id = %s
                    """, (task_id, question_id))

                    logger.info(f"[RESOLVED] Question #{question_id}: {user_question}")
                    resolved_count += 1

                if resolved_count > 0:
                    conn.commit()
                    logger.info(f"Resolved {resolved_count} questions")

                conn.close()

                return completed_questions

        except Exception as e:
            logger.error(f"Error checking resolved questions: {e}")
            return []


def main():
    """Main execution for task creator"""
    import argparse

    parser = argparse.ArgumentParser(description='Create Legion tasks from AVA feedback')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be created without actually creating')
    parser.add_argument('--resolve', action='store_true', help='Check and resolve completed questions')
    args = parser.parse_args()

    creator = LegionTaskCreator()

    if args.resolve:
        # Check for resolved questions
        resolved = creator.check_and_resolve_questions()
        print(f"\nResolved {len(resolved)} questions")
    else:
        # Create tasks
        tasks = creator.scan_and_create_tasks(dry_run=args.dry_run)
        print(f"\n{'Would create' if args.dry_run else 'Created'} {len(tasks)} Legion tasks")

        if tasks:
            print("\nTasks:")
            for task in tasks:
                print(f"  - [{task['priority'].upper()}] {task['task_title']}")
                if not args.dry_run:
                    print(f"    Task ID: {task['legion_task_id']}")


if __name__ == "__main__":
    main()
