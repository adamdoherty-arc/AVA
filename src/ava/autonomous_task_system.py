"""
Autonomous Task System for AVA
==============================

This system enables AVA to recognize task commands from users and execute them autonomously.
When a user says "task: do something", AVA will:
1. Parse and understand the task request
2. Store it in the database
3. Assign it to an appropriate executor agent
4. Execute the task autonomously (code updates, UI changes, system config, etc.)
5. Report back to the user

Usage in chat:
- "task: fix the login button"
- "task: add a dark mode toggle to settings"
- "task: update the database connection timeout"
- "task: create a new endpoint for user preferences"
"""

import os
import re
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

# Database imports
try:
    from src.task_db_manager import TaskDBManager
    TASK_DB_AVAILABLE = True
except ImportError:
    TASK_DB_AVAILABLE = False
    TaskDBManager = None

# LLM imports for task interpretation
try:
    from src.magnus_local_llm import MagnusLocalLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    MagnusLocalLLM = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks the system can execute"""
    CODE_UPDATE = "code_update"          # Modify existing code
    CODE_CREATE = "code_create"          # Create new files/modules
    UI_CHANGE = "ui_change"              # Frontend/UI modifications
    DATABASE_CHANGE = "database_change"  # Schema or data changes
    CONFIG_UPDATE = "config_update"      # Configuration changes
    DOCUMENTATION = "documentation"       # Docs/README updates
    BUG_FIX = "bug_fix"                  # Bug fixes
    FEATURE = "feature"                  # New features
    REFACTOR = "refactor"                # Code refactoring
    RESEARCH = "research"                # Research/analysis tasks
    GENERAL = "general"                  # Catch-all for other tasks


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class ParsedTask:
    """Represents a parsed task from user input"""
    raw_input: str
    task_type: TaskType
    description: str
    priority: TaskPriority
    target_files: List[str] = field(default_factory=list)
    feature_area: str = "general"
    estimated_complexity: str = "medium"
    requires_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecutionResult:
    """Result of task execution"""
    task_id: int
    success: bool
    message: str
    files_modified: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    error_details: Optional[str] = None
    output: Optional[str] = None


class TaskRecognizer:
    """Recognizes and parses task commands from user input"""

    # Patterns to recognize task commands
    TASK_PATTERNS = [
        r'^task:\s*(.+)$',                    # "task: do something"
        r'^task\s+(.+)$',                     # "task do something"
        r'^create\s+task:\s*(.+)$',           # "create task: do something"
        r'^add\s+task:\s*(.+)$',              # "add task: do something"
        r'^please\s+task:\s*(.+)$',           # "please task: do something"
        r'^i\s+need\s+you\s+to\s+task:\s*(.+)$',  # "I need you to task: ..."
    ]

    # Keywords to detect task type
    TYPE_KEYWORDS = {
        TaskType.CODE_UPDATE: ['update', 'modify', 'change', 'edit', 'fix in', 'adjust'],
        TaskType.CODE_CREATE: ['create', 'add new', 'build', 'implement', 'write new'],
        TaskType.UI_CHANGE: ['ui', 'interface', 'button', 'page', 'component', 'style', 'css', 'frontend', 'display', 'show'],
        TaskType.DATABASE_CHANGE: ['database', 'schema', 'table', 'column', 'migration', 'sql', 'query'],
        TaskType.CONFIG_UPDATE: ['config', 'setting', 'environment', 'env', 'parameter', 'option'],
        TaskType.DOCUMENTATION: ['document', 'readme', 'doc', 'guide', 'tutorial', 'comment'],
        TaskType.BUG_FIX: ['fix', 'bug', 'error', 'issue', 'problem', 'broken', 'crash', 'not working'],
        TaskType.FEATURE: ['feature', 'add', 'new functionality', 'enhance', 'improvement'],
        TaskType.REFACTOR: ['refactor', 'clean up', 'optimize', 'improve code', 'reorganize'],
        TaskType.RESEARCH: ['research', 'analyze', 'investigate', 'find out', 'look into'],
    }

    # Priority keywords
    PRIORITY_KEYWORDS = {
        TaskPriority.CRITICAL: ['critical', 'urgent', 'asap', 'immediately', 'emergency', 'now'],
        TaskPriority.HIGH: ['high priority', 'important', 'soon', 'quickly'],
        TaskPriority.LOW: ['low priority', 'when possible', 'eventually', 'someday', 'backlog'],
    }

    def __init__(self) -> None:
        self.llm = MagnusLocalLLM() if LLM_AVAILABLE else None

    def is_task_command(self, message: str) -> bool:
        """Check if the message is a task command"""
        message_lower = message.strip().lower()

        for pattern in self.TASK_PATTERNS:
            if re.match(pattern, message_lower, re.IGNORECASE):
                return True

        return False

    def extract_task_content(self, message: str) -> Optional[str]:
        """Extract the task description from a task command"""
        message_stripped = message.strip()

        for pattern in self.TASK_PATTERNS:
            match = re.match(pattern, message_stripped, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def detect_task_type(self, description: str) -> TaskType:
        """Detect the type of task from the description"""
        desc_lower = description.lower()

        # Check for keyword matches
        for task_type, keywords in self.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return task_type

        # Default to general
        return TaskType.GENERAL

    def detect_priority(self, description: str) -> TaskPriority:
        """Detect task priority from the description"""
        desc_lower = description.lower()

        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return priority

        # Default to medium
        return TaskPriority.MEDIUM

    def detect_feature_area(self, description: str) -> str:
        """Detect which feature area the task belongs to"""
        desc_lower = description.lower()

        feature_areas = {
            'dashboard': ['dashboard', 'main page', 'home'],
            'ava': ['ava', 'chatbot', 'assistant', 'chat'],
            'options': ['option', 'premium', 'put', 'call', 'strike'],
            'positions': ['position', 'portfolio', 'holdings'],
            'earnings': ['earnings', 'calendar'],
            'watchlist': ['watchlist', 'watch list'],
            'settings': ['setting', 'config', 'preference'],
            'database': ['database', 'db', 'postgres', 'sql'],
            'api': ['api', 'endpoint', 'route', 'backend'],
            'frontend': ['frontend', 'react', 'ui', 'component'],
        }

        for area, keywords in feature_areas.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return area

        return 'general'

    def detect_target_files(self, description: str) -> List[str]:
        """Try to detect which files might be affected"""
        files = []

        # Look for explicit file references
        file_pattern = r'[\w/\\]+\.\w{1,5}'
        matches = re.findall(file_pattern, description)
        files.extend(matches)

        return files

    def parse_task(self, message: str) -> Optional[ParsedTask]:
        """Parse a task command into a structured ParsedTask"""
        task_content = self.extract_task_content(message)
        if not task_content:
            return None

        # Use LLM for enhanced parsing if available
        if self.llm:
            enhanced_parse = self._llm_enhanced_parse(task_content)
            if enhanced_parse:
                return enhanced_parse

        # Fallback to rule-based parsing
        return ParsedTask(
            raw_input=message,
            task_type=self.detect_task_type(task_content),
            description=task_content,
            priority=self.detect_priority(task_content),
            target_files=self.detect_target_files(task_content),
            feature_area=self.detect_feature_area(task_content),
            estimated_complexity="medium"
        )

    def _llm_enhanced_parse(self, task_content: str) -> Optional[ParsedTask]:
        """Use LLM to better understand the task"""
        try:
            prompt = f"""Analyze this task request and extract structured information.

Task: {task_content}

Respond in JSON format:
{{
    "task_type": "code_update|code_create|ui_change|database_change|config_update|documentation|bug_fix|feature|refactor|research|general",
    "description": "Clear description of what needs to be done",
    "priority": "critical|high|medium|low",
    "feature_area": "dashboard|ava|options|positions|earnings|watchlist|settings|database|api|frontend|general",
    "complexity": "simple|medium|complex",
    "potential_files": ["list of files that might need changes"],
    "requires_approval": true/false (true if this could break things)
}}"""

            response = self.llm.query(prompt)
            if response:
                # Parse JSON from response
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())

                    return ParsedTask(
                        raw_input=task_content,
                        task_type=TaskType(data.get('task_type', 'general')),
                        description=data.get('description', task_content),
                        priority=TaskPriority(data.get('priority', 'medium')),
                        target_files=data.get('potential_files', []),
                        feature_area=data.get('feature_area', 'general'),
                        estimated_complexity=data.get('complexity', 'medium'),
                        requires_approval=data.get('requires_approval', False)
                    )
        except Exception as e:
            logger.warning(f"LLM parsing failed, using rule-based: {e}")

        return None


class TaskExecutor:
    """Executes tasks autonomously"""

    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.llm = MagnusLocalLLM() if LLM_AVAILABLE else None
        self.db = TaskDBManager() if TASK_DB_AVAILABLE else None

    def execute_task(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a parsed task"""
        start_time = datetime.now()

        try:
            # Update status to in_progress
            if self.db:
                self.db.update_task_status(task_id, TaskStatus.IN_PROGRESS.value)
                self.db.log_execution(
                    task_id=task_id,
                    agent_name="autonomous_executor",
                    action_type="started",
                    message=f"Starting execution of {task.task_type.value} task"
                )

            # Execute based on task type
            result = self._execute_by_type(task, task_id)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time_seconds = execution_time

            # Update database with result
            if self.db:
                if result.success:
                    self.db.update_task_status(task_id, TaskStatus.COMPLETED.value)
                    self.db.log_execution(
                        task_id=task_id,
                        agent_name="autonomous_executor",
                        action_type="completed",
                        message=result.message,
                        files_modified=result.files_modified,
                        duration_seconds=int(execution_time)
                    )
                else:
                    self.db.update_task_status(task_id, TaskStatus.FAILED.value)
                    self.db.log_execution(
                        task_id=task_id,
                        agent_name="autonomous_executor",
                        action_type="failed",
                        message=result.message,
                        error_details=result.error_details,
                        duration_seconds=int(execution_time)
                    )

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Task execution error: {e}")

            if self.db:
                self.db.update_task_status(task_id, TaskStatus.FAILED.value)
                self.db.log_execution(
                    task_id=task_id,
                    agent_name="autonomous_executor",
                    action_type="failed",
                    message=str(e),
                    error_details=str(e),
                    duration_seconds=int(execution_time)
                )

            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                message=f"Execution failed: {e}",
                error_details=str(e),
                execution_time_seconds=execution_time
            )

    def _execute_by_type(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Route execution based on task type"""

        if task.task_type == TaskType.CODE_UPDATE:
            return self._execute_code_update(task, task_id)
        elif task.task_type == TaskType.CODE_CREATE:
            return self._execute_code_create(task, task_id)
        elif task.task_type == TaskType.UI_CHANGE:
            return self._execute_ui_change(task, task_id)
        elif task.task_type == TaskType.CONFIG_UPDATE:
            return self._execute_config_update(task, task_id)
        elif task.task_type == TaskType.DATABASE_CHANGE:
            return self._execute_database_change(task, task_id)
        elif task.task_type == TaskType.BUG_FIX:
            return self._execute_bug_fix(task, task_id)
        elif task.task_type == TaskType.FEATURE:
            return self._execute_feature(task, task_id)
        elif task.task_type == TaskType.DOCUMENTATION:
            return self._execute_documentation(task, task_id)
        elif task.task_type == TaskType.RESEARCH:
            return self._execute_research(task, task_id)
        else:
            return self._execute_general(task, task_id)

    def _generate_code_changes(self, task: ParsedTask) -> Optional[Dict[str, str]]:
        """Use LLM to generate code changes for a task"""
        if not self.llm:
            return None

        prompt = f"""You are an expert software developer. Generate the code changes needed for this task.

Task Type: {task.task_type.value}
Description: {task.description}
Feature Area: {task.feature_area}
Target Files: {', '.join(task.target_files) if task.target_files else 'Auto-detect'}

Project Context:
- This is a Python/React trading dashboard application
- Backend: FastAPI with PostgreSQL
- Frontend: React with TypeScript
- Uses Streamlit for some pages
- Key directories: src/, frontend/src/, backend/

Please provide:
1. File path (relative to project root)
2. The code changes (full file content or diff)
3. Brief explanation of changes

Respond in JSON format:
{{
    "changes": [
        {{
            "file_path": "path/to/file.py",
            "action": "modify|create|delete",
            "content": "full file content or changes",
            "explanation": "why this change is needed"
        }}
    ],
    "summary": "Overall summary of changes"
}}"""

        try:
            response = self.llm.query(prompt)
            if response:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Error generating code changes: {e}")

        return None

    def _apply_code_changes(self, changes: Dict[str, Any], task_id: int) -> Tuple[bool, List[str], str]:
        """Apply generated code changes to files"""
        modified_files = []
        messages = []

        try:
            for change in changes.get('changes', []):
                file_path = change.get('file_path', '')
                action = change.get('action', 'modify')
                content = change.get('content', '')

                full_path = os.path.join(self.project_root, file_path)

                if action == 'create':
                    # Create new file
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    modified_files.append(file_path)
                    messages.append(f"Created: {file_path}")

                    # Track in database
                    if self.db:
                        self.db.track_file_change(
                            task_id=task_id,
                            file_path=file_path,
                            change_type='created',
                            lines_added=len(content.split('\n'))
                        )

                elif action == 'modify':
                    # Read existing file
                    existing_content = ''
                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            existing_content = f.read()

                    # Write modified content
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    modified_files.append(file_path)
                    messages.append(f"Modified: {file_path}")

                    # Track in database
                    if self.db:
                        old_lines = len(existing_content.split('\n'))
                        new_lines = len(content.split('\n'))
                        self.db.track_file_change(
                            task_id=task_id,
                            file_path=file_path,
                            change_type='modified',
                            lines_added=max(0, new_lines - old_lines),
                            lines_removed=max(0, old_lines - new_lines)
                        )

                elif action == 'delete':
                    if os.path.exists(full_path):
                        os.remove(full_path)
                        modified_files.append(file_path)
                        messages.append(f"Deleted: {file_path}")

                        if self.db:
                            self.db.track_file_change(
                                task_id=task_id,
                                file_path=file_path,
                                change_type='deleted'
                            )

            summary = changes.get('summary', 'Changes applied successfully')
            return True, modified_files, f"{summary}\n" + "\n".join(messages)

        except Exception as e:
            logger.error(f"Error applying code changes: {e}")
            return False, modified_files, str(e)

    def _execute_code_update(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a code update task"""
        changes = self._generate_code_changes(task)

        if not changes:
            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                message="Could not generate code changes. LLM may not be available.",
                error_details="LLM generation failed"
            )

        success, files, message = self._apply_code_changes(changes, task_id)

        return TaskExecutionResult(
            task_id=task_id,
            success=success,
            message=message,
            files_modified=files,
            output=json.dumps(changes, indent=2)
        )

    def _execute_code_create(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a code creation task"""
        return self._execute_code_update(task, task_id)

    def _execute_ui_change(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a UI change task"""
        return self._execute_code_update(task, task_id)

    def _execute_config_update(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a config update task"""
        return self._execute_code_update(task, task_id)

    def _execute_database_change(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a database change task"""
        # For database changes, we should be more careful
        if task.requires_approval:
            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                message="Database changes require explicit approval. Please review and approve manually.",
                error_details="Awaiting approval for database changes"
            )

        return self._execute_code_update(task, task_id)

    def _execute_bug_fix(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a bug fix task"""
        return self._execute_code_update(task, task_id)

    def _execute_feature(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a feature task"""
        return self._execute_code_update(task, task_id)

    def _execute_documentation(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a documentation task"""
        return self._execute_code_update(task, task_id)

    def _execute_research(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a research task"""
        if not self.llm:
            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                message="Research tasks require LLM to be available",
                error_details="LLM not available"
            )

        prompt = f"""You are a software research assistant. Analyze and research the following:

Task: {task.description}
Context: {task.feature_area}

Provide a comprehensive analysis including:
1. Current state of the codebase relevant to this research
2. Potential approaches or solutions
3. Recommendations
4. Any risks or considerations

Be thorough and technical."""

        try:
            response = self.llm.query(prompt)

            if self.db:
                self.db.log_execution(
                    task_id=task_id,
                    agent_name="autonomous_executor",
                    action_type="research_complete",
                    message=f"Research completed: {len(response)} chars of analysis"
                )

            return TaskExecutionResult(
                task_id=task_id,
                success=True,
                message="Research completed successfully",
                output=response
            )
        except Exception as e:
            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                message=f"Research failed: {e}",
                error_details=str(e)
            )

    def _execute_general(self, task: ParsedTask, task_id: int) -> TaskExecutionResult:
        """Execute a general task"""
        return self._execute_code_update(task, task_id)


class AutonomousTaskSystem:
    """
    Main interface for the autonomous task system.
    Integrates with AVA chatbot to handle "task:" commands.
    """

    def __init__(self, auto_execute: bool = True):
        """
        Initialize the autonomous task system.

        Args:
            auto_execute: If True, tasks are executed immediately.
                         If False, tasks are queued for later execution.
        """
        self.recognizer = TaskRecognizer()
        self.executor = TaskExecutor()
        self.db = TaskDBManager() if TASK_DB_AVAILABLE else None
        self.auto_execute = auto_execute

    def process_message(self, message: str, user_id: str = "user") -> Dict[str, Any]:
        """
        Process a user message and check if it's a task command.

        Args:
            message: User's input message
            user_id: Identifier for the user

        Returns:
            Dict with processing results
        """
        # Check if this is a task command
        if not self.recognizer.is_task_command(message):
            return {
                "is_task": False,
                "message": None
            }

        # Parse the task
        parsed_task = self.recognizer.parse_task(message)
        if not parsed_task:
            return {
                "is_task": True,
                "success": False,
                "message": "Could not parse the task. Please try rephrasing."
            }

        # Store in database
        task_id = self._store_task(parsed_task, user_id)
        if not task_id:
            return {
                "is_task": True,
                "success": False,
                "message": "Failed to create task in database."
            }

        # Execute or queue based on settings
        if self.auto_execute and not parsed_task.requires_approval:
            result = self.executor.execute_task(parsed_task, task_id)

            return {
                "is_task": True,
                "success": result.success,
                "task_id": task_id,
                "task_type": parsed_task.task_type.value,
                "message": result.message,
                "files_modified": result.files_modified,
                "execution_time": result.execution_time_seconds,
                "output": result.output
            }
        else:
            # Task queued for later
            return {
                "is_task": True,
                "success": True,
                "task_id": task_id,
                "task_type": parsed_task.task_type.value,
                "message": f"Task #{task_id} created and queued for execution. "
                          f"Type: {parsed_task.task_type.value}, Priority: {parsed_task.priority.value}",
                "requires_approval": parsed_task.requires_approval
            }

    def _store_task(self, task: ParsedTask, user_id: str) -> Optional[int]:
        """Store a parsed task in the database"""
        if not self.db:
            logger.warning("Database not available, task will not be persisted")
            return None

        try:
            task_id = self.db.create_task(
                title=task.description[:100],  # First 100 chars as title
                description=task.description,
                task_type=task.task_type.value,
                priority=task.priority.value,
                assigned_agent="autonomous_executor",
                feature_area=task.feature_area,
                tags=["autonomous", task.estimated_complexity],
                created_by=user_id
            )

            return task_id

        except Exception as e:
            logger.error(f"Failed to store task: {e}")
            return None

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get all pending tasks"""
        if not self.db:
            return []

        return self.db.get_tasks_by_status(TaskStatus.PENDING.value)

    def get_task_status(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if not self.db:
            return None

        return self.db.get_task(task_id)

    def execute_pending_tasks(self) -> List[TaskExecutionResult]:
        """Execute all pending tasks"""
        results = []
        pending = self.get_pending_tasks()

        for task_data in pending:
            # Reconstruct ParsedTask from database data
            parsed = ParsedTask(
                raw_input=task_data.get('description', ''),
                task_type=TaskType(task_data.get('task_type', 'general')),
                description=task_data.get('description', ''),
                priority=TaskPriority(task_data.get('priority', 'medium')),
                feature_area=task_data.get('feature_area', 'general')
            )

            result = self.executor.execute_task(parsed, task_data['id'])
            results.append(result)

        return results

    def cancel_task(self, task_id: int) -> bool:
        """Cancel a pending task"""
        if not self.db:
            return False

        return self.db.update_task_status(task_id, TaskStatus.CANCELLED.value)

    def format_response(self, result: Dict[str, Any]) -> str:
        """Format task result for chat response"""
        if not result.get('is_task'):
            return ""

        if not result.get('success'):
            return f"Task creation/execution failed: {result.get('message', 'Unknown error')}"

        response_parts = [
            f"Task #{result.get('task_id', '?')} - {result.get('task_type', 'unknown').replace('_', ' ').title()}"
        ]

        if result.get('files_modified'):
            response_parts.append(f"\nFiles modified: {', '.join(result['files_modified'])}")

        if result.get('execution_time'):
            response_parts.append(f"\nCompleted in {result['execution_time']:.1f} seconds")

        response_parts.append(f"\n{result.get('message', '')}")

        if result.get('output') and len(result['output']) < 500:
            response_parts.append(f"\n\nDetails:\n{result['output']}")

        return "\n".join(response_parts)


# Singleton instance for easy access
_task_system_instance: Optional[AutonomousTaskSystem] = None


def get_task_system(auto_execute: bool = True) -> AutonomousTaskSystem:
    """Get or create the singleton task system instance"""
    global _task_system_instance

    if _task_system_instance is None:
        _task_system_instance = AutonomousTaskSystem(auto_execute=auto_execute)

    return _task_system_instance


# Example usage
if __name__ == "__main__":
    print("Testing Autonomous Task System...")

    # Create task system
    system = AutonomousTaskSystem(auto_execute=False)

    # Test messages
    test_messages = [
        "task: fix the login button on the dashboard",
        "task: add a dark mode toggle to settings page",
        "task: update the database connection timeout to 30 seconds",
        "task: create a new API endpoint for user preferences",
        "Hello, how are you?",  # Not a task
        "task: research how to optimize the premium scanner queries",
    ]

    for msg in test_messages:
        print(f"\n{'='*50}")
        print(f"Message: {msg}")
        result = system.process_message(msg, user_id="test_user")
        print(f"Is Task: {result.get('is_task')}")
        if result.get('is_task'):
            print(f"Task ID: {result.get('task_id')}")
            print(f"Type: {result.get('task_type')}")
            print(f"Response: {result.get('message')}")
