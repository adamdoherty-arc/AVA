"""
Background Task Infrastructure

Provides:
- Async task queue for long-running operations
- Progress tracking and status updates
- Task cancellation support
- Memory-based queue (Redis-backed in production)
"""

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a background task"""
    task_id: str
    status: TaskStatus
    progress: float = 0.0  # 0-100
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata
        }


class BackgroundTaskManager:
    """
    Manages background task execution.

    Features:
    - Async task execution
    - Progress tracking
    - Task cancellation
    - Result caching

    Usage:
        manager = BackgroundTaskManager()

        # Submit a task
        task_id = await manager.submit(analyze_portfolio, user_id="123")

        # Check status
        status = await manager.get_status(task_id)

        # Get result when complete
        if status.status == TaskStatus.COMPLETED:
            result = status.result
    """

    def __init__(self, max_concurrent: int = 10, result_ttl: int = 3600):
        self._tasks: Dict[str, TaskResult] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent
        self._result_ttl = result_ttl
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()

    async def submit(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for background execution.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for the function
            task_id: Optional custom task ID
            metadata: Optional metadata to store with task

        Returns:
            Task ID for tracking
        """
        task_id = task_id or str(uuid.uuid4())

        # Create task result
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            metadata=metadata or {}
        )

        async with self._lock:
            self._tasks[task_id] = task_result

        # Start execution in background
        asyncio.create_task(self._execute_task(task_id, func, *args, **kwargs))

        logger.info(f"Task {task_id} submitted: {func.__name__}")
        return task_id

    async def _execute_task(self, task_id: str, func: Callable, *args, **kwargs):
        """Execute a task with semaphore limiting"""
        async with self._semaphore:
            task_result = self._tasks.get(task_id)
            if not task_result:
                return

            # Update status to running
            task_result.status = TaskStatus.RUNNING
            task_result.started_at = datetime.now()

            try:
                # Create cancellable task
                if asyncio.iscoroutinefunction(func):
                    coro = func(*args, **kwargs)
                else:
                    coro = asyncio.to_thread(func, *args, **kwargs)

                async_task = asyncio.create_task(coro)

                async with self._lock:
                    self._running_tasks[task_id] = async_task

                result = await async_task

                # Success
                task_result.status = TaskStatus.COMPLETED
                task_result.result = result
                task_result.progress = 100.0
                task_result.completed_at = datetime.now()

                logger.info(f"Task {task_id} completed successfully")

            except asyncio.CancelledError:
                task_result.status = TaskStatus.CANCELLED
                task_result.completed_at = datetime.now()
                logger.info(f"Task {task_id} was cancelled")

            except Exception as e:
                task_result.status = TaskStatus.FAILED
                task_result.error = str(e)
                task_result.completed_at = datetime.now()
                logger.error(f"Task {task_id} failed: {e}")

            finally:
                async with self._lock:
                    self._running_tasks.pop(task_id, None)

    async def get_status(self, task_id: str) -> Optional[TaskResult]:
        """Get current status of a task"""
        return self._tasks.get(task_id)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task"""
        async with self._lock:
            async_task = self._running_tasks.get(task_id)
            if async_task and not async_task.done():
                async_task.cancel()
                logger.info(f"Task {task_id} cancellation requested")
                return True
            return False

    async def update_progress(self, task_id: str, progress: float, metadata: Optional[Dict] = None):
        """Update task progress (call from within task function)"""
        task_result = self._tasks.get(task_id)
        if task_result:
            task_result.progress = min(100.0, max(0.0, progress))
            if metadata:
                task_result.metadata.update(metadata)

    async def list_tasks(self, status: Optional[TaskStatus] = None) -> List[TaskResult]:
        """List all tasks, optionally filtered by status"""
        if status:
            return [t for t in self._tasks.values() if t.status == status]
        return list(self._tasks.values())

    async def cleanup_old_tasks(self) -> int:
        """Remove completed tasks older than TTL"""
        cutoff = datetime.now()
        removed = 0

        async with self._lock:
            to_remove = []
            for task_id, task_result in self._tasks.items():
                if task_result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    if task_result.completed_at:
                        age = (cutoff - task_result.completed_at).total_seconds()
                        if age > self._result_ttl:
                            to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old tasks")

        return removed

    def get_stats(self) -> dict:
        """Get task manager statistics"""
        by_status = {}
        for status in TaskStatus:
            by_status[status.value] = sum(1 for t in self._tasks.values() if t.status == status)

        return {
            "total_tasks": len(self._tasks),
            "running_tasks": len(self._running_tasks),
            "max_concurrent": self._max_concurrent,
            "by_status": by_status
        }


# =============================================================================
# Background Task Decorator
# =============================================================================

_task_manager: Optional[BackgroundTaskManager] = None


def get_task_manager() -> BackgroundTaskManager:
    """Get global task manager instance"""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


def background_task(metadata: Optional[Dict] = None):
    """
    Decorator to run a function as a background task.

    Usage:
        @background_task(metadata={"type": "analysis"})
        async def analyze_portfolio(user_id: str):
            # Long-running operation
            return results

        # Call returns task_id immediately
        task_id = await analyze_portfolio(user_id="123")
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> str:
            manager = get_task_manager()
            return await manager.submit(func, *args, metadata=metadata, **kwargs)
        return wrapper
    return decorator


class ProgressReporter:
    """
    Helper class for reporting progress within a background task.

    Usage:
        async def my_task(task_id: str, reporter: ProgressReporter):
            for i in range(100):
                await do_work()
                await reporter.update(i + 1, {"current_item": i})
    """

    def __init__(self, task_id: str, manager: Optional[BackgroundTaskManager] = None):
        self.task_id = task_id
        self._manager = manager or get_task_manager()

    async def update(self, progress: float, metadata: Optional[Dict] = None):
        """Update progress (0-100)"""
        await self._manager.update_progress(self.task_id, progress, metadata)
