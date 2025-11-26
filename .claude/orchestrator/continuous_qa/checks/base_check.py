"""
Base Check Module

Abstract base class for all QA check modules.
Defines the interface and priority system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckPriority(Enum):
    """
    Check execution priority.

    CRITICAL: Run first, sequentially. Stop on failure.
    HIGH: Run second, parallel (max 5 concurrent).
    MEDIUM: Run third, parallel (max 10 concurrent).
    LOW: Run last, parallel (max 3 concurrent), non-blocking.
    """
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class CheckStatus(Enum):
    """Status of a check result."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class CheckResult:
    """Result of a single check within a module."""
    check_name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    auto_fixable: bool = False
    fix_applied: bool = False
    fix_description: str = ""
    files_affected: List[str] = field(default_factory=list)
    execution_time_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def passed(self) -> bool:
        """Check if the result indicates success."""
        return self.status in (CheckStatus.PASSED, CheckStatus.WARNING)

    @property
    def is_critical_failure(self) -> bool:
        """Check if this is a critical failure that should stop execution."""
        return self.status == CheckStatus.FAILED


@dataclass
class ModuleCheckResult:
    """Aggregated result from a check module."""
    module_name: str
    priority: CheckPriority
    results: List[CheckResult] = field(default_factory=list)
    total_execution_time_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def passed(self) -> bool:
        """Check if all individual checks passed."""
        return all(r.passed for r in self.results)

    @property
    def total_passed(self) -> int:
        """Count of passed checks."""
        return sum(1 for r in self.results if r.passed)

    @property
    def total_failed(self) -> int:
        """Count of failed checks."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def fixes_applied(self) -> int:
        """Count of auto-fixes applied."""
        return sum(1 for r in self.results if r.fix_applied)

    @property
    def has_critical_failure(self) -> bool:
        """Check if any result is a critical failure."""
        return any(r.is_critical_failure for r in self.results)


class BaseCheck(ABC):
    """
    Abstract base class for all check modules.

    Subclasses must implement:
    - name: The module name
    - priority: When to run (CRITICAL, HIGH, MEDIUM, LOW)
    - run(): Execute all checks and return results
    - get_checks_list(): List of individual check names
    """

    def __init__(self):
        """Initialize the check module."""
        self._results: List[CheckResult] = []
        self._start_time: Optional[datetime] = None
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Module name for logging and identification."""
        pass

    @property
    @abstractmethod
    def priority(self) -> CheckPriority:
        """Execution priority of this module."""
        pass

    @abstractmethod
    def run(self) -> ModuleCheckResult:
        """
        Execute all checks in this module.

        Returns:
            ModuleCheckResult containing all individual check results.
        """
        pass

    @abstractmethod
    def get_checks_list(self) -> List[str]:
        """
        Get list of individual check names.

        Returns:
            List of check names that this module will run.
        """
        pass

    def can_auto_fix(self, check_name: str) -> bool:
        """
        Check if a specific check can be auto-fixed.

        Override in subclass to enable auto-fixing.

        Args:
            check_name: Name of the specific check

        Returns:
            True if auto-fix is supported for this check.
        """
        return False

    def apply_fix(self, check_name: str, result: CheckResult) -> bool:
        """
        Apply an auto-fix for a failed check.

        Override in subclass to implement auto-fixing.

        Args:
            check_name: Name of the check to fix
            result: The failed check result

        Returns:
            True if fix was successfully applied.
        """
        return False

    # Helper methods for subclasses

    def _start_module(self):
        """Call at start of run() to initialize timing."""
        self._results = []
        self._start_time = datetime.now()
        self.logger.info(f"Starting {self.name} checks...")

    def _end_module(self) -> ModuleCheckResult:
        """Call at end of run() to finalize and return results."""
        elapsed = (datetime.now() - self._start_time).total_seconds() * 1000 \
            if self._start_time else 0

        result = ModuleCheckResult(
            module_name=self.name,
            priority=self.priority,
            results=self._results,
            total_execution_time_ms=int(elapsed),
        )

        self.logger.info(
            f"Completed {self.name}: {result.total_passed} passed, "
            f"{result.total_failed} failed, {result.fixes_applied} fixed "
            f"({elapsed:.0f}ms)"
        )

        return result

    def _add_result(self, check_name: str, status: CheckStatus, message: str,
                    details: Dict = None, auto_fixable: bool = False,
                    fix_applied: bool = False, fix_description: str = "",
                    files: List[str] = None):
        """Add a check result to the module results."""
        self._results.append(CheckResult(
            check_name=check_name,
            status=status,
            message=message,
            details=details or {},
            auto_fixable=auto_fixable,
            fix_applied=fix_applied,
            fix_description=fix_description,
            files_affected=files or [],
        ))

    def _pass(self, check_name: str, message: str, details: Dict = None):
        """Shorthand to add a passing result."""
        self._add_result(check_name, CheckStatus.PASSED, message, details)

    def _fail(self, check_name: str, message: str, details: Dict = None,
              auto_fixable: bool = False, files: List[str] = None):
        """Shorthand to add a failing result."""
        self._add_result(check_name, CheckStatus.FAILED, message, details,
                         auto_fixable=auto_fixable, files=files)

    def _warn(self, check_name: str, message: str, details: Dict = None,
              files: List[str] = None):
        """Shorthand to add a warning result."""
        self._add_result(check_name, CheckStatus.WARNING, message, details,
                         files=files)

    def _error(self, check_name: str, message: str, details: Dict = None):
        """Shorthand to add an error result (check couldn't run)."""
        self._add_result(check_name, CheckStatus.ERROR, message, details)

    def _skip(self, check_name: str, message: str):
        """Shorthand to skip a check."""
        self._add_result(check_name, CheckStatus.SKIPPED, message)

    def _fixed(self, check_name: str, message: str, fix_description: str,
               files: List[str], details: Dict = None):
        """Shorthand to record a successful auto-fix."""
        self._add_result(
            check_name, CheckStatus.PASSED, message, details,
            auto_fixable=True, fix_applied=True,
            fix_description=fix_description, files=files
        )


class CheckRunner:
    """
    Runs check modules according to their priority.

    Priority execution order:
    1. CRITICAL - Sequential, stop on failure
    2. HIGH - Parallel (max 5)
    3. MEDIUM - Parallel (max 10)
    4. LOW - Parallel (max 3)
    """

    def __init__(self, checks: List[BaseCheck]):
        """Initialize with list of check modules."""
        self.checks = checks
        self.logger = logging.getLogger(__name__)

    def run_all(self, stop_on_critical_failure: bool = True) -> List[ModuleCheckResult]:
        """
        Run all checks in priority order.

        Args:
            stop_on_critical_failure: If True, stop if a CRITICAL check fails.

        Returns:
            List of all module results.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []

        # Group checks by priority
        by_priority = {p: [] for p in CheckPriority}
        for check in self.checks:
            by_priority[check.priority].append(check)

        # Run CRITICAL checks sequentially
        self.logger.info("Running CRITICAL checks...")
        for check in by_priority[CheckPriority.CRITICAL]:
            result = check.run()
            results.append(result)
            if result.has_critical_failure and stop_on_critical_failure:
                self.logger.error(
                    f"CRITICAL check {check.name} failed - stopping execution"
                )
                return results

        # Run HIGH priority in parallel (max 5)
        results.extend(self._run_parallel(
            by_priority[CheckPriority.HIGH],
            max_workers=5,
            label="HIGH"
        ))

        # Run MEDIUM priority in parallel (max 10)
        results.extend(self._run_parallel(
            by_priority[CheckPriority.MEDIUM],
            max_workers=10,
            label="MEDIUM"
        ))

        # Run LOW priority in parallel (max 3)
        results.extend(self._run_parallel(
            by_priority[CheckPriority.LOW],
            max_workers=3,
            label="LOW"
        ))

        return results

    def _run_parallel(self, checks: List[BaseCheck], max_workers: int,
                      label: str) -> List[ModuleCheckResult]:
        """Run checks in parallel with limited concurrency."""
        if not checks:
            return []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.logger.info(f"Running {len(checks)} {label} priority checks...")
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(c.run): c for c in checks}
            for future in as_completed(futures):
                check = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Check {check.name} raised exception: {e}")
                    results.append(ModuleCheckResult(
                        module_name=check.name,
                        priority=check.priority,
                        results=[CheckResult(
                            check_name="module_execution",
                            status=CheckStatus.ERROR,
                            message=f"Module raised exception: {e}",
                        )],
                    ))

        return results
