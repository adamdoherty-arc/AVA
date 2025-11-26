#!/usr/bin/env python3
"""
Magnus QA Runner

Main entry point for the continuous QA and enhancement system.
Runs every 20 minutes, invokes Claude Code with --dangerously-skip-permissions,
performs comprehensive checks, and logs accomplishments (additive only).

Usage:
    # Run once
    python qa_runner.py --once

    # Run continuously (every 20 minutes)
    python qa_runner.py --daemon

    # Run with visible console output
    python qa_runner.py --daemon --visible

    # Custom interval
    python qa_runner.py --daemon --interval 30
"""

import os
import sys
import time
import json
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from .accomplishments_tracker import AccomplishmentsTracker, RunSummary, get_tracker
from .telegram_notifier import get_qa_notifier, send_critical_alert
from .checks.base_check import CheckRunner, CheckPriority, ModuleCheckResult


class QARunner:
    """
    Main QA runner that coordinates all checks and enhancements.

    Runs every 20 minutes (configurable) and:
    1. Executes all check modules by priority
    2. Applies safe auto-fixes
    3. Logs accomplishments (additive only)
    4. Sends Telegram notifications for critical issues
    5. Updates health score
    """

    def __init__(self, interval_minutes: int = 20, visible: bool = False):
        """
        Initialize the QA runner.

        Args:
            interval_minutes: Time between QA cycles (default 20)
            visible: Show visible console output
        """
        self.interval_minutes = interval_minutes
        self.visible = visible
        self.project_root = PROJECT_ROOT
        self.running = False
        self._stop_event = threading.Event()

        # Core components
        self.tracker = get_tracker()
        self.notifier = get_qa_notifier()

        # Check modules (will be populated)
        self.check_modules = []

        # Statistics
        self.total_cycles = 0
        self.total_issues_found = 0
        self.total_issues_fixed = 0

        logger.info(f"QA Runner initialized (interval: {interval_minutes} min)")

    def load_check_modules(self):
        """Load all available check modules."""
        from .checks.base_check import BaseCheck

        # Import check modules dynamically
        check_classes = []

        # Try to import each check module
        try:
            from .checks.api_endpoints import APIEndpointsCheck
            check_classes.append(APIEndpointsCheck)
        except ImportError as e:
            logger.warning(f"Could not load api_endpoints check: {e}")

        try:
            from .checks.api_connectivity import APIConnectivityCheck
            check_classes.append(APIConnectivityCheck)
        except ImportError as e:
            logger.warning(f"Could not load api_connectivity check: {e}")

        try:
            from .checks.database_health import DatabaseHealthCheck
            check_classes.append(DatabaseHealthCheck)
        except ImportError as e:
            logger.warning(f"Could not load database_health check: {e}")

        try:
            from .checks.code_quality import CodeQualityCheck
            check_classes.append(CodeQualityCheck)
        except ImportError as e:
            logger.warning(f"Could not load code_quality check: {e}")

        try:
            from .checks.dummy_data_detector import DummyDataDetectorCheck
            check_classes.append(DummyDataDetectorCheck)
        except ImportError as e:
            logger.warning(f"Could not load dummy_data_detector check: {e}")

        try:
            from .checks.ui_components import UIComponentsCheck
            check_classes.append(UIComponentsCheck)
        except ImportError as e:
            logger.warning(f"Could not load ui_components check: {e}")

        # Instantiate check modules
        self.check_modules = [cls() for cls in check_classes]

        logger.info(f"Loaded {len(self.check_modules)} check modules")

    def run_once(self) -> Dict[str, Any]:
        """
        Execute a single QA cycle.

        Returns:
            Dict with cycle results
        """
        start_time = datetime.now()
        run_id = self.tracker.start_run()

        self._log(f"Starting QA Cycle {run_id}...")
        self._log(f"Project root: {self.project_root}")

        # Results tracking
        all_results: List[ModuleCheckResult] = []
        issues_found = 0
        issues_fixed = 0
        critical_failures = 0
        health_score_before = self._get_current_health_score()

        try:
            # Load check modules if not already loaded
            if not self.check_modules:
                self.load_check_modules()

            if not self.check_modules:
                self._log("No check modules available. Running Claude Code review...")
                # Fall back to Claude Code invocation
                claude_result = self._invoke_claude_review()
                if claude_result:
                    self._process_claude_result(claude_result)
            else:
                # Run checks using CheckRunner
                runner = CheckRunner(self.check_modules)
                all_results = runner.run_all(stop_on_critical_failure=True)

                # Process results
                for result in all_results:
                    issues_found += result.total_failed
                    issues_fixed += result.fixes_applied

                    if result.has_critical_failure:
                        critical_failures += 1
                        self._handle_critical_failure(result)

                    # Log individual accomplishments
                    self._log_module_results(result)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Calculate new health score
            health_score_after = self._calculate_health_score(all_results)

            # Update statistics
            self.total_cycles += 1
            self.total_issues_found += issues_found
            self.total_issues_fixed += issues_fixed

            # End run and log summary
            summary = RunSummary(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
                checks_performed=sum(len(r.results) for r in all_results),
                issues_found=issues_found,
                issues_fixed=issues_fixed,
                health_score_before=health_score_before,
                health_score_after=health_score_after,
                critical_failures=critical_failures,
            )
            self.tracker.end_run(summary)

            # Send Telegram summary if enabled
            self._send_cycle_summary(summary)

            # Check health score threshold
            if health_score_after < 70 and self.notifier.is_available():
                self.notifier.send_health_warning(
                    health_score_after, health_score_before
                )

            self._log(
                f"Cycle complete: {issues_found} issues, {issues_fixed} fixed, "
                f"health: {health_score_after:.1f}/100 ({duration:.1f}s)"
            )

            return {
                "run_id": run_id,
                "duration_seconds": duration,
                "issues_found": issues_found,
                "issues_fixed": issues_fixed,
                "critical_failures": critical_failures,
                "health_score": health_score_after,
                "success": critical_failures == 0,
            }

        except Exception as e:
            logger.error(f"QA cycle failed with error: {e}", exc_info=True)
            send_critical_alert(
                f"QA cycle {run_id} failed: {str(e)}",
                module="qa_runner",
                details={"error": str(e)},
            )
            return {
                "run_id": run_id,
                "success": False,
                "error": str(e),
            }

    def run_continuous(self):
        """Run QA cycles continuously every interval_minutes."""
        self.running = True
        self._stop_event.clear()

        self._log(f"Starting continuous QA (every {self.interval_minutes} minutes)")
        self._log("Press Ctrl+C to stop")

        try:
            while not self._stop_event.is_set():
                # Run a cycle
                result = self.run_once()

                if self._stop_event.is_set():
                    break

                # Calculate next run time
                next_run = datetime.now() + timedelta(minutes=self.interval_minutes)
                self._log(f"Next run at: {next_run.strftime('%H:%M:%S')}")

                # Wait for next cycle (checking stop event periodically)
                wait_seconds = self.interval_minutes * 60
                for _ in range(wait_seconds):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            self._log("Received shutdown signal")
        finally:
            self.running = False
            self._log("QA Runner stopped")

    def stop(self):
        """Stop the continuous runner."""
        self._stop_event.set()

    # =========================================================================
    # Claude Code Integration
    # =========================================================================

    def _invoke_claude_review(self) -> Optional[str]:
        """
        Invoke Claude Code for comprehensive review.

        Returns:
            Claude's output or None if failed
        """
        prompt = self._build_review_prompt()

        try:
            result = subprocess.run(
                ["claude", "--dangerously-skip-permissions", "--print", prompt],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=600,  # 10 minute timeout
            )

            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Claude Code returned error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Claude Code timed out")
            return None
        except FileNotFoundError:
            logger.error("Claude CLI not found. Is Claude Code installed?")
            return None
        except Exception as e:
            logger.error(f"Failed to invoke Claude Code: {e}")
            return None

    def _build_review_prompt(self) -> str:
        """Build the review prompt for Claude Code."""
        return """
You are reviewing the Magnus financial trading platform. Perform a COMPREHENSIVE review.

## Focus Areas

1. **API Endpoints** - Test all 28+ FastAPI routes return real data (no mocks)
2. **React Components** - Verify frontend builds and all routes work
3. **Code Quality** - Find dead code, DRY violations, security issues
4. **Performance** - Identify bottlenecks, inefficient patterns
5. **Data Sync** - Verify data freshness from all sources
6. **Dummy Data** - Find and flag any mock/dummy data patterns

## Output Format

For each issue found, report:
```
[SEVERITY: CRITICAL|HIGH|MEDIUM|LOW]
FILE: <filepath>
LINE: <if applicable>
ISSUE: <description>
FIX: <recommendation>
AUTO_FIXABLE: true|false
```

## Summary

At the end, provide:
1. Total issues by severity
2. Top 3 urgent fixes
3. Health score (1-100)
4. List of accomplishments if any auto-fixes were applied
"""

    def _process_claude_result(self, output: str):
        """Process Claude Code output and log accomplishments."""
        # Parse output for issues and fixes
        lines = output.split('\n')
        current_issue = {}

        for line in lines:
            line = line.strip()
            if line.startswith('[SEVERITY:'):
                if current_issue:
                    self._log_claude_issue(current_issue)
                current_issue = {'severity': line.split(':')[1].strip().rstrip(']')}
            elif line.startswith('FILE:'):
                current_issue['file'] = line.split(':', 1)[1].strip()
            elif line.startswith('ISSUE:'):
                current_issue['issue'] = line.split(':', 1)[1].strip()
            elif line.startswith('FIX:'):
                current_issue['fix'] = line.split(':', 1)[1].strip()
            elif line.startswith('AUTO_FIXABLE:'):
                current_issue['auto_fixable'] = 'true' in line.lower()

        if current_issue:
            self._log_claude_issue(current_issue)

    def _log_claude_issue(self, issue: Dict):
        """Log a parsed Claude issue to accomplishments."""
        severity = issue.get('severity', 'MEDIUM').lower()
        file_path = issue.get('file', '')
        description = issue.get('issue', 'Unknown issue')

        self.tracker.log_issue(
            module="claude_review",
            message=description,
            severity=severity if severity in ('info', 'warning', 'error', 'critical') else 'warning',
            details=issue,
            files=[file_path] if file_path else [],
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _log(self, message: str):
        """Log message to console if visible mode enabled."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"[{timestamp}] {message}"

        logger.info(message)
        if self.visible:
            print(formatted)

    def _handle_critical_failure(self, result: ModuleCheckResult):
        """Handle a critical failure by sending alert."""
        failed_checks = [r for r in result.results if r.is_critical_failure]

        for check in failed_checks:
            send_critical_alert(
                check.message,
                module=result.module_name,
                details=check.details,
            )

    def _log_module_results(self, result: ModuleCheckResult):
        """Log individual check results as accomplishments."""
        for check_result in result.results:
            if check_result.fix_applied:
                self.tracker.log_auto_fix(
                    module=result.module_name,
                    message=check_result.fix_description or check_result.message,
                    files=check_result.files_affected,
                    details=check_result.details,
                )
            elif not check_result.passed:
                self.tracker.log_issue(
                    module=result.module_name,
                    message=check_result.message,
                    severity="error" if check_result.is_critical_failure else "warning",
                    details=check_result.details,
                    files=check_result.files_affected,
                )

    def _get_current_health_score(self) -> float:
        """Get current health score from status file."""
        status_path = self.tracker.status_path
        try:
            if status_path.exists():
                with open(status_path, 'r') as f:
                    status = json.load(f)
                    return status.get('health_score', 80.0)
        except Exception:
            pass
        return 80.0  # Default

    def _calculate_health_score(self, results: List[ModuleCheckResult]) -> float:
        """
        Calculate health score from check results.

        Scoring:
        - Start at 100
        - -10 for each critical failure
        - -5 for each high priority failure
        - -2 for each medium priority failure
        - -1 for each low priority failure
        - +1 for each auto-fix applied
        """
        score = 100.0

        for result in results:
            for check in result.results:
                if not check.passed:
                    if result.priority == CheckPriority.CRITICAL:
                        score -= 10
                    elif result.priority == CheckPriority.HIGH:
                        score -= 5
                    elif result.priority == CheckPriority.MEDIUM:
                        score -= 2
                    else:
                        score -= 1

                if check.fix_applied:
                    score += 1

        return max(0.0, min(100.0, score))

    def _send_cycle_summary(self, summary: RunSummary):
        """Send Telegram summary if conditions met."""
        # Send if critical failures or significant issues
        if summary.critical_failures > 0 or summary.issues_found >= 5:
            self.notifier.send_qa_cycle_summary({
                'run_id': summary.run_id,
                'duration_seconds': summary.duration_seconds,
                'checks_performed': summary.checks_performed,
                'issues_found': summary.issues_found,
                'issues_fixed': summary.issues_fixed,
                'health_score': summary.health_score_after,
                'critical_failures': summary.critical_failures,
            })


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Magnus QA Runner')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')
    parser.add_argument('--daemon', action='store_true',
                        help='Run continuously')
    parser.add_argument('--interval', type=int, default=20,
                        help='Interval between runs in minutes (default: 20)')
    parser.add_argument('--visible', action='store_true',
                        help='Show visible console output')

    args = parser.parse_args()

    runner = QARunner(
        interval_minutes=args.interval,
        visible=args.visible or not args.daemon,
    )

    if args.once:
        result = runner.run_once()
        sys.exit(0 if result.get('success', False) else 1)
    elif args.daemon:
        runner.run_continuous()
    else:
        # Default: run once with visible output
        runner.visible = True
        result = runner.run_once()
        sys.exit(0 if result.get('success', False) else 1)


if __name__ == "__main__":
    main()
