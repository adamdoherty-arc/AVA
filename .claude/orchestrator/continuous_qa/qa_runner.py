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

# Import QA Issues Database Manager
try:
    from src.qa_issues_db_manager import QAIssuesDBManager
    QA_DB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"QA Issues Database Manager not available: {e}")
    QA_DB_AVAILABLE = False

# Import SpecAgent system
try:
    from src.spec_agents import SpecAgentOrchestrator, SpecAgentRegistry
    from src.spec_agents.agents import (
        PositionsSpecAgent,
        DashboardSpecAgent,
        PremiumScannerSpecAgent,
        OptionsAnalysisSpecAgent,
        GameCardsSpecAgent,
    )
    SPEC_AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SpecAgent system not available: {e}")
    SPEC_AGENTS_AVAILABLE = False


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

        # SpecAgent orchestrator
        self.spec_orchestrator = None

        # Database manager for persistent issue tracking
        self.qa_db = None
        if QA_DB_AVAILABLE:
            try:
                self.qa_db = QAIssuesDBManager()
                if self.qa_db.check_schema_exists():
                    logger.info("QA Issues Database connected")
                else:
                    logger.warning("QA Issues Database schema not found - will log to files only")
                    self.qa_db = None
            except Exception as e:
                logger.warning(f"Could not connect to QA Issues Database: {e}")
                self.qa_db = None

        logger.info(f"QA Runner initialized (interval: {interval_minutes} min)")

    def load_check_modules(self) -> None:
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

        try:
            from .checks.data_sync import DataSyncCheck
            check_classes.append(DataSyncCheck)
        except ImportError as e:
            logger.warning(f"Could not load data_sync check: {e}")

        try:
            from .checks.shared_code_analyzer import SharedCodeAnalyzerCheck
            check_classes.append(SharedCodeAnalyzerCheck)
        except ImportError as e:
            logger.warning(f"Could not load shared_code_analyzer check: {e}")

        # Instantiate check modules
        self.check_modules = [cls() for cls in check_classes]

        logger.info(f"Loaded {len(self.check_modules)} check modules")

    def load_enhancement_engine(self) -> None:
        """Load the enhancement engine for proactive improvements."""
        try:
            from .enhancement_engine import EnhancementEngine
            self.enhancement_engine = EnhancementEngine(self.project_root)
            logger.info("Enhancement engine loaded")
        except ImportError as e:
            logger.warning(f"Could not load enhancement engine: {e}")
            self.enhancement_engine = None

    def load_pattern_learner(self) -> None:
        """Load the pattern learner for recurring issue detection."""
        try:
            from .pattern_learner import PatternLearner
            self.pattern_learner = PatternLearner(self.project_root)
            logger.info("Pattern learner loaded")
        except ImportError as e:
            logger.warning(f"Could not load pattern learner: {e}")
            self.pattern_learner = None

    def load_spec_agents(self) -> None:
        """Load the SpecAgent system for feature-specific testing."""
        if not SPEC_AGENTS_AVAILABLE:
            logger.warning("SpecAgent system not available")
            return

        try:
            self.spec_orchestrator = SpecAgentOrchestrator(max_parallel=3)
            agent_count = SpecAgentRegistry.get_agent_count()
            logger.info(f"SpecAgent system loaded with {agent_count} agents")
        except Exception as e:
            logger.error(f"Failed to load SpecAgent system: {e}")
            self.spec_orchestrator = None

    async def run_spec_agents(self) -> Dict[str, Any]:
        """
        Run all SpecAgents for comprehensive feature testing.

        Returns:
            Dict with SpecAgent results
        """
        if not self.spec_orchestrator:
            self.load_spec_agents()

        if not self.spec_orchestrator:
            return {'error': 'SpecAgent system not available', 'issues': []}

        self._log("Running SpecAgents for feature testing...")

        try:
            # Run priority agents (Tier 1: critical features)
            results = await self.spec_orchestrator.run_priority_agents()

            # Log issues to accomplishments
            for issue in results.get('issues', []):
                severity = issue.get('severity', 'medium')
                self.tracker.log_issue(
                    module=f"spec_agent:{issue.get('feature', 'unknown')}",
                    message=issue.get('title', 'Unknown issue'),
                    severity=severity if severity in ('info', 'warning', 'error', 'critical') else 'warning',
                    details=issue,
                    files=[issue.get('file_path')] if issue.get('file_path') else [],
                )

            # Report critical issues to chatbot
            if results.get('critical_issues', 0) > 0:
                await self.spec_orchestrator.report_to_chatbot(results)

            self._log(
                f"SpecAgents complete: {results.get('total_passed', 0)}/{results.get('total_tests', 0)} passed, "
                f"{results.get('total_issues', 0)} issues found"
            )

            return results

        except Exception as e:
            logger.error(f"SpecAgent execution failed: {e}")
            return {'error': str(e), 'issues': []}

    def _run_spec_agents_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for running SpecAgents."""
        import asyncio

        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.run_spec_agents())
                    return future.result(timeout=300)
            else:
                return loop.run_until_complete(self.run_spec_agents())
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.run_spec_agents())

    def run_once(self) -> Dict[str, Any]:
        """
        Execute a single QA cycle.

        Returns:
            Dict with cycle results
        """
        start_time = datetime.now()
        run_id = self.tracker.start_run()

        # Start database run tracking
        db_run_id = None
        if self.qa_db:
            try:
                db_run_id = self.qa_db.start_run(
                    run_id=run_id,
                    triggered_by="scheduler"
                )
                self._log(f"Database run started: {db_run_id}")
            except Exception as e:
                logger.warning(f"Could not start database run: {e}")
                db_run_id = None

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

            # Load enhancement engine and pattern learner
            if not hasattr(self, 'enhancement_engine') or self.enhancement_engine is None:
                self.load_enhancement_engine()
            if not hasattr(self, 'pattern_learner') or self.pattern_learner is None:
                self.load_pattern_learner()

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
                    self._log_module_results(result, db_run_id=db_run_id)

                # If issues found but not fixed, invoke Claude Code to fix them
                unfixed_issues = issues_found - issues_fixed
                if unfixed_issues > 0:
                    self._log(f"Found {unfixed_issues} unfixed issues. Invoking Claude Code to fix...")
                    claude_fixes = self._invoke_claude_fixer(all_results)
                    if claude_fixes > 0:
                        issues_fixed += claude_fixes
                        self._log(f"Claude Code fixed {claude_fixes} additional issues")

            # Run enhancement engine for proactive improvements
            enhancement_results = None
            if hasattr(self, 'enhancement_engine') and self.enhancement_engine:
                self._log("Running enhancement engine...")
                enhancement_results = self.enhancement_engine.run()
                if enhancement_results.total_applied > 0:
                    self._log(f"Applied {enhancement_results.total_applied} enhancements")
                    issues_fixed += enhancement_results.total_applied
                    # Log enhancements as accomplishments
                    for enhancement in enhancement_results.enhancements:
                        if enhancement.applied:
                            self.tracker.log_auto_fix(
                                module="enhancement_engine",
                                message=f"{enhancement.category}: {enhancement.description}",
                                files=[enhancement.file_path],
                                details={'category': enhancement.category},
                            )

            # Run pattern learner for recurring issue detection
            if hasattr(self, 'pattern_learner') and self.pattern_learner:
                self._log("Running pattern learner...")
                pattern_matches = self.pattern_learner.scan_codebase()
                if pattern_matches:
                    self._log(f"Found {len(pattern_matches)} pattern matches")
                    stats = self.pattern_learner.get_statistics()
                    hot_spots = stats.get('hot_spots', [])
                    if hot_spots:
                        self._log(f"Hot spot files: {[h['file'] for h in hot_spots[:3]]}")

            # Run SpecAgents for feature-specific testing
            spec_agent_results = None
            if SPEC_AGENTS_AVAILABLE:
                self._log("Running SpecAgents for feature testing...")
                spec_agent_results = self._run_spec_agents_sync()
                if spec_agent_results:
                    spec_issues = len(spec_agent_results.get('issues', []))
                    spec_critical = spec_agent_results.get('critical_issues', 0)
                    issues_found += spec_issues
                    critical_failures += spec_critical
                    self._log(f"SpecAgents found {spec_issues} issues ({spec_critical} critical)")

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

            # Complete database run tracking
            if self.qa_db and db_run_id:
                try:
                    # Count issues by severity
                    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                    total_checks = 0
                    passed_checks = 0
                    failed_checks = 0
                    warned_checks = 0
                    skipped_checks = 0

                    for result in all_results:
                        for check in result.results:
                            total_checks += 1
                            if check.passed:
                                passed_checks += 1
                            elif check.skipped:
                                skipped_checks += 1
                            elif check.is_critical_failure:
                                failed_checks += 1
                                severity_counts['critical'] += 1
                            elif hasattr(result, 'priority') and result.priority == CheckPriority.HIGH:
                                failed_checks += 1
                                severity_counts['high'] += 1
                            else:
                                warned_checks += 1
                                severity_counts['medium'] += 1

                    self.qa_db.complete_run(
                        run_db_id=db_run_id,
                        health_score=health_score_after,
                        total_checks=total_checks,
                        passed_checks=passed_checks,
                        failed_checks=failed_checks,
                        warned_checks=warned_checks,
                        skipped_checks=skipped_checks,
                        critical_issues=severity_counts['critical'],
                        high_issues=severity_counts['high'],
                        medium_issues=severity_counts['medium'],
                        low_issues=severity_counts['low'],
                        auto_fixes_attempted=issues_fixed,
                        auto_fixes_succeeded=issues_fixed,
                        status="completed"
                    )
                    self._log(f"Database run completed: {db_run_id}")
                except Exception as e:
                    logger.warning(f"Could not complete database run: {e}")

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

    def run_continuous(self) -> None:
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

    def stop(self) -> None:
        """Stop the continuous runner."""
        self._stop_event.set()

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the QA runner.

        Returns:
            Dict containing runner status information
        """
        return {
            'running': self.running,
            'interval_minutes': self.interval_minutes,
            'visible': self.visible,
            'total_cycles': self.total_cycles,
            'total_issues_found': self.total_issues_found,
            'total_issues_fixed': self.total_issues_fixed,
            'check_modules_loaded': len(self.check_modules),
            'spec_orchestrator_available': self.spec_orchestrator is not None,
            'enhancement_engine_available': hasattr(self, 'enhancement_engine') and self.enhancement_engine is not None,
            'pattern_learner_available': hasattr(self, 'pattern_learner') and self.pattern_learner is not None,
            'current_health_score': self._get_current_health_score(),
        }

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
            # Find Claude CLI path
            import shutil
            claude_path = shutil.which("claude")

            # On Windows, also try with .cmd extension
            if not claude_path and sys.platform == 'win32':
                claude_path = shutil.which("claude.cmd")

            if not claude_path:
                # Try common installation paths
                username = os.environ.get('USERNAME', os.environ.get('USER', 'New User'))
                possible_paths = [
                    f"C:/Users/{username}/AppData/Roaming/npm/claude.cmd",
                    f"C:/Users/{username}/AppData/Roaming/npm/claude",
                    os.path.expanduser("~/AppData/Roaming/npm/claude.cmd"),
                    os.path.expanduser("~/AppData/Roaming/npm/claude"),
                    os.path.expanduser("~/.npm/bin/claude"),
                    "/usr/local/bin/claude",
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        claude_path = p
                        break

            if not claude_path:
                logger.error("Claude CLI not found. Is Claude Code installed?")
                return None

            self._log(f"Using Claude CLI at: {claude_path}")

            result = subprocess.run(
                [claude_path, "--dangerously-skip-permissions", "--print", prompt],
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

    def _invoke_claude_fixer(self, results: List[ModuleCheckResult]) -> int:
        """
        Invoke Claude Code to actually FIX issues found by checks.

        Args:
            results: List of check results with issues

        Returns:
            Number of fixes applied
        """
        # Build a list of specific issues to fix
        issues_to_fix = []
        for result in results:
            for check in result.results:
                if not check.passed and not check.fix_applied:
                    issues_to_fix.append({
                        'module': result.module_name,
                        'check': check.check_name,
                        'message': check.message,
                        'details': check.details,
                    })

        if not issues_to_fix:
            return 0

        # Build fix prompt
        prompt = self._build_fix_prompt(issues_to_fix)

        try:
            self._log("Invoking Claude Code to fix issues...")

            # Use Claude Code SDK or CLI to actually fix
            # Try multiple possible paths for claude CLI
            import shutil
            claude_path = shutil.which("claude")

            # On Windows, also try with .cmd extension
            if not claude_path and sys.platform == 'win32':
                claude_path = shutil.which("claude.cmd")

            if not claude_path:
                # Try common installation paths - get actual username
                username = os.environ.get('USERNAME', os.environ.get('USER', 'New User'))
                possible_paths = [
                    # Windows paths with actual username
                    f"C:/Users/{username}/AppData/Roaming/npm/claude.cmd",
                    f"C:/Users/{username}/AppData/Roaming/npm/claude",
                    # Also try with os.path.expanduser
                    os.path.expanduser("~/AppData/Roaming/npm/claude.cmd"),
                    os.path.expanduser("~/AppData/Roaming/npm/claude"),
                    # Linux/Mac paths
                    os.path.expanduser("~/.npm/bin/claude"),
                    "/usr/local/bin/claude",
                    # Common Windows paths
                    "C:/Users/New User/AppData/Roaming/npm/claude.cmd",
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        claude_path = p
                        self._log(f"Found Claude CLI at: {claude_path}")
                        break

            if not claude_path:
                logger.warning("Claude CLI not found in PATH or common locations.")
                logger.warning(f"Searched paths including: ~/AppData/Roaming/npm/claude.cmd")
                return 0

            self._log(f"Using Claude CLI at: {claude_path}")

            result = subprocess.run(
                [claude_path, "--dangerously-skip-permissions", "-p", prompt],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=900,  # 15 minute timeout for fixes
            )

            if result.returncode == 0:
                # Count fixes from output
                output = result.stdout
                fixes_applied = output.lower().count('fixed') + output.lower().count('created') + output.lower().count('added')

                # Log the fix
                self.tracker.log_auto_fix(
                    module="claude_code_fixer",
                    message=f"Claude Code applied fixes for {len(issues_to_fix)} issues",
                    files=[],
                    details={'issues_addressed': len(issues_to_fix), 'output_preview': output[:500]},
                )

                return max(fixes_applied, 1) if 'error' not in output.lower() else 0
            else:
                logger.error(f"Claude Code fixer returned error: {result.stderr}")
                return 0

        except subprocess.TimeoutExpired:
            logger.error("Claude Code fixer timed out")
            return 0
        except FileNotFoundError:
            logger.warning("Claude CLI not found. Install Claude Code to enable auto-fixing.")
            return 0
        except Exception as e:
            logger.error(f"Failed to invoke Claude Code fixer: {e}")
            return 0

    def _build_fix_prompt(self, issues: List[Dict]) -> str:
        """Build a prompt for Claude Code to fix specific issues."""
        issue_list = "\n".join([
            f"- {i['module']}/{i['check']}: {i['message']}"
            for i in issues[:10]  # Limit to top 10 issues
        ])

        return f"""You are a code fixer for the Magnus trading platform.
FIX these specific issues found during QA:

{issue_list}

## Instructions
1. Read the relevant files
2. Make the necessary code changes to FIX each issue
3. Do NOT just report - actually WRITE the fixes
4. Focus on:
   - Adding missing API endpoints to backend routers
   - Fixing 404/500 errors by implementing the missing routes
   - Creating any missing files or functions

## Specific Actions Needed
For missing API endpoints like /api/sports/games:
- Open the relevant router file (backend/routers/sports.py)
- Add the missing endpoint with appropriate handler
- Use existing code patterns in the file as reference

Be concise. Make the fixes. Do not ask for confirmation.
"""

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

    def _log_module_results(self, result: ModuleCheckResult, db_run_id: int = None):
        """Log individual check results as accomplishments and to database."""
        for check_result in result.results:
            # Determine severity for database logging
            if check_result.is_critical_failure:
                severity = 'critical'
            elif hasattr(result, 'priority') and result.priority == CheckPriority.HIGH:
                severity = 'high'
            elif hasattr(result, 'priority') and result.priority == CheckPriority.MEDIUM:
                severity = 'medium'
            else:
                severity = 'low'

            # Determine status for database
            if check_result.passed:
                db_status = 'passed'
            elif getattr(check_result, 'skipped', False):
                db_status = 'skipped'
            elif check_result.is_critical_failure:
                db_status = 'failed'
            else:
                db_status = 'warned'

            # Log to database
            if self.qa_db and db_run_id:
                try:
                    # Log check result
                    check_result_id = self.qa_db.log_check_result(
                        run_id=db_run_id,
                        module_name=result.module_name,
                        check_name=check_result.check_name,
                        status=db_status,
                        message=check_result.message,
                        details=check_result.details,
                        auto_fixable=getattr(check_result, 'auto_fixable', False),
                        fix_attempted=check_result.fix_applied,
                        fix_succeeded=check_result.fix_applied,
                        fix_message=check_result.fix_description if check_result.fix_applied else None
                    )

                    # Log issue if check failed
                    if not check_result.passed and not getattr(check_result, 'skipped', False):
                        self.qa_db.log_issue(
                            run_id=db_run_id,
                            module_name=result.module_name,
                            check_name=check_result.check_name,
                            title=check_result.message[:500] if check_result.message else "Unknown issue",
                            severity=severity,
                            category=result.module_name.split('_')[0] if '_' in result.module_name else result.module_name,
                            files_affected=check_result.files_affected,
                            primary_file=check_result.files_affected[0] if check_result.files_affected else None,
                            details=check_result.details,
                            check_result_id=check_result_id
                        )
                except Exception as e:
                    logger.warning(f"Could not log to database: {e}")

            # Log to file-based tracker
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
    parser.add_argument('--force', action='store_true',
                        help='Force start (kill existing instance)')
    parser.add_argument('--status', action='store_true',
                        help='Show status of QA runner')
    parser.add_argument('--stop', action='store_true',
                        help='Stop running QA runner')

    args = parser.parse_args()

    # Import process manager
    try:
        from src.utils.process_manager import ProcessManager, stop_service
    except ImportError:
        # Fallback if import fails - allow running without singleton enforcement
        logger.warning("ProcessManager not available, running without singleton enforcement")
        ProcessManager = None

    # Handle status command
    if args.status:
        if ProcessManager:
            pm = ProcessManager('qa_runner')
            status = pm.get_status()
            print(f"\nQA Runner Status:")
            print(f"  Running: {status['is_running']}")
            print(f"  PID: {status['pid'] or 'N/A'}")
            print(f"  PID File: {status['pid_file']}")
        else:
            print("ProcessManager not available")
        sys.exit(0)

    # Handle stop command
    if args.stop:
        if ProcessManager:
            stop_service('qa_runner')
        else:
            print("ProcessManager not available")
        sys.exit(0)

    # Enforce singleton for daemon mode
    process_manager = None
    if args.daemon and ProcessManager:
        pm = ProcessManager('qa_runner')
        is_running, existing_pid = pm.is_already_running()

        if is_running and not args.force:
            print(f"\nERROR: QA Runner is already running (PID {existing_pid})")
            print(f"\nOptions:")
            print(f"  1. Stop existing:  python qa_runner.py --stop")
            print(f"  2. Force restart:  python qa_runner.py --daemon --force")
            print(f"  3. Check status:   python qa_runner.py --status")
            sys.exit(1)

        if not pm.acquire_lock(force=args.force):
            print("Failed to acquire lock")
            sys.exit(1)

        process_manager = pm

    runner = QARunner(
        interval_minutes=args.interval,
        visible=args.visible or not args.daemon,
    )

    try:
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
    finally:
        if process_manager:
            process_manager.release_lock()


if __name__ == "__main__":
    main()
