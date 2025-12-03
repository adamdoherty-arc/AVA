"""
Orchestrator Integration Module

Integrates the Continuous QA system with the Main Orchestrator to:
1. Run QA checks during post-execution phase
2. Feed pattern learning into pre-flight validation
3. Apply enhancements during QA cycles
4. Report health scores to orchestrator
5. Trigger Telegram notifications on critical issues
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Import QA components
from .accomplishments_tracker import AccomplishmentsTracker, Accomplishment
from .health_scorer import HealthScorer
from .enhancement_engine import EnhancementEngine
from .pattern_learner import PatternLearner
from .telegram_notifier import QATelegramNotifier as TelegramNotifier
from .checks.base_check import CheckRunner, CheckPriority

# Import check modules
from .checks.api_endpoints import APIEndpointsCheck
from .checks.code_quality import CodeQualityCheck
from .checks.dummy_data_detector import DummyDataDetectorCheck
from .checks.ui_components import UIComponentsCheck
from .checks.data_sync import DataSyncCheck
from .checks.shared_code_analyzer import SharedCodeAnalyzerCheck


class ContinuousQAIntegration:
    """
    Integrates Continuous QA with Main Orchestrator.

    This class serves as the bridge between:
    - Main Orchestrator (pre-flight, post-execution)
    - Continuous QA system (health checks, enhancements)
    - Pattern Learning (prevention rules)
    - Notifications (Telegram via AVA)
    """

    def __init__(self, project_root: Path = None, config: Dict[str, Any] = None):
        """
        Initialize the integration.

        Args:
            project_root: Root directory of the project
            config: Configuration dictionary
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.config = config or {}

        # Initialize components
        self.accomplishments = AccomplishmentsTracker()
        self.health_scorer = HealthScorer()
        self.enhancement_engine = EnhancementEngine(self.project_root)
        self.pattern_learner = PatternLearner(self.project_root)
        self.notifier = TelegramNotifier()

        # Initialize check modules
        self.check_modules = self._initialize_checks()

        logger.info("Continuous QA Integration initialized")

    def _initialize_checks(self) -> List:
        """Initialize all check modules."""
        checks = []

        try:
            checks.append(APIEndpointsCheck())
        except Exception as e:
            logger.warning(f"Could not initialize APIEndpointsCheck: {e}")

        try:
            checks.append(CodeQualityCheck(self.project_root))
        except Exception as e:
            logger.warning(f"Could not initialize CodeQualityCheck: {e}")

        try:
            checks.append(DummyDataDetectorCheck(self.project_root))
        except Exception as e:
            logger.warning(f"Could not initialize DummyDataDetectorCheck: {e}")

        try:
            checks.append(UIComponentsCheck(self.project_root))
        except Exception as e:
            logger.warning(f"Could not initialize UIComponentsCheck: {e}")

        try:
            checks.append(DataSyncCheck(self.project_root))
        except Exception as e:
            logger.warning(f"Could not initialize DataSyncCheck: {e}")

        try:
            checks.append(SharedCodeAnalyzerCheck(self.project_root))
        except Exception as e:
            logger.warning(f"Could not initialize SharedCodeAnalyzerCheck: {e}")

        return checks

    def run_full_qa_cycle(self) -> Dict[str, Any]:
        """
        Run a complete QA cycle including all checks and enhancements.

        Returns:
            Dictionary with cycle results
        """
        cycle_start = datetime.utcnow()
        results = {
            'timestamp': cycle_start.isoformat(),
            'checks': {},
            'enhancements': {},
            'patterns': {},
            'health_score': 0,
            'accomplishments': [],
            'notifications_sent': [],
        }

        logger.info("Starting full QA cycle...")

        # === PHASE 1: Run all checks ===
        check_runner = CheckRunner(self.check_modules)
        check_results = check_runner.run_all()
        results['checks'] = check_results

        # Track check accomplishments
        for module_name, module_result in check_results.items():
            if hasattr(module_result, 'status'):
                status = module_result.status.value if hasattr(module_result.status, 'value') else str(module_result.status)
                accomplishment = Accomplishment(
                    category='qa_check',
                    action=f"Ran {module_name} check",
                    description=f"Status: {status}",
                    files_affected=[],
                    auto_fixed=False,
                )
                self.accomplishments.log_accomplishment(accomplishment)
                results['accomplishments'].append(accomplishment.action)

        # === PHASE 2: Run enhancements ===
        enhancement_results = self.enhancement_engine.run()
        results['enhancements'] = self.enhancement_engine.get_enhancement_summary(enhancement_results)

        # Track enhancement accomplishments
        if enhancement_results.total_applied > 0:
            accomplishment = Accomplishment(
                category='enhancement',
                action=f"Applied {enhancement_results.total_applied} code enhancements",
                description=f"Categories: {list(results['enhancements'].get('by_category', {}).keys())}",
                files_affected=[e.file_path for e in enhancement_results.enhancements if e.applied],
                auto_fixed=True,
            )
            self.accomplishments.log_accomplishment(accomplishment)
            results['accomplishments'].append(accomplishment.action)

        # === PHASE 3: Run pattern learning ===
        pattern_matches = self.pattern_learner.scan_codebase()
        results['patterns'] = self.pattern_learner.get_statistics()

        # Track pattern learning
        if pattern_matches:
            accomplishment = Accomplishment(
                category='pattern_learning',
                action=f"Detected {len(pattern_matches)} pattern matches",
                description=f"Categories: {list(results['patterns'].get('by_category', {}).keys())}",
                files_affected=[],
                auto_fixed=False,
            )
            self.accomplishments.log_accomplishment(accomplishment)

        # === PHASE 4: Calculate health score ===
        health_score = self.health_scorer.calculate_score(check_results)
        results['health_score'] = health_score
        self.health_scorer.record_score(health_score, check_results)

        # === PHASE 5: Send notifications ===
        # Critical issues
        critical_issues = self._extract_critical_issues(check_results)
        if critical_issues:
            try:
                self.notifier.send_critical_alert(critical_issues)
                results['notifications_sent'].append('critical_alert')
            except Exception as e:
                logger.error(f"Failed to send critical alert: {e}")

        # Health warning if score is low
        if health_score < 70:
            try:
                self.notifier.send_health_warning(health_score, check_results)
                results['notifications_sent'].append('health_warning')
            except Exception as e:
                logger.error(f"Failed to send health warning: {e}")

        # Cycle summary
        cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
        results['duration_seconds'] = cycle_duration

        try:
            self.notifier.send_qa_cycle_summary(results)
            results['notifications_sent'].append('cycle_summary')
        except Exception as e:
            logger.error(f"Failed to send cycle summary: {e}")

        logger.info(f"QA cycle completed in {cycle_duration:.1f}s. Health: {health_score}")

        return results

    def run_post_execution_qa(self, files_modified: List[str]) -> Dict[str, Any]:
        """
        Run QA checks after code modifications (called by Main Orchestrator).

        Args:
            files_modified: List of files that were modified

        Returns:
            QA results dictionary
        """
        results = {
            'passed': True,
            'checks_run': 0,
            'violations': [],
            'warnings': [],
            'auto_fixed': [],
        }

        logger.info(f"Running post-execution QA on {len(files_modified)} files...")

        # Run relevant checks based on file types
        for check in self.check_modules:
            try:
                check_result = check.run()
                results['checks_run'] += 1

                # Process results
                if hasattr(check_result, 'results'):
                    for item_result in check_result.results.values():
                        if hasattr(item_result, 'status'):
                            status = item_result.status
                            if hasattr(status, 'value'):
                                status = status.value

                            if status == 'failed':
                                results['passed'] = False
                                results['violations'].append({
                                    'check': check.name,
                                    'message': getattr(item_result, 'message', 'Unknown'),
                                })
                            elif status == 'warning':
                                results['warnings'].append({
                                    'check': check.name,
                                    'message': getattr(item_result, 'message', 'Unknown'),
                                })
                            elif status == 'fixed':
                                results['auto_fixed'].append({
                                    'check': check.name,
                                    'message': getattr(item_result, 'fix_message', 'Auto-fixed'),
                                })

            except Exception as e:
                logger.error(f"Check {check.name} failed: {e}")

        # Log accomplishment
        accomplishment = Accomplishment(
            category='post_execution_qa',
            action=f"Post-execution QA on {len(files_modified)} files",
            description=f"Passed: {results['passed']}, Violations: {len(results['violations'])}",
            files_affected=files_modified,
            auto_fixed=len(results['auto_fixed']) > 0,
        )
        self.accomplishments.log_accomplishment(accomplishment)

        return results

    def get_prevention_rules_for_preflight(self) -> List[Dict[str, Any]]:
        """
        Get prevention rules from pattern learning for pre-flight validation.

        Returns:
            List of prevention rules
        """
        return self.pattern_learner.generate_prevention_rules()

    def get_hot_spots(self) -> List[Dict[str, Any]]:
        """
        Get files that frequently have issues.

        Returns:
            List of hot spot files with statistics
        """
        return self.pattern_learner.get_hot_spots()

    def get_health_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get health score history.

        Args:
            days: Number of days of history

        Returns:
            List of health score entries
        """
        return self.health_scorer.get_history(days)

    def get_accomplishments_summary(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get recent accomplishments summary.

        Args:
            limit: Maximum number of accomplishments

        Returns:
            Summary dictionary
        """
        return self.accomplishments.get_summary(limit)

    def _extract_critical_issues(self, check_results: Dict) -> List[Dict[str, Any]]:
        """Extract critical issues from check results."""
        critical = []

        for module_name, module_result in check_results.items():
            if hasattr(module_result, 'results'):
                for check_name, result in module_result.results.items():
                    if hasattr(result, 'status'):
                        status = result.status
                        if hasattr(status, 'value'):
                            status = status.value

                        if status == 'failed':
                            # Check if this is a critical check
                            if hasattr(module_result, 'priority'):
                                priority = module_result.priority
                                if hasattr(priority, 'value'):
                                    priority = priority.value
                                if priority == 1:  # CRITICAL
                                    critical.append({
                                        'module': module_name,
                                        'check': check_name,
                                        'message': getattr(result, 'message', 'Unknown'),
                                    })

        return critical


def integrate_with_orchestrator(orchestrator) -> 'ContinuousQAIntegration':
    """
    Integrate Continuous QA with an existing Main Orchestrator instance.

    Args:
        orchestrator: MainOrchestrator instance

    Returns:
        ContinuousQAIntegration instance
    """
    integration = ContinuousQAIntegration(
        config=orchestrator.config if hasattr(orchestrator, 'config') else {}
    )

    # Inject QA into orchestrator's post-execution hook
    original_post_qa = orchestrator.post_execution_qa

    def enhanced_post_qa(files_modified, context=None):
        # Run original QA
        original_results = original_post_qa(files_modified, context)

        # Run continuous QA checks
        continuous_results = integration.run_post_execution_qa(files_modified)

        # Merge results
        original_results['continuous_qa'] = continuous_results
        original_results['passed'] = original_results.get('passed', True) and continuous_results.get('passed', True)

        return original_results

    orchestrator.post_execution_qa = enhanced_post_qa

    # Inject prevention rules into pre-flight
    if hasattr(orchestrator, 'pre_flight'):
        orchestrator.pre_flight.prevention_rules = integration.get_prevention_rules_for_preflight()

    logger.info("Continuous QA integrated with Main Orchestrator")

    return integration


# CLI for standalone testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuous QA Integration")
    parser.add_argument("--full-cycle", action="store_true", help="Run full QA cycle")
    parser.add_argument("--health", action="store_true", help="Show health history")
    parser.add_argument("--accomplishments", action="store_true", help="Show accomplishments")
    parser.add_argument("--hot-spots", action="store_true", help="Show hot spot files")
    parser.add_argument("--prevention-rules", action="store_true", help="Show prevention rules")

    args = parser.parse_args()

    integration = ContinuousQAIntegration()

    if args.full_cycle:
        print("Running full QA cycle...")
        results = integration.run_full_qa_cycle()
        print(f"\nHealth Score: {results['health_score']}")
        print(f"Checks Run: {len(results['checks'])}")
        print(f"Enhancements Applied: {results['enhancements'].get('total_applied', 0)}")
        print(f"Accomplishments: {len(results['accomplishments'])}")

    elif args.health:
        history = integration.get_health_history(7)
        print("Health History (Last 7 Days):")
        for entry in history[-10:]:
            print(f"  {entry.get('timestamp', 'N/A')}: {entry.get('score', 'N/A')}")

    elif args.accomplishments:
        summary = integration.get_accomplishments_summary(20)
        print("Recent Accomplishments:")
        for acc in summary.get('recent', [])[:10]:
            print(f"  - {acc.get('action', 'N/A')}")

    elif args.hot_spots:
        spots = integration.get_hot_spots()
        print("Hot Spot Files:")
        for spot in spots[:10]:
            print(f"  - {spot['file']}: {spot['severity_score']} severity score")

    elif args.prevention_rules:
        rules = integration.get_prevention_rules_for_preflight()
        print("Prevention Rules:")
        for rule in rules[:10]:
            print(f"  - [{rule['severity']}] {rule['description']}")
            print(f"    Prevention: {rule['prevention_rule']}")

    else:
        parser.print_help()
