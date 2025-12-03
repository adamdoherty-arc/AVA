"""
SpecAgent Orchestrator - Coordinates testing across all SpecAgents

Features:
- Parallel test execution
- Result aggregation
- Baseline/regression detection
- Issue prioritization
- Integration with QA Runner
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_spec_agent import BaseSpecAgent, Issue, IssueSeverity
from .spec_agent_registry import SpecAgentRegistry
from .dependency_graph import get_dependency_graph, FeatureDependencyGraph

logger = logging.getLogger(__name__)


# ==================== Baseline & Regression Detection ====================

@dataclass
class FeatureBaseline:
    """Baseline metrics for a feature"""
    feature_name: str
    test_count: int = 0
    pass_rate: float = 0.0
    issue_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    avg_duration_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    sample_count: int = 0  # Number of runs used to calculate baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'test_count': self.test_count,
            'pass_rate': self.pass_rate,
            'issue_count': self.issue_count,
            'critical_count': self.critical_count,
            'high_count': self.high_count,
            'avg_duration_ms': self.avg_duration_ms,
            'last_updated': self.last_updated.isoformat(),
            'sample_count': self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureBaseline':
        last_updated = data.get('last_updated', datetime.now().isoformat())
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        return cls(
            feature_name=data['feature_name'],
            test_count=data.get('test_count', 0),
            pass_rate=data.get('pass_rate', 0.0),
            issue_count=data.get('issue_count', 0),
            critical_count=data.get('critical_count', 0),
            high_count=data.get('high_count', 0),
            avg_duration_ms=data.get('avg_duration_ms', 0.0),
            last_updated=last_updated,
            sample_count=data.get('sample_count', 0),
        )


@dataclass
class RegressionResult:
    """Result of regression comparison"""
    feature_name: str
    is_regression: bool
    regression_type: str  # 'pass_rate', 'critical_issues', 'high_issues', 'new_failures'
    baseline_value: float
    current_value: float
    delta: float
    severity: str  # 'critical', 'high', 'medium', 'low'
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'is_regression': self.is_regression,
            'regression_type': self.regression_type,
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'delta': self.delta,
            'severity': self.severity,
            'message': self.message,
        }


class BaselineManager:
    """Manages baselines and detects regressions"""

    def __init__(self, baselines_dir: Optional[Path] = None):
        self.baselines_dir = baselines_dir or Path(".claude/orchestrator/continuous_qa/data/baselines")
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_file = self.baselines_dir / "baselines.json"
        self.history_file = self.baselines_dir / "history.json"
        self._baselines: Dict[str, FeatureBaseline] = {}
        self._history: List[Dict[str, Any]] = []
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load baselines from file"""
        try:
            if self.baselines_file.exists():
                with open(self.baselines_file, 'r') as f:
                    data = json.load(f)
                    for feature_name, baseline_data in data.items():
                        self._baselines[feature_name] = FeatureBaseline.from_dict(baseline_data)
                logger.debug(f"Loaded {len(self._baselines)} baselines")

            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    self._history = json.load(f)
                logger.debug(f"Loaded {len(self._history)} history entries")

        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")

    def _save_baselines(self) -> None:
        """Save baselines to file"""
        try:
            data = {name: baseline.to_dict() for name, baseline in self._baselines.items()}
            with open(self.baselines_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Also save history (keep last 100 entries)
            with open(self.history_file, 'w') as f:
                json.dump(self._history[-100:], f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def update_baseline(self, feature_name: str, result: Dict[str, Any]):
        """
        Update baseline with new result using exponential moving average.

        Args:
            feature_name: Feature to update
            result: Test result dictionary
        """
        total_tests = result.get('total_tests', 0)
        passed = result.get('passed', 0)
        pass_rate = (passed / max(total_tests, 1)) * 100
        duration_ms = result.get('duration_ms', 0)
        issue_count = result.get('total_issues', 0)
        critical_count = result.get('critical_issues', 0)
        high_count = result.get('high_issues', 0)

        if feature_name not in self._baselines:
            # Create new baseline
            self._baselines[feature_name] = FeatureBaseline(
                feature_name=feature_name,
                test_count=total_tests,
                pass_rate=pass_rate,
                issue_count=issue_count,
                critical_count=critical_count,
                high_count=high_count,
                avg_duration_ms=duration_ms,
                sample_count=1,
            )
        else:
            # Update with exponential moving average (alpha = 0.2)
            baseline = self._baselines[feature_name]
            alpha = 0.2

            baseline.test_count = int(baseline.test_count * (1 - alpha) + total_tests * alpha)
            baseline.pass_rate = baseline.pass_rate * (1 - alpha) + pass_rate * alpha
            baseline.issue_count = baseline.issue_count * (1 - alpha) + issue_count * alpha
            baseline.critical_count = baseline.critical_count * (1 - alpha) + critical_count * alpha
            baseline.high_count = baseline.high_count * (1 - alpha) + high_count * alpha
            baseline.avg_duration_ms = baseline.avg_duration_ms * (1 - alpha) + duration_ms * alpha
            baseline.sample_count += 1
            baseline.last_updated = datetime.now()

        # Add to history
        self._history.append({
            'timestamp': datetime.now().isoformat(),
            'feature_name': feature_name,
            'pass_rate': pass_rate,
            'issue_count': issue_count,
            'critical_count': critical_count,
            'high_count': high_count,
            'duration_ms': duration_ms,
        })

        self._save_baselines()

    def detect_regressions(
        self,
        feature_name: str,
        result: Dict[str, Any],
        pass_rate_threshold: float = 10.0,  # 10% decrease triggers regression
        issue_increase_threshold: int = 2,  # +2 issues triggers regression
    ) -> List[RegressionResult]:
        """
        Detect regressions by comparing current result against baseline.

        Args:
            feature_name: Feature to check
            result: Current test result
            pass_rate_threshold: Pass rate decrease % that triggers regression
            issue_increase_threshold: Issue count increase that triggers regression

        Returns:
            List of detected regressions
        """
        regressions = []

        if feature_name not in self._baselines:
            # No baseline yet - skip regression check
            return regressions

        baseline = self._baselines[feature_name]

        # Skip if not enough samples for reliable baseline
        if baseline.sample_count < 3:
            return regressions

        total_tests = result.get('total_tests', 0)
        passed = result.get('passed', 0)
        current_pass_rate = (passed / max(total_tests, 1)) * 100
        current_issues = result.get('total_issues', 0)
        current_critical = result.get('critical_issues', 0)
        current_high = result.get('high_issues', 0)

        # Check pass rate regression
        pass_rate_delta = baseline.pass_rate - current_pass_rate
        if pass_rate_delta > pass_rate_threshold:
            severity = 'critical' if pass_rate_delta > 30 else 'high' if pass_rate_delta > 20 else 'medium'
            regressions.append(RegressionResult(
                feature_name=feature_name,
                is_regression=True,
                regression_type='pass_rate',
                baseline_value=baseline.pass_rate,
                current_value=current_pass_rate,
                delta=pass_rate_delta,
                severity=severity,
                message=f"Pass rate dropped from {baseline.pass_rate:.1f}% to {current_pass_rate:.1f}% (-{pass_rate_delta:.1f}%)",
            ))

        # Check critical issues increase
        critical_delta = current_critical - baseline.critical_count
        if critical_delta >= 1:  # Any new critical issue is significant
            regressions.append(RegressionResult(
                feature_name=feature_name,
                is_regression=True,
                regression_type='critical_issues',
                baseline_value=baseline.critical_count,
                current_value=current_critical,
                delta=critical_delta,
                severity='critical',
                message=f"Critical issues increased from {baseline.critical_count:.1f} to {current_critical} (+{critical_delta:.1f})",
            ))

        # Check high issues increase
        high_delta = current_high - baseline.high_count
        if high_delta >= issue_increase_threshold:
            regressions.append(RegressionResult(
                feature_name=feature_name,
                is_regression=True,
                regression_type='high_issues',
                baseline_value=baseline.high_count,
                current_value=current_high,
                delta=high_delta,
                severity='high',
                message=f"High-severity issues increased from {baseline.high_count:.1f} to {current_high} (+{high_delta:.1f})",
            ))

        # Check total issues increase
        issue_delta = current_issues - baseline.issue_count
        if issue_delta >= issue_increase_threshold * 2:
            severity = 'high' if issue_delta >= 5 else 'medium'
            regressions.append(RegressionResult(
                feature_name=feature_name,
                is_regression=True,
                regression_type='total_issues',
                baseline_value=baseline.issue_count,
                current_value=current_issues,
                delta=issue_delta,
                severity=severity,
                message=f"Total issues increased from {baseline.issue_count:.1f} to {current_issues} (+{issue_delta:.1f})",
            ))

        return regressions

    def get_baseline(self, feature_name: str) -> Optional[FeatureBaseline]:
        """Get baseline for a feature"""
        return self._baselines.get(feature_name)

    def get_all_baselines(self) -> Dict[str, FeatureBaseline]:
        """Get all baselines"""
        return self._baselines.copy()

    def get_trend(self, feature_name: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get historical trend for a feature"""
        cutoff = datetime.now() - timedelta(days=days)
        trend = []
        for entry in self._history:
            if entry.get('feature_name') == feature_name:
                ts = datetime.fromisoformat(entry['timestamp'])
                if ts >= cutoff:
                    trend.append(entry)
        return trend


class SpecAgentOrchestrator:
    """
    Coordinates all SpecAgents for comprehensive feature testing.

    Provides:
    - Parallel test execution
    - Result aggregation
    - Baseline tracking
    - Regression detection
    - Issue prioritization
    - Integration with QA Runner
    - AVA chatbot integration
    """

    def __init__(
        self,
        max_parallel: int = 5,
        output_dir: Optional[Path] = None,
        enable_regression_detection: bool = True,
        use_dependency_order: bool = True,
    ):
        """
        Initialize orchestrator

        Args:
            max_parallel: Maximum number of agents to run in parallel
            output_dir: Directory for results output
            enable_regression_detection: Whether to track baselines and detect regressions
            use_dependency_order: Whether to use dependency graph for test execution order
        """
        self.max_parallel = max_parallel
        self.output_dir = output_dir or Path(".claude/orchestrator/continuous_qa/data/spec_agents")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._results: Dict[str, Any] = {}
        self._all_issues: List[Issue] = []
        self._regressions: List[RegressionResult] = []
        self._cascade_impacts: Dict[str, Any] = {}

        # Initialize baseline manager for regression detection
        self.enable_regression_detection = enable_regression_detection
        self.baseline_manager = BaselineManager() if enable_regression_detection else None

        # Initialize dependency graph
        self.use_dependency_order = use_dependency_order
        self.dependency_graph = get_dependency_graph() if use_dependency_order else None

    async def run_agent(self, agent: BaseSpecAgent) -> Dict[str, Any]:
        """
        Run a single SpecAgent's tests

        Args:
            agent: SpecAgent to run

        Returns:
            Test results dictionary
        """
        try:
            logger.info(f"Starting SpecAgent: {agent.feature_name}")
            result = await agent.run_all_tests()
            return result
        except Exception as e:
            logger.error(f"SpecAgent {agent.feature_name} failed: {e}")
            return {
                'feature': agent.feature_name,
                'error': str(e),
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'total_issues': 1,
                'critical_issues': 1,
                'issues': [{
                    'title': f'Agent execution failed',
                    'description': str(e),
                    'severity': 'critical',
                    'feature': agent.feature_name,
                }],
                'timestamp': datetime.now().isoformat(),
            }
        finally:
            await agent.cleanup()

    async def run_all_agents(
        self,
        feature_names: Optional[List[str]] = None,
        tier: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run all registered SpecAgents

        Args:
            feature_names: Specific features to test (None = all)
            tier: Only run agents in specific tier (1, 2, or 3)

        Returns:
            Aggregated results dictionary
        """
        import time
        start = time.time()

        # Get agents to run
        if feature_names:
            agents = [SpecAgentRegistry.get(name) for name in feature_names]
            agents = [a for a in agents if a is not None]
        else:
            agents = SpecAgentRegistry.get_all()

        # Sort agents by dependency order if enabled
        if self.use_dependency_order and self.dependency_graph:
            execution_order = self.dependency_graph.get_execution_order()
            agent_map = {a.feature_name: a for a in agents}
            ordered_agents = []
            for feature in execution_order:
                if feature in agent_map:
                    ordered_agents.append(agent_map[feature])
            # Add any agents not in the dependency graph
            for agent in agents:
                if agent not in ordered_agents:
                    ordered_agents.append(agent)
            agents = ordered_agents
            logger.info(f"Running agents in dependency order: {[a.feature_name for a in agents]}")

        if not agents:
            logger.warning("No SpecAgents registered to run")
            return {
                'run_id': f"SPEC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                'total_agents': 0,
                'results': {},
                'summary': {},
            }

        logger.info(f"Running {len(agents)} SpecAgents...")

        # Run agents with concurrency limit
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def run_with_semaphore(agent: BaseSpecAgent) -> Dict[str, Any]:
            async with semaphore:
                return await self.run_agent(agent)

        tasks = [run_with_semaphore(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        run_id = f"SPEC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        agent_results = {}
        all_issues = []
        all_regressions = []

        total_tests = 0
        total_passed = 0
        total_failed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                feature_name = agents[i].feature_name
                agent_results[feature_name] = {
                    'error': str(result),
                    'critical_issues': 1,
                }
                all_issues.append({
                    'title': f'Agent exception: {feature_name}',
                    'description': str(result),
                    'severity': 'critical',
                    'feature': feature_name,
                })
            else:
                feature_name = result.get('feature', f'agent_{i}')
                agent_results[feature_name] = result
                all_issues.extend(result.get('issues', []))

                total_tests += result.get('total_tests', 0)
                total_passed += result.get('passed', 0)
                total_failed += result.get('failed', 0)

                # Regression detection
                if self.enable_regression_detection and self.baseline_manager:
                    # Detect regressions before updating baseline
                    regressions = self.baseline_manager.detect_regressions(feature_name, result)
                    if regressions:
                        all_regressions.extend(regressions)
                        agent_results[feature_name]['regressions'] = [r.to_dict() for r in regressions]
                        logger.warning(f"[{feature_name}] {len(regressions)} regression(s) detected!")

                    # Update baseline with current result
                    self.baseline_manager.update_baseline(feature_name, result)

        # Store results
        self._results = agent_results
        self._regressions = all_regressions

        total_time = (time.time() - start) * 1000

        # Build summary
        summary = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'total_agents': len(agents),
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_issues': len(all_issues),
            'critical_issues': sum(1 for i in all_issues if i.get('severity') == 'critical'),
            'high_issues': sum(1 for i in all_issues if i.get('severity') == 'high'),
            'medium_issues': sum(1 for i in all_issues if i.get('severity') == 'medium'),
            'duration_ms': total_time,
            'results': agent_results,
            'issues': all_issues,
            # Regression data
            'regressions': [r.to_dict() for r in all_regressions],
            'total_regressions': len(all_regressions),
            'critical_regressions': sum(1 for r in all_regressions if r.severity == 'critical'),
            'high_regressions': sum(1 for r in all_regressions if r.severity == 'high'),
        }

        # Add cascade impact analysis for failed features
        if self.use_dependency_order and self.dependency_graph:
            failed_features = [
                name for name, result in agent_results.items()
                if result.get('critical_issues', 0) > 0 or result.get('error')
            ]
            cascade_impacts = {}
            for feature in failed_features:
                impact = self.dependency_graph.analyze_cascade_impact(feature)
                if impact.get('total_affected', 0) > 0:
                    cascade_impacts[feature] = impact
                    logger.warning(
                        f"[CASCADE] {feature} failure affects {impact.get('total_affected')} features"
                    )
            summary['cascade_impacts'] = cascade_impacts
            summary['features_at_risk'] = list(set(
                f for impact in cascade_impacts.values()
                for f in impact.get('affected_features', [])
            ))
            self._cascade_impacts = cascade_impacts

        # Save results
        await self._save_results(summary)

        logger.info(
            f"SpecAgent run complete: "
            f"{total_passed}/{total_tests} tests passed, "
            f"{len(all_issues)} issues found, "
            f"{len(all_regressions)} regressions detected"
        )

        return summary

    async def run_priority_agents(self) -> Dict[str, Any]:
        """
        Run only Tier 1 (critical) agents

        Tier 1 features:
        - positions
        - dashboard
        - premium-scanner
        - options-analysis
        - game-cards
        """
        tier1_features = [
            'positions',
            'dashboard',
            'premium-scanner',
            'options-analysis',
            'game-cards',
        ]
        return await self.run_all_agents(feature_names=tier1_features)

    async def _save_results(self, summary: Dict[str, Any]):
        """Save results to output directory"""
        try:
            # Save full results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"results_{timestamp}.json"

            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            # Update latest results file
            latest_file = self.output_dir / "latest_results.json"
            with open(latest_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Results saved to {results_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def get_critical_issues(self) -> List[Dict[str, Any]]:
        """Get all critical issues from last run"""
        issues = []
        for feature_name, result in self._results.items():
            for issue in result.get('issues', []):
                if issue.get('severity') in ('critical', 'high'):
                    issues.append(issue)
        return issues

    def get_issues_by_feature(self, feature_name: str) -> List[Dict[str, Any]]:
        """Get issues for a specific feature"""
        result = self._results.get(feature_name, {})
        return result.get('issues', [])

    def get_health_score(self) -> float:
        """
        Calculate overall health score (0-100)

        Based on:
        - Test pass rate (40%)
        - Critical issues (25%)
        - High issues (15%)
        - Medium issues (10%)
        - Regressions (10%)
        """
        if not self._results:
            return 100.0

        total_tests = sum(r.get('total_tests', 0) for r in self._results.values())
        total_passed = sum(r.get('passed', 0) for r in self._results.values())

        critical = sum(r.get('critical_issues', 0) for r in self._results.values())
        high = sum(r.get('high_issues', 0) for r in self._results.values())
        medium = sum(r.get('medium_issues', 0) for r in self._results.values())

        # Pass rate component (40%)
        pass_rate = (total_passed / max(total_tests, 1)) * 100
        pass_score = pass_rate * 0.4

        # Critical issues component (25%) - each critical = -10 points
        critical_score = max(0, 25 - (critical * 10))

        # High issues component (15%) - each high = -5 points
        high_score = max(0, 15 - (high * 5))

        # Medium issues component (10%) - each medium = -2 points
        medium_score = max(0, 10 - (medium * 2))

        # Regressions component (10%) - each regression = -5 points
        regression_count = len(self._regressions)
        regression_score = max(0, 10 - (regression_count * 5))

        return min(100.0, pass_score + critical_score + high_score + medium_score + regression_score)

    def get_regressions(self) -> List[RegressionResult]:
        """Get all regressions from last run"""
        return self._regressions.copy()

    def get_regressions_by_feature(self, feature_name: str) -> List[RegressionResult]:
        """Get regressions for a specific feature"""
        return [r for r in self._regressions if r.feature_name == feature_name]

    def get_feature_trend(self, feature_name: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get historical trend for a feature"""
        if self.baseline_manager:
            return self.baseline_manager.get_trend(feature_name, days)
        return []

    def get_feature_baseline(self, feature_name: str) -> Optional[FeatureBaseline]:
        """Get baseline for a feature"""
        if self.baseline_manager:
            return self.baseline_manager.get_baseline(feature_name)
        return None

    def get_cascade_impacts(self) -> Dict[str, Any]:
        """Get cascade impacts from last run"""
        return self._cascade_impacts.copy()

    def get_dependency_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get dependency information for a feature"""
        if self.dependency_graph:
            return self.dependency_graph.get_feature_info(feature_name)
        return None

    def get_execution_order(self) -> List[str]:
        """Get the test execution order based on dependencies"""
        if self.dependency_graph:
            return self.dependency_graph.get_execution_order()
        return []

    def visualize_dependencies(self) -> str:
        """Get ASCII visualization of the dependency graph"""
        if self.dependency_graph:
            return self.dependency_graph.visualize_ascii()
        return "Dependency graph not enabled"

    async def report_to_chatbot(self, summary: Dict[str, Any]):
        """
        Report critical issues and regressions to AVA chatbot

        Integrates with existing AVA alert_agent
        """
        try:
            # Only report if there are critical/high issues or regressions
            critical_count = summary.get('critical_issues', 0)
            high_count = summary.get('high_issues', 0)
            regression_count = summary.get('total_regressions', 0)
            critical_regressions = summary.get('critical_regressions', 0)

            if critical_count == 0 and high_count == 0 and regression_count == 0:
                return

            # Build alert message
            message = f"**SpecAgent Alert**\n\n"
            message += f"Run ID: {summary.get('run_id')}\n"
            message += f"Health Score: {self.get_health_score():.1f}%\n\n"

            message += "**Issues:**\n"
            message += f"- Critical: {critical_count}\n"
            message += f"- High: {high_count}\n\n"

            # Add regression info
            if regression_count > 0:
                message += "**Regressions Detected:**\n"
                message += f"- Total: {regression_count}\n"
                message += f"- Critical: {critical_regressions}\n\n"

                # Add top regressions
                regressions = summary.get('regressions', [])[:3]
                for reg in regressions:
                    message += f"- [{reg.get('feature_name')}] {reg.get('message')}\n"
                message += "\n"

            # Add top issues
            issues = summary.get('issues', [])
            critical_issues = [i for i in issues if i.get('severity') == 'critical'][:3]

            if critical_issues:
                message += "**Critical Issues:**\n"
                for issue in critical_issues:
                    message += f"- [{issue.get('feature')}] {issue.get('title')}\n"

            # Determine severity
            if critical_count > 0 or critical_regressions > 0:
                severity = "critical"
            elif high_count > 0 or regression_count > 0:
                severity = "high"
            else:
                severity = "medium"

            # Try to send via AVA
            try:
                from src.ava.agents.monitoring.alert_agent import AlertAgent
                alert_agent = AlertAgent()
                await alert_agent.send_alert(
                    title="SpecAgent Issues Detected",
                    message=message,
                    severity=severity,
                )
            except ImportError:
                logger.warning("AlertAgent not available for chatbot integration")

        except Exception as e:
            logger.error(f"Failed to report to chatbot: {e}")


# Singleton orchestrator instance
_orchestrator: Optional[SpecAgentOrchestrator] = None


def get_orchestrator() -> SpecAgentOrchestrator:
    """Get the singleton SpecAgentOrchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SpecAgentOrchestrator()
    return _orchestrator
