"""
QA Dashboard SpecAgent - Tier 3

Deeply understands the QA Dashboard feature.
Tests: QA metrics, issue tracking, system health display.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('qa-dashboard')
class QADashboardSpecAgent(BaseSpecAgent):
    """
    SpecAgent for QA Dashboard feature.

    Validates:
    - QA status API endpoints
    - Issue tracking accuracy
    - Health metrics display
    - Historical trend data
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='qa-dashboard',
            description='QA metrics and system health dashboard',
            enable_browser=True,
            enable_database=True,
        )
        self._qa_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all QA dashboard API endpoints"""
        results = []

        # Test /api/qa-dashboard/status (actual endpoint)
        result = await self.test_endpoint(
            method='GET',
            path='/qa-dashboard/status',
            expected_status=200,
            validate_response=self._validate_qa_status_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/qa-dashboard/status')
            if response.status_code == 200:
                self._qa_data = response.json()

        # Test /api/qa-dashboard/issues
        result = await self.test_endpoint(
            method='GET',
            path='/qa-dashboard/issues',
            expected_status=200,
        )
        results.append(result)

        # Test /api/qa-dashboard/history
        result = await self.test_endpoint(
            method='GET',
            path='/qa-dashboard/history',
            expected_status=200,
        )
        results.append(result)

        # Test /api/qa-dashboard/metrics
        result = await self.test_endpoint(
            method='GET',
            path='/qa-dashboard/metrics',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_qa_status_response(self, response) -> List[Issue]:
        """Custom validation for QA status response"""
        issues = []
        data = response.json()

        # Check required fields
        required = ['health_score', 'last_run', 'issues_count']
        for field in required:
            if field not in data:
                issues.append(Issue(
                    title=f"Missing field in QA status: {field}",
                    description=f"QA status missing required field",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="api",
                ))

        # Validate health score
        health_score = data.get('health_score', 0)
        if health_score < 0 or health_score > 100:
            issues.append(Issue(
                title="Invalid health score",
                description=f"Health score {health_score} should be 0-100",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="data_quality",
            ))

        # Validate issues count is non-negative
        issues_count = data.get('issues_count', 0)
        if issues_count < 0:
            issues.append(Issue(
                title="Negative issues count",
                description=f"Issues count {issues_count} should be >= 0",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="data_quality",
            ))

        # Check for severity breakdown
        severity_counts = data.get('severity_counts', data.get('issues_by_severity', {}))
        if severity_counts:
            total_by_severity = sum(severity_counts.values())
            if total_by_severity != issues_count:
                issues.append(Issue(
                    title="Severity count mismatch",
                    description=f"Sum of severities ({total_by_severity}) != total issues ({issues_count})",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="calculation",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            success = await self.navigate_to('/qa')
            if not success:
                success = await self.navigate_to('/qa-dashboard')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to QA dashboard",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to QA dashboard",
                        description="Could not load /qa route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test health score display
            results.append(await self.test_element_exists(
                "[data-testid='health-score'], .health-gauge, .score-display",
                "Health Score Display"
            ))

            # Test issues list
            results.append(await self.test_element_exists(
                "[data-testid='issues-list'], table, .issues-table",
                "Issues List"
            ))

            # Test trend chart
            results.append(await self.test_element_exists(
                "[data-testid='trend-chart'], canvas, .chart",
                "Trend Chart"
            ))

            await self.take_screenshot("qa_dashboard_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="QA Dashboard UI Tests",
                passed=False,
                issues=[Issue(
                    title="UI testing exception",
                    description=str(e),
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="ui",
                )],
            ))

        return results

    async def test_business_logic(self) -> List[TestResult]:
        """Test QA metrics calculations"""
        results = []
        issues = []

        if self._qa_data:
            # Validate last run is recent
            last_run = self._qa_data.get('last_run')
            if last_run:
                from datetime import datetime, timedelta
                try:
                    if isinstance(last_run, str):
                        lr = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                    else:
                        lr = datetime.fromtimestamp(last_run)

                    now = datetime.now(lr.tzinfo) if lr.tzinfo else datetime.now()
                    age = now - lr

                    # QA should run at least every hour
                    if age > timedelta(hours=1):
                        issues.append(Issue(
                            title="QA cycle overdue",
                            description=f"Last QA run was {age.total_seconds() / 3600:.1f} hours ago",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="monitoring",
                        ))
                except Exception as e:
                    logger.debug(f"Could not parse last_run: {e}")

            # Validate health score matches issues
            health_score = self._qa_data.get('health_score', 100)
            critical_count = self._qa_data.get('severity_counts', {}).get('CRITICAL', 0)

            # Health should be low if there are critical issues
            if critical_count > 0 and health_score > 80:
                issues.append(Issue(
                    title="Health score inconsistent with critical issues",
                    description=f"Health {health_score}% but {critical_count} critical issues exist",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="calculation",
                ))

        results.append(TestResult(
            test_name="QA Metrics Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test QA data consistency"""
        results = []
        issues = []

        # Verify issues endpoint matches status count
        try:
            issues_response = await self.http_get('/qa-dashboard/issues')
            if issues_response.status_code == 200 and self._qa_data:
                issues_list = issues_response.json()
                issues_array = issues_list.get('issues', issues_list) if isinstance(issues_list, dict) else issues_list

                if isinstance(issues_array, list):
                    actual_count = len(issues_array)
                    reported_count = self._qa_data.get('issues_count', 0)

                    if actual_count != reported_count:
                        issues.append(Issue(
                            title="Issues count mismatch",
                            description=f"Status reports {reported_count} issues, but /issues returns {actual_count}",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="consistency",
                        ))
        except Exception as e:
            logger.debug(f"Could not verify issues consistency: {e}")

        results.append(TestResult(
            test_name="QA Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
