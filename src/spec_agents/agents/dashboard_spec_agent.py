"""
Dashboard SpecAgent - Tier 1 (Critical)

Deeply understands the Dashboard feature.
Tests: Portfolio overview, summary cards, balance forecast.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('dashboard')
class DashboardSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Dashboard feature.

    Validates:
    - Dashboard summary API endpoints
    - Portfolio value calculations
    - Summary cards display
    - Data freshness
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='dashboard',
            description='Main dashboard with portfolio overview',
            enable_browser=True,
            enable_database=True,
        )
        self._dashboard_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all dashboard-related API endpoints"""
        results = []

        # Test /api/dashboard/summary
        result = await self.test_endpoint(
            method='GET',
            path='/dashboard/summary',
            expected_status=200,
            expected_fields=['total_value'],
            validate_response=self._validate_dashboard_response,
        )
        results.append(result)

        # Store for cross-validation
        if result.passed:
            response = await self.http_get('/dashboard/summary')
            self._dashboard_data = response.json()

        # Test /api/health
        result = await self.test_endpoint(
            method='GET',
            path='/health',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_dashboard_response(self, response) -> List[Issue]:
        """Custom validation for dashboard response"""
        issues = []
        data = response.json()

        # Check for null values
        if data.get('total_value') is None:
            issues.append(Issue(
                title="Missing total_value in dashboard",
                description="Dashboard summary should include total_value",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))

        # Check for negative values (suspicious)
        if data.get('total_value', 0) < 0:
            issues.append(Issue(
                title="Negative portfolio value",
                description=f"total_value is ${data.get('total_value'):.2f}",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))

        # Check day_change makes sense
        day_change = data.get('day_change')
        day_change_pct = data.get('day_change_pct')
        total_value = data.get('total_value', 0)

        if day_change is not None and day_change_pct is not None and total_value > 0:
            # Verify percentage matches dollar change
            expected_pct = (day_change / (total_value - day_change)) * 100 if (total_value - day_change) != 0 else 0
            if abs(day_change_pct - expected_pct) > 1:  # 1% tolerance
                issues.append(Issue(
                    title="Day change percentage mismatch",
                    description=f"Reported {day_change_pct:.2f}%, calculated {expected_pct:.2f}%",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="calculation",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to dashboard (root)
            success = await self.navigate_to('/')
            if not success:
                results.append(TestResult(
                    test_name="Navigate to dashboard",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to dashboard",
                        description="Could not load / route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test portfolio value card exists
            results.append(await self.test_element_exists(
                "[data-testid='portfolio-value'], .portfolio-value, .total-value",
                "Portfolio Value Card"
            ))

            # Test buying power card
            results.append(await self.test_element_exists(
                "[data-testid='buying-power'], .buying-power",
                "Buying Power Card"
            ))

            # Test navigation menu
            results.append(await self.test_element_exists(
                "nav, .sidebar, [role='navigation']",
                "Navigation Menu"
            ))

            # Take screenshot
            await self.take_screenshot("dashboard_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Dashboard UI Tests",
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
        """Test dashboard calculations"""
        results = []
        issues = []

        if self._dashboard_data is None:
            response = await self.http_get('/dashboard/summary')
            if response.status_code == 200:
                self._dashboard_data = response.json()
            else:
                results.append(TestResult(
                    test_name="Fetch dashboard data",
                    passed=False,
                    issues=[Issue(
                        title="Could not fetch dashboard data",
                        description=f"API returned status {response.status_code}",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="business_logic",
                    )],
                ))
                return results

        # Verify allocation percentages sum to 100%
        allocations = self._dashboard_data.get('allocations', {})
        if allocations:
            total_pct = sum([
                allocations.get('stocks', 0),
                allocations.get('options', 0),
                allocations.get('cash', 0),
            ])
            if abs(total_pct - 100) > 1:  # 1% tolerance
                issues.append(Issue(
                    title="Allocations don't sum to 100%",
                    description=f"Total allocation is {total_pct:.1f}%",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="calculation",
                ))

        results.append(TestResult(
            test_name="Dashboard Calculations",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test dashboard data consistency"""
        results = []
        issues = []

        # Fetch positions to compare
        positions_response = await self.http_get('/portfolio/positions')
        if positions_response.status_code != 200:
            issues.append(Issue(
                title="Cannot verify dashboard against positions",
                description="Positions API unavailable for comparison",
                severity=IssueSeverity.MEDIUM,
                feature=self.feature_name,
                component="consistency",
            ))
        else:
            positions_data = positions_response.json()
            positions_total = positions_data.get('summary', {}).get('total_equity', 0)
            dashboard_total = self._dashboard_data.get('total_value', 0) if self._dashboard_data else 0

            if positions_total and dashboard_total:
                diff = abs(positions_total - dashboard_total)
                if diff > 0.01:
                    issues.append(Issue(
                        title="Dashboard doesn't match positions total",
                        description=f"Dashboard: ${dashboard_total:.2f}, Positions: ${positions_total:.2f}",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="consistency",
                    ))

        results.append(TestResult(
            test_name="Dashboard Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
