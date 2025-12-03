"""
Calendar Spreads SpecAgent - Tier 3

Deeply understands the Calendar Spreads feature.
Tests: Spread calculations, IV analysis, strategy recommendations.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('calendar-spreads')
class CalendarSpreadsSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Calendar Spreads feature.

    Validates:
    - Calendar spread API endpoints
    - IV skew calculations
    - Spread P&L projections
    - Strategy recommendations
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='calendar-spreads',
            description='Calendar spread analysis and recommendations',
            enable_browser=True,
            enable_database=True,
        )
        self._spreads_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all calendar spread API endpoints"""
        results = []

        # Test /api/spreads/calendar
        result = await self.test_endpoint(
            method='GET',
            path='/spreads/calendar',
            expected_status=200,
            validate_response=self._validate_spreads_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/spreads/calendar')
            if response.status_code == 200:
                self._spreads_data = response.json()

        # Test symbol-specific spread analysis
        result = await self.test_endpoint(
            method='POST',
            path='/spreads/analyze',
            expected_status=200,
            body={'symbol': 'AAPL', 'strike': 150, 'near_exp': '2024-01-19', 'far_exp': '2024-02-16'},
        )
        results.append(result)

        return results

    def _validate_spreads_response(self, response) -> List[Issue]:
        """Custom validation for spreads response"""
        issues = []
        data = response.json()

        spreads = data.get('spreads', data) if isinstance(data, dict) else data
        if not isinstance(spreads, list):
            issues.append(Issue(
                title="Spreads response not an array",
                description=f"Expected array of spreads",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        for i, spread in enumerate(spreads[:5]):
            # Check required fields
            required = ['symbol', 'strike', 'near_expiration', 'far_expiration', 'net_debit']
            for field in required:
                if field not in spread:
                    issues.append(Issue(
                        title=f"Missing field in spread {i}: {field}",
                        description=f"Calendar spread missing required field",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Validate net debit is positive (calendar spreads are debit spreads)
            net_debit = spread.get('net_debit', 0)
            if net_debit < 0:
                issues.append(Issue(
                    title=f"Negative net debit: {spread.get('symbol', i)}",
                    description=f"Calendar spreads should have positive net debit, got {net_debit}",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="calculation",
                ))

            # Validate far expiration is after near expiration
            near_exp = spread.get('near_expiration', '')
            far_exp = spread.get('far_expiration', '')
            if near_exp and far_exp and near_exp >= far_exp:
                issues.append(Issue(
                    title=f"Invalid expiration order: {spread.get('symbol', i)}",
                    description=f"Near exp ({near_exp}) should be before far exp ({far_exp})",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="data_quality",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            success = await self.navigate_to('/calendar-spreads')
            if not success:
                success = await self.navigate_to('/spreads')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to calendar spreads page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to calendar spreads page",
                        description="Could not load /calendar-spreads route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test spreads table
            results.append(await self.test_element_exists(
                "[data-testid='spreads-table'], table, .spreads-list",
                "Spreads Table"
            ))

            # Test P&L chart
            results.append(await self.test_element_exists(
                "[data-testid='pl-chart'], canvas, .chart",
                "P&L Chart"
            ))

            await self.take_screenshot("calendar_spreads_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Calendar Spreads UI Tests",
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
        """Test calendar spread calculations"""
        results = []
        issues = []

        if self._spreads_data:
            spreads = self._spreads_data.get('spreads', self._spreads_data) if isinstance(self._spreads_data, dict) else self._spreads_data

            for spread in spreads[:5]:
                # Validate IV relationship
                near_iv = spread.get('near_iv', 0)
                far_iv = spread.get('far_iv', 0)
                iv_differential = spread.get('iv_differential', None)

                if near_iv > 0 and far_iv > 0:
                    expected_diff = far_iv - near_iv
                    if iv_differential is not None and abs(iv_differential - expected_diff) > 0.01:
                        issues.append(Issue(
                            title=f"IV differential mismatch: {spread.get('symbol', 'unknown')}",
                            description=f"Reported {iv_differential:.2%}, calculated {expected_diff:.2%}",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="calculation",
                        ))

                # Validate max profit is reasonable
                net_debit = spread.get('net_debit', 0)
                max_profit = spread.get('max_profit', 0)

                if net_debit > 0 and max_profit > 0:
                    # Max profit should be reasonable (typically < 100% for calendar spreads)
                    profit_pct = (max_profit / net_debit) * 100
                    if profit_pct > 200:
                        issues.append(Issue(
                            title=f"Unrealistic max profit: {spread.get('symbol', 'unknown')}",
                            description=f"Max profit {profit_pct:.0f}% of debit seems too high",
                            severity=IssueSeverity.LOW,
                            feature=self.feature_name,
                            component="calculation",
                        ))

        results.append(TestResult(
            test_name="Calendar Spread Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test calendar spread data consistency"""
        results = []
        issues = []

        if self._spreads_data:
            spreads = self._spreads_data.get('spreads', self._spreads_data) if isinstance(self._spreads_data, dict) else self._spreads_data

            if isinstance(spreads, list):
                # Check for unique spread identifiers
                seen = set()
                for spread in spreads:
                    key = f"{spread.get('symbol', '')}_{spread.get('strike', '')}_{spread.get('near_expiration', '')}_{spread.get('far_expiration', '')}"
                    if key in seen:
                        issues.append(Issue(
                            title="Duplicate spread entry",
                            description=f"Same spread configuration appears multiple times",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="data_quality",
                        ))
                    seen.add(key)

        results.append(TestResult(
            test_name="Calendar Spread Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
