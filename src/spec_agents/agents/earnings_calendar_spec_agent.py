"""
Earnings Calendar SpecAgent - Tier 2

Deeply understands the Earnings Calendar feature.
Tests: Earnings data, date accuracy, earnings avoidance logic.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('earnings-calendar')
class EarningsCalendarSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Earnings Calendar feature.

    Validates:
    - Earnings data API endpoints
    - Date accuracy
    - Before/after market hours timing
    - Earnings avoidance warnings
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='earnings-calendar',
            description='Earnings calendar and avoidance system',
            enable_browser=True,
            enable_database=True,
        )
        self._earnings_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all earnings-related API endpoints"""
        results = []

        # Test /api/earnings/upcoming
        result = await self.test_endpoint(
            method='GET',
            path='/earnings/upcoming',
            expected_status=200,
            validate_response=self._validate_earnings_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/earnings/upcoming')
            if response.status_code == 200:
                self._earnings_data = response.json()

        # Test /api/earnings/check/{symbol}
        result = await self.test_endpoint(
            method='GET',
            path='/earnings/check/AAPL',
            expected_status=200,
        )
        results.append(result)

        # Test /api/earnings/this-week
        result = await self.test_endpoint(
            method='GET',
            path='/earnings/this-week',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_earnings_response(self, response) -> List[Issue]:
        """Custom validation for earnings response"""
        issues = []
        data = response.json()

        earnings = data.get('earnings', data) if isinstance(data, dict) else data
        if not isinstance(earnings, list):
            issues.append(Issue(
                title="Earnings response not an array",
                description=f"Expected array of earnings",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        for i, earning in enumerate(earnings[:10]):
            # Check required fields
            required = ['symbol', 'earnings_date']
            for field in required:
                if field not in earning:
                    issues.append(Issue(
                        title=f"Missing field in earning {i}: {field}",
                        description=f"Earnings data missing required field",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Validate date format and range
            earnings_date_str = earning.get('earnings_date', '')
            if earnings_date_str:
                try:
                    # Try parsing the date
                    if 'T' in earnings_date_str:
                        earnings_date = datetime.fromisoformat(earnings_date_str.replace('Z', '+00:00'))
                    else:
                        earnings_date = datetime.strptime(earnings_date_str, '%Y-%m-%d')

                    # Check if date is in past (shouldn't show past earnings in upcoming)
                    now = datetime.now()
                    if earnings_date.date() < now.date():
                        issues.append(Issue(
                            title=f"Past earnings in upcoming: {earning.get('symbol')}",
                            description=f"Earnings date {earnings_date_str} is in the past",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="data_quality",
                        ))
                except ValueError as e:
                    issues.append(Issue(
                        title=f"Invalid date format: {earning.get('symbol')}",
                        description=f"Could not parse date: {earnings_date_str}",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

            # Validate timing (BMO/AMC)
            timing = earning.get('timing', earning.get('time', ''))
            valid_timings = ['bmo', 'amc', 'before market open', 'after market close', 'before open', 'after close', 'unknown', '']
            if timing and timing.lower() not in valid_timings:
                issues.append(Issue(
                    title=f"Invalid timing value: {earning.get('symbol')}",
                    description=f"Timing '{timing}' not recognized",
                    severity=IssueSeverity.LOW,
                    feature=self.feature_name,
                    component="data_quality",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to earnings page
            success = await self.navigate_to('/earnings')
            if not success:
                success = await self.navigate_to('/earnings-calendar')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to earnings page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to earnings page",
                        description="Could not load /earnings route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test calendar exists
            results.append(await self.test_element_exists(
                "[data-testid='earnings-calendar'], .calendar, table",
                "Earnings Calendar"
            ))

            # Test week navigation
            results.append(await self.test_element_exists(
                "[data-testid='week-nav'], .week-navigation, button:has-text('Next')",
                "Week Navigation"
            ))

            # Take screenshot
            await self.take_screenshot("earnings_calendar_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Earnings Calendar UI Tests",
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
        """Test earnings avoidance logic"""
        results = []
        issues = []

        # Test symbol check endpoint
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']

        for symbol in test_symbols:
            try:
                response = await self.http_get(f'/earnings/check/{symbol}')
                if response.status_code == 200:
                    data = response.json()

                    has_upcoming = data.get('has_upcoming_earnings', False)
                    days_until = data.get('days_until_earnings', None)

                    # If has upcoming, should have days_until
                    if has_upcoming and days_until is None:
                        issues.append(Issue(
                            title=f"Missing days_until for {symbol}",
                            description="has_upcoming_earnings=True but no days_until",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="business_logic",
                        ))

                    # Days until should be non-negative
                    if days_until is not None and days_until < 0:
                        issues.append(Issue(
                            title=f"Negative days_until for {symbol}",
                            description=f"days_until_earnings = {days_until}",
                            severity=IssueSeverity.HIGH,
                            feature=self.feature_name,
                            component="calculation",
                        ))
            except Exception as e:
                logger.debug(f"Error checking {symbol}: {e}")

        results.append(TestResult(
            test_name="Earnings Check Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test earnings data consistency"""
        results = []
        issues = []

        # Check data is sorted by date
        if self._earnings_data:
            earnings = self._earnings_data.get('earnings', self._earnings_data) if isinstance(self._earnings_data, dict) else self._earnings_data

            if isinstance(earnings, list) and len(earnings) > 1:
                dates = []
                for e in earnings:
                    date_str = e.get('earnings_date', '')
                    if date_str:
                        try:
                            if 'T' in date_str:
                                dates.append(datetime.fromisoformat(date_str.replace('Z', '+00:00')))
                            else:
                                dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                        except ValueError:
                            pass

                if dates and dates != sorted(dates):
                    issues.append(Issue(
                        title="Earnings not sorted by date",
                        description="Upcoming earnings should be sorted chronologically",
                        severity=IssueSeverity.LOW,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        # Check for duplicate symbols in same date
        if self._earnings_data:
            earnings = self._earnings_data.get('earnings', self._earnings_data) if isinstance(self._earnings_data, dict) else self._earnings_data

            if isinstance(earnings, list):
                seen = set()
                for e in earnings:
                    key = f"{e.get('symbol', '')}_{e.get('earnings_date', '')}"
                    if key in seen:
                        issues.append(Issue(
                            title=f"Duplicate earnings entry",
                            description=f"{e.get('symbol')} appears multiple times for same date",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="data_quality",
                        ))
                    seen.add(key)

        results.append(TestResult(
            test_name="Earnings Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
