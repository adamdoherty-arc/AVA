"""
7-Day DTE Scanner SpecAgent - Tier 2

Deeply understands the 7-Day DTE Scanner feature.
Tests: Short-term options scanning, DTE calculations, premium analysis.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent
from ..validators.dte_calculator import DTECalculator

logger = logging.getLogger(__name__)


@register_spec_agent('dte-scanner')
class DTEScannerSpecAgent(BaseSpecAgent):
    """
    SpecAgent for 7-Day DTE Scanner feature.

    Validates:
    - DTE Scanner API endpoints
    - DTE calculation accuracy
    - Premium calculations for short-term options
    - Theta decay analysis
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='dte-scanner',
            description='7-Day DTE options scanner for theta decay',
            enable_browser=True,
            enable_database=True,
        )
        self._scanner_results = None
        self.dte_calculator = DTECalculator()

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all DTE scanner-related API endpoints"""
        results = []

        # Test /api/scanner/dte with 7-day filter
        result = await self.test_endpoint(
            method='POST',
            path='/scanner/dte',
            expected_status=200,
            body={'max_dte': 7, 'limit': 20},
            validate_response=self._validate_dte_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_post('/scanner/dte', json={'max_dte': 7, 'limit': 20})
            if response.status_code == 200:
                self._scanner_results = response.json()

        # Test with different DTE ranges
        for max_dte in [3, 5, 7]:
            result = await self.test_endpoint(
                method='POST',
                path='/scanner/dte',
                expected_status=200,
                body={'max_dte': max_dte, 'limit': 10},
            )
            results.append(result)

        return results

    def _validate_dte_response(self, response) -> List[Issue]:
        """Custom validation for DTE scanner response"""
        issues = []
        data = response.json()

        results = data.get('results', data) if isinstance(data, dict) else data
        if not isinstance(results, list):
            issues.append(Issue(
                title="DTE scanner didn't return array",
                description=f"Expected array, got {type(data).__name__}",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        for i, result in enumerate(results[:10]):
            # Check required fields
            required = ['symbol', 'expiration', 'dte', 'strike', 'premium']
            for field in required:
                if field not in result:
                    issues.append(Issue(
                        title=f"Missing field in result {i}: {field}",
                        description=f"DTE scanner result missing required field",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Validate DTE is within expected range
            dte = result.get('dte', 0)
            if dte > 7:
                issues.append(Issue(
                    title=f"DTE out of range: {result.get('symbol')}",
                    description=f"DTE is {dte}, expected <= 7",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="filter",
                ))

            if dte < 0:
                issues.append(Issue(
                    title=f"Negative DTE: {result.get('symbol')}",
                    description=f"DTE is {dte}, options shouldn't have negative DTE",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="calculation",
                ))

            # Validate expiration matches DTE
            expiration = result.get('expiration', '')
            if expiration:
                try:
                    if 'T' in expiration:
                        exp_date = datetime.fromisoformat(expiration.replace('Z', '+00:00'))
                    else:
                        exp_date = datetime.strptime(expiration, '%Y-%m-%d')

                    calculated_dte = self.dte_calculator.calculate_calendar_days(exp_date)

                    if abs(calculated_dte - dte) > 1:  # Allow 1 day tolerance
                        issues.append(Issue(
                            title=f"DTE mismatch: {result.get('symbol')}",
                            description=f"Reported DTE {dte}, calculated {calculated_dte}",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="calculation",
                        ))
                except Exception as e:
                    issues.append(Issue(
                        title=f"Invalid expiration format: {result.get('symbol')}",
                        description=f"Could not parse: {expiration}",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to DTE scanner page
            success = await self.navigate_to('/dte-scanner')
            if not success:
                success = await self.navigate_to('/scanner/dte')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to DTE scanner",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to DTE scanner page",
                        description="Could not load /dte-scanner route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test DTE filter
            results.append(await self.test_element_exists(
                "[data-testid='dte-filter'], input[type='number'], select",
                "DTE Filter"
            ))

            # Test results table
            results.append(await self.test_element_exists(
                "[data-testid='dte-results'], table, .results-table",
                "Results Table"
            ))

            # Take screenshot
            await self.take_screenshot("dte_scanner_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="DTE Scanner UI Tests",
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
        """Test DTE scanner business logic"""
        results = []
        issues = []

        if self._scanner_results is None:
            response = await self.http_post('/scanner/dte', json={'max_dte': 7, 'limit': 20})
            if response.status_code == 200:
                self._scanner_results = response.json()

        scanner_results = self._scanner_results.get('results', self._scanner_results) if isinstance(self._scanner_results, dict) else self._scanner_results

        if scanner_results:
            for result in scanner_results[:10]:
                symbol = result.get('symbol', 'UNKNOWN')
                dte = result.get('dte', 0)
                premium = result.get('premium', 0)
                theta = result.get('theta', result.get('greeks', {}).get('theta', 0))

                # Verify theta decay rate makes sense for short DTE
                if dte > 0 and dte <= 7 and premium > 0:
                    # Theta should be significant for short-term options
                    daily_decay_pct = abs(theta) / premium * 100 if premium > 0 else 0

                    # For 7-day options, daily decay should be noticeable
                    if daily_decay_pct < 1 and dte <= 3:
                        issues.append(Issue(
                            title=f"Low theta decay: {symbol}",
                            description=f"Daily decay {daily_decay_pct:.2f}% seems low for {dte} DTE",
                            severity=IssueSeverity.LOW,
                            feature=self.feature_name,
                            component="calculation",
                        ))

                # Validate premium is reasonable
                stock_price = result.get('stock_price', result.get('underlying_price', 0))
                strike = result.get('strike', 0)

                if stock_price > 0 and premium > 0:
                    premium_pct = premium / stock_price * 100

                    # For short DTE, premium % should be relatively low
                    if premium_pct > 20:
                        issues.append(Issue(
                            title=f"High premium % for short DTE: {symbol}",
                            description=f"Premium {premium_pct:.1f}% seems high for {dte} DTE",
                            severity=IssueSeverity.LOW,
                            feature=self.feature_name,
                            component="data_quality",
                        ))

        results.append(TestResult(
            test_name="DTE Scanner Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test DTE scanner data consistency"""
        results = []
        issues = []

        # Check results are sorted by DTE
        if self._scanner_results:
            scanner_results = self._scanner_results.get('results', self._scanner_results) if isinstance(self._scanner_results, dict) else self._scanner_results

            if isinstance(scanner_results, list) and len(scanner_results) > 1:
                dtes = [r.get('dte', 0) for r in scanner_results]
                if dtes != sorted(dtes):
                    issues.append(Issue(
                        title="Results not sorted by DTE",
                        description="DTE scanner results should be sorted by DTE ascending",
                        severity=IssueSeverity.LOW,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        # Check for duplicate entries
        if self._scanner_results:
            scanner_results = self._scanner_results.get('results', self._scanner_results) if isinstance(self._scanner_results, dict) else self._scanner_results

            if isinstance(scanner_results, list):
                seen = set()
                for r in scanner_results:
                    key = f"{r.get('symbol', '')}_{r.get('strike', '')}_{r.get('expiration', '')}_{r.get('option_type', '')}"
                    if key in seen:
                        issues.append(Issue(
                            title="Duplicate scanner result",
                            description=f"Same option appears multiple times",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="data_quality",
                        ))
                    seen.add(key)

        results.append(TestResult(
            test_name="DTE Scanner Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
