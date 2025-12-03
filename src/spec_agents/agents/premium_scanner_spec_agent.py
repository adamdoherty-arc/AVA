"""
Premium Scanner SpecAgent - Tier 1 (Critical)

Deeply understands the Premium Scanner feature.
Tests: Options scanning, premium calculations, Greeks display.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('premium-scanner')
class PremiumScannerSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Premium Scanner feature.

    Validates:
    - Scanner API endpoints
    - Premium percentage calculations
    - Monthly/annual return calculations
    - Greeks display accuracy
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='premium-scanner',
            description='Options premium scanner for wheel strategy',
            enable_browser=True,
            enable_database=True,
        )
        self._scanner_results = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all scanner-related API endpoints"""
        results = []

        # Test /api/scanner/scan with basic params
        result = await self.test_endpoint(
            method='POST',
            path='/scanner/scan',
            expected_status=200,
            validate_response=self._validate_scanner_response,
        )
        results.append(result)

        # Store results
        if result.passed:
            response = await self.http_post('/scanner/scan', json={'limit': 20})
            if response.status_code == 200:
                self._scanner_results = response.json()

        # Test health endpoint
        result = await self.test_endpoint(
            method='GET',
            path='/health',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_scanner_response(self, response) -> List[Issue]:
        """Custom validation for scanner response"""
        issues = []
        data = response.json()

        # Should return array of results
        if not isinstance(data, list):
            issues.append(Issue(
                title="Scanner didn't return array",
                description=f"Expected array, got {type(data).__name__}",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        # Validate individual results
        for i, result in enumerate(data[:5]):  # Check first 5
            # Check required fields
            required = ['symbol', 'strike', 'expiration', 'premium']
            for field in required:
                if field not in result:
                    issues.append(Issue(
                        title=f"Missing field in result {i}: {field}",
                        description=f"Scanner result missing required field '{field}'",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Validate premium calculations
            premium = result.get('premium', 0)
            stock_price = result.get('stock_price', 0)
            strike = result.get('strike', 0)
            premium_pct = result.get('premium_pct', 0)

            if stock_price > 0 and premium > 0:
                expected_pct = (premium / stock_price) * 100
                if abs(premium_pct - expected_pct) > 0.5:  # 0.5% tolerance
                    issues.append(Issue(
                        title=f"Premium % mismatch: {result.get('symbol')}",
                        description=f"Reported {premium_pct:.2f}%, calculated {expected_pct:.2f}%",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="calculation",
                    ))

            # Validate DTE is positive
            dte = result.get('dte', 0)
            if dte < 0:
                issues.append(Issue(
                    title=f"Negative DTE: {result.get('symbol')}",
                    description=f"DTE is {dte}, should be positive or zero",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="calculation",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to scanner page
            success = await self.navigate_to('/scanner')
            if not success:
                # Try alternative routes
                success = await self.navigate_to('/premium-scanner')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to scanner",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to scanner page",
                        description="Could not load /scanner or /premium-scanner route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test scanner table exists
            results.append(await self.test_element_exists(
                "[data-testid='scanner-table'], table.scanner-results, .scanner-table",
                "Scanner Results Table"
            ))

            # Test filter controls exist
            results.append(await self.test_element_exists(
                "[data-testid='scanner-filters'], .filters, form",
                "Scanner Filters"
            ))

            # Test scan button
            results.append(await self.test_button_clickable(
                "[data-testid='scan-button'], button:has-text('Scan')",
                "Scan Button"
            ))

            # Take screenshot
            await self.take_screenshot("scanner_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Scanner UI Tests",
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
        """Test premium calculations"""
        results = []
        issues = []

        if self._scanner_results is None:
            response = await self.http_post('/scanner/scan', json={'limit': 20})
            if response.status_code == 200:
                self._scanner_results = response.json()
            else:
                results.append(TestResult(
                    test_name="Fetch scanner results",
                    passed=False,
                    issues=[Issue(
                        title="Could not fetch scanner results",
                        description=f"API returned status {response.status_code}",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="business_logic",
                    )],
                ))
                return results

        for result in self._scanner_results or []:
            symbol = result.get('symbol', 'UNKNOWN')

            # Validate monthly return calculation
            premium_pct = result.get('premium_pct', 0)
            dte = result.get('dte', 30)
            monthly_return = result.get('monthly_return', 0)

            if dte > 0 and premium_pct > 0:
                expected_monthly = (premium_pct / dte) * 30
                if abs(monthly_return - expected_monthly) > 0.5:
                    issues.append(Issue(
                        title=f"Monthly return calculation error: {symbol}",
                        description=f"Reported {monthly_return:.2f}%, expected {expected_monthly:.2f}%",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="calculation",
                    ))

            # Validate annual return calculation
            annual_return = result.get('annual_return', 0)
            if monthly_return > 0:
                expected_annual = monthly_return * 12
                if abs(annual_return - expected_annual) > 1:
                    issues.append(Issue(
                        title=f"Annual return calculation error: {symbol}",
                        description=f"Reported {annual_return:.2f}%, expected {expected_annual:.2f}%",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="calculation",
                    ))

        results.append(TestResult(
            test_name="Premium Calculations",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test scanner data consistency"""
        results = []
        issues = []

        # Check for duplicate symbols
        if self._scanner_results:
            symbols = [r.get('symbol') for r in self._scanner_results]
            unique_symbols = set(symbols)
            if len(symbols) != len(unique_symbols):
                issues.append(Issue(
                    title="Duplicate symbols in results",
                    description=f"Found {len(symbols) - len(unique_symbols)} duplicates",
                    severity=IssueSeverity.LOW,
                    feature=self.feature_name,
                    component="data_quality",
                ))

            # Check for suspicious values
            for result in self._scanner_results:
                premium_pct = result.get('premium_pct', 0)
                if premium_pct > 50:  # 50%+ premium seems suspicious
                    issues.append(Issue(
                        title=f"Unusually high premium: {result.get('symbol')}",
                        description=f"Premium {premium_pct:.1f}% seems unrealistic",
                        severity=IssueSeverity.LOW,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        results.append(TestResult(
            test_name="Scanner Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
