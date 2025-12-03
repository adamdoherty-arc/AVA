"""
Options Analysis SpecAgent - Tier 1 (Critical)

Deeply understands the Options Analysis feature.
Tests: Options chain, Greeks, IV calculations.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent
from ..validators.greeks_validator import GreeksValidator

logger = logging.getLogger(__name__)


@register_spec_agent('options-analysis')
class OptionsAnalysisSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Options Analysis feature.

    Validates:
    - Options chain API endpoints
    - Greeks calculations
    - IV surface data
    - Strike price display
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='options-analysis',
            description='Options chain and Greeks analysis',
            enable_browser=True,
            enable_database=True,
        )
        self._chain_data = None
        self.greeks_validator = GreeksValidator()

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all options analysis API endpoints"""
        results = []

        # Test /api/options/{symbol}/chain with a common symbol
        test_symbol = 'AAPL'
        result = await self.test_endpoint(
            method='GET',
            path=f'/options/{test_symbol}/chain',
            expected_status=200,
            validate_response=self._validate_chain_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get(f'/options/{test_symbol}/chain')
            if response.status_code == 200:
                self._chain_data = response.json()

        # Test health endpoint
        result = await self.test_endpoint(
            method='GET',
            path='/health',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_chain_response(self, response) -> List[Issue]:
        """Custom validation for options chain response"""
        issues = []
        data = response.json()

        # Check for calls and puts
        calls = data.get('calls', [])
        puts = data.get('puts', [])

        if not calls and not puts:
            issues.append(Issue(
                title="Empty options chain",
                description="No calls or puts in response",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        # Validate option data structure
        for option_type, options in [('calls', calls), ('puts', puts)]:
            for i, opt in enumerate(options[:3]):  # Check first 3
                required = ['strike', 'expiration', 'bid', 'ask']
                for field in required:
                    if field not in opt:
                        issues.append(Issue(
                            title=f"Missing field in {option_type}[{i}]: {field}",
                            description=f"Options chain missing required field",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="api",
                        ))

                # Validate bid/ask spread makes sense
                bid = opt.get('bid', 0)
                ask = opt.get('ask', 0)
                if bid > ask and bid > 0:
                    issues.append(Issue(
                        title=f"Invalid bid/ask: {option_type}[{i}]",
                        description=f"Bid ${bid} > Ask ${ask}",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to options analysis page
            success = await self.navigate_to('/options')
            if not success:
                success = await self.navigate_to('/options-analysis')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to options analysis",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to options page",
                        description="Could not load /options route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test symbol input exists
            results.append(await self.test_element_exists(
                "input[placeholder*='symbol'], input[name='symbol'], [data-testid='symbol-input']",
                "Symbol Input"
            ))

            # Test options chain table
            results.append(await self.test_element_exists(
                "[data-testid='options-chain'], table, .options-table",
                "Options Chain Table"
            ))

            # Take screenshot
            await self.take_screenshot("options_analysis_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Options Analysis UI Tests",
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
        """Test Greeks and IV calculations"""
        results = []
        issues = []

        if self._chain_data is None:
            response = await self.http_get('/options/AAPL/chain')
            if response.status_code == 200:
                self._chain_data = response.json()
            else:
                results.append(TestResult(
                    test_name="Fetch options chain",
                    passed=False,
                    issues=[Issue(
                        title="Could not fetch options chain",
                        description=f"API returned status {response.status_code}",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="business_logic",
                    )],
                ))
                return results

        underlying_price = self._chain_data.get('underlying_price', 0)

        # Validate calls
        for call in self._chain_data.get('calls', [])[:5]:
            greeks = call.get('greeks', {})
            if greeks:
                # Call delta should be between 0 and 1
                delta = greeks.get('delta', 0)
                error = self.greeks_validator.validate_delta_range(delta, is_call=True)
                if error:
                    issues.append(Issue(
                        title=f"Invalid call delta: {call.get('strike')}",
                        description=error,
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="calculation",
                    ))

                # Theta should be negative
                theta = greeks.get('theta', 0)
                error = self.greeks_validator.validate_theta_sign(theta)
                if error:
                    issues.append(Issue(
                        title=f"Invalid theta: {call.get('strike')}",
                        description=error,
                        severity=IssueSeverity.LOW,
                        feature=self.feature_name,
                        component="calculation",
                    ))

        # Validate puts
        for put in self._chain_data.get('puts', [])[:5]:
            greeks = put.get('greeks', {})
            if greeks:
                # Put delta should be between -1 and 0
                delta = greeks.get('delta', 0)
                error = self.greeks_validator.validate_delta_range(delta, is_call=False)
                if error:
                    issues.append(Issue(
                        title=f"Invalid put delta: {put.get('strike')}",
                        description=error,
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="calculation",
                    ))

        results.append(TestResult(
            test_name="Greeks Validation",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test options data consistency"""
        results = []
        issues = []

        if self._chain_data:
            # Check strikes are sorted
            calls = self._chain_data.get('calls', [])
            if calls:
                strikes = [c.get('strike', 0) for c in calls]
                if strikes != sorted(strikes):
                    issues.append(Issue(
                        title="Calls not sorted by strike",
                        description="Options chain should be sorted by strike price",
                        severity=IssueSeverity.LOW,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

            # Check for zero IV (suspicious)
            for opt in calls[:5]:
                iv = opt.get('iv', opt.get('implied_volatility', 0))
                if iv == 0:
                    issues.append(Issue(
                        title=f"Zero IV: strike {opt.get('strike')}",
                        description="Implied volatility should not be zero for active options",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        results.append(TestResult(
            test_name="Options Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
