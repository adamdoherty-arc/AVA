"""
Positions SpecAgent - Tier 1 (Critical)

Deeply understands the Positions feature.
Tests: API endpoints, data consistency, P&L calculations, Greeks.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent
from ..validators.greeks_validator import GreeksValidator
from ..validators.pl_calculator import PLCalculator, StockPosition, OptionPosition
from ..validators.dte_calculator import DTECalculator

logger = logging.getLogger(__name__)


@register_spec_agent('positions')
class PositionsSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Positions feature.

    Validates:
    - Portfolio positions API endpoints
    - P&L calculations accuracy
    - Options Greeks correctness
    - DTE calculations
    - Data consistency with Dashboard
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='positions',
            description='Portfolio positions management - stocks and options',
            enable_browser=True,
            enable_database=True,
        )

        # Initialize validators
        self.greeks_validator = GreeksValidator()
        self.pl_calculator = PLCalculator()
        self.dte_calculator = DTECalculator()

        # Store fetched data for cross-validation
        self._positions_data = None
        self._dashboard_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all positions-related API endpoints"""
        results = []

        # Test /api/portfolio/positions
        result = await self.test_endpoint(
            method='GET',
            path='/portfolio/positions',
            expected_status=200,
            expected_fields=['summary', 'stocks', 'options'],
            validate_response=self._validate_positions_response,
        )
        results.append(result)

        # Store for cross-validation
        if result.passed:
            response = await self.http_get('/portfolio/positions')
            self._positions_data = response.json()

        # Test /api/portfolio/summary
        result = await self.test_endpoint(
            method='GET',
            path='/portfolio/summary',
            expected_status=200,
            expected_fields=['total_value', 'buying_power'],
        )
        results.append(result)

        # Test /api/health for service availability
        result = await self.test_endpoint(
            method='GET',
            path='/health',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_positions_response(self, response) -> List[Issue]:
        """Custom validation for positions response"""
        issues = []
        data = response.json()

        # Check summary has required fields
        summary = data.get('summary', {})
        if summary.get('total_equity') is None:
            issues.append(Issue(
                title="Missing total_equity in summary",
                description="Summary should include total_equity value",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))

        # Check for empty data
        stocks = data.get('stocks', [])
        options = data.get('options', [])

        if len(stocks) == 0 and len(options) == 0:
            issues.append(Issue(
                title="Empty positions data",
                description="No stocks or options in response (may be correct if no positions)",
                severity=IssueSeverity.INFO,
                feature=self.feature_name,
                component="api",
            ))

        # Check options have required Greeks
        for i, opt in enumerate(options):
            if not opt.get('greeks') and opt.get('dte', 0) > 0:
                issues.append(Issue(
                    title=f"Missing Greeks for {opt.get('symbol', f'option_{i}')}",
                    description="Active options should have Greeks data",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="api",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to positions page
            success = await self.navigate_to('/positions')
            if not success:
                results.append(TestResult(
                    test_name="Navigate to /positions",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to positions page",
                        description="Could not load /positions route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test summary cards exist
            results.append(await self.test_element_exists(
                "[data-testid='total-equity'], .total-equity, .portfolio-value",
                "Total Equity Card"
            ))

            # Test positions table exists
            results.append(await self.test_element_exists(
                "[data-testid='stocks-table'], table, .positions-table",
                "Positions Table"
            ))

            # Test sync button exists and is clickable
            results.append(await self.test_button_clickable(
                "[data-testid='sync-button'], button:has-text('Sync')",
                "Sync Button"
            ))

            # Take screenshot for visual verification
            await self.take_screenshot("positions_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="UI Component Tests",
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
        """Test P&L calculations, Greeks, and DTE"""
        results = []

        # Get positions data if not already fetched
        if self._positions_data is None:
            response = await self.http_get('/portfolio/positions')
            if response.status_code == 200:
                self._positions_data = response.json()
            else:
                results.append(TestResult(
                    test_name="Fetch positions for validation",
                    passed=False,
                    issues=[Issue(
                        title="Could not fetch positions data",
                        description=f"API returned status {response.status_code}",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="business_logic",
                    )],
                ))
                return results

        # Validate Stock P&L
        stock_issues = []
        for stock in self._positions_data.get('stocks', []):
            position = StockPosition(
                symbol=stock.get('symbol', 'UNKNOWN'),
                quantity=stock.get('quantity', 0),
                avg_price=stock.get('avg_buy_price', 0),
                current_price=stock.get('current_price', 0),
                reported_pl=stock.get('pl'),
                reported_pl_pct=stock.get('pl_pct'),
            )
            issues = self.pl_calculator.validate_stock_pl(position)
            for issue in issues:
                stock_issues.append(Issue(
                    title=f"Stock P&L Error: {position.symbol}",
                    description=issue.get('message', 'P&L calculation error'),
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="calculation",
                    expected=str(issue.get('expected')),
                    actual=str(issue.get('reported')),
                ))

        results.append(TestResult(
            test_name="Stock P&L Calculations",
            passed=len(stock_issues) == 0,
            issues=stock_issues,
        ))

        # Validate Option P&L and Greeks
        option_issues = []
        for opt in self._positions_data.get('options', []):
            # Validate P&L
            position = OptionPosition(
                symbol=opt.get('symbol', 'UNKNOWN'),
                quantity=opt.get('quantity', 0),
                avg_price=opt.get('avg_price', 0),
                current_price=opt.get('current_price', 0),
                reported_pl=opt.get('pl'),
            )
            pl_issues = self.pl_calculator.validate_option_pl(position)
            for issue in pl_issues:
                option_issues.append(Issue(
                    title=f"Option P&L Error: {position.symbol}",
                    description=issue.get('message', 'P&L calculation error'),
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="calculation",
                ))

            # Validate DTE
            expiration = opt.get('expiration')
            reported_dte = opt.get('dte')
            if expiration and reported_dte is not None:
                dte_issues = self.dte_calculator.validate_dte(reported_dte, expiration)
                for issue in dte_issues:
                    option_issues.append(Issue(
                        title=f"DTE Error: {opt.get('symbol')}",
                        description=issue.get('message', 'DTE calculation error'),
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="calculation",
                    ))

            # Validate Greeks if present
            greeks = opt.get('greeks', {})
            if greeks and opt.get('dte', 0) > 0:
                stock_price = opt.get('underlying_price', 0)
                strike = opt.get('strike', 0)
                dte = opt.get('dte', 0)
                iv = opt.get('iv', 0.3)
                is_call = opt.get('option_type', '').lower() == 'call'

                if stock_price > 0 and strike > 0 and dte > 0:
                    greek_issues = self.greeks_validator.validate_greeks(
                        reported=greeks,
                        stock_price=stock_price,
                        strike=strike,
                        days_to_expiry=dte,
                        volatility=iv,
                        is_call=is_call,
                    )
                    for issue in greek_issues:
                        option_issues.append(Issue(
                            title=f"Greeks Error: {opt.get('symbol')} - {issue.get('greek')}",
                            description=issue.get('message', 'Greeks validation error'),
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="calculation",
                        ))

        results.append(TestResult(
            test_name="Option Calculations (P&L, DTE, Greeks)",
            passed=len(option_issues) == 0,
            issues=option_issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test data consistency across views"""
        results = []
        issues = []

        # Fetch dashboard data for comparison
        response = await self.http_get('/dashboard/summary')
        if response.status_code == 200:
            self._dashboard_data = response.json()
        else:
            issues.append(Issue(
                title="Could not fetch dashboard data",
                description="Cannot compare positions total with dashboard",
                severity=IssueSeverity.MEDIUM,
                feature=self.feature_name,
                component="consistency",
            ))

        # Compare totals
        if self._positions_data and self._dashboard_data:
            positions_total = self._positions_data.get('summary', {}).get('total_equity', 0)
            dashboard_total = self._dashboard_data.get('total_value', 0)

            if positions_total and dashboard_total:
                diff = abs(positions_total - dashboard_total)
                if diff > 0.01:  # $0.01 tolerance
                    issues.append(Issue(
                        title="Portfolio totals don't match",
                        description=f"Positions: ${positions_total:.2f}, Dashboard: ${dashboard_total:.2f}",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="consistency",
                        expected=f"${positions_total:.2f}",
                        actual=f"${dashboard_total:.2f}",
                    ))

            # Check position count matches
            positions_count = self._positions_data.get('summary', {}).get('total_positions', 0)
            stocks_count = len(self._positions_data.get('stocks', []))
            options_count = len(self._positions_data.get('options', []))

            if positions_count != (stocks_count + options_count):
                issues.append(Issue(
                    title="Position count mismatch",
                    description=f"Summary says {positions_count}, but found {stocks_count} stocks + {options_count} options",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="consistency",
                ))

        results.append(TestResult(
            test_name="Data Consistency Checks",
            passed=len(issues) == 0,
            issues=issues,
        ))

        # Check for null values in critical fields
        null_issues = await self._check_null_values()
        results.append(TestResult(
            test_name="Null Value Checks",
            passed=len(null_issues) == 0,
            issues=null_issues,
        ))

        return results

    async def _check_null_values(self) -> List[Issue]:
        """Check for unexpected null values"""
        issues = []

        if not self._positions_data:
            return issues

        # Check stocks for nulls
        for stock in self._positions_data.get('stocks', []):
            symbol = stock.get('symbol', 'UNKNOWN')
            critical_fields = ['quantity', 'current_price', 'pl']
            for field in critical_fields:
                if stock.get(field) is None:
                    issues.append(Issue(
                        title=f"Null value: {symbol}.{field}",
                        description=f"Stock {symbol} has null {field}",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        # Check options for nulls
        for opt in self._positions_data.get('options', []):
            symbol = opt.get('symbol', 'UNKNOWN')
            critical_fields = ['quantity', 'current_price', 'strike', 'expiration']
            for field in critical_fields:
                if opt.get(field) is None:
                    issues.append(Issue(
                        title=f"Null value: {symbol}.{field}",
                        description=f"Option {symbol} has null {field}",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        return issues
