"""
Kalshi Markets SpecAgent - Tier 2

Deeply understands the Kalshi prediction markets feature.
Tests: Markets API, odds display, event tracking.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('kalshi-markets')
class KalshiMarketsSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Kalshi Markets feature.

    Validates:
    - Kalshi API endpoints
    - Market data accuracy
    - Odds calculations
    - Event tracking
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='kalshi-markets',
            description='Kalshi prediction markets integration',
            enable_browser=True,
            enable_database=True,
        )
        self._markets_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all Kalshi-related API endpoints"""
        results = []

        # Test /api/predictions/kalshi (actual endpoint)
        result = await self.test_endpoint(
            method='GET',
            path='/predictions/kalshi',
            expected_status=200,
            validate_response=self._validate_markets_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/predictions/kalshi')
            if response.status_code == 200:
                self._markets_data = response.json()

        # Test /api/predictions/kalshi/categories
        result = await self.test_endpoint(
            method='GET',
            path='/predictions/kalshi/categories',
            expected_status=200,
        )
        results.append(result)

        # Test /api/predictions/kalshi/nfl (NFL-specific markets)
        result = await self.test_endpoint(
            method='GET',
            path='/predictions/kalshi/nfl',
            expected_status=200,
        )
        results.append(result)

        # Test /api/predictions/kalshi/opportunities
        result = await self.test_endpoint(
            method='GET',
            path='/predictions/kalshi/opportunities',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_markets_response(self, response) -> List[Issue]:
        """Custom validation for markets response"""
        issues = []
        data = response.json()

        markets = data.get('markets', data) if isinstance(data, dict) else data
        if not isinstance(markets, list):
            issues.append(Issue(
                title="Markets response not an array",
                description=f"Expected array of markets",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        for i, market in enumerate(markets[:5]):
            # Check required fields
            required = ['ticker', 'title', 'yes_bid', 'yes_ask']
            for field in required:
                if field not in market:
                    issues.append(Issue(
                        title=f"Missing field in market {i}: {field}",
                        description=f"Market data missing required field",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Validate yes/no prices
            yes_bid = market.get('yes_bid', 0)
            yes_ask = market.get('yes_ask', 0)
            no_bid = market.get('no_bid', 0)
            no_ask = market.get('no_ask', 0)

            # Check bid < ask (normal spread)
            if yes_bid > yes_ask and yes_bid > 0 and yes_ask > 0:
                issues.append(Issue(
                    title=f"Invalid yes bid/ask spread: {market.get('ticker', i)}",
                    description=f"Yes bid {yes_bid} > yes ask {yes_ask}",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="data_quality",
                ))

            # Check prices are in valid range (0-100 cents or 0-1)
            for price_name, price_val in [('yes_bid', yes_bid), ('yes_ask', yes_ask)]:
                if price_val < 0:
                    issues.append(Issue(
                        title=f"Negative price: {market.get('ticker', i)}",
                        description=f"{price_name} = {price_val}",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="data_quality",
                    ))
                elif price_val > 100:
                    issues.append(Issue(
                        title=f"Price out of range: {market.get('ticker', i)}",
                        description=f"{price_name} = {price_val} (>100)",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to Kalshi markets page
            success = await self.navigate_to('/kalshi')
            if not success:
                success = await self.navigate_to('/markets')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to Kalshi page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to Kalshi page",
                        description="Could not load /kalshi or /markets route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test markets list exists
            results.append(await self.test_element_exists(
                "[data-testid='markets-list'], .markets-table, .market-cards",
                "Markets List"
            ))

            # Test category filter
            results.append(await self.test_element_exists(
                "[data-testid='category-filter'], .category-tabs, select",
                "Category Filter"
            ))

            # Take screenshot
            await self.take_screenshot("kalshi_markets_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Kalshi Markets UI Tests",
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
        """Test Kalshi market logic"""
        results = []
        issues = []

        if self._markets_data is None:
            response = await self.http_get('/predictions/kalshi')
            if response.status_code == 200:
                self._markets_data = response.json()

        markets = self._markets_data.get('markets', self._markets_data) if isinstance(self._markets_data, dict) else self._markets_data

        if markets:
            for market in markets[:10]:
                # Check market isn't expired
                status = market.get('status', '').lower()
                if status in ['closed', 'settled', 'finalized']:
                    # Shouldn't show closed markets by default
                    issues.append(Issue(
                        title=f"Closed market displayed: {market.get('ticker', 'unknown')}",
                        description=f"Market status is '{status}' but still in active list",
                        severity=IssueSeverity.LOW,
                        feature=self.feature_name,
                        component="business_logic",
                    ))

                # Check for stale data
                volume = market.get('volume', market.get('volume_24h', 0))
                if volume == 0:
                    open_interest = market.get('open_interest', 0)
                    if open_interest > 100:  # Active market with no recent volume
                        issues.append(Issue(
                            title=f"Stale market data: {market.get('ticker', 'unknown')}",
                            description=f"Market has {open_interest} open interest but 0 volume",
                            severity=IssueSeverity.LOW,
                            feature=self.feature_name,
                            component="data_freshness",
                        ))

        results.append(TestResult(
            test_name="Market Logic Validation",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test Kalshi data consistency"""
        results = []
        issues = []

        # Check for duplicate tickers
        if self._markets_data:
            markets = self._markets_data.get('markets', self._markets_data) if isinstance(self._markets_data, dict) else self._markets_data

            if isinstance(markets, list):
                tickers = [m.get('ticker') for m in markets if m.get('ticker')]
                if len(tickers) != len(set(tickers)):
                    issues.append(Issue(
                        title="Duplicate market tickers",
                        description="Same ticker appears multiple times",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        results.append(TestResult(
            test_name="Kalshi Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
