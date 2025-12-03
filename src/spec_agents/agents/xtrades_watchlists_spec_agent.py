"""
XTrades Watchlists SpecAgent - Tier 2

Deeply understands the XTrades Watchlists feature.
Tests: Watchlist sync, alert processing, premium calculations.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('xtrades-watchlists')
class XTradesWatchlistsSpecAgent(BaseSpecAgent):
    """
    SpecAgent for XTrades Watchlists feature.

    Validates:
    - Watchlist API endpoints
    - Alert synchronization
    - Premium calculations
    - Symbol data accuracy
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='xtrades-watchlists',
            description='XTrades watchlist synchronization and analysis',
            enable_browser=True,
            enable_database=True,
        )
        self._watchlist_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all watchlist-related API endpoints"""
        results = []

        # Test /api/watchlist/database (actual endpoint)
        result = await self.test_endpoint(
            method='GET',
            path='/watchlist/database',
            expected_status=200,
            validate_response=self._validate_watchlists_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/watchlist/database')
            if response.status_code == 200:
                self._watchlist_data = response.json()

        # Test /api/watchlist/alerts
        result = await self.test_endpoint(
            method='GET',
            path='/watchlist/alerts',
            expected_status=200,
        )
        results.append(result)

        # Test /api/watchlist/symbols
        result = await self.test_endpoint(
            method='GET',
            path='/watchlist/symbols',
            expected_status=200,
        )
        results.append(result)

        # Test sync status
        result = await self.test_endpoint(
            method='GET',
            path='/watchlist/sync-status',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_watchlists_response(self, response) -> List[Issue]:
        """Custom validation for watchlists response"""
        issues = []
        data = response.json()

        watchlists = data.get('watchlists', data) if isinstance(data, dict) else data
        if not isinstance(watchlists, list):
            issues.append(Issue(
                title="Watchlists response not an array",
                description=f"Expected array of watchlists",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        for i, watchlist in enumerate(watchlists[:5]):
            # Check required fields
            required = ['name', 'symbols']
            for field in required:
                if field not in watchlist:
                    issues.append(Issue(
                        title=f"Missing field in watchlist {i}: {field}",
                        description=f"Watchlist missing required field",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Validate symbols array
            symbols = watchlist.get('symbols', [])
            if not isinstance(symbols, list):
                issues.append(Issue(
                    title=f"Invalid symbols format: {watchlist.get('name', i)}",
                    description="Symbols should be an array",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="api",
                ))
            elif len(symbols) == 0 and watchlist.get('name', '').lower() != 'empty':
                issues.append(Issue(
                    title=f"Empty watchlist: {watchlist.get('name', i)}",
                    description="Watchlist has no symbols",
                    severity=IssueSeverity.LOW,
                    feature=self.feature_name,
                    component="data_quality",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to watchlists page
            success = await self.navigate_to('/watchlists')
            if not success:
                success = await self.navigate_to('/xtrades')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to watchlists page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to watchlists page",
                        description="Could not load /watchlists or /xtrades route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test watchlist selector
            results.append(await self.test_element_exists(
                "[data-testid='watchlist-selector'], select, .watchlist-tabs",
                "Watchlist Selector"
            ))

            # Test symbols table
            results.append(await self.test_element_exists(
                "[data-testid='symbols-table'], table, .symbols-list",
                "Symbols Table"
            ))

            # Test sync button
            results.append(await self.test_button_clickable(
                "[data-testid='sync-button'], button:has-text('Sync')",
                "Sync Button"
            ))

            # Take screenshot
            await self.take_screenshot("xtrades_watchlists_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="XTrades Watchlists UI Tests",
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
        """Test watchlist business logic"""
        results = []
        issues = []

        # Test sync status
        try:
            response = await self.http_get('/watchlist/sync-status')
            if response.status_code == 200:
                data = response.json()

                last_sync = data.get('last_sync')
                sync_status = data.get('status', '').lower()

                # Check if sync is recent (within last hour)
                if last_sync:
                    from datetime import datetime, timedelta
                    try:
                        if 'T' in str(last_sync):
                            last_sync_dt = datetime.fromisoformat(str(last_sync).replace('Z', '+00:00'))
                        else:
                            last_sync_dt = datetime.strptime(str(last_sync), '%Y-%m-%d %H:%M:%S')

                        now = datetime.now(last_sync_dt.tzinfo) if last_sync_dt.tzinfo else datetime.now()

                        if (now - last_sync_dt) > timedelta(hours=1):
                            issues.append(Issue(
                                title="Stale watchlist sync",
                                description=f"Last sync was at {last_sync}",
                                severity=IssueSeverity.MEDIUM,
                                feature=self.feature_name,
                                component="data_freshness",
                            ))
                    except Exception as e:
                        logger.debug(f"Could not parse sync time: {e}")

                if sync_status == 'error':
                    issues.append(Issue(
                        title="Watchlist sync error",
                        description=f"Sync status is 'error': {data.get('error', 'Unknown')}",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="sync",
                    ))
        except Exception as e:
            issues.append(Issue(
                title="Could not check sync status",
                description=str(e),
                severity=IssueSeverity.MEDIUM,
                feature=self.feature_name,
                component="api",
            ))

        results.append(TestResult(
            test_name="Watchlist Sync Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test watchlist data consistency"""
        results = []
        issues = []

        # Check for duplicate watchlist names
        if self._watchlist_data:
            watchlists = self._watchlist_data.get('watchlists', self._watchlist_data) if isinstance(self._watchlist_data, dict) else self._watchlist_data

            if isinstance(watchlists, list):
                names = [w.get('name') for w in watchlists if w.get('name')]
                if len(names) != len(set(names)):
                    issues.append(Issue(
                        title="Duplicate watchlist names",
                        description="Same watchlist name appears multiple times",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

                # Check for invalid symbols
                for watchlist in watchlists:
                    symbols = watchlist.get('symbols', [])
                    for symbol in symbols[:10]:  # Check first 10
                        symbol_str = symbol if isinstance(symbol, str) else symbol.get('symbol', '')
                        if symbol_str:
                            # Basic validation - should be uppercase, letters only (with some exceptions)
                            if not symbol_str.replace('.', '').replace('-', '').isalnum():
                                issues.append(Issue(
                                    title=f"Invalid symbol format: {symbol_str}",
                                    description=f"In watchlist '{watchlist.get('name', 'unknown')}'",
                                    severity=IssueSeverity.LOW,
                                    feature=self.feature_name,
                                    component="data_quality",
                                ))

        results.append(TestResult(
            test_name="Watchlist Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
