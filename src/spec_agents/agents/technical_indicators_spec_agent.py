"""
Technical Indicators SpecAgent - Tier 3

Deeply understands the Technical Indicators feature.
Tests: Technical analysis, indicator calculations, chart data.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('technical-indicators')
class TechnicalIndicatorsSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Technical Indicators feature.

    Validates:
    - Technical data API endpoints
    - Indicator calculation accuracy
    - Chart data formatting
    - Signal generation
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='technical-indicators',
            description='Technical analysis and indicators',
            enable_browser=True,
            enable_database=True,
        )
        self._indicators_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all technical indicator API endpoints"""
        results = []

        # Test /api/technicals/{symbol}
        result = await self.test_endpoint(
            method='GET',
            path='/technicals/AAPL',
            expected_status=200,
            validate_response=self._validate_indicators_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/technicals/AAPL')
            if response.status_code == 200:
                self._indicators_data = response.json()

        # Test specific indicators
        for indicator in ['sma', 'ema', 'rsi', 'macd']:
            result = await self.test_endpoint(
                method='GET',
                path=f'/technicals/AAPL/{indicator}',
                expected_status=200,
            )
            results.append(result)

        return results

    def _validate_indicators_response(self, response) -> List[Issue]:
        """Custom validation for indicators response"""
        issues = []
        data = response.json()

        # Check for common indicators
        expected_indicators = ['sma', 'ema', 'rsi', 'macd', 'volume']
        for indicator in expected_indicators:
            if indicator not in data and f'{indicator}_20' not in data:
                issues.append(Issue(
                    title=f"Missing indicator: {indicator}",
                    description=f"Technical data should include {indicator}",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="api",
                ))

        # Validate RSI range (0-100)
        rsi = data.get('rsi', data.get('rsi_14', None))
        if rsi is not None:
            if isinstance(rsi, (int, float)):
                if rsi < 0 or rsi > 100:
                    issues.append(Issue(
                        title="RSI out of range",
                        description=f"RSI value {rsi} should be 0-100",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="calculation",
                    ))

        # Validate SMA/EMA values are positive
        for ma_type in ['sma_20', 'sma_50', 'ema_20', 'ema_50']:
            ma_value = data.get(ma_type)
            if ma_value is not None and ma_value < 0:
                issues.append(Issue(
                    title=f"Negative moving average: {ma_type}",
                    description=f"{ma_type} = {ma_value}",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="calculation",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            success = await self.navigate_to('/technicals')
            if not success:
                success = await self.navigate_to('/chart')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to technicals page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to technicals page",
                        description="Could not load /technicals route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test symbol input
            results.append(await self.test_element_exists(
                "[data-testid='symbol-input'], input[name='symbol']",
                "Symbol Input"
            ))

            # Test chart container
            results.append(await self.test_element_exists(
                "[data-testid='chart'], canvas, .chart-container",
                "Chart Container"
            ))

            # Test indicator selectors
            results.append(await self.test_element_exists(
                "[data-testid='indicator-select'], .indicator-checkboxes",
                "Indicator Selectors"
            ))

            await self.take_screenshot("technical_indicators_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Technical Indicators UI Tests",
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
        """Test technical indicator calculations"""
        results = []
        issues = []

        if self._indicators_data:
            # Validate SMA < current price for uptrend stocks
            current_price = self._indicators_data.get('price', self._indicators_data.get('close', 0))
            sma_20 = self._indicators_data.get('sma_20', 0)
            sma_50 = self._indicators_data.get('sma_50', 0)

            # SMA 20 should be between SMA 50 and price for consistent trend
            if sma_20 > 0 and sma_50 > 0 and current_price > 0:
                # Check for golden cross / death cross consistency
                signal = self._indicators_data.get('signal', '')
                if signal == 'bullish' and sma_20 < sma_50:
                    issues.append(Issue(
                        title="Inconsistent bullish signal",
                        description=f"Signal is bullish but SMA20 ({sma_20:.2f}) < SMA50 ({sma_50:.2f})",
                        severity=IssueSeverity.LOW,
                        feature=self.feature_name,
                        component="calculation",
                    ))

            # Validate MACD histogram
            macd = self._indicators_data.get('macd', {})
            if isinstance(macd, dict):
                macd_line = macd.get('macd_line', 0)
                signal_line = macd.get('signal_line', 0)
                histogram = macd.get('histogram', 0)

                expected_histogram = macd_line - signal_line
                if abs(histogram - expected_histogram) > 0.01:
                    issues.append(Issue(
                        title="MACD histogram mismatch",
                        description=f"Histogram {histogram:.4f} != MACD - Signal ({expected_histogram:.4f})",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="calculation",
                    ))

        results.append(TestResult(
            test_name="Technical Indicator Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test technical data consistency"""
        results = []
        issues = []

        if self._indicators_data:
            # Check timestamp freshness
            timestamp = self._indicators_data.get('timestamp', self._indicators_data.get('updated_at'))
            if timestamp:
                from datetime import datetime, timedelta
                try:
                    if isinstance(timestamp, str):
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        ts = datetime.fromtimestamp(timestamp)

                    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
                    age = now - ts

                    # Data shouldn't be more than 1 hour old during market hours
                    if age > timedelta(hours=1):
                        issues.append(Issue(
                            title="Stale technical data",
                            description=f"Data is {age.total_seconds() / 3600:.1f} hours old",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="data_freshness",
                        ))
                except Exception as e:
                    logger.debug(f"Could not parse timestamp: {e}")

        results.append(TestResult(
            test_name="Technical Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
