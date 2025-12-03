"""
Signal Dashboard SpecAgent - Tier 3

Deeply understands the Signal Dashboard feature.
Tests: Trading signals, alert generation, signal history.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('signal-dashboard')
class SignalDashboardSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Signal Dashboard feature.

    Validates:
    - Signal API endpoints
    - Alert generation accuracy
    - Signal history tracking
    - Real-time updates
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='signal-dashboard',
            description='Trading signals and alerts dashboard',
            enable_browser=True,
            enable_database=True,
        )
        self._signals_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all signal dashboard API endpoints"""
        results = []

        # Test /api/signals
        result = await self.test_endpoint(
            method='GET',
            path='/signals',
            expected_status=200,
            validate_response=self._validate_signals_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/signals')
            if response.status_code == 200:
                self._signals_data = response.json()

        # Test /api/signals/history
        result = await self.test_endpoint(
            method='GET',
            path='/signals/history',
            expected_status=200,
        )
        results.append(result)

        # Test /api/signals/active
        result = await self.test_endpoint(
            method='GET',
            path='/signals/active',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_signals_response(self, response) -> List[Issue]:
        """Custom validation for signals response"""
        issues = []
        data = response.json()

        signals = data.get('signals', data) if isinstance(data, dict) else data
        if not isinstance(signals, list):
            issues.append(Issue(
                title="Signals response not an array",
                description=f"Expected array of signals",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        for i, signal in enumerate(signals[:5]):
            # Check required fields
            required = ['symbol', 'signal_type', 'timestamp']
            for field in required:
                if field not in signal:
                    issues.append(Issue(
                        title=f"Missing field in signal {i}: {field}",
                        description=f"Signal missing required field",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Validate signal type
            signal_type = signal.get('signal_type', '').lower()
            valid_types = ['buy', 'sell', 'hold', 'bullish', 'bearish', 'neutral', 'entry', 'exit']
            if signal_type and signal_type not in valid_types:
                issues.append(Issue(
                    title=f"Unknown signal type: {signal_type}",
                    description=f"Signal {i} has unrecognized type",
                    severity=IssueSeverity.LOW,
                    feature=self.feature_name,
                    component="data_quality",
                ))

            # Validate confidence if present
            confidence = signal.get('confidence', signal.get('strength', None))
            if confidence is not None:
                if isinstance(confidence, (int, float)) and (confidence < 0 or confidence > 100):
                    issues.append(Issue(
                        title=f"Invalid signal confidence: {signal.get('symbol', i)}",
                        description=f"Confidence {confidence} should be 0-100",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            success = await self.navigate_to('/signals')
            if not success:
                success = await self.navigate_to('/alerts')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to signals page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to signals page",
                        description="Could not load /signals route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test signals list
            results.append(await self.test_element_exists(
                "[data-testid='signals-list'], .signals-table, .alert-cards",
                "Signals List"
            ))

            # Test filter controls
            results.append(await self.test_element_exists(
                "[data-testid='signal-filters'], .filter-controls, select",
                "Signal Filters"
            ))

            await self.take_screenshot("signal_dashboard_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Signal Dashboard UI Tests",
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
        """Test signal generation logic"""
        results = []
        issues = []

        if self._signals_data:
            signals = self._signals_data.get('signals', self._signals_data) if isinstance(self._signals_data, dict) else self._signals_data

            for signal in signals[:10]:
                # Check signal timestamp is recent
                timestamp = signal.get('timestamp', signal.get('created_at'))
                if timestamp:
                    from datetime import datetime, timedelta
                    try:
                        if isinstance(timestamp, str):
                            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            ts = datetime.fromtimestamp(timestamp)

                        now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
                        age = now - ts

                        # Active signals shouldn't be too old
                        status = signal.get('status', 'active').lower()
                        if status == 'active' and age > timedelta(days=1):
                            issues.append(Issue(
                                title=f"Stale active signal: {signal.get('symbol', 'unknown')}",
                                description=f"Active signal is {age.days} days old",
                                severity=IssueSeverity.MEDIUM,
                                feature=self.feature_name,
                                component="data_freshness",
                            ))
                    except Exception as e:
                        logger.debug(f"Could not parse timestamp: {e}")

        results.append(TestResult(
            test_name="Signal Logic Validation",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test signal data consistency"""
        results = []
        issues = []

        # Check for duplicate active signals
        if self._signals_data:
            signals = self._signals_data.get('signals', self._signals_data) if isinstance(self._signals_data, dict) else self._signals_data

            if isinstance(signals, list):
                active_signals = [s for s in signals if s.get('status', 'active').lower() == 'active']
                symbol_types = {}
                for signal in active_signals:
                    key = f"{signal.get('symbol', '')}_{signal.get('signal_type', '')}"
                    if key in symbol_types:
                        issues.append(Issue(
                            title="Duplicate active signal",
                            description=f"Multiple active {signal.get('signal_type')} signals for {signal.get('symbol')}",
                            severity=IssueSeverity.LOW,
                            feature=self.feature_name,
                            component="data_quality",
                        ))
                    symbol_types[key] = True

        results.append(TestResult(
            test_name="Signal Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
