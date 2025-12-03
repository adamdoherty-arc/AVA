"""
Best Bets SpecAgent - Tier 3

Deeply understands the Best Bets/Unified Betting feature.
Tests: Betting recommendations, odds aggregation, value calculations.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('best-bets')
class BestBetsSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Best Bets feature.

    Validates:
    - Best bets API endpoints
    - Value calculation accuracy
    - Multi-source odds aggregation
    - Recommendation quality
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='best-bets',
            description='AI-powered best betting recommendations',
            enable_browser=True,
            enable_database=True,
        )
        self._bets_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all best bets API endpoints"""
        results = []

        # Test /api/sports/best-bets (actual endpoint)
        result = await self.test_endpoint(
            method='GET',
            path='/sports/best-bets',
            expected_status=200,
            validate_response=self._validate_bets_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/sports/best-bets')
            if response.status_code == 200:
                self._bets_data = response.json()

        # Test /api/sports/value-bets
        result = await self.test_endpoint(
            method='GET',
            path='/sports/value-bets',
            expected_status=200,
        )
        results.append(result)

        # Test sport-specific endpoints
        for sport in ['nfl', 'nba', 'mlb']:
            result = await self.test_endpoint(
                method='GET',
                path=f'/sports/{sport}/bets',
                expected_status=200,
            )
            results.append(result)

        return results

    def _validate_bets_response(self, response) -> List[Issue]:
        """Custom validation for bets response"""
        issues = []
        data = response.json()

        bets = data.get('bets', data) if isinstance(data, dict) else data
        if not isinstance(bets, list):
            issues.append(Issue(
                title="Bets response not an array",
                description=f"Expected array of bets",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        for i, bet in enumerate(bets[:5]):
            # Check required fields
            required = ['game', 'pick', 'odds', 'confidence']
            for field in required:
                if field not in bet:
                    issues.append(Issue(
                        title=f"Missing field in bet {i}: {field}",
                        description=f"Bet recommendation missing required field",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Validate confidence score
            confidence = bet.get('confidence', 0)
            if confidence < 0 or confidence > 100:
                issues.append(Issue(
                    title=f"Invalid confidence score: bet {i}",
                    description=f"Confidence {confidence} should be 0-100",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="data_quality",
                ))

            # Validate odds format
            odds = bet.get('odds', 0)
            if isinstance(odds, (int, float)):
                # American odds should be < -100 or > +100
                if -100 < odds < 100 and odds != 0:
                    issues.append(Issue(
                        title=f"Invalid odds format: bet {i}",
                        description=f"Odds {odds} invalid for American format",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            success = await self.navigate_to('/best-bets')
            if not success:
                success = await self.navigate_to('/bets')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to best bets page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to best bets page",
                        description="Could not load /best-bets route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test bets list
            results.append(await self.test_element_exists(
                "[data-testid='bets-list'], .bets-cards, .bet-recommendations",
                "Bets List"
            ))

            # Test sport filter
            results.append(await self.test_element_exists(
                "[data-testid='sport-filter'], .sport-tabs, select",
                "Sport Filter"
            ))

            await self.take_screenshot("best_bets_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Best Bets UI Tests",
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
        """Test best bets business logic"""
        results = []
        issues = []

        if self._bets_data:
            bets = self._bets_data.get('bets', self._bets_data) if isinstance(self._bets_data, dict) else self._bets_data

            for bet in bets[:5]:
                # Validate expected value calculation
                odds = bet.get('odds', 0)
                confidence = bet.get('confidence', 0)
                ev = bet.get('expected_value', bet.get('ev', None))

                if ev is not None and odds != 0 and confidence > 0:
                    # Calculate expected EV
                    if odds > 0:
                        implied_prob = 100 / (odds + 100)
                    else:
                        implied_prob = abs(odds) / (abs(odds) + 100)

                    edge = (confidence / 100) - implied_prob
                    # EV should be positive if we're recommending the bet
                    if edge < 0 and bet.get('recommended', True):
                        issues.append(Issue(
                            title=f"Negative edge on recommended bet",
                            description=f"Bet on {bet.get('game', 'unknown')} has negative expected value",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="calculation",
                        ))

        results.append(TestResult(
            test_name="Best Bets Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test best bets data consistency"""
        results = []
        issues = []

        # Check for duplicate bets
        if self._bets_data:
            bets = self._bets_data.get('bets', self._bets_data) if isinstance(self._bets_data, dict) else self._bets_data

            if isinstance(bets, list):
                seen = set()
                for bet in bets:
                    key = f"{bet.get('game', '')}_{bet.get('pick', '')}_{bet.get('bet_type', '')}"
                    if key in seen:
                        issues.append(Issue(
                            title="Duplicate bet recommendation",
                            description=f"Same bet appears multiple times",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="data_quality",
                        ))
                    seen.add(key)

        results.append(TestResult(
            test_name="Best Bets Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
