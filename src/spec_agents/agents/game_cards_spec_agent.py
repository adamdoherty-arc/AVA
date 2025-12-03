"""
Game Cards SpecAgent - Tier 1 (Critical)

Deeply understands the Game Cards/Sports feature.
Tests: Live games, odds display, Kalshi integration.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('game-cards')
class GameCardsSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Game Cards/Sports feature.

    Validates:
    - Sports games API endpoints
    - Live game updates
    - Odds display accuracy
    - Kalshi market integration
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='game-cards',
            description='Sports betting cards with live games and odds',
            enable_browser=True,
            enable_database=True,
        )
        self._games_data = None
        self._markets_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all sports-related API endpoints"""
        results = []

        # Test /api/sports/live
        result = await self.test_endpoint(
            method='GET',
            path='/sports/live',
            expected_status=200,
            validate_response=self._validate_games_response,
        )
        results.append(result)

        # Store games data
        if result.passed:
            response = await self.http_get('/sports/live')
            if response.status_code == 200:
                self._games_data = response.json()

        # Test /api/sports/upcoming
        result = await self.test_endpoint(
            method='GET',
            path='/sports/upcoming',
            expected_status=200,
        )
        results.append(result)

        # Test /api/sports/markets (Kalshi integration)
        result = await self.test_endpoint(
            method='GET',
            path='/sports/markets',
            expected_status=200,
            validate_response=self._validate_markets_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_get('/sports/markets')
            if response.status_code == 200:
                self._markets_data = response.json()

        return results

    def _validate_games_response(self, response) -> List[Issue]:
        """Custom validation for games response"""
        issues = []
        data = response.json()

        # Should return array
        if not isinstance(data, list):
            issues.append(Issue(
                title="Games response not an array",
                description=f"Expected array, got {type(data).__name__}",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        # Validate game structure
        for i, game in enumerate(data[:5]):
            required = ['home_team', 'away_team', 'status']
            for field in required:
                if field not in game:
                    issues.append(Issue(
                        title=f"Missing field in game {i}: {field}",
                        description=f"Game data missing required field",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Check for valid status
            valid_statuses = ['scheduled', 'in_progress', 'live', 'final', 'completed', 'postponed']
            status = game.get('status', '').lower()
            if status and status not in valid_statuses:
                issues.append(Issue(
                    title=f"Unknown game status: {status}",
                    description=f"Game {i} has unexpected status",
                    severity=IssueSeverity.LOW,
                    feature=self.feature_name,
                    component="api",
                ))

            # Check live games have scores
            if status in ['in_progress', 'live']:
                if game.get('home_score') is None or game.get('away_score') is None:
                    issues.append(Issue(
                        title=f"Live game missing scores",
                        description=f"{game.get('away_team')} @ {game.get('home_team')} is live but missing scores",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="api",
                    ))

        return issues

    def _validate_markets_response(self, response) -> List[Issue]:
        """Custom validation for markets response"""
        issues = []
        data = response.json()

        markets = data.get('markets', data) if isinstance(data, dict) else data
        if not isinstance(markets, list):
            issues.append(Issue(
                title="Markets response structure invalid",
                description=f"Expected markets array",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))
            return issues

        for i, market in enumerate(markets[:5]):
            # Validate yes/no prices
            yes_price = market.get('yes_price', 0)
            no_price = market.get('no_price', 0)

            # Prices should be between 0 and 1 (or 0 and 100 for cents)
            if yes_price < 0 or no_price < 0:
                issues.append(Issue(
                    title=f"Negative price in market {i}",
                    description=f"yes_price={yes_price}, no_price={no_price}",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="data_quality",
                ))

            # Yes + No should approximately equal 1 (or 100)
            total = yes_price + no_price
            if total > 1.5 and total < 50:  # Allow for some spread
                # Prices are likely in 0-1 format
                if total > 1.1:
                    issues.append(Issue(
                        title=f"Price sum unusual: market {i}",
                        description=f"Yes ({yes_price}) + No ({no_price}) = {total}",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))
            elif total > 50 and total < 150:
                # Prices are likely in 0-100 format
                if abs(total - 100) > 10:
                    issues.append(Issue(
                        title=f"Price sum unusual: market {i}",
                        description=f"Yes ({yes_price}) + No ({no_price}) = {total}",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="data_quality",
                    ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to sports page
            success = await self.navigate_to('/sports')
            if not success:
                success = await self.navigate_to('/games')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to sports page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to sports page",
                        description="Could not load /sports or /games route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test game cards exist
            results.append(await self.test_element_exists(
                "[data-testid='game-card'], .game-card, .sports-card",
                "Game Cards"
            ))

            # Test league filter
            results.append(await self.test_element_exists(
                "[data-testid='league-filter'], select, .league-tabs",
                "League Filter"
            ))

            # Take screenshot
            await self.take_screenshot("sports_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Sports UI Tests",
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
        """Test game data logic"""
        results = []
        issues = []

        if self._games_data:
            # Check live games are actually live (game time in past but not too old)
            from datetime import datetime, timedelta

            for game in self._games_data:
                status = game.get('status', '').lower()
                game_time = game.get('game_time')

                if status in ['in_progress', 'live'] and game_time:
                    try:
                        game_dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
                        now = datetime.now(game_dt.tzinfo) if game_dt.tzinfo else datetime.now()

                        # Live game should have started (game_time in past)
                        if game_dt > now:
                            issues.append(Issue(
                                title=f"'Live' game hasn't started",
                                description=f"{game.get('away_team')} @ {game.get('home_team')} marked live but game time is in future",
                                severity=IssueSeverity.HIGH,
                                feature=self.feature_name,
                                component="business_logic",
                            ))

                        # Live game shouldn't be more than ~5 hours old
                        if (now - game_dt).total_seconds() > 5 * 3600:
                            issues.append(Issue(
                                title=f"Stale 'live' game",
                                description=f"Game started >5 hours ago but still marked live",
                                severity=IssueSeverity.MEDIUM,
                                feature=self.feature_name,
                                component="data_freshness",
                            ))
                    except Exception as e:
                        logger.debug(f"Could not parse game time: {e}")

        results.append(TestResult(
            test_name="Game Logic Validation",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test sports data consistency"""
        results = []
        issues = []

        # Check games are sorted by time
        if self._games_data:
            game_times = []
            for game in self._games_data:
                gt = game.get('game_time')
                if gt:
                    game_times.append(gt)

            if game_times and game_times != sorted(game_times):
                issues.append(Issue(
                    title="Games not sorted by time",
                    description="Games should be sorted chronologically",
                    severity=IssueSeverity.LOW,
                    feature=self.feature_name,
                    component="data_quality",
                ))

        # Check for duplicate games
        if self._games_data:
            game_ids = [g.get('id') for g in self._games_data if g.get('id')]
            if len(game_ids) != len(set(game_ids)):
                issues.append(Issue(
                    title="Duplicate games in response",
                    description="Same game appears multiple times",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="data_quality",
                ))

        results.append(TestResult(
            test_name="Sports Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
