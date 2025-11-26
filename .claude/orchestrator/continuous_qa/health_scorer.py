"""
Health Scorer for Magnus QA System

Calculates and tracks health scores across 6 dimensions:
1. Code Quality (25%)
2. API Health (20%)
3. Performance (15%)
4. Security (15%)
5. AVA Agents (15%)
6. Modernization (10%)
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class HealthDimension:
    """Single health dimension score."""
    name: str
    score: float  # 0-100
    weight: float  # 0-1
    details: Dict[str, Any]


@dataclass
class HealthScore:
    """Complete health score with all dimensions."""
    timestamp: str
    overall_score: float
    dimensions: Dict[str, HealthDimension]
    trend: str  # improving, stable, declining

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'overall_score': self.overall_score,
            'dimensions': {
                name: {
                    'score': dim.score,
                    'weight': dim.weight,
                    'details': dim.details,
                }
                for name, dim in self.dimensions.items()
            },
            'trend': self.trend,
        }


class HealthScorer:
    """
    Calculates multi-dimensional health scores for Magnus.

    Dimensions:
    - code_quality (25%): Rule violations, dead code, DRY compliance
    - api_health (20%): API endpoint availability, response times
    - performance (15%): Page load times, query performance
    - security (15%): Vulnerability count, exposed secrets
    - ava_agents (15%): Agent health, routing success
    - modernization (10%): Deprecated patterns, outdated dependencies
    """

    DIMENSION_WEIGHTS = {
        'code_quality': 0.25,
        'api_health': 0.20,
        'performance': 0.15,
        'security': 0.15,
        'ava_agents': 0.15,
        'modernization': 0.10,
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the health scorer."""
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.data_dir / "health_history.jsonl"

    def calculate_score(self, check_results: Dict[str, Any]) -> HealthScore:
        """
        Calculate health score from check results.

        Args:
            check_results: Dict with results from various check modules

        Returns:
            HealthScore with all dimensions
        """
        dimensions = {}

        # Code Quality (25%)
        dimensions['code_quality'] = self._score_code_quality(
            check_results.get('code_quality', {})
        )

        # API Health (20%)
        dimensions['api_health'] = self._score_api_health(
            check_results.get('api_health', {}),
            check_results.get('api_endpoints', {}),
        )

        # Performance (15%)
        dimensions['performance'] = self._score_performance(
            check_results.get('performance', {})
        )

        # Security (15%)
        dimensions['security'] = self._score_security(
            check_results.get('security', {})
        )

        # AVA Agents (15%)
        dimensions['ava_agents'] = self._score_ava_agents(
            check_results.get('ava_agents', {})
        )

        # Modernization (10%)
        dimensions['modernization'] = self._score_modernization(
            check_results.get('modernization', {})
        )

        # Calculate weighted overall score
        overall = sum(
            dim.score * dim.weight
            for dim in dimensions.values()
        )

        # Determine trend
        trend = self._calculate_trend()

        health = HealthScore(
            timestamp=datetime.now().isoformat(),
            overall_score=round(overall, 1),
            dimensions=dimensions,
            trend=trend,
        )

        # Save to history
        self._append_to_history(health)

        return health

    def _score_code_quality(self, results: Dict) -> HealthDimension:
        """Calculate code quality dimension score."""
        score = 100.0

        violations = results.get('violations', 0)
        dead_code_files = results.get('dead_code_files', 0)
        dry_violations = results.get('dry_violations', 0)

        # Deduct for violations
        score -= min(violations * 2, 30)  # Max -30 for violations
        score -= min(dead_code_files * 1, 20)  # Max -20 for dead code
        score -= min(dry_violations * 3, 30)  # Max -30 for DRY

        return HealthDimension(
            name='code_quality',
            score=max(0, score),
            weight=self.DIMENSION_WEIGHTS['code_quality'],
            details={
                'violations': violations,
                'dead_code_files': dead_code_files,
                'dry_violations': dry_violations,
            },
        )

    def _score_api_health(self, connectivity: Dict, endpoints: Dict) -> HealthDimension:
        """Calculate API health dimension score."""
        score = 100.0

        # API connectivity (Robinhood, Kalshi, etc.)
        apis_down = connectivity.get('apis_down', 0)
        apis_slow = connectivity.get('apis_slow', 0)

        # Endpoint health
        endpoints_failing = endpoints.get('failing', 0)
        endpoints_returning_mock = endpoints.get('returning_mock', 0)

        score -= apis_down * 20  # -20 per down API
        score -= apis_slow * 5  # -5 per slow API
        score -= endpoints_failing * 5  # -5 per failing endpoint
        score -= endpoints_returning_mock * 10  # -10 per mock data endpoint

        return HealthDimension(
            name='api_health',
            score=max(0, score),
            weight=self.DIMENSION_WEIGHTS['api_health'],
            details={
                'apis_down': apis_down,
                'apis_slow': apis_slow,
                'endpoints_failing': endpoints_failing,
                'endpoints_returning_mock': endpoints_returning_mock,
            },
        )

    def _score_performance(self, results: Dict) -> HealthDimension:
        """Calculate performance dimension score."""
        score = 100.0

        slow_queries = results.get('slow_queries', 0)
        slow_endpoints = results.get('slow_endpoints', 0)
        memory_issues = results.get('memory_issues', 0)
        cache_miss_rate = results.get('cache_miss_rate', 0)

        score -= slow_queries * 3
        score -= slow_endpoints * 5
        score -= memory_issues * 10
        score -= cache_miss_rate * 20  # High miss rate is bad

        return HealthDimension(
            name='performance',
            score=max(0, score),
            weight=self.DIMENSION_WEIGHTS['performance'],
            details={
                'slow_queries': slow_queries,
                'slow_endpoints': slow_endpoints,
                'memory_issues': memory_issues,
                'cache_miss_rate': cache_miss_rate,
            },
        )

    def _score_security(self, results: Dict) -> HealthDimension:
        """Calculate security dimension score."""
        score = 100.0

        vulnerabilities = results.get('vulnerabilities', 0)
        exposed_secrets = results.get('exposed_secrets', 0)
        sql_injection_risks = results.get('sql_injection_risks', 0)

        # Security issues are severe
        score -= vulnerabilities * 15
        score -= exposed_secrets * 50  # Critical
        score -= sql_injection_risks * 25

        return HealthDimension(
            name='security',
            score=max(0, score),
            weight=self.DIMENSION_WEIGHTS['security'],
            details={
                'vulnerabilities': vulnerabilities,
                'exposed_secrets': exposed_secrets,
                'sql_injection_risks': sql_injection_risks,
            },
        )

    def _score_ava_agents(self, results: Dict) -> HealthDimension:
        """Calculate AVA agents dimension score."""
        score = 100.0

        agents_failing = results.get('agents_failing', 0)
        routing_errors = results.get('routing_errors', 0)
        memory_issues = results.get('memory_issues', 0)
        expected_agents = results.get('expected_agents', 105)
        actual_agents = results.get('actual_agents', expected_agents)

        # Missing agents
        missing = expected_agents - actual_agents
        score -= missing * 5

        score -= agents_failing * 10
        score -= routing_errors * 5
        score -= memory_issues * 10

        return HealthDimension(
            name='ava_agents',
            score=max(0, score),
            weight=self.DIMENSION_WEIGHTS['ava_agents'],
            details={
                'expected_agents': expected_agents,
                'actual_agents': actual_agents,
                'agents_failing': agents_failing,
                'routing_errors': routing_errors,
            },
        )

    def _score_modernization(self, results: Dict) -> HealthDimension:
        """Calculate modernization dimension score."""
        score = 100.0

        deprecated_patterns = results.get('deprecated_patterns', 0)
        outdated_deps = results.get('outdated_dependencies', 0)
        old_syntax = results.get('old_syntax_patterns', 0)

        score -= deprecated_patterns * 5
        score -= min(outdated_deps, 10) * 3  # Cap at 10 deps
        score -= old_syntax * 2

        return HealthDimension(
            name='modernization',
            score=max(0, score),
            weight=self.DIMENSION_WEIGHTS['modernization'],
            details={
                'deprecated_patterns': deprecated_patterns,
                'outdated_dependencies': outdated_deps,
                'old_syntax_patterns': old_syntax,
            },
        )

    def _calculate_trend(self) -> str:
        """Calculate health trend based on recent history."""
        recent = self.get_recent_scores(hours=24)

        if len(recent) < 2:
            return 'stable'

        # Compare first and last scores
        first_score = recent[0].get('overall_score', 80)
        last_score = recent[-1].get('overall_score', 80)

        diff = last_score - first_score

        if diff > 5:
            return 'improving'
        elif diff < -5:
            return 'declining'
        else:
            return 'stable'

    def _append_to_history(self, health: HealthScore):
        """Append health score to history (JSONL format)."""
        try:
            with open(self.history_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(health.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to append health score: {e}")

    def get_recent_scores(self, hours: int = 24) -> List[Dict]:
        """Get recent health scores from history."""
        if not self.history_path.exists():
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        results = []

        with open(self.history_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if timestamp >= cutoff:
                        results.append(data)
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

        return results

    def get_latest_score(self) -> Optional[Dict]:
        """Get the most recent health score."""
        recent = self.get_recent_scores(hours=168)  # 1 week
        if recent:
            return recent[-1]
        return None

    def get_dimension_trend(self, dimension: str, hours: int = 24) -> Dict:
        """Get trend for a specific dimension."""
        recent = self.get_recent_scores(hours)

        if len(recent) < 2:
            return {'trend': 'stable', 'change': 0}

        first = recent[0].get('dimensions', {}).get(dimension, {}).get('score', 80)
        last = recent[-1].get('dimensions', {}).get(dimension, {}).get('score', 80)

        change = last - first

        if change > 5:
            trend = 'improving'
        elif change < -5:
            trend = 'declining'
        else:
            trend = 'stable'

        return {'trend': trend, 'change': round(change, 1)}


# Singleton instance
_scorer_instance: Optional[HealthScorer] = None


def get_health_scorer() -> HealthScorer:
    """Get the singleton health scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = HealthScorer()
    return _scorer_instance
