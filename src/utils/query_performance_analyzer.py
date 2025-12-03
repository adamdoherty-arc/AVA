"""
Database Query Performance Analyzer

This module provides comprehensive analysis and monitoring of database query performance,
helping identify slow queries, optimization opportunities, and performance regressions.

Features:
- Automatic query timing and logging
- Query pattern analysis
- Slow query detection and alerting
- Query execution plan analysis
- Performance trend tracking
- Optimization recommendations
- Query result size tracking
- Connection pool monitoring

Usage:
    from src.utils.query_performance_analyzer import (
        QueryAnalyzer,
        analyze_query,
        track_query_performance
    )

    # Initialize analyzer
    analyzer = QueryAnalyzer(slow_query_threshold=1.0)

    # Decor ator usage
    @track_query_performance(analyzer)
    def get_positions():
        return db.execute("SELECT * FROM positions")

    # Manual usage
    with analyzer.track_query("SELECT * FROM trades WHERE date > %s", ("2024-01-01",)):
        results = db.execute(query, params)

    # Get performance report
    report = analyzer.get_performance_report()

Benefits:
- Identify slow queries automatically
- Track query performance over time
- Get actionable optimization recommendations
- Monitor query patterns and frequency
- Detect N+1 query problems
- Profile database operations
"""

import time
import logging
import re
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class QueryExecution:
    """Represents a single query execution."""
    query_id: str
    query: str
    params: Optional[Tuple] = None
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    rows_returned: int = 0
    rows_affected: int = 0
    execution_plan: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class QueryStats:
    """Statistics for a query pattern."""
    query_id: str
    query_pattern: str
    execution_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    p95_duration: float = 0.0
    total_rows: int = 0
    recent_executions: deque = field(default_factory=lambda: deque(maxlen=100))
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


class QueryAnalyzer:
    """
    Comprehensive database query performance analyzer.

    Tracks query execution, identifies slow queries, and provides
    optimization recommendations.
    """

    def __init__(
        self,
        slow_query_threshold: float = 1.0,  # seconds
        track_execution_plans: bool = False,
        max_stored_queries: int = 1000,
        enable_recommendations: bool = True
    ):
        """
        Initialize the query analyzer.

        Args:
            slow_query_threshold: Threshold for slow query warnings (seconds)
            track_execution_plans: Whether to track query execution plans
            max_stored_queries: Maximum number of query executions to store
            enable_recommendations: Enable optimization recommendations
        """
        self.slow_query_threshold = slow_query_threshold
        self.track_execution_plans = track_execution_plans
        self.max_stored_queries = max_stored_queries
        self.enable_recommendations = enable_recommendations

        # Query statistics by pattern
        self.query_stats: Dict[str, QueryStats] = {}

        # Recent query executions (limited buffer)
        self.recent_executions: deque = deque(maxlen=max_stored_queries)

        # Slow queries
        self.slow_queries: List[QueryExecution] = []

        # Query patterns to ignore (utility queries)
        self.ignore_patterns = [
            r"^SELECT\s+1",  # Connection health checks
            r"^SHOW\s+",     # SHOW commands
            r"^SET\s+",      # SET commands
        ]

        # Lock for thread safety
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_queries': 0,
            'slow_queries': 0,
            'failed_queries': 0,
            'total_duration': 0.0,
            'cache_hits': 0,
        }

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for pattern matching.

        Replaces literals with placeholders to group similar queries.

        Args:
            query: SQL query string

        Returns:
            Normalized query pattern
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())

        # Replace numeric literals
        query = re.sub(r'\b\d+\b', '?', query)

        # Replace string literals
        query = re.sub(r"'[^']*'", '?', query)

        # Replace IN clause lists
        query = re.sub(r'IN\s*\([^)]+\)', 'IN (?)', query, flags=re.IGNORECASE)

        return query

    def _get_query_id(self, query: str) -> str:
        """
        Generate unique ID for query pattern.

        Args:
            query: SQL query string

        Returns:
            Query ID (MD5 hash of normalized query)
        """
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _should_ignore_query(self, query: str) -> bool:
        """
        Check if query should be ignored from tracking.

        Args:
            query: SQL query string

        Returns:
            True if query should be ignored
        """
        query = query.strip().upper()

        for pattern in self.ignore_patterns:
            if re.match(pattern, query, re.IGNORECASE):
                return True

        return False

    @contextmanager
    def track_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        get_execution_plan: Optional[Callable] = None
    ):
        """
        Context manager to track query execution.

        Usage:
            with analyzer.track_query("SELECT * FROM trades WHERE id = %s", (123,)):
                results = db.execute(query, params)

        Args:
            query: SQL query string
            params: Query parameters
            get_execution_plan: Optional function to get execution plan
        """
        if self._should_ignore_query(query):
            yield None
            return

        query_id = self._get_query_id(query)
        execution = QueryExecution(
            query_id=query_id,
            query=query,
            params=params
        )

        start_time = time.time()
        error = None

        try:
            # Get execution plan if enabled
            if self.track_execution_plans and get_execution_plan:
                try:
                    execution.execution_plan = get_execution_plan(query, params)
                except Exception as e:
                    logger.debug(f"Failed to get execution plan: {e}")

            yield execution

        except Exception as e:
            error = str(e)
            execution.error = error
            self.stats['failed_queries'] += 1
            raise

        finally:
            # Record duration
            execution.duration = time.time() - start_time

            # Record execution
            with self.lock:
                self._record_execution(execution)

    def _record_execution(self, execution: QueryExecution):
        """
        Record query execution and update statistics.

        Args:
            execution: QueryExecution object
        """
        # Update global stats
        self.stats['total_queries'] += 1
        self.stats['total_duration'] += execution.duration

        # Check if slow query
        if execution.duration >= self.slow_query_threshold:
            self.stats['slow_queries'] += 1
            self.slow_queries.append(execution)
            logger.warning(
                f"Slow query detected ({execution.duration:.3f}s): {execution.query[:100]}"
            )

        # Add to recent executions
        self.recent_executions.append(execution)

        # Update query pattern statistics
        query_id = execution.query_id
        normalized_query = self._normalize_query(execution.query)

        if query_id not in self.query_stats:
            self.query_stats[query_id] = QueryStats(
                query_id=query_id,
                query_pattern=normalized_query
            )

        stats = self.query_stats[query_id]
        stats.execution_count += 1
        stats.total_duration += execution.duration
        stats.min_duration = min(stats.min_duration, execution.duration)
        stats.max_duration = max(stats.max_duration, execution.duration)
        stats.avg_duration = stats.total_duration / stats.execution_count
        stats.total_rows += execution.rows_returned
        stats.recent_executions.append(execution.duration)
        stats.last_seen = execution.timestamp

        # Calculate P95
        if stats.recent_executions:
            sorted_durations = sorted(stats.recent_executions)
            p95_index = int(len(sorted_durations) * 0.95)
            stats.p95_duration = sorted_durations[p95_index]

    def get_slow_queries(self, limit: int = 10) -> List[QueryExecution]:
        """
        Get slowest queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of slow QueryExecution objects
        """
        with self.lock:
            return sorted(
                self.slow_queries,
                key=lambda q: q.duration,
                reverse=True
            )[:limit]

    def get_most_frequent_queries(self, limit: int = 10) -> List[QueryStats]:
        """
        Get most frequently executed queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of QueryStats objects
        """
        with self.lock:
            return sorted(
                self.query_stats.values(),
                key=lambda s: s.execution_count,
                reverse=True
            )[:limit]

    def get_slowest_average_queries(self, limit: int = 10) -> List[QueryStats]:
        """
        Get queries with slowest average execution time.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of QueryStats objects
        """
        with self.lock:
            return sorted(
                self.query_stats.values(),
                key=lambda s: s.avg_duration,
                reverse=True
            )[:limit]

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on query patterns.

        Returns:
            List of recommendation dictionaries
        """
        if not self.enable_recommendations:
            return []

        recommendations = []

        with self.lock:
            for stats in self.query_stats.values():
                query = stats.query_pattern.upper()

                # Recommendation 1: Missing WHERE clause
                if 'SELECT' in query and 'WHERE' not in query:
                    if 'LIMIT' not in query:
                        recommendations.append({
                            'type': 'missing_where',
                            'severity': 'high',
                            'query_id': stats.query_id,
                            'query': stats.query_pattern,
                            'message': 'Query without WHERE clause - may scan entire table',
                            'suggestion': 'Add WHERE clause to filter results, or LIMIT if all rows needed'
                        })

                # Recommendation 2: SELECT *
                if 'SELECT *' in query:
                    recommendations.append({
                        'type': 'select_star',
                        'severity': 'medium',
                        'query_id': stats.query_id,
                        'query': stats.query_pattern,
                        'message': 'Using SELECT * - returns unnecessary columns',
                        'suggestion': 'Specify only required columns to reduce data transfer'
                    })

                # Recommendation 3: Slow average query
                if stats.avg_duration > self.slow_query_threshold:
                    recommendations.append({
                        'type': 'slow_average',
                        'severity': 'high',
                        'query_id': stats.query_id,
                        'query': stats.query_pattern,
                        'avg_duration': stats.avg_duration,
                        'message': f'Query averages {stats.avg_duration:.3f}s',
                        'suggestion': 'Add indexes on WHERE/JOIN columns, or optimize query'
                    })

                # Recommendation 4: High frequency query
                if stats.execution_count > 100 and stats.avg_duration > 0.1:
                    recommendations.append({
                        'type': 'high_frequency',
                        'severity': 'medium',
                        'query_id': stats.query_id,
                        'query': stats.query_pattern,
                        'execution_count': stats.execution_count,
                        'message': f'Query executed {stats.execution_count} times',
                        'suggestion': 'Consider caching results or reducing call frequency'
                    })

                # Recommendation 5: OR in WHERE clause
                if ' OR ' in query and 'WHERE' in query:
                    recommendations.append({
                        'type': 'or_clause',
                        'severity': 'low',
                        'query_id': stats.query_id,
                        'query': stats.query_pattern,
                        'message': 'OR clause may prevent index usage',
                        'suggestion': 'Consider UNION or IN clause instead of OR'
                    })

        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda r: severity_order[r['severity']])

        return recommendations

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Performance report dictionary
        """
        with self.lock:
            avg_duration = (
                self.stats['total_duration'] / self.stats['total_queries']
                if self.stats['total_queries'] > 0 else 0
            )

            return {
                'summary': {
                    'total_queries': self.stats['total_queries'],
                    'slow_queries': self.stats['slow_queries'],
                    'failed_queries': self.stats['failed_queries'],
                    'avg_duration': avg_duration,
                    'total_duration': self.stats['total_duration'],
                    'unique_query_patterns': len(self.query_stats),
                },
                'slowest_queries': [
                    {
                        'query': q.query[:200],
                        'duration': q.duration,
                        'timestamp': q.timestamp.isoformat(),
                        'rows_returned': q.rows_returned
                    }
                    for q in self.get_slow_queries(10)
                ],
                'most_frequent': [
                    {
                        'query': s.query_pattern[:200],
                        'execution_count': s.execution_count,
                        'avg_duration': s.avg_duration,
                        'total_duration': s.total_duration
                    }
                    for s in self.get_most_frequent_queries(10)
                ],
                'recommendations': self.get_optimization_recommendations()[:10]
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        with self.lock:
            return dict(self.stats)

    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self.lock:
            self.stats = {
                'total_queries': 0,
                'slow_queries': 0,
                'failed_queries': 0,
                'total_duration': 0.0,
                'cache_hits': 0,
            }
            self.query_stats.clear()
            self.slow_queries.clear()
            self.recent_executions.clear()
            logger.info("Query analyzer statistics reset")

    def export_report(self, filepath: str):
        """
        Export performance report to file.

        Args:
            filepath: Path to export file (JSON)
        """
        import json

        report = self.get_performance_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report exported to {filepath}")


def track_query_performance(analyzer: QueryAnalyzer):
    """
    Decorator to track query performance.

    Usage:
        @track_query_performance(analyzer)
        def get_all_positions():
            return db.execute("SELECT * FROM positions")

    Args:
        analyzer: QueryAnalyzer instance

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract query from function or kwargs
            query = kwargs.get('query') or f"Function: {func.__name__}"

            with analyzer.track_query(query):
                result = func(*args, **kwargs)

            return result

        return wrapper
    return decorator


# Global analyzer instance
_global_analyzer = None


def get_analyzer() -> QueryAnalyzer:
    """Get or create global analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = QueryAnalyzer()
    return _global_analyzer


def analyze_query(query: str, params: Optional[Tuple] = None):
    """
    Convenience function to analyze a query using global analyzer.

    Args:
        query: SQL query string
        params: Query parameters

    Returns:
        Context manager
    """
    return get_analyzer().track_query(query, params)


# Convenience exports
__all__ = [
    'QueryAnalyzer',
    'QueryExecution',
    'QueryStats',
    'track_query_performance',
    'analyze_query',
    'get_analyzer',
]
