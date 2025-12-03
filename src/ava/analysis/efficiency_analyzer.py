"""
Deep Efficiency Analyzer for AVA Codebase
Scores features across 7 dimensions with weighted scoring

Dimensions:
1. Code Completeness - TODOs, NotImplementedError, pass stubs
2. Test Coverage - Test files, test count, async tests
3. Performance - Async patterns, caching, batch operations
4. Error Handling - try/except, custom errors, logging
5. Documentation - Docstrings, type hints, comments
6. Maintainability - Function length, class size, nesting
7. Dependencies - Import organization, circular deps
"""

import os
import re
import ast
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class DimensionScore:
    """Score for a single efficiency dimension"""
    dimension: str
    score: float  # 0-10
    weight: float  # 0-1
    weighted_score: float
    details: List[str] = field(default_factory=list)
    deductions: List[str] = field(default_factory=list)
    bonuses: List[str] = field(default_factory=list)


@dataclass
class EfficiencyScore:
    """Complete efficiency score for a feature"""
    feature_name: str
    category: str
    overall_score: float  # 0-10 weighted average
    priority_level: str  # 'critical', 'high', 'medium', 'low'

    # Individual dimension scores
    code_completeness: DimensionScore
    test_coverage: DimensionScore
    performance: DimensionScore
    error_handling: DimensionScore
    documentation: DimensionScore
    maintainability: DimensionScore
    dependencies: DimensionScore

    # Quick wins and hotspots
    quick_wins: List[str] = field(default_factory=list)
    tech_debt_items: List[str] = field(default_factory=list)

    # Metadata
    files_analyzed: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    analysis_timestamp: str = ""


@dataclass
class EfficiencyReport:
    """Full efficiency report across all features"""
    total_features: int
    avg_overall_score: float
    category_averages: Dict[str, float]
    critical_features: List[str]
    high_priority_features: List[str]
    quick_wins_summary: List[str]
    tech_debt_hotspots: List[str]
    features: List[EfficiencyScore]
    generated_at: str


# =============================================================================
# Efficiency Analyzer
# =============================================================================

class EfficiencyAnalyzer:
    """
    Deep Efficiency Analyzer for AVA Features

    Analyzes source files across 7 dimensions and provides:
    - Weighted overall score (0-10)
    - Priority classification
    - Quick wins identification
    - Tech debt hotspots
    """

    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        'code_completeness': 0.20,
        'test_coverage': 0.15,
        'performance': 0.20,
        'error_handling': 0.15,
        'documentation': 0.10,
        'maintainability': 0.15,
        'dependencies': 0.05
    }

    # Priority thresholds
    PRIORITY_THRESHOLDS = {
        'critical': 4.0,
        'high': 6.0,
        'medium': 8.0
        # 8.0+ is 'low' priority
    }

    def __init__(self, project_root: str = None):
        """
        Initialize efficiency analyzer

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root or os.getcwd())
        self._cache: Dict[str, Any] = {}

        logger.info(f"EfficiencyAnalyzer initialized for: {self.project_root}")

    # =========================================================================
    # Main Analysis Methods
    # =========================================================================

    async def analyze_feature(
        self,
        feature_name: str,
        source_files: List[str],
        category: str = "unknown",
        test_files: Optional[List[str]] = None
    ) -> EfficiencyScore:
        """
        Analyze a single feature across all dimensions

        Args:
            feature_name: Name of the feature
            source_files: List of source file paths
            category: Feature category
            test_files: Optional list of test files

        Returns:
            EfficiencyScore with complete analysis
        """
        logger.info(f"Analyzing feature: {feature_name}")

        # Read all source files
        source_contents = {}
        total_lines = 0

        for file_path in source_files:
            try:
                full_path = self.project_root / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        source_contents[file_path] = content
                        total_lines += len(content.splitlines())
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")

        # Find related test files if not provided
        if test_files is None:
            test_files = self._find_test_files(source_files)

        # Read test files
        test_contents = {}
        for file_path in test_files:
            try:
                full_path = self.project_root / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        test_contents[file_path] = f.read()
            except Exception as e:
                logger.warning(f"Could not read test file {file_path}: {e}")

        # Analyze each dimension
        dimensions = await asyncio.gather(
            self._analyze_code_completeness(source_contents),
            self._analyze_test_coverage(source_contents, test_contents),
            self._analyze_performance(source_contents),
            self._analyze_error_handling(source_contents),
            self._analyze_documentation(source_contents),
            self._analyze_maintainability(source_contents),
            self._analyze_dependencies(source_contents)
        )

        code_completeness, test_coverage, performance, error_handling, \
            documentation, maintainability, dependencies = dimensions

        # Calculate overall weighted score
        overall_score = (
            code_completeness.weighted_score +
            test_coverage.weighted_score +
            performance.weighted_score +
            error_handling.weighted_score +
            documentation.weighted_score +
            maintainability.weighted_score +
            dependencies.weighted_score
        )

        # Determine priority level
        priority_level = self._determine_priority(overall_score)

        # Identify quick wins
        quick_wins = self._identify_quick_wins(
            code_completeness, test_coverage, performance,
            error_handling, documentation, maintainability, dependencies
        )

        # Identify tech debt
        tech_debt = self._identify_tech_debt(
            code_completeness, test_coverage, performance,
            error_handling, documentation, maintainability, dependencies
        )

        return EfficiencyScore(
            feature_name=feature_name,
            category=category,
            overall_score=round(overall_score, 2),
            priority_level=priority_level,
            code_completeness=code_completeness,
            test_coverage=test_coverage,
            performance=performance,
            error_handling=error_handling,
            documentation=documentation,
            maintainability=maintainability,
            dependencies=dependencies,
            quick_wins=quick_wins,
            tech_debt_items=tech_debt,
            files_analyzed=list(source_contents.keys()),
            lines_of_code=total_lines,
            analysis_timestamp=datetime.now().isoformat()
        )

    async def analyze_all_features(
        self,
        features: List[Dict[str, Any]]
    ) -> EfficiencyReport:
        """
        Analyze all features and generate comprehensive report

        Args:
            features: List of feature dictionaries with:
                - name: Feature name
                - source_files: List of source file paths
                - category: Feature category
                - test_files: Optional test files

        Returns:
            EfficiencyReport with complete analysis
        """
        logger.info(f"Analyzing {len(features)} features")

        # Analyze all features
        tasks = [
            self.analyze_feature(
                feature_name=f['name'],
                source_files=f['source_files'],
                category=f.get('category', 'unknown'),
                test_files=f.get('test_files')
            )
            for f in features
        ]

        scores = await asyncio.gather(*tasks)

        # Calculate averages
        avg_score = sum(s.overall_score for s in scores) / len(scores) if scores else 0

        # Category averages
        category_scores: Dict[str, List[float]] = {}
        for score in scores:
            if score.category not in category_scores:
                category_scores[score.category] = []
            category_scores[score.category].append(score.overall_score)

        category_averages = {
            cat: sum(vals) / len(vals)
            for cat, vals in category_scores.items()
        }

        # Find critical and high priority features
        critical_features = [s.feature_name for s in scores if s.priority_level == 'critical']
        high_priority_features = [s.feature_name for s in scores if s.priority_level == 'high']

        # Aggregate quick wins
        all_quick_wins = []
        for score in scores:
            for qw in score.quick_wins[:2]:  # Top 2 per feature
                all_quick_wins.append(f"[{score.feature_name}] {qw}")

        # Identify tech debt hotspots (categories with avg < 6.0)
        tech_debt_hotspots = [
            f"{cat}: {avg:.1f}/10"
            for cat, avg in category_averages.items()
            if avg < 6.0
        ]

        return EfficiencyReport(
            total_features=len(scores),
            avg_overall_score=round(avg_score, 2),
            category_averages={k: round(v, 2) for k, v in category_averages.items()},
            critical_features=critical_features,
            high_priority_features=high_priority_features,
            quick_wins_summary=all_quick_wins[:20],  # Top 20
            tech_debt_hotspots=tech_debt_hotspots,
            features=scores,
            generated_at=datetime.now().isoformat()
        )

    # =========================================================================
    # Dimension Analyzers
    # =========================================================================

    async def _analyze_code_completeness(
        self,
        source_contents: Dict[str, str]
    ) -> DimensionScore:
        """
        Analyze code completeness

        Checks for:
        - TODO/FIXME comments (-0.5 each)
        - NotImplementedError (-1.0 each)
        - pass statements in non-abstract methods (-0.3 each)
        - Placeholder strings (-0.2 each)
        """
        score = 10.0
        details = []
        deductions = []

        todo_count = 0
        not_impl_count = 0
        pass_count = 0
        placeholder_count = 0

        for file_path, content in source_contents.items():
            # Count TODOs
            todos = re.findall(r'#\s*(TODO|FIXME|XXX|HACK)', content, re.IGNORECASE)
            todo_count += len(todos)

            # Count NotImplementedError
            not_impl = re.findall(r'raise\s+NotImplementedError', content)
            not_impl_count += len(not_impl)

            # Count isolated pass statements (simplified)
            # Look for 'pass' on its own line
            pass_matches = re.findall(r'^\s*pass\s*$', content, re.MULTILINE)
            pass_count += len(pass_matches)

            # Count placeholder strings
            placeholders = re.findall(r'["\'](?:placeholder|todo|implement|coming soon)["\']', content, re.IGNORECASE)
            placeholder_count += len(placeholders)

        # Apply deductions
        if todo_count > 0:
            deduction = min(todo_count * 0.5, 3.0)
            score -= deduction
            deductions.append(f"TODOs found: {todo_count} (-{deduction:.1f})")

        if not_impl_count > 0:
            deduction = min(not_impl_count * 1.0, 3.0)
            score -= deduction
            deductions.append(f"NotImplementedError: {not_impl_count} (-{deduction:.1f})")

        if pass_count > 0:
            deduction = min(pass_count * 0.3, 2.0)
            score -= deduction
            deductions.append(f"Pass statements: {pass_count} (-{deduction:.1f})")

        if placeholder_count > 0:
            deduction = min(placeholder_count * 0.2, 1.0)
            score -= deduction
            deductions.append(f"Placeholders: {placeholder_count} (-{deduction:.1f})")

        score = max(0.0, min(10.0, score))

        if score >= 9.0:
            details.append("Excellent code completeness")
        elif score >= 7.0:
            details.append("Good code completeness with minor gaps")
        elif score >= 5.0:
            details.append("Moderate incompleteness - needs attention")
        else:
            details.append("Significant incomplete code detected")

        weighted_score = score * self.WEIGHTS['code_completeness']

        return DimensionScore(
            dimension='code_completeness',
            score=round(score, 2),
            weight=self.WEIGHTS['code_completeness'],
            weighted_score=round(weighted_score, 3),
            details=details,
            deductions=deductions
        )

    async def _analyze_test_coverage(
        self,
        source_contents: Dict[str, str],
        test_contents: Dict[str, str]
    ) -> DimensionScore:
        """
        Analyze test coverage

        Scoring:
        - Test file exists: +3
        - Each test function: +0.5 (max 4)
        - Async tests present: +1
        - Error path tests: +1
        - Mock usage: +1
        """
        score = 0.0
        details = []
        bonuses = []
        deductions = []

        # Check if test files exist
        if test_contents:
            score += 3.0
            bonuses.append(f"Test files found: {len(test_contents)} (+3.0)")

            # Count test functions
            test_count = 0
            async_tests = 0
            error_tests = 0
            mock_usage = False

            for file_path, content in test_contents.items():
                # Count test functions
                tests = re.findall(r'def\s+(test_\w+|async\s+def\s+test_\w+)', content)
                test_count += len(tests)

                # Check for async tests
                async_matches = re.findall(r'async\s+def\s+test_', content)
                async_tests += len(async_matches)

                # Check for error path tests
                error_patterns = re.findall(r'test_.*(?:error|fail|invalid|exception)', content, re.IGNORECASE)
                error_tests += len(error_patterns)

                # Check for mock usage
                if 'mock' in content.lower() or 'patch' in content.lower():
                    mock_usage = True

            # Apply bonuses
            test_bonus = min(test_count * 0.5, 4.0)
            score += test_bonus
            if test_count > 0:
                bonuses.append(f"Test functions: {test_count} (+{test_bonus:.1f})")

            if async_tests > 0:
                score += 1.0
                bonuses.append(f"Async tests: {async_tests} (+1.0)")

            if error_tests > 0:
                score += 1.0
                bonuses.append(f"Error path tests: {error_tests} (+1.0)")

            if mock_usage:
                score += 1.0
                bonuses.append("Mock/patch usage detected (+1.0)")
        else:
            deductions.append("No test files found (-7.0 potential)")
            details.append("Missing test coverage")

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            details.append("Excellent test coverage")
        elif score >= 5.0:
            details.append("Moderate test coverage")
        elif score >= 3.0:
            details.append("Basic test coverage - needs improvement")
        else:
            details.append("Poor or no test coverage")

        weighted_score = score * self.WEIGHTS['test_coverage']

        return DimensionScore(
            dimension='test_coverage',
            score=round(score, 2),
            weight=self.WEIGHTS['test_coverage'],
            weighted_score=round(weighted_score, 3),
            details=details,
            deductions=deductions,
            bonuses=bonuses
        )

    async def _analyze_performance(
        self,
        source_contents: Dict[str, str]
    ) -> DimensionScore:
        """
        Analyze performance patterns

        Scoring:
        - Async patterns: +2
        - Caching: +2
        - Batch operations: +1
        - Rate limiting: +1
        - Connection pooling: +1
        - N+1 detection: -2 if found
        """
        score = 5.0  # Start at midpoint
        details = []
        bonuses = []
        deductions = []

        combined_content = "\n".join(source_contents.values())

        # Check for async patterns
        if re.search(r'async\s+def\s+\w+', combined_content):
            score += 2.0
            bonuses.append("Async patterns found (+2.0)")

        # Check for caching
        cache_patterns = [
            r'@cache', r'@lru_cache', r'@cached',
            r'cache\s*=', r'_cache\s*=', r'redis',
            r'\.cache\(', r'get_cached', r'set_cache'
        ]
        if any(re.search(p, combined_content, re.IGNORECASE) for p in cache_patterns):
            score += 2.0
            bonuses.append("Caching implementation found (+2.0)")

        # Check for batch operations
        batch_patterns = [r'batch', r'bulk', r'executemany', r'insert_many', r'gather\(']
        if any(re.search(p, combined_content, re.IGNORECASE) for p in batch_patterns):
            score += 1.0
            bonuses.append("Batch operations found (+1.0)")

        # Check for rate limiting
        rate_patterns = [r'rate.?limit', r'throttle', r'backoff', r'retry']
        if any(re.search(p, combined_content, re.IGNORECASE) for p in rate_patterns):
            score += 1.0
            bonuses.append("Rate limiting/retry logic found (+1.0)")

        # Check for connection pooling
        pool_patterns = [r'pool', r'connection.?pool', r'aiohttp.?session']
        if any(re.search(p, combined_content, re.IGNORECASE) for p in pool_patterns):
            score += 1.0
            bonuses.append("Connection pooling found (+1.0)")

        # Check for N+1 patterns (loop with query inside)
        n_plus_1_pattern = r'for\s+\w+\s+in.*:\s*\n\s*.*(?:query|select|fetch|get.*from)'
        if re.search(n_plus_1_pattern, combined_content, re.IGNORECASE | re.MULTILINE):
            score -= 2.0
            deductions.append("Potential N+1 query pattern detected (-2.0)")

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            details.append("Excellent performance patterns")
        elif score >= 6.0:
            details.append("Good performance with room for optimization")
        elif score >= 4.0:
            details.append("Basic performance - needs optimization")
        else:
            details.append("Performance issues detected")

        weighted_score = score * self.WEIGHTS['performance']

        return DimensionScore(
            dimension='performance',
            score=round(score, 2),
            weight=self.WEIGHTS['performance'],
            weighted_score=round(weighted_score, 3),
            details=details,
            deductions=deductions,
            bonuses=bonuses
        )

    async def _analyze_error_handling(
        self,
        source_contents: Dict[str, str]
    ) -> DimensionScore:
        """
        Analyze error handling patterns

        Scoring:
        - try/except present: +2
        - Custom exceptions: +2
        - Logging in except: +2
        - Specific exception types: +1
        - Fallback logic: +1
        - Circuit breakers: +1
        - Bare except: -1
        """
        score = 0.0
        details = []
        bonuses = []
        deductions = []

        combined_content = "\n".join(source_contents.values())

        # Check for try/except
        try_blocks = re.findall(r'try\s*:', combined_content)
        if try_blocks:
            score += 2.0
            bonuses.append(f"Try/except blocks: {len(try_blocks)} (+2.0)")

        # Check for custom exceptions
        custom_exc = re.findall(r'class\s+\w+(?:Error|Exception)\s*\(', combined_content)
        if custom_exc:
            score += 2.0
            bonuses.append(f"Custom exceptions defined: {len(custom_exc)} (+2.0)")

        # Check for logging in exception handlers
        log_in_except = re.findall(r'except.*:\s*\n\s*.*(?:logger|logging)', combined_content, re.MULTILINE)
        if log_in_except:
            score += 2.0
            bonuses.append("Logging in exception handlers (+2.0)")

        # Check for specific exception types
        specific_exc = re.findall(r'except\s+(?!Exception)(?!BaseException)\w+(?:Error|Exception)', combined_content)
        if specific_exc:
            score += 1.0
            bonuses.append(f"Specific exception handling: {len(specific_exc)} (+1.0)")

        # Check for fallback/default logic
        fallback_patterns = [r'fallback', r'default', r'or\s+\[\]', r'or\s+{}', r'\.get\(', r'getattr\(']
        if any(re.search(p, combined_content, re.IGNORECASE) for p in fallback_patterns):
            score += 1.0
            bonuses.append("Fallback/default patterns found (+1.0)")

        # Check for circuit breaker
        circuit_patterns = [r'circuit', r'breaker', r'fail.?fast']
        if any(re.search(p, combined_content, re.IGNORECASE) for p in circuit_patterns):
            score += 1.0
            bonuses.append("Circuit breaker pattern found (+1.0)")

        # Penalize bare except
        bare_except = re.findall(r'except\s*:', combined_content)
        if bare_except:
            deduction = min(len(bare_except) * 0.5, 2.0)
            score -= deduction
            deductions.append(f"Bare except clauses: {len(bare_except)} (-{deduction:.1f})")

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            details.append("Excellent error handling")
        elif score >= 5.0:
            details.append("Good error handling with some gaps")
        elif score >= 3.0:
            details.append("Basic error handling - needs improvement")
        else:
            details.append("Poor error handling")

        weighted_score = score * self.WEIGHTS['error_handling']

        return DimensionScore(
            dimension='error_handling',
            score=round(score, 2),
            weight=self.WEIGHTS['error_handling'],
            weighted_score=round(weighted_score, 3),
            details=details,
            deductions=deductions,
            bonuses=bonuses
        )

    async def _analyze_documentation(
        self,
        source_contents: Dict[str, str]
    ) -> DimensionScore:
        """
        Analyze documentation quality

        Scoring:
        - Module docstrings: +2
        - Class docstrings: +2
        - Method docstrings: +2
        - Type hints: +2
        - Inline comments: +1
        - Example usage: +1
        """
        score = 0.0
        details = []
        bonuses = []
        deductions = []

        module_docs = 0
        class_docs = 0
        method_docs = 0
        type_hints = 0

        for file_path, content in source_contents.items():
            try:
                tree = ast.parse(content)

                # Check module docstring
                if ast.get_docstring(tree):
                    module_docs += 1

                for node in ast.walk(tree):
                    # Check class docstrings
                    if isinstance(node, ast.ClassDef):
                        if ast.get_docstring(node):
                            class_docs += 1

                    # Check function/method docstrings
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if ast.get_docstring(node):
                            method_docs += 1

                        # Check for type hints
                        if node.returns or node.args.args:
                            for arg in node.args.args:
                                if arg.annotation:
                                    type_hints += 1
                            if node.returns:
                                type_hints += 1

            except SyntaxError:
                # Handle non-parseable files
                pass

        # Apply bonuses
        if module_docs > 0:
            score += 2.0
            bonuses.append(f"Module docstrings: {module_docs} (+2.0)")

        if class_docs > 0:
            score += 2.0
            bonuses.append(f"Class docstrings: {class_docs} (+2.0)")

        if method_docs > 0:
            bonus = min(method_docs * 0.3, 2.0)
            score += bonus
            bonuses.append(f"Method docstrings: {method_docs} (+{bonus:.1f})")

        if type_hints > 0:
            bonus = min(type_hints * 0.2, 2.0)
            score += bonus
            bonuses.append(f"Type hints: {type_hints} (+{bonus:.1f})")

        # Check for inline comments
        combined = "\n".join(source_contents.values())
        inline_comments = re.findall(r'#\s*[A-Za-z]', combined)
        if len(inline_comments) > 5:
            score += 1.0
            bonuses.append(f"Inline comments: {len(inline_comments)} (+1.0)")

        # Check for example usage
        if 'example' in combined.lower() or '>>>' in combined:
            score += 1.0
            bonuses.append("Example usage found (+1.0)")

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            details.append("Excellent documentation")
        elif score >= 5.0:
            details.append("Good documentation with some gaps")
        elif score >= 3.0:
            details.append("Basic documentation - needs improvement")
        else:
            details.append("Poor or missing documentation")

        weighted_score = score * self.WEIGHTS['documentation']

        return DimensionScore(
            dimension='documentation',
            score=round(score, 2),
            weight=self.WEIGHTS['documentation'],
            weighted_score=round(weighted_score, 3),
            details=details,
            deductions=deductions,
            bonuses=bonuses
        )

    async def _analyze_maintainability(
        self,
        source_contents: Dict[str, str]
    ) -> DimensionScore:
        """
        Analyze maintainability

        Scoring (starts at 10, deductions):
        - Functions > 50 lines: -0.5 each
        - Classes > 500 lines: -1.0 each
        - Deep nesting (>4): -0.5 each
        - Too many imports (>20): -1.0
        - Single responsibility patterns: +bonuses
        """
        score = 10.0
        details = []
        bonuses = []
        deductions = []

        long_functions = 0
        long_classes = 0
        deep_nesting = 0

        for file_path, content in source_contents.items():
            lines = content.splitlines()

            try:
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    # Check function length
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                            func_lines = node.end_lineno - node.lineno
                            if func_lines > 50:
                                long_functions += 1

                    # Check class size
                    if isinstance(node, ast.ClassDef):
                        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                            class_lines = node.end_lineno - node.lineno
                            if class_lines > 500:
                                long_classes += 1

            except SyntaxError:
                pass

            # Check for deep nesting
            max_indent = 0
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    indent_level = indent // 4  # Assuming 4-space indent
                    max_indent = max(max_indent, indent_level)

            if max_indent > 4:
                deep_nesting += 1

        # Apply deductions
        if long_functions > 0:
            deduction = min(long_functions * 0.5, 2.0)
            score -= deduction
            deductions.append(f"Long functions (>50 lines): {long_functions} (-{deduction:.1f})")

        if long_classes > 0:
            deduction = min(long_classes * 1.0, 2.0)
            score -= deduction
            deductions.append(f"Long classes (>500 lines): {long_classes} (-{deduction:.1f})")

        if deep_nesting > 0:
            deduction = min(deep_nesting * 0.5, 2.0)
            score -= deduction
            deductions.append(f"Deep nesting (>4 levels): {deep_nesting} files (-{deduction:.1f})")

        # Check total imports
        combined = "\n".join(source_contents.values())
        imports = re.findall(r'^(?:from|import)\s+', combined, re.MULTILINE)
        if len(imports) > 30:
            score -= 1.0
            deductions.append(f"Many imports: {len(imports)} (-1.0)")

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            details.append("Excellent maintainability")
        elif score >= 6.0:
            details.append("Good maintainability")
        elif score >= 4.0:
            details.append("Moderate complexity - consider refactoring")
        else:
            details.append("High complexity - refactoring recommended")

        weighted_score = score * self.WEIGHTS['maintainability']

        return DimensionScore(
            dimension='maintainability',
            score=round(score, 2),
            weight=self.WEIGHTS['maintainability'],
            weighted_score=round(weighted_score, 3),
            details=details,
            deductions=deductions,
            bonuses=bonuses
        )

    async def _analyze_dependencies(
        self,
        source_contents: Dict[str, str]
    ) -> DimensionScore:
        """
        Analyze dependency management

        Scoring:
        - Clean imports (organized): +3
        - Type imports separated: +2
        - No circular imports: +3
        - Standard library first: +2
        """
        score = 5.0  # Start at midpoint
        details = []
        bonuses = []
        deductions = []

        combined = "\n".join(source_contents.values())

        # Check for TYPE_CHECKING imports
        if 'TYPE_CHECKING' in combined:
            score += 2.0
            bonuses.append("TYPE_CHECKING used for type imports (+2.0)")

        # Check for organized imports (__future__ first)
        if re.search(r'^from\s+__future__', combined, re.MULTILINE):
            score += 1.0
            bonuses.append("Future imports properly placed (+1.0)")

        # Check for import grouping (blank lines between groups)
        import_sections = re.findall(r'((?:^(?:from|import)\s+.*$\n?)+)', combined, re.MULTILINE)
        if len(import_sections) > 1:
            score += 1.0
            bonuses.append("Import sections organized (+1.0)")

        # Check for relative imports (often cleaner)
        relative_imports = re.findall(r'from\s+\.', combined)
        if relative_imports:
            score += 1.0
            bonuses.append(f"Relative imports used: {len(relative_imports)} (+1.0)")

        # Check for potential circular imports (import in function)
        func_imports = re.findall(r'def\s+\w+.*:\s*\n(?:\s*.*\n)*?\s*(?:from|import)\s+', combined)
        if func_imports:
            score -= 1.0
            deductions.append(f"Possible circular import workarounds: {len(func_imports)} (-1.0)")

        score = max(0.0, min(10.0, score))

        if score >= 8.0:
            details.append("Well-organized dependencies")
        elif score >= 5.0:
            details.append("Acceptable dependency management")
        else:
            details.append("Dependencies need organization")

        weighted_score = score * self.WEIGHTS['dependencies']

        return DimensionScore(
            dimension='dependencies',
            score=round(score, 2),
            weight=self.WEIGHTS['dependencies'],
            weighted_score=round(weighted_score, 3),
            details=details,
            deductions=deductions,
            bonuses=bonuses
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _find_test_files(self, source_files: List[str]) -> List[str]:
        """Find related test files for source files"""
        test_files = []

        for source_file in source_files:
            path = Path(source_file)

            # Try common test file patterns
            patterns = [
                f"tests/test_{path.stem}.py",
                f"tests/{path.stem}_test.py",
                f"test_{path.stem}.py",
                f"{path.parent}/tests/test_{path.stem}.py"
            ]

            for pattern in patterns:
                test_path = self.project_root / pattern
                if test_path.exists():
                    test_files.append(pattern)
                    break

        return test_files

    def _determine_priority(self, score: float) -> str:
        """Determine priority level from score"""
        if score < self.PRIORITY_THRESHOLDS['critical']:
            return 'critical'
        elif score < self.PRIORITY_THRESHOLDS['high']:
            return 'high'
        elif score < self.PRIORITY_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'

    def _identify_quick_wins(self, *dimensions: DimensionScore) -> List[str]:
        """Identify quick wins from dimension scores"""
        quick_wins = []

        for dim in dimensions:
            # Single TODO is a quick win
            for ded in dim.deductions:
                if 'TODO' in ded and ': 1 ' in ded:
                    quick_wins.append(f"Fix single TODO in {dim.dimension}")

                # Missing docstring is easy to fix
                if 'docstring' in ded.lower():
                    quick_wins.append(f"Add missing docstrings")
                    break

        return quick_wins[:5]  # Top 5 quick wins

    def _identify_tech_debt(self, *dimensions: DimensionScore) -> List[str]:
        """Identify tech debt from dimension scores"""
        tech_debt = []

        for dim in dimensions:
            if dim.score < 5.0:
                tech_debt.append(f"{dim.dimension}: {dim.score}/10 - {dim.details[0] if dim.details else 'needs attention'}")

        return tech_debt

    def to_database_record(self, score: EfficiencyScore) -> Dict[str, Any]:
        """Convert EfficiencyScore to database-compatible record"""
        return {
            'feature_name': score.feature_name,
            'category': score.category,
            'overall_rating': score.overall_score,
            'code_completeness': score.code_completeness.score,
            'test_coverage': score.test_coverage.score,
            'performance': score.performance.score,
            'error_handling': score.error_handling.score,
            'documentation_quality': score.documentation.score,
            'maintainability': score.maintainability.score,
            'dependency_health': score.dependencies.score,
            'quick_wins': score.quick_wins,
            'tech_debt_items': score.tech_debt_items,
            'analysis_details': {
                'files_analyzed': score.files_analyzed,
                'lines_of_code': score.lines_of_code,
                'dimensions': {
                    'code_completeness': asdict(score.code_completeness),
                    'test_coverage': asdict(score.test_coverage),
                    'performance': asdict(score.performance),
                    'error_handling': asdict(score.error_handling),
                    'documentation': asdict(score.documentation),
                    'maintainability': asdict(score.maintainability),
                    'dependencies': asdict(score.dependencies)
                }
            },
            'rated_at': score.analysis_timestamp
        }


# =============================================================================
# Convenience Function
# =============================================================================

async def analyze_feature(
    feature_name: str,
    source_files: List[str],
    project_root: str = None,
    category: str = "unknown"
) -> EfficiencyScore:
    """
    Convenience function to analyze a single feature

    Args:
        feature_name: Name of the feature
        source_files: List of source file paths
        project_root: Project root directory
        category: Feature category

    Returns:
        EfficiencyScore with complete analysis
    """
    analyzer = EfficiencyAnalyzer(project_root)
    return await analyzer.analyze_feature(feature_name, source_files, category)


# =============================================================================
# Testing
# =============================================================================

async def test_analyzer():
    """Test the efficiency analyzer"""
    print("Testing Efficiency Analyzer")
    print("=" * 80)

    analyzer = EfficiencyAnalyzer("/Users/adam/code/AVA")

    # Test with portfolio agent
    score = await analyzer.analyze_feature(
        feature_name="portfolio_agent",
        source_files=["src/ava/agents/trading/portfolio_agent.py"],
        category="trading"
    )

    print(f"\nFeature: {score.feature_name}")
    print(f"Category: {score.category}")
    print(f"Overall Score: {score.overall_score}/10")
    print(f"Priority: {score.priority_level}")
    print(f"\nDimension Scores:")

    for dim_name in ['code_completeness', 'test_coverage', 'performance',
                     'error_handling', 'documentation', 'maintainability', 'dependencies']:
        dim = getattr(score, dim_name)
        print(f"  {dim_name}: {dim.score}/10 (weighted: {dim.weighted_score:.3f})")

    print(f"\nQuick Wins: {score.quick_wins}")
    print(f"Tech Debt: {score.tech_debt_items}")
    print(f"\nFiles Analyzed: {len(score.files_analyzed)}")
    print(f"Lines of Code: {score.lines_of_code}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_analyzer())
