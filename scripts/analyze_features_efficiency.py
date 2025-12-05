#!/usr/bin/env python3
"""
Analyze Feature Efficiency Ratings

Populates the ava_spec_efficiency_ratings table with calculated scores
based on source file analysis (code completeness, documentation, etc.)
"""

import os
import sys
import asyncio
import asyncpg
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class EfficiencyAnalyzer:
    """Analyzes features and calculates efficiency ratings"""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def analyze_all_features(self) -> None:
        """Analyze all features and populate ratings"""
        async with self.pool.acquire() as conn:
            # Get all features with their source files
            features = await conn.fetch("""
                SELECT
                    fs.id,
                    fs.feature_name,
                    fs.category,
                    fs.purpose,
                    fs.description,
                    COALESCE(
                        (SELECT array_agg(sf.file_path)
                         FROM ava_spec_source_files sf
                         WHERE sf.spec_id = fs.id),
                        ARRAY[]::text[]
                    ) as source_files,
                    (SELECT COUNT(*) FROM ava_spec_api_endpoints e
                     WHERE e.spec_id = fs.id) as endpoint_count,
                    (SELECT COUNT(*) FROM ava_spec_integrations i
                     WHERE i.spec_id = fs.id) as integration_count
                FROM ava_feature_specs fs
                WHERE fs.is_current = TRUE
                ORDER BY fs.id
            """)

            print(f"Analyzing {len(features)} features...")

            for i, feature in enumerate(features):
                ratings = await self.analyze_feature(feature)
                await self.save_rating(conn, feature['id'], ratings)

                if (i + 1) % 20 == 0:
                    print(f"  Analyzed {i + 1}/{len(features)} features")

            print(f"Completed efficiency analysis for all {len(features)} features")

    async def analyze_feature(self, feature: dict) -> dict:
        """Analyze a single feature and return ratings"""
        source_files = feature['source_files'] or []
        purpose = feature['purpose'] or ''
        description = feature['description'] or ''
        endpoint_count = feature['endpoint_count'] or 0
        integration_count = feature['integration_count'] or 0
        category = feature['category']

        # Calculate code completeness based on file existence and content
        code_completeness = await self._calculate_code_completeness(
            source_files, endpoint_count
        )

        # Calculate test coverage estimate based on file patterns
        test_coverage = await self._calculate_test_coverage(source_files)

        # Calculate performance estimate
        performance = await self._calculate_performance(
            source_files, category
        )

        # Calculate error handling estimate
        error_handling = await self._calculate_error_handling(source_files)

        # Calculate documentation quality
        documentation_quality = self._calculate_documentation(
            purpose, description
        )

        # Calculate maintainability
        maintainability = self._calculate_maintainability(
            source_files, category
        )

        # Calculate dependency health
        dependency_health = self._calculate_dependency_health(
            integration_count, category
        )

        # Calculate overall rating
        overall = (
            code_completeness * 0.2 +
            test_coverage * 0.15 +
            performance * 0.15 +
            error_handling * 0.15 +
            documentation_quality * 0.1 +
            maintainability * 0.15 +
            dependency_health * 0.1
        )

        return {
            'code_completeness': round(code_completeness, 2),
            'test_coverage': round(test_coverage, 2),
            'performance': round(performance, 2),
            'error_handling': round(error_handling, 2),
            'documentation_quality': round(documentation_quality, 2),
            'maintainability': round(maintainability, 2),
            'dependency_health': round(dependency_health, 2),
            'overall_rating': round(overall, 2)
        }

    async def _calculate_code_completeness(
        self, source_files: list, endpoint_count: int
    ) -> float:
        """Estimate code completeness based on file existence"""
        if not source_files:
            return 5.0

        existing_files = 0
        total_lines = 0

        for file_path in source_files:
            full_path = Path('/Users/adam/code/AVA') / file_path
            if full_path.exists():
                existing_files += 1
                try:
                    content = full_path.read_text()
                    total_lines += len(content.splitlines())
                except Exception:
                    pass

        if existing_files == 0:
            return 5.0

        # Score based on file existence and size
        file_ratio = existing_files / len(source_files)
        size_score = min(10, 5 + (total_lines / 200))
        endpoint_bonus = min(1.5, endpoint_count * 0.1)

        return min(10, (file_ratio * 5) + (size_score * 0.4) + endpoint_bonus)

    async def _calculate_test_coverage(self, source_files: list) -> float:
        """Estimate test coverage based on test file presence"""
        if not source_files:
            return 5.0

        has_tests = any(
            'test' in f.lower() or 'spec' in f.lower()
            for f in source_files
        )

        # Check for test files in tests directory
        test_count = 0
        for file_path in source_files:
            base_name = Path(file_path).stem
            test_patterns = [
                f'tests/test_{base_name}.py',
                f'tests/{base_name}_test.py',
                f'frontend/src/**/*.test.tsx',
            ]
            for pattern in test_patterns:
                test_path = Path('/Users/adam/code/AVA') / pattern.split('*')[0]
                if test_path.exists():
                    test_count += 1

        base_score = 6.0 if has_tests else 4.0
        return min(10, base_score + (test_count * 0.5))

    async def _calculate_performance(
        self, source_files: list, category: str
    ) -> float:
        """Estimate performance based on code patterns"""
        base_score = 7.0

        # Category-based adjustments
        if category in ('agents_trading', 'backend_services'):
            base_score = 7.5
        elif category == 'core':
            base_score = 8.0

        # Check for async patterns
        for file_path in source_files:
            full_path = Path('/Users/adam/code/AVA') / file_path
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    if 'async def' in content:
                        base_score = min(10, base_score + 0.5)
                    if 'asyncpg' in content or 'aiohttp' in content:
                        base_score = min(10, base_score + 0.3)
                except Exception:
                    pass

        return base_score

    async def _calculate_error_handling(self, source_files: list) -> float:
        """Estimate error handling quality"""
        if not source_files:
            return 5.0

        error_patterns = 0
        total_files = 0

        for file_path in source_files:
            full_path = Path('/Users/adam/code/AVA') / file_path
            if full_path.exists():
                total_files += 1
                try:
                    content = full_path.read_text()
                    if 'try:' in content or 'except' in content:
                        error_patterns += 1
                    if 'logging' in content or 'logger' in content:
                        error_patterns += 0.5
                    if 'raise' in content:
                        error_patterns += 0.3
                except Exception:
                    pass

        if total_files == 0:
            return 5.0

        ratio = error_patterns / total_files
        return min(10, 5 + (ratio * 3))

    def _calculate_documentation(self, purpose: str, description: str) -> float:
        """Calculate documentation quality"""
        purpose_len = len(purpose or '')
        desc_len = len(description or '')

        if purpose_len < 20 and desc_len < 50:
            return 4.0
        elif purpose_len < 50 and desc_len < 100:
            return 6.0
        elif purpose_len < 100 and desc_len < 200:
            return 7.0
        else:
            return min(10, 7 + (desc_len / 500))

    def _calculate_maintainability(
        self, source_files: list, category: str
    ) -> float:
        """Estimate maintainability"""
        base_score = 7.0

        # Well-structured categories get higher scores
        if category in ('core', 'backend_services'):
            base_score = 7.5

        # More organized file structure = better maintainability
        if source_files:
            unique_dirs = len(set(str(Path(f).parent) for f in source_files))
            if unique_dirs <= 2:
                base_score = min(10, base_score + 0.5)

        return base_score

    def _calculate_dependency_health(
        self, integration_count: int, category: str
    ) -> float:
        """Estimate dependency health"""
        # Base score
        base_score = 7.0

        # Too many integrations might indicate coupling
        if integration_count > 5:
            base_score -= 0.5
        elif integration_count > 0:
            base_score += 0.5

        # Core features should have fewer dependencies
        if category == 'core' and integration_count > 3:
            base_score -= 0.5

        return min(10, max(5, base_score))

    async def save_rating(
        self, conn: asyncpg.Connection, spec_id: int, ratings: dict
    ):
        """Save efficiency rating to database"""
        await conn.execute("""
            INSERT INTO ava_spec_efficiency_ratings (
                spec_id, overall_rating, code_completeness, test_coverage,
                performance, error_handling, documentation,
                maintainability, dependencies, quick_wins
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (spec_id) DO UPDATE SET
                overall_rating = EXCLUDED.overall_rating,
                code_completeness = EXCLUDED.code_completeness,
                test_coverage = EXCLUDED.test_coverage,
                performance = EXCLUDED.performance,
                error_handling = EXCLUDED.error_handling,
                documentation = EXCLUDED.documentation,
                maintainability = EXCLUDED.maintainability,
                dependencies = EXCLUDED.dependencies,
                updated_at = NOW()
        """, spec_id, ratings['overall_rating'], ratings['code_completeness'],
            ratings['test_coverage'], ratings['performance'],
            ratings['error_handling'], ratings['documentation_quality'],
            ratings['maintainability'], ratings['dependency_health'],
            []  # quick_wins to be populated later
        )


async def main():
    print("=" * 60)
    print("AVA Feature Efficiency Analyzer")
    print("=" * 60)

    pool = await asyncpg.create_pool(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres'),
        database=os.getenv('DB_NAME', 'magnus'),
        min_size=2,
        max_size=10
    )

    try:
        analyzer = EfficiencyAnalyzer(pool)
        await analyzer.analyze_all_features()

        # Show summary
        async with pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    AVG(overall_rating) as avg_rating,
                    MIN(overall_rating) as min_rating,
                    MAX(overall_rating) as max_rating
                FROM ava_spec_efficiency_ratings
            """)

            print("\n" + "=" * 60)
            print("Efficiency Analysis Summary")
            print("=" * 60)
            print(f"Total ratings: {stats['total']}")
            print(f"Average rating: {stats['avg_rating']:.2f}")
            print(f"Min rating: {stats['min_rating']:.2f}")
            print(f"Max rating: {stats['max_rating']:.2f}")

            # Show features needing attention (below 7.0)
            low_rated = await conn.fetch("""
                SELECT
                    fs.feature_name,
                    fs.category,
                    er.overall_rating
                FROM ava_spec_efficiency_ratings er
                JOIN ava_feature_specs fs ON er.spec_id = fs.id
                WHERE er.overall_rating < 7.0
                ORDER BY er.overall_rating
                LIMIT 10
            """)

            if low_rated:
                print(f"\nFeatures needing attention ({len(low_rated)} below 7.0):")
                for f in low_rated:
                    print(f"  - {f['feature_name']} ({f['category']}): {f['overall_rating']:.2f}")

    finally:
        await pool.close()

    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    asyncio.run(main())
