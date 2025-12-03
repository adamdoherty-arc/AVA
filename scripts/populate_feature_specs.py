#!/usr/bin/env python3
"""
AVA Feature Specs Population Script
Scans the codebase, analyzes all features, and populates the ava_feature_specs database.

This script:
1. Scans all agent, service, and frontend files
2. Analyzes each feature using the EfficiencyAnalyzer
3. Populates the database with specs, dependencies, and ratings
4. Generates embeddings for semantic search

Run: python scripts/populate_feature_specs.py
"""

import os
import sys
import re
import ast
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Feature Discovery
# =============================================================================

@dataclass
class DiscoveredFeature:
    """A discovered feature in the codebase"""
    name: str
    category: str
    source_files: List[str]
    test_files: List[str]
    purpose: str
    how_it_works: str
    technical_details: Dict[str, Any]
    key_exports: List[str]
    dependencies: List[str]
    api_endpoints: List[Dict[str, str]]
    database_tables: List[str]
    integrations: List[str]


class FeatureDiscovery:
    """Discovers all features in the AVA codebase"""

    # Feature patterns to scan
    FEATURE_PATTERNS = {
        'core': [
            'src/ava/core/*.py'
        ],
        'trading': [
            'src/ava/agents/trading/*.py',
            'src/ava/agents/trading/agents/*.py'
        ],
        'analysis': [
            'src/ava/agents/analysis/*.py'
        ],
        'sports': [
            'src/ava/agents/sports/*.py'
        ],
        'monitoring': [
            'src/ava/agents/monitoring/*.py'
        ],
        'research': [
            'src/ava/agents/research/*.py'
        ],
        'management': [
            'src/ava/agents/management/*.py'
        ],
        'code': [
            'src/ava/agents/code/*.py'
        ],
        'backend': [
            'backend/services/*.py',
            'backend/routers/*.py'
        ],
        'frontend': [
            'frontend/src/pages/*.tsx'
        ],
        'integration': [
            'src/services/*_client.py',
            'src/services/*_service.py'
        ]
    }

    # Known integrations to detect
    KNOWN_INTEGRATIONS = [
        'robinhood', 'robin_stocks', 'espn', 'kalshi', 'telegram', 'discord',
        'openai', 'anthropic', 'ollama', 'langchain', 'redis', 'postgresql'
    ]

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    def discover_all(self) -> List[DiscoveredFeature]:
        """Discover all features in the codebase"""
        all_features = []

        for category, patterns in self.FEATURE_PATTERNS.items():
            logger.info(f"Scanning {category} features...")
            features = self._scan_category(category, patterns)
            all_features.extend(features)
            logger.info(f"  Found {len(features)} features")

        logger.info(f"Total features discovered: {len(all_features)}")
        return all_features

    def _scan_category(
        self,
        category: str,
        patterns: List[str]
    ) -> List[DiscoveredFeature]:
        """Scan a category for features"""
        features = []

        for pattern in patterns:
            # Expand glob pattern
            glob_path = self.project_root / pattern
            parent = glob_path.parent
            file_pattern = glob_path.name

            if parent.exists():
                for file_path in parent.glob(file_pattern):
                    if file_path.name.startswith('__'):
                        continue

                    try:
                        feature = self._analyze_file(file_path, category)
                        if feature:
                            features.append(feature)
                    except Exception as e:
                        logger.warning(f"Error analyzing {file_path}: {e}")

        return features

    def _analyze_file(
        self,
        file_path: Path,
        category: str
    ) -> Optional[DiscoveredFeature]:
        """Analyze a single file to extract feature information"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Parse Python files
            if file_path.suffix == '.py':
                return self._analyze_python_file(file_path, content, category)

            # Parse TypeScript/TSX files
            elif file_path.suffix in ('.ts', '.tsx'):
                return self._analyze_tsx_file(file_path, content, category)

            return None

        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return None

    def _analyze_python_file(
        self,
        file_path: Path,
        content: str,
        category: str
    ) -> Optional[DiscoveredFeature]:
        """Analyze a Python file"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        # Get module docstring
        module_doc = ast.get_docstring(tree) or ""

        # Extract feature name
        name = file_path.stem
        if name in ('__init__', 'base', 'utils'):
            return None

        # Find classes
        classes = []
        key_exports = []
        capabilities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                key_exports.append(node.name)

                # Extract capabilities from metadata
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == 'metadata':
                                # Try to extract capabilities
                                pass

                # Get class docstring
                class_doc = ast.get_docstring(node)
                if class_doc and not module_doc:
                    module_doc = class_doc

            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                key_exports.append(node.name)

        # Extract dependencies (imports)
        dependencies = []
        integrations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
                    for known in self.KNOWN_INTEGRATIONS:
                        if known in alias.name.lower():
                            integrations.append(known)

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
                    for known in self.KNOWN_INTEGRATIONS:
                        if known in node.module.lower():
                            integrations.append(known)

        # Extract API endpoints (for routers)
        api_endpoints = []
        if 'router' in name.lower() or category == 'backend':
            # Find FastAPI decorators
            endpoint_pattern = r'@router\.(get|post|put|delete|patch)\s*\(["\']([^"\']+)'
            for match in re.finditer(endpoint_pattern, content, re.IGNORECASE):
                method = match.group(1).upper()
                path = match.group(2)
                api_endpoints.append({'method': method, 'path': path})

        # Extract database tables
        db_tables = []
        table_pattern = r'(?:FROM|INTO|UPDATE|JOIN)\s+([a-z_]+)'
        for match in re.finditer(table_pattern, content, re.IGNORECASE):
            table = match.group(1)
            if table.startswith('ava_') or table.startswith('telegram_'):
                if table not in db_tables:
                    db_tables.append(table)

        # Find related test files
        test_files = self._find_test_files(file_path)

        # Generate purpose and how_it_works
        purpose = self._extract_purpose(module_doc, name, classes)
        how_it_works = self._extract_how_it_works(module_doc, content, classes)

        return DiscoveredFeature(
            name=name,
            category=category,
            source_files=[str(file_path.relative_to(self.project_root))],
            test_files=test_files,
            purpose=purpose,
            how_it_works=how_it_works,
            technical_details={
                'classes': classes,
                'language': 'python',
                'capabilities': capabilities
            },
            key_exports=key_exports[:10],  # Top 10
            dependencies=list(set(dependencies))[:20],  # Top 20
            api_endpoints=api_endpoints,
            database_tables=db_tables,
            integrations=list(set(integrations))
        )

    def _analyze_tsx_file(
        self,
        file_path: Path,
        content: str,
        category: str
    ) -> Optional[DiscoveredFeature]:
        """Analyze a TSX file"""
        name = file_path.stem

        # Skip common files
        if name in ('index', 'App', 'main'):
            return None

        # Extract component name
        component_pattern = r'(?:export\s+(?:default\s+)?function|const)\s+(\w+)'
        component_match = re.search(component_pattern, content)
        component_name = component_match.group(1) if component_match else name

        # Extract imports to find dependencies
        dependencies = []
        integrations = []

        import_pattern = r'import\s+.*?from\s+["\']([^"\']+)'
        for match in re.finditer(import_pattern, content):
            dep = match.group(1)
            dependencies.append(dep)

            for known in self.KNOWN_INTEGRATIONS:
                if known in dep.lower():
                    integrations.append(known)

        # Extract API calls
        api_endpoints = []
        fetch_pattern = r'fetch\(["\']([^"\']+)'
        for match in re.finditer(fetch_pattern, content):
            api_endpoints.append({'method': 'GET', 'path': match.group(1)})

        use_query_pattern = r'useQuery.*?["\']([^"\']+)'
        for match in re.finditer(use_query_pattern, content):
            api_endpoints.append({'method': 'GET', 'path': match.group(1)})

        # Extract purpose from comments
        comment_pattern = r'/\*\*([^*]+)\*/'
        comments = re.findall(comment_pattern, content)
        purpose = comments[0].strip() if comments else f"{component_name} page component"

        # Generate how_it_works
        how_it_works = f"""
React component that renders the {component_name} page.
Uses TypeScript with React 19 and Vite.
Connects to the backend API for data.
"""

        if 'useMagnusApi' in content:
            how_it_works += "Integrates with Magnus API hooks for data fetching.\n"

        if 'Chart' in content or 'Recharts' in content:
            how_it_works += "Includes data visualization with Recharts.\n"

        return DiscoveredFeature(
            name=name,
            category='frontend',
            source_files=[str(file_path.relative_to(self.project_root))],
            test_files=[],
            purpose=purpose[:500],
            how_it_works=how_it_works.strip()[:1000],
            technical_details={
                'component': component_name,
                'language': 'typescript',
                'framework': 'react'
            },
            key_exports=[component_name],
            dependencies=list(set(dependencies))[:20],
            api_endpoints=api_endpoints,
            database_tables=[],
            integrations=list(set(integrations))
        )

    def _find_test_files(self, source_file: Path) -> List[str]:
        """Find test files for a source file"""
        test_files = []
        name = source_file.stem

        # Common test patterns
        patterns = [
            f"tests/test_{name}.py",
            f"tests/{name}_test.py",
            f"test_{name}.py",
            f"{source_file.parent}/tests/test_{name}.py"
        ]

        for pattern in patterns:
            test_path = self.project_root / pattern
            if test_path.exists():
                test_files.append(str(test_path.relative_to(self.project_root)))

        return test_files

    def _extract_purpose(
        self,
        docstring: str,
        name: str,
        classes: List[str]
    ) -> str:
        """Extract purpose from docstring or generate one"""
        if docstring:
            # Get first paragraph
            lines = docstring.strip().split('\n\n')
            purpose = lines[0].strip()
            if len(purpose) > 20:
                return purpose[:500]

        # Generate purpose from name
        readable_name = name.replace('_', ' ').title()
        if classes:
            return f"{readable_name}: {classes[0]} implementation for AVA platform"
        return f"{readable_name} module for AVA trading platform"

    def _extract_how_it_works(
        self,
        docstring: str,
        content: str,
        classes: List[str]
    ) -> str:
        """Extract how it works description"""
        if docstring:
            # Get second paragraph onward
            parts = docstring.strip().split('\n\n')
            if len(parts) > 1:
                return '\n\n'.join(parts[1:])[:1000]

        # Generate from content analysis
        how_it_works = []

        if classes:
            for cls in classes[:3]:
                how_it_works.append(f"Implements {cls} class for feature functionality.")

        if 'async def' in content:
            how_it_works.append("Uses async/await patterns for non-blocking operations.")

        if 'BaseAgent' in content:
            how_it_works.append("Extends BaseAgent with LangChain tool integration.")

        if '@tool' in content:
            how_it_works.append("Provides LangChain tools for agent capabilities.")

        if 'cache' in content.lower():
            how_it_works.append("Implements caching for performance optimization.")

        return '\n'.join(how_it_works) if how_it_works else "Implements feature functionality."


# =============================================================================
# Database Population
# =============================================================================

class SpecPopulator:
    """Populates the database with feature specs"""

    def __init__(self, db_connection):
        self.db = db_connection

    async def populate_feature(self, feature: DiscoveredFeature) -> int:
        """Insert a feature into the database"""
        try:
            # Map category to enum value
            category_map = {
                'core': 'core',
                'trading': 'agents_trading',
                'analysis': 'agents_analysis',
                'sports': 'agents_sports',
                'monitoring': 'agents_monitoring',
                'research': 'agents_research',
                'management': 'agents_management',
                'code': 'agents_code',
                'backend': 'backend_services',
                'frontend': 'frontend_pages',
                'integration': 'integrations'
            }
            db_category = category_map.get(feature.category, 'core')

            # Insert main spec
            insert_sql = """
                INSERT INTO ava_feature_specs (
                    feature_id, feature_name, category, purpose, description,
                    technical_details, created_at, updated_at
                ) VALUES ($1, $2, $3::spec_category, $4, $5, $6, NOW(), NOW())
                ON CONFLICT (feature_id) DO UPDATE SET
                    feature_name = EXCLUDED.feature_name,
                    category = EXCLUDED.category,
                    purpose = EXCLUDED.purpose,
                    description = EXCLUDED.description,
                    technical_details = EXCLUDED.technical_details,
                    updated_at = NOW()
                RETURNING id
            """

            result = await self.db.fetchrow(
                insert_sql,
                feature.name,
                feature.name.replace('_', ' ').title(),
                db_category,
                feature.purpose,
                feature.how_it_works,
                json.dumps(feature.technical_details)
            )

            spec_id = result['id']

            # Insert related data sequentially (asyncpg requires it)
            await self._insert_source_files(spec_id, feature)
            await self._insert_api_endpoints(spec_id, feature)
            await self._insert_database_tables(spec_id, feature)
            await self._insert_integrations(spec_id, feature)

            return spec_id

        except Exception as e:
            logger.error(f"Error populating feature {feature.name}: {e}")
            raise

    async def _insert_source_files(
        self,
        spec_id: int,
        feature: DiscoveredFeature
    ):
        """Insert source files for a feature"""
        for file_path in feature.source_files:
            # Determine file type from extension
            if file_path.endswith('.py'):
                file_type = 'python'
            elif file_path.endswith('.tsx'):
                file_type = 'typescript_react'
            elif file_path.endswith('.ts'):
                file_type = 'typescript'
            elif file_path.endswith('.sql'):
                file_type = 'sql'
            else:
                file_type = 'other'

            await self.db.execute("""
                INSERT INTO ava_spec_source_files (
                    spec_id, file_path, file_type, key_exports
                ) VALUES ($1, $2, $3, $4)
                ON CONFLICT (spec_id, file_path) DO UPDATE SET
                    key_exports = EXCLUDED.key_exports
            """, spec_id, file_path, file_type, feature.key_exports)

    async def _insert_api_endpoints(
        self,
        spec_id: int,
        feature: DiscoveredFeature
    ):
        """Insert API endpoints for a feature"""
        for endpoint in feature.api_endpoints:
            await self.db.execute("""
                INSERT INTO ava_spec_api_endpoints (
                    spec_id, method, path
                ) VALUES ($1, $2, $3)
                ON CONFLICT (spec_id, method, path) DO NOTHING
            """, spec_id, endpoint['method'], endpoint['path'])

    async def _insert_database_tables(
        self,
        spec_id: int,
        feature: DiscoveredFeature
    ):
        """Insert database tables for a feature"""
        for table in feature.database_tables:
            await self.db.execute("""
                INSERT INTO ava_spec_database_tables (
                    spec_id, table_name, usage_type
                ) VALUES ($1, $2, 'read_write')
                ON CONFLICT (spec_id, table_name) DO NOTHING
            """, spec_id, table)

    async def _insert_integrations(
        self,
        spec_id: int,
        feature: DiscoveredFeature
    ):
        """Insert integrations for a feature"""
        for integration in feature.integrations:
            await self.db.execute("""
                INSERT INTO ava_spec_integrations (
                    spec_id, integration_name, integration_type, is_critical
                ) VALUES ($1, $2, 'external_api', false)
                ON CONFLICT (spec_id, integration_name) DO NOTHING
            """, spec_id, integration)


# =============================================================================
# Efficiency Analysis
# =============================================================================

async def run_efficiency_analysis(
    features: List[DiscoveredFeature],
    db_connection
) -> Dict[str, Any]:
    """Run efficiency analysis on all features"""
    from src.ava.analysis.efficiency_analyzer import EfficiencyAnalyzer

    analyzer = EfficiencyAnalyzer(str(PROJECT_ROOT))
    results = []

    for feature in features:
        try:
            score = await analyzer.analyze_feature(
                feature_name=feature.name,
                source_files=feature.source_files,
                category=feature.category,
                test_files=feature.test_files if feature.test_files else None
            )

            # Insert efficiency rating (column names match schema)
            await db_connection.execute("""
                INSERT INTO ava_spec_efficiency_ratings (
                    spec_id, overall_rating, code_completeness,
                    test_coverage, performance, error_handling,
                    documentation, maintainability, dependencies,
                    priority_level, metrics
                )
                SELECT id, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                FROM ava_feature_specs WHERE feature_id = $1
                ON CONFLICT (spec_id) DO UPDATE SET
                    overall_rating = EXCLUDED.overall_rating,
                    code_completeness = EXCLUDED.code_completeness,
                    test_coverage = EXCLUDED.test_coverage,
                    performance = EXCLUDED.performance,
                    error_handling = EXCLUDED.error_handling,
                    documentation = EXCLUDED.documentation,
                    maintainability = EXCLUDED.maintainability,
                    dependencies = EXCLUDED.dependencies,
                    priority_level = EXCLUDED.priority_level,
                    metrics = EXCLUDED.metrics
            """, feature.name, score.overall_score,
                score.code_completeness.score, score.test_coverage.score,
                score.performance.score, score.error_handling.score,
                score.documentation.score, score.maintainability.score,
                score.dependencies.score, score.priority_level,
                json.dumps({'quick_wins': score.quick_wins, 'tech_debt_items': score.tech_debt_items})
            )

            results.append({
                'name': feature.name,
                'score': score.overall_score,
                'priority': score.priority_level
            })

            logger.info(f"  {feature.name}: {score.overall_score}/10 ({score.priority_level})")

        except Exception as e:
            logger.error(f"Error analyzing {feature.name}: {e}")

    return {
        'analyzed': len(results),
        'results': results
    }


# =============================================================================
# Embedding Generation
# =============================================================================

async def generate_embeddings(db_connection) -> int:
    """Generate embeddings for all specs"""
    try:
        import openai

        client = openai.OpenAI()

        # Get all specs without embeddings
        specs = await db_connection.fetch("""
            SELECT id, feature_id, feature_name, purpose, description
            FROM ava_feature_specs
            WHERE embedding IS NULL
        """)

        count = 0
        for spec in specs:
            # Create text to embed
            name = spec['feature_name'] or spec['feature_id']
            purpose = spec['purpose'] or ''
            desc = spec['description'] or ''
            text = f"{name}: {purpose}. {desc}"

            # Generate embedding
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:8000]  # Limit to model max
            )

            embedding = list(response.data[0].embedding)

            # Update in database
            await db_connection.execute("""
                UPDATE ava_feature_specs
                SET embedding = $2
                WHERE id = $1
            """, spec['id'], embedding)

            count += 1
            logger.info(f"  Generated embedding for {name}")

        return count

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return 0


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main execution"""
    print("=" * 80)
    print("AVA Feature Specs Population")
    print("=" * 80)

    # 1. Discover all features
    print("\n1. Discovering features...")
    discovery = FeatureDiscovery(str(PROJECT_ROOT))
    features = discovery.discover_all()

    print(f"\nDiscovered {len(features)} features:")
    categories = {}
    for f in features:
        if f.category not in categories:
            categories[f.category] = 0
        categories[f.category] += 1

    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")

    # 2. Connect to database
    print("\n2. Connecting to database...")
    try:
        import asyncpg
        db = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres'),
            database=os.getenv('DB_NAME', 'magnus')
        )
        print("  Connected successfully!")
    except Exception as e:
        print(f"  Database connection failed: {e}")
        print("  Writing to JSON file instead...")

        # Write to JSON as fallback
        output = {
            'features': [asdict(f) for f in features],
            'categories': categories,
            'generated_at': datetime.now().isoformat()
        }

        output_path = PROJECT_ROOT / 'data' / 'feature_specs.json'
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  Wrote {len(features)} features to {output_path}")
        return

    # 3. Populate database
    print("\n3. Populating database...")
    populator = SpecPopulator(db)

    for feature in features:
        try:
            feature_id = await populator.populate_feature(feature)
            logger.info(f"  Inserted: {feature.name} (id={feature_id})")
        except Exception as e:
            logger.error(f"  Failed: {feature.name} - {e}")

    # 4. Run efficiency analysis
    print("\n4. Running efficiency analysis...")
    analysis_results = await run_efficiency_analysis(features, db)
    print(f"  Analyzed {analysis_results['analyzed']} features")

    # 5. Generate embeddings (if OpenAI key available)
    if os.getenv('OPENAI_API_KEY'):
        print("\n5. Generating embeddings...")
        embedding_count = await generate_embeddings(db)
        print(f"  Generated {embedding_count} embeddings")
    else:
        print("\n5. Skipping embeddings (OPENAI_API_KEY not set)")

    # 6. Summary
    print("\n" + "=" * 80)
    print("Population Complete!")
    print("=" * 80)

    # Get summary stats
    stats = await db.fetchrow("""
        SELECT
            COUNT(*) as total_features,
            COUNT(embedding) as with_embeddings,
            AVG(
                (SELECT overall_rating FROM ava_spec_efficiency_ratings
                 WHERE spec_id = fs.id ORDER BY rated_at DESC LIMIT 1)
            ) as avg_rating
        FROM ava_feature_specs fs
    """)

    print(f"\nTotal Features: {stats['total_features']}")
    print(f"With Embeddings: {stats['with_embeddings']}")
    avg_rating = stats['avg_rating'] or 0
    print(f"Average Rating: {avg_rating:.2f}/10")

    # Show category breakdown
    cat_stats = await db.fetch("""
        SELECT category, COUNT(*) as count
        FROM ava_feature_specs
        GROUP BY category
        ORDER BY count DESC
    """)

    print("\nBy Category:")
    for row in cat_stats:
        print(f"  - {row['category']}: {row['count']}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
