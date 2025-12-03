"""
Enhancement Engine

Proactively improves the codebase by:
1. Applying safe optimizations automatically
2. Modernizing deprecated patterns
3. Improving code consistency
4. Batching API calls for efficiency
5. Optimizing database queries
6. Adding missing type hints
"""

import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Enhancement:
    """Represents a code enhancement."""
    id: str
    category: str
    description: str
    file_path: str
    line_number: int
    original_code: str
    enhanced_code: str
    risk_level: str  # 'safe', 'moderate', 'risky'
    auto_apply: bool
    applied: bool = False
    applied_at: Optional[datetime] = None


@dataclass
class EnhancementResult:
    """Result of enhancement run."""
    total_found: int
    total_applied: int
    total_skipped: int
    enhancements: List[Enhancement] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class EnhancementEngine:
    """
    Proactively enhances the Magnus codebase.

    Safe enhancements are applied automatically.
    Risky enhancements are logged for manual review.
    """

    # Enhancement categories and their handlers
    ENHANCEMENT_CATEGORIES = {
        'deprecated_patterns': {
            'description': 'Replace deprecated code patterns',
            'risk': 'safe',
            'auto_apply': True,
        },
        'type_hints': {
            'description': 'Add missing type hints',
            'risk': 'safe',
            'auto_apply': True,
        },
        'api_batching': {
            'description': 'Batch multiple API calls',
            'risk': 'moderate',
            'auto_apply': False,
        },
        'query_optimization': {
            'description': 'Optimize database queries',
            'risk': 'moderate',
            'auto_apply': False,
        },
        'code_modernization': {
            'description': 'Use modern Python features',
            'risk': 'safe',
            'auto_apply': True,
        },
        'error_handling': {
            'description': 'Improve error handling',
            'risk': 'moderate',
            'auto_apply': False,
        },
        'performance': {
            'description': 'Performance optimizations',
            'risk': 'moderate',
            'auto_apply': False,
        },
    }

    # Safe pattern replacements
    SAFE_REPLACEMENTS = [
        # Deprecated patterns
        (r'@st\.cache\s*\(', '@st.cache_data(', 'deprecated_patterns'),
        (r'\.format\(([^)]+)\)', lambda m: f'f-string', 'code_modernization'),  # Suggest f-strings

        # Modern Python patterns - use negative lookbehind to avoid matching .tolist() / .to_dict()
        (r'(?<![.\w])dict\(\)', '{}', 'code_modernization'),
        (r'(?<![.\w])list\(\)', '[]', 'code_modernization'),
        (r'is None\b', 'is None', 'code_modernization'),
        (r'is not None\b', 'is not None', 'code_modernization'),
        (r'is True\b', 'is True', 'code_modernization'),
        (r'is False\b', 'is False', 'code_modernization'),

        # Remove unnecessary operations
        (r'\.keys\(\)\s*\)', ')', 'performance'),  # dict.keys() in iteration

        # Type hint suggestions (patterns only, actual addition is more complex)
        (r'def (\w+)\(self\):', r'def \1(self) -> None:', 'type_hints'),
    ]

    # Query optimization patterns
    QUERY_PATTERNS = [
        (r'SELECT \* FROM', 'Select specific columns', 'query_optimization'),
        (r'WHERE\s+\w+\s+LIKE\s+\'%', 'Leading wildcard prevents index use', 'query_optimization'),
        (r'ORDER BY\s+\w+\s+(?!ASC|DESC)', 'Add explicit sort direction', 'query_optimization'),
    ]

    def __init__(self, project_root: Path = None, auto_apply_safe: bool = True):
        """
        Initialize enhancement engine.

        Args:
            project_root: Root directory of the project
            auto_apply_safe: Automatically apply safe enhancements
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.auto_apply_safe = auto_apply_safe
        self.enhancements_log_path = Path(__file__).parent / "logs" / "enhancements.jsonl"
        self.enhancements_log_path.parent.mkdir(exist_ok=True)

    def run(self) -> EnhancementResult:
        """
        Run the enhancement engine on the entire codebase.

        Returns:
            EnhancementResult with all findings and applied changes.
        """
        result = EnhancementResult(
            total_found=0,
            total_applied=0,
            total_skipped=0,
        )

        logger.info("Starting enhancement engine...")

        # Scan all Python files
        for py_file in self.project_root.rglob('*.py'):
            if self._should_skip(py_file):
                continue

            try:
                file_enhancements = self._analyze_file(py_file)
                result.total_found += len(file_enhancements)

                for enhancement in file_enhancements:
                    if enhancement.auto_apply and self.auto_apply_safe:
                        success = self._apply_enhancement(enhancement)
                        if success:
                            result.total_applied += 1
                            enhancement.applied = True
                            enhancement.applied_at = datetime.utcnow()
                        else:
                            result.total_skipped += 1
                    else:
                        result.total_skipped += 1

                    result.enhancements.append(enhancement)

            except Exception as e:
                error_msg = f"Error processing {py_file}: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Log results
        self._log_enhancements(result)

        logger.info(
            f"Enhancement complete: {result.total_found} found, "
            f"{result.total_applied} applied, {result.total_skipped} skipped"
        )

        return result

    def _analyze_file(self, file_path: Path) -> List[Enhancement]:
        """Analyze a single file for enhancement opportunities."""
        enhancements = []

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            relative_path = str(file_path.relative_to(self.project_root))
            lines = content.split('\n')

            # Check safe replacements
            for pattern, replacement, category in self.SAFE_REPLACEMENTS:
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    original = match.group()

                    # Calculate replacement
                    if callable(replacement):
                        new_code = replacement(match)
                    else:
                        new_code = re.sub(pattern, replacement, original)

                    # Skip if no actual change
                    if original == new_code:
                        continue

                    cat_config = self.ENHANCEMENT_CATEGORIES.get(category, {})

                    enhancements.append(Enhancement(
                        id=f"{relative_path}:{line_num}:{category}",
                        category=category,
                        description=cat_config.get('description', category),
                        file_path=relative_path,
                        line_number=line_num,
                        original_code=original,
                        enhanced_code=new_code,
                        risk_level=cat_config.get('risk', 'moderate'),
                        auto_apply=cat_config.get('auto_apply', False) and
                                   cat_config.get('risk') == 'safe',
                    ))

            # Check for missing type hints on functions
            enhancements.extend(self._find_missing_type_hints(content, relative_path))

            # Check for query optimization opportunities
            enhancements.extend(self._find_query_issues(content, relative_path))

            # Check for API batching opportunities
            enhancements.extend(self._find_api_batching_opportunities(content, relative_path))

        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")

        return enhancements

    def _find_missing_type_hints(self, content: str, file_path: str) -> List[Enhancement]:
        """Find functions missing return type hints."""
        enhancements = []

        # Match function definitions without return type
        pattern = r'^(\s*)(def\s+(\w+)\s*\([^)]*\))\s*:'
        matches = list(re.finditer(pattern, content, re.MULTILINE))

        for match in matches:
            indent = match.group(1)
            func_def = match.group(2)
            func_name = match.group(3)

            # Skip if already has return type
            if '->' in func_def:
                continue

            # Skip dunder methods (they have known return types)
            if func_name.startswith('__') and func_name.endswith('__'):
                continue

            line_num = content[:match.start()].count('\n') + 1

            # Suggest None return type for simple cases
            suggested = f"{func_def} -> None:"

            enhancements.append(Enhancement(
                id=f"{file_path}:{line_num}:type_hints",
                category='type_hints',
                description=f"Add return type hint to '{func_name}'",
                file_path=file_path,
                line_number=line_num,
                original_code=f"{func_def}:",
                enhanced_code=suggested,
                risk_level='safe',
                auto_apply=False,  # Type hints need review
            ))

        return enhancements[:10]  # Limit per file

    def _find_query_issues(self, content: str, file_path: str) -> List[Enhancement]:
        """Find database query optimization opportunities."""
        enhancements = []

        for pattern, description, category in self.QUERY_PATTERNS:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1

                enhancements.append(Enhancement(
                    id=f"{file_path}:{line_num}:{category}",
                    category=category,
                    description=description,
                    file_path=file_path,
                    line_number=line_num,
                    original_code=match.group()[:50],
                    enhanced_code=f"[Manual optimization needed: {description}]",
                    risk_level='moderate',
                    auto_apply=False,
                ))

        return enhancements

    def _find_api_batching_opportunities(self, content: str, file_path: str) -> List[Enhancement]:
        """Find opportunities to batch API calls."""
        enhancements = []

        # Find files with multiple sequential API calls
        api_call_pattern = r'requests\.(get|post|put|delete)\s*\('
        matches = list(re.finditer(api_call_pattern, content))

        if len(matches) >= 3:
            # Check if calls are in sequence (within 10 lines of each other)
            lines_with_calls = []
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                lines_with_calls.append(line_num)

            # Find clusters of API calls
            clusters = self._find_clusters(lines_with_calls, max_gap=10)

            for cluster in clusters:
                if len(cluster) >= 3:
                    enhancements.append(Enhancement(
                        id=f"{file_path}:{cluster[0]}:api_batching",
                        category='api_batching',
                        description=f"Consider batching {len(cluster)} sequential API calls",
                        file_path=file_path,
                        line_number=cluster[0],
                        original_code=f"[{len(cluster)} API calls on lines {cluster[0]}-{cluster[-1]}]",
                        enhanced_code="[Use asyncio.gather() or batch endpoint]",
                        risk_level='moderate',
                        auto_apply=False,
                    ))

        return enhancements

    def _find_clusters(self, lines: List[int], max_gap: int) -> List[List[int]]:
        """Find clusters of line numbers within max_gap of each other."""
        if not lines:
            return []

        clusters = []
        current_cluster = [lines[0]]

        for line in lines[1:]:
            if line - current_cluster[-1] <= max_gap:
                current_cluster.append(line)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [line]

        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        return clusters

    def _apply_enhancement(self, enhancement: Enhancement) -> bool:
        """Apply a single enhancement to the file."""
        try:
            file_path = self.project_root / enhancement.file_path

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return False

            content = file_path.read_text(encoding='utf-8')

            # Simple replacement
            new_content = content.replace(
                enhancement.original_code,
                enhancement.enhanced_code,
                1  # Only replace first occurrence
            )

            if new_content == content:
                logger.warning(f"No change made for enhancement: {enhancement.id}")
                return False

            # Write back
            file_path.write_text(new_content, encoding='utf-8')
            logger.info(f"Applied enhancement: {enhancement.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply enhancement {enhancement.id}: {e}")
            return False

    def _log_enhancements(self, result: EnhancementResult):
        """Log enhancements to JSONL file."""
        try:
            with open(self.enhancements_log_path, 'a', encoding='utf-8') as f:
                for enhancement in result.enhancements:
                    log_entry = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'id': enhancement.id,
                        'category': enhancement.category,
                        'description': enhancement.description,
                        'file_path': enhancement.file_path,
                        'line_number': enhancement.line_number,
                        'risk_level': enhancement.risk_level,
                        'applied': enhancement.applied,
                    }
                    f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log enhancements: {e}")

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_parts = [
            'venv', '.venv', 'node_modules', '__pycache__',
            'tests', 'test_', '.git', 'dist', 'build',
            'migrations', 'alembic',
        ]
        path_str = str(file_path)
        return any(p in path_str for p in skip_parts)

    def invoke_claude_for_enhancement(self, enhancement: Enhancement) -> Optional[str]:
        """
        Use Claude to generate a proper enhancement for complex cases.

        This is used for moderate/risky enhancements that need AI review.
        """
        try:
            prompt = f"""
            Review this code and suggest an enhancement.

            Category: {enhancement.category}
            Issue: {enhancement.description}
            File: {enhancement.file_path}
            Line: {enhancement.line_number}

            Original code:
            ```
            {enhancement.original_code}
            ```

            Provide the enhanced code only, no explanation.
            """

            result = subprocess.run(
                ["claude", "--dangerously-skip-permissions", "--print", prompt],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return result.stdout.strip()

        except Exception as e:
            logger.error(f"Claude enhancement failed: {e}")

        return None

    def get_enhancement_summary(self, result: EnhancementResult) -> Dict[str, Any]:
        """Get a summary of enhancements by category."""
        summary = {
            'total_found': result.total_found,
            'total_applied': result.total_applied,
            'total_skipped': result.total_skipped,
            'by_category': {},
            'by_risk': {'safe': 0, 'moderate': 0, 'risky': 0},
        }

        for enhancement in result.enhancements:
            # By category
            cat = enhancement.category
            if cat not in summary['by_category']:
                summary['by_category'][cat] = {'found': 0, 'applied': 0}
            summary['by_category'][cat]['found'] += 1
            if enhancement.applied:
                summary['by_category'][cat]['applied'] += 1

            # By risk
            risk = enhancement.risk_level
            if risk in summary['by_risk']:
                summary['by_risk'][risk] += 1

        return summary
