"""
Shared Code Analyzer Check Module

Detects duplicate code patterns across the codebase to:
1. Identify code that should be shared/consolidated
2. Find copy-pasted logic that could be refactored
3. Detect similar API call patterns
4. Suggest utility function extraction
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import logging

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)


class SharedCodeAnalyzerCheck(BaseCheck):
    """
    Analyzes codebase for duplicate code patterns.

    HIGH priority - Code duplication leads to maintenance issues.
    """

    # Minimum lines for a block to be considered for duplication
    MIN_BLOCK_LINES = 5

    # Minimum occurrences to flag as duplicate
    MIN_OCCURRENCES = 2

    # Patterns to extract for analysis
    EXTRACT_PATTERNS = {
        'api_calls': r'(requests\.(get|post|put|delete)\s*\([^)]+\))',
        'db_queries': r'(cursor\.execute\s*\([^)]+\)|\.query\s*\([^)]+\))',
        'error_handling': r'(try:\s*\n.*?except.*?:\s*\n.*?)(?=\n\S|$)',
        'list_comprehensions': r'(\[[^\]]+\s+for\s+[^\]]+\])',
        'dict_operations': r'(\{[^}]+:\s*[^}]+\s+for\s+[^}]+\})',
        'decorator_usage': r'(@\w+(?:\([^)]*\))?)',
        'class_methods': r'(def\s+\w+\s*\([^)]*self[^)]*\):.*?)(?=\n\s*def|\n\s*class|\Z)',
    }

    # Skip patterns
    SKIP_PATTERNS = [
        '**/venv/**', '**/.venv/**', '**/node_modules/**',
        '**/__pycache__/**', '**/tests/**', '**/test_*.py',
        '**/*.min.js', '**/dist/**', '**/build/**',
    ]

    def __init__(self, project_root: Path = None, similarity_threshold: float = 0.85):
        """
        Initialize shared code analyzer.

        Args:
            project_root: Root directory of the project
            similarity_threshold: Threshold for considering code similar (0-1)
        """
        super().__init__()
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent.parent
        self.similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        return "shared_code_analyzer"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.HIGH

    def get_checks_list(self) -> List[str]:
        return [
            "exact_duplicates",
            "similar_functions",
            "repeated_api_patterns",
            "duplicated_error_handling",
            "consolidation_opportunities",
        ]

    def run(self) -> ModuleCheckResult:
        """Run shared code analysis."""
        self._start_module()

        # Collect all code blocks
        code_blocks = self._collect_code_blocks()

        # Find exact duplicates
        exact_dupes = self._find_exact_duplicates(code_blocks)
        self._report_duplicates("exact_duplicates", "exact duplicate", exact_dupes)

        # Find similar functions
        similar_funcs = self._find_similar_functions(code_blocks)
        self._report_duplicates("similar_functions", "similar function", similar_funcs)

        # Find repeated API patterns
        api_patterns = self._find_pattern_duplicates('api_calls')
        self._report_duplicates("repeated_api_patterns", "repeated API pattern", api_patterns)

        # Find duplicated error handling
        error_patterns = self._find_pattern_duplicates('error_handling')
        self._report_duplicates("duplicated_error_handling", "duplicated error handling", error_patterns)

        # Find consolidation opportunities
        consolidation = self._find_consolidation_opportunities(code_blocks)
        if not consolidation:
            self._pass(
                "consolidation_opportunities",
                "No major consolidation opportunities found"
            )
        else:
            self._warn(
                "consolidation_opportunities",
                f"Found {len(consolidation)} consolidation opportunities",
                details={'opportunities': consolidation[:10]},
            )

        return self._end_module()

    def _collect_code_blocks(self) -> Dict[str, List[Dict]]:
        """Collect code blocks from all Python files."""
        blocks = {
            'functions': [],
            'classes': [],
            'methods': [],
        }

        for py_file in self.project_root.rglob('*.py'):
            if self._should_skip(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                relative_path = str(py_file.relative_to(self.project_root))

                # Extract functions
                functions = self._extract_functions(content, relative_path)
                blocks['functions'].extend(functions)

                # Extract classes
                classes = self._extract_classes(content, relative_path)
                blocks['classes'].extend(classes)

            except Exception as e:
                logger.warning(f"Could not process {py_file}: {e}")

        return blocks

    def _extract_functions(self, content: str, file_path: str) -> List[Dict]:
        """Extract function definitions from content."""
        functions = []

        # Match function definitions
        pattern = r'^(def\s+(\w+)\s*\([^)]*\):.*?)(?=\n(?:def|class)\s|\Z)'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

        for match in matches:
            func_body = match.group(1)
            func_name = match.group(2)

            # Skip very short functions
            lines = func_body.strip().split('\n')
            if len(lines) < self.MIN_BLOCK_LINES:
                continue

            line_num = content[:match.start()].count('\n') + 1

            functions.append({
                'name': func_name,
                'file': file_path,
                'line': line_num,
                'body': func_body,
                'normalized': self._normalize_code(func_body),
                'hash': self._hash_code(func_body),
                'line_count': len(lines),
            })

        return functions

    def _extract_classes(self, content: str, file_path: str) -> List[Dict]:
        """Extract class definitions from content."""
        classes = []

        # Match class definitions
        pattern = r'^(class\s+(\w+).*?:.*?)(?=\nclass\s|\Z)'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

        for match in matches:
            class_body = match.group(1)
            class_name = match.group(2)

            line_num = content[:match.start()].count('\n') + 1

            classes.append({
                'name': class_name,
                'file': file_path,
                'line': line_num,
                'body': class_body,
                'normalized': self._normalize_code(class_body),
                'hash': self._hash_code(class_body),
            })

        return classes

    def _normalize_code(self, code: str) -> str:
        """
        Normalize code for comparison by:
        - Removing comments
        - Normalizing whitespace
        - Replacing variable names with placeholders
        """
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '""""""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''''''", code, flags=re.DOTALL)

        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)

        # Replace string literals
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)

        # Replace numbers
        code = re.sub(r'\b\d+\b', '0', code)

        return code.strip()

    def _hash_code(self, code: str) -> str:
        """Create a hash for code block."""
        normalized = self._normalize_code(code)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _find_exact_duplicates(self, code_blocks: Dict) -> List[Dict]:
        """Find exact code duplicates using hashes."""
        duplicates = []

        # Group by hash
        by_hash = defaultdict(list)
        for func in code_blocks['functions']:
            by_hash[func['hash']].append(func)

        # Find duplicates
        for hash_val, funcs in by_hash.items():
            if len(funcs) >= self.MIN_OCCURRENCES:
                duplicates.append({
                    'type': 'exact_duplicate',
                    'count': len(funcs),
                    'locations': [
                        {'file': f['file'], 'line': f['line'], 'name': f['name']}
                        for f in funcs
                    ],
                    'line_count': funcs[0]['line_count'],
                })

        return duplicates

    def _find_similar_functions(self, code_blocks: Dict) -> List[Dict]:
        """Find similar (but not identical) functions."""
        similar = []
        functions = code_blocks['functions']

        # Compare each pair
        compared = set()
        for i, func1 in enumerate(functions):
            for j, func2 in enumerate(functions[i + 1:], i + 1):
                pair_key = (func1['hash'], func2['hash'])
                if pair_key in compared:
                    continue
                compared.add(pair_key)

                # Skip if exact match (handled elsewhere)
                if func1['hash'] == func2['hash']:
                    continue

                # Calculate similarity
                similarity = self._calculate_similarity(
                    func1['normalized'],
                    func2['normalized']
                )

                if similarity >= self.similarity_threshold:
                    similar.append({
                        'type': 'similar_functions',
                        'similarity': round(similarity, 2),
                        'locations': [
                            {'file': func1['file'], 'line': func1['line'], 'name': func1['name']},
                            {'file': func2['file'], 'line': func2['line'], 'name': func2['name']},
                        ],
                    })

        return similar[:20]  # Limit results

    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code blocks."""
        # Simple token-based similarity
        tokens1 = set(code1.split())
        tokens2 = set(code2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def _find_pattern_duplicates(self, pattern_name: str) -> List[Dict]:
        """Find duplicates of a specific pattern type."""
        duplicates = []
        pattern = self.EXTRACT_PATTERNS.get(pattern_name)

        if not pattern:
            return duplicates

        # Collect all matches
        matches_by_file = defaultdict(list)

        for py_file in self.project_root.rglob('*.py'):
            if self._should_skip(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                relative_path = str(py_file.relative_to(self.project_root))

                matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    normalized = self._normalize_code(match.group(1))

                    matches_by_file[normalized].append({
                        'file': relative_path,
                        'line': line_num,
                        'code': match.group(1)[:100],
                    })

            except Exception as e:
                logger.warning(f"Could not process {py_file}: {e}")

        # Find duplicates
        for normalized, locations in matches_by_file.items():
            if len(locations) >= self.MIN_OCCURRENCES:
                duplicates.append({
                    'type': pattern_name,
                    'count': len(locations),
                    'pattern': locations[0]['code'][:50],
                    'locations': locations[:5],  # Limit locations
                })

        return duplicates[:10]  # Limit results

    def _find_consolidation_opportunities(self, code_blocks: Dict) -> List[Dict]:
        """Find opportunities for code consolidation."""
        opportunities = []

        # Find functions with similar names that could be consolidated
        name_groups = defaultdict(list)
        for func in code_blocks['functions']:
            # Extract base name (remove prefixes/suffixes like get_, _async, etc.)
            base_name = re.sub(r'^(get_|set_|is_|has_|do_|_)', '', func['name'])
            base_name = re.sub(r'(_async|_sync|_impl)$', '', base_name)
            name_groups[base_name].append(func)

        for base_name, funcs in name_groups.items():
            if len(funcs) >= 3:  # At least 3 similar functions
                # Check if they have similar structure
                if self._have_similar_structure(funcs):
                    opportunities.append({
                        'type': 'similar_named_functions',
                        'base_name': base_name,
                        'count': len(funcs),
                        'suggestion': f'Consider consolidating {len(funcs)} functions with similar names',
                        'locations': [
                            {'file': f['file'], 'name': f['name']}
                            for f in funcs[:5]
                        ],
                    })

        return opportunities

    def _have_similar_structure(self, functions: List[Dict]) -> bool:
        """Check if functions have similar structure."""
        if len(functions) < 2:
            return False

        # Compare line counts
        line_counts = [f['line_count'] for f in functions]
        avg_lines = sum(line_counts) / len(line_counts)

        # If all functions are within 50% of average, consider similar
        return all(abs(lc - avg_lines) < avg_lines * 0.5 for lc in line_counts)

    def _report_duplicates(self, check_name: str, desc: str, duplicates: List[Dict]):
        """Report duplicate findings."""
        if not duplicates:
            self._pass(check_name, f"No {desc}s found")
        else:
            total_instances = sum(d.get('count', 2) for d in duplicates)
            self._warn(
                check_name,
                f"Found {len(duplicates)} {desc}s ({total_instances} total instances)",
                details={'duplicates': duplicates[:5]},
            )

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path)
        skip_parts = ['venv', '.venv', 'node_modules', '__pycache__',
                      'tests', 'test_', '.git', 'dist', 'build']
        return any(p in path_str for p in skip_parts)

    def can_auto_fix(self, check_name: str) -> bool:
        """Code consolidation requires careful manual review."""
        return False

    def get_refactoring_suggestions(self, duplicates: List[Dict]) -> List[str]:
        """Generate refactoring suggestions for duplicates."""
        suggestions = []

        for dup in duplicates:
            if dup['type'] == 'exact_duplicate':
                locations = dup.get('locations', [])
                if locations:
                    suggestions.append(
                        f"Extract '{locations[0]['name']}' to a shared utility module "
                        f"(found in {len(locations)} places)"
                    )
            elif dup['type'] == 'similar_functions':
                locations = dup.get('locations', [])
                if len(locations) >= 2:
                    suggestions.append(
                        f"Consider parameterizing '{locations[0]['name']}' and "
                        f"'{locations[1]['name']}' into a single function"
                    )

        return suggestions
