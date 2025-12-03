"""
Dummy Data Detector Check

Scans the entire codebase for mock/dummy/fake data patterns and flags or removes them.
This is a CRITICAL check - Magnus must use real data only.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)


class DummyDataDetectorCheck(BaseCheck):
    """
    Detects and optionally removes dummy/mock data patterns from codebase.

    CRITICAL check - Mock data in production is unacceptable for a financial tool.
    """

    # Patterns that indicate dummy/mock data
    DUMMY_PATTERNS = [
        # Random data generation
        (r'random\.(uniform|randint|choice|random)\s*\(', 'random_data', 'Random data generation'),
        (r'np\.random\.(rand|randn|randint|uniform|choice)', 'numpy_random', 'NumPy random data'),

        # Explicit mock/dummy/fake markers
        (r'mock_data|dummy_data|fake_data|sample_data', 'mock_marker', 'Mock data marker'),
        (r'test_data\s*=|sample_\w+\s*=', 'test_data', 'Test data assignment'),

        # Placeholder patterns
        (r"'TODO'|'PLACEHOLDER'|'FIXME'", 'placeholder', 'Placeholder value'),
        (r'"TODO"|"PLACEHOLDER"|"FIXME"', 'placeholder', 'Placeholder value'),

        # Lorem ipsum
        (r'lorem\s+ipsum', 'lorem_ipsum', 'Lorem ipsum text'),

        # Hardcoded test values
        (r'=\s*\[\s*1,\s*2,\s*3\s*\]', 'test_list', 'Hardcoded test list'),
        (r"=\s*'test'|=\s*\"test\"", 'test_string', 'Test string value'),

        # Mock objects
        (r'Mock\(\)|MagicMock\(\)|patch\(', 'mock_object', 'Mock object (OK in tests)'),

        # Fake generators
        (r'faker\.|Faker\(', 'faker', 'Faker library usage'),
    ]

    # Paths to exclude from checks
    EXCLUDE_PATTERNS = [
        '**/tests/**',
        '**/test_*.py',
        '**/*_test.py',
        '**/fixtures/**',
        '**/conftest.py',
        '**/__pycache__/**',
        '**/node_modules/**',
        '**/.git/**',
        '**/venv/**',
        '**/.venv/**',
    ]

    def __init__(self, project_root: Path = None, auto_remove: bool = False):
        """
        Initialize the dummy data detector.

        Args:
            project_root: Root directory to scan
            auto_remove: If True, remove detected dummy data
        """
        super().__init__()
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent.parent
        self.auto_remove = auto_remove

    @property
    def name(self) -> str:
        return "dummy_data_detector"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.CRITICAL

    def get_checks_list(self) -> List[str]:
        return [
            "python_files_scan",
            "typescript_files_scan",
            "json_files_scan",
        ]

    def run(self) -> ModuleCheckResult:
        """Run the dummy data detection."""
        self._start_module()

        # Scan Python files
        python_violations = self._scan_python_files()

        # Scan TypeScript/JavaScript files
        ts_violations = self._scan_typescript_files()

        # Scan JSON files for obvious test data
        json_violations = self._scan_json_files()

        # Total violations
        total = len(python_violations) + len(ts_violations) + len(json_violations)

        if total == 0:
            self._pass(
                "python_files_scan",
                "No dummy data patterns found in Python files"
            )
            self._pass(
                "typescript_files_scan",
                "No dummy data patterns found in TypeScript files"
            )
            self._pass(
                "json_files_scan",
                "No dummy data patterns found in JSON files"
            )
        else:
            # Report Python violations
            if python_violations:
                self._fail(
                    "python_files_scan",
                    f"Found {len(python_violations)} dummy data patterns in Python files",
                    details={'violations': python_violations[:10]},  # First 10
                    auto_fixable=True,
                    files=[v['file'] for v in python_violations],
                )
            else:
                self._pass("python_files_scan", "No dummy data in Python files")

            # Report TypeScript violations
            if ts_violations:
                self._fail(
                    "typescript_files_scan",
                    f"Found {len(ts_violations)} dummy data patterns in TypeScript files",
                    details={'violations': ts_violations[:10]},
                    auto_fixable=True,
                    files=[v['file'] for v in ts_violations],
                )
            else:
                self._pass("typescript_files_scan", "No dummy data in TypeScript files")

            # Report JSON violations
            if json_violations:
                self._warn(
                    "json_files_scan",
                    f"Found {len(json_violations)} potential test data in JSON files",
                    details={'violations': json_violations[:10]},
                    files=[v['file'] for v in json_violations],
                )
            else:
                self._pass("json_files_scan", "No dummy data in JSON files")

        return self._end_module()

    def _scan_python_files(self) -> List[Dict]:
        """Scan all Python files for dummy data patterns."""
        violations = []

        # Get all Python files
        for py_file in self.project_root.rglob('*.py'):
            # Skip excluded paths
            if self._should_exclude(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                file_violations = self._scan_content(content, py_file)
                violations.extend(file_violations)
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")

        return violations

    def _scan_typescript_files(self) -> List[Dict]:
        """Scan TypeScript/JavaScript files for dummy data patterns."""
        violations = []

        # Get all TS/JS files
        for ext in ['*.ts', '*.tsx', '*.js', '*.jsx']:
            for ts_file in self.project_root.rglob(ext):
                if self._should_exclude(ts_file):
                    continue

                try:
                    content = ts_file.read_text(encoding='utf-8', errors='ignore')
                    file_violations = self._scan_content(content, ts_file)
                    violations.extend(file_violations)
                except Exception as e:
                    logger.warning(f"Could not read {ts_file}: {e}")

        return violations

    def _scan_json_files(self) -> List[Dict]:
        """Scan JSON files for obvious test data."""
        violations = []

        json_patterns = [
            (r'"name":\s*"test|"name":\s*"dummy', 'test_name', 'Test name in JSON'),
            (r'"data":\s*\[\s*1,\s*2,\s*3', 'test_array', 'Test array in JSON'),
        ]

        for json_file in self.project_root.rglob('*.json'):
            if self._should_exclude(json_file):
                continue

            # Skip package.json, package-lock.json, etc.
            if json_file.name in ['package.json', 'package-lock.json', 'tsconfig.json']:
                continue

            try:
                content = json_file.read_text(encoding='utf-8', errors='ignore')

                for pattern, pattern_type, description in json_patterns:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        violations.append({
                            'file': str(json_file.relative_to(self.project_root)),
                            'line': line_num,
                            'pattern_type': pattern_type,
                            'description': description,
                            'match': match.group()[:50],
                        })
            except Exception as e:
                logger.warning(f"Could not read {json_file}: {e}")

        return violations

    def _scan_content(self, content: str, file_path: Path) -> List[Dict]:
        """Scan file content for dummy data patterns."""
        violations = []
        relative_path = str(file_path.relative_to(self.project_root))

        for pattern, pattern_type, description in self.DUMMY_PATTERNS:
            # Skip mock patterns in test context
            if pattern_type == 'mock_object':
                continue  # These are OK in tests

            matches = list(re.finditer(pattern, content, re.IGNORECASE))

            for match in matches:
                # Get line number
                line_num = content[:match.start()].count('\n') + 1

                # Get the line content
                lines = content.split('\n')
                line_content = lines[line_num - 1] if line_num <= len(lines) else ''

                # Skip if it's in a comment
                if self._is_in_comment(line_content, match.group()):
                    continue

                # Skip if it's a test file indicator in the import
                if 'import' in line_content.lower() and pattern_type in ['mock_object', 'faker']:
                    continue

                violations.append({
                    'file': relative_path,
                    'line': line_num,
                    'pattern_type': pattern_type,
                    'description': description,
                    'match': match.group()[:50],
                    'context': line_content.strip()[:100],
                })

        return violations

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from scanning."""
        relative = str(file_path.relative_to(self.project_root))

        for pattern in self.EXCLUDE_PATTERNS:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace('**/', '(.*/)?').replace('*', '[^/]*')
            if re.match(regex_pattern, relative):
                return True

        return False

    def _is_in_comment(self, line: str, match: str) -> bool:
        """Check if match is in a comment."""
        # Python comment
        if '#' in line:
            comment_start = line.index('#')
            match_pos = line.find(match)
            if match_pos > comment_start:
                return True

        # JavaScript/TypeScript comment
        if '//' in line:
            comment_start = line.index('//')
            match_pos = line.find(match)
            if match_pos > comment_start:
                return True

        return False

    def can_auto_fix(self, check_name: str) -> bool:
        """Dummy data generally cannot be auto-fixed safely."""
        return False  # Require manual review

    def get_violation_summary(self, violations: List[Dict]) -> Dict[str, int]:
        """Get summary of violations by type."""
        summary = {}
        for v in violations:
            pattern_type = v.get('pattern_type', 'unknown')
            summary[pattern_type] = summary.get(pattern_type, 0) + 1
        return summary
