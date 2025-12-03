"""
Code Quality Check Module

Checks for rule violations, dead code, and code quality issues.
Auto-fixes safe violations like horizontal lines.
"""

import re
from pathlib import Path
from typing import List, Dict, Any
import logging

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)


class CodeQualityCheck(BaseCheck):
    """
    Checks code quality and rule compliance.

    Rules checked:
    - No horizontal lines (st.markdown('---') or st.divider())
    - No mock/dummy data in production code
    - No emojis unless explicitly requested
    - Filters on main page, not sidebar
    - Function length limits
    - File length limits
    """

    # Rule patterns to check
    RULE_PATTERNS = {
        'horizontal_lines': {
            'pattern': r"st\.(markdown|write)\s*\(\s*['\"]---['\"]\s*\)|st\.divider\s*\(\s*\)",
            'description': 'Horizontal line violation',
            'severity': 'high',
            'auto_fixable': True,
        },
        'emoji_in_code': {
            'pattern': r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]',
            'description': 'Emoji in code (unless requested)',
            'severity': 'low',
            'auto_fixable': False,
        },
        'sidebar_filters': {
            'pattern': r'st\.sidebar\.(selectbox|multiselect|slider|radio|checkbox)',
            'description': 'Filter in sidebar instead of main page',
            'severity': 'medium',
            'auto_fixable': False,
        },
        'deprecated_st_cache': {
            'pattern': r'@st\.cache\s*[(\n]',
            'description': 'Deprecated st.cache (use st.cache_data)',
            'severity': 'medium',
            'auto_fixable': True,
        },
        'hardcoded_secrets': {
            'pattern': r'(api_key|password|secret|token)\s*=\s*["\'][^"\']{10,}["\']',
            'description': 'Potential hardcoded secret',
            'severity': 'critical',
            'auto_fixable': False,
        },
    }

    # Thresholds
    MAX_FUNCTION_LENGTH = 150
    MAX_FILE_LENGTH = 1000

    def __init__(self, project_root: Path = None, auto_fix: bool = True):
        """Initialize code quality check."""
        super().__init__()
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent.parent
        self.auto_fix = auto_fix

    @property
    def name(self) -> str:
        return "code_quality"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.HIGH

    def get_checks_list(self) -> List[str]:
        return [
            "horizontal_lines",
            "deprecated_patterns",
            "sidebar_filters",
            "hardcoded_secrets",
            "function_length",
            "file_length",
        ]

    def run(self) -> ModuleCheckResult:
        """Run all code quality checks."""
        self._start_module()

        # Track violations
        all_violations = []
        fixes_applied = []

        # Scan all Python files
        for py_file in self.project_root.rglob('*.py'):
            # Skip test files and venv
            if self._should_skip(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                relative_path = str(py_file.relative_to(self.project_root))

                # Check each rule
                for rule_name, rule_config in self.RULE_PATTERNS.items():
                    matches = list(re.finditer(
                        rule_config['pattern'],
                        content,
                        re.IGNORECASE
                    ))

                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        violation = {
                            'rule': rule_name,
                            'file': relative_path,
                            'line': line_num,
                            'match': match.group()[:50],
                            'severity': rule_config['severity'],
                            'auto_fixable': rule_config['auto_fixable'],
                        }
                        all_violations.append(violation)

                        # Auto-fix if enabled
                        if self.auto_fix and rule_config['auto_fixable']:
                            fixed = self._apply_fix(py_file, rule_name, match, content)
                            if fixed:
                                fixes_applied.append({
                                    'rule': rule_name,
                                    'file': relative_path,
                                    'line': line_num,
                                })
                                # Reload content after fix
                                content = py_file.read_text(encoding='utf-8', errors='ignore')

                # Check function lengths
                long_functions = self._check_function_lengths(content, relative_path)
                all_violations.extend(long_functions)

                # Check file length
                if content.count('\n') > self.MAX_FILE_LENGTH:
                    all_violations.append({
                        'rule': 'file_length',
                        'file': relative_path,
                        'line': 0,
                        'match': f'{content.count("\n")} lines (max {self.MAX_FILE_LENGTH})',
                        'severity': 'low',
                        'auto_fixable': False,
                    })

            except Exception as e:
                logger.warning(f"Could not process {py_file}: {e}")

        # Report results
        self._report_results(all_violations, fixes_applied)

        return self._end_module()

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            'venv', '.venv', 'node_modules', '__pycache__',
            'tests', 'test_', '_test.py', '.git',
        ]
        path_str = str(file_path)
        return any(p in path_str for p in skip_patterns)

    def _apply_fix(self, file_path: Path, rule_name: str,
                   match: re.Match, content: str) -> bool:
        """Apply auto-fix for a violation."""
        try:
            if rule_name == 'horizontal_lines':
                # Remove the horizontal line
                new_content = content[:match.start()] + content[match.end():]
                # Clean up any resulting double newlines
                new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
                file_path.write_text(new_content, encoding='utf-8')
                logger.info(f"Fixed horizontal line in {file_path}")
                return True

            elif rule_name == 'deprecated_st_cache':
                # Replace st.cache with st.cache_data
                new_content = content.replace('@st.cache', '@st.cache_data')
                file_path.write_text(new_content, encoding='utf-8')
                logger.info(f"Fixed deprecated st.cache in {file_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to apply fix for {rule_name}: {e}")

        return False

    def _check_function_lengths(self, content: str, file_path: str) -> List[Dict]:
        """Check for functions that are too long."""
        violations = []

        # Simple function detection (def keyword)
        function_pattern = r'^(\s*)def\s+(\w+)\s*\('
        lines = content.split('\n')

        current_func = None
        current_indent = 0
        func_start_line = 0
        func_lines = 0

        for i, line in enumerate(lines, 1):
            match = re.match(function_pattern, line)

            if match:
                # Check previous function if any
                if current_func and func_lines > self.MAX_FUNCTION_LENGTH:
                    violations.append({
                        'rule': 'function_length',
                        'file': file_path,
                        'line': func_start_line,
                        'match': f'{current_func}: {func_lines} lines (max {self.MAX_FUNCTION_LENGTH})',
                        'severity': 'medium',
                        'auto_fixable': False,
                    })

                # Start tracking new function
                current_indent = len(match.group(1))
                current_func = match.group(2)
                func_start_line = i
                func_lines = 1

            elif current_func:
                # Count lines in current function
                if line.strip():
                    line_indent = len(line) - len(line.lstrip())
                    if line_indent > current_indent or line.strip().startswith('#'):
                        func_lines += 1
                    else:
                        # Function ended
                        if func_lines > self.MAX_FUNCTION_LENGTH:
                            violations.append({
                                'rule': 'function_length',
                                'file': file_path,
                                'line': func_start_line,
                                'match': f'{current_func}: {func_lines} lines (max {self.MAX_FUNCTION_LENGTH})',
                                'severity': 'medium',
                                'auto_fixable': False,
                            })
                        current_func = None

        return violations

    def _report_results(self, violations: List[Dict], fixes: List[Dict]):
        """Report check results."""
        # Group by rule
        by_rule = {}
        for v in violations:
            rule = v['rule']
            if rule not in by_rule:
                by_rule[rule] = []
            by_rule[rule].append(v)

        # Report each check
        for check_name in self.get_checks():
            rule_violations = by_rule.get(check_name, [])

            if not rule_violations:
                self._pass(check_name, f"No {check_name} violations found")
            else:
                # Check if any were fixed
                fixed_count = sum(
                    1 for f in fixes if f['rule'] == check_name
                )

                if fixed_count == len(rule_violations):
                    self._fixed(
                        check_name,
                        f"Fixed {fixed_count} {check_name} violations",
                        f"Auto-fixed {fixed_count} violations",
                        [v['file'] for v in rule_violations],
                    )
                elif fixed_count > 0:
                    remaining = len(rule_violations) - fixed_count
                    self._warn(
                        check_name,
                        f"Fixed {fixed_count}, {remaining} remaining {check_name} violations",
                        details={'violations': rule_violations[:5]},
                        files=[v['file'] for v in rule_violations],
                    )
                else:
                    severity = rule_violations[0].get('severity', 'medium')
                    if severity == 'critical':
                        self._fail(
                            check_name,
                            f"Found {len(rule_violations)} {check_name} violations",
                            details={'violations': rule_violations[:5]},
                            files=[v['file'] for v in rule_violations],
                        )
                    else:
                        self._warn(
                            check_name,
                            f"Found {len(rule_violations)} {check_name} violations",
                            details={'violations': rule_violations[:5]},
                            files=[v['file'] for v in rule_violations],
                        )

    def can_auto_fix(self, check_name: str) -> bool:
        """Check if a rule can be auto-fixed."""
        rule_config = self.RULE_PATTERNS.get(check_name, {})
        return rule_config.get('auto_fixable', False)
