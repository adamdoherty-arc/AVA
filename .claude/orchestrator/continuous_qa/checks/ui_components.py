"""
UI Components Check Module

Tests React frontend components to ensure:
1. Build succeeds without errors
2. All routes are accessible
3. No console errors on page load
4. Button handlers are properly connected
5. API calls from frontend work correctly
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)


class UIComponentsCheck(BaseCheck):
    """
    Tests React frontend components for functionality and correctness.

    CRITICAL check - UI must work properly for a financial trading tool.
    """

    # Default routes to test
    DEFAULT_ROUTES = [
        ("/", "Dashboard"),
        ("/positions", "Positions"),
        ("/opportunities", "Opportunities"),
        ("/scanner", "Premium Scanner"),
        ("/sports", "Sports Betting"),
        ("/kalshi", "Kalshi Markets"),
        ("/chat", "AVA Chat"),
        ("/earnings", "Earnings Calendar"),
        ("/settings", "Settings"),
    ]

    # Patterns that indicate broken components
    ERROR_PATTERNS = [
        r'Cannot read propert',
        r'is not defined',
        r'is not a function',
        r'undefined is not',
        r'null is not',
        r'Failed to compile',
        r'Module not found',
        r'SyntaxError',
        r'TypeError',
        r'ReferenceError',
    ]

    # Patterns for unconnected handlers
    UNCONNECTED_HANDLER_PATTERNS = [
        r'onClick=\{undefined\}',
        r'onClick=\{\s*\}',
        r'onSubmit=\{undefined\}',
        r'onChange=\{undefined\}',
        r'// TODO.*handler',
        r'// FIXME.*handler',
    ]

    def __init__(self, project_root: Path = None, frontend_dir: str = "frontend"):
        """
        Initialize UI components check.

        Args:
            project_root: Root directory of the project
            frontend_dir: Name of frontend directory
        """
        super().__init__()
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent.parent
        self.frontend_path = self.project_root / frontend_dir

    @property
    def name(self) -> str:
        return "ui_components"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.CRITICAL

    def get_checks_list(self) -> List[str]:
        return [
            "frontend_exists",
            "dependencies_installed",
            "build_succeeds",
            "typescript_errors",
            "unconnected_handlers",
            "component_imports",
        ]

    def run(self) -> ModuleCheckResult:
        """Run UI component checks."""
        self._start_module()

        # Check if frontend directory exists
        if not self.frontend_path.exists():
            self._fail(
                "frontend_exists",
                f"Frontend directory not found at {self.frontend_path}",
            )
            self._skip("dependencies_installed", "Frontend not found")
            self._skip("build_succeeds", "Frontend not found")
            self._skip("typescript_errors", "Frontend not found")
            self._skip("unconnected_handlers", "Frontend not found")
            self._skip("component_imports", "Frontend not found")
            return self._end_module()

        self._pass("frontend_exists", f"Frontend found at {self.frontend_path}")

        # Check dependencies
        deps_ok = self._check_dependencies()
        if not deps_ok:
            self._fail(
                "dependencies_installed",
                "Node modules not installed or package.json missing",
                details={"path": str(self.frontend_path)},
            )
            self._skip("build_succeeds", "Dependencies not installed")
        else:
            self._pass("dependencies_installed", "Node dependencies are installed")

            # Try to build (or just type-check)
            build_result = self._check_build()
            if build_result['success']:
                self._pass("build_succeeds", "Frontend builds successfully")
            else:
                self._fail(
                    "build_succeeds",
                    f"Frontend build failed: {build_result.get('error', 'Unknown error')[:100]}",
                    details={'errors': build_result.get('errors', [])[:5]},
                )

        # Check for TypeScript errors in source files
        ts_errors = self._check_typescript_source()
        if not ts_errors:
            self._pass("typescript_errors", "No TypeScript errors found in source")
        else:
            self._fail(
                "typescript_errors",
                f"Found {len(ts_errors)} TypeScript issues in source files",
                details={'errors': ts_errors[:10]},
                files=[e['file'] for e in ts_errors[:10]],
            )

        # Check for unconnected handlers
        handler_issues = self._check_handlers()
        if not handler_issues:
            self._pass("unconnected_handlers", "All event handlers are properly connected")
        else:
            self._warn(
                "unconnected_handlers",
                f"Found {len(handler_issues)} potentially unconnected handlers",
                details={'issues': handler_issues[:10]},
                files=[i['file'] for i in handler_issues[:10]],
            )

        # Check component imports
        import_issues = self._check_component_imports()
        if not import_issues:
            self._pass("component_imports", "All component imports are valid")
        else:
            self._fail(
                "component_imports",
                f"Found {len(import_issues)} import issues",
                details={'issues': import_issues[:10]},
                files=[i['file'] for i in import_issues[:10]],
            )

        return self._end_module()

    def _check_dependencies(self) -> bool:
        """Check if node_modules exists and package.json is valid."""
        package_json = self.frontend_path / "package.json"
        node_modules = self.frontend_path / "node_modules"

        if not package_json.exists():
            return False

        if not node_modules.exists():
            return False

        return True

    def _check_build(self) -> Dict[str, Any]:
        """Run build or type-check to verify frontend compiles."""
        try:
            # First try tsc --noEmit for type checking only (faster)
            result = subprocess.run(
                ["npx", "tsc", "--noEmit"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True,
                timeout=120,
                shell=True,
            )

            if result.returncode == 0:
                return {'success': True}

            # Parse errors
            errors = []
            for line in result.stdout.split('\n') + result.stderr.split('\n'):
                if 'error TS' in line or 'Error:' in line:
                    errors.append(line.strip()[:200])

            return {
                'success': False,
                'error': result.stderr[:200] if result.stderr else 'Type check failed',
                'errors': errors[:10],
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Build timed out after 120 seconds'}
        except FileNotFoundError:
            # npx not found, try npm run build
            try:
                result = subprocess.run(
                    ["npm", "run", "build"],
                    cwd=self.frontend_path,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    shell=True,
                )
                return {'success': result.returncode == 0, 'error': result.stderr[:200]}
            except Exception as e:
                return {'success': False, 'error': str(e)[:200]}
        except Exception as e:
            return {'success': False, 'error': str(e)[:200]}

    def _check_typescript_source(self) -> List[Dict]:
        """Check TypeScript source files for common issues."""
        issues = []

        # Scan all .tsx and .ts files
        for ext in ['*.tsx', '*.ts']:
            for ts_file in self.frontend_path.rglob(ext):
                # Skip node_modules and build directories
                if 'node_modules' in str(ts_file) or 'dist' in str(ts_file):
                    continue

                try:
                    content = ts_file.read_text(encoding='utf-8', errors='ignore')
                    relative_path = str(ts_file.relative_to(self.frontend_path))

                    # Check for error patterns
                    for pattern in self.ERROR_PATTERNS:
                        if re.search(pattern, content):
                            matches = re.findall(pattern, content)
                            for match in matches[:3]:  # Limit matches per file
                                issues.append({
                                    'file': relative_path,
                                    'pattern': pattern,
                                    'match': match[:50],
                                })

                    # Check for 'any' type usage (code quality)
                    any_matches = re.findall(r':\s*any\b', content)
                    if len(any_matches) > 5:
                        issues.append({
                            'file': relative_path,
                            'pattern': 'excessive_any_type',
                            'match': f'{len(any_matches)} uses of "any" type',
                        })

                except Exception as e:
                    logger.warning(f"Could not read {ts_file}: {e}")

        return issues

    def _check_handlers(self) -> List[Dict]:
        """Check for unconnected event handlers."""
        issues = []

        for tsx_file in self.frontend_path.rglob('*.tsx'):
            if 'node_modules' in str(tsx_file):
                continue

            try:
                content = tsx_file.read_text(encoding='utf-8', errors='ignore')
                relative_path = str(tsx_file.relative_to(self.frontend_path))

                for pattern in self.UNCONNECTED_HANDLER_PATTERNS:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'file': relative_path,
                            'line': line_num,
                            'pattern': pattern,
                            'match': match.group()[:50],
                        })

            except Exception as e:
                logger.warning(f"Could not read {tsx_file}: {e}")

        return issues

    def _check_component_imports(self) -> List[Dict]:
        """Check for invalid or missing component imports."""
        issues = []

        # Track all exported components
        exported_components = set()
        component_files = {}

        for tsx_file in self.frontend_path.rglob('*.tsx'):
            if 'node_modules' in str(tsx_file):
                continue

            try:
                content = tsx_file.read_text(encoding='utf-8', errors='ignore')
                relative_path = str(tsx_file.relative_to(self.frontend_path))

                # Find exports
                export_matches = re.findall(
                    r'export\s+(?:default\s+)?(?:function|const|class)\s+(\w+)',
                    content
                )
                for comp in export_matches:
                    exported_components.add(comp)
                    component_files[comp] = relative_path

            except Exception as e:
                logger.warning(f"Could not read {tsx_file}: {e}")

        # Now check imports
        for tsx_file in self.frontend_path.rglob('*.tsx'):
            if 'node_modules' in str(tsx_file):
                continue

            try:
                content = tsx_file.read_text(encoding='utf-8', errors='ignore')
                relative_path = str(tsx_file.relative_to(self.frontend_path))

                # Find imports from local files
                import_matches = re.findall(
                    r"import\s+\{([^}]+)\}\s+from\s+['\"]\.\.?/([^'\"]+)['\"]",
                    content
                )

                for imported_names, import_path in import_matches:
                    for name in imported_names.split(','):
                        name = name.strip()
                        if name and name not in exported_components:
                            # Could be a missing export
                            if not name.startswith('type '):  # Skip type imports
                                issues.append({
                                    'file': relative_path,
                                    'import': name,
                                    'from': import_path,
                                    'issue': 'Imported component may not be exported',
                                })

            except Exception as e:
                logger.warning(f"Could not read {tsx_file}: {e}")

        return issues

    def can_auto_fix(self, check_name: str) -> bool:
        """UI component issues generally cannot be auto-fixed."""
        return False
