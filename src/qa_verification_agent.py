"""
QA Verification Agent
=====================

Intelligent automated testing and verification system for development tasks.

Automatically runs comprehensive checks after task completion:
- File existence verification
- Python syntax validation
- Test execution for feature areas
- Integration testing (page load tests)
- Database schema validation

Usage:
    # From command line
    python src/qa_verification_agent.py <task_id>

    # From Python code
    from src.qa_verification_agent import get_qa_agent

    qa_agent = get_qa_agent()
    results = qa_agent.verify_task(task_id=123)

    if results['passed']:
        print("Task verification passed!")
    else:
        print(f"Task verification failed: {results['failures']}")

Features:
- Comprehensive test coverage across multiple dimensions
- Structured result reporting with detailed pass/fail tracking
- Automatic verification logging to database
- Integration with existing test infrastructure
"""

import os
import sys
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.task_db_manager import TaskDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QAVerificationAgent:
    """
    Intelligent QA agent that automatically verifies task completion

    Runs multiple verification checks:
    1. File existence verification
    2. Python syntax validation
    3. Feature-specific test execution
    4. Integration testing (page loads)
    5. Database schema validation
    """

    def __init__(self) -> None:
        """Initialize QA agent with database manager"""
        self.task_mgr = TaskDBManager()
        self.project_root = project_root
        logger.info("QA Verification Agent initialized")

    def verify_task(self, task_id: int) -> Dict[str, Any]:
        """
        Main entry point for task verification

        Args:
            task_id: ID of task to verify

        Returns:
            Dict containing verification results:
            {
                'task_id': int,
                'passed': bool,
                'checks_run': List[Dict],
                'failures': List[Dict],
                'warnings': List[str],
                'summary': str
            }
        """
        logger.info(f"Starting verification for task #{task_id}")
        print(f"\n{'='*80}")
        print(f"[QA AGENT] Verifying task #{task_id}")
        print(f"{'='*80}\n")

        # Get task details
        task = self.task_mgr.get_task(task_id)

        if not task:
            error_result = {
                'task_id': task_id,
                'passed': False,
                'error': 'Task not found',
                'checks_run': [],
                'failures': [],
                'warnings': []
            }
            logger.error(f"Task {task_id} not found")
            return error_result

        print(f"Task: {task['title']}")
        print(f"Feature Area: {task['feature_area']}")
        print(f"Status: {task['status']}\n")

        # Initialize results
        results = {
            'task_id': task_id,
            'task_title': task['title'],
            'feature_area': task['feature_area'],
            'checks_run': [],
            'passed': True,
            'failures': [],
            'warnings': []
        }

        # Get list of modified files
        files = self.task_mgr.get_task_files(task_id)
        logger.info(f"Found {len(files)} modified files for task {task_id}")

        # Run verification checks
        print("Running verification checks...\n")

        # Check 1: Verify files exist
        print("Check 1/5: File Existence Verification")
        file_check = self._verify_files_exist(files)
        results['checks_run'].append(file_check)
        self._print_check_result(file_check)

        if not file_check['passed']:
            results['passed'] = False
            results['failures'].append(file_check)

        # Check 2: Python syntax validation
        print("\nCheck 2/5: Python Syntax Validation")
        python_files = [f['file_path'] for f in files if f['file_path'].endswith('.py')]

        if python_files:
            syntax_check = self._check_python_syntax(python_files)
            results['checks_run'].append(syntax_check)
            self._print_check_result(syntax_check)

            if not syntax_check['passed']:
                results['passed'] = False
                results['failures'].append(syntax_check)
        else:
            results['checks_run'].append({
                'name': 'Python Syntax Validation',
                'passed': True,
                'details': ['[SKIP] No Python files to validate'],
                'skipped': True
            })
            print("  [SKIP] No Python files to validate")

        # Check 3: Run relevant tests
        print("\nCheck 3/5: Feature Test Execution")
        test_check = self._run_tests(task['feature_area'])
        results['checks_run'].append(test_check)
        self._print_check_result(test_check)

        if not test_check['passed'] and not test_check.get('skipped'):
            results['passed'] = False
            results['failures'].append(test_check)
        elif test_check.get('skipped'):
            results['warnings'].append(f"No tests available for {task['feature_area']}")

        # Check 4: Integration test (page loads)
        print("\nCheck 4/5: Integration Testing (Page Loads)")
        if task['feature_area'] in ['comprehensive_strategy', 'dashboard', 'enhancement_manager', 'ai_options_agent']:
            integration_check = self._test_page_loads(task['feature_area'])
            results['checks_run'].append(integration_check)
            self._print_check_result(integration_check)

            if not integration_check['passed']:
                results['passed'] = False
                results['failures'].append(integration_check)
        else:
            results['checks_run'].append({
                'name': 'Integration Testing',
                'passed': True,
                'details': [f"[SKIP] No integration tests for {task['feature_area']}"],
                'skipped': True
            })
            print(f"  [SKIP] No integration tests for {task['feature_area']}")

        # Check 5: Database schema validation (if database-related)
        print("\nCheck 5/5: Database Schema Validation")
        if self._is_database_task(task):
            db_check = self._verify_database_schema()
            results['checks_run'].append(db_check)
            self._print_check_result(db_check)

            if not db_check['passed']:
                results['passed'] = False
                results['failures'].append(db_check)
        else:
            results['checks_run'].append({
                'name': 'Database Schema Validation',
                'passed': True,
                'details': ['[SKIP] Not a database-related task'],
                'skipped': True
            })
            print("  [SKIP] Not a database-related task")

        # Generate summary
        results['summary'] = self._generate_summary(results)

        # Print final results
        print(f"\n{'='*80}")
        if results['passed']:
            print("[PASS] ALL VERIFICATION CHECKS PASSED")
        else:
            print("[FAIL] VERIFICATION FAILED")
            print(f"\nFailed checks: {len(results['failures'])}")
            for failure in results['failures']:
                print(f"  - {failure['name']}")

        if results['warnings']:
            print(f"\nWarnings: {len(results['warnings'])}")
            for warning in results['warnings']:
                print(f"  [WARN] {warning}")

        print(f"{'='*80}\n")

        # Save verification results to database
        try:
            self.task_mgr.add_verification(
                task_id=task_id,
                verified_by='qa_agent',
                passed=results['passed'],
                verification_notes=results['summary'],
                test_results=results
            )
            logger.info(f"Verification results saved to database for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to save verification results: {e}")
            results['warnings'].append(f"Failed to save verification to database: {e}")

        return results

    def _verify_files_exist(self, files: List[Dict]) -> Dict:
        """
        Verify all modified files exist

        Args:
            files: List of file change records

        Returns:
            Check result dict
        """
        check = {
            'name': 'File Existence Check',
            'passed': True,
            'details': []
        }

        if not files:
            check['details'].append("[WARN] No files tracked for this task")
            check['passed'] = True
            return check

        for file_info in files:
            file_path = file_info['file_path']

            # Handle both absolute and relative paths
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.project_root, file_path)

            if not os.path.exists(file_path):
                check['passed'] = False
                check['details'].append(f"[FAIL] Missing: {file_info['file_path']}")
                logger.warning(f"File not found: {file_path}")
            else:
                file_size = os.path.getsize(file_path)
                check['details'].append(f"[PASS] Found: {file_info['file_path']} ({file_size:,} bytes)")

        return check

    def _check_python_syntax(self, python_files: List[str]) -> Dict:
        """
        Validate Python syntax for all Python files

        Args:
            python_files: List of Python file paths

        Returns:
            Check result dict
        """
        check = {
            'name': 'Python Syntax Validation',
            'passed': True,
            'details': []
        }

        for file_path in python_files:
            # Handle both absolute and relative paths
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.project_root, file_path)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Compile to check syntax
                compile(code, file_path, 'exec')
                check['details'].append(f"[PASS] Valid syntax: {os.path.basename(file_path)}")
                logger.info(f"Syntax valid: {file_path}")

            except SyntaxError as e:
                check['passed'] = False
                error_msg = f"Line {e.lineno}: {e.msg}"
                check['details'].append(f"[FAIL] Syntax error in {os.path.basename(file_path)}: {error_msg}")
                logger.error(f"Syntax error in {file_path}: {e}")

            except Exception as e:
                check['passed'] = False
                check['details'].append(f"[FAIL] Error reading {os.path.basename(file_path)}: {str(e)}")
                logger.error(f"Error reading {file_path}: {e}")

        return check

    def _run_tests(self, feature_area: str) -> Dict:
        """
        Run relevant tests for feature area

        Args:
            feature_area: Feature area to test

        Returns:
            Check result dict
        """
        check = {
            'name': 'Test Execution',
            'passed': True,
            'details': []
        }

        # Map feature areas to test files
        test_mapping = {
            'comprehensive_strategy': 'test_comprehensive_integration_live.py',
            'xtrades': 'test_xtrades_complete.py',
            'dashboard': 'test_dashboard_display.py',
            'task_management': 'test_task_management_system.py',
            'ai_options_agent': 'test_ai_options_agent.py'
        }

        test_file = test_mapping.get(feature_area)

        if not test_file:
            check['details'].append(f"[WARN] No test mapping for feature area: {feature_area}")
            check['skipped'] = True
            return check

        test_path = os.path.join(self.project_root, test_file)

        if not os.path.exists(test_path):
            check['details'].append(f"[WARN] Test file not found: {test_file}")
            check['skipped'] = True
            logger.warning(f"Test file not found: {test_path}")
            return check

        # Run the test
        try:
            logger.info(f"Running test: {test_file}")
            result = subprocess.run(
                [sys.executable, test_path],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=str(self.project_root)
            )

            if result.returncode == 0:
                check['details'].append(f"[PASS] Tests passed: {test_file}")
                # Add output preview
                if result.stdout:
                    output_preview = result.stdout[:500].strip()
                    check['details'].append(f"   Output preview: {output_preview}")
                logger.info(f"Tests passed: {test_file}")
            else:
                check['passed'] = False
                check['details'].append(f"[FAIL] Tests failed: {test_file}")
                check['details'].append(f"   Return code: {result.returncode}")

                # Add error output
                if result.stderr:
                    error_preview = result.stderr[:500].strip()
                    check['details'].append(f"   Error: {error_preview}")

                if result.stdout:
                    output_preview = result.stdout[:500].strip()
                    check['details'].append(f"   Output: {output_preview}")

                logger.error(f"Tests failed: {test_file} (exit code {result.returncode})")

        except subprocess.TimeoutExpired:
            check['passed'] = False
            check['details'].append(f"[FAIL] Test timeout: {test_file} (exceeded 120 seconds)")
            logger.error(f"Test timeout: {test_file}")

        except Exception as e:
            check['passed'] = False
            check['details'].append(f"[FAIL] Error running tests: {str(e)}")
            logger.error(f"Error running tests for {feature_area}: {e}")

        return check

    def _test_page_loads(self, feature_area: str) -> Dict:
        """
        Test that Streamlit page loads without errors

        Args:
            feature_area: Feature area to test

        Returns:
            Check result dict
        """
        check = {
            'name': 'Page Load Test',
            'passed': True,
            'details': []
        }

        # Map feature areas to page modules
        page_mapping = {
            'comprehensive_strategy': 'comprehensive_strategy_page',
            'enhancement_manager': 'enhancement_manager_page',
            'ai_options_agent': 'ai_options_agent_page'
        }

        page_module = page_mapping.get(feature_area)

        if not page_module:
            check['details'].append(f"[WARN] No page mapping for {feature_area}")
            check['skipped'] = True
            return check

        # Try to import the page module
        try:
            logger.info(f"Testing page load: {page_module}")

            # Import the module
            module = __import__(page_module)

            # Check for render function
            render_functions = [attr for attr in dir(module) if 'render' in attr.lower() and callable(getattr(module, attr))]

            if render_functions:
                check['details'].append(f"[PASS] {page_module} imports successfully")
                check['details'].append(f"   Found render functions: {', '.join(render_functions)}")
                logger.info(f"Page loads successfully: {page_module}")
            else:
                check['details'].append(f"[WARN] {page_module} imports but no render function found")
                logger.warning(f"No render function in {page_module}")

        except ImportError as e:
            check['passed'] = False
            check['details'].append(f"[FAIL] Import failed: {page_module}")
            check['details'].append(f"   Error: {str(e)}")
            logger.error(f"Failed to import {page_module}: {e}")

        except Exception as e:
            check['passed'] = False
            check['details'].append(f"[FAIL] Error loading page: {str(e)}")
            logger.error(f"Error loading page {page_module}: {e}")

        return check

    def _verify_database_schema(self) -> Dict:
        """
        Verify database schema is valid

        Returns:
            Check result dict
        """
        check = {
            'name': 'Database Schema Validation',
            'passed': True,
            'details': []
        }

        try:
            import psycopg2
            from dotenv import load_dotenv
            load_dotenv()

            # Connect to database
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 5432)),
                database=os.getenv('DB_NAME', 'magnus'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD')
            )
            cur = conn.cursor()

            # Check that all expected task management tables exist
            expected_tables = ['development_tasks', 'task_execution_log', 'task_verification', 'task_files']

            for table in expected_tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    check['details'].append(f"[PASS] Table '{table}' exists ({count:,} rows)")
                except psycopg2.Error as e:
                    check['passed'] = False
                    check['details'].append(f"[FAIL] Table '{table}' check failed: {str(e)}")
                    logger.error(f"Table check failed for {table}: {e}")

            cur.close()
            conn.close()

            logger.info("Database schema validation completed")

        except psycopg2.Error as e:
            check['passed'] = False
            check['details'].append(f"[FAIL] Database connection failed: {str(e)}")
            logger.error(f"Database connection failed: {e}")

        except Exception as e:
            check['passed'] = False
            check['details'].append(f"[FAIL] Database check error: {str(e)}")
            logger.error(f"Database check error: {e}")

        return check

    def _is_database_task(self, task: Dict) -> bool:
        """
        Determine if task is database-related

        Args:
            task: Task dictionary

        Returns:
            True if database-related
        """
        database_keywords = [
            'database', 'schema', 'table', 'migration', 'sql',
            'postgres', 'query', 'index', 'db_manager'
        ]

        title_lower = task['title'].lower()
        desc_lower = (task['description'] or '').lower()

        return any(keyword in title_lower or keyword in desc_lower for keyword in database_keywords)

    def _generate_summary(self, results: Dict) -> str:
        """
        Generate human-readable summary of verification results

        Args:
            results: Verification results dict

        Returns:
            Summary string
        """
        summary_lines = [
            f"QA Verification for Task #{results['task_id']}: {results['task_title']}",
            f"Feature Area: {results['feature_area']}",
            f"Verification Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Overall Result: {'PASSED' if results['passed'] else 'FAILED'}",
            f"Checks Run: {len(results['checks_run'])}",
            f"Checks Passed: {sum(1 for c in results['checks_run'] if c['passed'])}",
            f"Checks Failed: {len(results['failures'])}",
            f"Warnings: {len(results['warnings'])}",
            "",
            "Checks Performed:"
        ]

        for check in results['checks_run']:
            status = "PASS" if check['passed'] else "FAIL"
            if check.get('skipped'):
                status = "SKIP"
            summary_lines.append(f"  - {check['name']}: {status}")

        if results['failures']:
            summary_lines.append("\nFailed Checks:")
            for failure in results['failures']:
                summary_lines.append(f"  - {failure['name']}")
                for detail in failure['details'][:3]:  # First 3 details
                    summary_lines.append(f"    {detail}")

        if results['warnings']:
            summary_lines.append("\nWarnings:")
            for warning in results['warnings']:
                summary_lines.append(f"  - {warning}")

        return "\n".join(summary_lines)

    def _print_check_result(self, check: Dict):
        """Print check result to console"""
        status = "[PASS]" if check['passed'] else "[FAIL]"
        if check.get('skipped'):
            status = "[SKIP]"

        print(f"  {status}")

        # Print details (limit to 5)
        for detail in check['details'][:5]:
            print(f"  {detail}")

        if len(check['details']) > 5:
            print(f"  ... and {len(check['details']) - 5} more")


# Singleton instance
_qa_agent = None


def get_qa_agent() -> QAVerificationAgent:
    """
    Get singleton QA agent instance

    Returns:
        QAVerificationAgent instance
    """
    global _qa_agent
    if _qa_agent is None:
        _qa_agent = QAVerificationAgent()
    return _qa_agent


# Command-line interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/qa_verification_agent.py <task_id>")
        print("\nExample: python src/qa_verification_agent.py 123")
        sys.exit(1)

    try:
        task_id = int(sys.argv[1])
        agent = QAVerificationAgent()
        results = agent.verify_task(task_id)

        # Print JSON results for programmatic consumption
        if '--json' in sys.argv:
            print("\n" + "="*80)
            print("JSON Results:")
            print("="*80)
            print(json.dumps(results, indent=2, default=str))

        # Exit with appropriate code
        sys.exit(0 if results['passed'] else 1)

    except ValueError:
        print(f"Error: Invalid task ID '{sys.argv[1]}'. Must be an integer.")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
