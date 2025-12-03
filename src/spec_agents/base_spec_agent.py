"""
Base Spec Agent - Foundation for All Feature-Specific SpecAgents

Provides:
- HTTP testing via httpx
- Browser automation via Playwright
- Database queries
- Spec file loading
- Issue detection and reporting
- Schema validation for API responses
- Async context manager for resource management
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import yaml
import json
import re

logger = logging.getLogger(__name__)


# ==================== Schema Validation ====================

class SchemaType(Enum):
    """Supported schema types for validation"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    ANY = "any"


@dataclass
class SchemaField:
    """Defines a field in a response schema"""
    name: str
    field_type: SchemaType
    required: bool = True
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    items_schema: Optional['ResponseSchema'] = None  # For arrays
    nested_schema: Optional['ResponseSchema'] = None  # For objects
    enum_values: Optional[List[Any]] = None  # Allowed values


@dataclass
class ResponseSchema:
    """Defines expected schema for an API response"""
    fields: List[SchemaField]
    allow_extra_fields: bool = True  # Whether to allow undeclared fields
    array_response: bool = False  # True if response is an array at root level
    array_min_items: int = 0
    array_max_items: Optional[int] = None


class SchemaValidator:
    """Validates API responses against defined schemas"""

    @staticmethod
    def validate(data: Any, schema: ResponseSchema, path: str = "root") -> List['Issue']:
        """
        Validate data against a schema.

        Args:
            data: The data to validate
            schema: The ResponseSchema to validate against
            path: Current path for error messages

        Returns:
            List of Issues for validation failures
        """
        issues = []

        # Handle array responses
        if schema.array_response:
            if not isinstance(data, list):
                issues.append(Issue(
                    title=f"Expected array at {path}",
                    description=f"Got {type(data).__name__} instead of list",
                    severity=IssueSeverity.HIGH,
                    feature="schema_validation",
                    component="api",
                    expected="array",
                    actual=type(data).__name__,
                ))
                return issues

            # Validate array constraints
            if len(data) < schema.array_min_items:
                issues.append(Issue(
                    title=f"Array too small at {path}",
                    description=f"Expected at least {schema.array_min_items} items, got {len(data)}",
                    severity=IssueSeverity.MEDIUM,
                    feature="schema_validation",
                    component="api",
                ))

            if schema.array_max_items and len(data) > schema.array_max_items:
                issues.append(Issue(
                    title=f"Array too large at {path}",
                    description=f"Expected at most {schema.array_max_items} items, got {len(data)}",
                    severity=IssueSeverity.MEDIUM,
                    feature="schema_validation",
                    component="api",
                ))

            # Validate first few items
            for i, item in enumerate(data[:5]):
                item_issues = SchemaValidator._validate_object(item, schema.fields, f"{path}[{i}]", schema.allow_extra_fields)
                issues.extend(item_issues)
        else:
            # Object response
            if not isinstance(data, dict):
                issues.append(Issue(
                    title=f"Expected object at {path}",
                    description=f"Got {type(data).__name__} instead of dict",
                    severity=IssueSeverity.HIGH,
                    feature="schema_validation",
                    component="api",
                    expected="object",
                    actual=type(data).__name__,
                ))
                return issues

            issues.extend(SchemaValidator._validate_object(data, schema.fields, path, schema.allow_extra_fields))

        return issues

    @staticmethod
    def _validate_object(data: dict, fields: List[SchemaField], path: str, allow_extra: bool) -> List['Issue']:
        """Validate an object against a list of field definitions"""
        issues = []

        # Check required fields
        for field in fields:
            field_path = f"{path}.{field.name}"

            if field.name not in data:
                if field.required:
                    issues.append(Issue(
                        title=f"Missing required field: {field.name}",
                        description=f"Field '{field_path}' is required but missing",
                        severity=IssueSeverity.MEDIUM,
                        feature="schema_validation",
                        component="api",
                    ))
                continue

            value = data[field.name]

            # Handle null values
            if value is None:
                if not field.nullable:
                    issues.append(Issue(
                        title=f"Unexpected null: {field.name}",
                        description=f"Field '{field_path}' is null but not nullable",
                        severity=IssueSeverity.MEDIUM,
                        feature="schema_validation",
                        component="api",
                    ))
                continue

            # Type validation
            issues.extend(SchemaValidator._validate_field_type(value, field, field_path))

        # Check for extra fields if not allowed
        if not allow_extra:
            declared_fields = {f.name for f in fields}
            extra_fields = set(data.keys()) - declared_fields
            if extra_fields:
                issues.append(Issue(
                    title=f"Unexpected fields at {path}",
                    description=f"Extra fields found: {', '.join(extra_fields)}",
                    severity=IssueSeverity.LOW,
                    feature="schema_validation",
                    component="api",
                ))

        return issues

    @staticmethod
    def _validate_field_type(value: Any, field: SchemaField, path: str) -> List['Issue']:
        """Validate a single field value against its type definition"""
        issues = []

        # Type checking
        type_map = {
            SchemaType.STRING: str,
            SchemaType.INTEGER: int,
            SchemaType.NUMBER: (int, float),
            SchemaType.BOOLEAN: bool,
            SchemaType.ARRAY: list,
            SchemaType.OBJECT: dict,
        }

        if field.field_type != SchemaType.ANY:
            expected_type = type_map.get(field.field_type)
            if expected_type and not isinstance(value, expected_type):
                # Special case: int is valid for number type
                if not (field.field_type == SchemaType.NUMBER and isinstance(value, (int, float))):
                    issues.append(Issue(
                        title=f"Type mismatch: {field.name}",
                        description=f"Expected {field.field_type.value} at '{path}', got {type(value).__name__}",
                        severity=IssueSeverity.MEDIUM,
                        feature="schema_validation",
                        component="api",
                        expected=field.field_type.value,
                        actual=type(value).__name__,
                    ))
                    return issues

        # Enum validation
        if field.enum_values and value not in field.enum_values:
            issues.append(Issue(
                title=f"Invalid enum value: {field.name}",
                description=f"Value '{value}' at '{path}' not in allowed values: {field.enum_values}",
                severity=IssueSeverity.MEDIUM,
                feature="schema_validation",
                component="api",
            ))

        # Numeric range validation
        if isinstance(value, (int, float)):
            if field.min_value is not None and value < field.min_value:
                issues.append(Issue(
                    title=f"Value below minimum: {field.name}",
                    description=f"Value {value} at '{path}' is below minimum {field.min_value}",
                    severity=IssueSeverity.MEDIUM,
                    feature="schema_validation",
                    component="api",
                ))
            if field.max_value is not None and value > field.max_value:
                issues.append(Issue(
                    title=f"Value above maximum: {field.name}",
                    description=f"Value {value} at '{path}' is above maximum {field.max_value}",
                    severity=IssueSeverity.MEDIUM,
                    feature="schema_validation",
                    component="api",
                ))

        # String validation
        if isinstance(value, str):
            if field.min_length is not None and len(value) < field.min_length:
                issues.append(Issue(
                    title=f"String too short: {field.name}",
                    description=f"Length {len(value)} at '{path}' is below minimum {field.min_length}",
                    severity=IssueSeverity.LOW,
                    feature="schema_validation",
                    component="api",
                ))
            if field.max_length is not None and len(value) > field.max_length:
                issues.append(Issue(
                    title=f"String too long: {field.name}",
                    description=f"Length {len(value)} at '{path}' exceeds maximum {field.max_length}",
                    severity=IssueSeverity.LOW,
                    feature="schema_validation",
                    component="api",
                ))
            if field.pattern:
                if not re.match(field.pattern, value):
                    issues.append(Issue(
                        title=f"Pattern mismatch: {field.name}",
                        description=f"Value '{value}' at '{path}' doesn't match pattern '{field.pattern}'",
                        severity=IssueSeverity.MEDIUM,
                        feature="schema_validation",
                        component="api",
                    ))

        # Array item validation
        if isinstance(value, list) and field.items_schema:
            for i, item in enumerate(value[:5]):  # Validate first 5 items
                issues.extend(SchemaValidator.validate(item, field.items_schema, f"{path}[{i}]"))

        # Nested object validation
        if isinstance(value, dict) and field.nested_schema:
            issues.extend(SchemaValidator.validate(value, field.nested_schema, path))

        return issues


class IssueSeverity(Enum):
    """Severity levels for detected issues"""
    CRITICAL = "critical"   # Blocks functionality, data loss
    HIGH = "high"           # Major bug, incorrect data
    MEDIUM = "medium"       # Minor bug, UI issue
    LOW = "low"             # Cosmetic, suggestion
    INFO = "info"           # Informational finding


@dataclass
class Issue:
    """Represents a detected issue"""
    title: str
    description: str
    severity: IssueSeverity
    feature: str
    component: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    screenshot_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    auto_fixable: bool = False
    fix_suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'feature': self.feature,
            'component': self.component,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'expected': self.expected,
            'actual': self.actual,
            'screenshot_path': self.screenshot_path,
            'timestamp': self.timestamp.isoformat(),
            'auto_fixable': self.auto_fixable,
            'fix_suggestion': self.fix_suggestion,
        }


@dataclass
class TestResult:
    """Result of a test execution"""
    test_name: str
    passed: bool
    issues: List[Issue] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSpecAgent(ABC):
    """
    Abstract base class for feature-specific SpecAgents.

    Each SpecAgent:
    1. Loads its spec files from .claude/specs/<feature>/
    2. Tests API endpoints against api-contract.yaml
    3. Tests UI with Playwright against ui-components.yaml
    4. Validates business logic against validation-rules.yaml
    5. Reports issues to the orchestrator
    """

    # Class-level configuration
    API_BASE_URL = "http://localhost:8002/api"
    FRONTEND_URL = "http://localhost:5173"
    SPECS_DIR = Path(".claude/specs")

    def __init__(
        self,
        feature_name: str,
        description: str,
        enable_browser: bool = True,
        enable_database: bool = True,
    ):
        """
        Initialize a SpecAgent

        Args:
            feature_name: Name of the feature (e.g., 'positions', 'dashboard')
            description: Human-readable description
            enable_browser: Whether to use Playwright for UI testing
            enable_database: Whether to enable database queries
        """
        self.feature_name = feature_name
        self.description = description
        self.enable_browser = enable_browser
        self.enable_database = enable_database

        # HTTP client (lazy init)
        self._http_client = None

        # Playwright (lazy init)
        self._browser = None
        self._page = None

        # Database connection (lazy init)
        self._db_pool = None

        # Loaded specs
        self._specs: Dict[str, Any] = {}

        # Test results
        self._results: List[TestResult] = []
        self._issues: List[Issue] = []

        logger.info(f"Initialized SpecAgent: {feature_name}")

    @property
    def tier(self) -> int:
        """Get the feature tier (1=Critical, 2=Important, 3=Additional)"""
        manifest = self.get_spec('manifest')
        if manifest and isinstance(manifest, dict):
            return manifest.get('tier', 3)
        return 3  # Default to tier 3 if not specified

    @property
    def priority(self) -> int:
        """Get the feature priority within its tier"""
        manifest = self.get_spec('manifest')
        if manifest and isinstance(manifest, dict):
            return manifest.get('priority', 1)
        return 1

    @property
    def spec(self) -> Optional[Dict[str, Any]]:
        """Get the loaded spec dictionary"""
        if not self._specs:
            self.load_specs()
        return self._specs if self._specs else None

    # ==================== Spec File Loading ====================

    def load_specs(self) -> Dict[str, Any]:
        """Load all spec files for this feature"""
        spec_dir = self.SPECS_DIR / self.feature_name

        if not spec_dir.exists():
            logger.warning(f"Spec directory not found: {spec_dir}")
            return {}

        spec_files = {
            'manifest': 'manifest.yaml',
            'requirements': 'requirements.md',
            'api_contract': 'api-contract.yaml',
            'ui_components': 'ui-components.yaml',
            'validation_rules': 'validation-rules.yaml',
        }

        for key, filename in spec_files.items():
            file_path = spec_dir / filename
            if file_path.exists():
                try:
                    if filename.endswith('.yaml'):
                        with open(file_path, 'r') as f:
                            self._specs[key] = yaml.safe_load(f)
                    elif filename.endswith('.md'):
                        with open(file_path, 'r') as f:
                            self._specs[key] = f.read()
                    logger.debug(f"Loaded spec: {filename}")
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")

        return self._specs

    def get_spec(self, key: str) -> Any:
        """Get a specific spec file content"""
        if not self._specs:
            self.load_specs()
        return self._specs.get(key)

    # ==================== HTTP Testing ====================

    async def _get_http_client(self) -> None:
        """Get or create httpx async client"""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=self.API_BASE_URL,
                timeout=30.0,
                follow_redirects=True,
            )
        return self._http_client

    async def http_get(self, path: str, **kwargs) -> 'httpx.Response':
        """Make HTTP GET request"""
        client = await self._get_http_client()
        return await client.get(path, **kwargs)

    async def http_post(self, path: str, **kwargs) -> 'httpx.Response':
        """Make HTTP POST request"""
        client = await self._get_http_client()
        return await client.post(path, **kwargs)

    async def test_endpoint(
        self,
        method: str,
        path: str,
        expected_status: int = 200,
        expected_fields: Optional[List[str]] = None,
        validate_response: Optional[callable] = None,
    ) -> TestResult:
        """
        Test an API endpoint

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            expected_status: Expected HTTP status code
            expected_fields: Fields that should be present in response
            validate_response: Custom validation function
        """
        import time
        start = time.time()
        issues = []

        try:
            client = await self._get_http_client()
            response = await client.request(method, path)

            # Check status code
            if response.status_code != expected_status:
                issues.append(Issue(
                    title=f"Unexpected status code: {path}",
                    description=f"Expected {expected_status}, got {response.status_code}",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="api",
                    expected=str(expected_status),
                    actual=str(response.status_code),
                ))

            # Check for expected fields
            if expected_fields and response.status_code == 200:
                try:
                    data = response.json()
                    for field in expected_fields:
                        if field not in data:
                            issues.append(Issue(
                                title=f"Missing field: {field}",
                                description=f"Response from {path} missing expected field '{field}'",
                                severity=IssueSeverity.MEDIUM,
                                feature=self.feature_name,
                                component="api",
                            ))
                except json.JSONDecodeError:
                    issues.append(Issue(
                        title=f"Invalid JSON response: {path}",
                        description="Response is not valid JSON",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="api",
                    ))

            # Custom validation
            if validate_response and response.status_code == 200:
                try:
                    custom_issues = validate_response(response)
                    if custom_issues:
                        issues.extend(custom_issues)
                except Exception as e:
                    logger.error(f"Custom validation failed: {e}")

            duration_ms = (time.time() - start) * 1000

            return TestResult(
                test_name=f"API {method} {path}",
                passed=len(issues) == 0,
                issues=issues,
                duration_ms=duration_ms,
                metadata={'status_code': response.status_code},
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            issues.append(Issue(
                title=f"API request failed: {path}",
                description=str(e),
                severity=IssueSeverity.CRITICAL,
                feature=self.feature_name,
                component="api",
            ))
            return TestResult(
                test_name=f"API {method} {path}",
                passed=False,
                issues=issues,
                duration_ms=duration_ms,
            )

    # ==================== Browser Testing (Playwright) ====================

    async def _get_browser(self) -> None:
        """Get or create Playwright browser"""
        if not self.enable_browser:
            raise RuntimeError("Browser testing is disabled for this agent")

        if self._browser is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._context = await self._browser.new_context(
                viewport={'width': 1920, 'height': 1080},
            )
            self._page = await self._context.new_page()

        return self._page

    async def navigate_to(self, path: str) -> bool:
        """Navigate browser to a path"""
        try:
            page = await self._get_browser()
            url = f"{self.FRONTEND_URL}{path}"
            await page.goto(url, wait_until='networkidle')
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False

    async def take_screenshot(self, name: str) -> Optional[str]:
        """Take a screenshot and return the path"""
        try:
            page = await self._get_browser()
            screenshot_dir = Path(".claude/orchestrator/continuous_qa/data/screenshots")
            screenshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.feature_name}_{name}_{timestamp}.png"
            filepath = screenshot_dir / filename

            await page.screenshot(path=str(filepath))
            return str(filepath)
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    async def test_element_exists(self, selector: str, description: str) -> TestResult:
        """Test that an element exists on the page"""
        import time
        start = time.time()
        issues = []

        try:
            page = await self._get_browser()
            element = await page.query_selector(selector)

            if element is None:
                screenshot = await self.take_screenshot(f"missing_{description.replace(' ', '_')}")
                issues.append(Issue(
                    title=f"Element not found: {description}",
                    description=f"Selector '{selector}' not found on page",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="ui",
                    screenshot_path=screenshot,
                ))

            duration_ms = (time.time() - start) * 1000
            return TestResult(
                test_name=f"Element exists: {description}",
                passed=len(issues) == 0,
                issues=issues,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            issues.append(Issue(
                title=f"Element test failed: {description}",
                description=str(e),
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="ui",
            ))
            return TestResult(
                test_name=f"Element exists: {description}",
                passed=False,
                issues=issues,
                duration_ms=duration_ms,
            )

    async def test_button_clickable(self, selector: str, description: str) -> TestResult:
        """Test that a button is clickable"""
        import time
        start = time.time()
        issues = []

        try:
            page = await self._get_browser()
            button = await page.query_selector(selector)

            if button is None:
                issues.append(Issue(
                    title=f"Button not found: {description}",
                    description=f"Button selector '{selector}' not found",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="ui",
                ))
            else:
                is_disabled = await button.is_disabled()
                if is_disabled:
                    issues.append(Issue(
                        title=f"Button disabled: {description}",
                        description=f"Button '{selector}' is unexpectedly disabled",
                        severity=IssueSeverity.MEDIUM,
                        feature=self.feature_name,
                        component="ui",
                    ))

            duration_ms = (time.time() - start) * 1000
            return TestResult(
                test_name=f"Button clickable: {description}",
                passed=len(issues) == 0,
                issues=issues,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            issues.append(Issue(
                title=f"Button test failed: {description}",
                description=str(e),
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="ui",
            ))
            return TestResult(
                test_name=f"Button clickable: {description}",
                passed=False,
                issues=issues,
                duration_ms=duration_ms,
            )

    # ==================== Database Testing ====================

    async def _get_db_pool(self) -> None:
        """Get database connection pool"""
        if not self.enable_database:
            raise RuntimeError("Database testing is disabled for this agent")

        if self._db_pool is None:
            try:
                from src.database.connection_pool import get_pool
                self._db_pool = await get_pool()
            except ImportError:
                import asyncpg
                import os
                self._db_pool = await asyncpg.create_pool(
                    os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/magnus'),
                    min_size=1,
                    max_size=5,
                )

        return self._db_pool

    async def db_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a database query"""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def db_query_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and return one row"""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    # ==================== Abstract Methods (Must Implement) ====================

    @abstractmethod
    async def test_api_endpoints(self) -> List[TestResult]:
        """
        Test all API endpoints for this feature.
        Must be implemented by each SpecAgent.
        """
        pass

    @abstractmethod
    async def test_ui_components(self) -> List[TestResult]:
        """
        Test UI components using Playwright.
        Must be implemented by each SpecAgent.
        """
        pass

    @abstractmethod
    async def test_business_logic(self) -> List[TestResult]:
        """
        Test business logic and calculations.
        Must be implemented by each SpecAgent.
        """
        pass

    @abstractmethod
    async def test_data_consistency(self) -> List[TestResult]:
        """
        Test data consistency across different views/APIs.
        Must be implemented by each SpecAgent.
        """
        pass

    # ==================== Main Execution ====================

    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests for this feature

        Returns:
            Dict with test results summary
        """
        import time
        start = time.time()

        all_results: List[TestResult] = []
        all_issues: List[Issue] = []

        # Load specs first
        self.load_specs()

        # Run each test suite
        test_suites = [
            ('api_endpoints', self.test_api_endpoints),
            ('ui_components', self.test_ui_components),
            ('business_logic', self.test_business_logic),
            ('data_consistency', self.test_data_consistency),
        ]

        for suite_name, test_func in test_suites:
            try:
                logger.info(f"[{self.feature_name}] Running {suite_name} tests...")
                results = await test_func()
                all_results.extend(results)

                for result in results:
                    all_issues.extend(result.issues)

            except Exception as e:
                logger.error(f"[{self.feature_name}] {suite_name} suite failed: {e}")
                all_issues.append(Issue(
                    title=f"Test suite failed: {suite_name}",
                    description=str(e),
                    severity=IssueSeverity.CRITICAL,
                    feature=self.feature_name,
                    component=suite_name,
                ))

        # Store results
        self._results = all_results
        self._issues = all_issues

        total_time = (time.time() - start) * 1000

        # Build summary
        summary = {
            'feature': self.feature_name,
            'total_tests': len(all_results),
            'passed': sum(1 for r in all_results if r.passed),
            'failed': sum(1 for r in all_results if not r.passed),
            'total_issues': len(all_issues),
            'critical_issues': sum(1 for i in all_issues if i.severity == IssueSeverity.CRITICAL),
            'high_issues': sum(1 for i in all_issues if i.severity == IssueSeverity.HIGH),
            'medium_issues': sum(1 for i in all_issues if i.severity == IssueSeverity.MEDIUM),
            'duration_ms': total_time,
            'timestamp': datetime.now().isoformat(),
            'issues': [i.to_dict() for i in all_issues],
        }

        logger.info(
            f"[{self.feature_name}] Tests complete: "
            f"{summary['passed']}/{summary['total_tests']} passed, "
            f"{summary['total_issues']} issues found"
        )

        return summary

    # ==================== Schema Validation ====================

    async def test_endpoint_with_schema(
        self,
        method: str,
        path: str,
        schema: ResponseSchema,
        expected_status: int = 200,
        **kwargs,
    ) -> TestResult:
        """
        Test an API endpoint with schema validation.

        Args:
            method: HTTP method
            path: API path
            schema: ResponseSchema defining expected structure
            expected_status: Expected HTTP status code
            **kwargs: Additional arguments passed to the request

        Returns:
            TestResult with schema validation issues
        """
        import time
        start = time.time()
        issues = []

        try:
            client = await self._get_http_client()
            response = await client.request(method, path, **kwargs)

            # Check status code
            if response.status_code != expected_status:
                issues.append(Issue(
                    title=f"Unexpected status code: {path}",
                    description=f"Expected {expected_status}, got {response.status_code}",
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="api",
                    expected=str(expected_status),
                    actual=str(response.status_code),
                ))
            elif response.status_code == 200:
                # Parse and validate response
                try:
                    data = response.json()

                    # Run schema validation
                    schema_issues = SchemaValidator.validate(data, schema, path)
                    for issue in schema_issues:
                        issue.feature = self.feature_name  # Set feature name
                    issues.extend(schema_issues)

                except json.JSONDecodeError:
                    issues.append(Issue(
                        title=f"Invalid JSON response: {path}",
                        description="Response is not valid JSON",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="api",
                    ))

            duration_ms = (time.time() - start) * 1000
            return TestResult(
                test_name=f"API {method} {path} (schema)",
                passed=len(issues) == 0,
                issues=issues,
                duration_ms=duration_ms,
                metadata={'status_code': response.status_code, 'schema_validated': True},
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            issues.append(Issue(
                title=f"API request failed: {path}",
                description=str(e),
                severity=IssueSeverity.CRITICAL,
                feature=self.feature_name,
                component="api",
            ))
            return TestResult(
                test_name=f"API {method} {path} (schema)",
                passed=False,
                issues=issues,
                duration_ms=duration_ms,
            )

    def define_schema(
        self,
        fields: List[Dict[str, Any]],
        array_response: bool = False,
        **kwargs,
    ) -> ResponseSchema:
        """
        Helper to define a schema from a simple dict format.

        Args:
            fields: List of field definitions as dicts
            array_response: Whether response is an array
            **kwargs: Additional ResponseSchema options

        Example:
            schema = self.define_schema([
                {'name': 'id', 'type': 'integer', 'required': True},
                {'name': 'symbol', 'type': 'string', 'pattern': r'^[A-Z]{1,5}$'},
                {'name': 'price', 'type': 'number', 'min': 0},
                {'name': 'status', 'type': 'string', 'enum': ['active', 'inactive']},
            ], array_response=True)
        """
        schema_fields = []
        type_map = {
            'string': SchemaType.STRING,
            'integer': SchemaType.INTEGER,
            'int': SchemaType.INTEGER,
            'number': SchemaType.NUMBER,
            'float': SchemaType.NUMBER,
            'boolean': SchemaType.BOOLEAN,
            'bool': SchemaType.BOOLEAN,
            'array': SchemaType.ARRAY,
            'object': SchemaType.OBJECT,
            'any': SchemaType.ANY,
        }

        for f in fields:
            field_type = type_map.get(f.get('type', 'any'), SchemaType.ANY)
            schema_fields.append(SchemaField(
                name=f['name'],
                field_type=field_type,
                required=f.get('required', True),
                nullable=f.get('nullable', False),
                min_value=f.get('min'),
                max_value=f.get('max'),
                min_length=f.get('min_length'),
                max_length=f.get('max_length'),
                pattern=f.get('pattern'),
                enum_values=f.get('enum'),
            ))

        return ResponseSchema(
            fields=schema_fields,
            array_response=array_response,
            **kwargs,
        )

    # ==================== Async Context Manager ====================

    async def __aenter__(self) -> 'BaseSpecAgent':
        """Enter async context - initialize resources"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context - cleanup resources"""
        await self.cleanup()

    @asynccontextmanager
    async def managed_browser(self) -> None:
        """Context manager for browser operations"""
        try:
            page = await self._get_browser()
            yield page
        finally:
            # Browser stays open for reuse, cleanup happens at agent cleanup
            pass

    @asynccontextmanager
    async def managed_db(self) -> None:
        """Context manager for database operations"""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            yield conn

    # ==================== Cleanup ====================

    async def cleanup(self) -> None:
        """Clean up resources"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._browser:
            await self._browser.close()
            await self._playwright.stop()
            self._browser = None
            self._page = None

        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None

        logger.debug(f"[{self.feature_name}] Resources cleaned up")

    def __repr__(self) -> str:
        return f"<SpecAgent:{self.feature_name}>"
