"""
Modern Observability Infrastructure
====================================

Production-grade observability with:
- Structured logging (structlog)
- Distributed tracing (OpenTelemetry)
- Metrics collection (Prometheus)
- Health checks and readiness probes
- Request correlation IDs
- Performance monitoring

Author: AVA Trading Platform
Updated: 2025-11-29
"""

import asyncio
import functools
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

# Structured logging
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

# Metrics
try:
    from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest
    from prometheus_client.core import CollectorRegistry

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Tracing
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Span = None
    SpanKind = None

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ObservabilityConfig:
    """Configuration for observability components."""

    # Service info
    service_name: str = "ava-trading-platform"
    service_version: str = "2.0.0"
    environment: str = "development"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    # Tracing
    tracing_enabled: bool = True
    otlp_endpoint: str = "http://localhost:4317"
    trace_sample_rate: float = 1.0  # 100% sampling

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Create config from environment."""
        return cls(
            service_name=os.getenv("SERVICE_NAME", "ava-trading-platform"),
            service_version=os.getenv("SERVICE_VERSION", "2.0.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            tracing_enabled=os.getenv("TRACING_ENABLED", "true").lower() == "true",
            otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4317"),
            trace_sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "1.0")),
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
        )


# =============================================================================
# Structured Logging Setup
# =============================================================================


def setup_logging(config: Optional[ObservabilityConfig] = None) -> None:
    """
    Configure structured logging with correlation ID support.

    This sets up structlog with processors for:
    - Timestamp addition
    - Log level handling
    - Correlation ID tracking
    - Exception formatting
    - JSON or console output
    """
    import logging as stdlib_logging

    config = config or ObservabilityConfig.from_env()

    # Map log level string to logging module constant
    log_level = getattr(stdlib_logging, config.log_level.upper(), stdlib_logging.INFO)

    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.log_format == "json":
        # JSON output for production
        structlog.configure(
            processors=shared_processors
            + [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Console output for development
        structlog.configure(
            processors=shared_processors
            + [
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a bound logger with the given name."""
    return structlog.get_logger(name)


# =============================================================================
# Distributed Tracing
# =============================================================================


class TracingManager:
    """
    Manages distributed tracing with OpenTelemetry.

    Features:
    - Automatic span creation
    - Context propagation
    - Custom attributes
    - Exception recording
    """

    _instance: Optional["TracingManager"] = None
    _tracer: Optional["trace.Tracer"] = None

    def __new__(cls) -> "TracingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, config: Optional[ObservabilityConfig] = None) -> None:
        """Initialize OpenTelemetry tracing."""
        if not OTEL_AVAILABLE:
            return

        config = config or ObservabilityConfig.from_env()

        if not config.tracing_enabled:
            return

        # Create resource with service info
        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.version": config.service_version,
                "deployment.environment": config.environment,
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter
        try:
            exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception as e:
            structlog.get_logger().warning(
                "otlp_exporter_failed",
                error=str(e),
                endpoint=config.otlp_endpoint,
            )

        # Set as global tracer provider
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(config.service_name)

    @property
    def tracer(self) -> Optional["trace.Tracer"]:
        """Get the tracer instance."""
        return self._tracer

    @contextmanager
    def span(
        self,
        name: str,
        kind: Optional["SpanKind"] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a trace span context manager.

        Usage:
            with tracing.span("process_order", attributes={"order_id": "123"}):
                # Do work
        """
        if not self._tracer:
            yield None
            return

        kind = kind or SpanKind.INTERNAL
        with self._tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
            except Exception as e:
                if span:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        kind: Optional["SpanKind"] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Async version of span context manager."""
        if not self._tracer:
            yield None
            return

        kind = kind or SpanKind.INTERNAL
        with self._tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
            except Exception as e:
                if span:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


# Global tracing manager
tracing = TracingManager()


def traced(
    name: Optional[str] = None,
    kind: Optional["SpanKind"] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to automatically trace a function.

    Usage:
        @traced("process_order")
        async def process_order(order_id: str):
            ...
    """

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with tracing.async_span(span_name, kind, attributes):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracing.span(span_name, kind, attributes):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Metrics Collection
# =============================================================================


class MetricsCollector:
    """
    Prometheus metrics collector for application monitoring.

    Provides:
    - Request counters
    - Latency histograms
    - Active connection gauges
    - Business metrics
    """

    _instance: Optional["MetricsCollector"] = None

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, config: Optional[ObservabilityConfig] = None) -> None:
        """Initialize metrics collectors."""
        if not PROMETHEUS_AVAILABLE or self._initialized:
            return

        config = config or ObservabilityConfig.from_env()

        if not config.metrics_enabled:
            return

        # Request metrics
        self.request_count = Counter(
            "ava_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )

        self.request_latency = Histogram(
            "ava_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # Database metrics
        self.db_query_count = Counter(
            "ava_db_queries_total",
            "Total database queries",
            ["operation", "status"],
        )

        self.db_query_latency = Histogram(
            "ava_db_query_duration_seconds",
            "Database query latency",
            ["operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        self.db_pool_size = Gauge(
            "ava_db_pool_connections",
            "Database connection pool size",
            ["state"],  # "active", "idle", "total"
        )

        # Cache metrics
        self.cache_hits = Counter(
            "ava_cache_hits_total",
            "Total cache hits",
            ["cache_name"],
        )

        self.cache_misses = Counter(
            "ava_cache_misses_total",
            "Total cache misses",
            ["cache_name"],
        )

        # AI/LLM metrics
        self.llm_requests = Counter(
            "ava_llm_requests_total",
            "Total LLM requests",
            ["provider", "model", "status"],
        )

        self.llm_latency = Histogram(
            "ava_llm_request_duration_seconds",
            "LLM request latency",
            ["provider", "model"],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )

        self.llm_tokens = Counter(
            "ava_llm_tokens_total",
            "Total LLM tokens used",
            ["provider", "model", "type"],  # token type: "input" or "output"
        )

        # Trading metrics
        self.trades_executed = Counter(
            "ava_trades_executed_total",
            "Total trades executed",
            ["strategy", "status"],
        )

        self.portfolio_value = Gauge(
            "ava_portfolio_value_dollars",
            "Current portfolio value",
            ["account"],
        )

        self.positions_count = Gauge(
            "ava_positions_count",
            "Number of open positions",
            ["type"],  # "stock", "option"
        )

        # Sports betting metrics
        self.bets_placed = Counter(
            "ava_bets_placed_total",
            "Total bets placed",
            ["sport", "bet_type"],
        )

        self.prediction_accuracy = Gauge(
            "ava_prediction_accuracy",
            "AI prediction accuracy",
            ["sport", "market_type"],
        )

        # System metrics
        self.active_websockets = Gauge(
            "ava_websocket_connections",
            "Active WebSocket connections",
        )

        self.background_tasks = Gauge(
            "ava_background_tasks",
            "Running background tasks",
            ["task_name"],
        )

        # Service info
        self.service_info = Info(
            "ava_service",
            "Service information",
        )
        self.service_info.info(
            {
                "version": config.service_version,
                "environment": config.environment,
            }
        )

        self._initialized = True

    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float,
    ) -> None:
        """Record an HTTP request."""
        if not PROMETHEUS_AVAILABLE or not hasattr(self, "request_count"):
            return
        self.request_count.labels(
            method=method, endpoint=endpoint, status=str(status)
        ).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(
            duration
        )

    def record_db_query(
        self,
        operation: str,
        duration: float,
        success: bool = True,
    ) -> None:
        """Record a database query."""
        if not PROMETHEUS_AVAILABLE or not hasattr(self, "db_query_count"):
            return
        status = "success" if success else "error"
        self.db_query_count.labels(operation=operation, status=status).inc()
        self.db_query_latency.labels(operation=operation).observe(duration)

    def record_cache(self, cache_name: str, hit: bool) -> None:
        """Record a cache hit or miss."""
        if not PROMETHEUS_AVAILABLE:
            return
        if hit:
            self.cache_hits.labels(cache_name=cache_name).inc()
        else:
            self.cache_misses.labels(cache_name=cache_name).inc()

    def record_llm_request(
        self,
        provider: str,
        model: str,
        duration: float,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
    ) -> None:
        """Record an LLM request."""
        if not PROMETHEUS_AVAILABLE or not hasattr(self, "llm_requests"):
            return
        status = "success" if success else "error"
        self.llm_requests.labels(
            provider=provider, model=model, status=status
        ).inc()
        self.llm_latency.labels(provider=provider, model=model).observe(duration)
        self.llm_tokens.labels(
            provider=provider, model=model, type="input"
        ).inc(input_tokens)
        self.llm_tokens.labels(
            provider=provider, model=model, type="output"
        ).inc(output_tokens)


# Global metrics collector
metrics = MetricsCollector()


# =============================================================================
# Request Correlation
# =============================================================================


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return str(uuid.uuid4())


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """
    Context manager for request correlation.

    Usage:
        with correlation_context(request_id):
            logger.info("Processing request")  # Includes correlation_id
    """
    cid = correlation_id or generate_correlation_id()
    bind_contextvars(correlation_id=cid)
    try:
        yield cid
    finally:
        clear_contextvars()


# =============================================================================
# Health Checks
# =============================================================================


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Complete health report."""

    status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class HealthChecker:
    """
    Health check manager for service health monitoring.

    Features:
    - Liveness checks (is the service alive?)
    - Readiness checks (is the service ready to handle requests?)
    - Custom health check registration
    """

    def __init__(self):
        self._checks: Dict[str, Callable[[], Awaitable[HealthCheck]]] = {}
        self._logger = get_logger("health_checker")

    def register(
        self,
        name: str,
        check: Callable[[], Awaitable[HealthCheck]],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check

    async def check_database(self) -> HealthCheck:
        """Check database connectivity."""
        try:
            from backend.infrastructure.database import get_database

            start = time.perf_counter()
            db = await get_database()
            result = await db.fetchval("SELECT 1")
            latency = (time.perf_counter() - start) * 1000

            if result == 1:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="PostgreSQL connection OK",
                    latency_ms=round(latency, 2),
                    details=db.get_stats()["pool_stats"],
                )
            else:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Unexpected response from database",
                )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}",
            )

    async def check_redis(self) -> HealthCheck:
        """Check Redis connectivity."""
        try:
            from backend.infrastructure.cache import get_cache

            start = time.perf_counter()
            cache = get_cache()
            stats = cache.get_stats()
            latency = (time.perf_counter() - start) * 1000

            if stats.get("connected"):
                return HealthCheck(
                    name="redis",
                    status=HealthStatus.HEALTHY,
                    message="Redis connection OK",
                    latency_ms=round(latency, 2),
                    details=stats,
                )
            else:
                return HealthCheck(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message="Using in-memory cache fallback",
                    details=stats,
                )
        except Exception as e:
            return HealthCheck(
                name="redis",
                status=HealthStatus.DEGRADED,
                message=f"Redis unavailable: {str(e)}",
            )

    async def run_all(self) -> HealthReport:
        """Run all registered health checks."""
        # Default checks
        default_checks = [
            self.check_database(),
            self.check_redis(),
        ]

        # Custom checks
        custom_checks = [check() for check in self._checks.values()]

        # Run all checks concurrently
        results = await asyncio.gather(
            *default_checks, *custom_checks, return_exceptions=True
        )

        checks = []
        for result in results:
            if isinstance(result, Exception):
                checks.append(
                    HealthCheck(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=str(result),
                    )
                )
            else:
                checks.append(result)

        # Determine overall status
        statuses = [c.status for c in checks]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return HealthReport(status=overall, checks=checks)

    async def liveness(self) -> bool:
        """Simple liveness check - is the service running?"""
        return True

    async def readiness(self) -> HealthReport:
        """Readiness check - is the service ready to handle requests?"""
        return await self.run_all()


# Global health checker
health_checker = HealthChecker()


# =============================================================================
# Initialization
# =============================================================================


def setup_observability(config: Optional[ObservabilityConfig] = None) -> None:
    """Initialize all observability components."""
    config = config or ObservabilityConfig.from_env()

    # Setup logging first
    setup_logging(config)

    # Initialize tracing
    tracing.initialize(config)

    # Initialize metrics
    metrics.initialize(config)

    structlog.get_logger().info(
        "observability_initialized",
        tracing_enabled=config.tracing_enabled,
        metrics_enabled=config.metrics_enabled,
        environment=config.environment,
    )
