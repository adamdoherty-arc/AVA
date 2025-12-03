"""
OpenTelemetry Tracer for Local Orchestrator
100% Free, no cloud services required
Exports to local files and console
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor
)
from opentelemetry.sdk.resources import Resource
from typing import Optional, Dict, Any, List
import logging
import json
from pathlib import Path
from datetime import datetime
import sqlite3

logger = logging.getLogger(__name__)


class SQLiteSpanExporter:
    """Export spans to local SQLite database"""

    def __init__(self, db_path: str = ".claude/orchestrator/databases/traces.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT,
                parent_span_id TEXT,
                name TEXT,
                start_time REAL,
                end_time REAL,
                duration_ms REAL,
                attributes TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trace_id ON spans(trace_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON spans(created_at)
        """)
        conn.commit()
        conn.close()

    def export(self, spans: List) -> None:
        """Export spans to SQLite"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        for span in spans:
            try:
                span_context = span.get_span_context()
                duration_ms = (span.end_time - span.start_time) / 1_000_000  # Convert ns to ms

                cursor.execute("""
                    INSERT OR REPLACE INTO spans
                    (span_id, trace_id, parent_span_id, name, start_time, end_time,
                     duration_ms, attributes, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    format(span_context.span_id, '016x'),
                    format(span_context.trace_id, '032x'),
                    format(span.parent.span_id, '016x') if span.parent else None,
                    span.name,
                    span.start_time / 1_000_000_000,  # Convert ns to seconds
                    span.end_time / 1_000_000_000,
                    duration_ms,
                    json.dumps(dict(span.attributes)) if span.attributes else '{}',
                    str(span.status)
                ))
            except Exception as e:
                logger.error(f"Failed to export span: {e}")

        conn.commit()
        conn.close()

    def shutdown(self) -> None:
        """Shutdown the exporter (required by OpenTelemetry)"""
        # No persistent connections to close for SQLite
        pass


class OrchestratorTracer:
    """
    Main tracer for orchestrator operations
    - Traces all agent executions
    - Tracks state transitions
    - Records tool calls
    - Exports to local SQLite + console
    """

    def __init__(self, service_name: str = "magnus-orchestrator"):
        # Create resource with service information
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "2.0",
            "deployment.environment": "local",
            "host.name": "localhost"
        })

        # Initialize tracer provider
        self.provider = TracerProvider(resource=resource)

        # Add console exporter for development
        console_exporter = ConsoleSpanExporter()
        self.provider.add_span_processor(
            SimpleSpanProcessor(console_exporter)
        )

        # Add SQLite exporter for persistence
        sqlite_exporter = SQLiteSpanExporter()
        self.provider.add_span_processor(
            SimpleSpanProcessor(sqlite_exporter)
        )

        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(__name__)

        logger.info(f"OpenTelemetry tracer initialized for {service_name}")

    def start_agent_execution(self, agent_name: str, context: Dict[str, Any]) -> trace.Span:
        """Start tracing an agent execution"""
        span = self.tracer.start_span(
            f"agent.execute.{agent_name}",
            attributes={
                "agent.name": agent_name,
                "agent.priority": str(context.get("priority", "unknown")),
                "feature.name": str(context.get("feature", "unknown")),
                "request.id": str(context.get("request_id", "unknown")),
                "model.name": str(context.get("model", "local")),
            }
        )
        return span

    def end_agent_execution(self, span: trace.Span, success: bool,
                           tokens_used: int = 0, error: str = None):
        """End agent execution span"""
        span.set_attribute("success", success)
        span.set_attribute("tokens.used", tokens_used)
        if error:
            span.set_attribute("error", error)
            span.set_status(trace.Status(trace.StatusCode.ERROR, error))
        else:
            span.set_status(trace.Status(trace.StatusCode.OK))
        span.end()

    def trace_state_transition(self, from_state: str, to_state: str,
                               context: Dict[str, Any]):
        """Trace a state machine transition"""
        with self.tracer.start_as_current_span(
            f"state.transition.{from_state}_to_{to_state}",
            attributes={
                "state.from": from_state,
                "state.to": to_state,
                "transition.valid": str(context.get("valid", True)),
                "context.keys": str(list(context.keys()))
            }
        ):
            pass

    def trace_tool_call(self, tool_name: str, parameters: Dict[str, Any],
                       result: Any = None, duration_ms: float = 0):
        """Trace a tool/MCP call"""
        with self.tracer.start_as_current_span(
            f"tool.call.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "tool.parameter_count": len(parameters),
                "tool.duration_ms": duration_ms,
                "tool.result_type": type(result).__name__ if result else "none"
            }
        ):
            pass

    def trace_error(self, error: Exception, context: Dict[str, Any]):
        """Trace an error occurrence"""
        with self.tracer.start_as_current_span(
            "error.occurred",
            attributes={
                "error.type": type(error).__name__,
                "error.message": str(error),
                "error.context": str(context)
            }
        ) as span:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))

    def trace_spec_validation(self, spec_type: str, feature_name: str,
                              valid: bool, violations: List[str] = None):
        """Trace spec validation"""
        with self.tracer.start_as_current_span(
            f"spec.validate.{spec_type}",
            attributes={
                "spec.type": spec_type,
                "feature.name": feature_name,
                "validation.valid": valid,
                "validation.violation_count": len(violations) if violations else 0
            }
        ):
            pass

    def get_trace_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get trace summary from SQLite"""
        db_path = ".claude/orchestrator/databases/traces.db"
        if not Path(db_path).exists():
            return {"message": "No traces yet"}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get summary stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_spans,
                AVG(duration_ms) as avg_duration,
                MAX(duration_ms) as max_duration,
                COUNT(DISTINCT trace_id) as total_traces
            FROM spans
            WHERE created_at > datetime('now', '-' || ? || ' hours')
        """, (hours,))

        stats = cursor.fetchone()
        conn.close()

        return {
            "total_spans": stats[0],
            "avg_duration_ms": round(stats[1], 2) if stats[1] else 0,
            "max_duration_ms": round(stats[2], 2) if stats[2] else 0,
            "total_traces": stats[3],
            "period_hours": hours
        }


# Singleton instance
_tracer_instance: Optional[OrchestratorTracer] = None


def get_tracer() -> OrchestratorTracer:
    """Get singleton tracer instance"""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = OrchestratorTracer()
    return _tracer_instance


if __name__ == "__main__":
    # Test the tracer
    tracer = get_tracer()

    # Test agent execution
    span = tracer.start_agent_execution(
        "test-agent",
        {"priority": "high", "feature": "test", "request_id": "test-123"}
    )
    tracer.end_agent_execution(span, success=True, tokens_used=150)

    # Test state transition
    tracer.trace_state_transition("idle", "executing", {"valid": True})

    # Test tool call
    tracer.trace_tool_call("test_tool", {"param1": "value1"}, result="success", duration_ms=50)

    # Get summary
    summary = tracer.get_trace_summary()
    print(f"\nTrace Summary: {json.dumps(summary, indent=2)}")
    print("\nTracer test complete!")
