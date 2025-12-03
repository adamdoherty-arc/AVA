"""
Metrics Collection for Local Orchestrator
Tracks: Latency, throughput, success rate, token usage
100% Local - stores in SQLite
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict
import statistics
import sqlite3
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and aggregates orchestrator metrics locally
    - No cloud services
    - SQLite storage
    - Real-time aggregation
    - Prometheus-compatible export
    """

    def __init__(self, db_path: str = ".claude/orchestrator/databases/metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache for current session
        self.session_metrics = {
            "agent_executions": [],
            "latencies": defaultdict(list),
            "token_usage": defaultdict(int),
            "success_counts": defaultdict(int),
            "failure_counts": defaultdict(int),
            "errors": []
        }

        self._init_database()
        logger.info("Metrics collector initialized")

    def _init_database(self) -> None:
        """Initialize SQLite schema for metrics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Agent execution metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_name TEXT NOT NULL,
                duration_ms REAL,
                success BOOLEAN,
                tokens_used INTEGER DEFAULT 0,
                model_name TEXT,
                feature_name TEXT,
                error_message TEXT
            )
        """)

        # Aggregated metrics (hourly rollups)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_hourly (
                hour TIMESTAMP PRIMARY KEY,
                total_executions INTEGER,
                successful_executions INTEGER,
                avg_duration_ms REAL,
                total_tokens_used INTEGER,
                unique_agents_used INTEGER
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_timestamp
            ON agent_executions(agent_name, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON agent_executions(timestamp)
        """)

        conn.commit()
        conn.close()

    def record_agent_execution(
        self,
        agent_name: str,
        duration_ms: float,
        success: bool,
        tokens_used: int = 0,
        model_name: str = "local",
        feature_name: str = "unknown",
        error_message: str = None
    ):
        """Record an agent execution"""
        # Add to in-memory cache
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "duration_ms": duration_ms,
            "success": success,
            "tokens_used": tokens_used,
            "model_name": model_name,
            "feature_name": feature_name
        }

        self.session_metrics["agent_executions"].append(record)
        self.session_metrics["latencies"][agent_name].append(duration_ms)
        self.session_metrics["token_usage"][agent_name] += tokens_used

        if success:
            self.session_metrics["success_counts"][agent_name] += 1
        else:
            self.session_metrics["failure_counts"][agent_name] += 1
            if error_message:
                self.session_metrics["errors"].append({
                    "timestamp": datetime.now().isoformat(),
                    "agent": agent_name,
                    "error": error_message
                })

        # Persist to SQLite
        self._persist_execution(
            agent_name, duration_ms, success, tokens_used,
            model_name, feature_name, error_message
        )

    def _persist_execution(
        self,
        agent_name: str,
        duration_ms: float,
        success: bool,
        tokens_used: int,
        model_name: str,
        feature_name: str,
        error_message: str
    ):
        """Persist execution to SQLite"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO agent_executions
            (agent_name, duration_ms, success, tokens_used, model_name, feature_name, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (agent_name, duration_ms, success, tokens_used, model_name, feature_name, error_message))

        conn.commit()
        conn.close()

    def get_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for last N hours"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_executions,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                AVG(duration_ms) as avg_duration,
                SUM(tokens_used) as total_tokens,
                COUNT(DISTINCT agent_name) as unique_agents
            FROM agent_executions
            WHERE timestamp > datetime('now', '-' || ? || ' hours')
        """, (hours,))

        overall = cursor.fetchone()

        # Per-agent stats
        cursor.execute("""
            SELECT
                agent_name,
                COUNT(*) as executions,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                AVG(duration_ms) as avg_duration,
                SUM(tokens_used) as total_tokens
            FROM agent_executions
            WHERE timestamp > datetime('now', '-' || ? || ' hours')
            GROUP BY agent_name
            ORDER BY executions DESC
        """, (hours,))

        agent_stats = []
        for row in cursor.fetchall():
            agent_stats.append({
                "agent_name": row[0],
                "executions": row[1],
                "success_rate": round(row[2] / row[1] * 100, 2) if row[1] > 0 else 0,
                "avg_duration_ms": round(row[3], 2) if row[3] else 0,
                "total_tokens": row[4]
            })

        # Recent errors
        cursor.execute("""
            SELECT timestamp, agent_name, error_message
            FROM agent_executions
            WHERE error_message IS NOT NULL
            AND timestamp > datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
            LIMIT 10
        """, (hours,))

        recent_errors = [
            {
                "timestamp": row[0],
                "agent": row[1],
                "error": row[2]
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "period_hours": hours,
            "overall": {
                "total_executions": overall[0] or 0,
                "successful": overall[1] or 0,
                "success_rate": round((overall[1] or 0) / (overall[0] or 1) * 100, 2),
                "avg_duration_ms": round(overall[2], 2) if overall[2] else 0,
                "total_tokens": overall[3] or 0,
                "unique_agents": overall[4] or 0
            },
            "by_agent": agent_stats,
            "recent_errors": recent_errors
        }

    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trends over time"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                DATE(timestamp) as day,
                COUNT(*) as executions,
                AVG(duration_ms) as avg_duration,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            FROM agent_executions
            WHERE timestamp > datetime('now', '-' || ? || ' days')
            GROUP BY DATE(timestamp)
            ORDER BY day ASC
        """, (days,))

        trends = [
            {
                "day": row[0],
                "executions": row[1],
                "avg_duration_ms": round(row[2], 2) if row[2] else 0,
                "success_rate": round(row[3], 2) if row[3] else 0
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return {"days": days, "trends": trends}

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        summary = self.get_summary(hours=1)  # Last hour

        metrics = []

        # Overall metrics
        overall = summary["overall"]
        metrics.append(f"orchestrator_executions_total {overall['total_executions']}")
        metrics.append(f"orchestrator_success_rate {overall['success_rate'] / 100}")
        metrics.append(f"orchestrator_avg_duration_ms {overall['avg_duration_ms']}")
        metrics.append(f"orchestrator_tokens_total {overall['total_tokens']}")

        # Per-agent metrics
        for agent in summary["by_agent"]:
            name = agent["agent_name"].replace("-", "_")
            metrics.append(f'orchestrator_agent_executions{{agent="{agent["agent_name"]}"}} {agent["executions"]}')
            metrics.append(f'orchestrator_agent_success_rate{{agent="{agent["agent_name"]}"}} {agent["success_rate"] / 100}')
            metrics.append(f'orchestrator_agent_duration_ms{{agent="{agent["agent_name"]}"}} {agent["avg_duration_ms"]}')

        return "\n".join(metrics)

    def rollup_hourly_metrics(self) -> None:
        """Rollup metrics into hourly aggregates (run periodically)"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO metrics_hourly
            SELECT
                datetime(strftime('%Y-%m-%d %H:00:00', timestamp)) as hour,
                COUNT(*) as total_executions,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions,
                AVG(duration_ms) as avg_duration_ms,
                SUM(tokens_used) as total_tokens_used,
                COUNT(DISTINCT agent_name) as unique_agents_used
            FROM agent_executions
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY hour
        """)

        conn.commit()
        conn.close()
        logger.info("Hourly metrics rolled up")

    def cleanup_old_data(self, keep_days: int = 30):
        """Clean up old raw data (keep aggregates)"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM agent_executions
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (keep_days,))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted} old execution records")
        return deleted


# Singleton
_collector_instance: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get singleton metrics collector instance"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = MetricsCollector()
    return _collector_instance


if __name__ == "__main__":
    # Test metrics collector
    collector = get_metrics_collector()

    # Record some test executions
    collector.record_agent_execution(
        "test-agent", 150.5, True, tokens_used=200,
        model_name="llama3.2", feature_name="test"
    )
    collector.record_agent_execution(
        "test-agent", 200.3, True, tokens_used=150,
        model_name="llama3.2", feature_name="test"
    )
    collector.record_agent_execution(
        "test-agent-2", 100.0, False, tokens_used=50,
        model_name="mistral", feature_name="test",
        error_message="Test error"
    )

    # Get summary
    summary = collector.get_summary()
    print("\nMetrics Summary:")
    print(json.dumps(summary, indent=2))

    # Get Prometheus format
    print("\nPrometheus Format:")
    print(collector.get_prometheus_metrics())

    print("\nMetrics collector test complete!")
