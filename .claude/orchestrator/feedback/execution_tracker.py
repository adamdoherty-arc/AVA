"""
Execution Tracker for Self-Healing System
Tracks all agent executions for pattern analysis
100% Local - SQLite storage
"""
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ExecutionTracker:
    """
    Tracks all agent executions with full context
    - Stores execution details in SQLite
    - Enables pattern analysis
    - Supports self-healing decisions
    """

    def __init__(self, db_path: str = ".claude/orchestrator/databases/execution_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("Execution tracker initialized")

    def _init_database(self) -> None:
        """Initialize SQLite schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Main executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_name TEXT NOT NULL,
                feature_name TEXT,
                request_text TEXT,
                success BOOLEAN NOT NULL,
                duration_ms REAL,
                tokens_used INTEGER,
                model_name TEXT,
                error_message TEXT,
                error_type TEXT,
                context_json TEXT,
                files_modified TEXT,
                retry_count INTEGER DEFAULT 0,
                parent_execution_id TEXT
            )
        """)

        # Patterns table (learned patterns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.5,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_timestamp ON executions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_agent ON executions(agent_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exec_success ON executions(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON learned_patterns(pattern_type)")

        conn.commit()
        conn.close()

    def track_execution(
        self,
        execution_id: str,
        agent_name: str,
        success: bool,
        duration_ms: float,
        request_text: str = None,
        feature_name: str = None,
        tokens_used: int = 0,
        model_name: str = "local",
        error_message: str = None,
        error_type: str = None,
        context: Dict[str, Any] = None,
        files_modified: List[str] = None,
        retry_count: int = 0,
        parent_execution_id: str = None
    ):
        """Track a single execution"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO executions (
                execution_id, agent_name, feature_name, request_text,
                success, duration_ms, tokens_used, model_name,
                error_message, error_type, context_json, files_modified,
                retry_count, parent_execution_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution_id,
            agent_name,
            feature_name,
            request_text,
            success,
            duration_ms,
            tokens_used,
            model_name,
            error_message,
            error_type,
            json.dumps(context) if context else None,
            json.dumps(files_modified) if files_modified else None,
            retry_count,
            parent_execution_id
        ))

        conn.commit()
        conn.close()

        logger.debug(f"Tracked execution: {execution_id} - Success: {success}")

    def get_recent_failures(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent failed executions"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT execution_id, agent_name, feature_name, request_text,
                   error_message, error_type, context_json, timestamp
            FROM executions
            WHERE success = 0
            AND timestamp > datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
            LIMIT ?
        """, (hours, limit))

        failures = []
        for row in cursor.fetchall():
            failures.append({
                "execution_id": row[0],
                "agent_name": row[1],
                "feature_name": row[2],
                "request_text": row[3],
                "error_message": row[4],
                "error_type": row[5],
                "context": json.loads(row[6]) if row[6] else {},
                "timestamp": row[7]
            })

        conn.close()
        return failures

    def get_success_patterns(self, agent_name: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent successful executions for pattern learning"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if agent_name:
            cursor.execute("""
                SELECT execution_id, agent_name, feature_name, request_text,
                       duration_ms, context_json, files_modified
                FROM executions
                WHERE success = 1 AND agent_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (agent_name, limit))
        else:
            cursor.execute("""
                SELECT execution_id, agent_name, feature_name, request_text,
                       duration_ms, context_json, files_modified
                FROM executions
                WHERE success = 1
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        successes = []
        for row in cursor.fetchall():
            successes.append({
                "execution_id": row[0],
                "agent_name": row[1],
                "feature_name": row[2],
                "request_text": row[3],
                "duration_ms": row[4],
                "context": json.loads(row[5]) if row[5] else {},
                "files_modified": json.loads(row[6]) if row[6] else []
            })

        conn.close()
        return successes

    def store_learned_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        confidence: float = 0.5
    ):
        """Store a learned pattern"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO learned_patterns (pattern_type, pattern_data, confidence)
            VALUES (?, ?, ?)
        """, (pattern_type, json.dumps(pattern_data), confidence))

        conn.commit()
        conn.close()

        logger.info(f"Stored learned pattern: {pattern_type}")

    def get_learned_patterns(self, pattern_type: str = None) -> List[Dict[str, Any]]:
        """Get learned patterns"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if pattern_type:
            cursor.execute("""
                SELECT id, pattern_type, pattern_data, success_count,
                       failure_count, confidence, last_updated
                FROM learned_patterns
                WHERE pattern_type = ?
                ORDER BY confidence DESC
            """, (pattern_type,))
        else:
            cursor.execute("""
                SELECT id, pattern_type, pattern_data, success_count,
                       failure_count, confidence, last_updated
                FROM learned_patterns
                ORDER BY confidence DESC
            """)

        patterns = []
        for row in cursor.fetchall():
            patterns.append({
                "id": row[0],
                "pattern_type": row[1],
                "pattern_data": json.loads(row[2]),
                "success_count": row[3],
                "failure_count": row[4],
                "confidence": row[5],
                "last_updated": row[6]
            })

        conn.close()
        return patterns

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get execution statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                AVG(duration_ms) as avg_duration,
                COUNT(DISTINCT agent_name) as unique_agents,
                SUM(retry_count) as total_retries
            FROM executions
            WHERE timestamp > datetime('now', '-' || ? || ' hours')
        """, (hours,))

        stats = cursor.fetchone()
        conn.close()

        total = stats[0] or 0
        successes = stats[1] or 0

        return {
            "total_executions": total,
            "successful": successes,
            "failed": total - successes,
            "success_rate": (successes / total * 100) if total > 0 else 0,
            "avg_duration_ms": round(stats[2], 2) if stats[2] else 0,
            "unique_agents": stats[3] or 0,
            "total_retries": stats[4] or 0,
            "period_hours": hours
        }


# Singleton
_tracker_instance: Optional[ExecutionTracker] = None


def get_execution_tracker() -> ExecutionTracker:
    """Get singleton execution tracker"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ExecutionTracker()
    return _tracker_instance


if __name__ == "__main__":
    # Test execution tracker
    import uuid

    tracker = get_execution_tracker()

    # Track some test executions
    tracker.track_execution(
        execution_id=str(uuid.uuid4()),
        agent_name="test-agent",
        success=True,
        duration_ms=150.5,
        request_text="Test request",
        tokens_used=200
    )

    tracker.track_execution(
        execution_id=str(uuid.uuid4()),
        agent_name="test-agent",
        success=False,
        duration_ms=50.0,
        error_message="Test error",
        error_type="TestError"
    )

    # Get statistics
    stats = tracker.get_statistics()
    print(f"\nExecution Statistics: {json.dumps(stats, indent=2)}")

    # Get failures
    failures = tracker.get_recent_failures()
    print(f"\nRecent Failures: {len(failures)}")

    print("\nExecution tracker test complete!")
