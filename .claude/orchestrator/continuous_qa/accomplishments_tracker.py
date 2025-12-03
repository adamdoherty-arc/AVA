"""
ADDITIVE-ONLY Accomplishments Tracker

CRITICAL: This file uses JSONL (JSON Lines) format and ONLY APPENDS.
It NEVER overwrites or modifies existing entries.

Each line is a complete, independent JSON record.
"""

import json
import os
import sys
from pathlib import Path

# Cross-platform file locking
if sys.platform == 'win32':
    import msvcrt
    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl
    def lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Accomplishment:
    """Single accomplishment entry."""
    timestamp: str
    run_id: str
    category: str  # auto_fix, issue_found, health_check, learning, enhancement
    module: str  # Which check module generated this
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    impact_score: int = 1  # 1-10, how significant
    severity: str = "info"  # info, warning, error, critical
    files_affected: List[str] = field(default_factory=list)
    fix_applied: bool = False

    def to_json_line(self) -> str:
        """Convert to JSON line (single line, no trailing newline)."""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(',', ':'))

    @classmethod
    def from_json_line(cls, line: str) -> 'Accomplishment':
        """Parse from JSON line."""
        data = json.loads(line.strip())
        return cls(**data)


@dataclass
class RunSummary:
    """Summary of a single QA run."""
    run_id: str
    timestamp: str
    duration_seconds: float
    checks_performed: int
    issues_found: int
    issues_fixed: int
    health_score_before: float
    health_score_after: float
    critical_failures: int = 0

    def to_json_line(self) -> str:
        """Convert to JSON line."""
        data = asdict(self)
        data['_type'] = 'run_summary'
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


class AccomplishmentsTracker:
    """
    Manages ADDITIVE accomplishments log.

    CRITICAL DESIGN PRINCIPLE: This class ONLY APPENDS to files.
    It NEVER overwrites, truncates, or modifies existing content.

    File format: JSONL (JSON Lines)
    - Each line is a complete JSON object
    - Append-only operations
    - Corruption-resistant (one bad line doesn't break file)
    - Easy to parse line-by-line
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize tracker with data directory."""
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.accomplishments_path = self.data_dir / "accomplishments.jsonl"
        self.status_path = self.data_dir / "status.json"

        # Track current run
        self._current_run_id: Optional[str] = None
        self._run_start_time: Optional[datetime] = None

    def start_run(self) -> str:
        """Start a new QA run and return run_id."""
        self._current_run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._run_start_time = datetime.now()

        # Update status file (this one can be overwritten - it's just current state)
        self._update_status({
            "service_status": "running",
            "is_currently_running": True,
            "current_run_id": self._current_run_id,
            "run_started_at": self._run_start_time.isoformat(),
        })

        return self._current_run_id

    def end_run(self, summary: RunSummary):
        """End current run and log summary."""
        # Append summary to accomplishments log
        self._append_line(summary.to_json_line())

        # Update status file
        self._update_status({
            "service_status": "running",
            "is_currently_running": False,
            "last_run": summary.timestamp,
            "last_run_id": summary.run_id,
            "next_run": (datetime.now() + timedelta(minutes=20)).isoformat(),
            "current_cycle": self._get_cycle_count() + 1,
            "last_accomplishments_count": summary.issues_fixed,
            "health_score": summary.health_score_after,
        })

        self._current_run_id = None
        self._run_start_time = None

    def log(self, entry: Accomplishment):
        """
        APPEND an accomplishment to the log.

        CRITICAL: This method ONLY APPENDS. It NEVER modifies existing content.
        """
        # Ensure timestamp is set
        if not entry.timestamp:
            entry.timestamp = datetime.now().isoformat()

        # Ensure run_id is set
        if not entry.run_id and self._current_run_id:
            entry.run_id = self._current_run_id

        # Append to file
        self._append_line(entry.to_json_line())

        logger.info(f"Logged accomplishment: [{entry.category}] {entry.message}")

    def log_auto_fix(self, module: str, message: str, files: List[str],
                     details: Dict = None, impact: int = 3):
        """Convenience method to log an auto-fix accomplishment."""
        self.log(Accomplishment(
            timestamp=datetime.now().isoformat(),
            run_id=self._current_run_id or "MANUAL",
            category="auto_fix",
            module=module,
            message=message,
            details=details or {},
            impact_score=impact,
            severity="info",
            files_affected=files,
            fix_applied=True,
        ))

    def log_issue(self, module: str, message: str, severity: str = "warning",
                  details: Dict = None, files: List[str] = None):
        """Convenience method to log an issue found."""
        self.log(Accomplishment(
            timestamp=datetime.now().isoformat(),
            run_id=self._current_run_id or "MANUAL",
            category="issue_found",
            module=module,
            message=message,
            details=details or {},
            impact_score=5 if severity in ("error", "critical") else 2,
            severity=severity,
            files_affected=files or [],
            fix_applied=False,
        ))

    def log_enhancement(self, module: str, message: str, files: List[str],
                        details: Dict = None, impact: int = 5):
        """Convenience method to log a proactive enhancement."""
        self.log(Accomplishment(
            timestamp=datetime.now().isoformat(),
            run_id=self._current_run_id or "MANUAL",
            category="enhancement",
            module=module,
            message=message,
            details=details or {},
            impact_score=impact,
            severity="info",
            files_affected=files,
            fix_applied=True,
        ))

    def log_learning(self, pattern_id: str, description: str,
                     occurrences: int, prevention_suggestions: List[str]):
        """Log a learned pattern for future prevention."""
        self.log(Accomplishment(
            timestamp=datetime.now().isoformat(),
            run_id=self._current_run_id or "MANUAL",
            category="learning",
            module="pattern_learner",
            message=f"Learned pattern: {description}",
            details={
                "pattern_id": pattern_id,
                "occurrences": occurrences,
                "prevention_suggestions": prevention_suggestions,
            },
            impact_score=4,
            severity="info",
            files_affected=[],
            fix_applied=False,
        ))

    def get_recent(self, hours: int = 24) -> List[Accomplishment]:
        """
        Read recent accomplishments without modifying file.

        This is a READ-ONLY operation.
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        results = []

        if not self.accomplishments_path.exists():
            return results

        with open(self.accomplishments_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Skip run summaries
                    if data.get('_type') == 'run_summary':
                        continue
                    # Check timestamp
                    entry_time = datetime.fromisoformat(data.get('timestamp', ''))
                    if entry_time >= cutoff:
                        results.append(Accomplishment(**{
                            k: v for k, v in data.items() if k != '_type'
                        }))
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    # Skip malformed lines - don't let one bad line break everything
                    logger.warning(f"Skipping malformed line: {e}")
                    continue

        return results

    def get_run_summaries(self, days: int = 7) -> List[RunSummary]:
        """Get run summaries from the past N days."""
        cutoff = datetime.now() - timedelta(days=days)
        results = []

        if not self.accomplishments_path.exists():
            return results

        with open(self.accomplishments_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('_type') != 'run_summary':
                        continue
                    entry_time = datetime.fromisoformat(data.get('timestamp', ''))
                    if entry_time >= cutoff:
                        # Remove _type before creating RunSummary
                        data.pop('_type', None)
                        results.append(RunSummary(**data))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        return results

    def get_accomplishments_by_category(self, category: str,
                                        hours: int = 24) -> List[Accomplishment]:
        """Get accomplishments filtered by category."""
        all_recent = self.get_recent(hours)
        return [a for a in all_recent if a.category == category]

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get statistics about recent accomplishments."""
        recent = self.get_recent(hours)

        by_category = {}
        by_severity = {}
        total_impact = 0

        for a in recent:
            by_category[a.category] = by_category.get(a.category, 0) + 1
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
            total_impact += a.impact_score

        return {
            "total_entries": len(recent),
            "by_category": by_category,
            "by_severity": by_severity,
            "total_impact_score": total_impact,
            "average_impact": total_impact / len(recent) if recent else 0,
            "fixes_applied": sum(1 for a in recent if a.fix_applied),
            "unique_modules": len(set(a.module for a in recent)),
        }

    def _append_line(self, json_line: str):
        """
        APPEND a single line to the accomplishments file.

        CRITICAL: This method ONLY opens in append mode ('a').
        It NEVER uses write mode ('w') or truncates the file.
        """
        try:
            with open(self.accomplishments_path, 'a', encoding='utf-8') as f:
                # On Windows, we can't use fcntl, so we use a simple approach
                # The 'a' mode is atomic enough for single-process use
                f.write(json_line + '\n')
                f.flush()
                os.fsync(f.fileno())  # Ensure write is persisted
        except Exception as e:
            logger.error(f"Failed to append to accomplishments log: {e}")
            raise

    def _update_status(self, status: Dict[str, Any]):
        """Update the status file (this one CAN be overwritten - it's current state only)."""
        try:
            # Read existing status if any
            existing = {}
            if self.status_path.exists():
                with open(self.status_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)

            # Merge with new status
            existing.update(status)
            existing['last_updated'] = datetime.now().isoformat()

            # Write status (this file is OK to overwrite)
            with open(self.status_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update status file: {e}")

    def _get_cycle_count(self) -> int:
        """Get current cycle count from status file."""
        try:
            if self.status_path.exists():
                with open(self.status_path, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                    return status.get('current_cycle', 0)
        except Exception:
            pass
        return 0

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the accomplishments log.

        Returns statistics about the log and any issues found.
        This is a READ-ONLY operation.
        """
        if not self.accomplishments_path.exists():
            return {"status": "empty", "total_lines": 0, "valid_lines": 0, "errors": []}

        total_lines = 0
        valid_lines = 0
        errors = []

        with open(self.accomplishments_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: {e}")

        return {
            "status": "ok" if not errors else "has_errors",
            "total_lines": total_lines,
            "valid_lines": valid_lines,
            "error_count": len(errors),
            "errors": errors[:10],  # Only show first 10 errors
        }


# Singleton instance for easy access
_tracker_instance: Optional[AccomplishmentsTracker] = None


def get_tracker() -> AccomplishmentsTracker:
    """Get the singleton tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = AccomplishmentsTracker()
    return _tracker_instance
