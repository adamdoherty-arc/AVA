"""
Alerting System for Local Orchestrator
Monitors metrics and sends alerts (log, file, email)
100% Local - no cloud dependencies
"""
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import logging
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert:
    """Represents an alert"""
    def __init__(self, severity: AlertSeverity, title: str, message: str,
                 context: Dict[str, Any] = None):
        self.severity = severity
        self.title = title
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class AlertingSystem:
    """
    Local alerting system
    - Monitors metrics
    - Evaluates conditions
    - Sends alerts via configured channels
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.handlers: List[Callable] = []
        self.alert_history: List[Alert] = []
        self.alert_log_path = Path(".claude/orchestrator/logs/alerts.log")
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._setup_handlers()
        logger.info("Alerting system initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default alerting configuration"""
        return {
            "enabled": True,
            "channels": ["log", "file"],
            "conditions": [
                {"type": "error_rate", "threshold": 0.1, "severity": "warning"},
                {"type": "latency_p95", "threshold": 5000, "severity": "warning"},
                {"type": "consecutive_failures", "threshold": 3, "severity": "error"}
            ]
        }

    def _setup_handlers(self) -> None:
        """Setup alert handlers based on config"""
        channels = self.config.get('channels', ['log', 'file'])

        if 'log' in channels:
            self.handlers.append(self._log_alert)

        if 'file' in channels:
            self.handlers.append(self._file_alert)

    def send_alert(self, alert: Alert):
        """Send alert through all configured channels"""
        if not self.config.get('enabled', True):
            return

        # Add to history
        self.alert_history.append(alert)

        # Send through handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def _log_alert(self, alert: Alert):
        """Log alert"""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)

        logger.log(
            log_level,
            f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        )

    def _file_alert(self, alert: Alert):
        """Write alert to file"""
        with open(self.alert_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(alert.to_dict()) + "\n")

    def check_metrics(self, metrics: Dict[str, Any]):
        """Check metrics against alert conditions"""
        conditions = self.config.get('conditions', [])

        for condition in conditions:
            if self._evaluate_condition(condition, metrics):
                severity = AlertSeverity(condition.get('severity', 'warning'))
                self.send_alert(Alert(
                    severity=severity,
                    title=f"Condition triggered: {condition.get('type')}",
                    message=f"Threshold: {condition.get('threshold')}, Current metrics: {metrics}",
                    context={"condition": condition, "metrics": metrics}
                ))

    def _evaluate_condition(self, condition: Dict[str, Any],
                           metrics: Dict[str, Any]) -> bool:
        """Evaluate a single alert condition"""
        try:
            condition_type = condition.get('type')
            threshold = condition.get('threshold')

            if condition_type == 'error_rate':
                overall = metrics.get('overall', {})
                success_rate = overall.get('success_rate', 100)
                error_rate = (100 - success_rate) / 100
                return error_rate > threshold

            elif condition_type == 'latency_p95':
                overall = metrics.get('overall', {})
                avg_latency = overall.get('avg_duration_ms', 0)
                # Approximate p95 as 1.5x average (rough estimate)
                p95_latency = avg_latency * 1.5
                return p95_latency > threshold

            elif condition_type == 'consecutive_failures':
                # Check recent errors
                errors = metrics.get('recent_errors', [])
                return len(errors) >= threshold

            elif condition_type == 'total_executions':
                overall = metrics.get('overall', {})
                total = overall.get('total_executions', 0)
                return total > threshold

            return False

        except Exception as e:
            logger.error(f"Failed to evaluate condition {condition}: {e}")
            return False

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        return [
            alert.to_dict()
            for alert in self.alert_history
            if alert.timestamp.timestamp() > cutoff
        ]

    def clear_history(self) -> None:
        """Clear alert history"""
        self.alert_history.clear()
        logger.info("Alert history cleared")


# Singleton
_alerting_system: Optional[AlertingSystem] = None


def get_alerting_system(config: Dict[str, Any] = None) -> AlertingSystem:
    """Get singleton alerting system instance"""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem(config)
    return _alerting_system


if __name__ == "__main__":
    # Test alerting system
    alerting = get_alerting_system()

    # Send test alert
    alerting.send_alert(Alert(
        severity=AlertSeverity.WARNING,
        title="Test Alert",
        message="This is a test alert",
        context={"test": True}
    ))

    # Test metric checking
    test_metrics = {
        "overall": {
            "success_rate": 85,  # Will trigger error_rate alert
            "avg_duration_ms": 2000
        },
        "recent_errors": []
    }

    alerting.check_metrics(test_metrics)

    print(f"\nRecent alerts: {len(alerting.get_recent_alerts())}")
    print(f"Alert log: {alerting.alert_log_path}")
    print("\nAlerting system test complete!")
