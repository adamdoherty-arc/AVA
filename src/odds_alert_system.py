"""
Automated Alert System for Odds Anomalies

Features:
- Real-time anomaly detection and alerting
- Multi-channel notifications (email, Slack, database)
- Alert prioritization and deduplication
- Configurable alert thresholds
- Alert acknowledgment and resolution workflow
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime
import psycopg2
import psycopg2.extras
from dataclasses import dataclass
import os

from src.odds_validator import ValidationResult, ValidationSeverity, ValidationRuleType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlertChannel:
    """Configuration for alert notification channel"""
    name: str
    enabled: bool
    config: Dict


class OddsAlertSystem:
    """
    Automated alert system for odds data quality issues

    Handles:
    - Creating alerts from validation failures
    - Sending notifications via multiple channels
    - Alert deduplication
    - Alert lifecycle management
    """

    def __init__(self, db_config: Dict, alert_channels: Optional[List[AlertChannel]] = None):
        """
        Initialize alert system

        Args:
            db_config: Database configuration
            alert_channels: List of alert channels (email, Slack, etc.)
        """
        self.db_config = db_config
        self.alert_channels = alert_channels or []

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def process_validation_results(
        self,
        ticker: str,
        away_team: str,
        home_team: str,
        away_win_price: float,
        home_win_price: float,
        validation_results: List[ValidationResult],
        metadata: Optional[Dict] = None
    ) -> List[int]:
        """
        Process validation results and create alerts for failures

        Args:
            ticker: Market ticker
            away_team: Away team name
            home_team: Home team name
            away_win_price: Away team win probability
            home_win_price: Home team win probability
            validation_results: List of validation results
            metadata: Additional metadata to include

        Returns:
            List of created alert IDs
        """
        alert_ids = []

        for result in validation_results:
            # Only create alerts for failures
            if result.passed:
                continue

            # Skip info-level alerts unless configured otherwise
            if result.severity == ValidationSeverity.INFO:
                continue

            # Create alert
            alert_id = self._create_alert(
                ticker=ticker,
                away_team=away_team,
                home_team=home_team,
                away_win_price=away_win_price,
                home_win_price=home_win_price,
                validation_result=result,
                metadata=metadata
            )

            if alert_id:
                alert_ids.append(alert_id)

                # Send notifications for critical alerts
                if result.severity == ValidationSeverity.CRITICAL:
                    self._send_notifications(
                        alert_id=alert_id,
                        ticker=ticker,
                        severity=result.severity.value,
                        title=self._get_alert_title(result),
                        description=result.message,
                        away_team=away_team,
                        home_team=home_team,
                        away_win_price=away_win_price,
                        home_win_price=home_win_price
                    )

        return alert_ids

    def _create_alert(
        self,
        ticker: str,
        away_team: str,
        home_team: str,
        away_win_price: float,
        home_win_price: float,
        validation_result: ValidationResult,
        metadata: Optional[Dict] = None
    ) -> Optional[int]:
        """
        Create anomaly alert in database

        Args:
            ticker: Market ticker
            away_team: Away team name
            home_team: Home team name
            away_win_price: Away team win probability
            home_win_price: Home team win probability
            validation_result: Validation result that failed
            metadata: Additional metadata

        Returns:
            Alert ID or None if creation failed
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Prepare alert metadata
            alert_metadata = {
                'rule_type': validation_result.rule_type.value,
                'validation_details': validation_result.details,
                **(metadata or {})
            }

            # Call database function to create alert (handles deduplication)
            cur.execute("""
                SELECT create_odds_anomaly_alert(
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                ticker,
                validation_result.rule_type.value,
                validation_result.severity.value,
                self._get_alert_title(validation_result),
                validation_result.message,
                away_team,
                home_team,
                away_win_price,
                home_win_price,
                psycopg2.extras.Json(alert_metadata)
            ))

            alert_id = cur.fetchone()[0]
            conn.commit()

            logger.info(f"Created alert #{alert_id} for {ticker}: {validation_result.message}")
            return alert_id

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error creating alert: {e}")
            return None
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def _get_alert_title(self, result: ValidationResult) -> str:
        """Generate alert title from validation result"""
        if result.rule_type == ValidationRuleType.TEAM_RECORD_CORRELATION:
            return "🚨 REVERSED ODDS DETECTED"
        elif result.rule_type == ValidationRuleType.PROBABILITY_SUM:
            return "⚠️ Invalid Probability Sum"
        elif result.rule_type == ValidationRuleType.HOME_ADVANTAGE:
            return "⚠️ Missing Home Field Advantage"
        elif result.rule_type == ValidationRuleType.ODDS_RANGE:
            return "🚨 CRITICAL: Invalid Odds Range"
        elif result.rule_type == ValidationRuleType.HISTORICAL_PERFORMANCE:
            return "⚠️ Historical Performance Mismatch"
        elif result.rule_type == ValidationRuleType.DATA_FRESHNESS:
            return "⚠️ Stale Data Detected"
        elif result.rule_type == ValidationRuleType.UPSET_DETECTION:
            return "🎯 Potential Upset Alert"
        else:
            return f"⚠️ {result.rule_type.value.replace('_', ' ').title()}"

    def _send_notifications(
        self,
        alert_id: int,
        ticker: str,
        severity: str,
        title: str,
        description: str,
        away_team: str,
        home_team: str,
        away_win_price: float,
        home_win_price: float
    ) -> None:
        """
        Send notifications through configured channels

        Args:
            alert_id: Alert ID
            ticker: Market ticker
            severity: Alert severity
            title: Alert title
            description: Alert description
            away_team: Away team name
            home_team: Home team name
            away_win_price: Away team win probability
            home_win_price: Home team win probability
        """
        for channel in self.alert_channels:
            if not channel.enabled:
                continue

            try:
                if channel.name == 'email':
                    self._send_email_alert(
                        alert_id=alert_id,
                        ticker=ticker,
                        severity=severity,
                        title=title,
                        description=description,
                        away_team=away_team,
                        home_team=home_team,
                        away_win_price=away_win_price,
                        home_win_price=home_win_price,
                        config=channel.config
                    )
                elif channel.name == 'slack':
                    self._send_slack_alert(
                        alert_id=alert_id,
                        ticker=ticker,
                        severity=severity,
                        title=title,
                        description=description,
                        away_team=away_team,
                        home_team=home_team,
                        away_win_price=away_win_price,
                        home_win_price=home_win_price,
                        config=channel.config
                    )
                elif channel.name == 'console':
                    self._send_console_alert(
                        alert_id=alert_id,
                        ticker=ticker,
                        severity=severity,
                        title=title,
                        description=description,
                        away_team=away_team,
                        home_team=home_team,
                        away_win_price=away_win_price,
                        home_win_price=home_win_price
                    )
            except Exception as e:
                logger.error(f"Error sending notification via {channel.name}: {e}")

    def _send_email_alert(
        self,
        alert_id: int,
        ticker: str,
        severity: str,
        title: str,
        description: str,
        away_team: str,
        home_team: str,
        away_win_price: float,
        home_win_price: float,
        config: Dict
    ) -> None:
        """Send email alert"""

        try:
            # Email configuration
            smtp_host = config.get('smtp_host', 'localhost')
            smtp_port = config.get('smtp_port', 587)
            smtp_user = config.get('smtp_user')
            smtp_password = config.get('smtp_password')
            from_email = config.get('from_email', 'alerts@trading.local')
            to_emails = config.get('to_emails', [])

            if not to_emails:
                logger.warning("No email recipients configured")
                return

            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{severity.upper()}] Odds Alert: {title}"
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)

            # Email body
            html = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <h2 style="color: {'#cc0000' if severity == 'critical' else '#ff9900'};">{title}</h2>

                    <p><strong>Alert ID:</strong> #{alert_id}</p>
                    <p><strong>Ticker:</strong> {ticker}</p>
                    <p><strong>Severity:</strong> <span style="color: {'#cc0000' if severity == 'critical' else '#ff9900'}; font-weight: bold;">{severity.upper()}</span></p>

                    <hr>

                    <h3>Game Details</h3>
                    <table style="border-collapse: collapse; width: 100%;">
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Away Team:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{away_team}</td>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Win Price:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{away_win_price:.1%}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Home Team:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{home_team}</td>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Win Price:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{home_win_price:.1%}</td>
                        </tr>
                    </table>

                    <hr>

                    <h3>Issue Description</h3>
                    <p>{description}</p>

                    <hr>

                    <p style="color: #666; font-size: 12px;">
                        This is an automated alert from the Odds Validation System.<br>
                        Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </body>
            </html>
            """

            msg.attach(MIMEText(html, 'html'))

            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)

            logger.info(f"Email alert sent for alert #{alert_id}")

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")

    def _send_slack_alert(
        self,
        alert_id: int,
        ticker: str,
        severity: str,
        title: str,
        description: str,
        away_team: str,
        home_team: str,
        away_win_price: float,
        home_win_price: float,
        config: Dict
    ) -> None:
        """Send Slack alert"""

        try:
            import requests

            webhook_url = config.get('webhook_url')
            if not webhook_url:
                logger.warning("No Slack webhook URL configured")
                return

            # Slack message
            color = '#cc0000' if severity == 'critical' else '#ff9900'
            emoji = '🚨' if severity == 'critical' else '⚠️'

            payload = {
                'attachments': [
                    {
                        'color': color,
                        'title': f"{emoji} {title}",
                        'fields': [
                            {
                                'title': 'Alert ID',
                                'value': f"#{alert_id}",
                                'short': True
                            },
                            {
                                'title': 'Ticker',
                                'value': ticker,
                                'short': True
                            },
                            {
                                'title': 'Severity',
                                'value': severity.upper(),
                                'short': True
                            },
                            {
                                'title': 'Matchup',
                                'value': f"{away_team} @ {home_team}",
                                'short': True
                            },
                            {
                                'title': 'Odds',
                                'value': f"Away: {away_win_price:.1%}, Home: {home_win_price:.1%}",
                                'short': False
                            },
                            {
                                'title': 'Description',
                                'value': description,
                                'short': False
                            }
                        ],
                        'footer': 'Odds Validation System',
                        'ts': int(datetime.now().timestamp())
                    }
                ]
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Slack alert sent for alert #{alert_id}")

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    def _send_console_alert(
        self,
        alert_id: int,
        ticker: str,
        severity: str,
        title: str,
        description: str,
        away_team: str,
        home_team: str,
        away_win_price: float,
        home_win_price: float
    ) -> None:
        """Send console alert (print to stdout)"""

        border = "=" * 80
        print(f"\n{border}")
        print(f"ODDS ALERT #{alert_id} - {severity.upper()}")
        print(border)
        print(f"Title:       {title}")
        print(f"Ticker:      {ticker}")
        print(f"Matchup:     {away_team} @ {home_team}")
        print(f"Odds:        Away: {away_win_price:.1%}, Home: {home_win_price:.1%}")
        print(f"Description: {description}")
        print(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(border + "\n")

    def acknowledge_alert(
        self,
        alert_id: int,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: Username/email of person acknowledging
            notes: Optional acknowledgment notes

        Returns:
            True if successful, False otherwise
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE odds_anomaly_alerts
                SET status = 'acknowledged',
                    acknowledged_by = %s,
                    acknowledged_at = NOW(),
                    resolution_notes = COALESCE(%s, resolution_notes),
                    updated_at = NOW()
                WHERE id = %s
                AND status = 'open'
            """, (acknowledged_by, notes, alert_id))

            conn.commit()
            success = cur.rowcount > 0

            if success:
                logger.info(f"Alert #{alert_id} acknowledged by {acknowledged_by}")
            else:
                logger.warning(f"Alert #{alert_id} not found or already acknowledged")

            return success

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error acknowledging alert: {e}")
            return False
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def resolve_alert(
        self,
        alert_id: int,
        resolution_notes: str,
        resolved_by: Optional[str] = None
    ) -> bool:
        """
        Resolve an alert

        Args:
            alert_id: Alert ID to resolve
            resolution_notes: Resolution notes
            resolved_by: Username/email of person resolving

        Returns:
            True if successful, False otherwise
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE odds_anomaly_alerts
                SET status = 'resolved',
                    resolution_notes = %s,
                    acknowledged_by = COALESCE(acknowledged_by, %s),
                    acknowledged_at = COALESCE(acknowledged_at, NOW()),
                    updated_at = NOW()
                WHERE id = %s
            """, (resolution_notes, resolved_by, alert_id))

            conn.commit()
            success = cur.rowcount > 0

            if success:
                logger.info(f"Alert #{alert_id} resolved")
            else:
                logger.warning(f"Alert #{alert_id} not found")

            return success

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error resolving alert: {e}")
            return False
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def get_active_alerts(
        self,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get active alerts

        Args:
            severity: Filter by severity ('critical', 'warning', 'info')
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            if severity:
                cur.execute("""
                    SELECT * FROM v_odds_active_alerts
                    WHERE severity = %s
                    LIMIT %s
                """, (severity, limit))
            else:
                cur.execute("""
                    SELECT * FROM v_odds_active_alerts
                    LIMIT %s
                """, (limit,))

            alerts = cur.fetchall()
            return [dict(alert) for alert in alerts]

        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()


# Convenience function for integration
def send_odds_alert(
    ticker: str,
    away_team: str,
    home_team: str,
    away_win_price: float,
    home_win_price: float,
    validation_results: List[ValidationResult],
    db_config: Dict,
    alert_channels: Optional[List[AlertChannel]] = None
) -> List[int]:
    """
    Quick function to send alerts for validation failures

    Usage:
        from src.odds_alert_system import send_odds_alert, AlertChannel

        alert_channels = [
            AlertChannel(name='console', enabled=True, config={}),
            AlertChannel(name='email', enabled=True, config={
                'smtp_host': 'smtp.gmail.com',
                'smtp_port': 587,
                'smtp_user': 'your@email.com',
                'smtp_password': 'your_password',
                'from_email': 'alerts@trading.com',
                'to_emails': ['admin@trading.com']
            })
        ]

        alert_ids = send_odds_alert(
            ticker="KXNFL-DAL-PHI",
            away_team="Dallas Cowboys",
            home_team="Philadelphia Eagles",
            away_win_price=0.35,
            home_win_price=0.65,
            validation_results=results,
            db_config=db_config,
            alert_channels=alert_channels
        )
    """
    alert_system = OddsAlertSystem(db_config, alert_channels)
    return alert_system.process_validation_results(
        ticker=ticker,
        away_team=away_team,
        home_team=home_team,
        away_win_price=away_win_price,
        home_win_price=home_win_price,
        validation_results=validation_results
    )


if __name__ == "__main__":
    # Test alert system
    import os
    from src.odds_validator import OddsValidator

    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'magnus',
        'user': 'postgres',
        'password': os.getenv('DB_PASSWORD')
    }

    # Setup alert channels (console only for testing)
    alert_channels = [
        AlertChannel(name='console', enabled=True, config={})
    ]

    # Create validator and alert system
    validator = OddsValidator(db_config)
    alert_system = OddsAlertSystem(db_config, alert_channels)

    print("\n" + "="*80)
    print("ODDS ALERT SYSTEM - Test")
    print("="*80)

    # Test with reversed odds (should trigger CRITICAL alert)
    print("\nTesting with REVERSED ODDS...")
    is_valid, results = validator.validate_game_odds(
        ticker="TEST-ALERT-1",
        away_team="Buffalo Bills",
        home_team="New York Jets",
        away_win_price=0.35,  # Better team has LOWER odds
        home_win_price=0.65,
        away_record="9-1",
        home_record="3-7"
    )

    print(f"\nValidation Result: {'VALID' if is_valid else 'INVALID'}")

    # Process results and create alerts
    alert_ids = alert_system.process_validation_results(
        ticker="TEST-ALERT-1",
        away_team="Buffalo Bills",
        home_team="New York Jets",
        away_win_price=0.35,
        home_win_price=0.65,
        validation_results=results
    )

    print(f"\nCreated {len(alert_ids)} alert(s): {alert_ids}")

    # Get active alerts
    active_alerts = alert_system.get_active_alerts()
    print(f"\nActive alerts: {len(active_alerts)}")

    print("\n" + "="*80)
    print("Alert System Test Complete!")
    print("="*80)
