"""
Comprehensive Odds Validation System
Bulletproof validation to catch data quality issues in betting odds

Features:
- Multi-level validation rules with severity levels
- Historical performance correlation checks
- Home advantage validation
- Market efficiency checks (odds sum to ~100¢)
- Automated anomaly detection and alerting
- Detailed validation reports and metrics
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import psycopg2
import psycopg2.extras
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"  # Blocking issue - do not display
    WARNING = "warning"    # Suspicious - flag for review
    INFO = "info"          # Minor anomaly - informational


class ValidationRuleType(Enum):
    """Types of validation rules"""
    PROBABILITY_SUM = "probability_sum"
    TEAM_RECORD_CORRELATION = "team_record_correlation"
    HOME_ADVANTAGE = "home_advantage"
    HISTORICAL_PERFORMANCE = "historical_performance"
    MARKET_EFFICIENCY = "market_efficiency"
    ODDS_RANGE = "odds_range"
    UPSET_DETECTION = "upset_detection"
    DATA_FRESHNESS = "data_freshness"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    rule_type: ValidationRuleType
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict
    timestamp: datetime


class OddsValidator:
    """
    Comprehensive odds validation system

    Validates betting odds for data quality issues including:
    - Reversed odds (better team showing lower probability)
    - Invalid probability sums
    - Missing home field advantage
    - Historical performance misalignment
    """

    # Validation thresholds
    PROBABILITY_SUM_MIN = 0.95  # 95¢ - accounting for market spread
    PROBABILITY_SUM_MAX = 1.05  # 105¢ - accounting for market spread
    HOME_ADVANTAGE_MIN = 0.02   # Minimum 2% home advantage expected
    HOME_ADVANTAGE_MAX = 0.15   # Maximum 15% home advantage
    MIN_VALID_ODDS = 0.01       # Minimum 1% probability
    MAX_VALID_ODDS = 0.99       # Maximum 99% probability
    RECORD_CORRELATION_THRESHOLD = 0.10  # 10% win rate difference threshold
    DATA_FRESHNESS_HOURS = 24   # Data older than 24h is stale

    def __init__(self, db_config: Dict):
        """
        Initialize validator with database connection

        Args:
            db_config: PostgreSQL connection configuration
        """
        self.db_config = db_config
        self.validation_history: List[ValidationResult] = []

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def validate_game_odds(
        self,
        ticker: str,
        away_team: str,
        home_team: str,
        away_win_price: float,
        home_win_price: float,
        game_date: Optional[datetime] = None,
        away_record: Optional[str] = None,
        home_record: Optional[str] = None,
        last_updated: Optional[datetime] = None
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate odds for a single game

        Args:
            ticker: Market ticker symbol
            away_team: Away team name
            home_team: Home team name
            away_win_price: Away team win probability (0-1)
            home_win_price: Home team win probability (0-1)
            game_date: Scheduled game date
            away_record: Away team record (e.g., "7-3")
            home_record: Home team record (e.g., "6-4")
            last_updated: When odds were last updated

        Returns:
            Tuple of (is_valid, list of ValidationResult objects)
        """
        results: List[ValidationResult] = []

        # Rule 1: Validate odds are in valid range
        results.append(self._validate_odds_range(
            ticker, away_team, home_team,
            away_win_price, home_win_price
        ))

        # Rule 2: Validate probability sum (should be ~1.0)
        results.append(self._validate_probability_sum(
            ticker, away_win_price, home_win_price
        ))

        # Rule 3: Validate team record correlation
        if away_record and home_record:
            results.append(self._validate_record_correlation(
                ticker, away_team, home_team,
                away_win_price, home_win_price,
                away_record, home_record
            ))

        # Rule 4: Validate home field advantage
        results.append(self._validate_home_advantage(
            ticker, away_team, home_team,
            away_win_price, home_win_price,
            away_record, home_record
        ))

        # Rule 5: Validate against historical performance
        results.append(self._validate_historical_performance(
            ticker, away_team, home_team,
            away_win_price, home_win_price
        ))

        # Rule 6: Check for data freshness
        if last_updated:
            results.append(self._validate_data_freshness(
                ticker, last_updated
            ))

        # Rule 7: Detect potential upsets
        results.append(self._detect_upset(
            ticker, away_team, home_team,
            away_win_price, home_win_price,
            away_record, home_record
        ))

        # Store validation results
        self.validation_history.extend(results)

        # Determine if odds are valid (no CRITICAL issues)
        is_valid = all(
            r.passed or r.severity != ValidationSeverity.CRITICAL
            for r in results
        )

        # Log results
        critical_issues = [r for r in results if not r.passed and r.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            logger.error(f"CRITICAL validation failures for {ticker}:")
            for issue in critical_issues:
                logger.error(f"  - {issue.message}")

        warnings = [r for r in results if not r.passed and r.severity == ValidationSeverity.WARNING]
        if warnings:
            logger.warning(f"Validation warnings for {ticker}:")
            for warning in warnings:
                logger.warning(f"  - {warning.message}")

        return is_valid, results

    def _validate_odds_range(
        self,
        ticker: str,
        away_team: str,
        home_team: str,
        away_price: float,
        home_price: float
    ) -> ValidationResult:
        """Validate odds are within acceptable range"""

        passed = True
        severity = ValidationSeverity.CRITICAL
        details = {
            'away_price': away_price,
            'home_price': home_price,
            'min_valid': self.MIN_VALID_ODDS,
            'max_valid': self.MAX_VALID_ODDS
        }

        invalid_prices = []
        if away_price < self.MIN_VALID_ODDS or away_price > self.MAX_VALID_ODDS:
            invalid_prices.append(f"{away_team}: {away_price:.2%}")
            passed = False

        if home_price < self.MIN_VALID_ODDS or home_price > self.MAX_VALID_ODDS:
            invalid_prices.append(f"{home_team}: {home_price:.2%}")
            passed = False

        if passed:
            message = f"✓ Odds within valid range ({self.MIN_VALID_ODDS:.0%}-{self.MAX_VALID_ODDS:.0%})"
        else:
            message = f"✗ CRITICAL: Invalid odds detected - {', '.join(invalid_prices)}"
            details['invalid_prices'] = invalid_prices

        return ValidationResult(
            rule_type=ValidationRuleType.ODDS_RANGE,
            severity=severity,
            passed=passed,
            message=message,
            details=details,
            timestamp=datetime.now()
        )

    def _validate_probability_sum(
        self,
        ticker: str,
        away_price: float,
        home_price: float
    ) -> ValidationResult:
        """
        Validate that probabilities sum to approximately 100%
        (accounting for market spread/vig)
        """

        total = away_price + home_price
        passed = self.PROBABILITY_SUM_MIN <= total <= self.PROBABILITY_SUM_MAX

        details = {
            'away_price': away_price,
            'home_price': home_price,
            'sum': total,
            'expected_min': self.PROBABILITY_SUM_MIN,
            'expected_max': self.PROBABILITY_SUM_MAX,
            'deviation': abs(total - 1.0)
        }

        if passed:
            message = f"✓ Probability sum valid: {total:.2%} (within {self.PROBABILITY_SUM_MIN:.0%}-{self.PROBABILITY_SUM_MAX:.0%})"
            severity = ValidationSeverity.INFO
        else:
            if total < self.PROBABILITY_SUM_MIN:
                message = f"✗ CRITICAL: Probabilities sum too low ({total:.2%}). Possible data error."
                severity = ValidationSeverity.CRITICAL
            else:
                message = f"⚠ WARNING: Probabilities sum high ({total:.2%}). High market spread detected."
                severity = ValidationSeverity.WARNING

        return ValidationResult(
            rule_type=ValidationRuleType.PROBABILITY_SUM,
            severity=severity,
            passed=passed,
            message=message,
            details=details,
            timestamp=datetime.now()
        )

    def _validate_record_correlation(
        self,
        ticker: str,
        away_team: str,
        home_team: str,
        away_price: float,
        home_price: float,
        away_record: str,
        home_record: str
    ) -> ValidationResult:
        """
        Validate that team with better record has higher win probability
        """

        try:
            # Parse records (format: "7-3" or "7-3-0")
            away_wins, away_losses = self._parse_record(away_record)
            home_wins, home_losses = self._parse_record(home_record)

            if away_wins is None or home_wins is None:
                return ValidationResult(
                    rule_type=ValidationRuleType.TEAM_RECORD_CORRELATION,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="⚠ Could not parse team records",
                    details={'away_record': away_record, 'home_record': home_record},
                    timestamp=datetime.now()
                )

            # Calculate win percentages
            away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5
            home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5

            win_pct_diff = abs(away_win_pct - home_win_pct)

            details = {
                'away_team': away_team,
                'home_team': home_team,
                'away_record': away_record,
                'home_record': home_record,
                'away_win_pct': away_win_pct,
                'home_win_pct': home_win_pct,
                'away_price': away_price,
                'home_price': home_price,
                'win_pct_diff': win_pct_diff
            }

            # If records are similar, any odds are reasonable
            if win_pct_diff < self.RECORD_CORRELATION_THRESHOLD:
                return ValidationResult(
                    rule_type=ValidationRuleType.TEAM_RECORD_CORRELATION,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message=f"✓ Similar records ({away_record} vs {home_record}), odds correlation not applicable",
                    details=details,
                    timestamp=datetime.now()
                )

            # Check if better team has higher odds
            better_team_is_away = away_win_pct > home_win_pct
            better_team_has_higher_odds = away_price > home_price

            # CRITICAL: Better team should have higher win probability
            if better_team_is_away and not better_team_has_higher_odds:
                passed = False
                severity = ValidationSeverity.CRITICAL
                message = (f"✗ CRITICAL: ODDS REVERSED - {away_team} ({away_record}, {away_win_pct:.1%}) "
                          f"has LOWER odds ({away_price:.1%}) than {home_team} ({home_record}, {home_win_pct:.1%}, {home_price:.1%})")
                details['reversed'] = True
            elif not better_team_is_away and better_team_has_higher_odds:
                passed = False
                severity = ValidationSeverity.CRITICAL
                message = (f"✗ CRITICAL: ODDS REVERSED - {home_team} ({home_record}, {home_win_pct:.1%}) "
                          f"has LOWER odds ({home_price:.1%}) than {away_team} ({away_record}, {away_win_pct:.1%}, {away_price:.1%})")
                details['reversed'] = True
            else:
                passed = True
                severity = ValidationSeverity.INFO
                better_team = away_team if better_team_is_away else home_team
                better_record = away_record if better_team_is_away else home_record
                better_odds = away_price if better_team_is_away else home_price
                message = f"✓ Record correlation valid: {better_team} ({better_record}) has higher odds ({better_odds:.1%})"
                details['reversed'] = False

            return ValidationResult(
                rule_type=ValidationRuleType.TEAM_RECORD_CORRELATION,
                severity=severity,
                passed=passed,
                message=message,
                details=details,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error validating record correlation: {e}")
            return ValidationResult(
                rule_type=ValidationRuleType.TEAM_RECORD_CORRELATION,
                severity=ValidationSeverity.WARNING,
                passed=True,
                message=f"⚠ Could not validate record correlation: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.now()
            )

    def _validate_home_advantage(
        self,
        ticker: str,
        away_team: str,
        home_team: str,
        away_price: float,
        home_price: float,
        away_record: Optional[str] = None,
        home_record: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate that home field advantage is properly factored

        For evenly matched teams, home team should have slight edge
        """

        details = {
            'away_team': away_team,
            'home_team': home_team,
            'away_price': away_price,
            'home_price': home_price,
            'expected_home_advantage': f"{self.HOME_ADVANTAGE_MIN:.0%}-{self.HOME_ADVANTAGE_MAX:.0%}"
        }

        # If we have records, check if teams are evenly matched
        if away_record and home_record:
            try:
                away_wins, away_losses = self._parse_record(away_record)
                home_wins, home_losses = self._parse_record(home_record)

                if away_wins is not None and home_wins is not None:
                    away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5
                    home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5

                    win_pct_diff = abs(away_win_pct - home_win_pct)
                    details['away_win_pct'] = away_win_pct
                    details['home_win_pct'] = home_win_pct
                    details['win_pct_diff'] = win_pct_diff

                    # Only validate home advantage for evenly matched teams
                    if win_pct_diff < self.RECORD_CORRELATION_THRESHOLD:
                        # Teams are evenly matched - home team should have slight advantage
                        if home_price <= away_price:
                            # Home team doesn't have advantage - WARNING
                            passed = False
                            severity = ValidationSeverity.WARNING
                            message = (f"⚠ WARNING: Evenly matched teams ({away_record} vs {home_record}) "
                                      f"but home team has no advantage ({home_price:.1%} vs {away_price:.1%})")
                            details['missing_home_advantage'] = True
                        else:
                            home_advantage = home_price - away_price
                            if self.HOME_ADVANTAGE_MIN <= home_advantage <= self.HOME_ADVANTAGE_MAX:
                                passed = True
                                severity = ValidationSeverity.INFO
                                message = f"✓ Home advantage properly factored: {home_advantage:.1%}"
                                details['home_advantage'] = home_advantage
                            else:
                                passed = False
                                severity = ValidationSeverity.WARNING
                                message = f"⚠ WARNING: Home advantage unusual ({home_advantage:.1%})"
                                details['home_advantage'] = home_advantage

                        return ValidationResult(
                            rule_type=ValidationRuleType.HOME_ADVANTAGE,
                            severity=severity,
                            passed=passed,
                            message=message,
                            details=details,
                            timestamp=datetime.now()
                        )
            except Exception as e:
                logger.debug(f"Could not parse records for home advantage check: {e}")

        # If we can't validate, pass with INFO
        return ValidationResult(
            rule_type=ValidationRuleType.HOME_ADVANTAGE,
            severity=ValidationSeverity.INFO,
            passed=True,
            message="ℹ Home advantage check skipped (insufficient data)",
            details=details,
            timestamp=datetime.now()
        )

    def _validate_historical_performance(
        self,
        ticker: str,
        away_team: str,
        home_team: str,
        away_price: float,
        home_price: float
    ) -> ValidationResult:
        """
        Validate odds against historical head-to-head performance
        """

        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Query historical games between these teams
            cur.execute("""
                SELECT
                    home_team,
                    away_team,
                    CASE
                        WHEN result = 'yes' THEN 1
                        WHEN result = 'no' THEN 0
                        ELSE NULL
                    END as home_won
                FROM kalshi_markets
                WHERE status = 'settled'
                AND (
                    (home_team = %s AND away_team = %s)
                    OR (home_team = %s AND away_team = %s)
                )
                AND result IS NOT NULL
                AND created_at > NOW() - INTERVAL '2 years'
                ORDER BY created_at DESC
                LIMIT 10
            """, (home_team, away_team, away_team, home_team))

            historical_games = cur.fetchall()

            if not historical_games or len(historical_games) < 3:
                return ValidationResult(
                    rule_type=ValidationRuleType.HISTORICAL_PERFORMANCE,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="ℹ Insufficient historical data for validation",
                    details={'historical_game_count': len(historical_games) if historical_games else 0},
                    timestamp=datetime.now()
                )

            # Calculate historical win rates
            home_wins = sum(1 for g in historical_games if g['home_won'] == 1)
            away_wins = len(historical_games) - home_wins

            historical_home_pct = home_wins / len(historical_games)
            historical_away_pct = away_wins / len(historical_games)

            details = {
                'away_team': away_team,
                'home_team': home_team,
                'historical_game_count': len(historical_games),
                'historical_home_wins': home_wins,
                'historical_away_wins': away_wins,
                'historical_home_pct': historical_home_pct,
                'historical_away_pct': historical_away_pct,
                'current_home_price': home_price,
                'current_away_price': away_price
            }

            # Check if current odds are reasonable given history
            home_odds_deviation = abs(home_price - historical_home_pct)
            away_odds_deviation = abs(away_price - historical_away_pct)

            # If deviation is > 20%, flag as warning
            if home_odds_deviation > 0.20 or away_odds_deviation > 0.20:
                passed = False
                severity = ValidationSeverity.WARNING
                message = (f"⚠ WARNING: Odds deviate significantly from historical performance "
                          f"(Home: {home_price:.1%} vs {historical_home_pct:.1%} historical, "
                          f"Away: {away_price:.1%} vs {historical_away_pct:.1%} historical)")
                details['home_deviation'] = home_odds_deviation
                details['away_deviation'] = away_odds_deviation
            else:
                passed = True
                severity = ValidationSeverity.INFO
                message = f"✓ Odds align with historical performance (n={len(historical_games)} games)"

            return ValidationResult(
                rule_type=ValidationRuleType.HISTORICAL_PERFORMANCE,
                severity=severity,
                passed=passed,
                message=message,
                details=details,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.debug(f"Could not validate historical performance: {e}")
            return ValidationResult(
                rule_type=ValidationRuleType.HISTORICAL_PERFORMANCE,
                severity=ValidationSeverity.INFO,
                passed=True,
                message="ℹ Historical performance check skipped (database unavailable)",
                details={'error': str(e)},
                timestamp=datetime.now()
            )
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def _validate_data_freshness(
        self,
        ticker: str,
        last_updated: datetime
    ) -> ValidationResult:
        """Validate that odds data is fresh"""

        age = datetime.now() - last_updated
        age_hours = age.total_seconds() / 3600

        details = {
            'last_updated': last_updated.isoformat(),
            'age_hours': age_hours,
            'threshold_hours': self.DATA_FRESHNESS_HOURS
        }

        if age_hours > self.DATA_FRESHNESS_HOURS:
            passed = False
            severity = ValidationSeverity.WARNING
            message = f"⚠ WARNING: Stale data - last updated {age_hours:.1f}h ago (threshold: {self.DATA_FRESHNESS_HOURS}h)"
        else:
            passed = True
            severity = ValidationSeverity.INFO
            message = f"✓ Fresh data (updated {age_hours:.1f}h ago)"

        return ValidationResult(
            rule_type=ValidationRuleType.DATA_FRESHNESS,
            severity=severity,
            passed=passed,
            message=message,
            details=details,
            timestamp=datetime.now()
        )

    def _detect_upset(
        self,
        ticker: str,
        away_team: str,
        home_team: str,
        away_price: float,
        home_price: float,
        away_record: Optional[str] = None,
        home_record: Optional[str] = None
    ) -> ValidationResult:
        """
        Detect potential upsets (underdog has better record)
        This is not an error, but useful information
        """

        if not away_record or not home_record:
            return ValidationResult(
                rule_type=ValidationRuleType.UPSET_DETECTION,
                severity=ValidationSeverity.INFO,
                passed=True,
                message="ℹ Upset detection skipped (no records)",
                details={},
                timestamp=datetime.now()
            )

        try:
            away_wins, away_losses = self._parse_record(away_record)
            home_wins, home_losses = self._parse_record(home_record)

            if away_wins is None or home_wins is None:
                return ValidationResult(
                    rule_type=ValidationRuleType.UPSET_DETECTION,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="ℹ Upset detection skipped (invalid records)",
                    details={},
                    timestamp=datetime.now()
                )

            away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5
            home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5

            # Determine favorite and underdog
            if away_price > home_price:
                favorite = away_team
                favorite_record = away_record
                favorite_win_pct = away_win_pct
                underdog = home_team
                underdog_record = home_record
                underdog_win_pct = home_win_pct
            else:
                favorite = home_team
                favorite_record = home_record
                favorite_win_pct = home_win_pct
                underdog = away_team
                underdog_record = away_record
                underdog_win_pct = away_win_pct

            details = {
                'favorite': favorite,
                'favorite_record': favorite_record,
                'favorite_win_pct': favorite_win_pct,
                'underdog': underdog,
                'underdog_record': underdog_record,
                'underdog_win_pct': underdog_win_pct
            }

            # If underdog has significantly better record, flag as potential upset
            if underdog_win_pct > favorite_win_pct + 0.10:
                passed = True  # Not a failure, just informational
                severity = ValidationSeverity.INFO
                message = (f"🎯 POTENTIAL UPSET: Underdog {underdog} ({underdog_record}, {underdog_win_pct:.1%}) "
                          f"has better record than favorite {favorite} ({favorite_record}, {favorite_win_pct:.1%})")
                details['is_upset_alert'] = True
            else:
                passed = True
                severity = ValidationSeverity.INFO
                message = f"ℹ No upset detected"
                details['is_upset_alert'] = False

            return ValidationResult(
                rule_type=ValidationRuleType.UPSET_DETECTION,
                severity=severity,
                passed=passed,
                message=message,
                details=details,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.debug(f"Error in upset detection: {e}")
            return ValidationResult(
                rule_type=ValidationRuleType.UPSET_DETECTION,
                severity=ValidationSeverity.INFO,
                passed=True,
                message="ℹ Upset detection skipped (error)",
                details={'error': str(e)},
                timestamp=datetime.now()
            )

    def _parse_record(self, record: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse team record string

        Args:
            record: Record string (e.g., "7-3" or "7-3-0")

        Returns:
            Tuple of (wins, losses) or (None, None) if invalid
        """
        try:
            parts = record.split('-')
            if len(parts) >= 2:
                wins = int(parts[0])
                losses = int(parts[1])
                return wins, losses
            return None, None
        except (ValueError, AttributeError):
            return None, None

    def store_validation_results(
        self,
        ticker: str,
        results: List[ValidationResult]
    ) -> None:
        """
        Store validation results in database for tracking and analysis

        Args:
            ticker: Market ticker
            results: List of validation results
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            for result in results:
                cur.execute("""
                    INSERT INTO odds_data_quality_log (
                        ticker,
                        rule_type,
                        severity,
                        passed,
                        message,
                        details,
                        checked_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    ticker,
                    result.rule_type.value,
                    result.severity.value,
                    result.passed,
                    result.message,
                    psycopg2.extras.Json(result.details),
                    result.timestamp
                ))

            conn.commit()
            logger.info(f"Stored {len(results)} validation results for {ticker}")

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error storing validation results: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def get_validation_summary(
        self,
        hours: int = 24
    ) -> Dict:
        """
        Get summary of validation results over time period

        Args:
            hours: Time period to analyze

        Returns:
            Dictionary with validation statistics
        """
        conn = None
        cur = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get total checks
            cur.execute("""
                SELECT COUNT(*) as total_checks
                FROM odds_data_quality_log
                WHERE checked_at > NOW() - INTERVAL '%s hours'
            """, (hours,))
            total_checks = cur.fetchone()['total_checks']

            # Get failures by severity
            cur.execute("""
                SELECT
                    severity,
                    COUNT(*) as count
                FROM odds_data_quality_log
                WHERE checked_at > NOW() - INTERVAL '%s hours'
                AND passed = FALSE
                GROUP BY severity
            """, (hours,))
            failures_by_severity = {row['severity']: row['count'] for row in cur.fetchall()}

            # Get failures by rule type
            cur.execute("""
                SELECT
                    rule_type,
                    COUNT(*) as count
                FROM odds_data_quality_log
                WHERE checked_at > NOW() - INTERVAL '%s hours'
                AND passed = FALSE
                GROUP BY rule_type
                ORDER BY count DESC
            """, (hours,))
            failures_by_rule = {row['rule_type']: row['count'] for row in cur.fetchall()}

            # Get recent critical failures
            cur.execute("""
                SELECT
                    ticker,
                    rule_type,
                    message,
                    details,
                    checked_at
                FROM odds_data_quality_log
                WHERE checked_at > NOW() - INTERVAL '%s hours'
                AND severity = 'critical'
                AND passed = FALSE
                ORDER BY checked_at DESC
                LIMIT 10
            """, (hours,))
            critical_failures = [dict(row) for row in cur.fetchall()]

            return {
                'total_checks': total_checks,
                'failures_by_severity': failures_by_severity,
                'failures_by_rule': failures_by_rule,
                'critical_failures': critical_failures,
                'time_period_hours': hours
            }

        except Exception as e:
            logger.error(f"Error getting validation summary: {e}")
            return {}
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()


# Convenience function for easy integration
def validate_kalshi_market(
    ticker: str,
    away_team: str,
    home_team: str,
    away_win_price: float,
    home_win_price: float,
    db_config: Dict,
    **kwargs
) -> Tuple[bool, List[ValidationResult]]:
    """
    Quick validation function for Kalshi markets

    Usage:
        from src.odds_validator import validate_kalshi_market

        is_valid, results = validate_kalshi_market(
            ticker="KXNFL-DAL-PHI",
            away_team="Dallas Cowboys",
            home_team="Philadelphia Eagles",
            away_win_price=0.45,
            home_win_price=0.55,
            db_config=db_config,
            away_record="7-3",
            home_record="9-1"
        )

        if not is_valid:
            print("CRITICAL ISSUES DETECTED - DO NOT DISPLAY")
    """
    validator = OddsValidator(db_config)
    return validator.validate_game_odds(
        ticker=ticker,
        away_team=away_team,
        home_team=home_team,
        away_win_price=away_win_price,
        home_win_price=home_win_price,
        **kwargs
    )


if __name__ == "__main__":
    # Test validation system
    import os

    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'magnus',
        'user': 'postgres',
        'password': os.getenv('DB_PASSWORD')
    }

    validator = OddsValidator(db_config)

    print("\n" + "="*80)
    print("ODDS VALIDATOR - Test Suite")
    print("="*80)

    # Test Case 1: Normal valid odds
    print("\n--- Test 1: Valid Odds ---")
    is_valid, results = validator.validate_game_odds(
        ticker="TEST-1",
        away_team="Dallas Cowboys",
        home_team="Philadelphia Eagles",
        away_win_price=0.45,
        home_win_price=0.55,
        away_record="7-3",
        home_record="9-1"
    )
    print(f"Valid: {is_valid}")
    for r in results:
        print(f"  [{r.severity.value.upper()}] {r.message}")

    # Test Case 2: REVERSED ODDS (CRITICAL)
    print("\n--- Test 2: REVERSED ODDS (Should FAIL) ---")
    is_valid, results = validator.validate_game_odds(
        ticker="TEST-2",
        away_team="Buffalo Bills",
        home_team="New York Jets",
        away_win_price=0.35,  # Better team (9-1) has LOWER odds
        home_win_price=0.65,  # Worse team (3-7) has HIGHER odds
        away_record="9-1",
        home_record="3-7"
    )
    print(f"Valid: {is_valid}")
    for r in results:
        print(f"  [{r.severity.value.upper()}] {r.message}")

    # Test Case 3: Invalid probability sum
    print("\n--- Test 3: Invalid Probability Sum ---")
    is_valid, results = validator.validate_game_odds(
        ticker="TEST-3",
        away_team="Kansas City Chiefs",
        home_team="Las Vegas Raiders",
        away_win_price=0.70,
        home_win_price=0.70,  # Sum = 1.40 (too high)
        away_record="8-2",
        home_record="5-5"
    )
    print(f"Valid: {is_valid}")
    for r in results:
        print(f"  [{r.severity.value.upper()}] {r.message}")

    print("\n" + "="*80)
    print("Validation Test Complete!")
    print("="*80)
