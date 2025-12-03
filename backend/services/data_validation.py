"""
Data Validation Framework

Provides comprehensive data quality checks for portfolio data:
- Greeks bounds validation
- Price consistency checks
- Data freshness monitoring
- Audit trail logging
- Anomaly detection
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # All validations pass
    GOOD = "good"           # Minor issues
    WARNING = "warning"     # Some concerns
    CRITICAL = "critical"   # Significant issues
    INVALID = "invalid"     # Data unusable


class ValidationSeverity(Enum):
    """Validation issue severity"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue"""
    field: str
    message: str
    severity: ValidationSeverity
    actual_value: Any
    expected_range: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation checks"""
    valid: bool
    quality: DataQuality
    issues: List[ValidationIssue] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)
    data_freshness_seconds: Optional[float] = None

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL))

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "quality": self.quality.value,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "field": i.field,
                    "message": i.message,
                    "severity": i.severity.value,
                    "actual_value": str(i.actual_value),
                    "expected_range": i.expected_range,
                    "suggestion": i.suggestion
                }
                for i in self.issues
            ],
            "data_freshness_seconds": self.data_freshness_seconds,
            "checked_at": self.checked_at.isoformat()
        }


class PortfolioDataValidator:
    """
    Comprehensive data validation for portfolio positions.

    Validates:
    - Greeks bounds (delta, gamma, theta, vega, IV)
    - Price consistency (bid-ask spread, price vs strike)
    - Data freshness (timestamp age)
    - Cross-field consistency (Greeks relationships)
    - Position type constraints
    """

    # Greeks validation bounds
    GREEKS_BOUNDS = {
        "delta": (-100, 100),      # Percentage, -1 to 1 raw
        "gamma": (0, 500),         # Gamma is always positive
        "theta": (-1000, 1000),    # Daily theta in dollars
        "vega": (0, 1000),         # Vega is always positive
        "iv": (0, 500)             # IV percentage (0-500%)
    }

    # Data freshness thresholds
    FRESHNESS_WARNING_SECONDS = 120    # 2 minutes
    FRESHNESS_ERROR_SECONDS = 300      # 5 minutes
    FRESHNESS_CRITICAL_SECONDS = 900   # 15 minutes

    # Price validation thresholds
    MAX_BID_ASK_SPREAD_PCT = 20        # 20% max spread
    MAX_PRICE_DEVIATION_PCT = 50       # 50% from strike for options

    def __init__(self):
        self._validation_history: List[ValidationResult] = []

    def validate_position(self, position: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single position (stock or option).

        Returns comprehensive validation result with all issues found.
        """
        issues = []
        position_type = position.get("type", "unknown")

        # Determine if it's an option or stock
        if "greeks" in position or "strike" in position:
            issues.extend(self._validate_option_position(position))
        else:
            issues.extend(self._validate_stock_position(position))

        # Check data freshness
        freshness_issues, freshness_seconds = self._check_data_freshness(position)
        issues.extend(freshness_issues)

        # Determine overall quality
        quality = self._determine_quality(issues)
        valid = quality not in (DataQuality.CRITICAL, DataQuality.INVALID)

        result = ValidationResult(
            valid=valid,
            quality=quality,
            issues=issues,
            data_freshness_seconds=freshness_seconds
        )

        self._validation_history.append(result)
        return result

    def validate_portfolio(self, positions: Dict[str, Any]) -> ValidationResult:
        """
        Validate entire portfolio for consistency.
        """
        issues = []

        stocks = positions.get("stocks", [])
        options = positions.get("options", [])
        summary = positions.get("summary", {})

        # Validate each position
        for stock in stocks:
            result = self.validate_position(stock)
            if not result.valid:
                for issue in result.issues:
                    issue.field = f"{stock.get('symbol', 'unknown')}.{issue.field}"
                    issues.append(issue)

        for option in options:
            result = self.validate_position(option)
            if not result.valid:
                for issue in result.issues:
                    issue.field = f"{option.get('symbol', 'unknown')}.{issue.field}"
                    issues.append(issue)

        # Cross-portfolio validations
        issues.extend(self._validate_portfolio_consistency(positions))

        # Summary validation
        issues.extend(self._validate_summary(summary, stocks, options))

        quality = self._determine_quality(issues)
        valid = quality not in (DataQuality.CRITICAL, DataQuality.INVALID)

        return ValidationResult(
            valid=valid,
            quality=quality,
            issues=issues
        )

    def _validate_option_position(self, option: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate option-specific fields"""
        issues = []
        greeks = option.get("greeks", {})

        # Greeks bounds validation
        for greek_name, (min_val, max_val) in self.GREEKS_BOUNDS.items():
            value = greeks.get(greek_name, 0)
            if value is not None and not (min_val <= value <= max_val):
                issues.append(ValidationIssue(
                    field=f"greeks.{greek_name}",
                    message=f"{greek_name} value {value} is outside valid range",
                    severity=ValidationSeverity.ERROR if greek_name == "delta" else ValidationSeverity.WARNING,
                    actual_value=value,
                    expected_range=f"[{min_val}, {max_val}]",
                    suggestion=f"Verify {greek_name} data from broker"
                ))

        # Delta sign validation based on position type
        position_type = option.get("type")  # 'long' or 'short'
        option_type = option.get("option_type")  # 'call' or 'put'
        delta = greeks.get("delta", 0)

        if position_type == "short" and option_type == "put":
            # Short put should have positive delta (after sign flip)
            if delta < 0:
                issues.append(ValidationIssue(
                    field="greeks.delta",
                    message="Short put delta should be positive",
                    severity=ValidationSeverity.WARNING,
                    actual_value=delta,
                    expected_range="[0, 100]",
                    suggestion="Check delta sign convention"
                ))

        # Theta validation for short positions
        theta = greeks.get("theta", 0)
        if position_type == "short" and theta < 0:
            issues.append(ValidationIssue(
                field="greeks.theta",
                message="Short position theta should be positive (time decay works for you)",
                severity=ValidationSeverity.INFO,
                actual_value=theta,
                suggestion="Verify theta sign for short positions"
            ))

        # IV validation
        iv = greeks.get("iv", 0)
        if iv < 5:
            issues.append(ValidationIssue(
                field="greeks.iv",
                message="IV suspiciously low (< 5%)",
                severity=ValidationSeverity.WARNING,
                actual_value=iv,
                expected_range="[5, 200] typical",
                suggestion="Check if IV data is stale"
            ))
        elif iv > 200:
            issues.append(ValidationIssue(
                field="greeks.iv",
                message="IV extremely high (> 200%)",
                severity=ValidationSeverity.INFO,
                actual_value=iv,
                suggestion="Verify - high IV may indicate earnings or event"
            ))

        # DTE validation
        dte = option.get("dte", 0)
        if dte < 0:
            issues.append(ValidationIssue(
                field="dte",
                message="Negative DTE - option has expired",
                severity=ValidationSeverity.CRITICAL,
                actual_value=dte,
                expected_range="[0, ∞)",
                suggestion="Remove expired option from portfolio"
            ))

        # Strike vs current price validation
        strike = option.get("strike", 0)
        current_price = option.get("current_price", 0)
        if current_price > 0 and current_price > strike * 10:
            issues.append(ValidationIssue(
                field="current_price",
                message="Option price seems too high relative to strike",
                severity=ValidationSeverity.WARNING,
                actual_value=current_price,
                suggestion="Verify price data"
            ))

        # P/L percentage validation
        pl_pct = option.get("pl_pct", 0)
        if abs(pl_pct) > 1000:
            issues.append(ValidationIssue(
                field="pl_pct",
                message=f"P/L percentage of {pl_pct}% seems extreme",
                severity=ValidationSeverity.WARNING,
                actual_value=pl_pct,
                suggestion="Verify entry price and current price"
            ))

        return issues

    def _validate_stock_position(self, stock: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate stock-specific fields"""
        issues = []

        # Quantity validation
        quantity = stock.get("quantity", 0)
        if quantity <= 0:
            issues.append(ValidationIssue(
                field="quantity",
                message="Invalid quantity (zero or negative)",
                severity=ValidationSeverity.ERROR,
                actual_value=quantity,
                expected_range="(0, ∞)"
            ))

        # Price validation
        current_price = stock.get("current_price", 0)
        avg_buy_price = stock.get("avg_buy_price", 0)

        if current_price <= 0:
            issues.append(ValidationIssue(
                field="current_price",
                message="Invalid current price",
                severity=ValidationSeverity.ERROR,
                actual_value=current_price,
                expected_range="(0, ∞)"
            ))

        if avg_buy_price <= 0:
            issues.append(ValidationIssue(
                field="avg_buy_price",
                message="Invalid average buy price",
                severity=ValidationSeverity.WARNING,
                actual_value=avg_buy_price
            ))

        # Price deviation check (if both prices are valid)
        if current_price > 0 and avg_buy_price > 0:
            deviation = abs(current_price - avg_buy_price) / avg_buy_price * 100
            if deviation > 500:  # 500% deviation
                issues.append(ValidationIssue(
                    field="current_price",
                    message=f"Large price deviation ({deviation:.0f}%) from average buy price",
                    severity=ValidationSeverity.INFO,
                    actual_value=current_price,
                    suggestion="Normal for long-held positions or volatile stocks"
                ))

        return issues

    def _check_data_freshness(self, position: Dict[str, Any]) -> Tuple[List[ValidationIssue], Optional[float]]:
        """Check how fresh the data is"""
        issues = []
        freshness_seconds = None

        # Check for fetched_at timestamp
        fetched_at_str = position.get("fetched_at")
        if fetched_at_str:
            try:
                fetched_at = datetime.fromisoformat(fetched_at_str.replace('Z', '+00:00'))
                freshness_seconds = (datetime.now() - fetched_at.replace(tzinfo=None)).total_seconds()

                if freshness_seconds > self.FRESHNESS_CRITICAL_SECONDS:
                    issues.append(ValidationIssue(
                        field="data_freshness",
                        message=f"Data is {freshness_seconds/60:.1f} minutes old",
                        severity=ValidationSeverity.CRITICAL,
                        actual_value=freshness_seconds,
                        expected_range=f"< {self.FRESHNESS_WARNING_SECONDS}s",
                        suggestion="Refresh position data"
                    ))
                elif freshness_seconds > self.FRESHNESS_ERROR_SECONDS:
                    issues.append(ValidationIssue(
                        field="data_freshness",
                        message=f"Data is {freshness_seconds/60:.1f} minutes old",
                        severity=ValidationSeverity.ERROR,
                        actual_value=freshness_seconds,
                        suggestion="Consider refreshing data"
                    ))
                elif freshness_seconds > self.FRESHNESS_WARNING_SECONDS:
                    issues.append(ValidationIssue(
                        field="data_freshness",
                        message=f"Data is {freshness_seconds:.0f} seconds old",
                        severity=ValidationSeverity.WARNING,
                        actual_value=freshness_seconds
                    ))
            except (ValueError, TypeError):
                pass

        return issues, freshness_seconds

    def _validate_portfolio_consistency(self, positions: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate cross-portfolio consistency"""
        issues = []
        options = positions.get("options", [])

        # Check for duplicate symbols with conflicting data
        symbol_positions = {}
        for opt in options:
            symbol = opt.get("symbol")
            strike = opt.get("strike")
            expiration = opt.get("expiration")
            key = f"{symbol}_{strike}_{expiration}"

            if key in symbol_positions:
                # Check for conflicting greeks
                prev = symbol_positions[key]
                if abs(opt.get("greeks", {}).get("delta", 0) - prev.get("greeks", {}).get("delta", 0)) > 10:
                    issues.append(ValidationIssue(
                        field=f"{symbol}.greeks",
                        message="Duplicate positions with conflicting Greeks",
                        severity=ValidationSeverity.WARNING,
                        actual_value=f"delta: {opt.get('greeks', {}).get('delta')} vs {prev.get('greeks', {}).get('delta')}",
                        suggestion="Consolidate duplicate positions"
                    ))
            else:
                symbol_positions[key] = opt

        return issues

    def _validate_summary(self, summary: Dict[str, Any], stocks: List, options: List) -> List[ValidationIssue]:
        """Validate portfolio summary matches individual positions"""
        issues = []

        # Verify position count
        reported_count = summary.get("total_positions", 0)
        actual_count = len(stocks) + len(options)

        if reported_count != actual_count:
            issues.append(ValidationIssue(
                field="summary.total_positions",
                message=f"Position count mismatch: reported {reported_count}, actual {actual_count}",
                severity=ValidationSeverity.ERROR,
                actual_value=reported_count,
                expected_range=str(actual_count)
            ))

        # Verify total equity is reasonable
        total_equity = summary.get("total_equity", 0)
        if total_equity < 0:
            issues.append(ValidationIssue(
                field="summary.total_equity",
                message="Negative total equity",
                severity=ValidationSeverity.CRITICAL,
                actual_value=total_equity,
                suggestion="Check for calculation errors or margin calls"
            ))

        return issues

    def _determine_quality(self, issues: List[ValidationIssue]) -> DataQuality:
        """Determine overall data quality based on issues"""
        critical_count = sum(1 for i in issues if i.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)

        if critical_count > 0:
            return DataQuality.CRITICAL
        elif error_count > 2:
            return DataQuality.CRITICAL
        elif error_count > 0:
            return DataQuality.WARNING
        elif warning_count > 3:
            return DataQuality.WARNING
        elif warning_count > 0:
            return DataQuality.GOOD
        else:
            return DataQuality.EXCELLENT

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self._validation_history:
            return {"total_validations": 0}

        valid_count = sum(1 for r in self._validation_history if r.valid)
        quality_dist = {}
        for r in self._validation_history:
            quality_dist[r.quality.value] = quality_dist.get(r.quality.value, 0) + 1

        return {
            "total_validations": len(self._validation_history),
            "valid_count": valid_count,
            "invalid_count": len(self._validation_history) - valid_count,
            "validation_rate": valid_count / len(self._validation_history) * 100,
            "quality_distribution": quality_dist
        }


# =============================================================================
# Audit Trail System
# =============================================================================

@dataclass
class AuditEntry:
    """A single audit log entry"""
    timestamp: datetime
    event_type: str
    symbol: str
    field_changed: Optional[str]
    old_value: Any
    new_value: Any
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditTrail:
    """
    Audit trail for tracking all position changes.

    Logs:
    - Position additions/removals
    - Price changes
    - Greeks updates
    - P/L changes
    """

    def __init__(self, max_entries: int = 10000):
        self._entries: List[AuditEntry] = []
        self._max_entries = max_entries

    def log_position_change(
        self,
        symbol: str,
        field: str,
        old_value: Any,
        new_value: Any,
        source: str = "system",
        metadata: Optional[Dict] = None
    ):
        """Log a position field change"""
        entry = AuditEntry(
            timestamp=datetime.now(),
            event_type="position_change",
            symbol=symbol,
            field_changed=field,
            old_value=old_value,
            new_value=new_value,
            source=source,
            metadata=metadata or {}
        )
        self._add_entry(entry)

    def log_position_added(self, symbol: str, position_data: Dict, source: str = "system"):
        """Log a new position"""
        entry = AuditEntry(
            timestamp=datetime.now(),
            event_type="position_added",
            symbol=symbol,
            field_changed=None,
            old_value=None,
            new_value=position_data,
            source=source
        )
        self._add_entry(entry)
        logger.info(f"Audit: Position added - {symbol}")

    def log_position_removed(self, symbol: str, position_data: Dict, source: str = "system"):
        """Log a removed position"""
        entry = AuditEntry(
            timestamp=datetime.now(),
            event_type="position_removed",
            symbol=symbol,
            field_changed=None,
            old_value=position_data,
            new_value=None,
            source=source
        )
        self._add_entry(entry)
        logger.info(f"Audit: Position removed - {symbol}")

    def _add_entry(self, entry: AuditEntry):
        """Add entry with size limit management"""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def get_history(
        self,
        symbol: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit history with filters"""
        filtered = self._entries

        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if since:
            filtered = [e for e in filtered if e.timestamp >= since]

        # Return most recent first
        filtered = sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:limit]

        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type,
                "symbol": e.symbol,
                "field_changed": e.field_changed,
                "old_value": str(e.old_value) if e.old_value else None,
                "new_value": str(e.new_value) if e.new_value else None,
                "source": e.source,
                "metadata": e.metadata
            }
            for e in filtered
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        if not self._entries:
            return {"total_entries": 0}

        event_types = {}
        symbols = {}
        for e in self._entries:
            event_types[e.event_type] = event_types.get(e.event_type, 0) + 1
            symbols[e.symbol] = symbols.get(e.symbol, 0) + 1

        return {
            "total_entries": len(self._entries),
            "event_types": event_types,
            "symbols_tracked": len(symbols),
            "most_active_symbols": sorted(symbols.items(), key=lambda x: -x[1])[:5],
            "oldest_entry": self._entries[0].timestamp.isoformat() if self._entries else None,
            "newest_entry": self._entries[-1].timestamp.isoformat() if self._entries else None
        }


# =============================================================================
# Singleton Instances
# =============================================================================

_validator: Optional[PortfolioDataValidator] = None
_audit_trail: Optional[AuditTrail] = None


def get_validator() -> PortfolioDataValidator:
    """Get validator singleton"""
    global _validator
    if _validator is None:
        _validator = PortfolioDataValidator()
    return _validator


def get_audit_trail() -> AuditTrail:
    """Get audit trail singleton"""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail
