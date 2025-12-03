"""
DTE Calculator - Validates Days To Expiration calculations

Validates:
- DTE calculations
- Expiration date parsing
- Timezone handling
- Weekend/holiday edge cases
"""

from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class DTECalculator:
    """
    Validates DTE (Days To Expiration) calculations.

    Common issues caught:
    - Off-by-one errors
    - Timezone issues (UTC vs local)
    - Weekend counting errors
    - Holiday handling
    """

    # US market holidays (simplified - major holidays only)
    US_MARKET_HOLIDAYS_2025 = [
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # MLK Day
        date(2025, 2, 17),  # Presidents Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving
        date(2025, 12, 25), # Christmas
    ]

    def __init__(self, include_current_day: bool = True):
        """
        Initialize DTE calculator

        Args:
            include_current_day: Whether to include today in DTE count
        """
        self.include_current_day = include_current_day

    def calculate_dte(
        self,
        expiration_date: Union[str, date, datetime],
        reference_date: Optional[Union[str, date, datetime]] = None,
    ) -> int:
        """
        Calculate Days To Expiration

        Args:
            expiration_date: Option expiration date
            reference_date: Reference date (default: today)

        Returns:
            Number of days until expiration
        """
        # Parse expiration date
        if isinstance(expiration_date, str):
            exp_date = self._parse_date(expiration_date)
        elif isinstance(expiration_date, datetime):
            exp_date = expiration_date.date()
        else:
            exp_date = expiration_date

        # Parse reference date
        if reference_date is None:
            ref_date = date.today()
        elif isinstance(reference_date, str):
            ref_date = self._parse_date(reference_date)
        elif isinstance(reference_date, datetime):
            ref_date = reference_date.date()
        else:
            ref_date = reference_date

        # Calculate difference
        delta = (exp_date - ref_date).days

        # Apply include_current_day logic
        if self.include_current_day and delta >= 0:
            return delta
        elif not self.include_current_day and delta > 0:
            return delta - 1
        else:
            return max(0, delta)

    def calculate_trading_days_dte(
        self,
        expiration_date: Union[str, date, datetime],
        reference_date: Optional[Union[str, date, datetime]] = None,
    ) -> int:
        """
        Calculate trading days until expiration (excludes weekends/holidays)

        Args:
            expiration_date: Option expiration date
            reference_date: Reference date (default: today)

        Returns:
            Number of trading days until expiration
        """
        # Parse dates
        if isinstance(expiration_date, str):
            exp_date = self._parse_date(expiration_date)
        elif isinstance(expiration_date, datetime):
            exp_date = expiration_date.date()
        else:
            exp_date = expiration_date

        if reference_date is None:
            ref_date = date.today()
        elif isinstance(reference_date, str):
            ref_date = self._parse_date(reference_date)
        elif isinstance(reference_date, datetime):
            ref_date = reference_date.date()
        else:
            ref_date = reference_date

        if exp_date <= ref_date:
            return 0

        # Count trading days
        trading_days = 0
        current = ref_date + timedelta(days=1)

        while current <= exp_date:
            if self._is_trading_day(current):
                trading_days += 1
            current += timedelta(days=1)

        return trading_days

    def _is_trading_day(self, d: date) -> bool:
        """Check if date is a trading day"""
        # Weekend check
        if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Holiday check
        if d in self.US_MARKET_HOLIDAYS_2025:
            return False

        return True

    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object"""
        # Try common formats
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%fZ',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.split('T')[0] if 'T' not in fmt else date_str, fmt).date()
            except ValueError:
                continue

        raise ValueError(f"Unable to parse date: {date_str}")

    def validate_dte(
        self,
        reported_dte: int,
        expiration_date: Union[str, date, datetime],
        reference_date: Optional[Union[str, date, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate reported DTE value

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        try:
            expected_dte = self.calculate_dte(expiration_date, reference_date)
        except ValueError as e:
            issues.append({
                'field': 'expiration_date',
                'error': 'parse_error',
                'message': f"Could not parse expiration date: {e}"
            })
            return issues

        # Check exact match
        if reported_dte != expected_dte:
            # Identify error type
            error_type = self._identify_dte_error(
                reported_dte, expected_dte, expiration_date, reference_date
            )

            issues.append({
                'field': 'dte',
                'expected': expected_dte,
                'reported': reported_dte,
                'difference': reported_dte - expected_dte,
                'error_type': error_type,
                'message': f"DTE mismatch: expected {expected_dte}, got {reported_dte}"
                          f" ({error_type})"
            })

        return issues

    def _identify_dte_error(
        self,
        reported: int,
        expected: int,
        expiration_date: Union[str, date, datetime],
        reference_date: Optional[Union[str, date, datetime]],
    ) -> str:
        """Identify the type of DTE calculation error"""

        diff = reported - expected

        # Off-by-one (common)
        if abs(diff) == 1:
            return 'off_by_one'

        # Possible timezone issue (can cause +/- 1 day shift)
        if abs(diff) == 1:
            return 'possible_timezone_issue'

        # Trading days vs calendar days
        trading_dte = self.calculate_trading_days_dte(expiration_date, reference_date)
        if reported == trading_dte:
            return 'using_trading_days'

        # Wrong reference date (maybe using yesterday?)
        yesterday_dte = self.calculate_dte(
            expiration_date,
            (date.today() - timedelta(days=1))
        )
        if reported == yesterday_dte:
            return 'stale_reference_date'

        return 'unknown'

    def get_expiration_category(self, dte: int) -> str:
        """
        Categorize option by DTE

        Returns:
            Category string: '0dte', 'weekly', 'monthly', 'quarterly', 'leaps'
        """
        if dte == 0:
            return '0dte'
        elif dte <= 7:
            return 'weekly'
        elif dte <= 45:
            return 'monthly'
        elif dte <= 180:
            return 'quarterly'
        else:
            return 'leaps'

    def is_expiring_soon(self, dte: int, threshold: int = 7) -> bool:
        """Check if option is expiring soon"""
        return 0 <= dte <= threshold

    def get_friday_expiration(self, reference_date: Optional[date] = None) -> date:
        """Get next Friday expiration date"""
        ref = reference_date or date.today()

        # Days until Friday (Friday = 4)
        days_until_friday = (4 - ref.weekday()) % 7
        if days_until_friday == 0:
            # If today is Friday, get next Friday
            days_until_friday = 7

        return ref + timedelta(days=days_until_friday)
