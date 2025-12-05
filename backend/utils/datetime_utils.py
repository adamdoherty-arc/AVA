"""
Timezone-Aware DateTime Utilities
=================================

Centralized datetime utilities for consistent timezone handling
across the AVA Trading Platform.

Usage:
    from backend.utils.datetime_utils import utc_now, local_now, to_utc

All datetime operations should use these utilities to ensure
consistent timezone handling, especially for:
- Database timestamps
- API responses
- Cache TTL calculations
- Scheduling tasks

Author: AVA Trading Platform
Updated: 2025-12-03
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import pytz


# Default timezone for the trading platform (US Eastern for market hours)
DEFAULT_TIMEZONE = pytz.timezone('America/New_York')
UTC = timezone.utc


def utc_now() -> datetime:
    """
    Get current UTC time with timezone info.

    Use this for:
    - Database timestamps
    - Cache expiration calculations
    - API response timestamps

    Returns:
        datetime: Current UTC time with tzinfo set
    """
    return datetime.now(UTC)


def local_now(tz: Optional[str] = None) -> datetime:
    """
    Get current local time with timezone info.

    Args:
        tz: Timezone string (e.g., 'America/New_York').
            Defaults to US Eastern (market timezone).

    Returns:
        datetime: Current local time with tzinfo set
    """
    if tz:
        local_tz = pytz.timezone(tz)
    else:
        local_tz = DEFAULT_TIMEZONE
    return datetime.now(local_tz)


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.

    Args:
        dt: datetime object (aware or naive)

    Returns:
        datetime: UTC datetime with tzinfo
    """
    if dt.tzinfo is None:
        # Assume naive datetime is in local timezone
        dt = DEFAULT_TIMEZONE.localize(dt)
    return dt.astimezone(UTC)


def to_local(dt: datetime, tz: Optional[str] = None) -> datetime:
    """
    Convert datetime to local timezone.

    Args:
        dt: datetime object (aware or naive)
        tz: Target timezone string. Defaults to US Eastern.

    Returns:
        datetime: Local datetime with tzinfo
    """
    if tz:
        local_tz = pytz.timezone(tz)
    else:
        local_tz = DEFAULT_TIMEZONE

    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = pytz.utc.localize(dt)
    return dt.astimezone(local_tz)


def is_market_open() -> bool:
    """
    Check if US stock market is currently open.

    Returns:
        bool: True if market is open (9:30 AM - 4:00 PM ET, weekdays)
    """
    now = local_now()

    # Check if it's a weekday (0=Monday, 6=Sunday)
    if now.weekday() >= 5:
        return False

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


def market_hours_until_open() -> Optional[timedelta]:
    """
    Calculate time until market opens.

    Returns:
        timedelta: Time until next market open, or None if market is open
    """
    if is_market_open():
        return None

    now = local_now()

    # Find next market open
    next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

    # If we're past today's open, move to tomorrow
    if now.hour >= 9 and now.minute >= 30:
        next_open += timedelta(days=1)

    # Skip weekends
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    return next_open - now


def format_iso(dt: datetime) -> str:
    """
    Format datetime as ISO 8601 string with timezone.

    Args:
        dt: datetime object

    Returns:
        str: ISO 8601 formatted string
    """
    if dt.tzinfo is None:
        dt = to_utc(dt)
    return dt.isoformat()


def parse_iso(iso_string: str) -> datetime:
    """
    Parse ISO 8601 string to timezone-aware datetime.

    Args:
        iso_string: ISO 8601 formatted string

    Returns:
        datetime: Parsed datetime with tzinfo
    """
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt


def seconds_since(dt: datetime) -> float:
    """
    Calculate seconds elapsed since given datetime.

    Args:
        dt: Start datetime

    Returns:
        float: Seconds elapsed
    """
    if dt.tzinfo is None:
        dt = to_utc(dt)
    return (utc_now() - dt).total_seconds()


def is_stale(dt: datetime, max_age_seconds: int) -> bool:
    """
    Check if datetime is older than max_age_seconds.

    Args:
        dt: datetime to check
        max_age_seconds: Maximum age in seconds

    Returns:
        bool: True if datetime is stale
    """
    return seconds_since(dt) > max_age_seconds
