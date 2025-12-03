"""
Spread Options Strategies
=========================

Defined risk spread strategies including verticals, calendars, and diagonals.

Strategies:
- Calendar Spread: Same strike, different expirations
- Diagonal Spread: Different strike and expiration
- Vertical Spreads: Bull/Bear call/put spreads
"""

from .calendar_diagonal import (
    CalendarSpreadStrategy,
    DiagonalSpreadStrategy,
    DoubleCalendarStrategy
)

from .verticals import (
    BullPutSpreadStrategy,
    BearCallSpreadStrategy,
    BullCallSpreadStrategy,
    BearPutSpreadStrategy
)

__all__ = [
    'CalendarSpreadStrategy',
    'DiagonalSpreadStrategy',
    'DoubleCalendarStrategy',
    'BullPutSpreadStrategy',
    'BearCallSpreadStrategy',
    'BullCallSpreadStrategy',
    'BearPutSpreadStrategy'
]
