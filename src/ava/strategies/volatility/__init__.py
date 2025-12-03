"""
Volatility Options Strategies
=============================

Strategies designed to profit from volatility expansion or contraction.

Strategies:
- Straddle: ATM call + ATM put for large moves
- Strangle: OTM call + OTM put for large moves (cheaper)
- Long Straddle: Buy volatility before events
- Short Straddle: Sell volatility in high IV environments
"""

from .straddle_strangle import (
    LongStraddleStrategy,
    ShortStraddleStrategy,
    LongStrangleStrategy,
    ShortStrangleStrategy
)

__all__ = [
    'LongStraddleStrategy',
    'ShortStraddleStrategy',
    'LongStrangleStrategy',
    'ShortStrangleStrategy'
]
