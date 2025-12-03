"""
AVA Options Strategies Module
=============================

Comprehensive options strategy library with:
- Income strategies (Wheel, Covered Calls, CSPs)
- Spread strategies (Verticals, Calendars, Diagonals)
- Neutral strategies (Iron Condors, Butterflies, Straddles)
- Directional strategies (Long Calls/Puts, Collars)
- Volatility strategies (Straddles, Strangles)
- Advanced strategies (0DTE, Gamma scalping)

Usage:
    from src.ava.strategies import WheelStrategy, IronCondorStrategy
    from src.ava.strategies import StrategyRegistry

    # Create strategy instance
    wheel = WheelStrategy(target_delta=0.30)

    # Or use registry
    wheel = StrategyRegistry.create('Wheel Strategy')
"""

# Base classes and data structures
from .base import (
    OptionsStrategy,
    StrategyType,
    OptionType,
    OptionSide,
    StrategySetup,
    OptionLeg,
    EntrySignal,
    ExitSignal,
    ExitReason,
    SignalStrength,
    StrategyResult,
    MarketContext,
    OptionsChain,
    StrategyRegistry,
    register_strategy
)

# Income strategies
from .income import WheelStrategy

# Neutral strategies
from .neutral import IronCondorStrategy

# Volatility strategies
from .volatility import (
    LongStraddleStrategy,
    ShortStraddleStrategy,
    LongStrangleStrategy,
    ShortStrangleStrategy
)

# Spread strategies
from .spreads import (
    CalendarSpreadStrategy,
    DiagonalSpreadStrategy,
    DoubleCalendarStrategy,
    BullPutSpreadStrategy,
    BearCallSpreadStrategy,
    BullCallSpreadStrategy,
    BearPutSpreadStrategy
)

# Advanced strategies
from .advanced import (
    ZeroDTEIronCondorStrategy,
    ZeroDTECreditSpreadStrategy,
    GammaScalpingStrategy
)

__all__ = [
    # Base
    'OptionsStrategy',
    'StrategyType',
    'OptionType',
    'OptionSide',
    'StrategySetup',
    'OptionLeg',
    'EntrySignal',
    'ExitSignal',
    'ExitReason',
    'SignalStrength',
    'StrategyResult',
    'MarketContext',
    'OptionsChain',
    'StrategyRegistry',
    'register_strategy',

    # Income
    'WheelStrategy',

    # Neutral
    'IronCondorStrategy',

    # Volatility
    'LongStraddleStrategy',
    'ShortStraddleStrategy',
    'LongStrangleStrategy',
    'ShortStrangleStrategy',

    # Spreads
    'CalendarSpreadStrategy',
    'DiagonalSpreadStrategy',
    'DoubleCalendarStrategy',
    'BullPutSpreadStrategy',
    'BearCallSpreadStrategy',
    'BullCallSpreadStrategy',
    'BearPutSpreadStrategy',

    # Advanced
    'ZeroDTEIronCondorStrategy',
    'ZeroDTECreditSpreadStrategy',
    'GammaScalpingStrategy'
]


def list_all_strategies() -> dict:
    """List all available strategies organized by type"""
    return {
        'income': [
            'WheelStrategy',
        ],
        'neutral': [
            'IronCondorStrategy',
        ],
        'volatility': [
            'LongStraddleStrategy',
            'ShortStraddleStrategy',
            'LongStrangleStrategy',
            'ShortStrangleStrategy',
        ],
        'spreads': [
            'CalendarSpreadStrategy',
            'DiagonalSpreadStrategy',
            'DoubleCalendarStrategy',
            'BullPutSpreadStrategy',
            'BearCallSpreadStrategy',
            'BullCallSpreadStrategy',
            'BearPutSpreadStrategy',
        ],
        'advanced': [
            'ZeroDTEIronCondorStrategy',
            'ZeroDTECreditSpreadStrategy',
            'GammaScalpingStrategy',
        ]
    }
