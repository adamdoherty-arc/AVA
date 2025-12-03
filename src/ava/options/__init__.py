"""
AVA Options Module
==================

High-performance options analysis, pricing, and trading.

Modules:
- greeks_engine: Advanced Greeks calculations with vectorization
"""

from .greeks_engine import (
    AdvancedGreeksEngine,
    GreeksResult,
    OptionContract,
    OptionType,
    PositionSide,
    StrategyGreeksCalculator,
    get_greeks_engine,
    calculate_greeks,
    calculate_iv
)

__all__ = [
    'AdvancedGreeksEngine',
    'GreeksResult',
    'OptionContract',
    'OptionType',
    'PositionSide',
    'StrategyGreeksCalculator',
    'get_greeks_engine',
    'calculate_greeks',
    'calculate_iv'
]
