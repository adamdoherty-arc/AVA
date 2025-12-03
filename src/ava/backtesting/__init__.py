"""
AVA Backtesting Engine
======================

High-performance options strategy backtesting with:
- Vectorized calculations for speed
- Support for all strategy types
- Monte Carlo simulations
- Comprehensive performance metrics
- Visual reports and analysis

Usage:
    from src.ava.backtesting import BacktestEngine, BacktestConfig

    config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2024-01-01',
        initial_capital=100000
    )

    engine = BacktestEngine(config)
    results = engine.run(WheelStrategy())
"""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    Trade,
    PerformanceMetrics
)

from .data_loader import (
    HistoricalDataLoader,
    OptionsDataSource
)

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'Trade',
    'PerformanceMetrics',
    'HistoricalDataLoader',
    'OptionsDataSource'
]
