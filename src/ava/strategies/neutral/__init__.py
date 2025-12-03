"""
Neutral Options Strategies
==========================

Strategies designed for range-bound markets with minimal directional bias.

Strategies:
- Iron Condor: Sell OTM strangle, buy protective wings
- Iron Butterfly: ATM short straddle with protective wings
- Butterfly Spread: Three-strike profit zone strategy
- Straddle (Short): ATM puts and calls for premium collection
- Strangle (Short): OTM puts and calls for wider profit zone
"""

from .iron_condor import IronCondorStrategy

__all__ = [
    'IronCondorStrategy'
]
