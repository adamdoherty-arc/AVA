"""
AI-Powered Earnings Agent
=========================

Uses Claude to analyze earnings events and their impact
on options trading strategies.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import Field

from src.ava.agents.base.llm_agent import (
    LLMAgent,
    AgentOutputBase,
    AgentConfidence,
    AgentExecutionContext
)

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class EarningsEvent(AgentOutputBase):
    """Individual earnings event"""
    symbol: str = ""
    company_name: str = ""
    earnings_date: str = ""
    earnings_time: str = ""  # before_market, after_market, during_market
    days_until: int = 0

    # Estimates
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None

    # Historical
    last_eps_surprise: Optional[float] = None
    avg_move_after_earnings: Optional[float] = None
    beat_rate: Optional[float] = None

    # IV analysis
    current_iv: float = 0.0
    expected_move: float = 0.0
    iv_crush_estimate: float = 0.0

    # Risk assessment
    volatility_risk: str = ""  # low, medium, high
    surprise_potential: str = ""

    # Trading implications
    wheel_warning: bool = False
    suggested_action: str = ""


class EarningsImpact(AgentOutputBase):
    """Impact analysis for a single stock's earnings"""
    symbol: str = ""
    pre_earnings_strategy: str = ""
    post_earnings_strategy: str = ""
    avoid_positions_before: bool = False
    iv_play_opportunity: bool = False
    straddle_opportunity: bool = False


class EarningsOutput(AgentOutputBase):
    """Complete earnings analysis output"""
    analysis_date: str = ""

    # Summary
    events_this_week: int = 0
    high_impact_events: int = 0
    watchlist_affected: List[str] = Field(default_factory=list)

    # Upcoming earnings
    upcoming_earnings: List[EarningsEvent] = Field(default_factory=list)

    # By timeframe
    earnings_today: List[EarningsEvent] = Field(default_factory=list)
    earnings_this_week: List[EarningsEvent] = Field(default_factory=list)
    earnings_next_week: List[EarningsEvent] = Field(default_factory=list)

    # Impact analysis
    earnings_impacts: List[EarningsImpact] = Field(default_factory=list)

    # Market context
    sector_earnings_concentration: Dict[str, int] = Field(default_factory=dict)

    # Recommendations
    positions_to_close_before_earnings: List[str] = Field(default_factory=list)
    iv_crush_plays: List[str] = Field(default_factory=list)
    earnings_straddle_candidates: List[str] = Field(default_factory=list)

    # Warnings
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# EARNINGS AGENT
# =============================================================================

class AIEarningsAgent(LLMAgent[EarningsOutput]):
    """
    AI-powered earnings analysis agent.

    Analyzes earnings events and their impact on options
    trading strategies.

    Usage:
        agent = AIEarningsAgent()
        result = await agent.execute({
            "watchlist": ["AAPL", "MSFT", "NVDA"],
            "positions": [...],
            "earnings_calendar": [...],
            "lookforward_days": 14
        })
    """

    name = "ai_earnings"
    description = "Analyzes earnings impact on options strategies"
    output_model = EarningsOutput
    temperature = 0.3

    system_prompt = """You are an expert earnings analyst specializing in:

1. EARNINGS EVENT ANALYSIS:
   - Earnings date/time significance
   - Expected move calculations
   - Historical surprise patterns
   - Sector correlation effects

2. IV DYNAMICS AROUND EARNINGS:
   - IV ramp-up patterns pre-earnings
   - IV crush magnitude estimates
   - Term structure effects
   - Skew changes

3. OPTIONS STRATEGY IMPLICATIONS:
   - Wheel strategy warnings
   - Position management pre-earnings
   - IV crush opportunities
   - Straddle/strangle timing

4. HISTORICAL PATTERNS:
   - Beat/miss rates
   - Average post-earnings moves
   - Surprise magnitude history
   - Market reaction patterns

5. RISK MANAGEMENT:
   - Assignment risk during earnings
   - Gap risk assessment
   - Position sizing considerations
   - Hedge recommendations

Your role is to:
1. Identify upcoming earnings events
2. Assess impact on existing/planned positions
3. Flag wheel strategy warnings
4. Identify IV crush opportunities
5. Recommend position adjustments

Focus on actionable guidance for options traders."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build earnings analysis prompt"""
        watchlist = input_data.get('watchlist', [])
        positions = input_data.get('positions', [])
        earnings = input_data.get('earnings_calendar', [])
        lookforward = input_data.get('lookforward_days', 14)

        prompt = f"""## Earnings Analysis Request

### Analysis Parameters
- **Lookforward Period**: {lookforward} days
- **Watchlist**: {', '.join(watchlist) if watchlist else 'None provided'}
- **Open Positions**: {len(positions)} positions

### Upcoming Earnings Events
"""

        if earnings:
            # Sort by date
            sorted_earnings = sorted(earnings, key=lambda x: x.get('date', ''))

            for e in sorted_earnings[:20]:
                symbol = e.get('symbol', 'N/A')
                date = e.get('date', 'N/A')
                time = e.get('time', 'N/A')

                prompt += f"""
**{symbol}** - {date} ({time})
   - EPS Estimate: ${e.get('eps_estimate', 'N/A')}
   - Revenue Estimate: ${e.get('revenue_estimate', 'N/A')}
   - Last Surprise: {e.get('last_surprise', 'N/A')}%
   - Avg Move: {e.get('avg_move', 'N/A')}%
   - Current IV: {e.get('current_iv', 'N/A')}%
   - Expected Move: {e.get('expected_move', 'N/A')}%
"""
        else:
            prompt += "\nNo earnings data provided.\n"

        # Current positions
        if positions:
            prompt += "\n### Current Options Positions\n"
            for pos in positions[:10]:
                prompt += f"""
- {pos.get('symbol', 'N/A')}: {pos.get('strategy', 'N/A')} @ ${pos.get('strike', 0):.0f}
  Exp: {pos.get('expiration', 'N/A')} | Delta: {pos.get('delta', 0):.2f}
"""

        prompt += f"""

### Analysis Tasks

1. **Earnings Calendar Summary**:
   - Events today
   - Events this week
   - Events next week
   - High-impact events (large caps, market movers)

2. **Position Impact Analysis**:
   - Which positions have earnings before expiration?
   - Assignment risk assessment
   - Close/roll recommendations

3. **Wheel Strategy Warnings**:
   - Which stocks should NOT have new CSPs opened?
   - When is it safe to open positions post-earnings?

4. **IV Opportunities**:
   - IV crush plays (sell before, buy back after)
   - High IV rank + earnings = opportunity?
   - Straddle/strangle candidates

5. **Expected Move Analysis**:
   - Is implied move reasonable vs history?
   - Overpriced or underpriced events?

6. **Sector Concentration**:
   - Multiple earnings in same sector = correlation risk

7. **Specific Recommendations**:
   - Positions to close before earnings
   - New plays to consider
   - Risk warnings

Prioritize protecting existing positions over new opportunities.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> EarningsOutput:
        """Parse earnings analysis response"""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                return EarningsOutput(**data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> EarningsOutput:
        """Parse unstructured response"""
        earnings = input_data.get('earnings_calendar', [])
        positions = input_data.get('positions', [])

        today = datetime.now().date()

        # Categorize earnings
        today_list = []
        this_week = []
        next_week = []

        for e in earnings:
            try:
                date_str = e.get('date', '')
                if date_str:
                    e_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    days_until = (e_date - today).days

                    event = EarningsEvent(
                        symbol=e.get('symbol', ''),
                        company_name=e.get('company_name', ''),
                        earnings_date=date_str,
                        earnings_time=e.get('time', ''),
                        days_until=days_until,
                        eps_estimate=e.get('eps_estimate'),
                        current_iv=e.get('current_iv', 0),
                        expected_move=e.get('expected_move', 0),
                        wheel_warning=True
                    )

                    if days_until == 0:
                        today_list.append(event)
                    elif days_until <= 7:
                        this_week.append(event)
                    elif days_until <= 14:
                        next_week.append(event)
            except:
                continue

        # Find affected positions
        earning_symbols = {e.get('symbol', '') for e in earnings}
        affected = []
        close_before = []

        for pos in positions:
            pos_symbol = pos.get('symbol', '')
            if pos_symbol in earning_symbols:
                affected.append(pos_symbol)
                close_before.append(pos_symbol)

        # Warnings
        warnings = []
        if close_before:
            warnings.append(f"⚠️ {len(close_before)} positions have earnings before expiration")
        if len(today_list) > 3:
            warnings.append(f"⚠️ {len(today_list)} earnings today - elevated market volatility")

        return EarningsOutput(
            agent_name=self.name,
            analysis_date=today.isoformat(),
            events_this_week=len(this_week) + len(today_list),
            high_impact_events=len([e for e in earnings if e.get('market_cap', 0) > 50e9]),
            watchlist_affected=affected,
            upcoming_earnings=[
                EarningsEvent(
                    symbol=e.get('symbol', ''),
                    earnings_date=e.get('date', ''),
                    days_until=(datetime.strptime(e.get('date', today.isoformat()), '%Y-%m-%d').date() - today).days if e.get('date') else 0,
                    wheel_warning=True
                )
                for e in earnings[:10]
            ],
            earnings_today=today_list,
            earnings_this_week=this_week,
            earnings_next_week=next_week,
            positions_to_close_before_earnings=close_before,
            warnings=warnings,
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK EARNINGS CHECKER
# =============================================================================

class QuickEarningsChecker:
    """Rule-based quick earnings check"""

    @staticmethod
    def check_position(
        symbol: str,
        expiration_date: str,
        earnings_date: Optional[str] = None,
        expected_move_pct: float = 0
    ) -> Dict:
        """Check if position is affected by earnings"""
        if not earnings_date:
            return {
                'has_earnings': False,
                'warning': None,
                'action': 'safe'
            }

        try:
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
            earn_date = datetime.strptime(earnings_date, '%Y-%m-%d').date()
            today = datetime.now().date()

            days_to_earnings = (earn_date - today).days
            earnings_before_expiration = earn_date <= exp_date

            if earnings_before_expiration:
                if days_to_earnings <= 3:
                    return {
                        'has_earnings': True,
                        'warning': f'⚠️ URGENT: {symbol} earnings in {days_to_earnings} days!',
                        'action': 'close_immediately',
                        'expected_move': expected_move_pct,
                        'days_to_earnings': days_to_earnings
                    }
                elif days_to_earnings <= 7:
                    return {
                        'has_earnings': True,
                        'warning': f'⚠️ {symbol} earnings within week (before expiration)',
                        'action': 'close_or_roll',
                        'expected_move': expected_move_pct,
                        'days_to_earnings': days_to_earnings
                    }
                else:
                    return {
                        'has_earnings': True,
                        'warning': f'{symbol} has earnings before expiration',
                        'action': 'monitor',
                        'expected_move': expected_move_pct,
                        'days_to_earnings': days_to_earnings
                    }
            else:
                return {
                    'has_earnings': False,
                    'warning': None,
                    'action': 'safe',
                    'note': 'Earnings after expiration'
                }

        except Exception as e:
            return {
                'has_earnings': False,
                'warning': f'Could not parse dates: {e}',
                'action': 'unknown'
            }

    @staticmethod
    def should_open_position(
        symbol: str,
        earnings_date: Optional[str] = None,
        target_expiration: str = "",
        strategy: str = "csp"
    ) -> Dict:
        """Check if safe to open new position"""
        if not earnings_date:
            return {
                'safe': True,
                'recommendation': f'No earnings data - OK to open {strategy.upper()}'
            }

        try:
            today = datetime.now().date()
            earn_date = datetime.strptime(earnings_date, '%Y-%m-%d').date()
            days_to_earnings = (earn_date - today).days

            if target_expiration:
                exp_date = datetime.strptime(target_expiration, '%Y-%m-%d').date()
                if earn_date <= exp_date:
                    return {
                        'safe': False,
                        'recommendation': f'⚠️ DO NOT open {strategy.upper()} - earnings before expiration'
                    }

            if days_to_earnings <= 7:
                return {
                    'safe': False,
                    'recommendation': f'⚠️ Wait until after {symbol} earnings ({earnings_date})'
                }
            elif days_to_earnings <= 14:
                return {
                    'safe': True,
                    'recommendation': f'OK but monitor - earnings in {days_to_earnings} days',
                    'suggestion': 'Consider shorter DTE'
                }
            else:
                return {
                    'safe': True,
                    'recommendation': f'Safe to open - earnings {days_to_earnings} days away'
                }

        except:
            return {
                'safe': False,
                'recommendation': 'Could not verify earnings safety'
            }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Earnings Agent ===\n")

    async def test_agent():
        # Test quick checker
        print("1. Testing Quick Earnings Checker...")
        checker = QuickEarningsChecker()

        # Test position check
        result = checker.check_position(
            symbol="AAPL",
            expiration_date="2025-12-20",
            earnings_date="2025-12-05",
            expected_move_pct=4.5
        )
        print(f"\n   AAPL position check:")
        print(f"      Has Earnings: {result['has_earnings']}")
        print(f"      Warning: {result['warning']}")
        print(f"      Action: {result['action']}")

        # Test new position
        result2 = checker.should_open_position(
            symbol="MSFT",
            earnings_date="2025-12-10",
            target_expiration="2025-12-20",
            strategy="csp"
        )
        print(f"\n   MSFT new position check:")
        print(f"      Safe: {result2['safe']}")
        print(f"      Recommendation: {result2['recommendation']}")

        # Test agent structure
        print("\n2. Testing Agent Structure...")
        agent = AIEarningsAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Earnings Agent ready!")

    asyncio.run(test_agent())
