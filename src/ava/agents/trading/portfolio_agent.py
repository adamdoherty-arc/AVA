"""
Portfolio Agent - Comprehensive Portfolio Management and Analysis
Integrates with Robinhood API and Local LLM for intelligent portfolio insights
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

from ...core.agent_base import BaseAgent, AgentState
from langchain_core.tools import tool
from src.services.robinhood_client import get_robinhood_client, RobinhoodClient
from src.magnus_local_llm import get_magnus_llm, TaskComplexity

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class PortfolioMetrics:
    """Portfolio-wide metrics and analytics"""
    total_value: float
    cash: float
    buying_power: float
    total_positions_value: float
    total_pnl: float
    total_pnl_pct: float
    num_positions: int
    num_stock_positions: int
    num_options_positions: int

    # Greeks exposure (aggregated)
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float

    # Risk metrics
    concentration_risk: float  # Largest position as % of portfolio
    sector_diversity: float  # Number of unique sectors
    options_exposure_pct: float  # Options value / total value

    # Performance
    daily_pnl: Optional[float] = None
    weekly_pnl: Optional[float] = None
    monthly_pnl: Optional[float] = None

    timestamp: str = ""


@dataclass
class SectorAllocation:
    """Sector allocation breakdown"""
    sector: str
    value: float
    percentage: float
    num_positions: int
    symbols: List[str]


@dataclass
class RiskAssessment:
    """Portfolio risk assessment"""
    overall_risk_level: str  # 'low', 'medium', 'high', 'critical'
    risk_score: int  # 0-100
    concentration_issues: List[str]
    correlation_warnings: List[str]
    greeks_warnings: List[str]
    liquidity_concerns: List[str]
    recommendations: List[str]


# =============================================================================
# Tools
# =============================================================================

@tool
def get_portfolio_summary_tool() -> str:
    """Get comprehensive portfolio summary with metrics and positions"""
    try:
        from src.services.robinhood_client import get_robinhood_client

        client = get_robinhood_client()
        account = client.get_account_info()
        positions = client.get_positions()

        summary = f"""Portfolio Summary:

Total Value: ${account.get('portfolio_value', 0):,.2f}
Cash: ${account.get('cash', 0):,.2f}
Buying Power: ${account.get('buying_power', 0):,.2f}
Day Trades Used: {account.get('day_trade_count', 0)}

Positions: {len(positions)} total
- Stock positions: {len([p for p in positions if p['type'] == 'stock'])}
- Options positions: {len([p for p in positions if p['type'] == 'option'])}

Top 5 Positions:
"""
        # Sort by market value
        sorted_positions = sorted(positions, key=lambda x: abs(x.get('market_value', 0)), reverse=True)
        for i, pos in enumerate(sorted_positions[:5], 1):
            summary += f"{i}. {pos['symbol']}: ${pos.get('market_value', 0):,.2f} ({pos.get('total_return_pct', 0):.2f}%)\n"

        return summary
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        return f"Error retrieving portfolio: {str(e)}"


@tool
def calculate_greeks_exposure_tool() -> str:
    """Calculate aggregated Greeks exposure across all options positions"""
    try:
        from src.services.robinhood_client import get_robinhood_client

        client = get_robinhood_client()
        positions = client.get_options_positions()

        # For now, return placeholder - full Greeks calculation requires market data
        return f"Greeks exposure for {len(positions)} options positions (detailed calculation in progress)"
    except Exception as e:
        return f"Error calculating Greeks: {str(e)}"


# =============================================================================
# Portfolio Agent
# =============================================================================

class PortfolioAgent(BaseAgent):
    """
    Portfolio Agent - Comprehensive Portfolio Management

    Capabilities:
    - Real-time portfolio metrics and valuations
    - Greeks exposure analysis (delta, gamma, theta, vega)
    - Sector allocation and diversification analysis
    - Risk assessment and concentration analysis
    - AI-powered rebalancing recommendations
    - Hedging strategy suggestions
    - Performance tracking and attribution

    Integrations:
    - Robinhood API for position data
    - Local LLM for intelligent analysis
    - Magnus trading strategies
    """

    def __init__(self, use_huggingface: bool = False):
        """Initialize Portfolio Agent"""
        tools = [
            get_portfolio_summary_tool,
            calculate_greeks_exposure_tool
        ]

        super().__init__(
            name="portfolio_agent",
            description="Manages and analyzes portfolio positions, risk, Greeks, and provides AI-powered recommendations",
            tools=tools,
            use_huggingface=use_huggingface
        )

        self.metadata['capabilities'] = [
            'portfolio_summary',
            'position_tracking',
            'greeks_analysis',
            'sector_allocation',
            'risk_assessment',
            'pnl_analysis',
            'performance_metrics',
            'rebalancing_recommendations',
            'hedging_strategies',
            'correlation_analysis'
        ]

        # Initialize clients
        self.robinhood_client: Optional[RobinhoodClient] = None
        self.llm = None

        # Cache for performance
        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)

        logger.info("PortfolioAgent initialized with Robinhood and Local LLM integration")

    def _get_robinhood_client(self) -> RobinhoodClient:
        """Get or initialize Robinhood client"""
        if self.robinhood_client is None:
            self.robinhood_client = get_robinhood_client()
            # Ensure logged in
            if not self.robinhood_client.logged_in:
                self.robinhood_client.login()
        return self.robinhood_client

    def _get_llm(self) -> None:
        """Get or initialize Local LLM"""
        if self.llm is None:
            self.llm = get_magnus_llm()
        return self.llm

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            if key in self._cache_expiry:
                if datetime.now() < self._cache_expiry[key]:
                    logger.debug(f"Cache hit for {key}")
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._cache_expiry[key]
        return None

    def _set_cache(self, key: str, value: Any):
        """Set cached value with expiry"""
        self._cache[key] = value
        self._cache_expiry[key] = datetime.now() + self._cache_ttl

    # =========================================================================
    # Core Portfolio Metrics
    # =========================================================================

    async def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Get comprehensive portfolio metrics

        Returns:
            PortfolioMetrics with all key metrics
        """
        try:
            # Check cache
            cached = self._get_cached('portfolio_metrics')
            if cached:
                return cached

            client = self._get_robinhood_client()

            # Get account info and positions
            account = await asyncio.to_thread(client.get_account_info)
            positions = await asyncio.to_thread(client.get_positions)

            # Separate stock and option positions
            stock_positions = [p for p in positions if p['type'] == 'stock']
            options_positions = [p for p in positions if p['type'] == 'option']

            # Calculate total P&L
            total_pnl = sum(p.get('total_return', 0) for p in positions)
            total_positions_value = sum(p.get('market_value', 0) for p in positions)

            # Calculate total P&L percentage
            total_cost_basis = sum(
                p.get('avg_cost', 0) * p.get('quantity', 0)
                for p in stock_positions
            )
            total_pnl_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0

            # Calculate Greeks exposure
            greeks = await self._calculate_greeks_exposure(options_positions)

            # Calculate concentration risk
            concentration = await self._calculate_concentration_risk(positions)

            # Calculate sector diversity (approximate)
            unique_symbols = set(p['symbol'] for p in positions)
            sector_diversity = len(unique_symbols)  # Simplified - would need sector data

            # Options exposure
            options_value = sum(p.get('market_value', 0) for p in options_positions)
            options_exposure_pct = (
                (options_value / total_positions_value * 100)
                if total_positions_value > 0 else 0
            )

            metrics = PortfolioMetrics(
                total_value=account.get('portfolio_value', 0),
                cash=account.get('cash', 0),
                buying_power=account.get('buying_power', 0),
                total_positions_value=total_positions_value,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                num_positions=len(positions),
                num_stock_positions=len(stock_positions),
                num_options_positions=len(options_positions),
                net_delta=greeks['net_delta'],
                net_gamma=greeks['net_gamma'],
                net_theta=greeks['net_theta'],
                net_vega=greeks['net_vega'],
                concentration_risk=concentration,
                sector_diversity=sector_diversity,
                options_exposure_pct=options_exposure_pct,
                timestamp=datetime.now().isoformat()
            )

            # Cache result
            self._set_cache('portfolio_metrics', metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            raise

    async def _calculate_greeks_exposure(self, options_positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate aggregated Greeks exposure across all options

        Args:
            options_positions: List of option position dictionaries

        Returns:
            Dict with net delta, gamma, theta, vega
        """
        try:
            # For now, use simplified Greeks estimation
            # In production, would fetch actual Greeks from options data API

            net_delta = 0.0
            net_gamma = 0.0
            net_theta = 0.0
            net_vega = 0.0

            for pos in options_positions:
                quantity = pos.get('quantity', 0)
                position_type = pos.get('position_type', 'long')  # long or short
                option_type = pos.get('option_type', 'put')  # call or put

                # Simplified Greeks estimation
                # Delta: ~0.5 for ATM, adjust for ITM/OTM
                strike = pos.get('strike_price', 0)
                current_price = pos.get('current_price', 0)

                if current_price > 0:
                    moneyness = strike / current_price if current_price > 0 else 1.0

                    # Estimate delta (very simplified)
                    if option_type == 'call':
                        delta_per_contract = max(0.1, min(0.9, 2 - moneyness))
                    else:  # put
                        delta_per_contract = -max(0.1, min(0.9, moneyness - 0.5))

                    # Adjust for position type
                    multiplier = -1 if position_type == 'short' else 1

                    net_delta += delta_per_contract * quantity * 100 * multiplier

                    # Simplified gamma (highest for ATM)
                    gamma_per_contract = max(0, 0.05 * (1 - abs(moneyness - 1)))
                    net_gamma += gamma_per_contract * quantity * 100 * abs(multiplier)

                    # Simplified theta (based on DTE - would need actual expiration)
                    theta_per_contract = -0.02  # Approximate daily decay
                    net_theta += theta_per_contract * quantity * 100 * multiplier

                    # Simplified vega
                    vega_per_contract = 0.10
                    net_vega += vega_per_contract * quantity * 100 * abs(multiplier)

            return {
                'net_delta': round(net_delta, 2),
                'net_gamma': round(net_gamma, 4),
                'net_theta': round(net_theta, 2),
                'net_vega': round(net_vega, 2)
            }

        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {
                'net_delta': 0.0,
                'net_gamma': 0.0,
                'net_theta': 0.0,
                'net_vega': 0.0
            }

    async def _calculate_concentration_risk(self, positions: List[Dict]) -> float:
        """
        Calculate concentration risk (largest position as % of portfolio)

        Args:
            positions: List of all positions

        Returns:
            Concentration percentage (0-100)
        """
        if not positions:
            return 0.0

        total_value = sum(abs(p.get('market_value', 0)) for p in positions)
        if total_value == 0:
            return 0.0

        largest_position_value = max(abs(p.get('market_value', 0)) for p in positions)
        concentration = (largest_position_value / total_value) * 100

        return round(concentration, 2)

    # =========================================================================
    # Sector Allocation
    # =========================================================================

    async def get_sector_allocation(self) -> List[SectorAllocation]:
        """
        Get sector allocation breakdown

        Returns:
            List of SectorAllocation objects
        """
        try:
            # Check cache
            cached = self._get_cached('sector_allocation')
            if cached:
                return cached

            client = self._get_robinhood_client()
            positions = await asyncio.to_thread(client.get_positions)

            # Group by symbol first (would need sector mapping in production)
            # For now, simplified by symbol
            symbol_allocation: Dict[str, Dict] = {}
            total_value = sum(abs(p.get('market_value', 0)) for p in positions)

            for pos in positions:
                symbol = pos['symbol']
                if symbol not in symbol_allocation:
                    symbol_allocation[symbol] = {
                        'value': 0.0,
                        'positions': []
                    }

                symbol_allocation[symbol]['value'] += abs(pos.get('market_value', 0))
                symbol_allocation[symbol]['positions'].append(pos)

            # Convert to SectorAllocation (using symbol as sector for now)
            allocations = []
            for symbol, data in symbol_allocation.items():
                percentage = (data['value'] / total_value * 100) if total_value > 0 else 0

                allocation = SectorAllocation(
                    sector=symbol,  # Would be actual sector in production
                    value=data['value'],
                    percentage=round(percentage, 2),
                    num_positions=len(data['positions']),
                    symbols=[symbol]
                )
                allocations.append(allocation)

            # Sort by value descending
            allocations.sort(key=lambda x: x.value, reverse=True)

            # Cache result
            self._set_cache('sector_allocation', allocations)

            return allocations

        except Exception as e:
            logger.error(f"Error calculating sector allocation: {e}")
            return []

    # =========================================================================
    # Risk Assessment
    # =========================================================================

    async def assess_portfolio_risk(self) -> RiskAssessment:
        """
        Comprehensive portfolio risk assessment

        Returns:
            RiskAssessment with risk level and recommendations
        """
        try:
            # Get metrics
            metrics = await self.get_portfolio_metrics()
            allocations = await self.get_sector_allocation()

            # Initialize risk factors
            risk_score = 0
            concentration_issues = []
            greeks_warnings = []
            liquidity_concerns = []
            recommendations = []

            # Check concentration risk
            if metrics.concentration_risk > 50:
                risk_score += 30
                concentration_issues.append(
                    f"High concentration risk: {metrics.concentration_risk:.1f}% in single position"
                )
                recommendations.append("Consider diversifying - largest position exceeds 50% of portfolio")
            elif metrics.concentration_risk > 30:
                risk_score += 15
                concentration_issues.append(
                    f"Moderate concentration risk: {metrics.concentration_risk:.1f}% in single position"
                )

            # Check Greeks exposure
            if abs(metrics.net_delta) > 100:
                risk_score += 20
                greeks_warnings.append(
                    f"High delta exposure: {metrics.net_delta:.0f} (consider hedging)"
                )
                recommendations.append(f"Net delta of {metrics.net_delta:.0f} suggests directional risk")

            if abs(metrics.net_theta) > 50:
                greeks_warnings.append(
                    f"High theta exposure: ${metrics.net_theta:.2f}/day time decay"
                )

            # Check options exposure
            if metrics.options_exposure_pct > 50:
                risk_score += 25
                liquidity_concerns.append(
                    f"Options represent {metrics.options_exposure_pct:.1f}% of portfolio"
                )
                recommendations.append("High options exposure - ensure adequate liquidity buffer")

            # Check number of positions
            if metrics.num_positions < 3:
                risk_score += 15
                concentration_issues.append("Low position diversity (< 3 positions)")
                recommendations.append("Consider adding positions to improve diversification")

            # Determine overall risk level
            if risk_score >= 70:
                risk_level = "critical"
            elif risk_score >= 50:
                risk_level = "high"
            elif risk_score >= 25:
                risk_level = "medium"
            else:
                risk_level = "low"

            return RiskAssessment(
                overall_risk_level=risk_level,
                risk_score=min(100, risk_score),
                concentration_issues=concentration_issues,
                correlation_warnings=[],  # Would need correlation analysis
                greeks_warnings=greeks_warnings,
                liquidity_concerns=liquidity_concerns,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return RiskAssessment(
                overall_risk_level="unknown",
                risk_score=0,
                concentration_issues=[],
                correlation_warnings=[],
                greeks_warnings=[],
                liquidity_concerns=[],
                recommendations=["Error assessing risk - manual review recommended"]
            )

    # =========================================================================
    # AI-Powered Analysis
    # =========================================================================

    async def generate_ai_recommendations(self, context: Optional[Dict] = None) -> str:
        """
        Generate AI-powered portfolio recommendations using Local LLM

        Args:
            context: Optional context (market conditions, user preferences, etc.)

        Returns:
            Detailed AI analysis and recommendations
        """
        try:
            # Get portfolio data
            metrics = await self.get_portfolio_metrics()
            risk = await self.assess_portfolio_risk()
            allocations = await self.get_sector_allocation()

            # Build comprehensive prompt
            prompt = f"""Analyze this portfolio and provide actionable recommendations:

PORTFOLIO METRICS:
- Total Value: ${metrics.total_value:,.2f}
- Cash Available: ${metrics.cash:,.2f}
- Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:.2f}%)
- Positions: {metrics.num_positions} total ({metrics.num_stock_positions} stocks, {metrics.num_options_positions} options)

GREEKS EXPOSURE:
- Net Delta: {metrics.net_delta:.2f}
- Net Theta: ${metrics.net_theta:.2f}/day
- Net Gamma: {metrics.net_gamma:.4f}
- Net Vega: {metrics.net_vega:.2f}

RISK ASSESSMENT:
- Risk Level: {risk.overall_risk_level.upper()}
- Risk Score: {risk.risk_score}/100
- Concentration Risk: {metrics.concentration_risk:.1f}%
- Options Exposure: {metrics.options_exposure_pct:.1f}%

TOP POSITIONS:
"""
            for i, allocation in enumerate(allocations[:5], 1):
                prompt += f"{i}. {allocation.sector}: ${allocation.value:,.2f} ({allocation.percentage:.1f}%)\n"

            if risk.recommendations:
                prompt += f"\nKEY ISSUES:\n"
                for rec in risk.recommendations:
                    prompt += f"- {rec}\n"

            prompt += """
Please provide:
1. Overall portfolio health assessment
2. Specific rebalancing recommendations
3. Suggested hedging strategies (if needed)
4. Risk mitigation priorities
5. Opportunities to optimize returns

Focus on actionable, specific recommendations aligned with the wheel strategy (CSP → assignment → covered calls).
"""

            # Query Local LLM
            llm = self._get_llm()
            analysis = await asyncio.to_thread(
                llm.query,
                prompt=prompt,
                complexity=TaskComplexity.BALANCED,
                use_trading_context=True,
                max_tokens=2000
            )

            return analysis

        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            return f"Error generating recommendations: {str(e)}\n\nPlease review portfolio manually."

    async def suggest_hedging_strategies(self) -> str:
        """
        Suggest specific hedging strategies based on portfolio Greeks

        Returns:
            Hedging strategy recommendations
        """
        try:
            metrics = await self.get_portfolio_metrics()

            prompt = f"""Given this portfolio's Greeks exposure, suggest specific hedging strategies:

Net Delta: {metrics.net_delta:.2f}
Net Gamma: {metrics.net_gamma:.4f}
Net Theta: ${metrics.net_theta:.2f}/day
Net Vega: {metrics.net_vega:.2f}

Portfolio has {metrics.num_options_positions} options positions and {metrics.num_stock_positions} stock positions.

Provide specific, actionable hedging strategies:
1. What type of hedge (if any) is needed?
2. Specific instruments to use (puts, calls, spreads)
3. Sizing recommendations
4. Expected impact on portfolio Greeks
5. Cost-benefit analysis

Keep recommendations practical for retail traders using Robinhood.
"""

            llm = self._get_llm()
            hedging_analysis = await asyncio.to_thread(
                llm.query,
                prompt=prompt,
                complexity=TaskComplexity.BALANCED,
                use_trading_context=True
            )

            return hedging_analysis

        except Exception as e:
            logger.error(f"Error generating hedging strategies: {e}")
            return f"Error: {str(e)}"

    # =========================================================================
    # Agent Execution
    # =========================================================================

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute portfolio agent based on input

        Args:
            state: Agent state with input and context

        Returns:
            Updated state with portfolio analysis
        """
        try:
            input_text = state.get('input', '').lower()
            context = state.get('context', {})

            # Determine what analysis is requested
            result = {}

            # Portfolio summary
            if any(keyword in input_text for keyword in ['summary', 'overview', 'status', 'portfolio']):
                metrics = await self.get_portfolio_metrics()
                result['metrics'] = asdict(metrics)

            # Risk assessment
            if any(keyword in input_text for keyword in ['risk', 'assess', 'danger', 'exposure']):
                risk = await self.assess_portfolio_risk()
                result['risk_assessment'] = asdict(risk)

            # Sector allocation
            if any(keyword in input_text for keyword in ['sector', 'allocation', 'diversif', 'breakdown']):
                allocations = await self.get_sector_allocation()
                result['sector_allocation'] = [asdict(a) for a in allocations]

            # Greeks analysis
            if any(keyword in input_text for keyword in ['greek', 'delta', 'theta', 'gamma', 'vega']):
                metrics = await self.get_portfolio_metrics()
                result['greeks'] = {
                    'net_delta': metrics.net_delta,
                    'net_gamma': metrics.net_gamma,
                    'net_theta': metrics.net_theta,
                    'net_vega': metrics.net_vega
                }

            # AI recommendations
            if any(keyword in input_text for keyword in ['recommend', 'suggest', 'advice', 'what should']):
                recommendations = await self.generate_ai_recommendations(context)
                result['ai_recommendations'] = recommendations

            # Hedging strategies
            if any(keyword in input_text for keyword in ['hedge', 'protect', 'insurance']):
                hedging = await self.suggest_hedging_strategies()
                result['hedging_strategies'] = hedging

            # If no specific request, provide comprehensive overview
            if not result:
                metrics = await self.get_portfolio_metrics()
                risk = await self.assess_portfolio_risk()
                recommendations = await self.generate_ai_recommendations(context)

                result = {
                    'metrics': asdict(metrics),
                    'risk_assessment': asdict(risk),
                    'ai_recommendations': recommendations,
                    'timestamp': datetime.now().isoformat()
                }

            state['result'] = result
            return state

        except Exception as e:
            logger.error(f"PortfolioAgent execution error: {e}", exc_info=True)
            state['error'] = str(e)
            state['result'] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return state


# =============================================================================
# Standalone Functions (for testing)
# =============================================================================

async def test_portfolio_agent():
    """Test the portfolio agent"""
    print("Testing Portfolio Agent")
    print("=" * 80)

    agent = PortfolioAgent()

    # Test 1: Portfolio Metrics
    print("\n1. Getting Portfolio Metrics...")
    try:
        metrics = await agent.get_portfolio_metrics()
        print(f"   Total Value: ${metrics.total_value:,.2f}")
        print(f"   Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:.2f}%)")
        print(f"   Positions: {metrics.num_positions}")
        print(f"   Net Delta: {metrics.net_delta:.2f}")
        print(f"   Net Theta: ${metrics.net_theta:.2f}/day")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Risk Assessment
    print("\n2. Assessing Portfolio Risk...")
    try:
        risk = await agent.assess_portfolio_risk()
        print(f"   Risk Level: {risk.overall_risk_level.upper()}")
        print(f"   Risk Score: {risk.risk_score}/100")
        print(f"   Recommendations: {len(risk.recommendations)}")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: AI Recommendations
    print("\n3. Generating AI Recommendations...")
    try:
        recommendations = await agent.generate_ai_recommendations()
        print(f"   Generated {len(recommendations)} characters of analysis")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 80)
    print("Portfolio Agent testing complete!")


if __name__ == "__main__":
    # Run async test
    asyncio.run(test_portfolio_agent())
