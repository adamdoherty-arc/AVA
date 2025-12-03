"""
AI-Powered Risk Management Agent
================================

Uses Claude to analyze portfolio risk and provide
intelligent risk management recommendations.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import Field

from src.ava.agents.base.llm_agent import (
    LLMAgent,
    AgentOutputBase,
    AgentConfidence,
    AgentExecutionContext
)
from src.ava.core.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class RiskAlert(AgentOutputBase):
    """Individual risk alert"""
    severity: str = "medium"  # low, medium, high, critical
    category: str = ""  # delta, gamma, theta, vega, concentration, var
    title: str = ""
    description: str = ""
    current_value: float = 0
    limit_value: float = 0
    recommended_action: str = ""


class HedgeRecommendation(AgentOutputBase):
    """Hedge recommendation"""
    hedge_type: str = ""  # protective_put, collar, hedge_ratio, reduce
    urgency: str = "medium"  # low, medium, high
    description: str = ""
    estimated_cost: float = 0
    risk_reduction_pct: float = 0


class PositionAdjustment(AgentOutputBase):
    """Position adjustment recommendation"""
    symbol: str = ""
    current_size: int = 0
    recommended_size: int = 0
    action: str = ""  # reduce, close, roll, hold
    reason: str = ""
    priority: int = 1


class RiskAnalysisOutput(AgentOutputBase):
    """Complete risk analysis output"""
    portfolio_value: float = 0
    risk_score: int = Field(default=50, ge=0, le=100)  # 0=safest, 100=riskiest
    risk_level: str = "moderate"  # low, moderate, elevated, high, critical

    # Greeks summary
    total_delta: float = 0
    total_gamma: float = 0
    total_theta: float = 0
    total_vega: float = 0
    beta_weighted_delta: float = 0

    # VaR
    var_95: float = 0
    var_95_pct: float = 0
    max_daily_loss: float = 0

    # Analysis
    risk_summary: str = ""
    biggest_risk_factor: str = ""

    # Recommendations
    alerts: List[RiskAlert] = Field(default_factory=list)
    hedge_recommendations: List[HedgeRecommendation] = Field(default_factory=list)
    position_adjustments: List[PositionAdjustment] = Field(default_factory=list)

    # Overall recommendation
    immediate_action_required: bool = False
    action_summary: str = ""


# =============================================================================
# RISK MANAGEMENT AGENT
# =============================================================================

class AIRiskManagementAgent(LLMAgent[RiskAnalysisOutput]):
    """
    AI-powered risk management agent.

    Analyzes portfolio Greeks, VaR, and positions to provide
    intelligent risk management recommendations.

    Usage:
        agent = AIRiskManagementAgent()
        result = await agent.execute({
            "portfolio_value": 100000,
            "positions": [...],
            "greeks": {...},
            "var_analysis": {...}
        })
    """

    name = "ai_risk_manager"
    description = "Analyzes portfolio risk and provides management recommendations"
    output_model = RiskAnalysisOutput
    temperature = 0.3

    system_prompt = """You are an expert portfolio risk manager specializing in options.

Your expertise includes:
1. GREEKS ANALYSIS:
   - Delta: Directional exposure (each delta point = 100 shares equivalent)
   - Gamma: Rate of delta change (acceleration risk)
   - Theta: Time decay (positive = collecting, negative = paying)
   - Vega: Volatility exposure ($ change per 1% IV move)

2. VALUE AT RISK (VaR):
   - 95% VaR: Expected max daily loss 95% of the time
   - Interpreting VaR as % of portfolio
   - Stress testing scenarios

3. POSITION CONCENTRATION:
   - Single position risk
   - Sector exposure
   - Correlated positions

4. RISK MITIGATION:
   - Position sizing adjustments
   - Delta hedging strategies
   - Protective puts and collars
   - Rolling and adjusting positions

Your role is to:
1. Assess the overall risk level (score 0-100)
2. Identify the biggest risk factors
3. Recommend specific hedges or adjustments
4. Flag any positions needing immediate attention
5. Provide clear, actionable guidance

Always prioritize capital preservation while maintaining profit potential."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build risk analysis prompt"""
        config = get_config()
        limits = config.risk

        portfolio_value = input_data.get('portfolio_value', 100000)
        positions = input_data.get('positions', [])
        greeks = input_data.get('greeks', {})
        var_analysis = input_data.get('var_analysis', {})
        stress_tests = input_data.get('stress_tests', {})

        prompt = f"""## Portfolio Risk Analysis Request

### Portfolio Overview
- **Total Value**: ${portfolio_value:,.2f}
- **Number of Positions**: {len(positions)}

### Current Greeks
- **Total Delta**: {greeks.get('total_delta', 0):.1f} (limit: {limits.max_portfolio_delta})
- **Total Gamma**: {greeks.get('total_gamma', 0):.3f} (limit: {limits.max_portfolio_gamma})
- **Total Theta**: ${greeks.get('total_theta', 0):.2f}/day
- **Total Vega**: ${greeks.get('total_vega', 0):.2f}
- **Beta-Weighted Delta (SPY)**: {greeks.get('beta_weighted_delta', 0):.1f}

### Value at Risk
- **95% VaR**: ${var_analysis.get('var_95', 0):,.2f} ({var_analysis.get('var_95_pct', 0):.1%} of portfolio)
- **99% VaR**: ${var_analysis.get('var_99', 0):,.2f}
- **Max Daily Loss Limit**: {limits.max_daily_loss_pct}%

### Risk Limits
- Max Portfolio Delta: {limits.max_portfolio_delta}
- Max Single Position: {limits.max_position_size_pct}%
- Max Single Underlying: {limits.max_single_underlying_pct}%
- Max Daily VaR: {limits.max_var_95_pct}%
"""

        # Add stress test results if available
        if stress_tests:
            prompt += "\n### Stress Test Results\n"
            for scenario, impact in stress_tests.items():
                prompt += f"- **{scenario}**: ${impact:,.2f}\n"

        # Add positions detail
        if positions:
            prompt += "\n### Current Positions\n"
            prompt += "| Symbol | Type | Qty | Delta | Theta | DTE | P&L % |\n"
            prompt += "|--------|------|-----|-------|-------|-----|-------|\n"

            for pos in positions[:20]:  # Limit to 20 positions
                prompt += f"| {pos.get('symbol', 'N/A')[:15]} | "
                prompt += f"{pos.get('option_type', 'stock')[:4]} | "
                prompt += f"{pos.get('quantity', 0)} | "
                prompt += f"{pos.get('delta', 0):.2f} | "
                prompt += f"${pos.get('theta', 0):.2f} | "
                prompt += f"{pos.get('days_to_expiration', 'N/A')} | "
                prompt += f"{pos.get('unrealized_pnl_pct', 0):.1%} |\n"

        prompt += """
### Your Analysis Task

1. **Risk Score (0-100)**: Calculate overall portfolio risk
   - 0-20: Low risk, well-hedged
   - 21-40: Moderate risk, acceptable
   - 41-60: Elevated risk, monitor closely
   - 61-80: High risk, action recommended
   - 81-100: Critical risk, immediate action required

2. **Risk Alerts**: Identify specific limit violations or concerns:
   - Delta exceeding limits
   - Concentration in single position
   - High VaR percentage
   - Positions near expiration
   - Large losing positions

3. **Hedge Recommendations**: If needed, suggest:
   - Protective puts (with estimated cost)
   - Delta reduction via spreads
   - Position reduction order

4. **Position Adjustments**: Rank positions by urgency:
   - Which to close immediately?
   - Which to reduce?
   - Which to roll?

5. **Action Summary**: One clear sentence on what to do NOW

Be specific and actionable. Include dollar amounts where relevant.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> RiskAnalysisOutput:
        """Parse risk analysis response"""
        import json
        import re

        try:
            # Try JSON extraction
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                data['portfolio_value'] = input_data.get('portfolio_value', 0)
                return RiskAnalysisOutput(**data)

        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        # Fallback parsing
        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> RiskAnalysisOutput:
        """Parse unstructured response"""
        import re

        # Extract risk score
        risk_score = 50
        score_match = re.search(r'risk\s*score[:\s]*(\d+)', response.lower())
        if score_match:
            risk_score = min(100, max(0, int(score_match.group(1))))

        # Determine risk level
        if risk_score <= 20:
            risk_level = "low"
        elif risk_score <= 40:
            risk_level = "moderate"
        elif risk_score <= 60:
            risk_level = "elevated"
        elif risk_score <= 80:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Check for immediate action keywords
        immediate_action = any(
            keyword in response.lower()
            for keyword in ['immediate', 'urgent', 'critical', 'now', 'asap']
        )

        greeks = input_data.get('greeks', {})
        var_analysis = input_data.get('var_analysis', {})

        return RiskAnalysisOutput(
            agent_name=self.name,
            portfolio_value=input_data.get('portfolio_value', 0),
            risk_score=risk_score,
            risk_level=risk_level,
            total_delta=greeks.get('total_delta', 0),
            total_gamma=greeks.get('total_gamma', 0),
            total_theta=greeks.get('total_theta', 0),
            total_vega=greeks.get('total_vega', 0),
            var_95=var_analysis.get('var_95', 0),
            var_95_pct=var_analysis.get('var_95_pct', 0),
            risk_summary=response[:500],
            immediate_action_required=immediate_action,
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK RISK CHECKER
# =============================================================================

class QuickRiskChecker:
    """
    Rule-based quick risk checker for when LLM is unavailable.
    """

    @staticmethod
    def check(
        portfolio_value: float,
        total_delta: float,
        total_theta: float,
        total_vega: float,
        var_95_pct: float,
        positions: List[Dict]
    ) -> Dict:
        """Quick risk assessment"""
        config = get_config()
        limits = config.risk

        alerts = []
        risk_score = 20  # Base score

        # Check delta limit
        if abs(total_delta) > limits.max_portfolio_delta:
            alerts.append({
                "severity": "high",
                "category": "delta",
                "title": "Delta Limit Exceeded",
                "description": f"Portfolio delta {total_delta:.0f} exceeds limit {limits.max_portfolio_delta}"
            })
            risk_score += 30

        # Check VaR
        if var_95_pct > limits.max_var_95_pct:
            alerts.append({
                "severity": "high",
                "category": "var",
                "title": "VaR Limit Exceeded",
                "description": f"VaR {var_95_pct:.1%} exceeds limit {limits.max_var_95_pct}%"
            })
            risk_score += 25

        # Check concentration
        for pos in positions:
            pos_value = abs(pos.get('market_value', 0))
            pos_pct = pos_value / portfolio_value * 100 if portfolio_value > 0 else 0

            if pos_pct > limits.max_position_size_pct:
                alerts.append({
                    "severity": "medium",
                    "category": "concentration",
                    "title": f"Position Concentration: {pos.get('symbol', 'Unknown')}",
                    "description": f"Position is {pos_pct:.1f}% of portfolio (limit: {limits.max_position_size_pct}%)"
                })
                risk_score += 10

        # Check for expiring positions
        for pos in positions:
            dte = pos.get('days_to_expiration', 999)
            if dte <= 3 and dte >= 0:
                alerts.append({
                    "severity": "medium",
                    "category": "expiration",
                    "title": f"Expiring Soon: {pos.get('symbol', 'Unknown')}",
                    "description": f"Position expires in {dte} days"
                })
                risk_score += 5

        # Determine risk level
        risk_score = min(100, risk_score)

        if risk_score <= 20:
            risk_level = "low"
        elif risk_score <= 40:
            risk_level = "moderate"
        elif risk_score <= 60:
            risk_level = "elevated"
        elif risk_score <= 80:
            risk_level = "high"
        else:
            risk_level = "critical"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "alerts": alerts,
            "immediate_action_required": risk_score >= 80
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Risk Management Agent ===\n")

    async def test_agent():
        # Test quick checker first
        print("1. Testing Quick Risk Checker...")
        checker = QuickRiskChecker()

        result = checker.check(
            portfolio_value=100000,
            total_delta=600,  # Over limit
            total_theta=-15,
            total_vega=500,
            var_95_pct=0.04,  # Over 3% limit
            positions=[
                {"symbol": "AAPL", "market_value": 25000, "days_to_expiration": 2},
                {"symbol": "MSFT", "market_value": 15000, "days_to_expiration": 30}
            ]
        )

        print(f"   Risk Score: {result['risk_score']}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Alerts: {len(result['alerts'])}")
        for alert in result['alerts']:
            print(f"      - [{alert['severity']}] {alert['title']}")

        # Test agent structure
        print("\n2. Testing Agent Structure...")
        agent = AIRiskManagementAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Risk Management Agent ready!")

    asyncio.run(test_agent())
