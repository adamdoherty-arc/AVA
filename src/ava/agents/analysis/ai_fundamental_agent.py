"""
AI-Powered Fundamental Analysis Agent
=====================================

Uses Claude to analyze company fundamentals and provide
actionable insights for options trading decisions.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
import os
from datetime import datetime
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

class FinancialMetric(AgentOutputBase):
    """Individual financial metric with analysis"""
    name: str = ""
    value: float = 0.0
    sector_average: float = 0.0
    percentile: int = 50
    interpretation: str = ""  # positive, negative, neutral


class ValuationAssessment(AgentOutputBase):
    """Valuation analysis result"""
    method: str = ""  # DCF, P/E comparison, P/S comparison
    fair_value: float = 0.0
    current_price: float = 0.0
    upside_percent: float = 0.0
    confidence: str = ""


class FundamentalOutput(AgentOutputBase):
    """Complete fundamental analysis output"""
    symbol: str = ""
    company_name: str = ""
    sector: str = ""
    industry: str = ""

    # Financial Health
    financial_health_score: int = Field(default=50, ge=0, le=100)
    health_grade: str = "C"  # A, B, C, D, F

    # Key Metrics
    pe_ratio: Optional[float] = None
    pe_vs_sector: str = ""  # premium, discount, inline
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None

    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None

    # Growth
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    projected_growth: Optional[float] = None

    # Balance Sheet
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None

    # Cash Flow
    free_cash_flow: Optional[float] = None
    fcf_yield: Optional[float] = None
    operating_cash_flow: Optional[float] = None

    # Dividends
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    dividend_growth_5y: Optional[float] = None

    # Valuation
    valuations: List[ValuationAssessment] = Field(default_factory=list)
    intrinsic_value_estimate: Optional[float] = None
    margin_of_safety: Optional[float] = None

    # Key metrics breakdown
    key_metrics: List[FinancialMetric] = Field(default_factory=list)

    # Strengths and risks
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    catalysts: List[str] = Field(default_factory=list)

    # Options trading implications
    suitable_for_wheel: bool = False
    wheel_rationale: str = ""
    recommended_strategies: List[str] = Field(default_factory=list)
    assignment_risk_level: str = "medium"  # low, medium, high


# =============================================================================
# FUNDAMENTAL ANALYSIS AGENT
# =============================================================================

class AIFundamentalAgent(LLMAgent[FundamentalOutput]):
    """
    AI-powered fundamental analysis agent.

    Analyzes company financials and provides insights
    relevant to options trading decisions.

    Usage:
        agent = AIFundamentalAgent()
        result = await agent.execute({
            "symbol": "AAPL",
            "financial_data": {...},
            "sector_data": {...}
        })
    """

    name = "ai_fundamental"
    description = "Analyzes company fundamentals for options trading"
    output_model = FundamentalOutput
    temperature = 0.3

    system_prompt = """You are an expert financial analyst specializing in:

1. FINANCIAL STATEMENT ANALYSIS:
   - Income statement: Revenue, margins, earnings quality
   - Balance sheet: Assets, liabilities, capital structure
   - Cash flow: Operating, investing, financing activities
   - Quality of earnings vs accounting adjustments

2. VALUATION METHODOLOGIES:
   - Discounted Cash Flow (DCF) analysis
   - Relative valuation (P/E, P/B, P/S, EV/EBITDA)
   - Sum-of-parts valuation
   - Comparable company analysis

3. KEY FINANCIAL RATIOS:
   - Profitability: ROE, ROA, ROIC, margins
   - Liquidity: Current ratio, quick ratio
   - Solvency: Debt/Equity, interest coverage
   - Efficiency: Asset turnover, inventory days

4. OPTIONS TRADING CONTEXT:
   - Wheel strategy suitability (would you own this stock?)
   - Assignment risk based on fundamentals
   - Stock quality for covered call writing
   - Cash-secured put safety

Your role is to:
1. Evaluate overall financial health (score 0-100)
2. Identify key strengths and risks
3. Determine if stock is suitable for wheel strategy
4. Recommend appropriate options strategies based on fundamentals
5. Highlight any red flags that affect trading decisions

Focus on actionable insights for options traders."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build fundamental analysis prompt"""
        symbol = input_data.get('symbol', 'UNKNOWN')
        financial_data = input_data.get('financial_data', {})
        sector_data = input_data.get('sector_data', {})
        price = input_data.get('current_price', 0)

        prompt = f"""## Fundamental Analysis Request: {symbol}

### Current Market Data
- **Current Price**: ${price:.2f}
- **Market Cap**: {financial_data.get('market_cap', 'N/A')}
- **Sector**: {financial_data.get('sector', 'Unknown')}
- **Industry**: {financial_data.get('industry', 'Unknown')}

### Valuation Metrics
"""
        # Add valuation metrics
        valuation_fields = ['pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio',
                          'peg_ratio', 'ev_ebitda', 'ev_revenue']
        for field in valuation_fields:
            value = financial_data.get(field)
            if value:
                sector_avg = sector_data.get(field, 'N/A')
                prompt += f"- **{field.replace('_', ' ').title()}**: {value}"
                if sector_avg != 'N/A':
                    prompt += f" (Sector avg: {sector_avg})"
                prompt += "\n"

        prompt += "\n### Profitability Metrics\n"
        profit_fields = ['profit_margin', 'operating_margin', 'gross_margin',
                        'roe', 'roa', 'roic']
        for field in profit_fields:
            value = financial_data.get(field)
            if value:
                prompt += f"- **{field.replace('_', ' ').title()}**: {value:.1%}\n"

        prompt += "\n### Growth Metrics\n"
        growth_fields = ['revenue_growth', 'earnings_growth', 'eps_growth_5y']
        for field in growth_fields:
            value = financial_data.get(field)
            if value:
                prompt += f"- **{field.replace('_', ' ').title()}**: {value:.1%}\n"

        prompt += "\n### Balance Sheet Strength\n"
        balance_fields = ['debt_to_equity', 'current_ratio', 'quick_ratio',
                         'interest_coverage', 'total_debt', 'total_cash']
        for field in balance_fields:
            value = financial_data.get(field)
            if value:
                if field in ['debt_to_equity', 'current_ratio', 'quick_ratio', 'interest_coverage']:
                    prompt += f"- **{field.replace('_', ' ').title()}**: {value:.2f}\n"
                else:
                    prompt += f"- **{field.replace('_', ' ').title()}**: ${value:,.0f}\n"

        prompt += "\n### Cash Flow\n"
        cf_fields = ['free_cash_flow', 'operating_cash_flow', 'fcf_yield']
        for field in cf_fields:
            value = financial_data.get(field)
            if value:
                if field == 'fcf_yield':
                    prompt += f"- **{field.replace('_', ' ').title()}**: {value:.2%}\n"
                else:
                    prompt += f"- **{field.replace('_', ' ').title()}**: ${value:,.0f}\n"

        prompt += "\n### Dividend Information\n"
        div_fields = ['dividend_yield', 'payout_ratio', 'dividend_growth_5y']
        for field in div_fields:
            value = financial_data.get(field)
            if value:
                prompt += f"- **{field.replace('_', ' ').title()}**: {value:.2%}\n"

        # Recent earnings info
        earnings = financial_data.get('recent_earnings', [])
        if earnings:
            prompt += "\n### Recent Earnings History\n"
            for e in earnings[:4]:
                prompt += f"- Q{e.get('quarter', '?')} {e.get('year', '?')}: "
                prompt += f"EPS ${e.get('eps_actual', 0):.2f} "
                prompt += f"vs est ${e.get('eps_estimate', 0):.2f} "
                prompt += f"({'beat' if e.get('eps_actual', 0) > e.get('eps_estimate', 0) else 'miss'})\n"

        prompt += """

### Analysis Tasks

1. **Financial Health Score (0-100)**:
   - 80-100: Excellent (A grade)
   - 60-79: Good (B grade)
   - 40-59: Average (C grade)
   - 20-39: Weak (D grade)
   - 0-19: Poor (F grade)

2. **Key Metrics Analysis**: Evaluate each metric vs sector/industry

3. **Valuation Assessment**:
   - Is the stock overvalued, undervalued, or fairly valued?
   - Estimate intrinsic value range if possible

4. **Strengths**: Top 3-5 positive factors

5. **Risks**: Top 3-5 concerns or red flags

6. **Catalysts**: Upcoming events that could move the stock

7. **Options Trading Suitability**:
   - Is this a stock you'd want to own long-term? (wheel strategy consideration)
   - Assignment risk level (low/medium/high)
   - Best options strategies for this fundamental profile

Focus on what matters most for options trading decisions.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> FundamentalOutput:
        """Parse fundamental analysis response"""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                data['symbol'] = input_data.get('symbol', '')
                return FundamentalOutput(**data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> FundamentalOutput:
        """Parse unstructured response"""
        import re

        response_lower = response.lower()
        financial_data = input_data.get('financial_data', {})

        # Determine health score
        score = 50
        if 'excellent' in response_lower or 'strong' in response_lower:
            score = 80
        elif 'good' in response_lower or 'solid' in response_lower:
            score = 70
        elif 'weak' in response_lower or 'poor' in response_lower:
            score = 30
        elif 'average' in response_lower or 'moderate' in response_lower:
            score = 50

        # Check for score mentions
        score_match = re.search(r'health\s*score[:\s]*(\d+)', response_lower)
        if score_match:
            score = min(100, max(0, int(score_match.group(1))))

        # Determine grade
        if score >= 80:
            grade = 'A'
        elif score >= 60:
            grade = 'B'
        elif score >= 40:
            grade = 'C'
        elif score >= 20:
            grade = 'D'
        else:
            grade = 'F'

        # Wheel suitability
        wheel_suitable = 'wheel' in response_lower and (
            'suitable' in response_lower or
            'appropriate' in response_lower or
            'good candidate' in response_lower
        )

        # Assignment risk
        if 'high risk' in response_lower or 'high assignment' in response_lower:
            assignment_risk = 'high'
        elif 'low risk' in response_lower or 'low assignment' in response_lower:
            assignment_risk = 'low'
        else:
            assignment_risk = 'medium'

        return FundamentalOutput(
            agent_name=self.name,
            symbol=input_data.get('symbol', ''),
            company_name=financial_data.get('company_name', ''),
            sector=financial_data.get('sector', ''),
            industry=financial_data.get('industry', ''),
            financial_health_score=score,
            health_grade=grade,
            pe_ratio=financial_data.get('pe_ratio'),
            pb_ratio=financial_data.get('pb_ratio'),
            ps_ratio=financial_data.get('ps_ratio'),
            profit_margin=financial_data.get('profit_margin'),
            roe=financial_data.get('roe'),
            debt_to_equity=financial_data.get('debt_to_equity'),
            current_ratio=financial_data.get('current_ratio'),
            free_cash_flow=financial_data.get('free_cash_flow'),
            dividend_yield=financial_data.get('dividend_yield'),
            suitable_for_wheel=wheel_suitable,
            assignment_risk_level=assignment_risk,
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK FUNDAMENTAL CHECKER
# =============================================================================

class QuickFundamentalChecker:
    """Rule-based quick fundamental check"""

    @staticmethod
    def check(
        pe_ratio: Optional[float] = None,
        debt_to_equity: Optional[float] = None,
        current_ratio: Optional[float] = None,
        profit_margin: Optional[float] = None,
        roe: Optional[float] = None,
        free_cash_flow: Optional[float] = None
    ) -> Dict:
        """Quick fundamental assessment"""
        score = 50
        flags = []
        strengths = []

        # P/E analysis
        if pe_ratio is not None:
            if pe_ratio < 0:
                flags.append("Negative earnings")
                score -= 20
            elif pe_ratio > 50:
                flags.append("High P/E - expensive or high growth")
                score -= 5
            elif pe_ratio < 15:
                strengths.append("Attractive P/E ratio")
                score += 10

        # Debt analysis
        if debt_to_equity is not None:
            if debt_to_equity > 2:
                flags.append("High debt levels")
                score -= 15
            elif debt_to_equity < 0.5:
                strengths.append("Low debt")
                score += 10

        # Liquidity
        if current_ratio is not None:
            if current_ratio < 1:
                flags.append("Liquidity concerns")
                score -= 10
            elif current_ratio > 2:
                strengths.append("Strong liquidity")
                score += 5

        # Profitability
        if profit_margin is not None:
            if profit_margin < 0:
                flags.append("Unprofitable")
                score -= 20
            elif profit_margin > 0.2:
                strengths.append("High profit margins")
                score += 15
            elif profit_margin > 0.1:
                strengths.append("Healthy profit margins")
                score += 5

        # ROE
        if roe is not None:
            if roe < 0:
                flags.append("Negative ROE")
                score -= 10
            elif roe > 0.2:
                strengths.append("Excellent ROE")
                score += 15
            elif roe > 0.1:
                strengths.append("Good ROE")
                score += 5

        # Cash flow
        if free_cash_flow is not None:
            if free_cash_flow < 0:
                flags.append("Negative free cash flow")
                score -= 10
            else:
                strengths.append("Positive free cash flow")
                score += 10

        score = min(100, max(0, score))

        # Grade
        if score >= 80:
            grade = 'A'
        elif score >= 60:
            grade = 'B'
        elif score >= 40:
            grade = 'C'
        elif score >= 20:
            grade = 'D'
        else:
            grade = 'F'

        # Wheel suitability
        wheel_suitable = (
            score >= 60 and
            'Negative earnings' not in flags and
            'Unprofitable' not in flags and
            'High debt levels' not in flags
        )

        return {
            'financial_health_score': score,
            'health_grade': grade,
            'strengths': strengths,
            'risks': flags,
            'suitable_for_wheel': wheel_suitable,
            'wheel_rationale': 'Fundamentally sound' if wheel_suitable else 'Fundamental concerns'
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Fundamental Agent ===\n")

    async def test_agent():
        # Test quick checker
        print("1. Testing Quick Fundamental Checker...")
        checker = QuickFundamentalChecker()

        test_cases = [
            {"pe_ratio": 15, "debt_to_equity": 0.3, "profit_margin": 0.25, "roe": 0.22},  # Strong
            {"pe_ratio": -5, "debt_to_equity": 3.0, "profit_margin": -0.1},  # Weak
            {"pe_ratio": 25, "debt_to_equity": 1.0, "profit_margin": 0.12, "current_ratio": 1.5},  # Average
        ]

        for case in test_cases:
            result = checker.check(**case)
            print(f"\n   P/E={case.get('pe_ratio')}, D/E={case.get('debt_to_equity')}:")
            print(f"      Score: {result['financial_health_score']}")
            print(f"      Grade: {result['health_grade']}")
            print(f"      Wheel: {result['suitable_for_wheel']}")

        # Test agent structure
        print("\n2. Testing Agent Structure...")
        agent = AIFundamentalAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Fundamental Agent ready!")

    asyncio.run(test_agent())
