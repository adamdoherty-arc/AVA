"""
AI Trading Orchestrator
=======================

Unified orchestrator that coordinates all AI agents for
comprehensive trading analysis and decision making.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from src.ava.agents.base.llm_agent import (
    LLMAgent,
    AgentOutputBase,
    AgentConfidence,
    AgentExecutionContext,
    MultiAgentExecutor
)
# Core AI agents (always used)
from src.ava.agents.trading.strategy_recommendation_agent import (
    StrategyRecommendationAgent,
    StrategyRecommendationOutput,
    QuickStrategySelector
)
from src.ava.agents.trading.ai_risk_agent import (
    AIRiskManagementAgent,
    RiskAnalysisOutput,
    QuickRiskChecker
)
from src.ava.agents.analysis.ai_sentiment_agent import (
    AISentimentAgent,
    MarketSentimentOutput,
    QuickSentimentChecker
)
# Extended AI agents (optional, for deep analysis)
from src.ava.agents.analysis.ai_fundamental_agent import (
    AIFundamentalAgent,
    FundamentalOutput,
    QuickFundamentalChecker
)
from src.ava.agents.trading.ai_options_analysis_agent import (
    AIOptionsAnalysisAgent,
    OptionsAnalysisOutput,
    QuickOptionsScorer
)
from src.ava.agents.analysis.ai_options_flow_agent import (
    AIOptionsFlowAgent,
    OptionsFlowOutput,
    QuickFlowAnalyzer
)
from src.ava.agents.trading.ai_earnings_agent import (
    AIEarningsAgent,
    EarningsOutput,
    QuickEarningsChecker
)
from src.ava.core.config import get_config
from src.ava.core.cache import TieredCache, get_general_cache
from src.ava.core.errors import get_error_handler

logger = logging.getLogger(__name__)


# =============================================================================
# ORCHESTRATOR OUTPUT
# =============================================================================

class TradingDecision(BaseModel):
    """Final trading decision from orchestrator"""
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Decision
    action: str = "hold"  # strong_buy, buy, hold, avoid, strong_avoid
    conviction: str = "medium"  # very_high, high, medium, low, very_low
    confidence_score: int = Field(default=50, ge=0, le=100)

    # Strategy
    recommended_strategy: str = ""
    strategy_parameters: Dict[str, Any] = Field(default_factory=dict)

    # Position sizing
    recommended_contracts: int = 1
    max_contracts: int = 1
    allocation_pct: float = 2.0

    # Analysis summaries
    strategy_analysis: str = ""
    risk_analysis: str = ""
    sentiment_analysis: str = ""

    # Scores
    strategy_score: int = 50
    risk_score: int = 50
    sentiment_score: int = 50
    composite_score: int = 50

    # Warnings and reasons
    key_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)

    # Agent results (for debugging)
    agent_results: Dict[str, Any] = Field(default_factory=dict)

    # Extended analysis (when deep_analysis=True)
    fundamental_score: Optional[int] = None
    options_analysis_score: Optional[int] = None
    flow_signal: Optional[str] = None
    earnings_impact: Optional[str] = None

    # Execution metadata
    analysis_time_ms: float = 0
    used_cache: bool = False
    fallback_used: bool = False
    deep_analysis_used: bool = False


# =============================================================================
# AI TRADING ORCHESTRATOR
# =============================================================================

class AITradingOrchestrator:
    """
    Orchestrates multiple AI agents for comprehensive trading analysis.

    Coordinates:
    - Strategy Recommendation Agent
    - Risk Management Agent
    - Sentiment Analysis Agent

    Provides unified trading decisions with confidence scoring.

    Usage:
        orchestrator = AITradingOrchestrator()
        await orchestrator.initialize()

        decision = await orchestrator.analyze(
            symbol="AAPL",
            underlying_price=230.50,
            iv_rank=45,
            portfolio_value=100000
        )

        print(f"Action: {decision.action}")
        print(f"Strategy: {decision.recommended_strategy}")
    """

    def __init__(
        self,
        use_parallel: bool = True,
        cache_enabled: bool = True,
        fallback_enabled: bool = True,
        deep_analysis_enabled: bool = False
    ):
        self.use_parallel = use_parallel
        self.cache_enabled = cache_enabled
        self.fallback_enabled = fallback_enabled
        self.deep_analysis_enabled = deep_analysis_enabled

        # Core AI agents (always used)
        self.strategy_agent = StrategyRecommendationAgent(
            cache_enabled=cache_enabled
        )
        self.risk_agent = AIRiskManagementAgent(
            cache_enabled=cache_enabled
        )
        self.sentiment_agent = AISentimentAgent(
            cache_enabled=cache_enabled
        )

        # Extended AI agents (for deep analysis)
        self.fundamental_agent = AIFundamentalAgent(
            cache_enabled=cache_enabled
        )
        self.options_analysis_agent = AIOptionsAnalysisAgent(
            cache_enabled=cache_enabled
        )
        self.options_flow_agent = AIOptionsFlowAgent(
            cache_enabled=cache_enabled
        )
        self.earnings_agent = AIEarningsAgent(
            cache_enabled=cache_enabled
        )

        # Quick checkers for fallback
        self.quick_strategy = QuickStrategySelector()
        self.quick_risk = QuickRiskChecker()
        self.quick_sentiment = QuickSentimentChecker()
        self.quick_fundamental = QuickFundamentalChecker()
        self.quick_options = QuickOptionsScorer()
        self.quick_flow = QuickFlowAnalyzer()
        self.quick_earnings = QuickEarningsChecker()

        # Multi-agent executor for core agents
        self.executor = MultiAgentExecutor(
            agents=[self.strategy_agent, self.risk_agent, self.sentiment_agent],
            max_concurrency=3
        )

        # Extended executor for deep analysis
        self.extended_executor = MultiAgentExecutor(
            agents=[
                self.fundamental_agent,
                self.options_analysis_agent,
                self.options_flow_agent,
                self.earnings_agent
            ],
            max_concurrency=4
        )

        # Cache
        self._cache = get_general_cache()

        # Error handler
        self._error_handler = get_error_handler()

        # Stats
        self._stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "fallback_uses": 0,
            "cache_hits": 0,
            "total_time_ms": 0,
            "deep_analyses": 0
        }

    async def initialize(self) -> None:
        """Initialize orchestrator and cache"""
        if self._cache:
            await self._cache.initialize()
        logger.info("AI Trading Orchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown orchestrator"""
        if self._cache:
            await self._cache.shutdown()

    async def analyze(
        self,
        symbol: str,
        underlying_price: float,
        iv_rank: float = 50,
        iv_percentile: float = None,
        hv_20: float = 0.25,
        vix: float = 15,
        trend: str = "neutral",
        days_to_earnings: Optional[int] = None,
        sector: str = "Unknown",
        portfolio_value: float = 100000,
        positions: List[Dict] = None,
        greeks: Dict = None,
        var_analysis: Dict = None,
        put_call_ratio: float = 1.0,
        recent_news: List = None,
        options_flow: List = None,
        risk_tolerance: str = "moderate",
        account_size: float = None,
        deep_analysis: bool = False,
        option_chain: Dict = None,
        financial_data: Dict = None,
        earnings_calendar: List = None,
        flow_data: List = None
    ) -> TradingDecision:
        """
        Run comprehensive trading analysis.

        Args:
            symbol: Stock symbol
            underlying_price: Current stock price
            iv_rank: IV Rank (0-100)
            deep_analysis: Enable extended agents (fundamental, options, flow, earnings)
            option_chain: Option chain data for options analysis
            financial_data: Financial data for fundamental analysis
            earnings_calendar: Earnings events for earnings analysis
            flow_data: Options flow data for flow analysis
            ... (other market data)

        Returns:
            TradingDecision with action, strategy, and analysis
        """
        start_time = datetime.now()
        self._stats["total_analyses"] += 1

        # Use instance setting if not explicitly specified
        run_deep = deep_analysis or self.deep_analysis_enabled

        account_size = account_size or portfolio_value
        positions = positions or []
        greeks = greeks or {}
        var_analysis = var_analysis or {}
        recent_news = recent_news or []
        options_flow = options_flow or []
        flow_data = flow_data or options_flow
        option_chain = option_chain or {}
        financial_data = financial_data or {}
        earnings_calendar = earnings_calendar or []
        iv_percentile = iv_percentile or iv_rank

        # Check cache
        cache_key = f"analysis:{symbol}:{underlying_price:.0f}:{iv_rank:.0f}"
        if self.cache_enabled:
            cached = await self._cache.get(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                cached.used_cache = True
                return cached

        # Build input data for agents
        strategy_input = {
            "symbol": symbol,
            "underlying_price": underlying_price,
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "hv_20": hv_20,
            "vix": vix,
            "trend": trend,
            "days_to_earnings": days_to_earnings,
            "sector": sector,
            "account_size": account_size,
            "risk_tolerance": risk_tolerance
        }

        risk_input = {
            "portfolio_value": portfolio_value,
            "positions": positions,
            "greeks": greeks,
            "var_analysis": var_analysis
        }

        sentiment_input = {
            "symbol": symbol,
            "vix": vix,
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "hv_20": hv_20,
            "put_call_ratio": put_call_ratio,
            "recent_news": recent_news,
            "options_flow": options_flow
        }

        # Extended agent inputs (for deep analysis)
        fundamental_input = {
            "symbol": symbol,
            "current_price": underlying_price,
            "financial_data": financial_data,
            "sector_data": {}
        }

        options_input = {
            "symbol": symbol,
            "underlying_price": underlying_price,
            "strategy": "csp",
            "option_chain": option_chain,
            "iv_data": {
                "iv_rank": iv_rank,
                "iv_percentile": iv_percentile,
                "current_iv": hv_20 * 1.2,
                "hv_20": hv_20
            },
            "target_delta": 0.30,
            "target_dte": 30
        }

        flow_input = {
            "symbol": symbol,
            "underlying_price": underlying_price,
            "flow_data": flow_data
        }

        earnings_input = {
            "watchlist": [symbol],
            "positions": positions,
            "earnings_calendar": earnings_calendar,
            "lookforward_days": 14
        }

        # Run analysis
        try:
            if self.use_parallel:
                results = await self._run_parallel_analysis(
                    strategy_input, risk_input, sentiment_input
                )
            else:
                results = await self._run_sequential_analysis(
                    strategy_input, risk_input, sentiment_input
                )

            # Run extended analysis if requested
            extended_results = {}
            if run_deep:
                self._stats["deep_analyses"] += 1
                extended_results = await self._run_extended_analysis(
                    fundamental_input, options_input, flow_input, earnings_input
                )

            decision = self._synthesize_decision(
                symbol, underlying_price, iv_rank, results, strategy_input,
                extended_results=extended_results
            )
            decision.deep_analysis_used = run_deep
            fallback_used = False

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")

            if self.fallback_enabled:
                decision = self._run_fallback_analysis(
                    symbol, underlying_price, iv_rank, trend,
                    days_to_earnings, portfolio_value, positions,
                    greeks, var_analysis, vix, put_call_ratio, risk_tolerance
                )
                fallback_used = True
                self._stats["fallback_uses"] += 1
            else:
                raise

        # Add metadata
        decision.analysis_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        decision.fallback_used = fallback_used

        # Cache result
        if self.cache_enabled:
            await self._cache.set(cache_key, decision, l1_ttl=60, l2_ttl=300)

        self._stats["successful_analyses"] += 1
        self._stats["total_time_ms"] += decision.analysis_time_ms

        return decision

    async def _run_parallel_analysis(
        self,
        strategy_input: Dict,
        risk_input: Dict,
        sentiment_input: Dict
    ) -> Dict[str, Any]:
        """Run all agents in parallel"""
        tasks = [
            self.strategy_agent.execute(strategy_input),
            self.risk_agent.execute(risk_input),
            self.sentiment_agent.execute(sentiment_input)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "strategy": results[0] if not isinstance(results[0], Exception) else None,
            "risk": results[1] if not isinstance(results[1], Exception) else None,
            "sentiment": results[2] if not isinstance(results[2], Exception) else None,
            "errors": [r for r in results if isinstance(r, Exception)]
        }

    async def _run_sequential_analysis(
        self,
        strategy_input: Dict,
        risk_input: Dict,
        sentiment_input: Dict
    ) -> Dict[str, Any]:
        """Run agents sequentially"""
        results = {"strategy": None, "risk": None, "sentiment": None, "errors": []}

        try:
            results["strategy"] = await self.strategy_agent.execute(strategy_input)
        except Exception as e:
            results["errors"].append(e)

        try:
            results["risk"] = await self.risk_agent.execute(risk_input)
        except Exception as e:
            results["errors"].append(e)

        try:
            results["sentiment"] = await self.sentiment_agent.execute(sentiment_input)
        except Exception as e:
            results["errors"].append(e)

        return results

    async def _run_extended_analysis(
        self,
        fundamental_input: Dict,
        options_input: Dict,
        flow_input: Dict,
        earnings_input: Dict
    ) -> Dict[str, Any]:
        """Run extended AI agents for deep analysis"""
        results = {
            "fundamental": None,
            "options": None,
            "flow": None,
            "earnings": None,
            "errors": []
        }

        if self.use_parallel:
            tasks = [
                self.fundamental_agent.execute(fundamental_input),
                self.options_analysis_agent.execute(options_input),
                self.options_flow_agent.execute(flow_input),
                self.earnings_agent.execute(earnings_input)
            ]
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            results["fundamental"] = agent_results[0] if not isinstance(agent_results[0], Exception) else None
            results["options"] = agent_results[1] if not isinstance(agent_results[1], Exception) else None
            results["flow"] = agent_results[2] if not isinstance(agent_results[2], Exception) else None
            results["earnings"] = agent_results[3] if not isinstance(agent_results[3], Exception) else None
            results["errors"] = [r for r in agent_results if isinstance(r, Exception)]
        else:
            # Sequential execution
            try:
                results["fundamental"] = await self.fundamental_agent.execute(fundamental_input)
            except Exception as e:
                results["errors"].append(e)

            try:
                results["options"] = await self.options_analysis_agent.execute(options_input)
            except Exception as e:
                results["errors"].append(e)

            try:
                results["flow"] = await self.options_flow_agent.execute(flow_input)
            except Exception as e:
                results["errors"].append(e)

            try:
                results["earnings"] = await self.earnings_agent.execute(earnings_input)
            except Exception as e:
                results["errors"].append(e)

        return results

    def _synthesize_decision(
        self,
        symbol: str,
        underlying_price: float,
        iv_rank: float,
        results: Dict[str, Any],
        strategy_input: Dict,
        extended_results: Dict[str, Any] = None
    ) -> TradingDecision:
        """Synthesize final decision from agent results"""
        extended_results = extended_results or {}

        strategy_result: Optional[StrategyRecommendationOutput] = results.get("strategy")
        risk_result: Optional[RiskAnalysisOutput] = results.get("risk")
        sentiment_result: Optional[MarketSentimentOutput] = results.get("sentiment")

        # Extended results
        fundamental_result: Optional[FundamentalOutput] = extended_results.get("fundamental")
        options_result: Optional[OptionsAnalysisOutput] = extended_results.get("options")
        flow_result: Optional[OptionsFlowOutput] = extended_results.get("flow")
        earnings_result: Optional[EarningsOutput] = extended_results.get("earnings")

        # Extract scores
        strategy_score = 50
        risk_score = 50
        sentiment_score = 50

        key_reasons = []
        warnings = []
        risks = []

        # Process strategy result
        recommended_strategy = ""
        strategy_params = {}
        strategy_analysis = ""

        if strategy_result and strategy_result.recommendations:
            top_rec = strategy_result.recommendations[0]
            strategy_score = top_rec.fit_score
            recommended_strategy = top_rec.strategy_name
            strategy_params = {
                "delta": top_rec.suggested_delta,
                "dte_min": top_rec.suggested_dte_min,
                "dte_max": top_rec.suggested_dte_max
            }
            key_reasons.append(f"Strategy fit: {top_rec.why_recommended}")
            strategy_analysis = strategy_result.reasoning or ""

        # Process risk result
        risk_analysis = ""
        if risk_result:
            risk_score = 100 - risk_result.risk_score  # Invert (lower risk = higher score)
            risk_analysis = risk_result.risk_summary or risk_result.reasoning or ""

            if risk_result.immediate_action_required:
                warnings.append("IMMEDIATE ACTION REQUIRED")

            for alert in (risk_result.alerts or [])[:3]:
                risks.append(f"{alert.category}: {alert.title}")

        # Process sentiment result
        sentiment_analysis = ""
        if sentiment_result:
            sentiment_score = sentiment_result.sentiment_score
            sentiment_analysis = sentiment_result.reasoning or ""

            if sentiment_result.overall_sentiment in ["bullish", "very_bullish"]:
                key_reasons.append(f"Sentiment: {sentiment_result.overall_sentiment}")
            elif sentiment_result.overall_sentiment in ["bearish", "very_bearish"]:
                warnings.append(f"Sentiment warning: {sentiment_result.overall_sentiment}")

            for caution in (sentiment_result.caution_factors or [])[:2]:
                warnings.append(caution)

        # Process extended results if available
        fundamental_score = None
        options_analysis_score = None
        flow_signal = None
        earnings_impact = None

        if fundamental_result:
            fundamental_score = fundamental_result.financial_health_score
            if not fundamental_result.suitable_for_wheel:
                warnings.append(f"Fundamental: Not ideal for wheel strategy")
            if fundamental_result.health_grade in ['D', 'F']:
                risks.append(f"Weak fundamentals (Grade {fundamental_result.health_grade})")

        if options_result:
            options_analysis_score = options_result.opportunity_score
            if options_result.recommendation in ['avoid']:
                warnings.append("Options analysis suggests avoiding this trade")
            for warn in (options_result.warnings or [])[:2]:
                warnings.append(warn)

        if flow_result:
            flow_signal = flow_result.flow_signal
            if flow_result.flow_signal in ['bearish', 'very_bearish']:
                warnings.append(f"Options flow: {flow_result.flow_signal}")
            elif flow_result.flow_signal in ['bullish', 'very_bullish']:
                key_reasons.append(f"Options flow: {flow_result.flow_signal}")

        if earnings_result:
            if earnings_result.positions_to_close_before_earnings:
                warnings.append(f"⚠️ Earnings warning: Close positions in {earnings_result.positions_to_close_before_earnings}")
                earnings_impact = "close_recommended"
            elif earnings_result.warnings:
                for warn in earnings_result.warnings[:2]:
                    warnings.append(warn)
                earnings_impact = "caution"
            else:
                earnings_impact = "safe"

        # Calculate composite score (with extended weights if available)
        if extended_results:
            # Weight includes extended analysis
            weights = {"strategy": 0.30, "risk": 0.25, "sentiment": 0.20, "fundamental": 0.15, "options": 0.10}
            base_score = (
                strategy_score * weights["strategy"] +
                risk_score * weights["risk"] +
                sentiment_score * weights["sentiment"]
            )
            extended_score = 0
            if fundamental_score is not None:
                extended_score += fundamental_score * weights["fundamental"]
            if options_analysis_score is not None:
                extended_score += options_analysis_score * weights["options"]
            composite_score = int(base_score + extended_score)
        else:
            weights = {"strategy": 0.4, "risk": 0.3, "sentiment": 0.3}
            composite_score = int(
                strategy_score * weights["strategy"] +
                risk_score * weights["risk"] +
                sentiment_score * weights["sentiment"]
            )

        # Determine action
        action, conviction = self._determine_action(
            composite_score, risk_result, strategy_input
        )

        # Position sizing
        config = get_config()
        if composite_score >= 70:
            allocation_pct = config.risk.max_position_size_pct
            max_contracts = 5
        elif composite_score >= 55:
            allocation_pct = config.risk.max_position_size_pct * 0.6
            max_contracts = 3
        else:
            allocation_pct = config.risk.max_position_size_pct * 0.3
            max_contracts = 1

        recommended_contracts = max(1, int(max_contracts * 0.5))

        # Build agent results dict
        agent_results_dict = {
            "strategy": strategy_result.model_dump() if strategy_result else None,
            "risk": risk_result.model_dump() if risk_result else None,
            "sentiment": sentiment_result.model_dump() if sentiment_result else None
        }
        if extended_results:
            agent_results_dict["fundamental"] = fundamental_result.model_dump() if fundamental_result else None
            agent_results_dict["options"] = options_result.model_dump() if options_result else None
            agent_results_dict["flow"] = flow_result.model_dump() if flow_result else None
            agent_results_dict["earnings"] = earnings_result.model_dump() if earnings_result else None

        return TradingDecision(
            symbol=symbol,
            action=action,
            conviction=conviction,
            confidence_score=composite_score,
            recommended_strategy=recommended_strategy,
            strategy_parameters=strategy_params,
            recommended_contracts=recommended_contracts,
            max_contracts=max_contracts,
            allocation_pct=allocation_pct,
            strategy_analysis=strategy_analysis[:500],
            risk_analysis=risk_analysis[:500],
            sentiment_analysis=sentiment_analysis[:500],
            strategy_score=strategy_score,
            risk_score=risk_score,
            sentiment_score=sentiment_score,
            composite_score=composite_score,
            key_reasons=key_reasons[:5],
            warnings=warnings[:8],  # More warnings with extended analysis
            risks=risks[:5],
            fundamental_score=fundamental_score,
            options_analysis_score=options_analysis_score,
            flow_signal=flow_signal,
            earnings_impact=earnings_impact,
            agent_results=agent_results_dict
        )

    def _determine_action(
        self,
        composite_score: int,
        risk_result: Optional[RiskAnalysisOutput],
        strategy_input: Dict
    ) -> tuple[str, str]:
        """Determine action and conviction from scores"""

        # Check for risk blocks
        if risk_result and risk_result.immediate_action_required:
            return "avoid", "high"

        # Check for earnings
        if strategy_input.get("days_to_earnings"):
            if strategy_input["days_to_earnings"] <= 7:
                return "hold", "medium"

        # Score-based decision
        if composite_score >= 80:
            return "strong_buy", "very_high"
        elif composite_score >= 65:
            return "buy", "high"
        elif composite_score >= 50:
            return "buy", "medium"
        elif composite_score >= 40:
            return "hold", "low"
        elif composite_score >= 25:
            return "avoid", "medium"
        else:
            return "strong_avoid", "high"

    def _run_fallback_analysis(
        self,
        symbol: str,
        underlying_price: float,
        iv_rank: float,
        trend: str,
        days_to_earnings: Optional[int],
        portfolio_value: float,
        positions: List[Dict],
        greeks: Dict,
        var_analysis: Dict,
        vix: float,
        put_call_ratio: float,
        risk_tolerance: str
    ) -> TradingDecision:
        """Run rule-based fallback analysis"""
        logger.info("Using fallback analysis (AI unavailable)")

        # Quick strategy check
        strategies = self.quick_strategy.select(
            iv_rank=iv_rank,
            trend=trend,
            days_to_earnings=days_to_earnings,
            risk_tolerance=risk_tolerance
        )

        # Quick risk check
        risk_result = self.quick_risk.check(
            portfolio_value=portfolio_value,
            total_delta=greeks.get("total_delta", 0),
            total_theta=greeks.get("total_theta", 0),
            total_vega=greeks.get("total_vega", 0),
            var_95_pct=var_analysis.get("var_95_pct", 0),
            positions=positions
        )

        # Quick sentiment check
        sentiment_result = self.quick_sentiment.check(
            vix=vix,
            iv_rank=iv_rank,
            put_call_ratio=put_call_ratio
        )

        # Determine scores
        strategy_score = 60 if strategies else 40
        risk_score = 100 - risk_result["risk_score"]
        sentiment_score = sentiment_result["sentiment_score"]

        composite = int(
            strategy_score * 0.4 +
            risk_score * 0.3 +
            sentiment_score * 0.3
        )

        # Determine action
        if risk_result["immediate_action_required"]:
            action, conviction = "avoid", "high"
        elif composite >= 60:
            action, conviction = "buy", "medium"
        elif composite >= 40:
            action, conviction = "hold", "low"
        else:
            action, conviction = "avoid", "medium"

        return TradingDecision(
            symbol=symbol,
            action=action,
            conviction=conviction,
            confidence_score=composite,
            recommended_strategy=strategies[0]["strategy"] if strategies else "",
            strategy_parameters={"delta": strategies[0].get("delta", 0.30)} if strategies else {},
            recommended_contracts=1,
            max_contracts=2,
            allocation_pct=2.0,
            strategy_analysis=f"Fallback: {strategies[0]['reason'] if strategies else 'No strategy'}",
            risk_analysis=f"Risk level: {risk_result['risk_level']}",
            sentiment_analysis=f"Sentiment: {sentiment_result['overall_sentiment']}",
            strategy_score=strategy_score,
            risk_score=risk_score,
            sentiment_score=sentiment_score,
            composite_score=composite,
            key_reasons=[s["reason"] for s in strategies[:2]],
            warnings=[a["title"] for a in risk_result["alerts"][:3]],
            risks=[],
            fallback_used=True
        )

    def get_stats(self) -> Dict:
        """Get orchestrator statistics"""
        total = self._stats["total_analyses"]
        return {
            "total_analyses": total,
            "successful_analyses": self._stats["successful_analyses"],
            "success_rate": self._stats["successful_analyses"] / total if total > 0 else 0,
            "fallback_uses": self._stats["fallback_uses"],
            "fallback_rate": self._stats["fallback_uses"] / total if total > 0 else 0,
            "deep_analyses": self._stats["deep_analyses"],
            "deep_analysis_rate": self._stats["deep_analyses"] / total if total > 0 else 0,
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": self._stats["cache_hits"] / total if total > 0 else 0,
            "avg_analysis_time_ms": self._stats["total_time_ms"] / total if total > 0 else 0,
            "agent_stats": {
                "core": {
                    "strategy": self.strategy_agent.get_stats(),
                    "risk": self.risk_agent.get_stats(),
                    "sentiment": self.sentiment_agent.get_stats()
                },
                "extended": {
                    "fundamental": self.fundamental_agent.get_stats(),
                    "options_analysis": self.options_analysis_agent.get_stats(),
                    "options_flow": self.options_flow_agent.get_stats(),
                    "earnings": self.earnings_agent.get_stats()
                }
            }
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def analyze_trade_opportunity(
    symbol: str,
    underlying_price: float,
    iv_rank: float = 50,
    portfolio_value: float = 100000,
    **kwargs
) -> TradingDecision:
    """
    Convenience function for quick trade analysis.

    Usage:
        decision = await analyze_trade_opportunity(
            symbol="AAPL",
            underlying_price=230.50,
            iv_rank=45
        )
    """
    orchestrator = AITradingOrchestrator()
    await orchestrator.initialize()

    try:
        return await orchestrator.analyze(
            symbol=symbol,
            underlying_price=underlying_price,
            iv_rank=iv_rank,
            portfolio_value=portfolio_value,
            **kwargs
        )
    finally:
        await orchestrator.shutdown()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Trading Orchestrator ===\n")

    async def test_orchestrator():
        print("1. Creating orchestrator...")
        orchestrator = AITradingOrchestrator(fallback_enabled=True)
        await orchestrator.initialize()

        print("\n2. Running fallback analysis (no API key)...")
        try:
            decision = await orchestrator.analyze(
                symbol="AAPL",
                underlying_price=230.50,
                iv_rank=45,
                trend="bullish",
                vix=15.5,
                portfolio_value=100000,
                put_call_ratio=0.85
            )

            print(f"\n   Decision Summary:")
            print(f"   Symbol: {decision.symbol}")
            print(f"   Action: {decision.action}")
            print(f"   Conviction: {decision.conviction}")
            print(f"   Confidence: {decision.confidence_score}")
            print(f"   Strategy: {decision.recommended_strategy}")
            print(f"   Composite Score: {decision.composite_score}")
            print(f"   Analysis Time: {decision.analysis_time_ms:.0f}ms")
            print(f"   Fallback Used: {decision.fallback_used}")

            print(f"\n   Key Reasons:")
            for reason in decision.key_reasons:
                print(f"      - {reason}")

            print(f"\n   Warnings:")
            for warning in decision.warnings:
                print(f"      - {warning}")

        except Exception as e:
            print(f"   Error: {e}")

        print("\n3. Orchestrator Stats:")
        stats = orchestrator.get_stats()
        print(f"   Total analyses: {stats['total_analyses']}")
        print(f"   Fallback rate: {stats['fallback_rate']:.0%}")

        await orchestrator.shutdown()
        print("\n✅ AI Trading Orchestrator ready!")

    asyncio.run(test_orchestrator())
