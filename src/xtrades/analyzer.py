"""
AI-Powered Trade Analyzer for Xtrades
======================================

Modern AI analysis using LangChain with support for multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models via Ollama

Features:
- Trade signal extraction with NER
- Sentiment analysis
- Risk assessment
- Quality scoring
- Async processing
"""

import os
import re
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal
import structlog

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import (
    XtradeAlert, AIAnalysis, TradeSignal,
    SentimentLevel, RiskLevel, TradeStrategy, TradeAction
)

logger = structlog.get_logger(__name__)


# Prompt templates for different analysis tasks
TRADE_EXTRACTION_PROMPT = """You are an expert trade signal parser. Extract trading information from the following alert text.

Alert Text:
{alert_text}

Extract and return a JSON object with these fields:
- ticker: Stock/ETF symbol (uppercase, 1-5 letters)
- strategy: One of [call, put, stock, spread, iron_condor, butterfly, straddle, strangle, covered_call, cash_secured_put, unknown]
- action: One of [bto, sto, btc, stc, buy, sell, unknown] (bto=buy to open, sto=sell to open, etc.)
- strike_price: Strike price for options (number or null)
- expiration_date: Option expiration date in YYYY-MM-DD format (or null)
- entry_price: Entry/current price (number or null)
- target_price: Target/profit price (number or null)
- stop_loss: Stop loss price (number or null)
- quantity: Number of contracts/shares (integer or null)
- confidence: Your confidence in this extraction from 0.0 to 1.0

Only extract information that is explicitly stated. Use null for missing fields.
Return ONLY the JSON object, no other text."""

SENTIMENT_ANALYSIS_PROMPT = """Analyze the sentiment of this trading alert.

Alert Text:
{alert_text}

Ticker: {ticker}

Analyze and return a JSON object:
- sentiment: One of [very_bullish, bullish, neutral, bearish, very_bearish]
- sentiment_score: Float from -1.0 (very bearish) to 1.0 (very bullish)
- reasoning: Brief explanation of the sentiment determination
- key_points: List of 2-4 key points from the alert

Return ONLY the JSON object."""

RISK_ASSESSMENT_PROMPT = """Assess the risk level of this trade.

Alert Text:
{alert_text}

Extracted Trade:
- Ticker: {ticker}
- Strategy: {strategy}
- Action: {action}
- Entry: {entry_price}
- Target: {target_price}
- Stop Loss: {stop_loss}

Analyze and return a JSON object:
- risk_level: One of [low, medium, high, extreme]
- risk_score: Float from 0.0 (no risk) to 1.0 (extreme risk)
- risk_factors: List of identified risk factors
- suggested_action: Brief suggestion for the trade
- quality_score: Trade setup quality from 0.0 to 1.0 based on completeness and clarity

Return ONLY the JSON object."""


class AITradeAnalyzer:
    """
    AI-powered trade alert analyzer using LangChain.

    Supports multiple LLM backends with automatic fallback.
    """

    # Token/cost tracking
    total_tokens_used: int = 0
    total_cost: float = 0.0

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout: int = 30
    ):
        """
        Initialize the AI analyzer.

        Args:
            provider: LLM provider - "openai", "anthropic", "ollama", or "auto"
            model: Specific model name (provider-dependent)
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._llm = None
        self._extraction_chain = None
        self._sentiment_chain = None
        self._risk_chain = None

        self.logger = logger.bind(component="AITradeAnalyzer")

    def _get_llm(self):
        """Get or create the LLM instance."""
        if self._llm is not None:
            return self._llm

        provider = self.provider

        # Auto-detect available provider
        if provider == "auto":
            if os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif self._check_ollama_available():
                provider = "ollama"
            else:
                raise ValueError("No LLM provider available. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama.")

        self.logger.info("Initializing LLM", provider=provider, model=self.model)

        if provider == "openai":
            self._llm = ChatOpenAI(
                model=self.model or "gpt-4o-mini",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
        elif provider == "anthropic":
            self._llm = ChatAnthropic(
                model=self.model or "claude-3-haiku-20240307",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
        elif provider == "ollama":
            self._llm = ChatOllama(
                model=self.model or "llama3.2",
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return self._llm

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _get_extraction_chain(self):
        """Get or create the trade extraction chain."""
        if self._extraction_chain is None:
            prompt = ChatPromptTemplate.from_template(TRADE_EXTRACTION_PROMPT)
            parser = JsonOutputParser()
            self._extraction_chain = prompt | self._get_llm() | parser
        return self._extraction_chain

    def _get_sentiment_chain(self):
        """Get or create the sentiment analysis chain."""
        if self._sentiment_chain is None:
            prompt = ChatPromptTemplate.from_template(SENTIMENT_ANALYSIS_PROMPT)
            parser = JsonOutputParser()
            self._sentiment_chain = prompt | self._get_llm() | parser
        return self._sentiment_chain

    def _get_risk_chain(self):
        """Get or create the risk assessment chain."""
        if self._risk_chain is None:
            prompt = ChatPromptTemplate.from_template(RISK_ASSESSMENT_PROMPT)
            parser = JsonOutputParser()
            self._risk_chain = prompt | self._get_llm() | parser
        return self._risk_chain

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    async def extract_trade_signal(self, alert_text: str) -> Optional[TradeSignal]:
        """
        Extract trade signal from alert text using AI.

        Args:
            alert_text: Raw alert text to parse

        Returns:
            TradeSignal if extraction successful, None otherwise
        """
        start_time = datetime.utcnow()

        try:
            chain = self._get_extraction_chain()
            result = await chain.ainvoke({"alert_text": alert_text})

            self.logger.debug("Trade extraction complete", result=result)

            # Validate and convert result
            ticker = result.get("ticker")
            if not ticker or len(ticker) > 10:
                return None

            # Map strategy
            strategy_map = {
                "call": TradeStrategy.CALL,
                "put": TradeStrategy.PUT,
                "stock": TradeStrategy.STOCK,
                "spread": TradeStrategy.SPREAD,
                "iron_condor": TradeStrategy.IRON_CONDOR,
                "butterfly": TradeStrategy.BUTTERFLY,
                "straddle": TradeStrategy.STRADDLE,
                "strangle": TradeStrategy.STRANGLE,
                "covered_call": TradeStrategy.COVERED_CALL,
                "cash_secured_put": TradeStrategy.CASH_SECURED_PUT,
            }
            strategy = strategy_map.get(result.get("strategy", "").lower(), TradeStrategy.UNKNOWN)

            # Map action
            action_map = {
                "bto": TradeAction.BTO,
                "sto": TradeAction.STO,
                "btc": TradeAction.BTC,
                "stc": TradeAction.STC,
                "buy": TradeAction.BUY,
                "sell": TradeAction.SELL,
            }
            action = action_map.get(result.get("action", "").lower(), TradeAction.UNKNOWN)

            # Parse expiration date
            exp_date = None
            if result.get("expiration_date"):
                try:
                    exp_date = datetime.strptime(result["expiration_date"], "%Y-%m-%d")
                except ValueError:
                    pass

            signal = TradeSignal(
                ticker=ticker.upper(),
                strategy=strategy,
                action=action,
                strike_price=Decimal(str(result["strike_price"])) if result.get("strike_price") else None,
                expiration_date=exp_date,
                entry_price=Decimal(str(result["entry_price"])) if result.get("entry_price") else None,
                target_price=Decimal(str(result["target_price"])) if result.get("target_price") else None,
                stop_loss=Decimal(str(result["stop_loss"])) if result.get("stop_loss") else None,
                quantity=int(result["quantity"]) if result.get("quantity") else None,
                confidence_score=float(result.get("confidence", 0.5)),
                raw_text=alert_text
            )

            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.logger.info("Trade signal extracted", ticker=ticker, duration_ms=duration_ms)

            return signal

        except Exception as e:
            self.logger.error("Trade extraction failed", error=str(e))
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    async def analyze_sentiment(self, alert_text: str, ticker: str) -> Tuple[SentimentLevel, float, str, List[str]]:
        """
        Analyze sentiment of an alert.

        Returns:
            Tuple of (sentiment_level, score, reasoning, key_points)
        """
        try:
            chain = self._get_sentiment_chain()
            result = await chain.ainvoke({
                "alert_text": alert_text,
                "ticker": ticker
            })

            # Map sentiment
            sentiment_map = {
                "very_bullish": SentimentLevel.VERY_BULLISH,
                "bullish": SentimentLevel.BULLISH,
                "neutral": SentimentLevel.NEUTRAL,
                "bearish": SentimentLevel.BEARISH,
                "very_bearish": SentimentLevel.VERY_BEARISH,
            }
            sentiment = sentiment_map.get(result.get("sentiment", "neutral"), SentimentLevel.NEUTRAL)
            score = float(result.get("sentiment_score", 0.0))
            reasoning = result.get("reasoning", "")
            key_points = result.get("key_points", [])

            return sentiment, score, reasoning, key_points

        except Exception as e:
            self.logger.error("Sentiment analysis failed", error=str(e))
            return SentimentLevel.NEUTRAL, 0.0, "", []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    async def assess_risk(
        self,
        alert_text: str,
        ticker: str,
        strategy: Optional[str] = None,
        action: Optional[str] = None,
        entry_price: Optional[Decimal] = None,
        target_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None
    ) -> Tuple[RiskLevel, float, List[str], str, float]:
        """
        Assess risk level of a trade.

        Returns:
            Tuple of (risk_level, risk_score, risk_factors, suggested_action, quality_score)
        """
        try:
            chain = self._get_risk_chain()
            result = await chain.ainvoke({
                "alert_text": alert_text,
                "ticker": ticker,
                "strategy": strategy or "unknown",
                "action": action or "unknown",
                "entry_price": str(entry_price) if entry_price else "not specified",
                "target_price": str(target_price) if target_price else "not specified",
                "stop_loss": str(stop_loss) if stop_loss else "not specified",
            })

            # Map risk level
            risk_map = {
                "low": RiskLevel.LOW,
                "medium": RiskLevel.MEDIUM,
                "high": RiskLevel.HIGH,
                "extreme": RiskLevel.EXTREME,
            }
            risk_level = risk_map.get(result.get("risk_level", "medium"), RiskLevel.MEDIUM)
            risk_score = float(result.get("risk_score", 0.5))
            risk_factors = result.get("risk_factors", [])
            suggested_action = result.get("suggested_action", "")
            quality_score = float(result.get("quality_score", 0.5))

            return risk_level, risk_score, risk_factors, suggested_action, quality_score

        except Exception as e:
            self.logger.error("Risk assessment failed", error=str(e))
            return RiskLevel.MEDIUM, 0.5, [], "", 0.5

    async def analyze_alert(self, alert: XtradeAlert) -> AIAnalysis:
        """
        Perform comprehensive AI analysis on an alert.

        Args:
            alert: XtradeAlert to analyze

        Returns:
            AIAnalysis with all analysis results
        """
        start_time = datetime.utcnow()
        tokens_used = 0

        self.logger.info("Starting AI analysis", alert_id=alert.alert_id)

        # Extract trade signal if not already done
        ticker = alert.ticker
        if not ticker:
            signal = await self.extract_trade_signal(alert.alert_text)
            if signal:
                ticker = signal.ticker
                alert.ticker = ticker
                alert.strategy = signal.strategy.value
                alert.action = signal.action.value
                alert.strike_price = signal.strike_price
                alert.expiration_date = signal.expiration_date
                alert.entry_price = signal.entry_price
                alert.target_price = signal.target_price
                alert.stop_loss = signal.stop_loss
                alert.quantity = signal.quantity

        if not ticker:
            # Cannot analyze without ticker
            return AIAnalysis(
                alert_id=alert.alert_id,
                ticker="UNKNOWN",
                summary="Could not extract ticker from alert",
                quality_score=0.0
            )

        # Run sentiment and risk analysis in parallel
        sentiment_task = self.analyze_sentiment(alert.alert_text, ticker)
        risk_task = self.assess_risk(
            alert.alert_text,
            ticker,
            alert.strategy,
            alert.action,
            alert.entry_price,
            alert.target_price,
            alert.stop_loss
        )

        (sentiment, sentiment_score, reasoning, key_points), \
        (risk_level, risk_score, risk_factors, suggested_action, quality_score) = \
            await asyncio.gather(sentiment_task, risk_task)

        # Calculate completeness based on filled fields
        filled_fields = sum([
            bool(alert.ticker),
            bool(alert.strategy),
            bool(alert.action),
            bool(alert.entry_price),
            bool(alert.target_price or alert.stop_loss),
        ])
        completeness_score = filled_fields / 5.0

        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        analysis = AIAnalysis(
            alert_id=alert.alert_id,
            ticker=ticker,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            sentiment_reasoning=reasoning,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            quality_score=quality_score,
            completeness_score=completeness_score,
            summary=reasoning,
            key_points=key_points,
            suggested_action=suggested_action,
            model_used=self.model or "auto",
            analysis_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time_ms,
            tokens_used=tokens_used
        )

        # Update alert with AI analysis
        alert.sentiment = sentiment
        alert.risk_level = risk_level
        alert.ai_summary = reasoning
        alert.ai_confidence = quality_score

        self.logger.info(
            "AI analysis complete",
            alert_id=alert.alert_id,
            ticker=ticker,
            sentiment=sentiment.value,
            risk=risk_level.value,
            duration_ms=processing_time_ms
        )

        return analysis

    async def batch_analyze(
        self,
        alerts: List[XtradeAlert],
        max_concurrent: int = 5
    ) -> List[AIAnalysis]:
        """
        Analyze multiple alerts concurrently.

        Args:
            alerts: List of alerts to analyze
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of AIAnalysis results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(alert: XtradeAlert) -> AIAnalysis:
            async with semaphore:
                return await self.analyze_alert(alert)

        tasks = [analyze_with_limit(alert) for alert in alerts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "Batch analysis failed for alert",
                    alert_id=alerts[i].alert_id,
                    error=str(result)
                )
            else:
                analyses.append(result)

        return analyses


# Convenience function for simple extraction
async def extract_trade_from_text(text: str, provider: str = "auto") -> Optional[TradeSignal]:
    """
    Quick helper to extract trade signal from text.

    Args:
        text: Alert text
        provider: LLM provider

    Returns:
        TradeSignal if found
    """
    analyzer = AITradeAnalyzer(provider=provider)
    return await analyzer.extract_trade_signal(text)
