"""
Modern AI Client Infrastructure
================================

Production-grade AI/LLM client with:
- Structured outputs using Instructor
- Multi-provider support (Anthropic, OpenAI, Groq, Ollama)
- Automatic retry with exponential backoff
- Response caching
- Streaming support
- Cost tracking
- Observability integration

Author: AVA Trading Platform
Updated: 2025-11-29
"""

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

import structlog
from pydantic import BaseModel, Field

# Instructor for structured outputs
try:
    import instructor
    from instructor import Mode

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    instructor = None
    Mode = None

# LLM Clients
try:
    import anthropic
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    AsyncAnthropic = None

try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    AsyncOpenAI = None

try:
    from groq import AsyncGroq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    AsyncGroq = None

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Enums & Configuration
# =============================================================================


class AIProvider(str, Enum):
    """Supported AI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"


class ModelTier(str, Enum):
    """Model capability tiers."""

    FAST = "fast"  # Quick, cheap responses
    BALANCED = "balanced"  # Good balance of speed/quality
    POWERFUL = "powerful"  # Best quality, slower/expensive


# Default models per provider and tier
DEFAULT_MODELS: Dict[AIProvider, Dict[ModelTier, str]] = {
    AIProvider.ANTHROPIC: {
        ModelTier.FAST: "claude-3-5-haiku-20241022",
        ModelTier.BALANCED: "claude-sonnet-4-20250514",
        ModelTier.POWERFUL: "claude-sonnet-4-20250514",
    },
    AIProvider.OPENAI: {
        ModelTier.FAST: "gpt-4o-mini",
        ModelTier.BALANCED: "gpt-4o",
        ModelTier.POWERFUL: "gpt-4o",
    },
    AIProvider.GROQ: {
        ModelTier.FAST: "llama-3.1-8b-instant",
        ModelTier.BALANCED: "llama-3.3-70b-versatile",
        ModelTier.POWERFUL: "llama-3.3-70b-versatile",
    },
    AIProvider.OLLAMA: {
        ModelTier.FAST: "llama3.2",
        ModelTier.BALANCED: "llama3.2",
        ModelTier.POWERFUL: "llama3.2:70b",
    },
}

# Token pricing per 1M tokens (input, output)
TOKEN_PRICING: Dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Groq (free tier, then paid)
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


@dataclass(frozen=True)
class AIConfig:
    """AI client configuration."""

    provider: AIProvider = AIProvider.ANTHROPIC
    model: Optional[str] = None
    tier: ModelTier = ModelTier.BALANCED

    # API keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    ollama_host: str = "http://localhost:11434"

    # Request settings
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: float = 60.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 300

    @classmethod
    def from_env(cls) -> "AIConfig":
        """Create config from environment."""
        provider = AIProvider(os.getenv("LLM_PROVIDER", "anthropic"))
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        )

    def get_model(self) -> str:
        """Get the model to use."""
        if self.model:
            return self.model
        return DEFAULT_MODELS[self.provider][self.tier]


# =============================================================================
# Response Types
# =============================================================================


@dataclass
class AIUsage:
    """Token usage and cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    def calculate_cost(self, model: str) -> float:
        """Calculate cost based on token usage."""
        if model not in TOKEN_PRICING:
            return 0.0
        input_price, output_price = TOKEN_PRICING[model]
        self.cost_usd = (
            (self.input_tokens * input_price / 1_000_000)
            + (self.output_tokens * output_price / 1_000_000)
        )
        return self.cost_usd


@dataclass
class AIResponse(Generic[T]):
    """Typed response from AI client."""

    content: T
    raw_content: str
    model: str
    provider: AIProvider
    usage: AIUsage
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_cached(self) -> bool:
        return self.usage.cached


# =============================================================================
# Structured Output Models
# =============================================================================


class TradeRecommendation(BaseModel):
    """Structured trade recommendation from AI."""

    action: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
    score: int = Field(ge=-100, le=100, description="Score -100 to 100")

    technical_analysis: str = Field(description="Technical analysis summary")
    fundamental_analysis: str = Field(description="Fundamental analysis summary")
    risk_assessment: str = Field(description="Risk assessment summary")

    bullish_factors: List[str] = Field(default_factory=list)
    bearish_factors: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)

    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = Field(
        default=0.02, ge=0.0, le=0.1, description="Position size as % of portfolio"
    )

    reasoning: str = Field(description="Detailed reasoning for recommendation")


class OptionsStrategy(BaseModel):
    """Structured options strategy recommendation."""

    strategy_name: str = Field(description="Name of the strategy")
    strategy_type: Literal[
        "cash_secured_put",
        "covered_call",
        "iron_condor",
        "credit_spread",
        "debit_spread",
        "calendar_spread",
        "straddle",
        "strangle",
    ]
    fit_score: int = Field(ge=0, le=100, description="How well this fits current conditions")

    legs: List[Dict[str, Any]] = Field(description="Option legs")

    max_profit: float
    max_loss: float
    break_even: List[float]
    probability_of_profit: float = Field(ge=0.0, le=1.0)

    iv_environment: Literal["low", "normal", "elevated", "extreme"]
    ideal_conditions: List[str]
    risks: List[str]
    adjustments: List[str] = Field(description="Potential adjustment strategies")

    reasoning: str


class MarketAnalysis(BaseModel):
    """Structured market analysis."""

    symbol: str
    analysis_type: Literal["technical", "fundamental", "sentiment", "comprehensive"]

    trend: Literal["bullish", "bearish", "neutral", "mixed"]
    trend_strength: float = Field(ge=0.0, le=1.0)

    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)

    key_indicators: Dict[str, Any] = Field(default_factory=dict)
    catalysts: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)

    price_target_low: Optional[float] = None
    price_target_mid: Optional[float] = None
    price_target_high: Optional[float] = None
    timeframe: str = Field(default="30d", description="Analysis timeframe")

    summary: str
    detailed_analysis: str


class SportsPrediction(BaseModel):
    """Structured sports prediction."""

    game_id: str
    sport: str
    home_team: str
    away_team: str

    predicted_winner: str
    win_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    predicted_spread: float
    spread_confidence: float = Field(ge=0.0, le=1.0)

    predicted_total: float
    over_under_pick: Literal["over", "under"]
    total_confidence: float = Field(ge=0.0, le=1.0)

    key_factors: List[str]
    injury_impact: str
    weather_impact: Optional[str] = None

    value_bets: List[Dict[str, Any]] = Field(default_factory=list)
    kelly_fraction: float = Field(ge=0.0, le=1.0, default=0.0)

    reasoning: str


# =============================================================================
# AI Client Implementation
# =============================================================================


class AIClient:
    """
    Modern AI client with structured outputs.

    Features:
    - Multi-provider support (Anthropic, OpenAI, Groq, Ollama)
    - Structured outputs using Instructor/Pydantic
    - Automatic retry with backoff
    - Response caching
    - Cost tracking
    - Streaming support

    Usage:
        client = AIClient()

        # Structured output
        recommendation = await client.generate(
            prompt="Analyze AAPL for a swing trade",
            response_model=TradeRecommendation,
        )

        # Streaming
        async for chunk in client.stream("Explain market conditions"):
            print(chunk, end="")
    """

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig.from_env()
        self._client = None
        self._instructor_client = None
        self._initialized = False
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._total_usage = AIUsage()

        logger.info(
            "ai_client_initialized",
            provider=self.config.provider.value,
            model=self.config.get_model(),
        )

    async def _ensure_initialized(self) -> None:
        """Lazy initialize the underlying client."""
        if self._initialized:
            return

        provider = self.config.provider

        if provider == AIProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic not installed: pip install anthropic")
            self._client = AsyncAnthropic(api_key=self.config.anthropic_api_key)
            if INSTRUCTOR_AVAILABLE:
                self._instructor_client = instructor.from_anthropic(
                    self._client, mode=Mode.ANTHROPIC_JSON
                )

        elif provider == AIProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed: pip install openai")
            self._client = AsyncOpenAI(api_key=self.config.openai_api_key)
            if INSTRUCTOR_AVAILABLE:
                self._instructor_client = instructor.from_openai(self._client)

        elif provider == AIProvider.GROQ:
            if not GROQ_AVAILABLE:
                raise ImportError("groq not installed: pip install groq")
            self._client = AsyncGroq(api_key=self.config.groq_api_key)
            if INSTRUCTOR_AVAILABLE:
                self._instructor_client = instructor.from_groq(self._client)

        elif provider == AIProvider.OLLAMA:
            # Ollama uses HTTP API
            if not HTTPX_AVAILABLE:
                raise ImportError("httpx not installed: pip install httpx")
            self._client = httpx.AsyncClient(
                base_url=self.config.ollama_host,
                timeout=self.config.timeout,
            )

        self._initialized = True

    def _cache_key(self, prompt: str, system: str, model: str) -> str:
        """Generate cache key."""
        content = f"{prompt}:{system}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached response if valid."""
        if not self.config.cache_enabled or key not in self._cache:
            return None

        response, timestamp = self._cache[key]
        age = (datetime.now() - timestamp).total_seconds()

        if age > self.config.cache_ttl:
            del self._cache[key]
            return None

        return response

    def _set_cached(self, key: str, response: Any) -> None:
        """Cache a response."""
        if self.config.cache_enabled:
            self._cache[key] = (response, datetime.now())

    async def generate(
        self,
        prompt: str,
        response_model: Type[T],
        system: str = "You are a helpful trading assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> AIResponse[T]:
        """
        Generate a structured response.

        Args:
            prompt: User prompt
            response_model: Pydantic model for structured output
            system: System prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            use_cache: Whether to use caching

        Returns:
            AIResponse with typed content
        """
        await self._ensure_initialized()

        model = self.config.get_model()
        cache_key = self._cache_key(prompt, system, model)

        # Check cache
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug("ai_cache_hit", model=model)
                cached.usage.cached = True
                return cached

        # Make request with retry
        start_time = time.perf_counter()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self._generate_with_instructor(
                    prompt=prompt,
                    system=system,
                    response_model=response_model,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                # Build response
                result = AIResponse(
                    content=response["content"],
                    raw_content=response.get("raw_content", ""),
                    model=model,
                    provider=self.config.provider,
                    usage=AIUsage(
                        input_tokens=response.get("input_tokens", 0),
                        output_tokens=response.get("output_tokens", 0),
                        total_tokens=response.get("total_tokens", 0),
                    ),
                    latency_ms=latency_ms,
                )

                # Calculate cost
                result.usage.calculate_cost(model)

                # Update totals
                self._total_usage.input_tokens += result.usage.input_tokens
                self._total_usage.output_tokens += result.usage.output_tokens
                self._total_usage.cost_usd += result.usage.cost_usd

                # Cache result
                if use_cache:
                    self._set_cached(cache_key, result)

                logger.info(
                    "ai_generation_complete",
                    model=model,
                    latency_ms=round(latency_ms, 2),
                    tokens=result.usage.total_tokens,
                    cost_usd=round(result.usage.cost_usd, 6),
                )

                return result

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (
                        self.config.retry_backoff ** attempt
                    )
                    logger.warning(
                        "ai_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        raise last_error or Exception("AI generation failed")

    async def _generate_with_instructor(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Generate using instructor for structured output."""
        provider = self.config.provider
        model = self.config.get_model()

        if provider == AIProvider.ANTHROPIC and self._instructor_client:
            response = await self._instructor_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
            )
            return {
                "content": response,
                "raw_content": response.model_dump_json(),
                "input_tokens": 0,  # Instructor doesn't expose these
                "output_tokens": 0,
                "total_tokens": 0,
            }

        elif provider == AIProvider.OPENAI and self._instructor_client:
            response = await self._instructor_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_model=response_model,
            )
            return {
                "content": response,
                "raw_content": response.model_dump_json(),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        elif provider == AIProvider.GROQ and self._instructor_client:
            response = await self._instructor_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_model=response_model,
            )
            return {
                "content": response,
                "raw_content": response.model_dump_json(),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        elif provider == AIProvider.OLLAMA:
            # Ollama requires manual JSON parsing
            return await self._generate_ollama_structured(
                prompt, system, response_model, temperature, max_tokens
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _generate_ollama_structured(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Generate structured output from Ollama."""
        model = self.config.get_model()

        # Add JSON schema to prompt for Ollama
        schema = response_model.model_json_schema()
        enhanced_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Only respond with the JSON object, no other text."""

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": enhanced_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "format": "json",
        }

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        content_str = data.get("message", {}).get("content", "{}")

        # Parse JSON and validate with Pydantic
        parsed = json.loads(content_str)
        validated = response_model.model_validate(parsed)

        return {
            "content": validated,
            "raw_content": content_str,
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "total_tokens": (
                data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            ),
        }

    async def generate_text(
        self,
        prompt: str,
        system: str = "You are a helpful trading assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate unstructured text response."""
        await self._ensure_initialized()

        provider = self.config.provider
        model = self.config.get_model()
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        if provider == AIProvider.ANTHROPIC:
            response = await self._client.messages.create(
                model=model,
                max_tokens=tokens,
                temperature=temp,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif provider == AIProvider.OPENAI:
            response = await self._client.chat.completions.create(
                model=model,
                max_tokens=tokens,
                temperature=temp,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""

        elif provider == AIProvider.GROQ:
            response = await self._client.chat.completions.create(
                model=model,
                max_tokens=tokens,
                temperature=temp,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""

        elif provider == AIProvider.OLLAMA:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": temp, "num_predict": tokens},
            }
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")

        raise ValueError(f"Unknown provider: {provider}")

    async def stream(
        self,
        prompt: str,
        system: str = "You are a helpful trading assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text response."""
        await self._ensure_initialized()

        provider = self.config.provider
        model = self.config.get_model()
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        if provider == AIProvider.ANTHROPIC:
            async with self._client.messages.stream(
                model=model,
                max_tokens=tokens,
                temperature=temp,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        elif provider in (AIProvider.OPENAI, AIProvider.GROQ):
            stream = await self._client.chat.completions.create(
                model=model,
                max_tokens=tokens,
                temperature=temp,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif provider == AIProvider.OLLAMA:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {"temperature": temp, "num_predict": tokens},
            }
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get total usage statistics."""
        return {
            "total_input_tokens": self._total_usage.input_tokens,
            "total_output_tokens": self._total_usage.output_tokens,
            "total_tokens": (
                self._total_usage.input_tokens + self._total_usage.output_tokens
            ),
            "total_cost_usd": round(self._total_usage.cost_usd, 6),
            "cache_size": len(self._cache),
        }


# =============================================================================
# Singleton & Factory
# =============================================================================

_ai_client: Optional[AIClient] = None


def get_ai_client(config: Optional[AIConfig] = None) -> AIClient:
    """Get or create the global AI client."""
    global _ai_client
    if _ai_client is None or config is not None:
        _ai_client = AIClient(config)
    return _ai_client


# =============================================================================
# Convenience Functions
# =============================================================================


async def analyze_trade(
    symbol: str,
    context: Dict[str, Any],
    client: Optional[AIClient] = None,
) -> AIResponse[TradeRecommendation]:
    """
    Analyze a potential trade and get a structured recommendation.

    Args:
        symbol: Stock symbol
        context: Market context (price, IV, fundamentals, etc.)
        client: Optional AI client override

    Returns:
        Structured TradeRecommendation
    """
    client = client or get_ai_client()

    prompt = f"""Analyze {symbol} for a potential trade:

Current Price: ${context.get('price', 0):.2f}
IV Rank: {context.get('iv_rank', 50):.1f}
IV Percentile: {context.get('iv_percentile', 50):.1f}
Trend: {context.get('trend', 'neutral')}
RSI: {context.get('rsi', 50):.1f}
MACD: {context.get('macd', 0):.4f}
Volume Ratio: {context.get('volume_ratio', 1.0):.2f}
Sector: {context.get('sector', 'Unknown')}
Days to Earnings: {context.get('days_to_earnings', 'N/A')}

52-Week High: ${context.get('week_52_high', 0):.2f}
52-Week Low: ${context.get('week_52_low', 0):.2f}

Provide a comprehensive trade analysis with specific entry, stop loss, and profit targets."""

    return await client.generate(
        prompt=prompt,
        response_model=TradeRecommendation,
        system="""You are an expert trading analyst. Analyze the provided data and give
a clear, actionable trade recommendation. Be specific about entry points, stop losses,
and profit targets. Consider both technical and fundamental factors.""",
    )


async def recommend_options_strategy(
    symbol: str,
    context: Dict[str, Any],
    account_size: float = 100000,
    risk_tolerance: str = "moderate",
    client: Optional[AIClient] = None,
) -> AIResponse[OptionsStrategy]:
    """
    Get an options strategy recommendation.

    Args:
        symbol: Underlying symbol
        context: Market context
        account_size: Account size for position sizing
        risk_tolerance: "conservative", "moderate", or "aggressive"
        client: Optional AI client override

    Returns:
        Structured OptionsStrategy recommendation
    """
    client = client or get_ai_client()

    prompt = f"""Recommend an options strategy for {symbol}:

Current Price: ${context.get('price', 0):.2f}
IV Rank: {context.get('iv_rank', 50):.1f}
Historical Volatility: {context.get('hv_20', 0.20):.1%}
Trend: {context.get('trend', 'neutral')}
Days to Earnings: {context.get('days_to_earnings', 'N/A')}

Account Size: ${account_size:,.2f}
Risk Tolerance: {risk_tolerance}

Consider IV environment, trend, and earnings when recommending a strategy.
Provide specific strikes, expirations, and expected outcomes."""

    return await client.generate(
        prompt=prompt,
        response_model=OptionsStrategy,
        system="""You are an expert options strategist. Based on the market conditions,
recommend the most appropriate options strategy. Consider IV environment (high IV favors
selling premium, low IV favors buying), trend direction, and upcoming events like earnings.
Provide specific, actionable recommendations.""",
    )


async def predict_sports_game(
    game_data: Dict[str, Any],
    client: Optional[AIClient] = None,
) -> AIResponse[SportsPrediction]:
    """
    Generate a sports prediction.

    Args:
        game_data: Game information (teams, stats, odds, etc.)
        client: Optional AI client override

    Returns:
        Structured SportsPrediction
    """
    client = client or get_ai_client()

    prompt = f"""Analyze this game and provide predictions:

Sport: {game_data.get('sport', 'NFL')}
Home Team: {game_data.get('home_team')}
Away Team: {game_data.get('away_team')}

Current Spread: {game_data.get('spread', 0)}
Current Total: {game_data.get('total', 0)}
Moneyline (Home/Away): {game_data.get('home_ml', 0)}/{game_data.get('away_ml', 0)}

Home Team Stats:
{json.dumps(game_data.get('home_stats', {}), indent=2)}

Away Team Stats:
{json.dumps(game_data.get('away_stats', {}), indent=2)}

Injuries: {game_data.get('injuries', 'None reported')}
Weather: {game_data.get('weather', 'N/A')}

Provide detailed predictions with probabilities and identify any value bets."""

    return await client.generate(
        prompt=prompt,
        response_model=SportsPrediction,
        system="""You are an expert sports analyst specializing in NFL, NBA, NCAA, and MLB.
Analyze the provided data to make predictions. Consider team stats, injuries, weather,
and identify value bets where the true probability differs from implied odds.""",
    )
