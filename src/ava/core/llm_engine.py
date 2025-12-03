"""
AVA LLM Decision Engine
=======================

Claude-powered AI decision engine for trading analysis.
Provides intelligent trade recommendations, risk assessment,
and natural language explanations.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Any, Callable, TypeVar
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
from functools import wraps

logger = logging.getLogger(__name__)

# Type variable for generic retry
T = TypeVar('T')


# =============================================================================
# LLM CLIENT ABSTRACTION
# =============================================================================

class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"  # Local LLM - FREE
    GROQ = "groq"  # Cloud with generous free tier
    HUGGINGFACE = "huggingface"  # HuggingFace Inference API


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    provider: LLMProvider
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False
    raw_response: Optional[Any] = None


class LLMClient:
    """
    Unified LLM client with caching, retry logic, and provider abstraction.

    Usage:
        client = LLMClient(provider=LLMProvider.ANTHROPIC)
        response = await client.generate(
            system="You are a trading analyst.",
            messages=[{"role": "user", "content": "Analyze AAPL"}]
        )
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.ANTHROPIC,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_enabled: bool = True,
        cache_ttl: int = 300
    ):
        self.provider = provider
        self.model = model or self._default_model()
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl

        # Response cache
        self._cache: Dict[str, tuple[LLMResponse, datetime]] = {}

        # Initialize client
        self._client = None
        self._initialized = False

    def _default_model(self) -> str:
        if self.provider == LLMProvider.ANTHROPIC:
            return "claude-sonnet-4-20250514"
        elif self.provider == LLMProvider.OPENAI:
            return "gpt-4-turbo"
        elif self.provider == LLMProvider.OLLAMA:
            return "llama3.2"  # Good default, can also use mistral, codellama
        elif self.provider == LLMProvider.GROQ:
            return "llama-3.3-70b-versatile"  # Free tier available
        elif self.provider == LLMProvider.HUGGINGFACE:
            return "mistralai/Mistral-7B-Instruct-v0.3"
        return "llama3.2"

    async def _ensure_initialized(self):
        """Lazy initialization of API client"""
        if self._initialized:
            return

        if self.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
                self._initialized = True
            except ImportError:
                logger.error("anthropic package not installed")
                raise

        elif self.provider == LLMProvider.OPENAI:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
                self._initialized = True
            except ImportError:
                logger.error("openai package not installed")
                raise

        elif self.provider == LLMProvider.OLLAMA:
            # Ollama uses HTTP API - no special client needed
            import os
            self._ollama_host = os.getenv(
                "OLLAMA_HOST", "http://localhost:11434"
            )
            self._initialized = True
            logger.info(f"Ollama initialized at {self._ollama_host}")

        elif self.provider == LLMProvider.GROQ:
            try:
                import os
                from groq import AsyncGroq
                api_key = self.api_key or os.getenv("GROQ_API_KEY")
                self._client = AsyncGroq(api_key=api_key)
                self._initialized = True
            except ImportError:
                logger.error("groq package not installed: pip install groq")
                raise

        elif self.provider == LLMProvider.HUGGINGFACE:
            import os
            self._hf_token = self.api_key or os.getenv("HF_TOKEN")
            self._initialized = True

    def _cache_key(self, system: str, messages: List[Dict]) -> str:
        """Generate cache key from request"""
        content = json.dumps({
            "system": system,
            "messages": messages,
            "model": self.model
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[LLMResponse]:
        """Get cached response if valid"""
        if not self.cache_enabled or key not in self._cache:
            return None

        response, timestamp = self._cache[key]
        age = (datetime.now() - timestamp).total_seconds()

        if age > self.cache_ttl:
            del self._cache[key]
            return None

        response.cached = True
        return response

    def _set_cached(self, key: str, response: LLMResponse):
        """Cache response"""
        if self.cache_enabled:
            self._cache[key] = (response, datetime.now())

    async def generate(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
        tools: Optional[List[Dict]] = None,
        use_cache: bool = True
    ) -> LLMResponse:
        """
        Generate LLM response with retry and caching.

        Args:
            system: System prompt
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            tools: Optional tool definitions for function calling
            use_cache: Whether to use response cache

        Returns:
            LLMResponse with content and metadata
        """
        await self._ensure_initialized()

        # Check cache
        if use_cache:
            cache_key = self._cache_key(system, messages)
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug(f"Cache hit for LLM request")
                return cached

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = datetime.now()

                if self.provider == LLMProvider.ANTHROPIC:
                    response = await self._call_anthropic(
                        system, messages, max_tokens, temperature, tools
                    )
                elif self.provider == LLMProvider.OPENAI:
                    response = await self._call_openai(
                        system, messages, max_tokens, temperature, tools
                    )
                elif self.provider == LLMProvider.OLLAMA:
                    response = await self._call_ollama(
                        system, messages, max_tokens, temperature
                    )
                elif self.provider == LLMProvider.GROQ:
                    response = await self._call_groq(
                        system, messages, max_tokens, temperature
                    )
                elif self.provider == LLMProvider.HUGGINGFACE:
                    response = await self._call_huggingface(
                        system, messages, max_tokens, temperature
                    )
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

                response.latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Cache successful response
                if use_cache:
                    self._set_cached(cache_key, response)

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise last_error or Exception("LLM call failed after retries")

    async def _call_anthropic(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[Dict]]
    ) -> LLMResponse:
        """Call Anthropic Claude API"""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages
        }

        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text if response.content else "",
            model=self.model,
            provider=self.provider,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=0,
            raw_response=response
        )

    async def _call_openai(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[Dict]]
    ) -> LLMResponse:
        """Call OpenAI API"""
        full_messages = [{"role": "system", "content": system}] + messages

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": full_messages
        }

        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]

        response = await self._client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            provider=self.provider,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=0,
            raw_response=response
        )

    async def _call_ollama(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Call local Ollama API - FREE"""
        import aiohttp

        # Build messages in Ollama format
        ollama_messages = [{"role": "system", "content": system}]
        for msg in messages:
            ollama_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self._ollama_host}/api/chat"
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Ollama error: {error_text}")
                data = await resp.json()

        content = data.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=0,
            raw_response=data
        )

    async def _call_groq(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Call Groq API - FREE tier available (generous limits)"""
        full_messages = [{"role": "system", "content": system}] + messages

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            provider=self.provider,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=0,
            raw_response=response
        )

    async def _call_huggingface(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Call HuggingFace Inference API - FREE tier available"""
        import aiohttp

        # Combine system and user messages into a prompt
        prompt = f"<|system|>\n{system}<|end|>\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|{role}|>\n{content}<|end|>\n"
        prompt += "<|assistant|>\n"

        headers = {}
        if self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }

        url = f"https://api-inference.huggingface.co/models/{self.model}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"HuggingFace error: {error_text}")
                data = await resp.json()

        # Handle response format
        if isinstance(data, list) and len(data) > 0:
            content = data[0].get("generated_text", "")
        else:
            content = str(data)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            input_tokens=0,  # HF doesn't return token counts
            output_tokens=0,
            latency_ms=0,
            raw_response=data
        )


# =============================================================================
# TRADING ANALYSIS ENGINE
# =============================================================================

@dataclass
class TradeAnalysis:
    """Comprehensive trade analysis from LLM"""
    symbol: str
    strategy: str
    recommendation: str  # "strong_buy", "buy", "hold", "avoid", "strong_avoid"
    confidence: float  # 0-1
    score: int  # -100 to 100

    # Analysis components
    technical_summary: str
    fundamental_summary: str
    options_summary: str
    risk_summary: str
    sentiment_summary: str

    # Reasons
    bullish_factors: List[str]
    bearish_factors: List[str]
    key_risks: List[str]

    # Position sizing
    recommended_contracts: int
    max_contracts: int
    stop_loss_suggestion: str
    profit_target_suggestion: str

    # Metadata
    reasoning: str
    generated_at: datetime = field(default_factory=datetime.now)


class TradingAnalysisEngine:
    """
    LLM-powered trading analysis engine.

    Provides comprehensive trade analysis by synthesizing:
    - Technical analysis
    - Fundamental analysis
    - Options-specific factors
    - Risk assessment
    - Market sentiment

    Usage:
        engine = TradingAnalysisEngine()
        analysis = await engine.analyze_opportunity(setup, context)
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

        # System prompts
        self._analysis_system = self._build_analysis_system_prompt()
        self._adjustment_system = self._build_adjustment_system_prompt()
        self._risk_system = self._build_risk_system_prompt()

    def _build_analysis_system_prompt(self) -> str:
        return """You are an expert options trading analyst with deep expertise in:
- Technical analysis (price action, indicators, patterns)
- Fundamental analysis (earnings, valuation, sector trends)
- Options mechanics (Greeks, IV, probability)
- Risk management (position sizing, portfolio impact)
- Market sentiment (VIX, put/call ratios, news)

Your role is to analyze trading opportunities and provide:
1. A clear recommendation (strong_buy, buy, hold, avoid, strong_avoid)
2. Confidence level (0-1)
3. Numerical score (-100 to +100)
4. Detailed reasoning for your recommendation
5. Key risks and mitigating factors
6. Position sizing suggestions

Always be objective and thorough. Consider both bullish and bearish scenarios.
Format your response as JSON matching the required schema."""

    def _build_adjustment_system_prompt(self) -> str:
        return """You are an expert options position manager specializing in:
- Position adjustments (rolling, spreading, hedging)
- Risk mitigation strategies
- Profit taking decisions
- Loss management

Analyze the current position and market conditions to recommend:
1. Whether to adjust, close, or hold the position
2. Specific adjustment strategy if needed
3. Timing considerations
4. Risk/reward of the adjustment

Format your response as JSON matching the required schema."""

    def _build_risk_system_prompt(self) -> str:
        return """You are a portfolio risk manager specializing in:
- Portfolio Greeks analysis
- VaR and stress testing interpretation
- Position concentration risks
- Correlation and sector exposure

Analyze the portfolio and identify:
1. Key risk exposures
2. Specific hedging recommendations
3. Position adjustments needed
4. Overall risk score and warnings

Format your response as JSON matching the required schema."""

    async def analyze_opportunity(
        self,
        setup: Dict,
        context: Dict,
        portfolio: Optional[Dict] = None
    ) -> TradeAnalysis:
        """
        Perform comprehensive analysis of a trading opportunity.

        Args:
            setup: Strategy setup with legs, strikes, premiums
            context: Market context with IV, price, fundamentals
            portfolio: Optional current portfolio for concentration analysis

        Returns:
            TradeAnalysis with recommendation and reasoning
        """
        # Build analysis prompt
        prompt = self._build_opportunity_prompt(setup, context, portfolio)

        # Call LLM
        response = await self.llm.generate(
            system=self._analysis_system,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Parse response
        return self._parse_analysis_response(response.content, setup)

    def _build_opportunity_prompt(
        self,
        setup: Dict,
        context: Dict,
        portfolio: Optional[Dict]
    ) -> str:
        """Build detailed prompt for opportunity analysis"""

        prompt = f"""Analyze this options trading opportunity:

## Strategy Setup
- Symbol: {setup.get('symbol')}
- Strategy: {setup.get('strategy_name')}
- Underlying Price: ${setup.get('underlying_price', 0):.2f}

### Legs
"""
        for i, leg in enumerate(setup.get('legs', []), 1):
            prompt += f"""
Leg {i}:
- Strike: ${leg.get('strike', 0):.2f}
- Type: {leg.get('option_type', 'unknown')}
- Expiration: {leg.get('expiration', 'unknown')}
- Delta: {leg.get('delta', 0):.3f}
- Quantity: {leg.get('quantity', 0)}
- Premium: ${leg.get('mid_price', leg.get('bid', 0)):.2f}
"""

        prompt += f"""
### P&L Profile
- Max Profit: ${setup.get('max_profit', 0):.2f}
- Max Loss: ${setup.get('max_loss', 0):.2f}
- Break Even: ${setup.get('break_even', 0):.2f}
- Probability of Profit: {setup.get('probability_of_profit', 0):.1%}

### Greeks
- Net Delta: {setup.get('net_delta', 0):.3f}
- Net Theta: ${setup.get('net_theta', 0):.2f}/day
- Net Vega: ${setup.get('net_vega', 0):.2f}

## Market Context
- Current Price: ${context.get('underlying_price', 0):.2f}
- IV Rank: {context.get('iv_rank', 50):.1f}
- IV Percentile: {context.get('iv_percentile', 50):.1f}
- Historical Vol (20d): {context.get('hv_20', 0.20):.1%}
- VIX: {context.get('vix', 15):.1f}
- Sector: {context.get('sector', 'Unknown')}
- Days to Earnings: {context.get('days_to_earnings', 'N/A')}
- Trend: {context.get('trend', 'neutral')}
"""

        if portfolio:
            prompt += f"""
## Current Portfolio
- Total Value: ${portfolio.get('total_value', 0):,.2f}
- Current Delta: {portfolio.get('total_delta', 0):.1f}
- Current Theta: ${portfolio.get('total_theta', 0):.2f}/day
- Exposure to {setup.get('symbol')}: {portfolio.get('symbol_exposure_pct', 0):.1%}
"""

        prompt += """
## Required Output (JSON format)
{
    "recommendation": "strong_buy|buy|hold|avoid|strong_avoid",
    "confidence": 0.0-1.0,
    "score": -100 to 100,
    "technical_summary": "Brief technical analysis",
    "fundamental_summary": "Brief fundamental view",
    "options_summary": "Options-specific analysis",
    "risk_summary": "Key risks",
    "sentiment_summary": "Market sentiment",
    "bullish_factors": ["factor1", "factor2"],
    "bearish_factors": ["factor1", "factor2"],
    "key_risks": ["risk1", "risk2"],
    "recommended_contracts": 1-5,
    "max_contracts": 1-10,
    "stop_loss_suggestion": "description",
    "profit_target_suggestion": "description",
    "reasoning": "Detailed explanation of recommendation"
}
"""
        return prompt

    def _parse_analysis_response(self, content: str, setup: Dict) -> TradeAnalysis:
        """Parse LLM response into TradeAnalysis"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            return TradeAnalysis(
                symbol=setup.get('symbol', ''),
                strategy=setup.get('strategy_name', ''),
                recommendation=data.get('recommendation', 'hold'),
                confidence=float(data.get('confidence', 0.5)),
                score=int(data.get('score', 0)),
                technical_summary=data.get('technical_summary', ''),
                fundamental_summary=data.get('fundamental_summary', ''),
                options_summary=data.get('options_summary', ''),
                risk_summary=data.get('risk_summary', ''),
                sentiment_summary=data.get('sentiment_summary', ''),
                bullish_factors=data.get('bullish_factors', []),
                bearish_factors=data.get('bearish_factors', []),
                key_risks=data.get('key_risks', []),
                recommended_contracts=int(data.get('recommended_contracts', 1)),
                max_contracts=int(data.get('max_contracts', 1)),
                stop_loss_suggestion=data.get('stop_loss_suggestion', ''),
                profit_target_suggestion=data.get('profit_target_suggestion', ''),
                reasoning=data.get('reasoning', content)
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return conservative defaults
            return TradeAnalysis(
                symbol=setup.get('symbol', ''),
                strategy=setup.get('strategy_name', ''),
                recommendation='hold',
                confidence=0.3,
                score=0,
                technical_summary='Analysis failed',
                fundamental_summary='',
                options_summary='',
                risk_summary='Unable to assess risk',
                sentiment_summary='',
                bullish_factors=[],
                bearish_factors=['Analysis parsing failed'],
                key_risks=['Unable to complete analysis'],
                recommended_contracts=1,
                max_contracts=1,
                stop_loss_suggestion='Conservative stop recommended',
                profit_target_suggestion='',
                reasoning=f"Analysis failed: {str(e)}"
            )

    async def analyze_adjustment(
        self,
        position: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Analyze whether a position needs adjustment.

        Returns recommended action and specific strategy.
        """
        prompt = f"""Analyze this option position for potential adjustments:

## Current Position
- Symbol: {position.get('symbol')}
- Type: {position.get('option_type')}
- Strike: ${position.get('strike', 0):.2f}
- Quantity: {position.get('quantity')}
- Entry Price: ${position.get('average_cost', 0):.2f}
- Current Price: ${position.get('current_price', 0):.2f}
- P&L: ${position.get('unrealized_pnl', 0):.2f} ({position.get('unrealized_pnl_pct', 0):.1%})
- Days to Expiration: {position.get('days_to_expiration', 0)}
- Delta: {position.get('delta', 0):.3f}

## Current Market
- Underlying Price: ${market_data.get('underlying_price', 0):.2f}
- IV Rank: {market_data.get('iv_rank', 50):.1f}
- Price Change Today: {market_data.get('price_change_pct', 0):.1%}

Recommend one of:
1. HOLD - Keep position as is
2. CLOSE - Close entire position
3. ROLL - Roll to different strike/expiration
4. SPREAD - Convert to spread
5. HEDGE - Add protective position

Output JSON:
{{
    "action": "hold|close|roll|spread|hedge",
    "urgency": "low|medium|high",
    "reasoning": "explanation",
    "specific_recommendation": "detailed action steps",
    "risk_if_no_action": "potential downside"
}}
"""

        response = await self.llm.generate(
            system=self._adjustment_system,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Failed to parse adjustment response: {e}")

        return {
            "action": "hold",
            "urgency": "low",
            "reasoning": "Unable to analyze",
            "specific_recommendation": "Monitor position",
            "risk_if_no_action": "Unknown"
        }

    async def explain_risk(self, portfolio_risk: Dict) -> str:
        """
        Generate plain-English explanation of portfolio risk.
        """
        prompt = f"""Explain this portfolio risk analysis in plain English:

## Portfolio Greeks
- Total Delta: {portfolio_risk.get('total_delta', 0):.1f}
- Total Gamma: {portfolio_risk.get('total_gamma', 0):.3f}
- Total Theta: ${portfolio_risk.get('total_theta', 0):.2f}/day
- Total Vega: ${portfolio_risk.get('total_vega', 0):.2f}

## Value at Risk
- 95% VaR: ${portfolio_risk.get('var_95', 0):,.2f} ({portfolio_risk.get('var_95_pct', 0):.1%} of portfolio)
- 99% VaR: ${portfolio_risk.get('var_99', 0):,.2f}

## Stress Test Results
{json.dumps(portfolio_risk.get('stress_test_results', {}), indent=2)}

## Violations
{portfolio_risk.get('violations', [])}

Provide:
1. A 2-3 sentence summary of the portfolio's risk exposure
2. The most concerning risk factor
3. One specific actionable recommendation
"""

        response = await self.llm.generate(
            system=self._risk_system,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.content


# =============================================================================
# STRATEGY RECOMMENDATION ENGINE
# =============================================================================

class StrategyRecommendationEngine:
    """
    LLM-powered strategy selection based on market conditions.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    async def recommend_strategies(
        self,
        symbol: str,
        context: Dict,
        account_size: float,
        risk_tolerance: str = "moderate"
    ) -> List[Dict]:
        """
        Recommend optimal strategies for current market conditions.

        Args:
            symbol: Underlying symbol
            context: Market context
            account_size: Account value
            risk_tolerance: "conservative", "moderate", or "aggressive"

        Returns:
            List of recommended strategies with reasoning
        """
        prompt = f"""Recommend options strategies for this situation:

## Symbol: {symbol}
- Current Price: ${context.get('underlying_price', 0):.2f}
- IV Rank: {context.get('iv_rank', 50):.1f}
- VIX: {context.get('vix', 15):.1f}
- Trend: {context.get('trend', 'neutral')}
- Days to Earnings: {context.get('days_to_earnings', 'N/A')}

## Account
- Size: ${account_size:,.2f}
- Risk Tolerance: {risk_tolerance}

Available strategies:
1. Cash-Secured Put (Wheel)
2. Covered Call
3. Iron Condor
4. Credit Spread (Bull Put / Bear Call)
5. Debit Spread (Bull Call / Bear Put)
6. Calendar Spread
7. Diagonal Spread
8. Straddle/Strangle (Long or Short)

Recommend top 3 strategies. For each provide:
- Strategy name
- Why it fits current conditions
- Suggested parameters (delta, DTE, etc.)
- Expected outcome
- Key risks

Output JSON array:
[
    {{
        "strategy": "name",
        "fit_score": 0-100,
        "reasoning": "why this strategy",
        "parameters": {{"delta": 0.3, "dte": 30}},
        "expected_outcome": "description",
        "key_risks": ["risk1"]
    }}
]
"""

        response = await self.llm.generate(
            system="""You are an options strategy specialist. Recommend strategies
that match the current market conditions, IV environment, and risk tolerance.
Prioritize probability of profit and risk-adjusted returns.""",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        try:
            import re
            json_match = re.search(r'\[[\s\S]*\]', response.content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Failed to parse strategy recommendations: {e}")

        return []


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing LLM Engine ===\n")

    async def test_analysis():
        # Mock setup and context (would normally come from real data)
        setup = {
            "symbol": "AAPL",
            "strategy_name": "Cash-Secured Put",
            "underlying_price": 230.50,
            "legs": [{
                "strike": 225,
                "option_type": "put",
                "expiration": "2024-12-20",
                "delta": -0.28,
                "quantity": -1,
                "mid_price": 3.50
            }],
            "max_profit": 350,
            "max_loss": -22150,
            "break_even": 221.50,
            "probability_of_profit": 0.72,
            "net_delta": -0.28,
            "net_theta": 0.12,
            "net_vega": -0.08
        }

        context = {
            "underlying_price": 230.50,
            "iv_rank": 45,
            "iv_percentile": 48,
            "hv_20": 0.22,
            "vix": 15.5,
            "sector": "Technology",
            "days_to_earnings": 45,
            "trend": "bullish"
        }

        print("Testing LLM Client initialization...")
        # Note: This will fail without API key, but shows structure
        try:
            engine = TradingAnalysisEngine()
            print("Engine created (will need API key for actual calls)")

            # Show the prompt that would be sent
            prompt = engine._build_opportunity_prompt(setup, context, None)
            print("\nSample Analysis Prompt:")
            print("-" * 50)
            print(prompt[:1000] + "...")

        except Exception as e:
            print(f"Note: Full test requires API key: {e}")

        print("\n✅ LLM Engine structure validated!")

    asyncio.run(test_analysis())
