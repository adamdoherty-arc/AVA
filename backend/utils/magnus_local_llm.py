"""
Magnus Local LLM Service
=========================

Unified local LLM service for Magnus Trading Platform
Optimized for NVIDIA RTX 4090 (24GB VRAM)

Primary Model: Qwen 2.5 32B (Q4_K_M) - Balanced performance
Secondary Model: Llama 3.3 70B (Q4_K_M) - Complex analysis
Fast Model: Qwen 2.5 14B (Q4_K_M) - Quick responses

Author: Magnus AI Team
Created: 2025-01-20
"""

import os
import logging
import time
from typing import Optional, Dict, List, Any, Literal
from dataclasses import dataclass
from enum import Enum
import requests
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for model selection"""
    FAST = "fast"          # Quick queries, chat, simple analysis
    BALANCED = "balanced"  # Most trading analysis, research
    COMPLEX = "complex"    # Deep multi-step analysis, research reports
    CODING = "coding"      # Code generation, debugging, refactoring


class ModelTier(Enum):
    """Model tiers with their specifications"""
    FAST = "qwen2.5:14b-instruct-q4_K_M"
    BALANCED = "qwen2.5:32b-instruct-q4_K_M"
    COMPLEX = "llama3.3:70b-instruct-q4_K_M"
    CODING = "qwen2.5-coder:32b"  # Specialized coding model


@dataclass
class ModelSpecs:
    """Model specifications and performance characteristics"""
    name: str
    tier: ModelTier
    vram_gb: float
    tokens_per_second: int
    context_window: int
    use_cases: List[str]


class MagnusLocalLLM:
    """
    Unified local LLM service for Magnus Trading Platform

    Features:
    - Automatic model selection based on task complexity
    - Intelligent caching and memory management
    - Fallback mechanisms for reliability
    - Performance monitoring and optimization
    - Trading-specific prompt engineering
    """

    # Model configurations
    MODELS = {
        TaskComplexity.FAST: ModelSpecs(
            name="Qwen 2.5 14B",
            tier=ModelTier.FAST,
            vram_gb=9.0,
            tokens_per_second=90,
            context_window=32768,
            use_cases=["chat", "quick_query", "price_check", "simple_analysis"]
        ),
        TaskComplexity.BALANCED: ModelSpecs(
            name="Qwen 2.5 32B",
            tier=ModelTier.BALANCED,
            vram_gb=20.0,
            tokens_per_second=45,
            context_window=32768,
            use_cases=["trade_analysis", "options_strategy", "risk_assessment", "market_research"]
        ),
        TaskComplexity.COMPLEX: ModelSpecs(
            name="Llama 3.3 70B",
            tier=ModelTier.COMPLEX,
            vram_gb=14.0,  # + 25GB RAM for hybrid deployment
            tokens_per_second=10,
            context_window=131072,
            use_cases=["deep_research", "complex_modeling", "multi_step_analysis"]
        ),
        TaskComplexity.CODING: ModelSpecs(
            name="Qwen 2.5 Coder 32B",
            tier=ModelTier.CODING,
            vram_gb=20.0,
            tokens_per_second=45,
            context_window=32768,
            use_cases=["code_generation", "code_review", "debugging", "refactoring", "documentation"]
        )
    }

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        default_complexity: TaskComplexity = TaskComplexity.BALANCED,
        enable_caching: bool = True,
        enable_fallback: bool = True,
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        """
        Initialize Magnus Local LLM service

        Args:
            ollama_host: Ollama server URL
            default_complexity: Default task complexity
            enable_caching: Enable response caching
            enable_fallback: Enable model fallback on failure
            temperature: LLM temperature (0.0-1.0)
            max_retries: Maximum retry attempts
        """
        self.ollama_host = ollama_host
        self.default_complexity = default_complexity
        self.enable_caching = enable_caching
        self.enable_fallback = enable_fallback
        self.temperature = temperature
        self.max_retries = max_retries

        # Cache for responses
        self._cache: Dict[str, Any] = {}

        # Performance metrics
        self._metrics = {
            "requests": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0
        }

        # Trading-specific system prompts
        self.trading_system_prompt = """You are AVA, an advanced AI trading assistant for the Magnus Trading Platform.

Your expertise includes:
- Options trading (cash-secured puts, covered calls, spreads)
- Technical analysis and chart patterns
- Fundamental analysis and company valuation
- Market sentiment and news analysis
- Risk management and position sizing
- Trading psychology and discipline

Guidelines:
1. Provide data-driven, objective analysis
2. Always consider risk management
3. Be clear about assumptions and limitations
4. Use precise financial terminology
5. Cite specific metrics and indicators
6. Consider both bullish and bearish scenarios

Current trading focus: Wheel strategy (CSP → assignment → covered calls)
Risk tolerance: Conservative to moderate
Preferred position size: Single options contracts
Strategy: Income generation through premium collection"""

    def _get_ollama_client(self, complexity: TaskComplexity) -> OllamaLLM:
        """Get Ollama client for specified complexity level"""
        model = self.MODELS[complexity].tier.value
        return OllamaLLM(
            base_url=self.ollama_host,
            model=model,
            temperature=self.temperature,
            keep_alive="30m"  # Keep model loaded for 30 minutes
        )

    def _check_model_availability(self, model_name: str) -> bool:
        """Check if model is available in Ollama"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name") == model_name for m in models)
            return False
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    def _generate_cache_key(self, prompt: str, complexity: TaskComplexity) -> str:
        """Generate cache key for prompt"""
        import hashlib
        key_str = f"{prompt}:{complexity.value}:{self.temperature}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def query(
        self,
        prompt: str,
        complexity: Optional[TaskComplexity] = None,
        system_prompt: Optional[str] = None,
        use_trading_context: bool = True,
        max_tokens: int = 4000,
        stream: bool = False
    ) -> str:
        """
        Query the local LLM

        Args:
            prompt: User prompt/question
            complexity: Task complexity (auto-selected if None)
            system_prompt: Custom system prompt
            use_trading_context: Use trading-specific system prompt
            max_tokens: Maximum tokens in response
            stream: Stream response (for UI)

        Returns:
            LLM response text
        """
        start_time = time.time()
        complexity = complexity or self.default_complexity

        # Check cache
        if self.enable_caching:
            cache_key = self._generate_cache_key(prompt, complexity)
            if cache_key in self._cache:
                logger.info(f"Cache hit for prompt (length: {len(prompt)})")
                self._metrics["cache_hits"] += 1
                return self._cache[cache_key]

        # Prepare system prompt
        if system_prompt is None and use_trading_context:
            system_prompt = self.trading_system_prompt

        # Build full prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser Query: {prompt}"
        else:
            full_prompt = prompt

        # Attempt query with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Querying {self.MODELS[complexity].name} (attempt {attempt + 1}/{self.max_retries})")

                # Get client
                llm = self._get_ollama_client(complexity)

                # Generate response
                response = llm.invoke(full_prompt)

                # Update metrics
                latency_ms = (time.time() - start_time) * 1000
                self._metrics["requests"] += 1
                self._metrics["avg_latency_ms"] = (
                    (self._metrics["avg_latency_ms"] * (self._metrics["requests"] - 1) + latency_ms)
                    / self._metrics["requests"]
                )

                # Cache response
                if self.enable_caching:
                    self._cache[cache_key] = response

                logger.info(f"Response generated in {latency_ms:.0f}ms")
                return response

            except Exception as e:
                logger.error(f"Error querying LLM (attempt {attempt + 1}): {e}")
                self._metrics["errors"] += 1

                # Try fallback to simpler model
                if self.enable_fallback and attempt < self.max_retries - 1:
                    if complexity == TaskComplexity.COMPLEX:
                        complexity = TaskComplexity.BALANCED
                        logger.info("Falling back to BALANCED complexity")
                    elif complexity == TaskComplexity.BALANCED:
                        complexity = TaskComplexity.FAST
                        logger.info("Falling back to FAST complexity")

        raise Exception(f"Failed to get LLM response after {self.max_retries} attempts")

    def analyze_trade(
        self,
        symbol: str,
        analysis_type: Literal["technical", "fundamental", "sentiment", "options"],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Specialized method for trade analysis

        Args:
            symbol: Stock symbol
            analysis_type: Type of analysis
            context: Additional context (price, volume, etc.)

        Returns:
            Analysis result
        """
        # Build analysis prompt
        prompt_templates = {
            "technical": """Analyze {symbol} from a technical analysis perspective.

Current Data:
{context}

Provide:
1. Key support/resistance levels
2. Trend direction and strength
3. Important chart patterns
4. Volume analysis
5. Key indicators (RSI, MACD, Moving Averages)
6. Trading recommendation""",

            "fundamental": """Analyze {symbol} from a fundamental perspective.

Company Data:
{context}

Provide:
1. Business model assessment
2. Revenue and earnings trends
3. Valuation metrics (P/E, P/S, etc.)
4. Competitive position
5. Growth prospects
6. Investment recommendation""",

            "sentiment": """Analyze sentiment for {symbol}.

Recent Data:
{context}

Provide:
1. Overall market sentiment (bullish/bearish/neutral)
2. News sentiment analysis
3. Social media trends
4. Insider trading activity
5. Analyst ratings summary
6. Sentiment-based outlook""",

            "options": """Analyze options strategy for {symbol}.

Options Data:
{context}

Provide:
1. Implied volatility analysis
2. Options flow insights
3. Recommended strikes and expirations
4. Strategy suggestions (CSP, CC, spreads)
5. Risk/reward assessment
6. Execution recommendations"""
        }

        template = prompt_templates.get(analysis_type, "")
        context_str = str(context) if context else "Not provided"
        prompt = templatef-string

        # Use BALANCED complexity for most trading analysis
        return self.query(
            prompt=prompt,
            complexity=TaskComplexity.BALANCED,
            use_trading_context=True
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self._metrics,
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self._metrics["cache_hits"] / self._metrics["requests"] * 100
                if self._metrics["requests"] > 0 else 0
            )
        }

    def clear_cache(self) -> None:
        """Clear response cache"""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_model_info(self, complexity: TaskComplexity) -> ModelSpecs:
        """Get model specifications"""
        return self.MODELS[complexity]

    @classmethod
    def install_instructions(cls) -> str:
        """Return Ollama installation instructions"""
        return """
# Ollama Installation Instructions

## Windows Installation

1. Download Ollama for Windows:
   https://ollama.ai/download/windows

2. Install Ollama (double-click installer)

3. Open PowerShell and verify installation:
   ollama --version

4. Pull required models:
   ollama pull qwen2.5:32b-instruct-q4_K_M
   ollama pull qwen2.5:14b-instruct-q4_K_M
   ollama pull llama3.3:70b-instruct-q4_K_M

5. Verify models are ready:
   ollama list

## Model Sizes
- Qwen 2.5 14B: ~9GB
- Qwen 2.5 32B: ~20GB
- Llama 3.3 70B: ~40GB

Total disk space needed: ~70GB

## Testing
Run in PowerShell:
ollama run qwen2.5:32b-instruct-q4_K_M "Explain cash-secured puts"

## Troubleshooting
- Ensure NVIDIA drivers are up to date
- Verify CUDA is working: nvidia-smi
- Check Ollama service: Get-Service Ollama
"""


# Singleton instance
_magnus_llm_instance: Optional[MagnusLocalLLM] = None


def get_magnus_llm() -> MagnusLocalLLM:
    """Get singleton instance of Magnus Local LLM"""
    global _magnus_llm_instance
    if _magnus_llm_instance is None:
        _magnus_llm_instance = MagnusLocalLLM()
        logger.info("Initialized Magnus Local LLM service")
    return _magnus_llm_instance


if __name__ == "__main__":
    # Test the service
    print("Magnus Local LLM Service")
    print("=" * 60)
    print(MagnusLocalLLM.install_instructions())

    # Try to initialize
    try:
        llm = get_magnus_llm()
        print("\n✓ Service initialized successfully")
        print(f"\nMetrics: {llm.get_metrics()}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure Ollama is installed and running!")
