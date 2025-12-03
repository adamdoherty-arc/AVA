"""
LLM Service
Unified LLM service with multi-provider support, automatic fallback, caching, and cost tracking
"""

import os
import hashlib
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv

from src.services.config import get_service_config, calculate_cost
from src.services.rate_limiter import rate_limit

load_dotenv()


# =============================================================================
# Response Cache
# =============================================================================

class ResponseCache:
    """Simple in-memory cache for LLM responses"""

    def __init__(self, ttl: int = 3600):
        """
        Initialize cache

        Args:
            ttl: Time to live in seconds (default 1 hour)
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self.ttl = ttl

    def _make_key(self, prompt: str, provider: str, model: str, **kwargs) -> str:
        """Create cache key from parameters"""
        key_string = f"{provider}:{model}:{prompt}:{kwargs}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, prompt: str, provider: str, model: str, **kwargs) -> Optional[str]:
        """Get cached response if exists and not expired"""
        key = self._make_key(prompt, provider, model, **kwargs)

        with self._lock:
            if key in self._cache:
                response, timestamp = self._cache[key]

                # Check if expired
                if time.time() - timestamp < self.ttl:
                    logger.debug(f"Cache hit for {provider}/{model}")
                    return response
                else:
                    # Expired, remove
                    del self._cache[key]

        return None

    def set(self, prompt: str, provider: str, model: str, response: str, **kwargs):
        """Cache a response"""
        key = self._make_key(prompt, provider, model, **kwargs)

        with self._lock:
            self._cache[key] = (response, time.time())
            logger.debug(f"Cached response for {provider}/{model}")

    def clear(self) -> None:
        """Clear all cached responses"""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared LLM response cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            now = time.time()
            valid_entries = sum(
                1 for _, timestamp in self._cache.values()
                if now - timestamp < self.ttl
            )

            return {
                'total_entries': len(self._cache),
                'valid_entries': valid_entries,
                'ttl': self.ttl
            }


# =============================================================================
# Usage Tracker
# =============================================================================

class UsageTracker:
    """Track LLM usage and costs per provider"""

    def __init__(self) -> None:
        self._usage: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int, cost: float):
        """Record an API call"""
        with self._lock:
            key = f"{provider}:{model}"

            if key not in self._usage:
                self._usage[key] = {
                    'provider': provider,
                    'model': model,
                    'calls': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_cost': 0.0,
                    'first_call': datetime.now().isoformat(),
                    'last_call': datetime.now().isoformat()
                }

            self._usage[key]['calls'] += 1
            self._usage[key]['input_tokens'] += input_tokens
            self._usage[key]['output_tokens'] += output_tokens
            self._usage[key]['total_cost'] += cost
            self._usage[key]['last_call'] = datetime.now().isoformat()

    def get_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage statistics

        Args:
            provider: Filter by provider (None for all)

        Returns:
            Usage statistics
        """
        with self._lock:
            if provider:
                # Filter by provider
                filtered = {
                    k: v for k, v in self._usage.items()
                    if v['provider'] == provider
                }
                return filtered
            else:
                # Return all
                return self._usage.copy()

    def get_total_cost(self) -> float:
        """Get total cost across all providers"""
        with self._lock:
            return sum(usage['total_cost'] for usage in self._usage.values())

    def reset(self) -> None:
        """Reset all usage statistics"""
        with self._lock:
            self._usage.clear()
            logger.info("Reset LLM usage statistics")


# =============================================================================
# LLM Service
# =============================================================================

class LLMService:
    """
    Unified LLM service with multi-provider support

    Features:
    - Multiple provider support (Claude, DeepSeek, Gemini, etc.)
    - Automatic provider fallback on errors
    - Response caching for identical prompts
    - Cost tracking per provider
    - Rate limiting per provider
    - Thread-safe singleton pattern
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize LLM service (singleton pattern)"""
        if not hasattr(self, '_initialized'):
            self._providers = {}
            self._cache = ResponseCache(ttl=3600)  # 1 hour cache
            self._usage = UsageTracker()
            self._router = None  # Intelligent router (initialized after providers)
            self._initialize_providers()
            self._initialize_router()
            self._initialized = True
            logger.info("LLM service initialized")

    def _initialize_providers(self) -> None:
        """Initialize all available LLM providers"""
        # Import existing LLM manager
        try:
            from src.ai_options_agent.llm_manager import (
                OllamaProvider, GroqProvider, DeepSeekProvider, HuggingFaceProvider,
                GeminiProvider, OpenAIProvider, AnthropicProvider,
                GrokProvider, KimiProvider
            )

            # Ollama (local, always try)
            ollama = OllamaProvider()
            if ollama.is_available():
                self._providers["ollama"] = ollama
                logger.info("✓ Ollama available (local)")

            # Groq (free tier)
            groq = GroqProvider()
            if groq.is_available():
                self._providers["groq"] = groq
                logger.info("✓ Groq available (free tier)")

            # DeepSeek (very cheap)
            deepseek = DeepSeekProvider()
            if deepseek.is_available():
                self._providers["deepseek"] = deepseek
                logger.info("✓ DeepSeek available ($0.14/$0.28 per 1M)")

            # Hugging Face (free tier)
            huggingface = HuggingFaceProvider()
            if huggingface.is_available():
                self._providers["huggingface"] = huggingface
                logger.info("✓ Hugging Face available (free tier: 300 req/hour)")

            # Gemini (Google)
            gemini = GeminiProvider()
            if gemini.is_available():
                self._providers["gemini"] = gemini
                logger.info("✓ Gemini available")

            # OpenAI
            openai = OpenAIProvider()
            if openai.is_available():
                self._providers["openai"] = openai
                logger.info("✓ OpenAI available")

            # Anthropic
            anthropic = AnthropicProvider()
            if anthropic.is_available():
                self._providers["anthropic"] = anthropic
                logger.info("✓ Anthropic available")

            # Grok
            grok = GrokProvider()
            if grok.is_available():
                self._providers["grok"] = grok
                logger.info("✓ Grok available")

            # Kimi
            kimi = KimiProvider()
            if kimi.is_available():
                self._providers["kimi"] = kimi
                logger.info("✓ Kimi available")

            if not self._providers:
                logger.warning("No LLM providers available!")

        except ImportError as e:
            logger.error(f"Failed to import LLM providers: {e}")

    def _initialize_router(self) -> None:
        """Initialize intelligent LLM router for cost optimization"""
        try:
            from src.services.intelligent_llm_router import IntelligentLLMRouter

            available_providers = list(self._providers.keys())
            self._router = IntelligentLLMRouter(available_providers)
            logger.info("✅ Intelligent LLM router initialized for cost optimization")
        except ImportError as e:
            logger.warning(f"Intelligent router not available: {e}")
            self._router = None

    def _get_provider(self, provider_name: str):
        """Get provider instance by name"""
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        return self._providers[provider_name]

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count"""
        # Average: 1 token ≈ 4 characters or 0.75 words
        return int(len(text.split()) * 1.3)

    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text using specified provider or auto-select

        Args:
            prompt: The prompt to send
            provider: Provider to use (auto-selects if None)
            model: Model to use (uses default if None)
            max_tokens: Max tokens to generate
            temperature: Temperature for sampling
            use_cache: Whether to use response cache

        Returns:
            Dict with 'text', 'provider', 'model', 'tokens_used', 'cost', 'cached'
        """
        # Auto-select provider if not specified
        if provider is None or provider not in self._providers:
            # Use intelligent router for cost optimization
            if self._router:
                routing = self._router.get_optimal_provider(prompt)
                provider = routing['provider']
                if not model:  # Only use router's model if not explicitly specified
                    model = routing['model']
                logger.info(f"🎯 Intelligent routing: {routing['routing_reason']} -> {provider}/{model}")
            else:
                # Fallback to simple priority if router not available
                # Priority: Ollama (free) > Groq (free) > HuggingFace (free tier) > DeepSeek (cheap) > Gemini > others
                for fallback in ["ollama", "groq", "huggingface", "deepseek", "gemini"]:
                    if fallback in self._providers:
                        provider = fallback
                        logger.info(f"Auto-selected provider (no router): {fallback}")
                        break

                if provider is None:
                    # Use any available provider
                    if self._providers:
                        provider = list(self._providers.keys())[0]
                    else:
                        raise RuntimeError("No LLM providers available")

        provider_instance = self._get_provider(provider)

        # Update model if specified
        if model:
            provider_instance.model = model

        current_model = provider_instance.model

        # Check cache first
        if use_cache:
            cached_response = self._cache.get(
                prompt, provider, current_model,
                max_tokens=max_tokens, temperature=temperature
            )
            if cached_response:
                return {
                    "text": cached_response,
                    "provider": provider,
                    "model": current_model,
                    "input_tokens": self._estimate_tokens(prompt),
                    "output_tokens": self._estimate_tokens(cached_response),
                    "cost": 0.0,  # No cost for cached response
                    "cached": True
                }

        # Generate response with rate limiting
        @rate_limit(provider, tokens=1)
        def _generate():
            return provider_instance.generate(prompt, max_tokens, temperature)

        try:
            text = _generate()

            # Calculate tokens and cost
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(text)
            cost = calculate_cost(current_model, input_tokens, output_tokens)

            # Record usage
            self._usage.record(provider, current_model, input_tokens, output_tokens, cost)

            # Cache response
            if use_cache and text:
                self._cache.set(
                    prompt, provider, current_model, text,
                    max_tokens=max_tokens, temperature=temperature
                )

            return {
                "text": text,
                "provider": provider,
                "model": current_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "cached": False
            }

        except Exception as e:
            logger.error(f"Error generating with {provider}/{current_model}: {e}")
            raise

    def generate_with_fallback(
        self,
        prompt: str,
        providers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with automatic fallback to other providers on failure

        Args:
            prompt: The prompt to send
            providers: List of providers to try (None = default order)
            **kwargs: Additional arguments to pass to generate()

        Returns:
            Dict with response from first successful provider
        """
        # Default provider order (free/cheap first)
        if providers is None:
            providers = ["ollama", "groq", "huggingface", "deepseek", "gemini", "openai", "anthropic"]

        # Filter to only available providers
        available_providers = [p for p in providers if p in self._providers]

        if not available_providers:
            raise RuntimeError("No providers available for fallback")

        last_error = None

        for provider in available_providers:
            try:
                logger.info(f"Trying provider: {provider}")
                result = self.generate(prompt, provider=provider, **kwargs)
                logger.info(f"Success with provider: {provider}")
                return result

            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
                last_error = e
                continue

        # All providers failed
        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self._providers.keys())

    def get_usage_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage statistics

        Args:
            provider: Filter by provider (None for all)

        Returns:
            Usage statistics including costs
        """
        stats = self._usage.get_stats(provider)
        total_cost = self._usage.get_total_cost()

        return {
            "provider_stats": stats,
            "total_cost": total_cost,
            "available_providers": self.get_available_providers(),
            "cache_stats": self._cache.get_stats()
        }

    def clear_cache(self) -> None:
        """Clear response cache"""
        self._cache.clear()

    def reset_usage(self) -> None:
        """Reset usage statistics"""
        self._usage.reset()

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get intelligent routing statistics and cost savings

        Returns:
            Routing statistics including tier distribution and savings
        """
        if not self._router:
            return {"intelligent_routing": False, "message": "Router not initialized"}

        # Calculate queries by tier from usage stats
        usage_stats = self._usage.get_stats()

        # Map providers to tiers
        provider_to_tier = {
            "ollama": "free",
            "groq": "free",
            "huggingface": "free",
            "deepseek": "cheap",
            "gemini": "cheap",
            "openai": "standard",
            "anthropic": "premium"
        }

        queries_by_tier = {"free": 0, "cheap": 0, "standard": 0, "premium": 0}

        for key, stats in usage_stats.items():
            provider = stats['provider']
            tier = provider_to_tier.get(provider, "standard")
            queries_by_tier[tier] += stats['calls']

        # Get savings report
        savings = self._router.get_savings_report(queries_by_tier)

        return {
            "intelligent_routing": True,
            "total_queries": savings['total_queries'],
            "actual_cost": savings['actual_cost'],
            "premium_cost": savings['premium_cost'],
            "savings": savings['savings'],
            "savings_percent": savings['savings_percent'],
            "cost_reduction": savings['cost_reduction'],
            "queries_by_tier": queries_by_tier,
            "free_tier_percentage": round(
                (queries_by_tier['free'] / savings['total_queries'] * 100)
                if savings['total_queries'] > 0 else 0, 1
            )
        }

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and health status"""
        return {
            "available_providers": self.get_available_providers(),
            "total_providers": len(self._providers),
            "cache_enabled": True,
            "cache_ttl": self._cache.ttl,
            "total_cost": self._usage.get_total_cost(),
            "intelligent_routing_enabled": self._router is not None
        }


# =============================================================================
# Singleton Access
# =============================================================================

_llm_service = None
_service_lock = threading.Lock()


def get_llm_service() -> LLMService:
    """
    Get the global LLM service instance

    Returns:
        Singleton LLMService instance
    """
    global _llm_service

    if _llm_service is None:
        with _service_lock:
            if _llm_service is None:
                _llm_service = LLMService()

    return _llm_service


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing LLM Service")
    print("=" * 60)

    service = get_llm_service()

    # Test 1: Service info
    print("\nTest 1: Service Info")
    info = service.get_service_info()
    print(f"  Available providers: {', '.join(info['available_providers'])}")
    print(f"  Cache enabled: {info['cache_enabled']}")
    print(f"  Total cost: ${info['total_cost']:.4f}")

    # Test 2: Generate with auto-select
    print("\nTest 2: Generate with Auto-Select Provider")
    test_prompt = "What is the capital of France? Answer in one word."

    try:
        result = service.generate(test_prompt, max_tokens=50)
        print(f"  Provider: {result['provider']}")
        print(f"  Model: {result['model']}")
        print(f"  Response: {result['text'][:100]}")
        print(f"  Cost: ${result['cost']:.6f}")
        print(f"  Cached: {result['cached']}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 3: Cache hit
    print("\nTest 3: Cache Hit (same prompt)")
    try:
        result = service.generate(test_prompt, max_tokens=50)
        print(f"  Cached: {result['cached']}")
        print(f"  Cost: ${result['cost']:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 4: Fallback
    print("\nTest 4: Generate with Fallback")
    try:
        result = service.generate_with_fallback(
            "Explain options trading in 10 words.",
            max_tokens=50
        )
        print(f"  Provider: {result['provider']}")
        print(f"  Response: {result['text'][:100]}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 5: Usage stats
    print("\nTest 5: Usage Statistics")
    stats = service.get_usage_stats()
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    print(f"  Providers used: {len(stats['provider_stats'])}")
    for key, usage in stats['provider_stats'].items():
        print(f"    {usage['provider']}/{usage['model']}: {usage['calls']} calls, ${usage['total_cost']:.4f}")

    # Test 6: Cache stats
    print("\nTest 6: Cache Statistics")
    cache_stats = stats['cache_stats']
    print(f"  Total entries: {cache_stats['total_entries']}")
    print(f"  Valid entries: {cache_stats['valid_entries']}")

    print("\n" + "=" * 60)
    print("LLM service tests complete!")
