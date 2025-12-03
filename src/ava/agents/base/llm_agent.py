"""
LLM-Integrated Agent Base Class
===============================

Modern agent base class with:
- Automatic Claude/GPT integration
- Response caching
- Rate limiting
- Circuit breaker pattern
- Structured outputs with Pydantic
- Async execution with retry

All agents should inherit from this class for AI-powered analysis.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import logging
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Dict, Any, Optional, List, Type, TypeVar, Generic,
    Callable, Union
)
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel

from src.ava.core.llm_engine import LLMClient, LLMProvider, LLMResponse
from src.ava.core.cache import LRUCache, async_cached
from src.ava.core.async_utils import (
    RateLimiter, CircuitBreaker, CircuitBreakerConfig,
    retry_with_backoff, RetryConfig
)
from src.ava.core.errors import (
    AVAError, APIError, RateLimitError,
    get_error_handler
)
from src.ava.core.config import get_config

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


# =============================================================================
# AGENT OUTPUT MODELS
# =============================================================================

class AgentConfidence(str, Enum):
    """Agent confidence levels"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class AgentOutputBase(BaseModel):
    """Base model for all agent outputs"""
    agent_name: str
    confidence: AgentConfidence = AgentConfidence.MEDIUM
    reasoning: str = ""
    timestamp: datetime = None
    cached: bool = False
    latency_ms: float = 0

    def __init__(self, **data):
        if data.get('timestamp') is None:
            data['timestamp'] = datetime.now()
        super().__init__(**data)


# =============================================================================
# AGENT STATE
# =============================================================================

@dataclass
class AgentExecutionContext:
    """Context for agent execution"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    portfolio_value: float = 100000.0
    risk_tolerance: str = "moderate"
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# LLM AGENT BASE CLASS
# =============================================================================

class LLMAgent(ABC, Generic[T]):
    """
    Base class for all LLM-powered agents.

    Provides:
    - Automatic LLM client management
    - Response caching
    - Rate limiting
    - Error handling with retry
    - Structured output parsing

    Usage:
        class MyAgent(LLMAgent[MyOutput]):
            name = "my_agent"
            output_model = MyOutput

            def build_prompt(self, input_data: Dict) -> str:
                return f"Analyze: {input_data}"

        agent = MyAgent()
        result = await agent.execute({"symbol": "AAPL"})
    """

    # Override in subclass
    name: str = "base_agent"
    description: str = "Base LLM agent"
    output_model: Type[T] = None

    # LLM settings (can override in subclass)
    system_prompt: str = "You are a helpful trading assistant."
    temperature: float = 0.3
    max_tokens: int = 4096

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 300,
        rate_limit: Optional[int] = None,
        use_circuit_breaker: bool = True
    ):
        # Get config
        config = get_config()

        # LLM client - use configured provider (defaults to Ollama for FREE)
        provider_map = {
            "ollama": LLMProvider.OLLAMA,
            "groq": LLMProvider.GROQ,
            "huggingface": LLMProvider.HUGGINGFACE,
            "anthropic": LLMProvider.ANTHROPIC,
            "openai": LLMProvider.OPENAI,
        }
        provider = provider_map.get(
            config.ai.provider.lower(),
            LLMProvider.OLLAMA  # Default to FREE local
        )

        self.llm = llm_client or LLMClient(
            provider=provider,
            model=config.ai.default_model,
            max_retries=config.ai.max_retries,
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl
        )

        # Response cache
        self.cache_enabled = cache_enabled
        self._cache = LRUCache(max_size=100, default_ttl=cache_ttl)

        # Rate limiter
        self._rate_limiter = RateLimiter(
            rate=rate_limit or 10,
            per=1.0
        ) if rate_limit else None

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            name=self.name,
            config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=2
            )
        ) if use_circuit_breaker else None

        # Error handler
        self._error_handler = get_error_handler()

        # Execution stats
        self._stats = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "cache_hits": 0,
            "total_latency_ms": 0
        }

    # =========================================================================
    # ABSTRACT METHODS (Override in subclass)
    # =========================================================================

    @abstractmethod
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        Build the prompt for the LLM.

        Args:
            input_data: Input data for analysis

        Returns:
            Formatted prompt string
        """
        pass

    def build_system_prompt(self, context: Optional[AgentExecutionContext] = None) -> str:
        """
        Build the system prompt. Override for custom system prompts.

        Args:
            context: Execution context

        Returns:
            System prompt string
        """
        return self.system_prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> T:
        """
        Parse LLM response into output model. Override for custom parsing.

        Args:
            response: Raw LLM response text
            input_data: Original input data

        Returns:
            Parsed output model instance
        """
        if self.output_model is None:
            return response

        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                return self.output_model(**data)
        except Exception as e:
            logger.warning(f"Failed to parse structured output: {e}")

        # Fallback: create minimal output
        return self.output_model(
            agent_name=self.name,
            confidence=AgentConfidence.LOW,
            reasoning=response
        )

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[AgentExecutionContext] = None
    ) -> T:
        """
        Execute the agent analysis.

        Args:
            input_data: Input data for analysis
            context: Optional execution context

        Returns:
            Parsed output model
        """
        start_time = datetime.now()
        self._stats["executions"] += 1

        try:
            # Check cache
            cache_key = self._build_cache_key(input_data)
            if self.cache_enabled:
                cached_result = self._cache.get(cache_key)
                if cached_result is not None:
                    self._stats["cache_hits"] += 1
                    cached_result.cached = True
                    return cached_result

            # Rate limiting
            if self._rate_limiter:
                await self._rate_limiter.acquire()

            # Build prompts
            system_prompt = self.build_system_prompt(context)
            user_prompt = self.build_prompt(input_data)

            # Add output format instructions
            if self.output_model:
                user_prompt += self._build_output_instructions()

            # Execute with circuit breaker
            if self._circuit_breaker:
                response = await self._circuit_breaker.call(
                    lambda: self._call_llm(system_prompt, user_prompt)
                )
            else:
                response = await self._call_llm(system_prompt, user_prompt)

            # Parse response
            result = self.parse_response(response.content, input_data)

            # Add metadata
            latency = (datetime.now() - start_time).total_seconds() * 1000
            if hasattr(result, 'latency_ms'):
                result.latency_ms = latency
            if hasattr(result, 'cached'):
                result.cached = False

            # Cache result
            if self.cache_enabled:
                self._cache.set(cache_key, result)

            # Update stats
            self._stats["successes"] += 1
            self._stats["total_latency_ms"] += latency

            return result

        except Exception as e:
            self._stats["failures"] += 1
            self._error_handler.handle_error(e)
            raise

    async def _call_llm(self, system: str, user: str) -> LLMResponse:
        """Call LLM with retry"""
        return await self.llm.generate(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def _build_cache_key(self, input_data: Dict) -> str:
        """Build cache key from input"""
        import hashlib
        content = json.dumps({
            "agent": self.name,
            "input": input_data
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _build_output_instructions(self) -> str:
        """Build JSON output format instructions"""
        if self.output_model is None:
            return ""

        # Get model schema
        schema = self.output_model.model_json_schema()

        return f"""

## Required Output Format (JSON)
Respond with a valid JSON object matching this schema:
```json
{json.dumps(schema.get('properties', {}), indent=2)}
```

Important:
- Include ALL required fields
- Use the exact field names shown
- Ensure valid JSON syntax
"""

    # =========================================================================
    # TOOL INTERFACE
    # =========================================================================

    def as_tool(self) -> Dict[str, Any]:
        """
        Return agent as a tool definition for function calling.
        """
        return {
            "name": f"{self.name}_analyze",
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input_data": {
                        "type": "object",
                        "description": "Input data for analysis"
                    }
                },
                "required": ["input_data"]
            }
        }

    async def invoke_as_tool(self, **kwargs) -> str:
        """Invoke agent as a tool and return string result"""
        try:
            result = await self.execute(kwargs)
            if hasattr(result, 'model_dump'):
                return json.dumps(result.model_dump(), default=str)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    # =========================================================================
    # STATS & HEALTH
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = self._stats["executions"]
        return {
            "agent_name": self.name,
            "total_executions": total,
            "successes": self._stats["successes"],
            "failures": self._stats["failures"],
            "cache_hits": self._stats["cache_hits"],
            "success_rate": self._stats["successes"] / total if total > 0 else 0,
            "cache_hit_rate": self._stats["cache_hits"] / total if total > 0 else 0,
            "avg_latency_ms": self._stats["total_latency_ms"] / total if total > 0 else 0,
            "circuit_breaker_state": self._circuit_breaker.state.value if self._circuit_breaker else "disabled"
        }

    def reset_stats(self):
        """Reset execution statistics"""
        self._stats = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "cache_hits": 0,
            "total_latency_ms": 0
        }


# =============================================================================
# MULTI-AGENT EXECUTOR
# =============================================================================

class MultiAgentExecutor:
    """
    Execute multiple agents in parallel with result aggregation.

    Usage:
        executor = MultiAgentExecutor([agent1, agent2, agent3])
        results = await executor.execute_all(input_data)
    """

    def __init__(
        self,
        agents: List[LLMAgent],
        max_concurrency: int = 5
    ):
        self.agents = {agent.name: agent for agent in agents}
        self.max_concurrency = max_concurrency

    async def execute_all(
        self,
        input_data: Dict[str, Any],
        context: Optional[AgentExecutionContext] = None
    ) -> Dict[str, Any]:
        """Execute all agents in parallel"""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_agent(agent: LLMAgent):
            async with semaphore:
                try:
                    result = await agent.execute(input_data, context)
                    return agent.name, result
                except Exception as e:
                    logger.error(f"Agent {agent.name} failed: {e}")
                    return agent.name, {"error": str(e)}

        tasks = [run_agent(agent) for agent in self.agents.values()]
        results = await asyncio.gather(*tasks)

        return dict(results)

    async def execute_selected(
        self,
        agent_names: List[str],
        input_data: Dict[str, Any],
        context: Optional[AgentExecutionContext] = None
    ) -> Dict[str, Any]:
        """Execute selected agents"""
        selected = [
            self.agents[name] for name in agent_names
            if name in self.agents
        ]

        temp_executor = MultiAgentExecutor(selected, self.max_concurrency)
        return await temp_executor.execute_all(input_data, context)


# =============================================================================
# AGENT CHAIN
# =============================================================================

class AgentChain:
    """
    Chain agents together where output of one feeds into the next.

    Usage:
        chain = AgentChain([
            research_agent,
            analysis_agent,
            recommendation_agent
        ])
        final_result = await chain.execute(initial_input)
    """

    def __init__(self, agents: List[LLMAgent]):
        self.agents = agents

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[AgentExecutionContext] = None
    ) -> Dict[str, Any]:
        """Execute chain sequentially"""
        current_data = input_data
        results = {}

        for agent in self.agents:
            try:
                result = await agent.execute(current_data, context)

                # Store result
                results[agent.name] = result

                # Merge result into input for next agent
                if hasattr(result, 'model_dump'):
                    current_data = {**current_data, **result.model_dump()}
                elif isinstance(result, dict):
                    current_data = {**current_data, **result}

            except Exception as e:
                logger.error(f"Chain failed at {agent.name}: {e}")
                results[agent.name] = {"error": str(e)}
                break

        return {
            "chain_results": results,
            "final_output": current_data
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing LLM Agent Base ===\n")

    # Define test output model
    class TestOutput(AgentOutputBase):
        analysis: str = ""
        score: int = 0
        recommendations: List[str] = []

    # Define test agent
    class TestAgent(LLMAgent[TestOutput]):
        name = "test_agent"
        description = "Test agent for demonstration"
        output_model = TestOutput
        system_prompt = "You are a test agent. Always respond with valid JSON."

        def build_prompt(self, input_data: Dict) -> str:
            return f"""
Analyze this test input: {input_data}

Provide:
1. A brief analysis
2. A score from 1-100
3. 2-3 recommendations
"""

    async def test_agent():
        print("1. Creating test agent...")
        agent = TestAgent(cache_enabled=True)

        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model}")

        print("\n2. Getting tool definition...")
        tool = agent.as_tool()
        print(f"   Tool: {tool['name']}")

        print("\n3. Testing execution (requires API key)...")
        try:
            result = await agent.execute({"symbol": "AAPL", "action": "analyze"})
            print(f"   Result type: {type(result)}")
            print(f"   Confidence: {result.confidence}")
        except Exception as e:
            print(f"   Note: Execution requires API key: {type(e).__name__}")

        print("\n4. Checking stats...")
        stats = agent.get_stats()
        print(f"   Stats: {stats}")

        print("\n✅ LLM Agent base tests complete!")

    asyncio.run(test_agent())
