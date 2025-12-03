"""
AI Model Client Implementations
Supports GPT-4, Claude, Gemini, and local Llama models
"""

import os
import logging
from typing import Dict, Optional
from abc import ABC, abstractmethod
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModelClient(ABC):
    """Abstract base class for AI model clients"""

    @abstractmethod
    async def analyze_market(self, prompt: str) -> Dict:
        """
        Analyze market and return prediction

        Args:
            prompt: Analysis prompt

        Returns:
            Model response dictionary
        """
        pass


class GPT4Client(BaseModelClient):
    """OpenAI GPT-4 Turbo client"""

    def __init__(self) -> None:
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - GPT-4 unavailable")
        self.model = "gpt-4-turbo-preview"

    async def analyze_market(self, prompt: str) -> Dict:
        """Call GPT-4 API"""

        if not self.api_key:
            raise Exception("OPENAI_API_KEY not configured")

        try:
            # Import here to avoid dependency if not used
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert NFL prediction analyst with deep knowledge "
                                 "of betting markets, statistics, and game dynamics."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more focused analysis
                max_tokens=1000,
                response_format={"type": "json_object"}  # Force JSON output
            )

            return {
                'content': response.choices[0].message.content,
                'model': self.model,
                'usage': {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens
                }
            }

        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            raise


class ClaudeClient(BaseModelClient):
    """Anthropic Claude 3.5 Sonnet client"""

    def __init__(self) -> None:
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set - Claude unavailable")
        self.model = "claude-3-5-sonnet-20241022"

    async def analyze_market(self, prompt: str) -> Dict:
        """Call Claude API"""

        if not self.api_key:
            raise Exception("ANTHROPIC_API_KEY not configured")

        try:
            # Import here to avoid dependency if not used
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            # Add JSON formatting instruction
            json_prompt = prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON matching the schema above."

            response = await client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": json_prompt
                    }
                ]
            )

            return {
                'content': response.content[0].text,
                'model': self.model,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


class GeminiClient(BaseModelClient):
    """Google Gemini Pro client"""

    def __init__(self) -> None:
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set - Gemini unavailable")
        self.model = "gemini-pro"

    async def analyze_market(self, prompt: str) -> Dict:
        """Call Gemini API"""

        if not self.api_key:
            raise Exception("GOOGLE_API_KEY not configured")

        try:
            # Import here to avoid dependency if not used
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)

            # Add JSON formatting instruction
            json_prompt = prompt + "\n\nRespond with valid JSON only."

            # Gemini doesn't have async support yet, so run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    json_prompt,
                    generation_config={
                        'temperature': 0.3,
                        'max_output_tokens': 1000,
                    }
                )
            )

            return {
                'content': response.text,
                'model': self.model,
                'usage': {
                    'input_tokens': 0,  # Gemini doesn't expose token counts
                    'output_tokens': 0
                }
            }

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class GroqClient(BaseModelClient):
    """Groq - FREE Llama 3.1 70B (30 req/min free tier)"""

    def __init__(self) -> None:
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set - Groq unavailable")
        self.model = "llama-3.1-70b-versatile"

    async def analyze_market(self, prompt: str) -> Dict:
        """Call Groq API"""

        if not self.api_key:
            raise Exception("GROQ_API_KEY not configured")

        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self.api_key)

            # Add JSON formatting instruction
            json_prompt = prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON."

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert options trading analyst."
                    },
                    {
                        "role": "user",
                        "content": json_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            return {
                'content': response.choices[0].message.content,
                'model': self.model,
                'usage': {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens
                }
            }

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


class DeepSeekClient(BaseModelClient):
    """DeepSeek - ULTRA CHEAP ($0.14/$0.28 per 1M tokens)"""

    def __init__(self) -> None:
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not set - DeepSeek unavailable")
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com"

    async def analyze_market(self, prompt: str) -> Dict:
        """Call DeepSeek API"""

        if not self.api_key:
            raise Exception("DEEPSEEK_API_KEY not configured")

        try:
            import openai  # DeepSeek uses OpenAI-compatible API

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            # Add JSON formatting instruction
            json_prompt = prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON."

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert options trading analyst."
                    },
                    {
                        "role": "user",
                        "content": json_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            return {
                'content': response.choices[0].message.content,
                'model': self.model,
                'usage': {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens
                }
            }

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise


class LocalModelClient(BaseModelClient):
    """
    Local Llama 3 70B client using Ollama

    Requires Ollama installed with llama3:70b model
    Install: ollama pull llama3:70b
    """

    def __init__(self) -> None:
        self.model = "llama3:70b"
        self.base_url = "http://localhost:11434"

    async def analyze_market(self, prompt: str) -> Dict:
        """Call local Ollama API"""

        try:
            import aiohttp

            # Add JSON formatting instruction
            json_prompt = prompt + "\n\nRespond ONLY with valid JSON."

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": json_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 1000
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Ollama API error: {resp.status}")

                    data = await resp.json()

                    return {
                        'content': data.get('response', ''),
                        'model': self.model,
                        'usage': {
                            'input_tokens': 0,
                            'output_tokens': 0
                        }
                    }

        except Exception as e:
            logger.error(f"Local model error: {e}")
            # If local model fails, return fallback response
            return {
                'content': '''{
                    "predicted_outcome": "no",
                    "confidence": 50,
                    "edge_percentage": 0,
                    "reasoning": "Local model unavailable",
                    "key_factors": []
                }''',
                'model': 'fallback',
                'usage': {'input_tokens': 0, 'output_tokens': 0}
            }


# ============================================================================
# Mock client for testing without API keys
# ============================================================================

class MockModelClient(BaseModelClient):
    """Mock client for testing"""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    async def analyze_market(self, prompt: str) -> Dict:
        """Return mock prediction"""

        # Simulate API delay
        await asyncio.sleep(0.5)

        # Extract market info from prompt for realistic mock
        import random

        outcome = random.choice(['yes', 'no'])
        confidence = random.uniform(60, 85)
        edge = random.uniform(-5, 15)

        return {
            'content': f'''{{
                "predicted_outcome": "{outcome}",
                "confidence": {confidence:.1f},
                "edge_percentage": {edge:.1f},
                "reasoning": "This is a {self.model_name} mock prediction based on market analysis.",
                "key_factors": [
                    "Mock factor 1",
                    "Mock factor 2",
                    "Mock factor 3"
                ],
                "risk_level": "medium",
                "recommended_action": "buy"
            }}''',
            'model': self.model_name,
            'usage': {
                'input_tokens': 2000,
                'output_tokens': 500
            }
        }


# ============================================================================
# Client factory for easy instantiation
# ============================================================================

def get_model_client(model_name: str, mock: bool = False) -> BaseModelClient:
    """
    Get model client by name

    Args:
        model_name: 'gpt4', 'claude', 'gemini', 'groq', 'deepseek', or 'llama3'
        mock: Use mock client for testing

    Returns:
        Model client instance
    """
    if mock:
        return MockModelClient(model_name)

    clients = {
        'gpt4': GPT4Client,
        'claude': ClaudeClient,
        'gemini': GeminiClient,
        'groq': GroqClient,        # FREE - Llama 3.1 70B
        'deepseek': DeepSeekClient,  # ULTRA CHEAP - $0.14/$0.28 per 1M tokens
        'llama3': LocalModelClient
    }

    if model_name not in clients:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(clients.keys())}")

    return clients[model_name]()


# ============================================================================
# Testing
# ============================================================================

async def test_clients():
    """Test all model clients"""

    test_prompt = """
    Analyze this NFL market:
    - Market: Will the Chiefs beat the Bills?
    - Yes Price: 0.48
    - No Price: 0.52

    Respond in JSON format:
    {
        "predicted_outcome": "yes" or "no",
        "confidence": 0-100,
        "edge_percentage": -50 to 50,
        "reasoning": "brief explanation",
        "key_factors": ["factor1", "factor2"]
    }
    """

    # Test with mock clients
    for model_name in ['gpt4', 'claude', 'gemini', 'llama3']:
        print(f"\nTesting {model_name}...")
        try:
            client = get_model_client(model_name, mock=True)
            response = await client.analyze_market(test_prompt)
            print(f"Response: {response['content'][:200]}...")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_clients())
