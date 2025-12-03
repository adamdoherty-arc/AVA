"""
LLM Manager - Multi-Provider AI Model Support
Supports 10+ LLM providers with unified interface

Free/Open Source:
- Ollama (local, 100% free)
- Groq (cloud, free tier, ultra-fast)

Low Cost:
- DeepSeek ($0.14/$0.28 per 1M tokens)
- Gemini Flash (very cheap)
- Kimi/Moonshot

Premium:
- OpenAI GPT-4o
- Anthropic Claude 3.5
- Gemini Pro
- Grok (xAI)
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class LLMProvider:
    """Base class for LLM providers"""

    def __init__(self, name: str, model: str, api_key: Optional[str] = None):
        self.name = name
        self.model = model
        self.api_key = api_key

    def is_available(self) -> bool:
        """Check if this provider is available (API key exists or local model)"""
        return self.api_key is not None and self.api_key != ""

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response from LLM"""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Ollama - Local, completely free, no API key needed"""

    def __init__(self, model: str = "llama3.1"):
        super().__init__("Ollama", model, api_key="local")
        self.base_url = "http://localhost:11434"

    def is_available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Ollama local API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return ""


class GroqProvider(LLMProvider):
    """Groq - Cloud, free tier, ultra-fast inference"""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        super().__init__("Groq", model, api_key)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Groq API (OpenAI-compatible)"""
        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                api_key=self.api_key,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = llm.invoke(prompt)
            return response.content

        except ImportError:
            # Fallback to direct API call
            logger.warning("langchain-groq not installed, using direct API")
            return self._generate_direct(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return ""

    def _generate_direct(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Direct API call fallback"""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"Groq API error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Groq direct API error: {e}")
            return ""


class DeepSeekProvider(LLMProvider):
    """DeepSeek - Very cheap Chinese model, excellent quality"""

    def __init__(self, model: str = "deepseek-chat"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        super().__init__("DeepSeek", model, api_key)
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using DeepSeek API (OpenAI-compatible)"""
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = llm.invoke(prompt)
            return response.content

        except ImportError:
            # Fallback to direct API call
            logger.warning("langchain-openai not installed, using direct API")
            return self._generate_direct(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"DeepSeek LangChain error: {e}, trying direct API")
            return self._generate_direct(prompt, max_tokens, temperature)

    def _generate_direct(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Direct API call fallback (OpenAI-compatible)"""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return ""

        except Exception as e:
            logger.error(f"DeepSeek direct API error: {e}")
            return ""


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference API - Free tier available"""

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        super().__init__("HuggingFace", model, api_key)
        self.base_url = f"https://api-inference.huggingface.co/models/{model}"

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Hugging Face Inference API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }

            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                return str(result)
            elif response.status_code == 503:
                # Model is loading, wait and retry once
                logger.warning("Hugging Face model loading, waiting 20s for model to load...")
                import time
                time.sleep(20)

                # Retry once
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                    return str(result)
                else:
                    logger.error(f"Hugging Face API error after retry: {response.status_code} - {response.text}")
                    return ""
            else:
                logger.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                return ""

        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
            return ""


class GeminiProvider(LLMProvider):
    """Google Gemini - Fast, cheap, high quality"""

    def __init__(self, model: str = "gemini-2.5-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        super().__init__("Gemini", model, api_key)

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Google Gemini"""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            # Configure model
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config
            )

            response = model.generate_content(prompt)

            if not response:
                return ""

            # Try to get text - handle both single and multi-part responses
            try:
                # Simple accessor for single-part responses
                return response.text
            except ValueError:
                # Multi-part response - extract from parts
                if response.candidates and len(response.candidates) > 0:
                    parts = response.candidates[0].content.parts
                    text_parts = [p.text for p in parts if hasattr(p, 'text')]
                    return ''.join(text_parts)
                return ""

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return ""


class OpenAIProvider(LLMProvider):
    """OpenAI GPT - Industry standard, high quality"""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        super().__init__("OpenAI", model, api_key)

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using OpenAI API"""
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return ""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude - Excellent reasoning, long context"""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        super().__init__("Anthropic", model, api_key)

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Anthropic Claude"""
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                api_key=self.api_key,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return ""


class GrokProvider(LLMProvider):
    """xAI Grok - Elon's AI, real-time X/Twitter integration"""

    def __init__(self, model: str = "grok-beta"):
        api_key = os.getenv("GROK_API_KEY")
        super().__init__("Grok", model, api_key)
        self.base_url = "https://api.x.ai/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Grok API (OpenAI-compatible)"""
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Grok generation error: {e}")
            return ""


class KimiProvider(LLMProvider):
    """Kimi (Moonshot AI) - Chinese model, long context"""

    def __init__(self, model: str = "moonshot-v1-8k"):
        api_key = os.getenv("KIMI_API_KEY")
        super().__init__("Kimi", model, api_key)
        self.base_url = "https://api.moonshot.cn/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Kimi API (OpenAI-compatible)"""
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Kimi generation error: {e}")
            return ""


class LLMManager:
    """
    Unified LLM Manager
    Handles multiple providers with automatic fallback
    """

    # All supported providers with metadata
    PROVIDERS = {
        "ollama": {
            "name": "Ollama (Local)",
            "description": "Free local models (Llama 3.1, Mistral, Phi)",
            "cost": "Free",
            "speed": "Medium",
            "quality": "Good",
            "models": ["llama3.1", "llama3.1:70b", "mistral", "phi3", "codellama"]
        },
        "groq": {
            "name": "Groq Cloud",
            "description": "Ultra-fast free inference",
            "cost": "Free tier",
            "speed": "Very Fast",
            "quality": "Excellent",
            "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
        },
        "deepseek": {
            "name": "DeepSeek",
            "description": "Very cheap Chinese model",
            "cost": "$0.14/$0.28 per 1M tokens",
            "speed": "Fast",
            "quality": "Excellent",
            "models": ["deepseek-chat", "deepseek-coder"]
        },
        "huggingface": {
            "name": "Hugging Face",
            "description": "Free tier inference API",
            "cost": "Free tier: 300 req/hour",
            "speed": "Medium (model loading on first request)",
            "quality": "Good to Excellent",
            "models": ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
        },
        "gemini": {
            "name": "Google Gemini",
            "description": "Fast, cheap, high quality",
            "cost": "Flash: Very cheap, Pro: $1.25/$5",
            "speed": "Very Fast",
            "quality": "Excellent",
            "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp", "gemini-exp-1206"]
        },
        "openai": {
            "name": "OpenAI GPT",
            "description": "Industry standard",
            "cost": "4o-mini: $0.15/$0.60, 4o: $2.50/$10",
            "speed": "Fast",
            "quality": "Excellent",
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        },
        "anthropic": {
            "name": "Anthropic Claude",
            "description": "Best reasoning, long context",
            "cost": "Sonnet 4.5: $3/$15, Sonnet 3.5: $3/$15, Opus: $15/$75",
            "speed": "Medium",
            "quality": "Best",
            "models": ["claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        },
        "grok": {
            "name": "xAI Grok",
            "description": "Real-time X/Twitter data",
            "cost": "TBD",
            "speed": "Fast",
            "quality": "Good",
            "models": ["grok-beta"]
        },
        "kimi": {
            "name": "Kimi/Moonshot",
            "description": "Chinese model, 200k context",
            "cost": "Cheap",
            "speed": "Medium",
            "quality": "Good",
            "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
        }
    }

    def __init__(self) -> None:
        """Initialize LLM Manager"""
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all available providers"""
        # Ollama (local, always try)
        ollama = OllamaProvider()
        if ollama.is_available():
            self.providers["ollama"] = ollama
            logger.info("✓ Ollama available (local)")

        # Groq (free tier)
        groq = GroqProvider()
        if groq.is_available():
            self.providers["groq"] = groq
            logger.info("✓ Groq available (free tier)")

        # DeepSeek (very cheap)
        deepseek = DeepSeekProvider()
        if deepseek.is_available():
            self.providers["deepseek"] = deepseek
            logger.info("✓ DeepSeek available ($0.14/$0.28 per 1M)")

        # Hugging Face (free tier)
        huggingface = HuggingFaceProvider()
        if huggingface.is_available():
            self.providers["huggingface"] = huggingface
            logger.info("✓ Hugging Face available (free tier: 300 req/hour)")

        # Gemini (Google)
        gemini = GeminiProvider()
        if gemini.is_available():
            self.providers["gemini"] = gemini
            logger.info("✓ Gemini available")

        # OpenAI
        openai = OpenAIProvider()
        if openai.is_available():
            self.providers["openai"] = openai
            logger.info("✓ OpenAI available")

        # Anthropic
        anthropic = AnthropicProvider()
        if anthropic.is_available():
            self.providers["anthropic"] = anthropic
            logger.info("✓ Anthropic available")

        # Grok
        grok = GrokProvider()
        if grok.is_available():
            self.providers["grok"] = grok
            logger.info("✓ Grok available")

        # Kimi
        kimi = KimiProvider()
        if kimi.is_available():
            self.providers["kimi"] = kimi
            logger.info("✓ Kimi available")

        if not self.providers:
            logger.warning("No LLM providers available!")

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available providers with metadata"""
        available = []
        for provider_id, provider in self.providers.items():
            info = self.PROVIDERS[provider_id].copy()
            info["id"] = provider_id
            info["current_model"] = provider.model
            info["is_local"] = provider_id == "ollama"
            available.append(info)

        return sorted(available, key=lambda x: (
            0 if x["cost"] == "Free" else 1 if "Free tier" in x["cost"] else 2
        ))

    def generate(self, prompt: str, provider_id: Optional[str] = None,
                model: Optional[str] = None, max_tokens: int = 1000,
                temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate response using specified provider or auto-select

        Args:
            prompt: The prompt to send
            provider_id: Provider to use (auto-selects if None)
            model: Model to use (uses default if None)
            max_tokens: Max tokens to generate
            temperature: Temperature for sampling

        Returns:
            Dict with 'text', 'provider', 'model', 'tokens_used'
        """
        # Auto-select provider if not specified
        if provider_id is None or provider_id not in self.providers:
            # Priority: Ollama (free) > Groq (free) > HuggingFace (free tier) > DeepSeek (cheap) > Gemini > others
            for fallback in ["ollama", "groq", "huggingface", "deepseek", "gemini"]:
                if fallback in self.providers:
                    provider_id = fallback
                    logger.info(f"Auto-selected provider: {fallback}")
                    break

            if provider_id is None:
                # Use any available provider
                provider_id = list(self.providers.keys())[0]

        provider = self.providers[provider_id]

        # Update model if specified
        if model:
            provider.model = model

        # Generate response
        text = provider.generate(prompt, max_tokens, temperature)

        return {
            "text": text,
            "provider": provider_id,
            "model": provider.model,
            "tokens_used": len(text.split()) * 1.3  # Rough estimate
        }


# Singleton instance
_llm_manager = None


def get_llm_manager() -> LLMManager:
    """Get or create singleton LLM manager"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


# Test
if __name__ == "__main__":
    manager = get_llm_manager()

    print("\n=== Available LLM Providers ===")
    for provider in manager.get_available_providers():
        print(f"\n{provider['name']} ({provider['id']})")
        print(f"  Description: {provider['description']}")
        print(f"  Cost: {provider['cost']}")
        print(f"  Speed: {provider['speed']}")
        print(f"  Quality: {provider['quality']}")
        print(f"  Current Model: {provider['current_model']}")

    # Test generation
    print("\n=== Testing Generation ===")
    test_prompt = "Analyze AAPL stock for options trading. Is it a good CSP candidate? Be very brief (2-3 sentences)."

    result = manager.generate(test_prompt, max_tokens=200)
    print(f"\nProvider: {result['provider']}")
    print(f"Model: {result['model']}")
    print(f"Response: {result['text']}")
    print(f"Tokens: ~{result['tokens_used']:.0f}")
