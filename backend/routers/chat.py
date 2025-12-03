from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from src.ava.agent_aware_nlp_handler import get_agent_aware_handler

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"]
)

# Initialize Ava Handler (singleton)
ava_handler = get_agent_aware_handler()

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    model: Optional[str] = "auto"  # Model selection: auto, fast, balanced, complex

# Available models for frontend - organized by provider tier
AVAILABLE_MODELS = {
    # Auto-selection
    "auto": {"name": "Auto Select", "description": "Intelligently route based on query complexity", "speed": "varies", "tier": "auto"},

    # Free Tier - Groq (ultra-fast)
    "groq-llama70b": {"name": "Llama 3.3 70B (Groq)", "description": "Best free model, ultra-fast", "speed": "~300 tok/s", "tier": "free"},
    "groq-mixtral": {"name": "Mixtral 8x7B (Groq)", "description": "Fast, excellent reasoning", "speed": "~200 tok/s", "tier": "free"},

    # Free Tier - Hugging Face
    "hf-llama8b": {"name": "Llama 3.1 8B (HF)", "description": "Free tier, good for quick queries", "speed": "~50 tok/s", "tier": "free"},
    "hf-mistral7b": {"name": "Mistral 7B (HF)", "description": "Free tier, fast responses", "speed": "~60 tok/s", "tier": "free"},
    "hf-mixtral": {"name": "Mixtral 8x7B (HF)", "description": "Free tier, excellent reasoning", "speed": "~30 tok/s", "tier": "free"},

    # Low Cost - DeepSeek
    "deepseek-chat": {"name": "DeepSeek Chat", "description": "$0.14/1M tokens, excellent quality", "speed": "~80 tok/s", "tier": "cheap"},
    "deepseek-coder": {"name": "DeepSeek Coder", "description": "Best for code analysis", "speed": "~80 tok/s", "tier": "cheap"},

    # Google Gemini
    "gemini-flash": {"name": "Gemini 2.5 Flash", "description": "Very fast, cost-effective", "speed": "~150 tok/s", "tier": "cheap"},
    "gemini-pro": {"name": "Gemini 2.5 Pro", "description": "High quality reasoning", "speed": "~100 tok/s", "tier": "standard"},

    # OpenAI
    "gpt-4o-mini": {"name": "GPT-4o Mini", "description": "Fast, affordable OpenAI", "speed": "~80 tok/s", "tier": "standard"},
    "gpt-4o": {"name": "GPT-4o", "description": "OpenAI flagship model", "speed": "~50 tok/s", "tier": "premium"},

    # Anthropic Claude
    "claude-haiku": {"name": "Claude 3 Haiku", "description": "Fast, affordable Claude", "speed": "~100 tok/s", "tier": "standard"},
    "claude-sonnet": {"name": "Claude Sonnet 4.5", "description": "Best reasoning, long context", "speed": "~50 tok/s", "tier": "premium"},

    # xAI Grok
    "grok": {"name": "Grok (xAI)", "description": "Real-time X/Twitter data", "speed": "~60 tok/s", "tier": "standard"},
}

# Model ID to actual provider/model mapping
MODEL_MAPPING = {
    "groq-llama70b": ("groq", "llama-3.3-70b-versatile"),
    "groq-mixtral": ("groq", "mixtral-8x7b-32768"),
    "hf-llama8b": ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
    "hf-mistral7b": ("huggingface", "mistralai/Mistral-7B-Instruct-v0.2"),
    "hf-mixtral": ("huggingface", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    "deepseek-chat": ("deepseek", "deepseek-chat"),
    "deepseek-coder": ("deepseek", "deepseek-coder"),
    "gemini-flash": ("gemini", "gemini-2.5-flash"),
    "gemini-pro": ("gemini", "gemini-2.5-pro"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4o": ("openai", "gpt-4o"),
    "claude-haiku": ("anthropic", "claude-3-haiku-20240307"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-5-20250929"),
    "grok": ("grok", "grok-beta"),
}

class ChatResponse(BaseModel):
    answer: str
    intent: str
    agent_used: Optional[str] = None
    service_used: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = []
    confidence: float
    processing_time_ms: float = 0.0



@router.get("/history")
async def get_chat_history():
    """
    Get chat history - returns empty list for now (history managed by frontend).
    """
    return {"history": [], "count": 0}

@router.get("/models")
async def get_models():
    """Get available models for chat."""
    return {"models": AVAILABLE_MODELS}

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with Ava (Agent-Aware).
    """
    try:
        # Process message using Ava's routing logic with model preference
        result = ava_handler.parse_query(
            request.message,
            context={
                'history': request.history,
                'model_preference': request.model
            }
        )
        
        # Extract response fields
        answer = result.get('response', "I'm not sure how to answer that.")
        intent = result.get('intent', 'UNKNOWN')
        agent_used = result.get('agent_used')
        service_used = result.get('service_used')
        confidence = result.get('confidence', 0.0)
        
        # Extract sources if available (from RAG or agents)
        sources = result.get('sources', [])
        if 'rag_context' in result:
             # If RAG was used but sources not explicitly structured, we might need to parse them
             # For now, we'll leave it as is, assuming RAGService integration handles it
             pass

        return ChatResponse(
            answer=answer,
            intent=intent,
            agent_used=agent_used,
            service_used=service_used,
            sources=sources,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
