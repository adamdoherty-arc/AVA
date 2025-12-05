from fastapi import APIRouter, HTTPException, Body, Request
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import structlog
from backend.infrastructure.observability import get_audit_logger, AuditEventType
from backend.infrastructure.rate_limiter import rate_limited, RateLimitExceeded
from backend.infrastructure.errors import safe_internal_error

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"]
)

# Lazy initialization to prevent import failures from breaking router
_ava_handler = None

def get_handler():
    """Lazy initialize Ava handler to prevent import failures from breaking router."""
    global _ava_handler
    if _ava_handler is None:
        try:
            from src.ava.agent_aware_nlp_handler import get_agent_aware_handler
            _ava_handler = get_agent_aware_handler()
            logger.info("ava_handler_initialized")
        except Exception as e:
            logger.error("ava_handler_init_failed", error=str(e))
            raise HTTPException(status_code=503, detail="Chat service temporarily unavailable")
    return _ava_handler

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

    # Local - DeepSeek R1 32B (Deep Reasoning) - R1-0528 Update
    "deepseek-r1": {"name": "DeepSeek R1 32B (Local)", "description": "Chain-of-thought reasoning approaching O3/Gemini 2.5 Pro (R1-0528)", "speed": "~25 tok/s", "tier": "local"},

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
    "deepseek-r1": ("ollama", "deepseek-r1:32b"),  # DeepSeek R1 32B - deep reasoning (R1-0528)
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
        handler = get_handler()
        result = handler.parse_query(
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
        safe_internal_error(e, "process chat message")


# Deep Reasoning endpoint using DeepSeek R1
class DeepReasoningRequest(BaseModel):
    problem: str
    context: Optional[Dict[str, Any]] = None
    depth: Optional[str] = "deep"  # standard, deep, exhaustive


class DeepReasoningResponse(BaseModel):
    status: str
    model: str
    reasoning_depth: str
    response: str
    processing_time_ms: float = 0.0


@router.post("/deep-reasoning", response_model=DeepReasoningResponse)
@rate_limited(requests=3, window=60)  # 3 requests per minute - very expensive operation
async def deep_reasoning(request: DeepReasoningRequest, req: Request):
    """
    Deep reasoning endpoint using DeepSeek R1 model.
    Rate limited to 3 requests per minute due to high computational cost.

    Use this for:
    - Complex multi-step analysis
    - Hypothesis testing
    - Portfolio optimization scenarios
    - Risk scenario analysis
    - What-if market scenarios
    - Strategy backtesting logic

    Depth levels:
    - standard: Basic step-by-step reasoning
    - deep: Full reasoning chain with multiple perspectives
    - exhaustive: Complete analysis with probabilities and edge cases
    """
    import time
    from src.magnus_local_llm import get_magnus_llm

    audit = get_audit_logger()
    start_time = time.time()

    try:
        llm = get_magnus_llm()

        # Validate depth
        depth = request.depth if request.depth in ["standard", "deep", "exhaustive"] else "deep"

        result = llm.deep_reasoning(
            problem=request.problem,
            context=request.context,
            reasoning_depth=depth
        )

        processing_time = (time.time() - start_time) * 1000

        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Deep reasoning failed: {result.get('error')}"
            )

        # Log successful deep reasoning query
        await audit.log(
            AuditEventType.DEEP_REASONING_QUERY,
            action=f"Deep reasoning query ({depth})",
            resource_type="ai_reasoning",
            details={
                "depth": depth,
                "problem_preview": request.problem[:200] if request.problem else "",
                "processing_time_ms": processing_time,
                "model": result.get("model", "DeepSeek R1"),
            },
            ip_address=req.client.host if req.client else None,
        )

        return DeepReasoningResponse(
            status=result.get("status", "success"),
            model=result.get("model", "DeepSeek R1"),
            reasoning_depth=result.get("reasoning_depth", depth),
            response=result.get("response", ""),
            processing_time_ms=processing_time
        )

    except Exception as e:
        safe_internal_error(e, "deep reasoning analysis")
