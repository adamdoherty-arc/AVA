"""
DeepSeek R1 32B Deep Reasoning API
===================================

Dedicated endpoint for DeepSeek R1's chain-of-thought reasoning capabilities.
Optimized for complex multi-step analysis, hypothesis testing, and decision-making.

Uses DeepSeek R1 32B (R1-0528 update) which approaches O3/Gemini 2.5 Pro performance.

Author: Magnus AI Team
Created: 2025-12-04
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
import structlog
import time

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/reasoning",
    tags=["reasoning", "deepseek-r1"]
)


# =============================================================================
# Request/Response Models
# =============================================================================

class ReasoningDepth(str, Enum):
    """Depth of reasoning analysis"""
    STANDARD = "standard"      # Quick step-by-step (~30s)
    DEEP = "deep"              # Thorough multi-perspective (~60s)
    EXHAUSTIVE = "exhaustive"  # Comprehensive with confidence levels (~90s)


class ReasoningCategory(str, Enum):
    """Category of reasoning task for optimized prompts"""
    TRADING = "trading"                    # Options, stocks, strategies
    PORTFOLIO = "portfolio"                # Portfolio optimization, allocation
    RISK = "risk"                          # Risk analysis, stress testing
    PREDICTION = "prediction"              # Market/sports predictions
    HYPOTHESIS = "hypothesis"              # Testing assumptions
    WHAT_IF = "what_if"                    # Scenario analysis
    GENERAL = "general"                    # General reasoning


class ReasoningRequest(BaseModel):
    """Request for deep reasoning analysis"""
    problem: str = Field(..., description="The problem or question requiring deep reasoning")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context data")
    depth: ReasoningDepth = Field(default=ReasoningDepth.DEEP, description="Reasoning depth level")
    category: ReasoningCategory = Field(default=ReasoningCategory.GENERAL, description="Category for optimized prompts")
    extract_thinking: bool = Field(default=True, description="Separate thinking process from final answer")
    max_tokens: int = Field(default=6000, ge=1000, le=16000, description="Maximum response tokens")


class ThinkingProcess(BaseModel):
    """Extracted chain-of-thought reasoning"""
    raw_thinking: str = Field(description="Full internal reasoning process")
    key_considerations: List[str] = Field(default_factory=list, description="Main points considered")
    assumptions_identified: List[str] = Field(default_factory=list, description="Assumptions made")
    alternatives_explored: List[str] = Field(default_factory=list, description="Alternative scenarios")


class ReasoningAnswer(BaseModel):
    """Structured final answer"""
    problem_understanding: str = Field(description="Restated core problem")
    key_variables: List[str] = Field(description="Critical factors identified")
    reasoning_chain: str = Field(description="Summary of logical analysis")
    conclusion: str = Field(description="Final recommendation")
    confidence: Literal["High", "Medium", "Low"] = Field(description="Confidence level")
    risks: List[str] = Field(default_factory=list, description="Identified risks or downsides")


class ReasoningResponse(BaseModel):
    """Response from deep reasoning analysis"""
    status: Literal["success", "error"] = Field(description="Request status")
    model: str = Field(default="DeepSeek R1 32B (R1-0528)", description="Model used")
    model_id: str = Field(default="deepseek-r1:32b", description="Ollama model ID")

    # Reasoning outputs
    thinking: Optional[ThinkingProcess] = Field(default=None, description="Chain-of-thought process")
    answer: Optional[ReasoningAnswer] = Field(default=None, description="Structured final answer")
    raw_response: str = Field(default="", description="Full raw model response")

    # Metadata
    depth: ReasoningDepth = Field(description="Reasoning depth used")
    category: ReasoningCategory = Field(description="Category used")
    confidence: str = Field(default="Medium", description="Overall confidence")
    latency_ms: float = Field(default=0, description="Processing time in milliseconds")

    # Error handling
    error: Optional[str] = Field(default=None, description="Error message if failed")
    fallback_available: bool = Field(default=True, description="Whether fallback model is available")


class QuickReasoningRequest(BaseModel):
    """Simplified request for quick reasoning queries"""
    question: str = Field(..., description="Question to reason about")
    context: Optional[str] = Field(default=None, description="Optional context string")


class QuickReasoningResponse(BaseModel):
    """Simplified response for quick reasoning"""
    answer: str = Field(description="Reasoning response")
    confidence: str = Field(default="Medium", description="Confidence level")
    model: str = Field(default="DeepSeek R1 32B", description="Model used")
    latency_ms: float = Field(default=0, description="Processing time")


class ModelHealthResponse(BaseModel):
    """Health check for reasoning models"""
    deepseek_r1_available: bool = Field(description="DeepSeek R1 32B availability")
    qwen_fallback_available: bool = Field(description="Qwen 2.5 32B fallback availability")
    ollama_running: bool = Field(description="Ollama server status")
    models_loaded: List[str] = Field(default_factory=list, description="Currently loaded models")
    gpu_available: bool = Field(default=False, description="GPU acceleration available")
    recommended_model: str = Field(default="deepseek-r1:32b", description="Recommended model")


# =============================================================================
# Category-specific System Prompts
# =============================================================================

CATEGORY_PROMPTS = {
    ReasoningCategory.TRADING: """You are AVA's expert trading analyst powered by DeepSeek R1.
Focus on: Options strategies (CSP, CC, spreads), Greeks analysis, entry/exit timing,
risk-reward ratios, and position sizing. Consider IV environment, earnings, and market regime.""",

    ReasoningCategory.PORTFOLIO: """You are AVA's portfolio optimization engine powered by DeepSeek R1.
Focus on: Asset allocation, diversification, correlation analysis, rebalancing strategies,
sector exposure, and risk-adjusted returns. Consider beta, Sharpe ratio, and drawdown metrics.""",

    ReasoningCategory.RISK: """You are AVA's risk assessment engine powered by DeepSeek R1.
Focus on: VaR calculations, stress testing scenarios, tail risk, margin requirements,
position concentration, and hedging strategies. Consider worst-case scenarios.""",

    ReasoningCategory.PREDICTION: """You are AVA's prediction analysis engine powered by DeepSeek R1.
Focus on: Probability estimation, factor weighting, confidence intervals, ensemble methods,
and edge detection. Quantify uncertainty and identify key predictive variables.""",

    ReasoningCategory.HYPOTHESIS: """You are AVA's hypothesis testing engine powered by DeepSeek R1.
Focus on: Assumption validation, counter-arguments, evidence evaluation, logical fallacies,
and conclusion strength. Challenge your own reasoning and identify weak points.""",

    ReasoningCategory.WHAT_IF: """You are AVA's scenario analysis engine powered by DeepSeek R1.
Focus on: Alternative outcomes, sensitivity analysis, cascading effects, probability trees,
and contingency planning. Explore the full possibility space.""",

    ReasoningCategory.GENERAL: """You are AVA's deep reasoning engine powered by DeepSeek R1.
Your reasoning capabilities approach frontier models like O3 and Gemini 2.5 Pro.
Use your full chain-of-thought reasoning abilities to solve this problem."""
}


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/analyze", response_model=ReasoningResponse)
async def deep_reasoning_analyze(request: ReasoningRequest):
    """
    Perform deep reasoning analysis using DeepSeek R1 32B.

    This endpoint leverages DeepSeek R1's chain-of-thought capabilities for:
    - Multi-step reasoning problems
    - Hypothesis testing and validation
    - Complex portfolio optimization
    - Risk scenario analysis
    - What-if market scenarios

    The R1-0528 update approaches O3/Gemini 2.5 Pro performance on reasoning benchmarks.
    """
    start_time = time.time()

    try:
        # Import here to avoid circular imports
        from src.magnus_local_llm import get_magnus_llm

        llm = get_magnus_llm()

        # Use the enhanced deep_reasoning method
        result = llm.deep_reasoning(
            problem=request.problem,
            context=request.context,
            reasoning_depth=request.depth.value,
            extract_thinking=request.extract_thinking
        )

        latency_ms = (time.time() - start_time) * 1000

        if result.get("status") == "success":
            # Parse thinking into structured format
            thinking = None
            if request.extract_thinking and result.get("thinking"):
                thinking = ThinkingProcess(
                    raw_thinking=result["thinking"],
                    key_considerations=_extract_bullet_points(result["thinking"], "consider"),
                    assumptions_identified=_extract_bullet_points(result["thinking"], "assum"),
                    alternatives_explored=_extract_bullet_points(result["thinking"], "alternat")
                )

            # Parse answer into structured format
            answer = None
            if result.get("answer"):
                answer = ReasoningAnswer(
                    problem_understanding=_extract_section(result["answer"], "Problem Understanding"),
                    key_variables=_extract_list_section(result["answer"], "Key Variables"),
                    reasoning_chain=_extract_section(result["answer"], "Reasoning Chain"),
                    conclusion=_extract_section(result["answer"], "Conclusion"),
                    confidence=result.get("confidence", "Medium"),
                    risks=_extract_list_section(result["answer"], "Risk")
                )

            logger.info(
                "deep_reasoning_complete",
                depth=request.depth.value,
                category=request.category.value,
                confidence=result.get("confidence"),
                latency_ms=round(latency_ms, 2)
            )

            return ReasoningResponse(
                status="success",
                thinking=thinking,
                answer=answer,
                raw_response=result.get("response", ""),
                depth=request.depth,
                category=request.category,
                confidence=result.get("confidence", "Medium"),
                latency_ms=latency_ms
            )
        else:
            return ReasoningResponse(
                status="error",
                error=result.get("error", "Unknown error"),
                depth=request.depth,
                category=request.category,
                latency_ms=latency_ms,
                fallback_available=result.get("fallback_available", True)
            )

    except Exception as e:
        logger.error("deep_reasoning_error", error=str(e))
        return ReasoningResponse(
            status="error",
            error=str(e),
            depth=request.depth,
            category=request.category,
            latency_ms=(time.time() - start_time) * 1000,
            fallback_available=True
        )


@router.post("/quick", response_model=QuickReasoningResponse)
async def quick_reasoning(request: QuickReasoningRequest):
    """
    Quick reasoning query using DeepSeek R1 32B.

    For simpler reasoning tasks that don't require full chain-of-thought extraction.
    Faster response times (~10-20 seconds).
    """
    start_time = time.time()

    try:
        from src.magnus_local_llm import get_magnus_llm

        llm = get_magnus_llm()
        response = llm.quick_reasoning(
            question=request.question,
            context=request.context
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract confidence if present
        confidence = "Medium"
        if "high confidence" in response.lower():
            confidence = "High"
        elif "low confidence" in response.lower():
            confidence = "Low"

        return QuickReasoningResponse(
            answer=response,
            confidence=confidence,
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error("quick_reasoning_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=ModelHealthResponse)
async def check_reasoning_model_health():
    """
    Check health and availability of reasoning models.

    Returns status of DeepSeek R1 32B and fallback models.
    Uses async HTTP client for non-blocking network calls.
    """
    import asyncio
    import httpx

    try:
        # Check Ollama server (async HTTP)
        ollama_running = False
        models_loaded = []
        deepseek_available = False
        qwen_available = False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    ollama_running = True
                    data = response.json()
                    models = data.get("models", [])
                    models_loaded = [m.get("name", "") for m in models]

                    deepseek_available = any("deepseek-r1" in m for m in models_loaded)
                    qwen_available = any("qwen2.5:32b" in m for m in models_loaded)
        except Exception:
            pass

        # Check GPU (run subprocess in thread pool to avoid blocking)
        gpu_available = False
        try:
            import subprocess
            def _check_gpu():
                result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
                return result.returncode == 0
            gpu_available = await asyncio.to_thread(_check_gpu)
        except Exception:
            pass

        recommended = "deepseek-r1:32b" if deepseek_available else "qwen2.5:32b-instruct-q4_K_M"

        return ModelHealthResponse(
            deepseek_r1_available=deepseek_available,
            qwen_fallback_available=qwen_available,
            ollama_running=ollama_running,
            models_loaded=models_loaded,
            gpu_available=gpu_available,
            recommended_model=recommended
        )

    except Exception as e:
        logger.error("health_check_error", error=str(e))
        return ModelHealthResponse(
            deepseek_r1_available=False,
            qwen_fallback_available=False,
            ollama_running=False,
            models_loaded=[],
            gpu_available=False,
            recommended_model="deepseek-r1:32b"
        )


@router.get("/capabilities")
async def get_reasoning_capabilities():
    """
    Get information about DeepSeek R1 32B reasoning capabilities.
    """
    return {
        "model": "DeepSeek R1 32B (R1-0528)",
        "model_id": "deepseek-r1:32b",
        "capabilities": [
            "Chain-of-thought reasoning",
            "Multi-step analysis",
            "Hypothesis testing",
            "Portfolio optimization",
            "Risk scenario analysis",
            "What-if simulations",
            "Strategy backtesting logic",
            "Correlation analysis"
        ],
        "performance": {
            "benchmark_level": "Approaches O3 and Gemini 2.5 Pro",
            "vram_required": "19 GB",
            "context_window": "131,072 tokens",
            "typical_speed": "~25 tokens/second"
        },
        "reasoning_depths": {
            "standard": "Quick step-by-step analysis (~30s)",
            "deep": "Thorough multi-perspective analysis (~60s)",
            "exhaustive": "Comprehensive with confidence levels (~90s)"
        },
        "categories": [
            "trading", "portfolio", "risk", "prediction",
            "hypothesis", "what_if", "general"
        ],
        "cost": "FREE (local execution)",
        "privacy": "All processing done locally - no data leaves your machine"
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_section(text: str, section_name: str) -> str:
    """Extract a section from structured text"""
    import re
    pattern = rf'\*?\*?{section_name}\*?\*?[:\s]*(.*?)(?=\n\*?\*?\d+\.|\n\*?\*?[A-Z]|\Z)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_list_section(text: str, section_name: str) -> List[str]:
    """Extract a list from a section"""
    section = _extract_section(text, section_name)
    if not section:
        return []

    import re
    # Match bullet points or numbered items
    items = re.findall(r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|\Z)', section, re.DOTALL)
    if not items:
        items = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|\Z)', section, re.DOTALL)

    return [item.strip() for item in items if item.strip()]


def _extract_bullet_points(text: str, keyword: str) -> List[str]:
    """Extract bullet points containing a keyword"""
    import re
    lines = text.split('\n')
    results = []
    for line in lines:
        if keyword.lower() in line.lower():
            # Clean up the line
            clean = re.sub(r'^[-•*\d.)\s]+', '', line).strip()
            if clean and len(clean) > 10:
                results.append(clean)
    return results[:5]  # Limit to 5 items
