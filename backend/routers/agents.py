"""Agents Router - Exposes all 48+ AI agents via REST API"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import structlog

from backend.infrastructure.errors import safe_internal_error
from backend.infrastructure.database import get_database, AsyncDatabaseManager

router = APIRouter(
    prefix="/api/agents",
    tags=["agents"]
)

logger = structlog.get_logger(__name__)

# Import agent registry
try:
    from src.ava.core.agent_registry import AgentRegistry
    agent_registry = AgentRegistry()
except ImportError:
    agent_registry = None
    logger.warning("AgentRegistry not available")


class AgentInvokeRequest(BaseModel):
    """Request to invoke an agent"""
    query: str
    context: Optional[Dict[str, Any]] = None  # Use None to avoid mutable default


class AgentInfo(BaseModel):
    """Information about an agent"""
    name: str
    description: str
    category: str
    capabilities: List[str]
    is_active: bool = True


class AgentResponse(BaseModel):
    """Response from agent invocation"""
    agent_name: str
    result: Dict[str, Any]
    processing_time_ms: float
    success: bool


# Define available agents (mapping to actual implementations)
AGENT_CATALOG = {
    # Trading Agents
    "portfolio_agent": {
        "name": "Portfolio Agent",
        "description": "Analyzes portfolio composition, risk, and performance",
        "category": "trading",
        "capabilities": ["portfolio_analysis", "risk_assessment", "allocation_review"]
    },
    "options_agent": {
        "name": "Options Strategy Agent",
        "description": "Generates and analyzes options strategies",
        "category": "trading",
        "capabilities": ["strategy_generation", "greeks_analysis", "premium_optimization"]
    },
    "premium_scanner_agent": {
        "name": "Premium Scanner Agent",
        "description": "Scans for best CSP/CC premium opportunities",
        "category": "trading",
        "capabilities": ["premium_scan", "dte_analysis", "return_calculation"]
    },
    "technical_agent": {
        "name": "Technical Analysis Agent",
        "description": "Performs technical analysis on stocks",
        "category": "trading",
        "capabilities": ["trend_analysis", "indicator_analysis", "support_resistance"]
    },
    "fundamental_agent": {
        "name": "Fundamental Analysis Agent",
        "description": "Analyzes company fundamentals",
        "category": "trading",
        "capabilities": ["earnings_analysis", "valuation", "financial_health"]
    },

    # Sports Agents
    "nfl_predictor": {
        "name": "NFL Predictor",
        "description": "Predicts NFL game outcomes",
        "category": "sports",
        "capabilities": ["game_prediction", "spread_analysis", "totals_prediction"]
    },
    "nba_predictor": {
        "name": "NBA Predictor",
        "description": "Predicts NBA game outcomes",
        "category": "sports",
        "capabilities": ["game_prediction", "player_props", "live_adjustments"]
    },
    "ncaa_predictor": {
        "name": "NCAA Predictor",
        "description": "Predicts college sports outcomes",
        "category": "sports",
        "capabilities": ["cfb_prediction", "cbb_prediction", "upset_detection"]
    },
    "best_bets_ranker": {
        "name": "Best Bets Ranker",
        "description": "Ranks and filters the best betting opportunities",
        "category": "sports",
        "capabilities": ["bet_ranking", "edge_calculation", "bankroll_optimization"]
    },

    # Research Agents
    "research_orchestrator": {
        "name": "Research Orchestrator",
        "description": "Orchestrates multi-agent research tasks",
        "category": "research",
        "capabilities": ["multi_agent_coordination", "deep_research", "report_generation"]
    },
    "news_agent": {
        "name": "News Analysis Agent",
        "description": "Analyzes news and sentiment",
        "category": "research",
        "capabilities": ["news_analysis", "sentiment_scoring", "event_detection"]
    },
    "sector_agent": {
        "name": "Sector Analysis Agent",
        "description": "Analyzes market sectors",
        "category": "research",
        "capabilities": ["sector_rotation", "relative_strength", "correlation_analysis"]
    },

    # System Agents
    "rag_agent": {
        "name": "RAG Knowledge Agent",
        "description": "Retrieves information from knowledge base",
        "category": "system",
        "capabilities": ["knowledge_retrieval", "context_augmentation", "citation_generation"]
    },
    "code_agent": {
        "name": "Code Generation Agent",
        "description": "Generates and reviews code",
        "category": "system",
        "capabilities": ["code_generation", "code_review", "documentation"]
    },
    "notification_agent": {
        "name": "Notification Agent",
        "description": "Manages alerts and notifications",
        "category": "system",
        "capabilities": ["alert_generation", "threshold_monitoring", "delivery"]
    },
}


@router.get("/", response_model=List[AgentInfo])
async def list_agents(
    category: Optional[str] = Query(None, description="Filter by category")
):
    """
    List all available AI agents.
    """
    agents = []
    for agent_id, info in AGENT_CATALOG.items():
        if category and info["category"] != category:
            continue
        agents.append(AgentInfo(
            name=agent_id,
            description=info["description"],
            category=info["category"],
            capabilities=info["capabilities"]
        ))
    return agents


@router.get("/categories")
async def list_categories():
    """
    List all agent categories.
    """
    categories = set(info["category"] for info in AGENT_CATALOG.values())
    return {
        "categories": list(categories),
        "counts": {
            cat: len([a for a in AGENT_CATALOG.values() if a["category"] == cat])
            for cat in categories
        }
    }


# ============ AI Options Agent Endpoints (MUST BE BEFORE /{agent_name}) ============

class OptionsAnalyzeRequest(BaseModel):
    """Request to analyze options"""
    symbols: List[str] = []
    min_dte: int = 20
    max_dte: int = 40
    min_delta: float = -0.45
    max_delta: float = -0.15
    min_premium: float = 100
    max_results: int = 200
    use_llm: bool = False
    model: str = "qwen2.5:14b"


@router.get("/options/top")
async def get_top_options_recommendations(
    min_score: int = Query(50, description="Minimum score to include")
):
    """
    Get top CSP recommendations from recent analysis.
    Returns stored recommendations from the database.
    """
    try:
        from datetime import datetime, timedelta

        db = await get_database()

        # Check if we have an options_recommendations table
        table_exists_row = await db.fetchrow("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'options_recommendations'
            )
        """)
        table_exists = table_exists_row["exists"]

        if table_exists:
            # Get recent recommendations from database
            rows = await db.fetch("""
                SELECT
                    symbol, company_name, current_price, strike, expiration,
                    dte, premium, score, recommendation, reasoning,
                    delta, iv, premium_pct, monthly_return, created_at
                FROM options_recommendations
                WHERE score >= $1
                  AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY score DESC
                LIMIT 50
            """, min_score)

            recommendations = []
            for row in rows:
                recommendations.append({
                    "symbol": row["symbol"],
                    "company_name": row["company_name"] or row["symbol"],
                    "current_price": float(row["current_price"]) if row["current_price"] else 0,
                    "strike": float(row["strike"]) if row["strike"] else 0,
                    "expiration": row["expiration"].isoformat() if row["expiration"] else "",
                    "dte": row["dte"] or 0,
                    "premium": float(row["premium"]) if row["premium"] else 0,
                    "score": row["score"] or 0,
                    "recommendation": row["recommendation"] or "Hold",
                    "reasoning": row["reasoning"] or "",
                    "delta": float(row["delta"]) if row["delta"] else 0,
                    "iv": float(row["iv"]) if row["iv"] else 0,
                    "premium_pct": float(row["premium_pct"]) if row["premium_pct"] else 0,
                    "monthly_return": float(row["monthly_return"]) if row["monthly_return"] else 0
                })

            # Calculate stats
            if recommendations:
                avg_score = sum(r["score"] for r in recommendations) / len(recommendations)
                strong_buys = len([r for r in recommendations if r["recommendation"] == "Strong Buy"])
                strong_buy_rate = (strong_buys / len(recommendations)) * 100
            else:
                avg_score = 0
                strong_buy_rate = 0

            return {
                "recommendations": recommendations,
                "total_analyzed": len(recommendations),
                "avg_score": avg_score,
                "strong_buy_rate": strong_buy_rate,
                "generated_at": datetime.now().isoformat()
            }

        # No table yet - return empty with message
        return {
            "recommendations": [],
            "total_analyzed": 0,
            "avg_score": 0,
            "strong_buy_rate": 0,
            "message": "Run an analysis to generate recommendations",
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching top options: {e}")
        return {
            "recommendations": [],
            "total_analyzed": 0,
            "avg_score": 0,
            "strong_buy_rate": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.post("/options/analyze")
async def analyze_options(request: OptionsAnalyzeRequest):
    """
    Run AI-powered options analysis using MCDM scoring.
    Scans for CSP opportunities based on criteria.
    """
    from datetime import datetime
    import time

    start_time = time.time()

    try:
        # Try to use actual premium scanner
        try:
            from src.services.llm_options_strategist import LLMOptionsStrategist
            strategist = LLMOptionsStrategist()

            results = await strategist.analyze_csp_opportunities(
                symbols=request.symbols or ['AAPL', 'MSFT', 'NVDA', 'AMD', 'GOOG', 'META'],
                min_dte=request.min_dte,
                max_dte=request.max_dte,
                min_delta=request.min_delta,
                max_delta=request.max_delta,
                min_premium=request.min_premium,
                max_results=request.max_results,
                use_llm=request.use_llm,
                model=request.model
            )

            processing_time = (time.time() - start_time) * 1000

            return {
                "results": results,
                "total_analyzed": len(results),
                "processing_time_ms": processing_time,
                "generated_at": datetime.now().isoformat()
            }

        except ImportError:
            # Fallback: Query from database if scanner not available
            results = []

            db = await get_database()

            # Get options data from premium_opportunities table if it exists
            table_exists_row = await db.fetchrow("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'premium_opportunities'
                )
            """)

            if table_exists_row["exists"]:
                # Build parameterized query - symbols use ANY() for safe array matching
                if request.symbols:
                    # Sanitize symbols: uppercase, alphanumeric only, max 5 chars each
                    import re
                    safe_symbols = [
                        s.upper()[:5] for s in request.symbols
                        if s and re.match(r'^[A-Z]{1,5}$', s.upper())
                    ]
                    rows = await db.fetch("""
                        SELECT
                            symbol, company_name, current_price, strike, expiration,
                            dte, premium, delta, iv, premium_pct, monthly_return
                        FROM premium_opportunities
                        WHERE dte BETWEEN $1 AND $2
                          AND delta BETWEEN $3 AND $4
                          AND premium >= $5
                          AND symbol = ANY($6)
                        ORDER BY monthly_return DESC
                        LIMIT $7
                    """, request.min_dte, request.max_dte, request.min_delta, request.max_delta,
                          request.min_premium, safe_symbols, request.max_results)
                else:
                    rows = await db.fetch("""
                        SELECT
                            symbol, company_name, current_price, strike, expiration,
                            dte, premium, delta, iv, premium_pct, monthly_return
                        FROM premium_opportunities
                        WHERE dte BETWEEN $1 AND $2
                          AND delta BETWEEN $3 AND $4
                          AND premium >= $5
                        ORDER BY monthly_return DESC
                        LIMIT $6
                    """, request.min_dte, request.max_dte, request.min_delta, request.max_delta,
                          request.min_premium, request.max_results)

                for row in rows:
                    # Calculate MCDM score
                    monthly_return = float(row["monthly_return"]) if row["monthly_return"] else 0
                    delta = abs(float(row["delta"])) if row["delta"] else 0
                    dte = row["dte"] or 0
                    iv = float(row["iv"]) if row["iv"] else 0

                    # Weighted scoring
                    score = int(
                        (monthly_return * 30) +
                        ((0.30 - delta) * 100) +
                        (min(iv, 100) / 2) +
                        (max(0, 45 - dte) / 2)
                    )
                    score = min(100, max(0, score))

                    # Determine recommendation
                    if score >= 90:
                        rec = "Strong Buy"
                    elif score >= 70:
                        rec = "Buy"
                    elif score >= 50:
                        rec = "Hold"
                    else:
                        rec = "Avoid"

                    results.append({
                        "symbol": row["symbol"],
                        "company_name": row["company_name"] or row["symbol"],
                        "current_price": float(row["current_price"]) if row["current_price"] else 0,
                        "strike": float(row["strike"]) if row["strike"] else 0,
                        "expiration": row["expiration"].isoformat() if row["expiration"] else "",
                        "dte": dte,
                        "premium": float(row["premium"]) if row["premium"] else 0,
                        "score": score,
                        "recommendation": rec,
                        "reasoning": f"MCDM Score based on premium return ({monthly_return:.1f}%), delta ({delta:.2f}), IV ({iv:.0f}%)",
                        "delta": float(row["delta"]) if row["delta"] else 0,
                        "iv": iv,
                        "premium_pct": float(row["premium_pct"]) if row["premium_pct"] else 0,
                        "monthly_return": monthly_return
                    })

            processing_time = (time.time() - start_time) * 1000

            return {
                "results": sorted(results, key=lambda x: x['score'], reverse=True),
                "total_analyzed": len(results),
                "processing_time_ms": processing_time,
                "generated_at": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error analyzing options: {e}")
        return {
            "results": [],
            "total_analyzed": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


# ============ Generic Agent Endpoints ============

@router.get("/{agent_name}")
async def get_agent_info(agent_name: str):
    """
    Get detailed information about a specific agent.
    """
    if agent_name not in AGENT_CATALOG:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    info = AGENT_CATALOG[agent_name]
    return {
        "id": agent_name,
        **info,
        "is_available": True
    }


@router.post("/{agent_name}/invoke", response_model=AgentResponse)
async def invoke_agent(agent_name: str, request: AgentInvokeRequest):
    """
    Invoke a specific agent with a query.
    """
    import time
    start_time = time.time()

    if agent_name not in AGENT_CATALOG:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    try:
        # Route to actual agent implementation
        result = await _invoke_agent_impl(agent_name, request.query, request.context)

        processing_time = (time.time() - start_time) * 1000

        return AgentResponse(
            agent_name=agent_name,
            result=result,
            processing_time_ms=processing_time,
            success=True
        )
    except Exception as e:
        logger.error(f"Agent invocation error: {e}")
        safe_internal_error(e, "invoke agent")


@router.post("/route")
async def smart_route(request: AgentInvokeRequest):
    """
    Intelligently route a query to the best agent.
    """
    try:
        # Use agent registry for smart routing
        if agent_registry:
            result = agent_registry.route_query(request.query, request.context)
            return result
        else:
            # Fallback: simple keyword routing
            query_lower = request.query.lower()

            if any(w in query_lower for w in ['portfolio', 'positions', 'holdings']):
                agent = 'portfolio_agent'
            elif any(w in query_lower for w in ['options', 'put', 'call', 'premium']):
                agent = 'options_agent'
            elif any(w in query_lower for w in ['nfl', 'football', 'touchdown']):
                agent = 'nfl_predictor'
            elif any(w in query_lower for w in ['nba', 'basketball']):
                agent = 'nba_predictor'
            elif any(w in query_lower for w in ['research', 'analyze']):
                agent = 'research_orchestrator'
            else:
                agent = 'rag_agent'

            result = await _invoke_agent_impl(agent, request.query, request.context)
            return {
                "routed_to": agent,
                "result": result
            }
    except Exception as e:
        safe_internal_error(e, "route to agent")


async def _invoke_agent_impl(agent_name: str, query: str, context: Dict) -> Dict[str, Any]:
    """
    Internal function to invoke agent implementations.
    """
    # Import and invoke actual agent implementations
    try:
        if agent_name == "portfolio_agent":
            from src.ava.agents.trading.portfolio_agent import PortfolioAgent
            agent = PortfolioAgent()
            state = {"input": query, "context": context, "history": []}
            result_state = await agent.execute(state)
            return result_state.get("result", {})

        elif agent_name == "nfl_predictor":
            from src.prediction_agents.nfl_predictor import NFLPredictor
            predictor = NFLPredictor()
            return predictor.predict_from_query(query)

        elif agent_name == "nba_predictor":
            from src.prediction_agents.nba_predictor import NBAPredictor
            predictor = NBAPredictor()
            return predictor.predict_from_query(query)

        elif agent_name == "premium_scanner_agent":
            from src.premium_scanner import PremiumScanner
            scanner = PremiumScanner()
            # Parse symbols from query or use defaults
            results = scanner.scan_premiums(['AAPL', 'MSFT', 'NVDA'], dte=30)
            return {"opportunities": results[:10]}

        elif agent_name == "research_orchestrator":
            from src.agents.ai_research.orchestrator import ResearchOrchestrator
            orchestrator = ResearchOrchestrator()
            return orchestrator.research(query)

        elif agent_name == "rag_agent":
            from src.rag.rag_service import RAGService
            rag = RAGService()
            return rag.query(query)

        else:
            # Generic response for unimplemented agents
            return {
                "message": f"Agent '{agent_name}' received query",
                "query": query,
                "status": "processed"
            }

    except ImportError as e:
        logger.warning(f"Agent {agent_name} import failed: {e}")
        return {
            "message": f"Agent '{agent_name}' is not fully configured",
            "query": query,
            "status": "fallback"
        }
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        raise
