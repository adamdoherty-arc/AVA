"""
Orchestrator Router - API endpoints for the Master Orchestrator Agent
Provides natural language queries, semantic search, dependency analysis,
and efficiency gap detection for all AVA features.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging
from backend.infrastructure.rate_limiter import rate_limited, RateLimitExceeded
from backend.infrastructure.errors import safe_internal_error

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/orchestrator",
    tags=["orchestrator"]
)


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Natural language query request"""
    query: str = Field(..., description="Natural language query about AVA")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class QueryResponse(BaseModel):
    """Query response"""
    response: str
    query: str
    timestamp: str


class SearchRequest(BaseModel):
    """Semantic search request"""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=50)
    category: Optional[str] = Field(default=None)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Single search result"""
    name: str
    category: str
    purpose: str
    similarity_score: Optional[float] = None


class DependencyRequest(BaseModel):
    """Dependency analysis request"""
    feature_name: str = Field(..., description="Feature to analyze")
    direction: str = Field(default="both", description="upstream, downstream, or both")
    max_depth: int = Field(default=3, ge=1, le=10)


class DependencyNode(BaseModel):
    """Dependency node"""
    feature_name: str
    category: str
    dependency_type: str
    is_critical: bool
    depth: int


class DependencyResponse(BaseModel):
    """Dependency analysis response"""
    feature: str
    upstream: Optional[List[DependencyNode]] = None
    downstream: Optional[List[DependencyNode]] = None


class EfficiencyGapRequest(BaseModel):
    """Efficiency gap analysis request"""
    threshold: float = Field(default=7.0, ge=0.0, le=10.0)
    category: Optional[str] = Field(default=None)
    dimension: Optional[str] = Field(default=None)


class EfficiencyGap(BaseModel):
    """Feature with low efficiency"""
    feature_name: str
    category: str
    overall_rating: float
    low_dimensions: List[str]
    quick_wins: List[str]


class ImpactRequest(BaseModel):
    """Impact analysis request"""
    feature_name: str = Field(..., description="Feature being changed")
    change_description: str = Field(..., description="Description of the proposed change")


class ImpactResponse(BaseModel):
    """Impact analysis response"""
    affected_features: List[str]
    breaking_changes: List[str]
    test_impact: List[str]
    recommended_actions: List[str]
    risk_level: str


class EnhancementRequest(BaseModel):
    """Enhancement suggestions request"""
    feature_name: str = Field(..., description="Feature to get enhancements for")


class FeatureSummary(BaseModel):
    """Feature summary"""
    name: str
    category: str
    purpose: str
    rating: Optional[float] = None


class CategorySummary(BaseModel):
    """Category summary"""
    category: str
    feature_count: int
    avg_rating: float
    features_needing_attention: int


class AgentInfo(BaseModel):
    """Agent information"""
    name: str
    category: str
    description: str
    capabilities: List[str]
    source_file: str


# =============================================================================
# Thread-Safe Agent Singleton
# =============================================================================

import asyncio
from contextlib import asynccontextmanager

_orchestrator_agent = None
_agent_lock = asyncio.Lock()
_agent_initialized = False


async def get_orchestrator_agent():
    """
    Get or create the Master Orchestrator Agent with thread-safe initialization.
    Uses double-checked locking pattern for async safety.
    """
    global _orchestrator_agent, _agent_initialized

    # Fast path - agent already initialized
    if _agent_initialized:
        return _orchestrator_agent

    # Slow path - acquire lock and initialize
    async with _agent_lock:
        # Double-check after acquiring lock
        if not _agent_initialized:
            try:
                from src.ava.agents.orchestration.master_orchestrator_agent import MasterOrchestratorAgent
                _orchestrator_agent = MasterOrchestratorAgent()
                _agent_initialized = True
                logger.info("MasterOrchestratorAgent initialized (thread-safe)")
            except Exception as e:
                logger.error(f"Failed to initialize MasterOrchestratorAgent: {e}")
                safe_internal_error(e, "initialize orchestrator")

    return _orchestrator_agent


async def shutdown_orchestrator():
    """Cleanup orchestrator resources on shutdown"""
    global _orchestrator_agent, _agent_initialized

    if _orchestrator_agent is not None:
        try:
            # Close database pool if exists
            if hasattr(_orchestrator_agent, '_db_pool') and _orchestrator_agent._db_pool:
                await _orchestrator_agent._db_pool.close()
                logger.info("Orchestrator database pool closed")
        except Exception as e:
            logger.warning(f"Error closing orchestrator resources: {e}")
        finally:
            _orchestrator_agent = None
            _agent_initialized = False


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check for the orchestrator service.

    Returns:
        Service status and capabilities
    """
    try:
        agent = await get_orchestrator_agent()
        return {
            "status": "healthy",
            "agent": agent.name,
            "capabilities": agent.get_capabilities(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/query", response_model=QueryResponse)
@rate_limited(requests=10, window=60)  # 10 requests per minute - AI operation
async def natural_language_query(request: QueryRequest) -> QueryResponse:
    """
    Process a natural language query about AVA.
    Rate limited to 10 requests per minute.

    This is the main entry point for asking questions about the codebase.

    Examples:
    - "How does the premium scanner work?"
    - "What agents are involved in sports betting?"
    - "What has efficiency below 7?"
    - "What would break if I change the Robinhood integration?"
    """
    try:
        agent = await get_orchestrator_agent()

        # Execute query
        state = {
            "input": request.query,
            "context": request.context or {},
            "history": []
        }

        result_state = await agent.execute(state)
        result = result_state.get('result', {})

        return QueryResponse(
            response=result.get('response', 'No response generated'),
            query=request.query,
            timestamp=result.get('timestamp', datetime.now().isoformat())
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        safe_internal_error(e, "process query")


@router.post("/search", response_model=List[SearchResult])
async def semantic_search(request: SearchRequest) -> List[SearchResult]:
    """
    Search feature specs using semantic similarity.

    Uses pgvector for embedding-based search across all feature specifications.
    Falls back to text search if embeddings are unavailable.
    """
    try:
        agent = await get_orchestrator_agent()

        results = await agent.search_specs(
            query=request.query,
            limit=request.limit,
            category=request.category,
            threshold=request.threshold
        )

        return [
            SearchResult(
                name=r['name'],
                category=r['category'],
                purpose=r.get('purpose', ''),
                similarity_score=r.get('similarity_score')
            )
            for r in results
        ]

    except Exception as e:
        logger.error(f"Search error: {e}")
        safe_internal_error(e, "semantic search")


@router.get("/features", response_model=List[FeatureSummary])
async def list_features(
    category: Optional[str] = Query(None, description="Filter by category")
) -> List[FeatureSummary]:
    """
    List all features, optionally filtered by category.

    Categories include: core, trading, analysis, sports, monitoring,
    research, management, code, backend, frontend, integration, database
    """
    try:
        agent = await get_orchestrator_agent()
        features = await agent.list_features(category)

        return [
            FeatureSummary(
                name=f['name'],
                category=f['category'],
                purpose=f.get('purpose', ''),
                rating=f.get('rating')
            )
            for f in features
        ]

    except Exception as e:
        logger.error(f"List features error: {e}")
        safe_internal_error(e, "list features")


@router.get("/features/{feature_name}")
async def get_feature_details(feature_name: str) -> Dict[str, Any]:
    """
    Get complete details for a specific feature.

    Returns full specification including:
    - Purpose and how it works
    - Source files
    - Dependencies
    - API endpoints
    - Database tables
    - Integrations
    - Efficiency rating
    - Known issues
    - Enhancement opportunities
    """
    try:
        agent = await get_orchestrator_agent()
        feature = await agent.get_feature_by_name(feature_name)

        if not feature:
            raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found")

        from dataclasses import asdict
        return asdict(feature)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get feature error: {e}")
        safe_internal_error(e, "get feature details")


@router.post("/dependencies", response_model=DependencyResponse)
async def analyze_dependencies(request: DependencyRequest) -> DependencyResponse:
    """
    Analyze dependencies for a feature.

    Returns upstream (what this feature depends on) and/or downstream
    (what depends on this feature) dependencies.
    """
    try:
        agent = await get_orchestrator_agent()

        deps = await agent.find_dependencies(
            feature_name=request.feature_name,
            direction=request.direction,
            max_depth=request.max_depth
        )

        if 'error' in deps:
            raise HTTPException(status_code=404, detail=deps['error'])

        upstream = None
        downstream = None

        if 'upstream' in deps:
            upstream = [
                DependencyNode(
                    feature_name=d.feature_name,
                    category=d.category,
                    dependency_type=d.dependency_type,
                    is_critical=d.is_critical,
                    depth=d.depth
                )
                for d in deps['upstream']
            ]

        if 'downstream' in deps:
            downstream = [
                DependencyNode(
                    feature_name=d.feature_name,
                    category=d.category,
                    dependency_type=d.dependency_type,
                    is_critical=d.is_critical,
                    depth=d.depth
                )
                for d in deps['downstream']
            ]

        return DependencyResponse(
            feature=request.feature_name,
            upstream=upstream,
            downstream=downstream
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dependency analysis error: {e}")
        safe_internal_error(e, "analyze dependencies")


@router.post("/efficiency-gaps", response_model=List[EfficiencyGap])
async def find_efficiency_gaps(request: EfficiencyGapRequest) -> List[EfficiencyGap]:
    """
    Find features with efficiency below threshold.

    Useful for identifying areas that need improvement.

    Dimensions that can be filtered:
    - code_completeness
    - test_coverage
    - performance
    - error_handling
    - documentation_quality
    - maintainability
    - dependency_health
    """
    try:
        agent = await get_orchestrator_agent()

        gaps = await agent.find_efficiency_gaps(
            threshold=request.threshold,
            category=request.category,
            dimension=request.dimension
        )

        return [
            EfficiencyGap(
                feature_name=g.feature_name,
                category=g.category,
                overall_rating=g.overall_rating,
                low_dimensions=g.low_dimensions,
                quick_wins=g.quick_wins
            )
            for g in gaps
        ]

    except Exception as e:
        logger.error(f"Efficiency gaps error: {e}")
        safe_internal_error(e, "find efficiency gaps")


@router.post("/impact", response_model=ImpactResponse)
async def analyze_impact(request: ImpactRequest) -> ImpactResponse:
    """
    Analyze the impact of changing a feature.

    Helps understand what might break and what needs to be tested
    when modifying a feature.
    """
    try:
        agent = await get_orchestrator_agent()

        impact = await agent.analyze_impact(
            feature_name=request.feature_name,
            change_description=request.change_description
        )

        return ImpactResponse(
            affected_features=impact.affected_features,
            breaking_changes=impact.breaking_changes,
            test_impact=impact.test_impact,
            recommended_actions=impact.recommended_actions,
            risk_level=impact.risk_level
        )

    except Exception as e:
        logger.error(f"Impact analysis error: {e}")
        safe_internal_error(e, "analyze impact")


@router.post("/enhancements")
async def suggest_enhancements(request: EnhancementRequest) -> Dict[str, Any]:
    """
    Get enhancement suggestions for a feature.

    Returns proposed improvements based on efficiency analysis
    and best practices.
    """
    try:
        agent = await get_orchestrator_agent()
        feature = await agent.get_feature_by_name(request.feature_name)

        if not feature:
            raise HTTPException(status_code=404, detail=f"Feature '{request.feature_name}' not found")

        return {
            "feature": request.feature_name,
            "enhancements": feature.enhancements,
            "efficiency_rating": feature.efficiency_rating,
            "known_issues": feature.known_issues
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhancement suggestions error: {e}")
        safe_internal_error(e, "suggest enhancements")


@router.get("/categories", response_model=List[CategorySummary])
async def get_category_summary() -> List[CategorySummary]:
    """
    Get summary statistics for each category.

    Returns feature counts, average ratings, and attention metrics
    for each feature category.
    """
    try:
        agent = await get_orchestrator_agent()
        summary = await agent.get_category_summary()

        return [
            CategorySummary(
                category=cat,
                feature_count=data['feature_count'],
                avg_rating=data['avg_rating'],
                features_needing_attention=data['features_needing_attention']
            )
            for cat, data in summary.items()
        ]

    except Exception as e:
        logger.error(f"Category summary error: {e}")
        safe_internal_error(e, "get category summary")


@router.get("/attention-needed")
async def get_features_needing_attention() -> List[Dict[str, Any]]:
    """
    Get features that need attention.

    Returns features with low efficiency scores or critical issues
    that should be prioritized for improvement.
    """
    try:
        agent = await get_orchestrator_agent()
        features = await agent.get_features_needing_attention()
        return features

    except Exception as e:
        logger.error(f"Attention needed error: {e}")
        safe_internal_error(e, "get features needing attention")


@router.get("/agents", response_model=List[AgentInfo])
async def list_agents(
    category: Optional[str] = Query(None, description="Filter by category")
) -> List[AgentInfo]:
    """
    Get information about all registered agents.

    Returns agent metadata including capabilities and source files.
    """
    try:
        agent = await get_orchestrator_agent()

        # Search for agent features
        results = await agent.search_specs("agent", limit=50, category=category)

        agents = []
        for r in results:
            if 'agent' in r['name'].lower():
                feature = await agent.get_feature_by_name(r['name'])
                if feature:
                    agents.append(AgentInfo(
                        name=feature.name,
                        category=feature.category,
                        description=feature.purpose,
                        capabilities=feature.technical_details.get('capabilities', []),
                        source_file=feature.source_files[0]['file_path'] if feature.source_files else ''
                    ))

        return agents

    except Exception as e:
        logger.error(f"List agents error: {e}")
        safe_internal_error(e, "list agents")


@router.get("/integrations")
async def list_integrations() -> Dict[str, Any]:
    """
    Get summary of all external integrations.

    Returns integration usage across all features.
    """
    try:
        agent = await get_orchestrator_agent()

        # Use the database view for integration summary
        sql = "SELECT * FROM v_ava_integration_usage"
        results = await agent._execute_query(sql)

        return {
            "integrations": results,
            "total": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"List integrations error: {e}")
        safe_internal_error(e, "list integrations")


# =============================================================================
# Batch Operations
# =============================================================================

class BatchQueryRequest(BaseModel):
    """Batch query request"""
    queries: List[str] = Field(..., description="List of queries to execute")


@router.post("/batch-query")
@rate_limited(requests=3, window=60)  # 3 requests per minute - very expensive batch operation
async def batch_query(request: BatchQueryRequest) -> List[QueryResponse]:
    """
    Execute multiple queries in batch.
    Rate limited to 3 requests per minute due to high computational cost.

    Useful for running several related queries at once.
    """
    try:
        agent = await get_orchestrator_agent()
        results = []

        for query in request.queries:
            state = {
                "input": query,
                "context": {},
                "history": []
            }

            result_state = await agent.execute(state)
            result = result_state.get('result', {})

            results.append(QueryResponse(
                response=result.get('response', 'No response'),
                query=query,
                timestamp=result.get('timestamp', datetime.now().isoformat())
            ))

        return results

    except Exception as e:
        logger.error(f"Batch query error: {e}")
        safe_internal_error(e, "batch query")
