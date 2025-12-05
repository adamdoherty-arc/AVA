"""
Portfolio Router V2 - Modern, Optimized Endpoints

Features:
- All new infrastructure integrations
- WebSocket real-time updates
- Advanced risk models
- Health monitoring
- Background task management
"""

from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import json
import logging

from backend.services.portfolio_service_v2 import (
    get_portfolio_service_v2,
    PortfolioServiceV2
)
from backend.services.advanced_risk_models import (
    get_risk_models,
    get_prediction_engine,
    get_anomaly_detector,
    get_recommendation_engine,
    get_ts_forecaster,
    get_vol_predictor,
    AdvancedRiskModels,
    AIPredictionEngine,
    PortfolioAnomalyDetector,
    TradeRecommendationEngine,
    TimeSeriesForecaster,
    VolatilityPredictor
)
from backend.infrastructure.cache import get_cache
from backend.infrastructure.async_yfinance import get_async_yfinance, AsyncYFinance
from backend.infrastructure.circuit_breaker import get_all_breaker_stats
from backend.infrastructure.rate_limiter import (
    robinhood_quota,
    RateLimitExceeded,
    rate_limited,
    get_endpoint_limit
)
from backend.infrastructure.background_tasks import (
    get_task_manager,
    TaskStatus
)
from backend.infrastructure.websocket_manager import (
    get_ws_manager,
    get_position_broadcaster
)
from backend.infrastructure.database import get_database
from backend.infrastructure.errors import safe_internal_error

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/portfolio/v2",
    tags=["portfolio-v2"]
)


# =============================================================================
# Position Endpoints with Modern Infrastructure
# =============================================================================

@router.get("/positions")
async def get_positions_v2(
    force_refresh: bool = Query(False, description="Bypass cache"),
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Get all portfolio positions with caching and circuit breaker.

    Features:
    - 30-second cache TTL
    - Circuit breaker protection
    - Rate limit aware
    - Fallback to stale cache on errors
    """
    try:
        positions = await service.get_positions(force_refresh=force_refresh)
        return positions

    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {e.retry_after:.1f}s"
        )
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        safe_internal_error(e, "fetch positions")


@router.get("/positions/enriched")
async def get_enriched_positions_v2(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Get positions enriched with metadata using parallel batch fetching.

    Much faster than V1 due to parallel API calls.
    """
    try:
        return await service.get_enriched_positions()
    except Exception as e:
        logger.error(f"Error fetching enriched positions: {e}")
        safe_internal_error(e, "fetch enriched positions")


@router.post("/positions/refresh")
async def refresh_positions(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Force refresh positions and invalidate cache.
    """
    await service.invalidate_cache()
    return await service.get_positions(force_refresh=True)


# =============================================================================
# Advanced Risk Model Endpoints
# =============================================================================

@router.get("/risk/var")
async def get_var_analysis(
    method: str = Query("both", description="VaR method: parametric, monte_carlo, or both"),
    simulations: int = Query(10000, description="Monte Carlo simulations"),
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2),
    risk_models: AdvancedRiskModels = Depends(get_risk_models)
) -> Dict[str, Any]:
    """
    Calculate Value at Risk using advanced models.

    Methods:
    - parametric: Cornish-Fisher adjusted (fast)
    - monte_carlo: Simulation-based (more accurate)
    - both: Run both methods
    """
    positions = await service.get_positions()

    results = {}

    if method in ("parametric", "both"):
        var_param = risk_models.calculate_var_parametric(positions)
        results["parametric"] = {
            "var_95": var_param.var_95,
            "var_99": var_param.var_99,
            "expected_shortfall": var_param.expected_shortfall_95,
            "method": var_param.method,
            "calculation_time_ms": var_param.calculation_time_ms
        }

    if method in ("monte_carlo", "both"):
        var_mc = risk_models.calculate_var_monte_carlo(
            positions,
            num_simulations=simulations
        )
        results["monte_carlo"] = {
            "var_95": var_mc.var_95,
            "var_99": var_mc.var_99,
            "expected_shortfall": var_mc.expected_shortfall_95,
            "method": var_mc.method,
            "scenarios_run": var_mc.scenarios_run,
            "calculation_time_ms": var_mc.calculation_time_ms
        }

    return {
        "var_analysis": results,
        "generated_at": datetime.now().isoformat()
    }


@router.get("/risk/stress-test")
async def run_stress_tests(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2),
    risk_models: AdvancedRiskModels = Depends(get_risk_models)
) -> Dict[str, Any]:
    """
    Run predefined stress test scenarios.

    Scenarios include:
    - Market Crash (2008-style)
    - Flash Crash
    - Normal Correction
    - VIX Spike
    - Strong Rally
    """
    positions = await service.get_positions()
    results = risk_models.run_stress_tests(positions)

    return {
        "stress_tests": [
            {
                "scenario": r.scenario_name,
                "market_move_pct": r.market_move * 100,
                "iv_change_pct": r.volatility_change * 100,
                "portfolio_impact": r.portfolio_impact,
                "portfolio_impact_pct": r.portfolio_impact_pct,
                "positions_at_risk": r.positions_at_risk,
                "worst_position": r.worst_position,
                "worst_loss": r.worst_position_loss
            }
            for r in results
        ],
        "generated_at": datetime.now().isoformat()
    }


@router.get("/risk/pnl-projection")
async def project_pnl(
    underlying_move: float = Query(0.0, description="Expected % move (e.g., 0.05 for 5% up)"),
    iv_change: float = Query(0.0, description="Expected IV change (e.g., -0.10 for 10% drop)"),
    days: int = Query(1, description="Days forward to project"),
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2),
    risk_models: AdvancedRiskModels = Depends(get_risk_models)
) -> Dict[str, Any]:
    """
    Project P/L based on hypothetical market scenarios.

    Use this to answer "what if" questions about your portfolio.
    """
    positions = await service.get_positions()
    projection = risk_models.project_pnl_greeks(
        positions,
        underlying_move_pct=underlying_move,
        iv_change_pct=iv_change,
        days_forward=days
    )

    return {
        "projection": {
            "delta_pnl": projection.delta_pnl,
            "gamma_pnl": projection.gamma_pnl,
            "theta_pnl": projection.theta_pnl,
            "vega_pnl": projection.vega_pnl,
            "total_pnl": projection.total_pnl,
            "underlying_move_pct": projection.underlying_move * 100,
            "iv_change_pct": projection.iv_change * 100,
            "days_forward": projection.days_forward
        },
        "generated_at": datetime.now().isoformat()
    }


@router.get("/risk/max-loss")
async def get_max_loss(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2),
    risk_models: AdvancedRiskModels = Depends(get_risk_models)
) -> Dict[str, Any]:
    """
    Calculate theoretical maximum loss for the portfolio.
    """
    positions = await service.get_positions()
    return risk_models.calculate_max_loss(positions)


# =============================================================================
# WebSocket Real-Time Updates
# =============================================================================

@router.websocket("/ws/positions")
async def websocket_positions(
    websocket: WebSocket,
    user_id: Optional[str] = None
):
    """
    WebSocket endpoint for real-time position updates.

    Clients receive position updates every 5 seconds while connected.

    Message format:
    {
        "type": "positions_update",
        "data": { ... positions ... },
        "timestamp": "2024-01-15T10:30:00"
    }
    """
    ws_manager = get_ws_manager()

    conn_info = await ws_manager.connect(
        websocket,
        user_id=user_id,
        rooms={"positions"}
    )

    try:
        service = get_portfolio_service_v2()

        # Send initial positions
        positions = await service.get_positions()
        await ws_manager.send_personal(websocket, {
            "type": "initial_positions",
            "data": positions,
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive and send updates
        while True:
            try:
                # Wait for client message (heartbeat or commands)
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Handle heartbeat
                if message == "ping":
                    await ws_manager.handle_heartbeat(websocket)
                    await websocket.send_text("pong")

                # Handle refresh request
                elif message == "refresh":
                    positions = await service.get_positions(force_refresh=True)
                    await ws_manager.send_personal(websocket, {
                        "type": "positions_update",
                        "data": positions,
                        "timestamp": datetime.now().isoformat()
                    })

            except asyncio.TimeoutError:
                # Send periodic update
                positions = await service.get_positions()
                await ws_manager.send_personal(websocket, {
                    "type": "positions_update",
                    "data": positions,
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)


@router.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    user_id: Optional[str] = None
):
    """
    WebSocket endpoint for real-time alert notifications.
    """
    ws_manager = get_ws_manager()

    await ws_manager.connect(
        websocket,
        user_id=user_id,
        rooms={"alerts"}
    )

    try:
        while True:
            message = await websocket.receive_text()

            if message == "ping":
                await ws_manager.handle_heartbeat(websocket)
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket)


# =============================================================================
# Background Task Endpoints
# =============================================================================

@router.post("/tasks/analyze-portfolio")
async def start_portfolio_analysis(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Start a background portfolio analysis task.

    Returns immediately with a task_id for tracking.
    """
    task_manager = get_task_manager()

    async def run_analysis():
        # Get positions
        positions = await service.get_positions()

        # Run all analytics
        risk_models = get_risk_models()
        var_result = risk_models.calculate_var_monte_carlo(positions, num_simulations=50000)
        stress_results = risk_models.run_stress_tests(positions)
        max_loss = risk_models.calculate_max_loss(positions)

        return {
            "positions": positions,
            "var": {
                "var_95": var_result.var_95,
                "var_99": var_result.var_99,
                "expected_shortfall": var_result.expected_shortfall_95
            },
            "stress_tests": [
                {"scenario": r.scenario_name, "impact": r.portfolio_impact}
                for r in stress_results
            ],
            "max_loss": max_loss
        }

    task_id = await task_manager.submit(
        run_analysis,
        metadata={"type": "portfolio_analysis"}
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Analysis started. Poll /tasks/{task_id} for results."
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status and result of a background task.
    """
    task_manager = get_task_manager()
    result = await task_manager.get_status(task_id)

    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    return result.to_dict()


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel a running background task.
    """
    task_manager = get_task_manager()
    cancelled = await task_manager.cancel(task_id)

    if cancelled:
        return {"message": "Task cancellation requested"}
    else:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")


# =============================================================================
# Health & Monitoring Endpoints
# =============================================================================

@router.get("/health")
async def get_health_status(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Comprehensive health check for portfolio service.

    Returns status of:
    - Robinhood connection
    - Circuit breakers
    - API quota
    - Cache
    - Database
    - WebSocket connections
    """
    # Service health
    service_health = await service.get_health()

    # Database health
    try:
        db = await get_database()
        db_health = await db.health_check()
    except Exception as e:
        db_health = {"healthy": False, "error": str(e)}

    # Cache health
    cache = get_cache()
    cache_health = {
        "healthy": await cache.health_check(),
        "stats": cache.get_stats()
    }

    # WebSocket stats
    ws_manager = get_ws_manager()
    ws_stats = ws_manager.get_stats()

    # Task manager stats
    task_manager = get_task_manager()
    task_stats = task_manager.get_stats()

    return {
        "status": "healthy" if db_health.get("healthy") and service_health.get("logged_in", False) else "degraded",
        "components": {
            "robinhood": service_health,
            "database": db_health,
            "cache": cache_health,
            "circuit_breakers": get_all_breaker_stats(),
            "websockets": ws_stats,
            "background_tasks": task_stats
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/metrics")
async def get_metrics(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Get performance metrics and statistics.
    """
    # Quota status
    quota = await robinhood_quota.get_status()

    # Cache stats
    cache = get_cache()
    cache_stats = cache.get_stats()

    # Circuit breaker stats
    breaker_stats = get_all_breaker_stats()

    return {
        "api_quota": quota,
        "cache": cache_stats,
        "circuit_breakers": breaker_stats,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/cache/invalidate")
async def invalidate_cache(
    pattern: str = Query("*", description="Cache key pattern to invalidate"),
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Manually invalidate cache entries.
    """
    cache = get_cache()
    deleted = await cache.invalidate_pattern(pattern)

    return {
        "message": f"Invalidated {deleted} cache entries",
        "pattern": pattern
    }


# =============================================================================
# AI-Powered Analysis Endpoints
# =============================================================================

@router.get("/ai/anomalies")
async def detect_anomalies(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2),
    detector: PortfolioAnomalyDetector = Depends(get_anomaly_detector)
) -> Dict[str, Any]:
    """
    Detect portfolio anomalies using AI.

    Checks for:
    - Unusual Greeks values
    - Concentration risk
    - Theta imbalance
    - Unusual volatility patterns
    """
    positions = await service.get_positions()
    anomalies = detector.analyze_portfolio(positions)

    return {
        "anomalies": [
            {
                "type": a.anomaly_type.value,
                "severity": a.severity,
                "description": a.description,
                "affected_positions": a.affected_positions,
                "metric_value": a.metric_value,
                "threshold": a.threshold,
                "recommendation": a.recommendation,
                "detected_at": a.detected_at.isoformat()
            }
            for a in anomalies
        ],
        "total_count": len(anomalies),
        "critical_count": sum(1 for a in anomalies if a.severity == "critical"),
        "generated_at": datetime.now().isoformat()
    }


@router.get("/ai/risk-score")
async def get_risk_score(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2),
    detector: PortfolioAnomalyDetector = Depends(get_anomaly_detector)
) -> Dict[str, Any]:
    """
    Get overall portfolio risk score (0-100).

    Score breakdown:
    - 0-30: Low risk
    - 30-50: Moderate risk
    - 50-70: Elevated risk
    - 70-100: High risk
    """
    positions = await service.get_positions()
    risk_score = detector.get_risk_score(positions)

    return {
        "risk_score": risk_score,
        "generated_at": datetime.now().isoformat()
    }


@router.get("/ai/recommendations")
async def get_trade_recommendations(
    risk_tolerance: str = Query(
        "moderate",
        description="Risk tolerance: conservative, moderate, aggressive"
    ),
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2),
    engine: TradeRecommendationEngine = Depends(get_recommendation_engine)
) -> Dict[str, Any]:
    """
    Get AI-powered trade recommendations.

    Analyzes your portfolio and provides actionable suggestions based on:
    - Current risk profile
    - Detected anomalies
    - Price predictions
    - Greeks optimization opportunities
    """
    positions = await service.get_positions()
    recommendations = engine.generate_recommendations(
        positions,
        risk_tolerance=risk_tolerance
    )

    return {
        "recommendations": recommendations,
        "risk_tolerance": risk_tolerance,
        "recommendation_count": len(recommendations),
        "generated_at": datetime.now().isoformat()
    }


@router.get("/ai/predictions/{symbol}")
async def get_price_prediction(
    symbol: str,
    engine: AIPredictionEngine = Depends(get_prediction_engine)
) -> Dict[str, Any]:
    """
    Get AI price prediction for a symbol.

    Uses ensemble model with:
    - Mean reversion signals
    - Momentum indicators
    - RSI analysis
    - Volatility adjustment
    """
    # Get stock data using async wrapper (non-blocking)
    yf_client = get_async_yfinance()
    try:
        stock_data = await yf_client.get_stock_data(symbol, period="3mo")
        current_price = stock_data.current_price
        historical_prices = stock_data.historical_prices
        iv_estimate = stock_data.iv_estimate
    except ImportError:
        raise HTTPException(status_code=500, detail="Market data service unavailable")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol}")
    except Exception as e:
        safe_internal_error(e, "fetch stock data")

    prediction = engine.predict_price(
        symbol=symbol.upper(),
        current_price=current_price,
        historical_prices=historical_prices,
        iv=iv_estimate / 100
    )

    return {
        "prediction": {
            "symbol": prediction.symbol,
            "current_price": prediction.current_price,
            "predicted_1d": prediction.predicted_price_1d,
            "predicted_5d": prediction.predicted_price_5d,
            "predicted_30d": prediction.predicted_price_30d,
            "range_low": prediction.prediction_range_low,
            "range_high": prediction.prediction_range_high,
            "confidence": prediction.confidence.value,
            "model": prediction.model_used,
            "features": prediction.features_used
        },
        "generated_at": datetime.now().isoformat()
    }


@router.get("/ai/trend/{symbol}")
async def get_trend_signal(
    symbol: str,
    engine: AIPredictionEngine = Depends(get_prediction_engine)
) -> Dict[str, Any]:
    """
    Get technical trend signal for a symbol.

    Returns bullish/bearish/neutral with strength (0-1).
    """
    yf_client = get_async_yfinance()
    try:
        stock_data = await yf_client.get_stock_data(symbol, period="3mo")
        historical_prices = stock_data.historical_prices
    except ImportError:
        raise HTTPException(status_code=500, detail="Market data service unavailable")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol}")
    except Exception as e:
        safe_internal_error(e, "fetch trend data")

    signal = engine.get_trend_signal(
        symbol=symbol.upper(),
        prices=historical_prices,
        timeframe="daily"
    )

    return {
        "trend": {
            "symbol": symbol.upper(),
            "signal": signal.signal_type,
            "strength": signal.strength,
            "indicators": signal.indicators,
            "timeframe": signal.timeframe
        },
        "generated_at": datetime.now().isoformat()
    }


@router.get("/ai/comprehensive")
async def get_comprehensive_ai_analysis(
    risk_tolerance: str = Query("moderate"),
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2),
    risk_models: AdvancedRiskModels = Depends(get_risk_models),
    anomaly_detector: PortfolioAnomalyDetector = Depends(get_anomaly_detector),
    recommendation_engine: TradeRecommendationEngine = Depends(
        get_recommendation_engine
    )
) -> Dict[str, Any]:
    """
    Get comprehensive AI analysis of portfolio.

    Combines:
    - Risk metrics (VaR, stress tests)
    - Anomaly detection
    - Risk scoring
    - Trade recommendations

    This is the most complete analysis endpoint.
    """
    positions = await service.get_positions()

    # Run all analyses in parallel conceptually
    var_result = risk_models.calculate_var_monte_carlo(
        positions, num_simulations=50000
    )
    stress_results = risk_models.run_stress_tests(positions)
    max_loss = risk_models.calculate_max_loss(positions)
    anomalies = anomaly_detector.analyze_portfolio(positions)
    risk_score = anomaly_detector.get_risk_score(positions)
    recommendations = recommendation_engine.generate_recommendations(
        positions, risk_tolerance
    )

    return {
        "portfolio_summary": positions.get("summary", {}),
        "risk_metrics": {
            "var_95": var_result.var_95,
            "var_99": var_result.var_99,
            "expected_shortfall": var_result.expected_shortfall_95,
            "max_theoretical_loss": max_loss.get("total_max_loss", 0),
            "percentile_distribution": var_result.percentile_distribution
        },
        "stress_tests": [
            {
                "scenario": r.scenario_name,
                "impact": r.portfolio_impact,
                "impact_pct": r.portfolio_impact_pct
            }
            for r in stress_results
        ],
        "risk_score": risk_score,
        "anomalies": [
            {
                "type": a.anomaly_type.value,
                "severity": a.severity,
                "description": a.description,
                "recommendation": a.recommendation
            }
            for a in anomalies
        ],
        "recommendations": recommendations[:5],  # Top 5
        "analysis_metadata": {
            "simulations_run": var_result.scenarios_run,
            "calculation_time_ms": var_result.calculation_time_ms,
            "risk_tolerance": risk_tolerance
        },
        "generated_at": datetime.now().isoformat()
    }


# =============================================================================
# Streaming Endpoint
# =============================================================================

@router.get("/stream/portfolio")
async def stream_portfolio_updates(
    interval: int = Query(5, description="Update interval in seconds"),
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
):
    """
    Stream portfolio updates using Server-Sent Events (SSE).

    Alternative to WebSocket for simpler client implementations.
    """
    async def event_generator():
        try:
            while True:
                positions = await service.get_positions()

                yield f"data: {json.dumps(positions, default=str)}\n\n"

                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =============================================================================
# Data Validation & Audit Trail Endpoints
# =============================================================================

@router.get("/validation/details")
async def get_validation_details(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Get detailed validation results from the last portfolio fetch.

    Returns:
    - Full validation result with all issues
    - Quality score (excellent, good, warning, critical)
    - Validation statistics
    """
    return service.get_validation_details()


@router.get("/validation/run")
async def run_validation(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Force a fresh portfolio fetch with full validation.

    Returns positions with embedded validation result.
    """
    positions = await service.get_positions(force_refresh=True)
    validation = service.get_validation_details()

    return {
        "positions": positions,
        "validation": validation,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/audit/history")
async def get_audit_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    event_type: Optional[str] = Query(
        None,
        description="Filter by event type (position_added, position_removed, position_change)"
    ),
    limit: int = Query(100, description="Max entries to return"),
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Get audit trail history for position changes.

    Tracks:
    - Positions added/removed
    - Significant P/L changes
    - Price movements
    """
    return service.get_audit_history(
        symbol=symbol,
        event_type=event_type,
        limit=limit
    )


@router.post("/audit/validate-and-track")
async def validate_and_audit(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Full validation and audit workflow.

    Fetches fresh data, validates it, and tracks changes from previous state.
    Use this for comprehensive portfolio monitoring.
    """
    return await service.validate_and_audit(force_refresh=True)


@router.get("/data-quality/summary")
async def get_data_quality_summary(
    service: PortfolioServiceV2 = Depends(get_portfolio_service_v2)
) -> Dict[str, Any]:
    """
    Get a high-level data quality summary.

    Quick overview of:
    - Overall data quality grade
    - Number of validation issues
    - Last validation timestamp
    """
    validation = service.get_validation_details()

    if validation.get("status") == "no_validation_run":
        return {
            "quality_grade": "unknown",
            "message": "No validation has been run yet",
            "recommendation": "Call GET /api/portfolio/v2/positions to run validation"
        }

    result = validation.get("result", {})

    return {
        "quality_grade": result.get("quality", "unknown").upper(),
        "is_valid": result.get("valid", False),
        "error_count": result.get("error_count", 0),
        "warning_count": result.get("warning_count", 0),
        "last_checked": result.get("checked_at"),
        "data_freshness_seconds": result.get("data_freshness_seconds"),
        "top_issues": result.get("issues", [])[:3],  # Top 3 issues only
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# Time Series Forecasting Endpoints
# =============================================================================

@router.get("/forecast/price/{symbol}")
async def get_price_forecast(
    symbol: str,
    horizon: int = Query(5, description="Days to forecast ahead (1-30)"),
    method: str = Query("holt", description="Method: holt or arima"),
    forecaster: TimeSeriesForecaster = Depends(get_ts_forecaster)
) -> Dict[str, Any]:
    """
    Get time series price forecast for a symbol.

    Uses Holt's Linear Trend Method or ARIMA approximation
    for short-term price forecasting.

    Returns:
    - Forecasted prices for each day in horizon
    - Confidence intervals
    - Trend direction
    - Model accuracy metrics (MAE, MAPE)
    """
    yf_client = get_async_yfinance()
    try:
        stock_data = await yf_client.get_stock_data(symbol, period="3mo")
        prices = stock_data.historical_prices

        # Limit horizon
        horizon = min(max(1, horizon), 30)

        if method == "arima":
            forecast = forecaster.forecast_with_arima_approximation(
                symbol=symbol.upper(),
                prices=prices,
                horizon=horizon
            )
        else:
            forecast = forecaster.forecast(
                symbol=symbol.upper(),
                prices=prices,
                horizon=horizon
            )

        return {
            "forecast": {
                "symbol": forecast.symbol,
                "horizon_days": forecast.forecast_horizon,
                "forecasted_prices": forecast.forecasted_values,
                "confidence_lower": forecast.confidence_interval_lower,
                "confidence_upper": forecast.confidence_interval_upper,
                "trend": forecast.trend,
                "seasonality_detected": forecast.seasonality_detected,
                "model": forecast.model_type,
                "accuracy": {
                    "mae": forecast.mae,
                    "mape": forecast.mape
                }
            },
            "current_price": round(prices[-1], 2),
            "generated_at": datetime.now().isoformat()
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="yfinance not available")
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        safe_internal_error(e, "price forecast")


# =============================================================================
# Volatility Prediction Endpoints
# =============================================================================

@router.get("/forecast/volatility/{symbol}")
async def get_volatility_forecast(
    symbol: str,
    predictor: VolatilityPredictor = Depends(get_vol_predictor)
) -> Dict[str, Any]:
    """
    Get volatility forecast using GARCH-like model.

    Returns:
    - Current realized and implied volatility
    - Forecasted volatility at 5, 10, 21 day horizons
    - Volatility regime (low/normal/elevated/extreme)
    - IV-RV spread analysis
    """
    yf_client = get_async_yfinance()
    try:
        stock_data = await yf_client.get_stock_data(symbol, period="6mo")
        prices = stock_data.historical_prices
        iv_estimate = stock_data.iv_estimate / 100  # Convert from percentage

        forecast = predictor.predict_volatility(
            symbol=symbol.upper(),
            prices=prices,
            current_iv=iv_estimate
        )

        return {
            "volatility_forecast": {
                "symbol": forecast.symbol,
                "current_realized_vol": forecast.current_realized_vol,
                "current_iv": forecast.current_iv,
                "forecasts": {
                    "5d": forecast.forecasted_vol_5d,
                    "10d": forecast.forecasted_vol_10d,
                    "21d": forecast.forecasted_vol_21d
                },
                "regime": forecast.vol_regime,
                "trend": forecast.vol_trend,
                "iv_rv_spread_pct": forecast.iv_rv_spread,
                "garch_persistence": forecast.garch_persistence,
                "confidence": forecast.confidence.value
            },
            "generated_at": datetime.now().isoformat()
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="yfinance not available")
    except Exception as e:
        logger.error(f"Volatility forecast error: {e}")
        safe_internal_error(e, "volatility forecast")


@router.get("/forecast/vol-surface/{symbol}")
async def get_volatility_surface(
    symbol: str,
    days_ahead: int = Query(30, description="Max days to forecast"),
    predictor: VolatilityPredictor = Depends(get_vol_predictor)
) -> Dict[str, Any]:
    """
    Get volatility term structure (surface).

    Returns expected volatility at multiple time horizons
    to understand how volatility is expected to evolve.
    """
    yf_client = get_async_yfinance()
    try:
        stock_data = await yf_client.get_stock_data(symbol, period="6mo")
        prices = stock_data.historical_prices
        iv_estimate = stock_data.iv_estimate / 100  # Convert from percentage

        surface = predictor.estimate_vol_surface(
            symbol=symbol.upper(),
            prices=prices,
            current_iv=iv_estimate,
            days_ahead=days_ahead
        )

        return surface

    except ImportError:
        raise HTTPException(status_code=500, detail="yfinance not available")
    except Exception as e:
        logger.error(f"Vol surface error: {e}")
        safe_internal_error(e, "volatility surface")


@router.get("/ai/full-analysis/{symbol}")
async def get_full_symbol_analysis(
    symbol: str,
    engine: AIPredictionEngine = Depends(get_prediction_engine),
    forecaster: TimeSeriesForecaster = Depends(get_ts_forecaster),
    vol_predictor: VolatilityPredictor = Depends(get_vol_predictor)
) -> Dict[str, Any]:
    """
    Comprehensive AI analysis for a single symbol.

    Combines:
    - Ensemble price prediction
    - Time series forecast
    - Volatility prediction
    - Trend signals

    This is the most complete single-symbol analysis.
    """
    yf_client = get_async_yfinance()
    try:
        stock_data = await yf_client.get_stock_data(symbol, period="6mo")
        prices = stock_data.historical_prices
        current_price = stock_data.current_price
        iv_estimate = stock_data.iv_estimate / 100  # Convert from percentage

        # Run all analyses
        price_prediction = engine.predict_price(
            symbol=symbol.upper(),
            current_price=current_price,
            historical_prices=prices,
            iv=iv_estimate
        )

        ts_forecast = forecaster.forecast(
            symbol=symbol.upper(),
            prices=prices,
            horizon=10
        )

        vol_forecast = vol_predictor.predict_volatility(
            symbol=symbol.upper(),
            prices=prices,
            current_iv=iv_estimate
        )

        trend = engine.get_trend_signal(
            symbol=symbol.upper(),
            prices=prices
        )

        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "price_prediction": {
                "1d": price_prediction.predicted_price_1d,
                "5d": price_prediction.predicted_price_5d,
                "30d": price_prediction.predicted_price_30d,
                "range": {
                    "low": price_prediction.prediction_range_low,
                    "high": price_prediction.prediction_range_high
                },
                "confidence": price_prediction.confidence.value
            },
            "time_series_forecast": {
                "prices": ts_forecast.forecasted_values,
                "trend": ts_forecast.trend,
                "model": ts_forecast.model_type,
                "accuracy_mape": ts_forecast.mape
            },
            "volatility": {
                "current_rv": vol_forecast.current_realized_vol,
                "forecast_21d": vol_forecast.forecasted_vol_21d,
                "regime": vol_forecast.vol_regime,
                "trend": vol_forecast.vol_trend
            },
            "trend_signal": {
                "direction": trend.signal_type,
                "strength": trend.strength,
                "indicators": trend.indicators
            },
            "generated_at": datetime.now().isoformat()
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="yfinance not available")
    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        safe_internal_error(e, "full symbol analysis")
