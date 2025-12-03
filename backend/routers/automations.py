"""
Automations Router - Developer Console API
==========================================

REST API for managing automation state (enable/disable),
querying execution history, triggering manual runs, and
monitoring automation health.

Endpoints:
- GET    /api/automations           - List all automations
- GET    /api/automations/dashboard - Get dashboard stats
- GET    /api/automations/categories - Get automation categories
- GET    /api/automations/{name}    - Get single automation
- PATCH  /api/automations/{name}    - Toggle enabled state
- POST   /api/automations/{name}/run - Manually trigger
- GET    /api/automations/{name}/history - Get execution history
- POST   /api/automations/bulk      - Bulk enable/disable

Author: AVA Trading Platform
Created: 2025-11-28
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import logging

# Import the control service
import sys
sys.path.insert(0, '/Users/adam/code/AVA')
from src.services.automation_control_service import (
    get_automation_control_service,
    ExecutionStatus
)
from src.services.automation_ai_analyzer import get_automation_ai_analyzer

router = APIRouter(prefix="/api/automations", tags=["automations"])
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class AutomationStateUpdate(BaseModel):
    """Request model for updating automation state."""
    enabled: bool
    revoke_running: bool = False
    reason: Optional[str] = None


class BulkStateUpdate(BaseModel):
    """Request model for bulk state updates."""
    automation_names: List[str]
    enabled: bool
    reason: Optional[str] = None


class TriggerRequest(BaseModel):
    """Request model for manual trigger."""
    pass  # No parameters needed currently


class AutomationResponse(BaseModel):
    """Response model for automation details."""
    id: int
    name: str
    display_name: str
    automation_type: str
    celery_task_name: Optional[str]
    schedule_type: Optional[str]
    schedule_config: Optional[dict]
    schedule_display: Optional[str]
    queue: Optional[str]
    category: str
    description: Optional[str]
    is_enabled: bool
    timeout_seconds: Optional[int]
    last_run_status: Optional[str]
    last_run_at: Optional[datetime]
    last_duration_seconds: Optional[float]
    success_rate_24h: Optional[float]
    total_runs_24h: Optional[int]


class ExecutionLogResponse(BaseModel):
    """Response model for execution log entry."""
    id: int
    automation_name: str
    display_name: str
    celery_task_id: Optional[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    result: Optional[dict]
    error_message: Optional[str]
    records_processed: Optional[int]
    triggered_by: str


# ============================================================================
# Endpoints
# ============================================================================

@router.get("")
async def list_automations(
    category: Optional[str] = Query(None, description="Filter by category"),
    enabled: Optional[bool] = Query(None, description="Filter by enabled state"),
    search: Optional[str] = Query(None, description="Search by name")
) -> Dict[str, Any]:
    """
    List all registered automations with optional filtering.

    Returns automations grouped by category with current status.
    """
    try:
        control = get_automation_control_service()
        automations = control.get_all_automations(category=category)

        # Apply additional filters
        if enabled is not None:
            automations = [a for a in automations if a.get('is_enabled') == enabled]

        if search:
            search_lower = search.lower()
            automations = [
                a for a in automations
                if search_lower in a.get('name', '').lower()
                or search_lower in a.get('display_name', '').lower()
            ]

        # Group by category
        grouped = {}
        for automation in automations:
            cat = automation.get('category', 'other')
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(automation)

        return {
            "automations": automations,
            "grouped": grouped,
            "total": len(automations),
            "enabled_count": len([a for a in automations if a.get('is_enabled')]),
            "disabled_count": len([a for a in automations if not a.get('is_enabled')])
        }

    except Exception as e:
        logger.error(f"Error listing automations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours")
) -> Dict[str, Any]:
    """
    Get dashboard statistics and overview.

    Returns:
    - Automation counts (total, enabled, disabled)
    - Execution stats (success rate, failures, running)
    - Recent failures
    - Category breakdown
    """
    try:
        control = get_automation_control_service()
        stats = control.get_dashboard_stats(hours=hours)

        return {
            "status": "success",
            **stats
        }

    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_categories() -> Dict[str, Any]:
    """
    Get all unique automation categories with counts.
    """
    try:
        control = get_automation_control_service()
        categories = control.get_categories()

        # Get counts per category
        all_automations = control.get_all_automations()
        category_counts = {}
        for cat in categories:
            cat_automations = [a for a in all_automations if a.get('category') == cat]
            category_counts[cat] = {
                "total": len(cat_automations),
                "enabled": len([a for a in cat_automations if a.get('is_enabled')]),
                "disabled": len([a for a in cat_automations if not a.get('is_enabled')])
            }

        return {
            "categories": categories,
            "counts": category_counts
        }

    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_all_history(
    status: Optional[str] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours")
) -> Dict[str, Any]:
    """
    Get execution history across all automations.
    """
    try:
        control = get_automation_control_service()
        since = datetime.now() - timedelta(hours=hours)

        executions = control.get_execution_history(
            status=status,
            limit=limit,
            offset=offset,
            since=since
        )

        # Filter by category if specified
        if category:
            executions = [e for e in executions if e.get('category') == category]

        # Calculate stats
        stats = {
            "total": len(executions),
            "success": len([e for e in executions if e.get('status') == 'success']),
            "failed": len([e for e in executions if e.get('status') == 'failed']),
            "skipped": len([e for e in executions if e.get('status') == 'skipped']),
            "running": len([e for e in executions if e.get('status') == 'running'])
        }

        return {
            "executions": executions,
            "stats": stats,
            "limit": limit,
            "offset": offset,
            "time_window_hours": hours
        }

    except Exception as e:
        logger.error(f"Error getting execution history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{automation_name}")
async def get_automation(automation_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific automation.
    """
    try:
        control = get_automation_control_service()
        automation = control.get_automation(automation_name)

        if not automation:
            raise HTTPException(
                status_code=404,
                detail=f"Automation '{automation_name}' not found"
            )

        return {
            "status": "success",
            "automation": automation
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting automation {automation_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{automation_name}/history")
async def get_automation_history(
    automation_name: str,
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    hours: Optional[int] = Query(None, ge=1, le=720)
) -> Dict[str, Any]:
    """
    Get execution history for a specific automation.
    """
    try:
        control = get_automation_control_service()

        # Verify automation exists
        automation = control.get_automation(automation_name)
        if not automation:
            raise HTTPException(
                status_code=404,
                detail=f"Automation '{automation_name}' not found"
            )

        since = None
        if hours:
            since = datetime.now() - timedelta(hours=hours)

        executions = control.get_execution_history(
            automation_name=automation_name,
            status=status,
            limit=limit,
            offset=offset,
            since=since
        )

        return {
            "automation_name": automation_name,
            "display_name": automation.get('display_name'),
            "executions": executions,
            "total": len(executions),
            "limit": limit,
            "offset": offset
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history for {automation_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{automation_name}")
async def update_automation_state(
    automation_name: str,
    update: AutomationStateUpdate
) -> Dict[str, Any]:
    """
    Enable or disable an automation.

    Simple toggle - no confirmation required.
    """
    try:
        control = get_automation_control_service()

        success, revoked_tasks = control.set_enabled(
            automation_name=automation_name,
            enabled=update.enabled,
            revoke_running=update.revoke_running,
            changed_by="api",
            reason=update.reason
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Automation '{automation_name}' not found"
            )

        return {
            "status": "success",
            "automation_name": automation_name,
            "is_enabled": update.enabled,
            "revoked_tasks": revoked_tasks or [],
            "message": f"Automation {'enabled' if update.enabled else 'disabled'} successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating automation {automation_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{automation_name}/run")
async def trigger_automation(
    automation_name: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Manually trigger an automation to run immediately.

    Bypasses the schedule and queues the task for immediate execution.
    """
    try:
        control = get_automation_control_service()

        # Check if automation exists and is enabled
        automation = control.get_automation(automation_name)
        if not automation:
            raise HTTPException(
                status_code=404,
                detail=f"Automation '{automation_name}' not found"
            )

        if not automation.get('is_enabled'):
            raise HTTPException(
                status_code=400,
                detail=f"Automation '{automation_name}' is disabled. Enable it first."
            )

        # Trigger the task
        task_id = control.trigger_automation(automation_name)

        if not task_id:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to trigger automation '{automation_name}'"
            )

        return {
            "status": "success",
            "automation_name": automation_name,
            "task_id": task_id,
            "message": "Automation triggered successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering automation {automation_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk")
async def bulk_update_state(update: BulkStateUpdate) -> Dict[str, Any]:
    """
    Enable or disable multiple automations at once.

    Useful for the "Danger Zone" bulk operations.
    """
    try:
        control = get_automation_control_service()

        results = control.bulk_set_enabled(
            automation_names=update.automation_names,
            enabled=update.enabled,
            changed_by="api",
            reason=update.reason
        )

        return {
            "status": "success",
            "updated": results["updated"],
            "failed": results["failed"],
            "unchanged": results.get("unchanged", []),
            "message": f"{'Enabled' if update.enabled else 'Disabled'} {len(results['updated'])} automations"
        }

    except Exception as e:
        logger.error(f"Error in bulk update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/enable-all")
async def enable_all_automations() -> Dict[str, Any]:
    """
    Enable all automations.

    Danger Zone operation - enables every automation in the system.
    """
    try:
        control = get_automation_control_service()
        all_automations = control.get_all_automations()

        disabled_names = [
            a['name'] for a in all_automations
            if not a.get('is_enabled')
        ]

        if not disabled_names:
            return {
                "status": "success",
                "message": "All automations are already enabled",
                "updated": []
            }

        results = control.bulk_set_enabled(
            automation_names=disabled_names,
            enabled=True,
            changed_by="api",
            reason="Bulk enable all"
        )

        return {
            "status": "success",
            "updated": results["updated"],
            "failed": results["failed"],
            "message": f"Enabled {len(results['updated'])} automations"
        }

    except Exception as e:
        logger.error(f"Error enabling all automations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/disable-all")
async def disable_all_automations() -> Dict[str, Any]:
    """
    Disable all automations.

    Danger Zone operation - disables every automation in the system.
    Use with caution!
    """
    try:
        control = get_automation_control_service()
        all_automations = control.get_all_automations()

        enabled_names = [
            a['name'] for a in all_automations
            if a.get('is_enabled')
        ]

        if not enabled_names:
            return {
                "status": "success",
                "message": "All automations are already disabled",
                "updated": []
            }

        results = control.bulk_set_enabled(
            automation_names=enabled_names,
            enabled=False,
            changed_by="api",
            reason="Bulk disable all"
        )

        return {
            "status": "success",
            "updated": results["updated"],
            "failed": results["failed"],
            "message": f"Disabled {len(results['updated'])} automations"
        }

    except Exception as e:
        logger.error(f"Error disabling all automations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/cleanup")
async def cleanup_old_history(
    days: int = Query(30, ge=7, le=365, description="Delete logs older than N days")
) -> Dict[str, Any]:
    """
    Delete old execution history logs.

    Danger Zone operation - permanently removes old execution logs.
    """
    try:
        control = get_automation_control_service()

        # Use direct database access for cleanup
        with control._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM automation_executions
                    WHERE started_at < NOW() - INTERVAL '%s days'
                    RETURNING id
                """, (days,))
                deleted_count = cur.rowcount
                conn.commit()

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} execution logs older than {days} days"
        }

    except Exception as e:
        logger.error(f"Error cleaning up history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AI-Powered Endpoints
# ============================================================================

class AIQuestionRequest(BaseModel):
    """Request model for AI question."""
    question: str = Field(..., min_length=5, max_length=500)


class FailureAnalysisRequest(BaseModel):
    """Request model for failure analysis."""
    error_message: str
    error_traceback: Optional[str] = None


@router.post("/ai/analyze-failure/{automation_name}")
async def analyze_failure(
    automation_name: str,
    request: FailureAnalysisRequest
) -> Dict[str, Any]:
    """
    AI-powered root cause analysis for automation failures.

    Uses LLM to analyze error patterns and provide:
    - Root cause identification
    - Severity assessment
    - Immediate fix suggestions
    - Long-term solutions
    """
    try:
        analyzer = get_automation_ai_analyzer()
        control = get_automation_control_service()

        # Get recent history for context
        recent_history = control.get_execution_history(
            automation_name=automation_name,
            limit=10
        )

        result = analyzer.analyze_failure(
            automation_name=automation_name,
            error_message=request.error_message,
            error_traceback=request.error_traceback,
            recent_history=recent_history
        )

        return result

    except Exception as e:
        logger.error(f"Error in AI failure analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/health-prediction")
async def get_health_prediction(
    automation_name: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168)
) -> Dict[str, Any]:
    """
    AI-powered health prediction for automations.

    Analyzes historical patterns to predict:
    - Health status (healthy/warning/critical)
    - Failure risk for next 24 hours
    - Trend analysis (improving/stable/declining)
    - AI-generated recommendations
    """
    try:
        analyzer = get_automation_ai_analyzer()
        result = analyzer.get_health_prediction(
            automation_name=automation_name,
            hours=hours
        )
        return result

    except Exception as e:
        logger.error(f"Error in health prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/recommendations")
async def get_recommendations(
    automation_name: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Get AI-powered optimization recommendations.

    Analyzes all automations and provides:
    - Performance issues
    - Optimization suggestions
    - Priority-ranked recommendations
    """
    try:
        analyzer = get_automation_ai_analyzer()
        result = analyzer.get_optimization_recommendations(
            automation_name=automation_name
        )
        return result

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/ask")
async def ask_ai(request: AIQuestionRequest) -> Dict[str, Any]:
    """
    Ask natural language questions about automations.

    Examples:
    - "Why did sync-kalshi-markets fail?"
    - "Which automations have the lowest success rate?"
    - "What's the overall system health?"
    - "Show me failing automations in the last hour"
    """
    try:
        analyzer = get_automation_ai_analyzer()
        result = analyzer.answer_question(request.question)
        return result

    except Exception as e:
        logger.error(f"Error in AI question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/insights")
async def get_ai_insights(
    hours: int = Query(24, ge=1, le=168)
) -> Dict[str, Any]:
    """
    Get comprehensive AI insights dashboard.

    Combines multiple AI analyses into a single view:
    - Health prediction
    - Top recommendations
    - Recent failure analyses
    - System summary
    """
    try:
        analyzer = get_automation_ai_analyzer()
        control = get_automation_control_service()

        # Gather multiple insights
        health = analyzer.get_health_prediction(hours=hours)
        recommendations = analyzer.get_optimization_recommendations()

        # Get recent failures for analysis
        recent_failures = control.get_execution_history(
            status='failed',
            limit=3,
            since=datetime.now() - timedelta(hours=hours)
        )

        # Analyze top failures
        failure_analyses = []
        for failure in recent_failures[:2]:  # Limit to avoid too many LLM calls
            if failure.get('error_message'):
                analysis = analyzer.analyze_failure(
                    automation_name=failure.get('automation_name', 'unknown'),
                    error_message=failure.get('error_message', 'Unknown error'),
                    error_traceback=failure.get('error_traceback')
                )
                failure_analyses.append(analysis)

        return {
            "status": "success",
            "health_prediction": health,
            "top_recommendations": recommendations.get('recommendations', [])[:5],
            "recent_failure_analyses": failure_analyses,
            "time_window_hours": hours
        }

    except Exception as e:
        logger.error(f"Error getting AI insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))
