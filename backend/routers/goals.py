"""
Goals Router - API endpoints for income goal tracking

Provides endpoints for:
- Getting current goal progress
- Updating targets
- Viewing historical progress
- Getting actionable advice
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import logging

from src.ava.services.goal_tracking_service import (
    GoalTrackingService,
    GoalProgress,
    GoalStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/goals", tags=["goals"])


class GoalProgressResponse(BaseModel):
    """Response model for goal progress"""
    goal_id: int
    goal_name: str
    goal_type: str
    target_amount: float
    current_amount: float
    progress_percent: float
    days_elapsed: int
    days_remaining: int
    daily_target: float
    daily_actual: float
    projected_amount: float
    status: str
    premium_income: float
    realized_gains: float
    advice: List[str]
    opportunities_needed: int
    avg_premium_needed: float


class UpdateTargetRequest(BaseModel):
    """Request model for updating goal target"""
    target_amount: float


class RecordIncomeRequest(BaseModel):
    """Request model for recording manual income"""
    amount: float
    source: str
    symbol: Optional[str] = None
    notes: Optional[str] = None


@router.get("/monthly")
async def get_monthly_goal(user_id: str = "default_user"):
    """
    Get current monthly income goal progress.

    Returns comprehensive goal tracking including:
    - Current progress vs target
    - Daily/projected metrics
    - Status assessment
    - Actionable advice
    """
    try:
        service = GoalTrackingService(user_id)
        goal = service.get_monthly_income_goal()

        if not goal:
            raise HTTPException(status_code=404, detail="No active monthly goal found")

        return {
            "goal_id": goal.goal_id,
            "goal_name": goal.goal_name,
            "goal_type": goal.goal_type,
            "target_amount": goal.target_amount,
            "current_amount": round(goal.current_amount, 2),
            "progress_percent": round(goal.progress_percent, 1),
            "days_elapsed": goal.days_elapsed,
            "days_remaining": goal.days_remaining,
            "daily_target": round(goal.daily_target, 2),
            "daily_actual": round(goal.daily_actual, 2),
            "projected_amount": round(goal.projected_amount, 2),
            "status": goal.status.value,
            "status_display": goal.status.value.replace('_', ' ').title(),
            "premium_income": round(goal.premium_income, 2),
            "realized_gains": round(goal.realized_gains, 2),
            "unrealized_gains": round(goal.unrealized_gains, 2),
            "advice": goal.advice,
            "opportunities_needed": goal.opportunities_needed,
            "avg_premium_needed": round(goal.avg_premium_needed, 2),
            "period_start": goal.period_start.isoformat() if goal.period_start else None,
            "period_end": goal.period_end.isoformat() if goal.period_end else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting monthly goal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly-progress")
async def get_weekly_progress(user_id: str = "default_user"):
    """Get week-by-week progress breakdown for current month"""
    try:
        service = GoalTrackingService(user_id)
        weeks = service.get_weekly_progress()

        return {"weeks": weeks}

    except Exception as e:
        logger.error(f"Error getting weekly progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_performance_trends(user_id: str = "default_user"):
    """Get performance trends over recent months"""
    try:
        service = GoalTrackingService(user_id)
        trends = service.get_performance_trends()

        return trends

    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_goals(user_id: str = "default_user"):
    """Get all active goals for the user"""
    try:
        service = GoalTrackingService(user_id)
        goals = service.get_active_goals()

        return {"goals": goals}

    except Exception as e:
        logger.error(f"Error getting all goals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/target")
async def update_goal_target(
    request: UpdateTargetRequest,
    user_id: str = "default_user"
):
    """Update the monthly income target amount"""
    try:
        if request.target_amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="Target amount must be positive"
            )

        service = GoalTrackingService(user_id)
        success = service.update_goal_target(request.target_amount)

        if not success:
            raise HTTPException(
                status_code=404,
                detail="No active goal found to update"
            )

        return {
            "success": True,
            "message": f"Target updated to ${request.target_amount:,.0f}",
            "new_target": request.target_amount
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating goal target: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record-income")
async def record_income(
    request: RecordIncomeRequest,
    user_id: str = "default_user"
):
    """
    Manually record income toward goal.

    Use this for income from sources not automatically tracked
    (e.g., dividend payments, manual premium adjustments).
    """
    try:
        service = GoalTrackingService(user_id)
        success = service.record_income(
            amount=request.amount,
            source=request.source,
            symbol=request.symbol,
            notes=request.notes
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to record income"
            )

        return {
            "success": True,
            "message": f"Recorded ${request.amount:,.2f} from {request.source}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording income: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_goal_summary(user_id: str = "default_user"):
    """
    Get a formatted text summary of goal progress.

    Useful for chat context or quick dashboard display.
    """
    try:
        service = GoalTrackingService(user_id)
        summary = service.get_goal_summary_for_chat()

        return {
            "summary": summary,
            "format": "markdown"
        }

    except Exception as e:
        logger.error(f"Error getting goal summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status-badge")
async def get_status_badge(user_id: str = "default_user"):
    """
    Get a quick status badge for the dashboard.

    Returns minimal data for UI status indicators.
    """
    try:
        service = GoalTrackingService(user_id)
        goal = service.get_monthly_income_goal()

        if not goal:
            return {
                "status": "no_goal",
                "color": "gray",
                "text": "No Goal Set"
            }

        status_colors = {
            GoalStatus.ACHIEVED: "green",
            GoalStatus.AHEAD: "blue",
            GoalStatus.ON_TRACK: "green",
            GoalStatus.BEHIND: "yellow",
            GoalStatus.AT_RISK: "red"
        }

        return {
            "status": goal.status.value,
            "color": status_colors.get(goal.status, "gray"),
            "text": goal.status.value.replace('_', ' ').title(),
            "progress_percent": round(goal.progress_percent, 0),
            "current_amount": round(goal.current_amount, 0),
            "target_amount": round(goal.target_amount, 0)
        }

    except Exception as e:
        logger.error(f"Error getting status badge: {e}")
        return {
            "status": "error",
            "color": "gray",
            "text": "Error"
        }
