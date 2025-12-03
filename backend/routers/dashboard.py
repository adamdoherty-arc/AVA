from fastapi import APIRouter, Query
from typing import Dict, List, Optional
from backend.services.dashboard_service import dashboard_service
from pydantic import BaseModel

class PortfolioSummary(BaseModel):
    total_value: float
    buying_power: float
    day_change: float
    day_change_pct: float
    allocations: Dict[str, float]

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"]
)

@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary():
    """
    Get portfolio summary metrics.
    """
    return await dashboard_service.get_portfolio_summary()

@router.get("/activity")
async def get_recent_activity(limit: int = 10):
    """
    Get recent activity.
    """
    return dashboard_service.get_recent_activity(limit)

@router.get("/performance")
async def get_performance_history(period: str = "1M"):
    """
    Get performance history.
    """
    history = dashboard_service.get_performance_history(period)
    # Format for frontend: expects {history: [{date, value}, ...]}
    return {
        "history": [
            {"date": str(h.get("date", ""))[:10], "value": h.get("portfolio_value", 0)}
            for h in history
        ]
    }
