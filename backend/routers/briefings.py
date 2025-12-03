"""
Briefings Router - API endpoints for morning briefings and reports

Provides endpoints for:
- Generating morning briefings
- Sending briefings via alerts
- Viewing briefing history

Performance: Uses asyncio.to_thread() for non-blocking DB calls
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any, List
import logging
import asyncio

from src.ava.services.morning_briefing_service import (
    MorningBriefingService,
    get_morning_briefing,
    send_morning_briefing
)
from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)


# ============ Sync Helper Functions (run via asyncio.to_thread) ============

def _fetch_briefing_history_sync(user_id: str, limit: int) -> Dict[str, Any]:
    """Sync function to fetch briefing history - called via asyncio.to_thread()"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, report_type, title, content, generated_at
            FROM ava_generated_reports
            WHERE user_id = %s AND report_type = 'morning_briefing'
            ORDER BY generated_at DESC
            LIMIT %s
        """, (user_id, limit))
        rows = cursor.fetchall()
        return {"briefings": [{
            "id": row[0],
            "report_type": row[1],
            "title": row[2],
            "content": row[3],
            "generated_at": row[4].isoformat() if row[4] else None
        } for row in rows]}


def _fetch_briefing_preview_sync(user_id: str) -> Dict[str, Any]:
    """Sync function to fetch briefing preview counts - called via asyncio.to_thread()"""
    from src.ava.services.goal_tracking_service import GoalTrackingService

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trade_history WHERE status = 'open'")
        open_positions = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT COUNT(*) FROM earnings_calendar
            WHERE report_date >= CURRENT_DATE
              AND report_date <= CURRENT_DATE + INTERVAL '7 days'
        """)
        upcoming_earnings = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT COUNT(*) FROM xtrades_alerts
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """)
        xtrades_24h = cursor.fetchone()[0] or 0

    # Get goal status
    goal_service = GoalTrackingService(user_id)
    goal = goal_service.get_monthly_income_goal()

    return {
        "preview": {
            "open_positions": open_positions,
            "upcoming_earnings": upcoming_earnings,
            "xtrades_activity": xtrades_24h,
            "goal_progress": {
                "percent": round(goal.progress_percent, 1) if goal else 0,
                "status": goal.status.value if goal else "no_goal"
            }
        },
        "has_urgent_items": (
            (goal and goal.status.value in ['behind', 'at_risk']) or
            upcoming_earnings > 0 or
            xtrades_24h > 0
        )
    }

router = APIRouter(prefix="/api/briefings", tags=["briefings"])


@router.get("/morning")
async def get_morning_briefing_endpoint(user_id: str = "default_user"):
    """
    Generate and return the morning briefing.

    This does NOT send the briefing - it just generates and returns it
    for display in the UI.
    """
    try:
        briefing = await get_morning_briefing(user_id)

        return {
            "generated_at": briefing.generated_at.isoformat(),
            "greeting": briefing.greeting,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "emoji": s.emoji,
                    "priority": s.priority
                }
                for s in sorted(briefing.sections, key=lambda x: x.priority)
            ],
            "action_items": briefing.action_items,
            "telegram_format": briefing.to_telegram_message(),
            "email_format": briefing.to_email_html()
        }

    except Exception as e:
        logger.error(f"Error generating briefing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/morning/send")
async def send_morning_briefing_endpoint(
    background_tasks: BackgroundTasks,
    user_id: str = "default_user",
    via_telegram: bool = True,
    via_email: bool = True
):
    """
    Generate and send the morning briefing via configured channels.

    Sends asynchronously in the background.
    """
    try:
        # Send in background to not block the response
        async def send_task():
            service = MorningBriefingService(user_id)
            await service.send_briefing(
                via_telegram=via_telegram,
                via_email=via_email
            )

        background_tasks.add_task(send_task)

        return {
            "success": True,
            "message": "Morning briefing is being generated and sent",
            "channels": {
                "telegram": via_telegram,
                "email": via_email
            }
        }

    except Exception as e:
        logger.error(f"Error scheduling briefing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_briefing_history(
    user_id: str = "default_user",
    limit: int = 10
):
    """Get history of generated briefings. Uses asyncio.to_thread() for non-blocking DB."""
    try:
        return await asyncio.to_thread(_fetch_briefing_history_sync, user_id, limit)
    except Exception as e:
        logger.error(f"Error getting briefing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview")
async def preview_briefing(user_id: str = "default_user"):
    """
    Get a quick preview/summary of what the briefing will contain.
    Uses asyncio.to_thread() for non-blocking DB and service calls.
    """
    try:
        return await asyncio.to_thread(_fetch_briefing_preview_sync, user_id)
    except Exception as e:
        logger.error(f"Error getting briefing preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))
