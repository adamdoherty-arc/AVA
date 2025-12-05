"""
Briefings Router - API endpoints for morning briefings and reports

Provides endpoints for:
- Generating morning briefings
- Sending briefings via alerts
- Viewing briefing history

Performance: Uses async database manager for non-blocking DB calls
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any, List
import structlog

from src.ava.services.morning_briefing_service import (
    MorningBriefingService,
    get_morning_briefing,
    send_morning_briefing
)
from backend.infrastructure.database import get_database, AsyncDatabaseManager
from backend.infrastructure.errors import safe_internal_error

logger = structlog.get_logger(__name__)


# ============ Async Helper Functions ============

async def _fetch_briefing_history_async(user_id: str, limit: int) -> Dict[str, Any]:
    """Async function to fetch briefing history"""
    db = await get_database()
    rows = await db.fetch("""
        SELECT id, report_type, title, content, generated_at
        FROM ava_generated_reports
        WHERE user_id = $1 AND report_type = 'morning_briefing'
        ORDER BY generated_at DESC
        LIMIT $2
    """, user_id, limit)

    return {"briefings": [{
        "id": row["id"],
        "report_type": row["report_type"],
        "title": row["title"],
        "content": row["content"],
        "generated_at": row["generated_at"].isoformat() if row["generated_at"] else None
    } for row in rows]}


async def _fetch_briefing_preview_async(user_id: str) -> Dict[str, Any]:
    """Async function to fetch briefing preview counts"""
    from src.ava.services.goal_tracking_service import GoalTrackingService

    db = await get_database()

    # Get open positions count
    open_positions_row = await db.fetchrow("SELECT COUNT(*) FROM trade_history WHERE status = 'open'")
    open_positions = open_positions_row["count"] if open_positions_row else 0

    # Get upcoming earnings count
    upcoming_earnings_row = await db.fetchrow("""
        SELECT COUNT(*) FROM earnings_calendar
        WHERE report_date >= CURRENT_DATE
          AND report_date <= CURRENT_DATE + INTERVAL '7 days'
    """)
    upcoming_earnings = upcoming_earnings_row["count"] if upcoming_earnings_row else 0

    # Get recent xtrades alerts count
    xtrades_24h_row = await db.fetchrow("""
        SELECT COUNT(*) FROM xtrades_alerts
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """)
    xtrades_24h = xtrades_24h_row["count"] if xtrades_24h_row else 0

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
        logger.error("generate_morning_briefing_error", error=str(e))
        safe_internal_error(e, "generate morning briefing")


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
        # Sync wrapper for BackgroundTasks (which expects sync functions)
        def send_briefing_sync(uid: str, telegram: bool, email: bool):
            """Sync wrapper that runs async service in new event loop"""
            import asyncio

            async def _run():
                service = MorningBriefingService(uid)
                await service.send_briefing(
                    via_telegram=telegram,
                    via_email=email
                )

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_run())
            except Exception as e:
                logger.error("background_briefing_send_failed", error=str(e))
            finally:
                loop.close()

        background_tasks.add_task(send_briefing_sync, user_id, via_telegram, via_email)

        return {
            "success": True,
            "message": "Morning briefing is being generated and sent",
            "channels": {
                "telegram": via_telegram,
                "email": via_email
            }
        }

    except Exception as e:
        logger.error("schedule_morning_briefing_error", error=str(e))
        safe_internal_error(e, "schedule morning briefing")


@router.get("/history")
async def get_briefing_history(
    user_id: str = "default_user",
    limit: int = 10
):
    """Get history of generated briefings. Uses async database manager for non-blocking DB."""
    try:
        return await _fetch_briefing_history_async(user_id, limit)
    except Exception as e:
        logger.error("fetch_briefing_history_error", error=str(e))
        safe_internal_error(e, "fetch briefing history")


@router.get("/preview")
async def preview_briefing(user_id: str = "default_user"):
    """
    Get a quick preview/summary of what the briefing will contain.
    Uses async database manager for non-blocking DB and service calls.
    """
    try:
        return await _fetch_briefing_preview_async(user_id)
    except Exception as e:
        logger.error("fetch_briefing_preview_error", error=str(e))
        safe_internal_error(e, "fetch briefing preview")
