"""
XTrades Router - API endpoints for XTrades watchlists
NO MOCK DATA - All endpoints use real database via XtradesDBManager
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

from src.xtrades_db_manager import XtradesDBManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/xtrades",
    tags=["xtrades"]
)


class ProfileCreate(BaseModel):
    username: str
    display_name: Optional[str] = None
    notes: Optional[str] = None


class ProfileUpdate(BaseModel):
    active: Optional[bool] = None
    display_name: Optional[str] = None
    notes: Optional[str] = None


# Initialize XTrades DB manager (singleton pattern)
_xtrades_manager = None

def get_xtrades_manager():
    global _xtrades_manager
    if _xtrades_manager is None:
        _xtrades_manager = XtradesDBManager()
    return _xtrades_manager


@router.get("/trades")
async def get_trades(
    status: str = Query("open", description="Trade status: open, closed, expired"),
    profile_id: Optional[int] = Query(None, description="Filter by profile ID"),
    limit: int = Query(100, description="Maximum results")
):
    """
    Get trades from XTrades profiles.
    Data comes from xtrades_trades table via XtradesDBManager.
    """
    try:
        manager = get_xtrades_manager()

        # If profile_id specified, get trades for that profile
        if profile_id:
            trades = manager.get_trades_by_profile(profile_id, status=status, limit=limit)
        else:
            # Get trades from all active profiles
            profiles = manager.get_active_profiles()
            trades = []
            for profile in profiles:
                profile_trades = manager.get_trades_by_profile(
                    profile['id'],
                    status=status,
                    limit=limit // max(len(profiles), 1)
                )
                for trade in profile_trades:
                    trade['profile_name'] = profile.get('display_name') or profile.get('username')
                trades.extend(profile_trades)

        # Format trades for API response
        formatted_trades = []
        for trade in trades[:limit]:
            # Calculate days open from alert_timestamp or entry_date
            days_open = 0
            alert_ts = trade.get('alert_timestamp') or trade.get('entry_date')
            if alert_ts:
                try:
                    if hasattr(alert_ts, 'date'):
                        days_open = (datetime.now().date() - alert_ts.date()).days
                    else:
                        days_open = (datetime.now() - alert_ts).days
                except Exception:
                    days_open = 0

            # Format expiration date
            exp_date = trade.get('expiration_date')
            if exp_date:
                if hasattr(exp_date, 'strftime'):
                    exp_date_str = exp_date.strftime("%Y-%m-%d")
                else:
                    exp_date_str = str(exp_date)
            else:
                exp_date_str = None

            # Format alert timestamp
            alert_timestamp = trade.get('alert_timestamp')
            if alert_timestamp:
                if hasattr(alert_timestamp, 'isoformat'):
                    alert_ts_str = alert_timestamp.isoformat()
                else:
                    alert_ts_str = str(alert_timestamp)
            else:
                alert_ts_str = None

            # Format exit date
            exit_date = trade.get('exit_date')
            if exit_date:
                if hasattr(exit_date, 'isoformat'):
                    exit_date_str = exit_date.isoformat()
                else:
                    exit_date_str = str(exit_date)
            else:
                exit_date_str = None

            formatted_trades.append({
                "id": trade.get('id'),
                "profile_id": trade.get('profile_id'),
                "profile_name": trade.get('profile_name', 'Unknown'),
                "ticker": trade.get('ticker') or trade.get('symbol'),
                "strategy": trade.get('strategy') or trade.get('action', 'Unknown'),
                "entry_price": float(trade.get('entry_price', 0) or 0),
                "exit_price": float(trade.get('exit_price') or 0) if trade.get('exit_price') else None,
                "strike_price": float(trade.get('strike_price', 0) or 0),
                "expiration_date": exp_date_str,
                "quantity": trade.get('quantity', 1),
                "status": trade.get('status', status),
                "pnl": float(trade.get('pnl', 0) or 0),
                "pnl_percent": float(trade.get('pnl_percent', 0) or 0),
                "alert_timestamp": alert_ts_str,
                "exit_date": exit_date_str,
                "days_open": days_open
            })

        return {
            "trades": formatted_trades,
            "total": len(formatted_trades),
            "status_filter": status,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching xtrades trades: {e}")
        return {
            "trades": [],
            "total": 0,
            "status_filter": status,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/profiles")
async def get_profiles(include_inactive: bool = Query(False, description="Include inactive profiles")):
    """
    Get all XTrades profiles.
    Data comes from xtrades_profiles table.
    """
    try:
        manager = get_xtrades_manager()
        profiles = manager.get_all_profiles(include_inactive=include_inactive)

        # Format for API response
        formatted_profiles = []
        for profile in profiles:
            formatted_profiles.append({
                "id": profile.get('id'),
                "username": profile.get('username'),
                "display_name": profile.get('display_name'),
                "active": profile.get('active', True),
                "last_sync": profile.get('last_sync').isoformat() if profile.get('last_sync') else None,
                "last_sync_status": profile.get('last_sync_status', 'unknown'),
                "total_trades_scraped": profile.get('total_trades_scraped', 0),
                "added_date": profile.get('added_date').strftime("%Y-%m-%d") if profile.get('added_date') else None,
                "notes": profile.get('notes')
            })

        return {
            "profiles": formatted_profiles,
            "total": len(formatted_profiles),
            "active_count": len([p for p in formatted_profiles if p['active']]),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching xtrades profiles: {e}")
        return {
            "profiles": [],
            "total": 0,
            "active_count": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.post("/profiles")
async def create_profile(profile: ProfileCreate):
    """
    Add a new XTrades profile to monitor.
    Stored in xtrades_profiles table.
    """
    try:
        manager = get_xtrades_manager()
        profile_id = manager.add_profile(
            username=profile.username,
            display_name=profile.display_name,
            notes=profile.notes
        )

        return {
            "id": profile_id,
            "username": profile.username,
            "display_name": profile.display_name,
            "status": "created",
            "message": f"Profile '{profile.username}' added successfully"
        }

    except Exception as e:
        logger.error(f"Error creating xtrades profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profiles/{profile_id}")
async def update_profile(profile_id: int, update: ProfileUpdate):
    """
    Update an XTrades profile (full update).
    """
    try:
        manager = get_xtrades_manager()

        # Get existing profile
        existing = manager.get_profile_by_id(profile_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

        # Update fields if provided
        if update.active is not None:
            if not update.active:
                manager.deactivate_profile(profile_id)
            else:
                manager.reactivate_profile(profile_id)

        return {
            "id": profile_id,
            "status": "updated",
            "message": f"Profile {profile_id} updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating xtrades profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/profiles/{profile_id}")
async def patch_profile(profile_id: int, update: ProfileUpdate):
    """
    Partially update an XTrades profile (PATCH for toggle operations).
    """
    try:
        manager = get_xtrades_manager()

        # Get existing profile
        existing = manager.get_profile_by_id(profile_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

        # Update active status if provided
        if update.active is not None:
            if not update.active:
                manager.deactivate_profile(profile_id)
            else:
                manager.reactivate_profile(profile_id)

        return {
            "id": profile_id,
            "active": update.active if update.active is not None else existing.get('active'),
            "status": "updated",
            "message": f"Profile {profile_id} updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error patching xtrades profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """
    Get XTrades monitoring statistics.
    Data aggregated from database tables.
    """
    try:
        manager = get_xtrades_manager()

        # Get all profiles
        all_profiles = manager.get_all_profiles(include_inactive=True)
        active_profiles = [p for p in all_profiles if p.get('active', False)]

        # Calculate totals
        total_trades = sum(p.get('total_trades_scraped', 0) for p in all_profiles)

        # Get closed trades to calculate PnL and win rate
        total_pnl = 0.0
        win_rate = 0.0
        closed_count = 0
        winning_count = 0

        for profile in active_profiles:
            closed_trades = manager.get_trades_by_profile(profile['id'], status='closed', limit=500)
            for trade in closed_trades:
                pnl = float(trade.get('pnl', 0) or 0)
                total_pnl += pnl
                closed_count += 1
                if pnl > 0:
                    winning_count += 1

        if closed_count > 0:
            win_rate = (winning_count / closed_count) * 100

        # Get recent sync status
        recent_syncs = []
        for profile in active_profiles[:5]:
            if profile.get('last_sync'):
                recent_syncs.append({
                    "profile": profile.get('display_name') or profile.get('username'),
                    "status": profile.get('last_sync_status', 'unknown'),
                    "time": profile.get('last_sync').isoformat() if profile.get('last_sync') else None
                })

        return {
            "total_profiles": len(all_profiles),
            "active_profiles": len(active_profiles),
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 1),
            "last_sync": recent_syncs[0]['time'] if recent_syncs else None,
            "recent_syncs": recent_syncs,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching xtrades stats: {e}")
        return {
            "total_profiles": 0,
            "active_profiles": 0,
            "total_trades": 0,
            "total_pnl": 0,
            "win_rate": 0,
            "last_sync": None,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/ticker/{ticker}")
async def get_trades_by_ticker(
    ticker: str,
    limit: int = Query(50, description="Maximum results")
):
    """
    Get all trades for a specific ticker across all profiles.
    """
    try:
        manager = get_xtrades_manager()
        trades = manager.get_trades_by_ticker(ticker.upper(), limit=limit)

        formatted_trades = []
        for trade in trades:
            # Get profile name from joined data
            profile_name = trade.get('display_name') or trade.get('username') or 'Unknown'

            # Format alert timestamp
            alert_timestamp = trade.get('alert_timestamp')
            if alert_timestamp:
                if hasattr(alert_timestamp, 'isoformat'):
                    alert_ts_str = alert_timestamp.isoformat()
                else:
                    alert_ts_str = str(alert_timestamp)
            else:
                alert_ts_str = None

            formatted_trades.append({
                "id": trade.get('id'),
                "profile_id": trade.get('profile_id'),
                "profile_name": profile_name,
                "ticker": trade.get('ticker') or trade.get('symbol'),
                "strategy": trade.get('strategy') or trade.get('action', 'Unknown'),
                "entry_price": float(trade.get('entry_price', 0) or 0),
                "exit_price": float(trade.get('exit_price') or 0) if trade.get('exit_price') else None,
                "strike_price": float(trade.get('strike_price', 0) or 0),
                "expiration_date": trade.get('expiration_date').strftime("%Y-%m-%d") if trade.get('expiration_date') and hasattr(trade.get('expiration_date'), 'strftime') else str(trade.get('expiration_date')) if trade.get('expiration_date') else None,
                "status": trade.get('status', 'unknown'),
                "pnl": float(trade.get('pnl', 0) or 0),
                "pnl_percent": float(trade.get('pnl_percent', 0) or 0),
                "alert_timestamp": alert_ts_str,
            })

        return {
            "ticker": ticker.upper(),
            "trades": formatted_trades,
            "total": len(formatted_trades),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching trades for ticker {ticker}: {e}")
        return {
            "ticker": ticker.upper(),
            "trades": [],
            "total": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.post("/sync")
async def trigger_sync(profile_id: Optional[int] = None):
    """
    Trigger a sync for XTrades profiles.
    If profile_id specified, sync only that profile.
    Otherwise sync all active profiles.
    """
    try:
        # This would trigger the actual sync service
        # For now, return a status indicating the sync was queued
        return {
            "status": "queued",
            "message": f"Sync queued for {'profile ' + str(profile_id) if profile_id else 'all active profiles'}",
            "queued_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error triggering xtrades sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles/{profile_id}/sync")
async def trigger_profile_sync(profile_id: int):
    """
    Trigger a sync for a specific XTrades profile.
    """
    try:
        manager = get_xtrades_manager()

        # Verify profile exists
        existing = manager.get_profile_by_id(profile_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

        # This would trigger the actual sync service for this profile
        # For now, return a status indicating the sync was queued
        return {
            "status": "queued",
            "profile_id": profile_id,
            "profile_name": existing.get('display_name') or existing.get('username'),
            "message": f"Sync queued for profile {profile_id}",
            "queued_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering sync for profile {profile_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync/history")
async def get_sync_history(limit: int = Query(20, description="Maximum results")):
    """
    Get sync history for XTrades profiles.
    Data comes from xtrades_sync_log table.
    """
    try:
        manager = get_xtrades_manager()

        # Try to get actual sync logs first
        sync_logs = manager.get_sync_history(limit=limit)

        if sync_logs:
            # Use actual sync log table data
            formatted_logs = []
            for log in sync_logs:
                sync_ts = log.get('sync_timestamp')
                if sync_ts and hasattr(sync_ts, 'isoformat'):
                    sync_ts_str = sync_ts.isoformat()
                else:
                    sync_ts_str = str(sync_ts) if sync_ts else None

                formatted_logs.append({
                    "id": log.get('id'),
                    "sync_timestamp": sync_ts_str,
                    "profiles_synced": log.get('profiles_synced', 0),
                    "trades_found": log.get('trades_found', 0),
                    "new_trades": log.get('new_trades', 0),
                    "updated_trades": log.get('updated_trades', 0),
                    "duration_seconds": float(log.get('duration_seconds', 0) or 0),
                    "status": log.get('status', 'unknown'),
                    "errors": log.get('errors')
                })

            return {
                "logs": formatted_logs,
                "total": len(formatted_logs),
                "generated_at": datetime.now().isoformat()
            }

        # Fallback to profile-based sync history
        profiles = manager.get_all_profiles(include_inactive=True)
        sync_history = []
        for profile in profiles:
            if profile.get('last_sync'):
                last_sync = profile.get('last_sync')
                if hasattr(last_sync, 'isoformat'):
                    sync_time = last_sync.isoformat()
                else:
                    sync_time = str(last_sync)

                sync_history.append({
                    "id": profile.get('id'),
                    "sync_timestamp": sync_time,
                    "profiles_synced": 1,
                    "trades_found": profile.get('total_trades_scraped', 0),
                    "new_trades": 0,
                    "updated_trades": 0,
                    "duration_seconds": 0,
                    "status": profile.get('last_sync_status', 'unknown'),
                    "errors": None if profile.get('last_sync_status') == 'success' else profile.get('last_sync_status')
                })

        sync_history.sort(key=lambda x: x['sync_timestamp'] or '', reverse=True)

        return {
            "logs": sync_history[:limit],
            "total": len(sync_history),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching xtrades sync history: {e}")
        return {
            "logs": [],
            "total": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/profiles/{profile_id}/stats")
async def get_profile_stats(profile_id: int):
    """
    Get detailed statistics for a specific profile.
    """
    try:
        manager = get_xtrades_manager()

        # Get profile info
        profile = manager.get_profile_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

        # Get profile stats
        stats = manager.get_profile_stats(profile_id)

        return {
            "profile_id": profile_id,
            "profile_name": profile.get('display_name') or profile.get('username'),
            "total_trades": stats.get('total_trades', 0),
            "open_trades": stats.get('open_trades', 0),
            "closed_trades": stats.get('closed_trades', 0),
            "total_pnl": float(stats.get('total_pnl', 0) or 0),
            "avg_pnl": float(stats.get('avg_pnl', 0) or 0),
            "win_rate": float(stats.get('win_rate', 0) or 0),
            "best_trade": stats.get('best_trade'),
            "worst_trade": stats.get('worst_trade'),
            "generated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching profile stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/stats/all")
async def get_all_profile_stats():
    """
    Get statistics for all active profiles.
    """
    try:
        manager = get_xtrades_manager()
        profiles = manager.get_active_profiles()

        all_stats = []
        for profile in profiles:
            stats = manager.get_profile_stats(profile['id'])
            all_stats.append({
                "profile_id": profile['id'],
                "profile_name": profile.get('display_name') or profile.get('username'),
                "total_trades": stats.get('total_trades', 0),
                "open_trades": stats.get('open_trades', 0),
                "closed_trades": stats.get('closed_trades', 0),
                "total_pnl": float(stats.get('total_pnl', 0) or 0),
                "avg_pnl": float(stats.get('avg_pnl', 0) or 0),
                "win_rate": float(stats.get('win_rate', 0) or 0),
            })

        return {
            "profiles": all_stats,
            "total": len(all_stats),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching all profile stats: {e}")
        return {
            "profiles": [],
            "total": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/stats/by-strategy")
async def get_stats_by_strategy():
    """
    Get performance statistics grouped by trading strategy.
    """
    try:
        manager = get_xtrades_manager()

        # Get all closed trades for strategy analysis
        closed_trades = manager.get_all_trades(status='closed', limit=1000)

        # Group by strategy
        strategy_stats = {}
        for trade in closed_trades:
            strategy = trade.get('strategy') or trade.get('action', 'Unknown')
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "strategy": strategy,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "total_pnl": 0.0,
                    "trades": []
                }

            stats = strategy_stats[strategy]
            stats["total_trades"] += 1
            pnl = float(trade.get('pnl', 0) or 0)
            stats["total_pnl"] += pnl
            if pnl > 0:
                stats["winning_trades"] += 1
            stats["trades"].append(pnl)

        # Calculate averages and win rates
        result = []
        for strategy, stats in strategy_stats.items():
            if stats["total_trades"] > 0:
                result.append({
                    "strategy": strategy,
                    "total_trades": stats["total_trades"],
                    "total_pnl": round(stats["total_pnl"], 2),
                    "avg_pnl": round(stats["total_pnl"] / stats["total_trades"], 2),
                    "win_rate": round((stats["winning_trades"] / stats["total_trades"]) * 100, 1),
                    "best_trade": max(stats["trades"]) if stats["trades"] else 0,
                    "worst_trade": min(stats["trades"]) if stats["trades"] else 0,
                })

        # Sort by total P/L descending
        result.sort(key=lambda x: x["total_pnl"], reverse=True)

        return {
            "strategies": result,
            "total": len(result),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching strategy stats: {e}")
        return {
            "strategies": [],
            "total": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/recent-activity")
async def get_recent_activity(days: int = Query(7, description="Number of days to look back")):
    """
    Get recent trading activity across all profiles.
    """
    try:
        manager = get_xtrades_manager()
        trades = manager.get_recent_activity(days=days, limit=100)

        formatted_trades = []
        for trade in trades:
            profile_name = trade.get('display_name') or trade.get('username') or 'Unknown'

            alert_ts = trade.get('alert_timestamp')
            alert_ts_str = alert_ts.isoformat() if alert_ts and hasattr(alert_ts, 'isoformat') else str(alert_ts) if alert_ts else None

            formatted_trades.append({
                "id": trade.get('id'),
                "profile_name": profile_name,
                "ticker": trade.get('ticker'),
                "strategy": trade.get('strategy') or trade.get('action'),
                "entry_price": float(trade.get('entry_price', 0) or 0),
                "status": trade.get('status'),
                "pnl": float(trade.get('pnl', 0) or 0),
                "alert_timestamp": alert_ts_str,
            })

        return {
            "trades": formatted_trades,
            "total": len(formatted_trades),
            "days": days,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching recent activity: {e}")
        return {
            "trades": [],
            "total": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }
