"""
Watchlist Router - API endpoints for database and TradingView watchlists
NO MOCK DATA - All endpoints use real database queries

Performance: Uses async database manager for non-blocking DB calls
"""

from fastapi import APIRouter, HTTPException, Query, Request
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from pydantic import BaseModel
import structlog

from backend.infrastructure.database import get_database, AsyncDatabaseManager
from backend.infrastructure.observability import get_audit_logger, AuditEventType
from backend.infrastructure.errors import safe_internal_error

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/watchlist",
    tags=["watchlist"]
)


class CreateWatchlistRequest(BaseModel):
    """Request to create a new watchlist"""
    name: str
    description: Optional[str] = None
    symbols: List[str] = []


class AddSymbolsRequest(BaseModel):
    """Request to add symbols to a watchlist"""
    symbols: List[str]


# ============ Async Helper Functions ============

async def _fetch_database_watchlists_async() -> Dict[str, Any]:
    """Async function to fetch watchlists"""
    db = await get_database()
    try:
        rows = await db.fetch("""
            SELECT w.watchlist_id, w.name, w.symbol_count, w.created_at, w.updated_at
            FROM tv_watchlists_api w
            ORDER BY w.symbol_count DESC, w.name
        """)
        watchlists = [{
            "id": row["watchlist_id"],
            "name": row["name"],
            "symbol_count": row["symbol_count"] or 0,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
        } for row in rows]
        return {"watchlists": watchlists, "total": len(watchlists), "source": "database"}
    except Exception as e:
        logger.warning("primary_table_failed", error=str(e), fallback="alternate")
        # Fallback to alternate table
        rows = await db.fetch("""
            SELECT id, name, symbol_count, created_at, updated_at
            FROM tv_watchlists WHERE is_active = TRUE
            ORDER BY symbol_count DESC, name
        """)
        watchlists = [{
            "id": row["id"],
            "name": row["name"],
            "symbol_count": row["symbol_count"] or 0,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
        } for row in rows]
        return {"watchlists": watchlists, "total": len(watchlists), "source": "database_alt"}


# ============ Database Watchlist Endpoints ============

@router.get("/database")
async def get_database_watchlists():
    """
    Get all watchlists from tv_watchlists_api table (TradingView synced data).
    Returns watchlists with symbol counts and metadata.
    Uses async database manager for non-blocking database access.
    """
    try:
        result = await _fetch_database_watchlists_async()
        result["generated_at"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error("fetch_database_watchlists_error", error=str(e))
        safe_internal_error(e, "fetch database watchlists")


async def _fetch_watchlist_symbols_async(watchlist_name: str, limit: int, stocks_only: bool) -> Dict[str, Any]:
    """Async function to fetch watchlist symbols"""
    db = await get_database()
    if stocks_only:
        rows = await db.fetch("""
            SELECT s.symbol, s.exchange, s.full_symbol, s.added_at
            FROM tv_symbols_api s
            JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
            WHERE w.name = $1 AND s.exchange IN ('NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS')
            ORDER BY s.symbol LIMIT $2
        """, watchlist_name, limit)
    else:
        rows = await db.fetch("""
            SELECT s.symbol, s.exchange, s.full_symbol, s.added_at
            FROM tv_symbols_api s
            JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
            WHERE w.name = $1
            ORDER BY s.symbol LIMIT $2
        """, watchlist_name, limit)

    symbols = [{
        "symbol": row["symbol"],
        "exchange": row["exchange"],
        "full_symbol": row["full_symbol"],
        "added_at": row["added_at"].isoformat() if row["added_at"] else None
    } for row in rows]
    return {"symbols": symbols, "count": len(symbols), "watchlist_name": watchlist_name}


@router.get("/database/{watchlist_name}/symbols")
async def get_database_watchlist_symbols(
    watchlist_name: str,
    limit: int = Query(200, description="Maximum symbols to return"),
    stocks_only: bool = Query(False, description="Filter to only show stock exchanges (exclude crypto)")
):
    """
    Get all symbols in a specific database watchlist with details.
    Uses tv_symbols_api table which has basic symbol info.
    """
    try:
        db = await get_database()

        # Build query based on filter
        if stocks_only:
            # Filter to stock exchanges only
            rows = await db.fetch("""
                SELECT
                    s.symbol,
                    s.exchange,
                    s.full_symbol,
                    s.added_at
                FROM tv_symbols_api s
                JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
                WHERE w.name = $1
                  AND s.exchange IN ('NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS')
                ORDER BY s.symbol
                LIMIT $2
            """, watchlist_name, limit)
        else:
            # Return all symbols
            rows = await db.fetch("""
                SELECT
                    s.symbol,
                    s.exchange,
                    s.full_symbol,
                    s.added_at
                FROM tv_symbols_api s
                JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
                WHERE w.name = $1
                ORDER BY s.symbol
                LIMIT $2
            """, watchlist_name, limit)

        symbols = []
        for row in rows:
            symbols.append({
                "symbol": row["symbol"],
                "exchange": row["exchange"] or "Unknown",
                "full_symbol": row["full_symbol"],
                "added_at": row["added_at"].isoformat() if row["added_at"] else None
            })

        return {
            "watchlist_name": watchlist_name,
            "symbols": symbols,
            "count": len(symbols),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("fetch_watchlist_symbols_error", watchlist_name=watchlist_name, error=str(e))
        return {
            "watchlist_name": watchlist_name,
            "symbols": [],
            "count": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


# ============ TradingView Watchlist Endpoints ============

@router.get("/tradingview")
async def get_tradingview_watchlists():
    """
    Get all TradingView synced watchlists.
    Data from tradingview_watchlists and tv_watchlist_symbols tables.
    """
    try:
        db = await get_database()

        # Get watchlists from tradingview_watchlists with symbol counts from tv_watchlist_symbols
        rows = await db.fetch("""
            SELECT
                tw.id,
                tw.name,
                COUNT(ts.id) as symbol_count,
                tw.created_date,
                tw.last_updated
            FROM tradingview_watchlists tw
            LEFT JOIN tv_watchlist_symbols ts ON ts.watchlist_id = tw.id
            GROUP BY tw.id, tw.name, tw.created_date, tw.last_updated
            ORDER BY COUNT(ts.id) DESC, tw.name
        """)

        watchlists = []
        for row in rows:
            watchlists.append({
                "id": row["id"],
                "name": row["name"],
                "symbol_count": row["symbol_count"] or 0,
                "created_at": row["created_date"].isoformat() if row["created_date"] else None,
                "last_synced": row["last_updated"].isoformat() if row["last_updated"] else None
            })

        return {
            "watchlists": watchlists,
            "total": len(watchlists),
            "source": "tradingview",
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("fetch_tradingview_watchlists_error", error=str(e))
        return {
            "watchlists": [],
            "total": 0,
            "error": str(e),
            "message": "TradingView watchlists table not found. Run TradingView sync to populate.",
            "generated_at": datetime.now().isoformat()
        }


@router.get("/tradingview/{watchlist_id}/symbols")
async def get_tradingview_watchlist_symbols(
    watchlist_id: int,
    limit: int = Query(200, description="Maximum symbols to return")
):
    """
    Get all symbols in a specific TradingView watchlist.
    Uses tv_watchlist_symbols table.
    """
    try:
        db = await get_database()

        # Get watchlist name
        name_row = await db.fetchrow("""
            SELECT name FROM tradingview_watchlists WHERE id = $1
        """, watchlist_id)
        watchlist_name = name_row["name"] if name_row else f"Watchlist {watchlist_id}"

        # Get symbols from tv_watchlist_symbols (correct table name)
        rows = await db.fetch("""
            SELECT
                ts.symbol,
                ts.company_name,
                ts.sector,
                ts.industry,
                ts.last_price,
                ts.volume,
                ts.market_cap,
                ts.added_at
            FROM tv_watchlist_symbols ts
            WHERE ts.watchlist_id = $1
            ORDER BY ts.symbol
            LIMIT $2
        """, watchlist_id, limit)

        symbols = []
        for row in rows:
            symbols.append({
                "symbol": row["symbol"],
                "company_name": row["company_name"] or row["symbol"],
                "sector": row["sector"] or "Unknown",
                "industry": row["industry"] or "Unknown",
                "last_price": float(row["last_price"]) if row["last_price"] else 0,
                "volume": int(row["volume"]) if row["volume"] else 0,
                "market_cap": int(row["market_cap"]) if row["market_cap"] else 0,
                "added_at": row["added_at"].isoformat() if row["added_at"] else None
            })

        return {
            "watchlist_id": watchlist_id,
            "watchlist_name": watchlist_name,
            "symbols": symbols,
            "count": len(symbols),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("fetch_tradingview_symbols_error", watchlist_id=watchlist_id, error=str(e))
        return {
            "watchlist_id": watchlist_id,
            "symbols": [],
            "count": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


# ============ Combined View Endpoints ============

@router.get("/all")
async def get_all_watchlists():
    """
    Get all watchlists from all sources combined.
    Includes both database (tv_watchlists_api) and TradingView synced watchlists.
    """
    all_watchlists = {
        "database": [],
        "tradingview": []
    }

    # Get database watchlists
    try:
        db_result = await get_database_watchlists()
        all_watchlists["database"] = db_result.get("watchlists", [])
    except Exception as e:
        logger.warning("fetch_database_watchlists_failed", error=str(e))

    # Get TradingView watchlists
    try:
        tv_result = await get_tradingview_watchlists()
        all_watchlists["tradingview"] = tv_result.get("watchlists", [])
    except Exception as e:
        logger.warning("fetch_tradingview_watchlists_failed", error=str(e))

    total = len(all_watchlists["database"]) + len(all_watchlists["tradingview"])

    return {
        "sources": all_watchlists,
        "total_watchlists": total,
        "generated_at": datetime.now().isoformat()
    }


@router.get("/symbols/search")
async def search_symbols(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, description="Maximum results")
):
    """
    Search for symbols across all watchlists.
    """
    try:
        db = await get_database()

        search_pattern = f"%{q.upper()}%"

        # Search in tv_symbols_api using actual columns
        rows = await db.fetch("""
            SELECT DISTINCT
                s.symbol,
                s.exchange,
                s.full_symbol,
                w.name as watchlist_name
            FROM tv_symbols_api s
            JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
            WHERE s.symbol ILIKE $1
            ORDER BY s.symbol
            LIMIT $2
        """, search_pattern, limit)

        results = []
        for row in rows:
            results.append({
                "symbol": row["symbol"],
                "exchange": row["exchange"] or "Unknown",
                "full_symbol": row["full_symbol"],
                "watchlist": row["watchlist_name"]
            })

        return {
            "query": q,
            "results": results,
            "count": len(results),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("search_symbols_error", query=q, error=str(e))
        return {
            "query": q,
            "results": [],
            "count": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


# ============ Watchlist Management Endpoints ============

@router.post("/create")
async def create_watchlist(request: CreateWatchlistRequest, req: Request):
    """
    Create a new custom watchlist.
    """
    audit = get_audit_logger()
    try:
        from src.tradingview_db_manager import TradingViewDBManager
        manager = TradingViewDBManager()

        watchlist_id = manager.create_watchlist(request.name, request.description)

        if watchlist_id and request.symbols:
            manager.add_symbols_to_watchlist(watchlist_id, request.symbols)

        # Log watchlist creation
        await audit.log(
            AuditEventType.WATCHLIST_CREATED,
            action=f"Watchlist created: {request.name}",
            resource_type="watchlist",
            resource_id=str(watchlist_id),
            details={
                "name": request.name,
                "description": request.description,
                "symbols": request.symbols,
                "symbol_count": len(request.symbols),
            },
            ip_address=req.client.host if req.client else None,
        )

        return {
            "id": watchlist_id,
            "name": request.name,
            "symbols_added": len(request.symbols),
            "message": f"Watchlist '{request.name}' created successfully",
            "created_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("create_watchlist_error", error=str(e))
        safe_internal_error(e, "create watchlist")


@router.post("/{watchlist_name}/symbols")
async def add_symbols_to_watchlist(watchlist_name: str, request: AddSymbolsRequest, req: Request):
    """
    Add symbols to an existing watchlist.
    """
    audit = get_audit_logger()
    try:
        from src.tradingview_db_manager import TradingViewDBManager
        manager = TradingViewDBManager()

        # Get or create watchlist
        watchlists = manager.get_all_watchlists()
        watchlist_id = None

        for w in watchlists:
            if w['name'] == watchlist_name:
                watchlist_id = w['id']
                break

        if not watchlist_id:
            watchlist_id = manager.create_watchlist(watchlist_name)

        if watchlist_id:
            count = manager.add_symbols_to_watchlist(watchlist_id, request.symbols)

            # Log watchlist modification
            await audit.log(
                AuditEventType.WATCHLIST_MODIFIED,
                action=f"Symbols added to watchlist: {watchlist_name}",
                resource_type="watchlist",
                resource_id=str(watchlist_id),
                details={
                    "watchlist_name": watchlist_name,
                    "symbols_added": request.symbols,
                    "count": count,
                },
                ip_address=req.client.host if req.client else None,
            )

            return {
                "watchlist_name": watchlist_name,
                "symbols_added": count,
                "message": f"Added {count} symbols to '{watchlist_name}'",
                "updated_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Could not create or find watchlist")

    except Exception as e:
        logger.error("add_symbols_error", watchlist_name=watchlist_name, error=str(e))
        safe_internal_error(e, "add symbols to watchlist")


@router.delete("/{watchlist_name}")
async def delete_watchlist(watchlist_name: str, req: Request):
    """
    Delete a watchlist (soft delete).
    """
    audit = get_audit_logger()
    try:
        from src.tradingview_db_manager import TradingViewDBManager
        manager = TradingViewDBManager()

        success = manager.delete_watchlist(watchlist_name)

        if success:
            # Log watchlist deletion
            await audit.log(
                AuditEventType.WATCHLIST_DELETED,
                action=f"Watchlist deleted: {watchlist_name}",
                resource_type="watchlist",
                details={"watchlist_name": watchlist_name},
                ip_address=req.client.host if req.client else None,
            )

            return {
                "watchlist_name": watchlist_name,
                "status": "deleted",
                "message": f"Watchlist '{watchlist_name}' deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Watchlist '{watchlist_name}' not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_watchlist_error", watchlist_name=watchlist_name, error=str(e))
        safe_internal_error(e, "delete watchlist")
