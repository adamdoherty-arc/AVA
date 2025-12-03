"""
Watchlist Router - API endpoints for database and TradingView watchlists
NO MOCK DATA - All endpoints use real database queries

Performance: Uses asyncio.to_thread() for non-blocking DB calls
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio
from pydantic import BaseModel

from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)

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


# ============ Sync Helper Functions (run via asyncio.to_thread) ============

def _fetch_database_watchlists_sync() -> Dict[str, Any]:
    """Sync function to fetch watchlists - called via asyncio.to_thread()"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT w.watchlist_id, w.name, w.symbol_count, w.created_at, w.updated_at
                FROM tv_watchlists_api w
                ORDER BY w.symbol_count DESC, w.name
            """)
            rows = cursor.fetchall()
            watchlists = [{
                "id": row[0],
                "name": row[1],
                "symbol_count": row[2] or 0,
                "created_at": row[3].isoformat() if row[3] else None,
                "updated_at": row[4].isoformat() if row[4] else None
            } for row in rows]
            return {"watchlists": watchlists, "total": len(watchlists), "source": "database"}
    except Exception as e:
        logger.warning(f"Primary table failed: {e}, trying alternate")
        # Fallback to alternate table
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, symbol_count, created_at, updated_at
                FROM tv_watchlists WHERE is_active = TRUE
                ORDER BY symbol_count DESC, name
            """)
            rows = cursor.fetchall()
            watchlists = [{
                "id": row[0],
                "name": row[1],
                "symbol_count": row[2] or 0,
                "created_at": row[3].isoformat() if row[3] else None,
                "updated_at": row[4].isoformat() if row[4] else None
            } for row in rows]
            return {"watchlists": watchlists, "total": len(watchlists), "source": "database_alt"}


# ============ Database Watchlist Endpoints ============

@router.get("/database")
async def get_database_watchlists():
    """
    Get all watchlists from tv_watchlists_api table (TradingView synced data).
    Returns watchlists with symbol counts and metadata.
    Uses asyncio.to_thread() for non-blocking database access.
    """
    try:
        result = await asyncio.to_thread(_fetch_database_watchlists_sync)
        result["generated_at"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error(f"Error fetching database watchlists: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _fetch_watchlist_symbols_sync(watchlist_name: str, limit: int, stocks_only: bool) -> Dict[str, Any]:
    """Sync function to fetch watchlist symbols - called via asyncio.to_thread()"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if stocks_only:
            cursor.execute("""
                SELECT s.symbol, s.exchange, s.full_symbol, s.added_at
                FROM tv_symbols_api s
                JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
                WHERE w.name = %s AND s.exchange IN ('NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS')
                ORDER BY s.symbol LIMIT %s
            """, (watchlist_name, limit))
        else:
            cursor.execute("""
                SELECT s.symbol, s.exchange, s.full_symbol, s.added_at
                FROM tv_symbols_api s
                JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
                WHERE w.name = %s
                ORDER BY s.symbol LIMIT %s
            """, (watchlist_name, limit))

        rows = cursor.fetchall()
        symbols = [{
            "symbol": row[0],
            "exchange": row[1],
            "full_symbol": row[2],
            "added_at": row[3].isoformat() if row[3] else None
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
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Build query based on filter
            if stocks_only:
                # Filter to stock exchanges only
                cursor.execute("""
                    SELECT
                        s.symbol,
                        s.exchange,
                        s.full_symbol,
                        s.added_at
                    FROM tv_symbols_api s
                    JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
                    WHERE w.name = %s
                      AND s.exchange IN ('NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS')
                    ORDER BY s.symbol
                    LIMIT %s
                """, (watchlist_name, limit))
            else:
                # Return all symbols
                cursor.execute("""
                    SELECT
                        s.symbol,
                        s.exchange,
                        s.full_symbol,
                        s.added_at
                    FROM tv_symbols_api s
                    JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
                    WHERE w.name = %s
                    ORDER BY s.symbol
                    LIMIT %s
                """, (watchlist_name, limit))

            rows = cursor.fetchall()

            symbols = []
            for row in rows:
                symbols.append({
                    "symbol": row[0],
                    "exchange": row[1] or "Unknown",
                    "full_symbol": row[2],
                    "added_at": row[3].isoformat() if row[3] else None
                })

            return {
                "watchlist_name": watchlist_name,
                "symbols": symbols,
                "count": len(symbols),
                "generated_at": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error fetching symbols for {watchlist_name}: {e}")
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
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get watchlists from tradingview_watchlists with symbol counts from tv_watchlist_symbols
            cursor.execute("""
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

            rows = cursor.fetchall()

            watchlists = []
            for row in rows:
                watchlists.append({
                    "id": row[0],
                    "name": row[1],
                    "symbol_count": row[2] or 0,
                    "created_at": row[3].isoformat() if row[3] else None,
                    "last_synced": row[4].isoformat() if row[4] else None
                })

            return {
                "watchlists": watchlists,
                "total": len(watchlists),
                "source": "tradingview",
                "generated_at": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error fetching TradingView watchlists: {e}")
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
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get watchlist name
            cursor.execute("""
                SELECT name FROM tradingview_watchlists WHERE id = %s
            """, (watchlist_id,))
            name_row = cursor.fetchone()
            watchlist_name = name_row[0] if name_row else f"Watchlist {watchlist_id}"

            # Get symbols from tv_watchlist_symbols (correct table name)
            cursor.execute("""
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
                WHERE ts.watchlist_id = %s
                ORDER BY ts.symbol
                LIMIT %s
            """, (watchlist_id, limit))

            rows = cursor.fetchall()

            symbols = []
            for row in rows:
                symbols.append({
                    "symbol": row[0],
                    "company_name": row[1] or row[0],
                    "sector": row[2] or "Unknown",
                    "industry": row[3] or "Unknown",
                    "last_price": float(row[4]) if row[4] else 0,
                    "volume": int(row[5]) if row[5] else 0,
                    "market_cap": int(row[6]) if row[6] else 0,
                    "added_at": row[7].isoformat() if row[7] else None
                })

            return {
                "watchlist_id": watchlist_id,
                "watchlist_name": watchlist_name,
                "symbols": symbols,
                "count": len(symbols),
                "generated_at": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error fetching TradingView symbols: {e}")
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
        logger.warning(f"Could not fetch database watchlists: {e}")

    # Get TradingView watchlists
    try:
        tv_result = await get_tradingview_watchlists()
        all_watchlists["tradingview"] = tv_result.get("watchlists", [])
    except Exception as e:
        logger.warning(f"Could not fetch TradingView watchlists: {e}")

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
        with get_db_connection() as conn:
            cursor = conn.cursor()

            search_pattern = f"%{q.upper()}%"

            # Search in tv_symbols_api using actual columns
            cursor.execute("""
                SELECT DISTINCT
                    s.symbol,
                    s.exchange,
                    s.full_symbol,
                    w.name as watchlist_name
                FROM tv_symbols_api s
                JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
                WHERE s.symbol ILIKE %s
                ORDER BY s.symbol
                LIMIT %s
            """, (search_pattern, limit))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "symbol": row[0],
                    "exchange": row[1] or "Unknown",
                    "full_symbol": row[2],
                    "watchlist": row[3]
                })

            return {
                "query": q,
                "results": results,
                "count": len(results),
                "generated_at": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        return {
            "query": q,
            "results": [],
            "count": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


# ============ Watchlist Management Endpoints ============

@router.post("/create")
async def create_watchlist(request: CreateWatchlistRequest):
    """
    Create a new custom watchlist.
    """
    try:
        from src.tradingview_db_manager import TradingViewDBManager
        manager = TradingViewDBManager()

        watchlist_id = manager.create_watchlist(request.name, request.description)

        if watchlist_id and request.symbols:
            manager.add_symbols_to_watchlist(watchlist_id, request.symbols)

        return {
            "id": watchlist_id,
            "name": request.name,
            "symbols_added": len(request.symbols),
            "message": f"Watchlist '{request.name}' created successfully",
            "created_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error creating watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{watchlist_name}/symbols")
async def add_symbols_to_watchlist(watchlist_name: str, request: AddSymbolsRequest):
    """
    Add symbols to an existing watchlist.
    """
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
            return {
                "watchlist_name": watchlist_name,
                "symbols_added": count,
                "message": f"Added {count} symbols to '{watchlist_name}'",
                "updated_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Could not create or find watchlist")

    except Exception as e:
        logger.error(f"Error adding symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{watchlist_name}")
async def delete_watchlist(watchlist_name: str):
    """
    Delete a watchlist (soft delete).
    """
    try:
        from src.tradingview_db_manager import TradingViewDBManager
        manager = TradingViewDBManager()

        success = manager.delete_watchlist(watchlist_name)

        if success:
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
        logger.error(f"Error deleting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))
