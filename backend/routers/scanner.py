"""Scanner Router - API endpoints for premium scanning
NO MOCK DATA - All endpoints use real data sources

MODERNIZED: Uses native async database (asyncpg) for non-blocking DB calls
Updated: 2025-12-04
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import structlog
import json
import asyncio
import uuid
import hashlib
from pydantic import BaseModel
from backend.services.scanner_service import get_scanner_service, ScannerService
from backend.infrastructure.database import get_database, AsyncDatabaseManager
from backend.infrastructure.errors import safe_internal_error
from backend.infrastructure.cache import get_cache

logger = structlog.get_logger(__name__)

# ============ Cache TTLs (seconds) ============
CACHE_TTL_WATCHLISTS = 300       # 5 minutes - watchlists rarely change
CACHE_TTL_HISTORY = 60           # 1 minute - scan history
CACHE_TTL_STORED_PREMIUMS = 600  # 10 minutes - expensive scans, worth caching longer
CACHE_TTL_DTE_COMPARISON = 600   # 10 minutes - expensive multi-DTE scan
CACHE_TTL_PREMIUM_STATS = 120    # 2 minutes - aggregate stats

# ============ Timeout Configuration (seconds) ============
TIMEOUT_LIVE_SCAN_MAX = 120      # Max 2 minutes for any live scan
TIMEOUT_PER_SYMBOL = 10          # 10 seconds per symbol (used to calculate adaptive timeout)
TIMEOUT_MIN = 30                 # Minimum 30 seconds for any scan
TIMEOUT_DB_QUERY = 30            # Database query timeout


# ============ Async Database Helper Functions (Native asyncpg) ============

async def _fetch_scan_history(db: AsyncDatabaseManager, limit: int) -> Dict[str, Any]:
    """Async function to fetch scan history using native asyncpg."""
    rows = await db.fetch("""
        SELECT scan_id, symbols, dte, max_price, min_premium_pct, result_count, created_at
        FROM premium_scan_history
        ORDER BY created_at DESC
        LIMIT $1
    """, limit)

    history = [{
        "scan_id": row["scan_id"],
        "symbols": row["symbols"],
        "symbol_count": len(row["symbols"]) if row["symbols"] else 0,
        "dte": row["dte"],
        "max_price": float(row["max_price"]) if row["max_price"] else 0,
        "min_premium_pct": float(row["min_premium_pct"]) if row["min_premium_pct"] else 0,
        "result_count": row["result_count"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None
    } for row in rows]

    return {"history": history, "count": len(history)}


async def _fetch_scan_by_id(db: AsyncDatabaseManager, scan_id: str) -> Optional[Dict[str, Any]]:
    """Async function to fetch a scan by ID using native asyncpg."""
    row = await db.fetchrow("""
        SELECT scan_id, symbols, dte, max_price, min_premium_pct, result_count, results, created_at
        FROM premium_scan_history
        WHERE scan_id = $1
    """, scan_id)

    if not row:
        return None

    return {
        "scan_id": row["scan_id"],
        "symbols": row["symbols"],
        "dte": row["dte"],
        "max_price": float(row["max_price"]) if row["max_price"] else 0,
        "min_premium_pct": float(row["min_premium_pct"]) if row["min_premium_pct"] else 0,
        "result_count": row["result_count"],
        "results": row["results"] if row["results"] else [],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None
    }


async def _persist_scan_results(db: AsyncDatabaseManager, results: List[Dict[str, Any]]) -> int:
    """
    Persist live scan results to premium_opportunities table for future caching.
    Uses upsert to update existing records or insert new ones.
    Returns number of records persisted.
    """
    if not results:
        return 0

    persisted = 0
    for opp in results:
        try:
            # Convert expiration string to date object if needed
            expiration = opp.get('expiration')
            if isinstance(expiration, str):
                expiration = datetime.strptime(expiration, '%Y-%m-%d').date()

            # Upsert: update if exists, insert if not
            await db.execute("""
                INSERT INTO premium_opportunities (
                    symbol, option_type, strike, expiration, dte,
                    stock_price, bid, ask, mid, premium, premium_pct,
                    annualized_return, monthly_return, delta, gamma, theta, vega,
                    implied_volatility, volume, open_interest, break_even, pop, last_updated
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10, $11,
                    $12, $13, $14, $15, $16, $17,
                    $18, $19, $20, $21, $22, NOW()
                )
                ON CONFLICT (symbol, strike, expiration, option_type)
                DO UPDATE SET
                    dte = EXCLUDED.dte,
                    stock_price = EXCLUDED.stock_price,
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    mid = EXCLUDED.mid,
                    premium = EXCLUDED.premium,
                    premium_pct = EXCLUDED.premium_pct,
                    annualized_return = EXCLUDED.annualized_return,
                    monthly_return = EXCLUDED.monthly_return,
                    delta = EXCLUDED.delta,
                    gamma = EXCLUDED.gamma,
                    theta = EXCLUDED.theta,
                    vega = EXCLUDED.vega,
                    implied_volatility = EXCLUDED.implied_volatility,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest,
                    break_even = EXCLUDED.break_even,
                    pop = EXCLUDED.pop,
                    last_updated = NOW()
            """,
                opp.get('symbol'),
                opp.get('option_type', 'PUT'),
                opp.get('strike'),
                expiration,
                opp.get('dte'),
                opp.get('stock_price'),
                opp.get('bid'),
                opp.get('ask'),
                opp.get('mid'),
                opp.get('premium'),
                opp.get('premium_pct'),
                opp.get('annualized_return'),
                opp.get('monthly_return'),
                opp.get('delta'),
                opp.get('gamma'),
                opp.get('theta'),
                opp.get('vega'),
                opp.get('implied_volatility'),
                opp.get('volume'),
                opp.get('open_interest'),
                opp.get('break_even'),
                opp.get('pop')
            )
            persisted += 1
        except Exception as e:
            logger.warning("persist_scan_result_failed",
                          symbol=opp.get('symbol'),
                          error=str(e)[:100])

    if persisted > 0:
        logger.info("scan_results_persisted", count=persisted)

    return persisted


router = APIRouter(
    prefix="/api/scanner",
    tags=["scanner"]
)

# Store active scan progress in memory
_scan_progress = {}


class ScanRequest(BaseModel):
    """Request body for scanning premiums"""
    symbols: List[str]
    max_price: float = 50
    min_premium_pct: float = 1.0
    dte: int = 30
    save_to_db: bool = True


class MultiDTEScanRequest(BaseModel):
    """Request body for multi-DTE scanning"""
    symbols: List[str]
    max_price: float = 50
    min_premium_pct: float = 1.0
    dte_targets: List[int] = [7, 14, 30, 45]


async def save_scan_results_to_db(scan_id: str, request: ScanRequest, results: List[dict]):
    """Save scan results to database for persistence (async)."""
    try:
        db = await get_database()

        # Create table if not exists
        await db.execute("""
            CREATE TABLE IF NOT EXISTS premium_scan_history (
                id SERIAL PRIMARY KEY,
                scan_id VARCHAR(50) UNIQUE NOT NULL,
                symbols TEXT[],
                dte INTEGER,
                max_price NUMERIC(10,2),
                min_premium_pct NUMERIC(5,2),
                result_count INTEGER,
                results JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Insert scan results using transaction
        async with db.transaction() as conn:
            await conn.execute("""
                INSERT INTO premium_scan_history
                (scan_id, symbols, dte, max_price, min_premium_pct, result_count, results, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (scan_id) DO UPDATE SET
                    results = EXCLUDED.results,
                    result_count = EXCLUDED.result_count,
                    created_at = NOW()
            """,
                scan_id,
                request.symbols,
                request.dte,
                request.max_price,
                request.min_premium_pct,
                len(results),
                json.dumps(results)
            )

        logger.info("scan_results_saved", scan_id=scan_id, result_count=len(results))

        # Also save individual premiums to stock_premiums table
        await save_premiums_to_db(scan_id, results)

    except Exception as e:
        logger.error("save_scan_results_error", error=str(e), scan_id=scan_id)


async def save_premiums_to_db(scan_id: str, results: List[dict], watchlist_source: str = None):
    """Save individual premium results to stock_premiums table for granular queries (async)."""
    if not results:
        return

    try:
        db = await get_database()

        # Prepare batch data for efficient upsert
        records = []
        for result in results:
            records.append({
                'symbol': result.get('symbol'),
                'strike': result.get('strike'),
                'expiration': result.get('expiration'),
                'option_type': 'PUT',  # Premium scanner focuses on puts for CSP
                'stock_price': result.get('stock_price'),
                'bid': result.get('bid'),
                'ask': result.get('ask'),
                'premium': result.get('premium'),
                'premium_pct': result.get('premium_pct'),
                'iv': result.get('iv'),
                'delta': result.get('delta'),
                'theta': result.get('theta'),
                'dte': result.get('dte'),
                'monthly_return': result.get('monthly_return'),
                'annual_return': result.get('annual_return'),
                'volume': result.get('volume'),
                'open_interest': result.get('open_interest'),
                'liquidity_score': result.get('liquidity_score'),
                'bid_ask_spread': result.get('bid_ask_spread'),
                'spread_pct': result.get('spread_pct'),
                'spread_quality': result.get('spread_quality'),
                'otm_pct': result.get('otm_pct'),
                'collateral': result.get('collateral'),
                'scan_id': scan_id,
                'watchlist_source': watchlist_source
            })

        # Use executemany for batch insert
        if records:
            await db.executemany("""
                INSERT INTO stock_premiums (
                    symbol, strike, expiration, option_type, stock_price,
                    bid, ask, premium, premium_pct,
                    iv, delta, theta,
                    dte, monthly_return, annual_return,
                    volume, open_interest, liquidity_score,
                    bid_ask_spread, spread_pct, spread_quality,
                    otm_pct, collateral,
                    scan_id, watchlist_source, scanned_at
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9,
                    $10, $11, $12,
                    $13, $14, $15,
                    $16, $17, $18,
                    $19, $20, $21,
                    $22, $23,
                    $24, $25, NOW()
                )
                ON CONFLICT (symbol, strike, expiration, option_type, scanned_at) DO NOTHING
            """, [
                (
                    r['symbol'], r['strike'], r['expiration'], r['option_type'], r['stock_price'],
                    r['bid'], r['ask'], r['premium'], r['premium_pct'],
                    r['iv'], r['delta'], r['theta'],
                    r['dte'], r['monthly_return'], r['annual_return'],
                    r['volume'], r['open_interest'], r['liquidity_score'],
                    r['bid_ask_spread'], r['spread_pct'], r['spread_quality'],
                    r['otm_pct'], r['collateral'],
                    r['scan_id'], r['watchlist_source']
                ) for r in records
            ])

            logger.info("premiums_saved", count=len(records), scan_id=scan_id)

    except Exception as e:
        logger.error("save_premiums_error", error=str(e), scan_id=scan_id)


async def _fetch_latest_premiums(db: AsyncDatabaseManager, symbol: str = None, limit: int = 100) -> List[Dict]:
    """Fetch latest premiums from stock_premiums table (async)."""
    if symbol:
        rows = await db.fetch("""
            SELECT symbol, strike, expiration, stock_price, premium, premium_pct,
                   iv, delta, dte, monthly_return, annual_return,
                   volume, open_interest, liquidity_score, spread_quality,
                   otm_pct, collateral, scanned_at
            FROM stock_premiums
            WHERE symbol = $1
            ORDER BY scanned_at DESC, annual_return DESC
            LIMIT $2
        """, symbol.upper(), limit)
    else:
        rows = await db.fetch("""
            SELECT symbol, strike, expiration, stock_price, premium, premium_pct,
                   iv, delta, dte, monthly_return, annual_return,
                   volume, open_interest, liquidity_score, spread_quality,
                   otm_pct, collateral, scanned_at
            FROM stock_premiums
            ORDER BY scanned_at DESC, annual_return DESC
            LIMIT $1
        """, limit)

    return [{
        "symbol": row["symbol"],
        "strike": float(row["strike"]) if row["strike"] else 0,
        "expiration": row["expiration"].isoformat() if row["expiration"] else None,
        "stock_price": float(row["stock_price"]) if row["stock_price"] else 0,
        "premium": float(row["premium"]) if row["premium"] else 0,
        "premium_pct": float(row["premium_pct"]) if row["premium_pct"] else 0,
        "iv": float(row["iv"]) if row["iv"] else 0,
        "delta": float(row["delta"]) if row["delta"] else 0,
        "dte": row["dte"],
        "monthly_return": float(row["monthly_return"]) if row["monthly_return"] else 0,
        "annual_return": float(row["annual_return"]) if row["annual_return"] else 0,
        "volume": row["volume"],
        "open_interest": row["open_interest"],
        "liquidity_score": row["liquidity_score"],
        "spread_quality": row["spread_quality"],
        "otm_pct": float(row["otm_pct"]) if row["otm_pct"] else 0,
        "collateral": float(row["collateral"]) if row["collateral"] else 0,
        "scanned_at": row["scanned_at"].isoformat() if row["scanned_at"] else None
    } for row in rows]


async def _fetch_top_premiums(db: AsyncDatabaseManager, min_annual_return: float = 20, limit: int = 50) -> List[Dict]:
    """Fetch top performing premiums from last 24 hours (async)."""
    rows = await db.fetch("""
        SELECT DISTINCT ON (symbol, strike, expiration)
               symbol, strike, expiration, stock_price, premium, premium_pct,
               iv, delta, dte, monthly_return, annual_return,
               volume, open_interest, liquidity_score, spread_quality,
               otm_pct, collateral, scanned_at
        FROM stock_premiums
        WHERE scanned_at > NOW() - INTERVAL '24 hours'
          AND annual_return >= $1
        ORDER BY symbol, strike, expiration, scanned_at DESC
        LIMIT $2
    """, min_annual_return, limit)

    return [{
        "symbol": row["symbol"],
        "strike": float(row["strike"]) if row["strike"] else 0,
        "expiration": row["expiration"].isoformat() if row["expiration"] else None,
        "stock_price": float(row["stock_price"]) if row["stock_price"] else 0,
        "premium": float(row["premium"]) if row["premium"] else 0,
        "premium_pct": float(row["premium_pct"]) if row["premium_pct"] else 0,
        "iv": float(row["iv"]) if row["iv"] else 0,
        "delta": float(row["delta"]) if row["delta"] else 0,
        "dte": row["dte"],
        "monthly_return": float(row["monthly_return"]) if row["monthly_return"] else 0,
        "annual_return": float(row["annual_return"]) if row["annual_return"] else 0,
        "volume": row["volume"],
        "open_interest": row["open_interest"],
        "liquidity_score": row["liquidity_score"],
        "spread_quality": row["spread_quality"],
        "otm_pct": float(row["otm_pct"]) if row["otm_pct"] else 0,
        "collateral": float(row["collateral"]) if row["collateral"] else 0,
        "scanned_at": row["scanned_at"].isoformat() if row["scanned_at"] else None
    } for row in rows]


@router.post("/scan")
async def scan_premiums(
    request: ScanRequest,
    background_tasks: BackgroundTasks,
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Scan for premium opportunities using real options data.
    Returns a list of options sorted by monthly return.
    """
    try:
        scan_id = str(uuid.uuid4())[:8]

        results = service.scan_premiums(
            symbols=request.symbols,
            max_price=request.max_price,
            min_premium_pct=request.min_premium_pct,
            dte=request.dte
        )

        # Save to database in background
        if request.save_to_db:
            background_tasks.add_task(save_scan_results_to_db, scan_id, request, results)

        return {
            "scan_id": scan_id,
            "count": len(results),
            "dte": request.dte,
            "results": results,
            "saved": request.save_to_db
        }
    except Exception as e:
        logger.error(f"Error scanning premiums: {e}")
        safe_internal_error(e, "scan premiums")


@router.post("/scan-stream")
async def scan_premiums_stream(request: ScanRequest):
    """
    Scan for premium opportunities with streaming progress updates.
    Uses Server-Sent Events to report per-symbol progress.
    """
    async def generate():
        service = get_scanner_service()
        scan_id = str(uuid.uuid4())[:8]
        total_symbols = len(request.symbols)

        try:
            # Send start event with all symbols list
            yield f"data: {json.dumps({'type': 'start', 'scan_id': scan_id, 'total': total_symbols, 'symbols': request.symbols})}\n\n"

            all_results = []
            symbol_status = {}  # Track status for each symbol

            # Process each symbol individually for granular progress
            for i, symbol in enumerate(request.symbols):
                completed = i + 1
                progress = round((completed / total_symbols) * 100)

                # Mark symbol as scanning
                symbol_status[symbol] = 'scanning'
                yield f"data: {json.dumps({'type': 'symbol_start', 'symbol': symbol, 'index': i, 'current': completed, 'total': total_symbols, 'percent': progress})}\n\n"

                # Scan this symbol
                try:
                    symbol_results = service.scan_premiums(
                        symbols=[symbol],
                        max_price=request.max_price,
                        min_premium_pct=request.min_premium_pct,
                        dte=request.dte
                    )
                    all_results.extend(symbol_results)
                    found_count = len(symbol_results)

                    # Mark symbol as complete with result count
                    symbol_status[symbol] = {'status': 'complete', 'found': found_count}
                    yield f"data: {json.dumps({'type': 'symbol_complete', 'symbol': symbol, 'index': i, 'found': found_count, 'current': completed, 'total': total_symbols, 'percent': progress, 'total_so_far': len(all_results)})}\n\n"

                except Exception as e:
                    # Mark symbol as error
                    symbol_status[symbol] = {'status': 'error', 'error': str(e)}
                    yield f"data: {json.dumps({'type': 'symbol_error', 'symbol': symbol, 'index': i, 'error': str(e), 'current': completed, 'total': total_symbols, 'percent': progress})}\n\n"

                # Small delay to prevent overwhelming (allow UI to update)
                await asyncio.sleep(0.05)

            # Sort all results by monthly return
            all_results.sort(key=lambda x: x.get('monthly_return', 0), reverse=True)

            # Save to database
            if request.save_to_db:
                await save_scan_results_to_db(scan_id, request, all_results)

            # Send complete event with results and final status
            yield f"data: {json.dumps({'type': 'complete', 'scan_id': scan_id, 'count': len(all_results), 'results': all_results, 'saved': request.save_to_db, 'symbol_status': symbol_status})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )


@router.get("/history")
async def get_scan_history(limit: int = Query(10, description="Maximum scans to return")):
    """
    Get previous scan results from database.
    Uses native async database (asyncpg) for non-blocking access.

    CACHED: 1 minute
    """
    cache = get_cache()
    cache_key = f"scanner:history:{limit}"

    # Try cache first
    cached = await cache.get(cache_key)
    if cached:
        logger.debug("history_cache_hit", limit=limit)
        return cached

    try:
        db = await get_database()
        result = await _fetch_scan_history(db, limit)

        # Cache the result
        await cache.set(cache_key, result, CACHE_TTL_HISTORY)

        return result
    except Exception as e:
        logger.error("fetch_scan_history_error", error=str(e))
        return {"history": [], "count": 0, "error": str(e)}


@router.get("/history/{scan_id}")
async def get_scan_by_id(scan_id: str):
    """
    Get a specific scan's full results by ID.
    Uses native async database (asyncpg) for non-blocking access.
    """
    try:
        db = await get_database()
        result = await _fetch_scan_by_id(db, scan_id)
        if not result:
            raise HTTPException(status_code=404, detail="Scan not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("fetch_scan_by_id_error", error=str(e), scan_id=scan_id)
        safe_internal_error(e, "fetch scan history")


# Valid sort fields for stored-premiums endpoint (SQL injection prevention)
VALID_SORT_FIELDS = {
    'annualized_return': 'annualized_return DESC NULLS LAST',
    'monthly_return': 'monthly_return DESC NULLS LAST',
    'premium_pct': 'premium_pct DESC NULLS LAST',
    'dte': 'dte ASC',
    'symbol': 'symbol ASC',
    'delta': 'ABS(delta) ASC NULLS LAST'
}


@router.get("/stored-premiums")
async def get_stored_premiums(
    symbol: Optional[str] = Query(None, description="Filter by single symbol"),
    symbols: Optional[str] = Query(None, description="Filter by comma-separated symbols (for watchlist filtering)"),
    watchlist_id: Optional[str] = Query(None, description="Watchlist ID to look up symbols from scanner_watchlists table"),
    option_type: Optional[str] = Query(None, pattern="^(PUT|CALL)$", description="PUT or CALL"),
    min_premium_pct: float = Query(1.0, ge=0.0, le=100.0, description="Minimum premium percentage"),
    max_price: float = Query(500.0, ge=0.0, le=10000.0, description="Maximum stock price"),
    max_dte: int = Query(45, ge=0, le=365, description="Maximum DTE"),
    min_dte: int = Query(0, ge=0, le=365, description="Minimum DTE"),
    sort_by: str = Query("annualized_return", description="Sort field: annualized_return, monthly_return, premium_pct, dte, symbol, delta"),
    limit: int = Query(500, ge=1, le=1000, description="Maximum results"),
    live_scan: bool = Query(False, description="Force live scan even if stored data exists"),
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get stored premium opportunities from database.
    Uses native async database (asyncpg) for non-blocking access.
    Supports watchlist_id to avoid sending large symbol lists via URL.

    SMART FALLBACK: If no stored data exists for the requested symbols,
    automatically triggers a live scan using real options data.

    CACHED: 2 minutes based on query parameters
    """
    # Generate cache key from query parameters
    cache = get_cache()
    cache_params = f"{symbol}:{symbols}:{watchlist_id}:{option_type}:{min_premium_pct}:{max_price}:{max_dte}:{min_dte}:{sort_by}:{limit}"
    cache_key = f"scanner:stored_premiums:{hashlib.md5(cache_params.encode()).hexdigest()}"

    # Try cache first
    cached = await cache.get(cache_key)
    if cached:
        logger.debug("stored_premiums_cache_hit")
        return cached

    try:
        db = await get_database()

        # Validate sort field for SQL injection safety
        if sort_by not in VALID_SORT_FIELDS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sort_by value. Must be one of: {', '.join(VALID_SORT_FIELDS.keys())}"
            )
        order_clause = VALID_SORT_FIELDS[sort_by]

        # If watchlist_id provided, look up symbols from scanner_watchlists table
        symbol_list = []
        if watchlist_id:
            watchlist_row = await db.fetchrow(
                "SELECT symbols FROM scanner_watchlists WHERE watchlist_id = $1 AND is_active = true",
                watchlist_id
            )
            if watchlist_row and watchlist_row["symbols"]:
                symbol_list = watchlist_row["symbols"]
                logger.info(f"Loaded {len(symbol_list)} symbols from watchlist '{watchlist_id}'")

        # Build dynamic query with proper parameterization
        params = [min_dte, max_dte, min_premium_pct, max_price]
        param_idx = 5

        base_query = """
            SELECT symbol, company_name, option_type, strike, expiration, dte,
                   stock_price, bid, ask, mid, premium, premium_pct,
                   annualized_return, monthly_return,
                   delta, gamma, theta, vega, implied_volatility,
                   volume, open_interest, break_even, pop, last_updated
            FROM premium_opportunities
            WHERE dte >= $1 AND dte <= $2
              AND premium_pct >= $3
              AND (stock_price IS NULL OR stock_price <= $4)
        """

        if symbol:
            base_query += f" AND symbol = ${param_idx}"
            params.append(symbol.upper())
            param_idx += 1
        elif symbol_list:
            # Use symbols from watchlist lookup
            placeholders = ', '.join(f'${param_idx + i}' for i in range(len(symbol_list)))
            base_query += f" AND symbol IN ({placeholders})"
            params.extend(symbol_list)
            param_idx += len(symbol_list)
        elif symbols:
            # Parse comma-separated symbols for watchlist filtering (fallback for small lists)
            parsed_symbols = [s.strip().upper() for s in symbols.split(',') if s.strip()]
            if parsed_symbols:
                placeholders = ', '.join(f'${param_idx + i}' for i in range(len(parsed_symbols)))
                base_query += f" AND symbol IN ({placeholders})"
                params.extend(parsed_symbols)
                param_idx += len(parsed_symbols)

        if option_type:
            base_query += f" AND option_type = ${param_idx}"
            params.append(option_type.upper())
            param_idx += 1

        base_query += f" ORDER BY {order_clause} LIMIT ${param_idx}"
        params.append(limit)

        # Skip database query if live_scan is requested
        if not live_scan:
            rows = await db.fetch(base_query, *params)
        else:
            rows = []

        results = [{
            'symbol': row["symbol"],
            'company_name': row["company_name"],
            'option_type': row["option_type"],
            'strike': float(row["strike"]) if row["strike"] else None,
            'expiration': row["expiration"].isoformat() if row["expiration"] else None,
            'dte': row["dte"],
            'stock_price': float(row["stock_price"]) if row["stock_price"] else None,
            'bid': float(row["bid"]) if row["bid"] else None,
            'ask': float(row["ask"]) if row["ask"] else None,
            'mid': float(row["mid"]) if row["mid"] else None,
            'premium': float(row["premium"]) if row["premium"] else None,
            'premium_pct': float(row["premium_pct"]) if row["premium_pct"] else None,
            'annualized_return': float(row["annualized_return"]) if row["annualized_return"] else None,
            'monthly_return': float(row["monthly_return"]) if row["monthly_return"] else None,
            'delta': float(row["delta"]) if row["delta"] else None,
            'gamma': float(row["gamma"]) if row["gamma"] else None,
            'theta': float(row["theta"]) if row["theta"] else None,
            'vega': float(row["vega"]) if row["vega"] else None,
            'implied_volatility': float(row["implied_volatility"]) if row["implied_volatility"] else None,
            'volume': row["volume"],
            'open_interest': row["open_interest"],
            'break_even': float(row["break_even"]) if row["break_even"] else None,
            'pop': float(row["pop"]) if row["pop"] else None,
            'last_updated': row["last_updated"].isoformat() if row["last_updated"] else None
        } for row in rows]

        # ============ SMART FALLBACK: Live scan when no stored data ============
        # Determine symbols for potential live scan
        scan_symbols = []
        if symbol:
            scan_symbols = [symbol.upper()]
        elif symbol_list:
            scan_symbols = symbol_list[:50]  # Limit to 50 for performance
        elif symbols:
            scan_symbols = [s.strip().upper() for s in symbols.split(',') if s.strip()][:50]

        # Trigger live scan if:
        # 1. No stored results found, OR
        # 2. live_scan=true parameter was passed, AND
        # 3. We have symbols to scan
        if len(results) == 0 and len(scan_symbols) > 0:
            # Calculate adaptive timeout based on symbol count (capped at max)
            adaptive_timeout = min(
                TIMEOUT_LIVE_SCAN_MAX,
                max(TIMEOUT_MIN, TIMEOUT_PER_SYMBOL * len(scan_symbols))
            )

            logger.info("stored_premiums_empty_triggering_live_scan",
                       symbol_count=len(scan_symbols),
                       timeout_seconds=adaptive_timeout,
                       live_scan_requested=live_scan)
            try:
                # Run live scan in thread pool with timeout protection
                loop = asyncio.get_event_loop()
                live_results = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: service.scan_premiums(
                            symbols=scan_symbols,
                            max_price=max_price,
                            min_premium_pct=min_premium_pct,
                            dte=max_dte
                        )
                    ),
                    timeout=adaptive_timeout
                )

                # Transform live results to match stored format
                results = []
                for opp in live_results:
                    # Calculate monthly return if not present
                    monthly_return = opp.get('monthly_return')
                    if monthly_return is None and opp.get('premium_pct') and opp.get('dte'):
                        monthly_return = (opp['premium_pct'] / opp['dte']) * 30 if opp['dte'] > 0 else 0

                    # Calculate annualized return if not present
                    annual_return = opp.get('annualized_return')
                    if annual_return is None and opp.get('premium_pct') and opp.get('dte'):
                        annual_return = (opp['premium_pct'] / opp['dte']) * 365 if opp['dte'] > 0 else 0

                    results.append({
                        'symbol': opp.get('symbol'),
                        'company_name': opp.get('company_name', opp.get('symbol')),
                        'option_type': opp.get('option_type', 'PUT'),
                        'strike': opp.get('strike'),
                        'expiration': opp.get('expiration'),
                        'dte': opp.get('dte'),
                        'stock_price': opp.get('stock_price', opp.get('current_price')),
                        'bid': opp.get('bid'),
                        'ask': opp.get('ask'),
                        'mid': opp.get('mid', opp.get('premium')),
                        'premium': opp.get('premium'),
                        'premium_pct': opp.get('premium_pct'),
                        'annualized_return': annual_return,
                        'monthly_return': monthly_return,
                        'delta': opp.get('delta'),
                        'gamma': opp.get('gamma'),
                        'theta': opp.get('theta'),
                        'vega': opp.get('vega'),
                        'implied_volatility': opp.get('iv', opp.get('implied_volatility')),
                        'volume': opp.get('volume'),
                        'open_interest': opp.get('open_interest'),
                        'break_even': opp.get('break_even'),
                        'pop': opp.get('pop'),
                        'last_updated': datetime.now().isoformat()
                    })

                # Sort results by the requested field
                if sort_by == 'annualized_return':
                    results.sort(key=lambda x: x.get('annualized_return') or 0, reverse=True)
                elif sort_by == 'monthly_return':
                    results.sort(key=lambda x: x.get('monthly_return') or 0, reverse=True)
                elif sort_by == 'premium_pct':
                    results.sort(key=lambda x: x.get('premium_pct') or 0, reverse=True)
                elif sort_by == 'dte':
                    results.sort(key=lambda x: x.get('dte') or 999)
                elif sort_by == 'delta':
                    results.sort(key=lambda x: abs(x.get('delta') or 1))

                # Apply limit
                results = results[:limit]

                logger.info("live_scan_completed", result_count=len(results))

                # Persist results to database in background for future caching
                try:
                    await _persist_scan_results(db, results)
                except Exception as persist_error:
                    logger.warning("persist_scan_results_failed", error=str(persist_error)[:100])

            except asyncio.TimeoutError:
                logger.warning("live_scan_timeout",
                             symbol_count=len(scan_symbols),
                             timeout_seconds=adaptive_timeout)
                # Return graceful timeout response instead of error
                return {
                    'count': 0,
                    'results': [],
                    'last_updated': None,
                    'source': 'timeout',
                    'error': f'Scan timeout after {adaptive_timeout}s - try fewer symbols or increase timeout',
                    'timeout_seconds': adaptive_timeout,
                    'symbols_attempted': len(scan_symbols),
                    'filters': {
                        'symbol': symbol,
                        'option_type': option_type,
                        'min_premium_pct': min_premium_pct,
                        'max_price': max_price,
                        'min_dte': min_dte,
                        'max_dte': max_dte
                    }
                }
            except Exception as scan_error:
                logger.error("live_scan_failed", error=str(scan_error))
                # Return empty results with error info
                return {
                    'count': 0,
                    'results': [],
                    'last_updated': None,
                    'source': 'live_scan_failed',
                    'error': str(scan_error),
                    'filters': {
                        'symbol': symbol,
                        'option_type': option_type,
                        'min_premium_pct': min_premium_pct,
                        'max_price': max_price,
                        'min_dte': min_dte,
                        'max_dte': max_dte
                    }
                }

        # Get last update time
        last_update = await db.fetchval("SELECT MAX(last_updated) FROM premium_opportunities")

        # Determine data source for transparency
        data_source = 'stored' if not live_scan and len(rows) > 0 else 'live_scan'

        result = {
            'count': len(results),
            'results': results,
            'last_updated': last_update.isoformat() if last_update else datetime.now().isoformat(),
            'source': data_source,
            'filters': {
                'symbol': symbol,
                'option_type': option_type,
                'min_premium_pct': min_premium_pct,
                'max_price': max_price,
                'min_dte': min_dte,
                'max_dte': max_dte
            }
        }

        # Cache the result (shorter TTL for live scans)
        cache_ttl = CACHE_TTL_STORED_PREMIUMS if data_source == 'stored' else 60
        await cache.set(cache_key, result, cache_ttl)
        logger.debug("stored_premiums_cached", count=len(results), source=data_source)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("fetch_stored_premiums_error", error=str(e))
        return {'count': 0, 'results': [], 'error': str(e)}


@router.get("/premium-stats")
async def get_premium_stats():
    """Get statistics about stored premium data (async)."""
    try:
        db = await get_database()

        row = await db.fetchrow("""
            SELECT
                COUNT(*) as total_opportunities,
                COUNT(DISTINCT symbol) as unique_symbols,
                AVG(premium_pct) as avg_premium_pct,
                MAX(annualized_return) as max_annualized,
                MIN(last_updated) as oldest_data,
                MAX(last_updated) as newest_data
            FROM premium_opportunities
        """)

        newest_data = row["newest_data"]
        return {
            'total_opportunities': row["total_opportunities"],
            'unique_symbols': row["unique_symbols"],
            'avg_premium_pct': round(float(row["avg_premium_pct"]), 2) if row["avg_premium_pct"] else 0,
            'max_annualized_return': round(float(row["max_annualized"]), 2) if row["max_annualized"] else 0,
            'oldest_data': row["oldest_data"].isoformat() if row["oldest_data"] else None,
            'newest_data': newest_data.isoformat() if newest_data else None,
            'data_fresh': newest_data is not None and (datetime.now() - newest_data).total_seconds() < 3600
        }
    except Exception as e:
        logger.error("fetch_premium_stats_error", error=str(e))
        return {'error': str(e)}


@router.post("/scan-multi-dte")
async def scan_multi_dte(
    request: MultiDTEScanRequest,
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Scan for premiums at multiple DTE targets using real options data.
    Returns opportunities organized by DTE.
    """
    try:
        results = service.scan_multiple_dte(
            symbols=request.symbols,
            max_price=request.max_price,
            min_premium_pct=request.min_premium_pct,
            dte_targets=request.dte_targets
        )
        return {
            "dte_targets": request.dte_targets,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in multi-DTE scan: {e}")
        safe_internal_error(e, "perform multi-DTE scan")


@router.get("/quick-scan")
async def quick_scan(
    dte: int = Query(30, description="Target days to expiration"),
    limit: int = Query(20, description="Maximum results to return"),
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Quick scan using predefined watchlist with real options data.
    """
    try:
        results = service.get_quick_scan(dte=dte, limit=limit)
        return {
            "count": len(results),
            "dte": dte,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in quick scan: {e}")
        safe_internal_error(e, "perform quick scan")


@router.get("/dte")
async def dte_scanner(
    symbols: str = Query(
        "TSLA,NVDA,AMD,PLTR,SOFI,SNAP",
        description="Comma-separated symbols (max 50)",
        max_length=500
    ),
    max_dte: int = Query(7, ge=0, le=45, description="Maximum days to expiration"),
    min_premium_pct: float = Query(0.5, ge=0.0, le=100.0, description="Minimum premium percentage"),
    max_strike: float = Query(100, ge=1.0, le=10000.0, description="Maximum strike price"),
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Scan for 0-7 DTE options opportunities using real options data.
    Returns short-term theta capture opportunities.
    """
    try:
        # Validate and parse symbols with limit
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        if len(symbol_list) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 symbols allowed per request"
            )
        if not symbol_list:
            raise HTTPException(
                status_code=400,
                detail="At least one symbol is required"
            )

        # Get short-term opportunities from real scanner
        results = service.scan_premiums(
            symbols=symbol_list,
            max_price=max_strike,
            min_premium_pct=min_premium_pct,
            dte=max_dte
        )

        # Format for DTE scanner
        opportunities = []
        for r in results:
            if r.get('dte', 999) <= max_dte:
                opportunities.append({
                    "symbol": r.get("symbol", ""),
                    "company_name": r.get("company_name", r.get("symbol", "")),
                    "current_price": r.get("stock_price", 0),
                    "strike": r.get("strike", 0),
                    "expiration": r.get("expiration", ""),
                    "dte": r.get("dte", 0),
                    "bid": r.get("bid", 0),
                    "ask": r.get("ask", 0),
                    "premium_pct": r.get("premium_pct", 0),
                    "annual_return": r.get("monthly_return", 0) * 12 if r.get("monthly_return") else 0,
                    "delta": r.get("delta", 0),
                    "theta": r.get("theta", 0),
                    "iv": r.get("iv", 0),
                    "volume": r.get("volume", 0),
                    "open_interest": r.get("open_interest", 0),
                })

        return {
            "opportunities": opportunities,
            "scanned_at": datetime.now().isoformat(),
            "symbols_scanned": len(symbol_list)
        }
    except Exception as e:
        logger.error(f"Error in DTE scanner: {e}")
        safe_internal_error(e, "scan DTE opportunities")


@router.get("/dte-comparison")
async def dte_comparison(
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get premium comparison across all DTE targets using real options data.
    Returns aggregated stats for 7, 14, 30, and 45 DTE.

    CACHED: 5 minutes (expensive multi-DTE scan)
    """
    cache = get_cache()
    cache_key = "scanner:dte_comparison"

    # Try cache first - this is an expensive operation
    cached = await cache.get(cache_key)
    if cached:
        logger.debug("dte_comparison_cache_hit")
        return cached

    default_symbols = [
        'AAPL', 'AMD', 'AMZN', 'BAC', 'CSCO', 'F', 'GOOG',
        'INTC', 'META', 'MSFT', 'NVDA', 'PLTR', 'TSLA'
    ]

    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: service.scan_multiple_dte(
                symbols=default_symbols,
                max_price=200,
                min_premium_pct=0.5,
                dte_targets=[7, 14, 30, 45]
            )
        )

        # Calculate summary stats for each DTE
        summary = {}
        for dte, opportunities in results.items():
            if opportunities:
                avg_monthly = sum(o['monthly_return'] for o in opportunities) / len(opportunities)
                avg_iv = sum(o['iv'] for o in opportunities) / len(opportunities)
                best = opportunities[0] if opportunities else None
                summary[dte] = {
                    "count": len(opportunities),
                    "avg_monthly_return": round(avg_monthly, 2),
                    "avg_iv": round(avg_iv, 1),
                    "best_opportunity": best,
                    "top_5": opportunities[:5]
                }
            else:
                summary[dte] = {
                    "count": 0,
                    "avg_monthly_return": 0,
                    "avg_iv": 0,
                    "best_opportunity": None,
                    "top_5": []
                }

        result = {
            "dte_targets": [7, 14, 30, 45],
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }

        # Cache the result - this is expensive to compute
        await cache.set(cache_key, result, CACHE_TTL_DTE_COMPARISON)
        logger.info("dte_comparison_cached")

        return result
    except Exception as e:
        logger.error(f"Error in DTE comparison: {e}")
        safe_internal_error(e, "compare DTE opportunities")


# ============ Stock Premiums Database Endpoints ============

@router.get("/premiums/latest")
async def get_latest_premiums(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(100, description="Maximum results")
):
    """
    Get latest premium scan results from stock_premiums table.
    Uses native async database (asyncpg) for non-blocking access.
    """
    try:
        db = await get_database()
        results = await _fetch_latest_premiums(db, symbol, limit)
        return {
            "count": len(results),
            "results": results,
            "symbol_filter": symbol
        }
    except Exception as e:
        logger.error("fetch_latest_premiums_error", error=str(e))
        return {"count": 0, "results": [], "error": str(e)}


@router.get("/premiums/top")
async def get_top_premiums(
    min_annual_return: float = Query(20, description="Minimum annual return %"),
    limit: int = Query(50, description="Maximum results")
):
    """
    Get top performing premiums from the last 24 hours.
    Uses native async database (asyncpg) for non-blocking access.
    """
    try:
        db = await get_database()
        results = await _fetch_top_premiums(db, min_annual_return, limit)
        return {
            "count": len(results),
            "results": results,
            "min_annual_return": min_annual_return
        }
    except Exception as e:
        logger.error("fetch_top_premiums_error", error=str(e))
        return {"count": 0, "results": [], "error": str(e)}


@router.get("/premiums/stats")
async def get_stock_premiums_stats():
    """Get statistics about the stock_premiums table (async)."""
    try:
        db = await get_database()

        row = await db.fetchrow("""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT scan_id) as total_scans,
                AVG(premium_pct) as avg_premium_pct,
                AVG(annual_return) as avg_annual_return,
                MAX(annual_return) as max_annual_return,
                MIN(scanned_at) as oldest_scan,
                MAX(scanned_at) as latest_scan
            FROM stock_premiums
        """)

        return {
            "total_records": row["total_records"] or 0,
            "unique_symbols": row["unique_symbols"] or 0,
            "total_scans": row["total_scans"] or 0,
            "avg_premium_pct": round(float(row["avg_premium_pct"]), 2) if row["avg_premium_pct"] else 0,
            "avg_annual_return": round(float(row["avg_annual_return"]), 2) if row["avg_annual_return"] else 0,
            "max_annual_return": round(float(row["max_annual_return"]), 2) if row["max_annual_return"] else 0,
            "oldest_scan": row["oldest_scan"].isoformat() if row["oldest_scan"] else None,
            "latest_scan": row["latest_scan"].isoformat() if row["latest_scan"] else None
        }
    except Exception as e:
        logger.error("fetch_stock_premiums_stats_error", error=str(e))
        return {"error": str(e)}


# ============ Options Flow Endpoints ============

@router.get("/flow")
async def get_options_flow(
    symbols: str = Query(None, description="Filter by symbols (comma-separated)"),
    min_premium: float = Query(100000, description="Minimum premium in dollars"),
    sentiment: str = Query("all", description="Filter: all, bullish, bearish"),
    limit: int = Query(50, description="Maximum results")
):
    """
    Get unusual options activity from database.
    Options flow data must be populated by external data feed.
    Uses async database for non-blocking queries.
    """
    try:
        db = await get_database()

        # Build dynamic query with async placeholders
        base_query = """
            SELECT id, timestamp, symbol, strike, expiry, option_type,
                   sentiment, premium, volume, open_interest, iv, delta,
                   size_category, spot_price, bid, ask
            FROM options_flow
            WHERE premium >= $1
        """
        params = [min_premium]
        param_idx = 2

        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
            placeholders = ','.join([f'${param_idx + i}' for i in range(len(symbol_list))])
            base_query += f" AND symbol IN ({placeholders})"
            params.extend(symbol_list)
            param_idx += len(symbol_list)

        if sentiment == "bullish":
            base_query += " AND sentiment = 'Bullish'"
        elif sentiment == "bearish":
            base_query += " AND sentiment = 'Bearish'"

        base_query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
        params.append(limit)

        rows = await db.fetch(base_query, *params)

        flows = []
        for row in rows:
            flows.append({
                "id": row["id"],
                "time": row["timestamp"].strftime("%H:%M:%S") if row["timestamp"] else "",
                "symbol": row["symbol"],
                "strike": float(row["strike"]) if row["strike"] else 0,
                "expiry": str(row["expiry"]) if row["expiry"] else "",
                "type": row["option_type"],
                "sentiment": row["sentiment"],
                "premium": float(row["premium"]) if row["premium"] else 0,
                "volume": int(row["volume"]) if row["volume"] else 0,
                "open_interest": int(row["open_interest"]) if row["open_interest"] else 0,
                "iv": float(row["iv"]) if row["iv"] else 0,
                "delta": float(row["delta"]) if row["delta"] else 0,
                "size": row["size_category"],
                "spot_price": float(row["spot_price"]) if row["spot_price"] else 0,
                "bid": float(row["bid"]) if row["bid"] else 0,
                "ask": float(row["ask"]) if row["ask"] else 0
            })

        # Calculate summary stats
        total_premium = sum(f["premium"] for f in flows)
        bullish_premium = sum(f["premium"] for f in flows if f["sentiment"] == "Bullish")
        bearish_premium = sum(f["premium"] for f in flows if f["sentiment"] == "Bearish")

        return {
            "flows": flows,
            "summary": {
                "total_premium": total_premium,
                "bullish_premium": bullish_premium,
                "bearish_premium": bearish_premium,
                "bullish_pct": round(bullish_premium / total_premium * 100, 1) if total_premium > 0 else 50,
                "flow_count": len(flows),
                "unique_symbols": len(set(f["symbol"] for f in flows))
            },
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("options_flow_error", error=str(e))
        # Return empty if table doesn't exist or no data
        return {
            "flows": [],
            "summary": {
                "total_premium": 0,
                "bullish_premium": 0,
                "bearish_premium": 0,
                "bullish_pct": 50,
                "flow_count": 0,
                "unique_symbols": 0
            },
            "message": "Options flow data not available. Configure options flow data feed.",
            "generated_at": datetime.now().isoformat()
        }


# ============ Symbol Aggregator Endpoints ============

@router.get("/symbols")
async def get_aggregated_symbols():
    """
    Get all symbols from multiple sources for the premium scanner dropdown.
    Sources: Database watchlists, TradingView symbols, Robinhood positions.
    NO MOCK DATA - Real database queries only.
    Uses async database for non-blocking queries.
    """
    symbols_by_source = {
        "database": [],
        "tradingview": [],
        "robinhood": [],
        "xtrades": []
    }

    # Get TradingView symbols from database (async)
    try:
        db = await get_database()
        rows = await db.fetch("""
            SELECT DISTINCT ts.symbol, tw.name as watchlist_name
            FROM tradingview_symbols ts
            JOIN tradingview_watchlists tw ON ts.watchlist_id = tw.id
            ORDER BY tw.name, ts.symbol
        """)

        watchlist_groups = {}
        for row in rows:
            symbol, watchlist = row["symbol"], row["watchlist_name"]
            if watchlist not in watchlist_groups:
                watchlist_groups[watchlist] = []
            watchlist_groups[watchlist].append(symbol)

        symbols_by_source["tradingview"] = [
            {"watchlist": name, "symbols": syms}
            for name, syms in watchlist_groups.items()
        ]

    except Exception as e:
        logger.warning("tradingview_symbols_error", error=str(e))

    # Get Robinhood positions
    try:
        from backend.services.portfolio_service import get_portfolio_service
        service = get_portfolio_service()
        positions = await service.get_positions()

        stock_symbols = [s.get("symbol") for s in positions.get("stocks", []) if s.get("symbol")]
        option_symbols = list(set(o.get("symbol") for o in positions.get("options", []) if o.get("symbol")))

        symbols_by_source["robinhood"] = [
            {"category": "Stock Positions", "symbols": stock_symbols},
            {"category": "Option Underlyings", "symbols": option_symbols}
        ]

    except Exception as e:
        logger.warning("robinhood_positions_error", error=str(e))

    # Get XTrades symbols (from active profiles) - run in thread pool for sync code
    try:
        import asyncio
        from src.xtrades_db_manager import XtradesDBManager

        def _fetch_xtrades_symbols():
            manager = XtradesDBManager()
            profiles = manager.get_active_profiles()
            xtrades_symbols = set()
            for profile in profiles:
                trades = manager.get_trades_by_profile(profile['id'], status='open', limit=50)
                for trade in trades:
                    symbol = trade.get('ticker') or trade.get('symbol')
                    if symbol:
                        xtrades_symbols.add(symbol.upper())
            return list(xtrades_symbols)

        xtrades_syms = await asyncio.to_thread(_fetch_xtrades_symbols)
        symbols_by_source["xtrades"] = [
            {"category": "Active Trades", "symbols": xtrades_syms}
        ]

    except Exception as e:
        logger.warning("xtrades_symbols_error", error=str(e))

    # Get database stocks (if stocks table exists) - async
    try:
        db = await get_database()
        rows = await db.fetch("""
            SELECT DISTINCT symbol FROM stocks
            WHERE active = true
            ORDER BY symbol
            LIMIT 100
        """)
        symbols_by_source["database"] = [
            {"category": "Database Stocks", "symbols": [row["symbol"] for row in rows]}
        ]
    except Exception as e:
        logger.warning("database_stocks_error", error=str(e))

    # Create combined unique symbols list
    all_symbols = set()
    for source, groups in symbols_by_source.items():
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, dict) and "symbols" in group:
                    all_symbols.update(group["symbols"])

    return {
        "sources": symbols_by_source,
        "all_symbols": sorted(list(all_symbols)),
        "total_unique": len(all_symbols),
        "generated_at": datetime.now().isoformat()
    }


@router.get("/watchlists")
async def get_scanner_watchlists():
    """
    Get all available watchlists for the premium scanner.
    Reads from scanner_watchlists cache table for fast response.
    Data is populated by scripts/sync_watchlists.py

    CACHED: 5 minutes (watchlists rarely change)
    """
    cache = get_cache()
    cache_key = "scanner:watchlists"

    # Try cache first
    cached = await cache.get(cache_key)
    if cached:
        logger.debug("watchlists_cache_hit")
        return cached

    try:
        db = await get_database()
        rows = await db.fetch("""
            SELECT watchlist_id, source, name, symbols, symbol_count, last_synced
            FROM scanner_watchlists
            WHERE is_active = true
            ORDER BY sort_order ASC, name ASC
        """)

        watchlists = []
        for row in rows:
            watchlists.append({
                "source": row["source"],
                "id": row["watchlist_id"],
                "name": row["name"],
                "symbols": row["symbols"] or []
            })

        # Get last sync time
        last_sync = await db.fetchval("SELECT MAX(last_synced) FROM scanner_watchlists WHERE is_active = true")

        result = {
            "watchlists": watchlists,
            "total": len(watchlists),
            "last_synced": last_sync.isoformat() if last_sync else None,
            "generated_at": datetime.now().isoformat()
        }

        # Cache the result
        await cache.set(cache_key, result, CACHE_TTL_WATCHLISTS)
        logger.debug("watchlists_cached", count=len(watchlists))

        return result

    except Exception as e:
        logger.error(f"Error fetching watchlists from cache: {e}")
        # Fallback to minimal predefined watchlists if cache is empty
        return {
            "watchlists": [
                {
                    "source": "predefined",
                    "id": "popular",
                    "name": "Popular Stocks",
                    "symbols": ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOG", "META", "PLTR", "SOFI"]
                }
            ],
            "total": 1,
            "error": "Cache not available, showing fallback",
            "generated_at": datetime.now().isoformat()
        }


# ============ Universe-Based Endpoints ============

@router.get("/universe/stats")
async def get_universe_stats(
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get statistics about the stock and ETF universe.
    Returns counts, optionable symbols, and summary metrics.
    """
    try:
        return service.get_universe_stats()
    except Exception as e:
        logger.error(f"Error getting universe stats: {e}")
        safe_internal_error(e, "fetch universe stats")


@router.get("/universe/sectors")
async def get_sectors(
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get all available sectors from the stock universe.
    """
    try:
        sectors = service.get_sectors()
        return {
            "sectors": sectors,
            "count": len(sectors)
        }
    except Exception as e:
        logger.error(f"Error getting sectors: {e}")
        safe_internal_error(e, "fetch sectors")


@router.get("/universe/categories")
async def get_etf_categories(
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get all available ETF categories from the universe.
    """
    try:
        categories = service.get_categories()
        return {
            "categories": categories,
            "count": len(categories)
        }
    except Exception as e:
        logger.error(f"Error getting ETF categories: {e}")
        safe_internal_error(e, "fetch ETF categories")


@router.get("/universe/optionable")
async def get_optionable_symbols(
    asset_type: str = Query("all", description="stock, etf, or all"),
    max_price: Optional[float] = Query(None, description="Max stock price"),
    sectors: Optional[str] = Query(None, description="Comma-separated sectors"),
    limit: int = Query(500, description="Maximum symbols to return"),
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get all optionable symbols from the universe.
    Pre-filtered based on has_options=true in the database.
    """
    try:
        sector_list = None
        if sectors:
            sector_list = [s.strip() for s in sectors.split(',')]

        symbols = service.get_optionable_symbols(
            asset_type=asset_type,
            max_price=max_price,
            sectors=sector_list,
            limit=limit
        )
        return {
            "symbols": symbols,
            "count": len(symbols),
            "asset_type": asset_type,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting optionable symbols: {e}")
        safe_internal_error(e, "fetch optionable symbols")


class UniverseScanRequest(BaseModel):
    """Request for scanning from universe"""
    max_price: float = 100.0
    min_premium_pct: float = 1.0
    dte: int = 30
    sectors: Optional[List[str]] = None
    include_etfs: bool = True
    min_volume: int = 100000
    limit: int = 200


@router.post("/universe/scan")
async def scan_from_universe(
    request: UniverseScanRequest,
    background_tasks: BackgroundTasks,
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Scan symbols automatically from the universe.

    This is the recommended method for production scans as it:
    - Automatically filters for optionable symbols
    - Applies price and volume filters
    - Supports sector filtering
    - Includes ETFs optionally
    """
    try:
        scan_id = str(uuid.uuid4())[:8]

        results = service.scan_from_universe(
            max_price=request.max_price,
            min_premium_pct=request.min_premium_pct,
            dte=request.dte,
            sectors=request.sectors,
            include_etfs=request.include_etfs,
            min_volume=request.min_volume,
            limit=request.limit
        )

        return {
            "scan_id": scan_id,
            "count": len(results),
            "dte": request.dte,
            "sectors": request.sectors,
            "include_etfs": request.include_etfs,
            "results": results,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error scanning from universe: {e}")
        safe_internal_error(e, "scan universe")


@router.get("/universe/scan-sector/{sector}")
async def scan_by_sector(
    sector: str,
    max_price: float = Query(100.0, description="Maximum stock price"),
    min_premium_pct: float = Query(1.0, description="Minimum premium %"),
    dte: int = Query(30, description="Target DTE"),
    limit: int = Query(50, description="Maximum results"),
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Scan all optionable stocks in a specific sector.
    """
    try:
        results = service.scan_by_sector(
            sector=sector,
            max_price=max_price,
            min_premium_pct=min_premium_pct,
            dte=dte,
            limit=limit
        )
        return {
            "sector": sector,
            "count": len(results),
            "dte": dte,
            "results": results,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error scanning sector {sector}: {e}")
        safe_internal_error(e, "scan sector")


@router.get("/universe/scan-etfs")
async def scan_etfs(
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    max_price: float = Query(100.0, description="Maximum ETF price"),
    min_premium_pct: float = Query(0.5, description="Minimum premium %"),
    dte: int = Query(30, description="Target DTE"),
    limit: int = Query(50, description="Maximum results"),
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Scan ETFs for premium opportunities.
    """
    try:
        category_list = None
        if categories:
            category_list = [c.strip() for c in categories.split(',')]

        results = service.scan_etfs(
            categories=category_list,
            max_price=max_price,
            min_premium_pct=min_premium_pct,
            dte=dte,
            limit=limit
        )
        return {
            "categories": category_list,
            "count": len(results),
            "dte": dte,
            "results": results,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error scanning ETFs: {e}")
        safe_internal_error(e, "scan ETFs")


@router.get("/universe/symbol/{symbol}")
async def get_symbol_details(
    symbol: str,
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get detailed information about a symbol from the universe.
    """
    try:
        details = service.get_symbol_details(symbol.upper())
        if details:
            return {
                "symbol": symbol.upper(),
                "details": details,
                "generated_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found in universe"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting symbol details: {e}")
        safe_internal_error(e, "fetch symbol details")


@router.post("/universe/cache/invalidate")
async def invalidate_universe_cache(
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Invalidate all universe and scanner caches.
    Useful after updating universe data.
    """
    try:
        result = service.invalidate_cache()
        return {
            "status": "success",
            "message": "Cache invalidated",
            **result
        }
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        safe_internal_error(e, "invalidate cache")


# ============ AI-Powered CSP Recommendation Endpoints ============

@router.get("/ai-picks")
async def get_ai_csp_picks(
    min_dte: int = Query(7, ge=0, le=365, description="Minimum DTE"),
    max_dte: int = Query(45, ge=0, le=365, description="Maximum DTE"),
    min_premium_pct: float = Query(0.5, ge=0.0, le=100.0, description="Minimum premium percentage"),
    force_refresh: bool = Query(False, description="Force refresh ignoring cache")
):
    """
    Get AI-powered CSP recommendations using DeepSeek R1 reasoning.

    This endpoint:
    1. Fetches stored premiums from database
    2. Applies multi-criteria scoring (MCDM)
    3. Uses DeepSeek R1 for chain-of-thought analysis
    4. Returns top 5-10 AI picks with explanations

    Returns ranked picks with scores, confidence levels, and detailed reasoning.
    """
    try:
        from backend.services.ai_csp_recommender import get_ai_csp_recommender

        db = await get_database()
        recommender = get_ai_csp_recommender()

        result = await recommender.get_recommendations(
            pool=db,
            min_dte=min_dte,
            max_dte=max_dte,
            min_premium_pct=min_premium_pct,
            force_refresh=force_refresh
        )

        return result

    except Exception as e:
        logger.error("ai_picks_error", error=str(e))
        return {
            'picks': [],
            'market_context': 'Error generating AI recommendations',
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }


@router.post("/ai-picks/refresh")
async def refresh_ai_csp_picks(
    min_dte: int = Query(7, ge=0, le=365, description="Minimum DTE"),
    max_dte: int = Query(45, ge=0, le=365, description="Maximum DTE"),
    min_premium_pct: float = Query(0.5, ge=0.0, le=100.0, description="Minimum premium percentage")
):
    """
    Force refresh AI CSP recommendations.

    Clears cache and regenerates picks with latest data.
    Use sparingly as it triggers full DeepSeek analysis.
    """
    try:
        from backend.services.ai_csp_recommender import get_ai_csp_recommender

        db = await get_database()
        recommender = get_ai_csp_recommender()

        # Clear cache first
        recommender.clear_cache()

        # Generate fresh recommendations
        result = await recommender.get_recommendations(
            pool=db,
            min_dte=min_dte,
            max_dte=max_dte,
            min_premium_pct=min_premium_pct,
            force_refresh=True
        )

        return {
            'status': 'refreshed',
            **result
        }

    except Exception as e:
        logger.error("ai_picks_refresh_error", error=str(e))
        return {
            'status': 'error',
            'picks': [],
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }


@router.get("/ai-picks/status")
async def get_ai_picks_status():
    """
    Get status of AI CSP recommender service.
    Returns details about enabled features (ensemble, Monte Carlo, etc).
    """
    try:
        from backend.services.ai_csp_recommender import get_ai_csp_recommender

        recommender = get_ai_csp_recommender()

        return {
            'service': 'ai_csp_recommender',
            'version': 'v2.0-ensemble',
            'status': 'available',

            # Configuration
            'cache_ttl_seconds': recommender.CACHE_TTL_SECONDS,
            'top_candidates': recommender.TOP_CANDIDATES,
            'final_picks': recommender.FINAL_PICKS,

            # World-class features
            'features': {
                'ensemble_ai': recommender.enable_ensemble,
                'monte_carlo_simulation': recommender.enable_monte_carlo,
                'market_regime_detection': True,
                'streaming_support': True,
                'pydantic_v2_models': True,
            },

            # Models used
            'models': {
                'primary': 'DeepSeek R1 32B (Reasoning)',
                'secondary': 'Qwen 32B (Balanced)',
                'ensemble_method': 'Multi-model voting consensus',
            },

            # Cache status
            'cache_valid': recommender._is_cache_valid(),
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("ai_picks_status_error", error=str(e))
        return {
            'service': 'ai_csp_recommender',
            'status': 'error',
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }


@router.get("/ai-picks/stream")
async def stream_ai_csp_picks(
    min_dte: int = Query(7, ge=0, le=365, description="Minimum DTE"),
    max_dte: int = Query(45, ge=0, le=365, description="Maximum DTE"),
    min_premium_pct: float = Query(0.5, ge=0.0, le=100.0, description="Minimum premium percentage")
):
    """
    Stream AI CSP recommendations with real-time progress updates.

    Returns Server-Sent Events (SSE) with:
    - progress: Step-by-step analysis progress
    - pick: Individual recommendation as analyzed
    - complete: Final summary

    Perfect for real-time UI updates during analysis.
    """
    from sse_starlette.sse import EventSourceResponse
    import json

    async def generate_events():
        try:
            from backend.services.ai_csp_recommender import get_ai_csp_recommender

            db = await get_database()
            recommender = get_ai_csp_recommender()

            async for event in recommender.stream_recommendations(
                pool=db,
                min_dte=min_dte,
                max_dte=max_dte,
                min_premium_pct=min_premium_pct
            ):
                yield {
                    "event": event.get('event', 'message'),
                    "data": json.dumps(event)
                }

        except Exception as e:
            logger.error("ai_picks_stream_error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({'error': str(e)})
            }

    return EventSourceResponse(generate_events())


# ============ World-Class Enhancement Endpoints ============

@router.get("/ai-picks/explain/{symbol}")
async def explain_ai_pick(symbol: str):
    """
    Get XAI (Explainable AI) feature importance for a specific pick.

    Returns SHAP-like feature importance scores explaining
    why this recommendation was made.

    Response includes:
    - Feature name and importance score (0-1)
    - Contribution direction (positive/negative/neutral)
    - Human-readable explanation for each factor
    """
    try:
        from backend.services.ai_csp_recommender import get_ai_csp_recommender
        from backend.services.ai_csp_enhancements import get_explainable_ai

        db = await get_database()
        recommender = get_ai_csp_recommender()
        xai = get_explainable_ai()

        # Get cached recommendations
        result = await recommender.get_recommendations(pool=db, force_refresh=False)
        picks = result.get('picks', [])

        # Find the specific pick
        pick = next((p for p in picks if p.get('symbol', '').upper() == symbol.upper()), None)

        if not pick:
            return {
                'symbol': symbol.upper(),
                'error': 'Symbol not found in current AI picks',
                'features': []
            }

        # Generate feature importance
        features = xai.explain_pick(pick)
        summary = xai.get_explanation_summary(features)

        return {
            'symbol': symbol.upper(),
            'features': [f.model_dump() for f in features],
            'summary': summary,
            'ai_score': pick.get('ai_score'),
            'confidence': pick.get('confidence'),
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("xai_explain_error", symbol=symbol, error=str(e))
        return {
            'symbol': symbol.upper(),
            'error': str(e),
            'features': [],
            'generated_at': datetime.now().isoformat()
        }


@router.get("/ai-picks/performance")
async def get_ai_performance_stats(days: int = Query(30, ge=1, le=365)):
    """
    Get AI recommendation performance statistics.

    Tracks historical accuracy of AI picks to enable
    continuous learning and improvement.

    Metrics returned:
    - Win rate (profitable recommendations)
    - Assignment rate
    - Score accuracy (correlation between AI score and outcome)
    - Average confidence levels

    Use this to monitor AI recommendation quality over time.
    """
    try:
        from backend.services.ai_csp_enhancements import get_performance_tracker

        db = await get_database()
        tracker = get_performance_tracker(db)

        stats = await tracker.get_performance_stats(days=days)

        return {
            'status': 'success',
            **stats
        }

    except Exception as e:
        logger.error("performance_stats_error", error=str(e))
        return {
            'status': 'error',
            'error': str(e),
            'message': 'Performance tracking requires outcome data',
            'generated_at': datetime.now().isoformat()
        }


@router.get("/ai-picks/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str):
    """
    Get market sentiment analysis for a symbol.

    Aggregates sentiment from multiple sources:
    - Discord community discussions
    - News headlines
    - Social media mentions

    Returns sentiment score (0-1), signal (bullish/bearish/neutral),
    and source breakdown.
    """
    try:
        from backend.services.ai_csp_enhancements import get_sentiment_analyzer

        db = await get_database()
        analyzer = get_sentiment_analyzer(db)

        sentiment = await analyzer.get_sentiment(symbol.upper())

        return {
            'status': 'success',
            **sentiment
        }

    except Exception as e:
        logger.error("sentiment_error", symbol=symbol, error=str(e))
        return {
            'status': 'error',
            'symbol': symbol.upper(),
            'score': 0.5,
            'signal': 'neutral',
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }


@router.get("/ai-picks/circuit-breaker")
async def get_circuit_breaker_status():
    """
    Get circuit breaker status for AI services.

    Circuit breaker protects against cascade failures
    by stopping calls to failing AI services.

    States:
    - CLOSED: Normal operation
    - OPEN: Service failing, calls rejected
    - HALF_OPEN: Testing recovery

    Monitor this endpoint to detect AI service health issues.
    """
    try:
        from backend.services.ai_csp_enhancements import get_circuit_breaker

        breaker = get_circuit_breaker()
        status = breaker.get_status()

        return {
            'status': 'success',
            'circuit_breaker': status,
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("circuit_breaker_status_error", error=str(e))
        return {
            'status': 'error',
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }


@router.post("/ai-picks/outcome")
async def record_recommendation_outcome(
    symbol: str = Query(..., description="Stock symbol"),
    expiration: str = Query(..., description="Expiration date YYYY-MM-DD"),
    was_profitable: bool = Query(..., description="Was the trade profitable?"),
    actual_premium: float = Query(None, description="Actual premium received"),
    was_assigned: bool = Query(False, description="Was the option assigned?"),
    profit_loss: float = Query(None, description="P/L in dollars")
):
    """
    Record the outcome of an AI recommendation.

    Used to track AI accuracy over time and enable learning.
    Call this endpoint after a CSP trade closes to record:
    - Whether it was profitable
    - Actual premium received
    - Assignment status
    - Final P/L

    This data improves future recommendations.
    """
    try:
        from backend.services.ai_csp_enhancements import get_performance_tracker

        db = await get_database()
        tracker = get_performance_tracker(db)

        await tracker.record_outcome(
            symbol=symbol.upper(),
            expiration=expiration,
            was_profitable=was_profitable,
            actual_premium=actual_premium,
            was_assigned=was_assigned,
            profit_loss=profit_loss
        )

        return {
            'status': 'success',
            'message': f'Outcome recorded for {symbol.upper()} {expiration}',
            'was_profitable': was_profitable,
            'recorded_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("record_outcome_error", symbol=symbol, error=str(e))
        return {
            'status': 'error',
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }


@router.get("/ai-picks/features")
async def get_all_enhanced_features():
    """
    Get list of all enhanced AI features and their status.

    Returns configuration for all world-class features:
    - Ensemble AI (multi-model consensus)
    - Monte Carlo simulation
    - Market regime detection
    - Circuit breaker (resilience)
    - Performance tracking (learning)
    - XAI (explainability)
    - Sentiment analysis
    - WebSocket streaming
    """
    try:
        from backend.services.ai_csp_recommender import get_ai_csp_recommender
        from backend.services.ai_csp_enhancements import (
            get_circuit_breaker,
            get_performance_tracker,
            get_explainable_ai,
            get_sentiment_analyzer
        )

        recommender = get_ai_csp_recommender()
        breaker = get_circuit_breaker()

        return {
            'version': 'v2.1-world-class',
            'features': {
                # Core AI features
                'ensemble_ai': {
                    'enabled': recommender.enable_ensemble,
                    'models': ['DeepSeek R1 32B', 'Qwen 32B'],
                    'method': 'Multi-model voting consensus'
                },
                'monte_carlo_simulation': {
                    'enabled': recommender.enable_monte_carlo,
                    'simulations': 10000,
                    'output': 'Confidence intervals (5th-95th percentile)'
                },
                'market_regime_detection': {
                    'enabled': True,
                    'regimes': ['trending_up', 'trending_down', 'ranging', 'high_volatility', 'low_volatility']
                },

                # Resilience
                'circuit_breaker': {
                    'enabled': True,
                    'state': breaker.state.value,
                    'config': {
                        'failure_threshold': breaker.config.failure_threshold,
                        'recovery_timeout': breaker.config.recovery_timeout
                    }
                },

                # Learning
                'performance_tracking': {
                    'enabled': True,
                    'tracks': ['win_rate', 'score_accuracy', 'assignment_rate']
                },

                # Explainability
                'explainable_ai': {
                    'enabled': True,
                    'features_analyzed': ['delta', 'premium_yield', 'iv', 'dte', 'liquidity']
                },

                # Sentiment
                'sentiment_analysis': {
                    'enabled': True,
                    'sources': ['discord', 'news']
                },

                # Real-time
                'streaming': {
                    'sse': True,
                    'websocket': True,
                    'endpoint': '/ws/ai-picks'
                }
            },
            'scoring': {
                'mcdm': 'Multi-Criteria Decision Making (5 scorers)',
                'weights': {
                    'premium_yield': '20-25%',
                    'risk_adjusted': '20-25%',
                    'liquidity': '20-25%',
                    'delta_positioning': '20-25%',
                    'time_value': '20-25%'
                }
            },
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("features_error", error=str(e))
        return {
            'status': 'error',
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }
