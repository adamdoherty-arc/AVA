"""Scanner Router - API endpoints for premium scanning
NO MOCK DATA - All endpoints use real data sources

Performance: Uses asyncio.to_thread() for non-blocking DB calls
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import json
import asyncio
import uuid
from pydantic import BaseModel
from backend.services.scanner_service import get_scanner_service, ScannerService
from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)


# ============ Sync Helper Functions (run via asyncio.to_thread) ============

def _fetch_scan_history_sync(limit: int) -> Dict[str, Any]:
    """Sync function to fetch scan history - called via asyncio.to_thread()"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT scan_id, symbols, dte, max_price, min_premium_pct, result_count, created_at
            FROM premium_scan_history
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        history = [{
            "scan_id": row[0],
            "symbols": row[1],
            "symbol_count": len(row[1]) if row[1] else 0,
            "dte": row[2],
            "max_price": float(row[3]) if row[3] else 0,
            "min_premium_pct": float(row[4]) if row[4] else 0,
            "result_count": row[5],
            "created_at": row[6].isoformat() if row[6] else None
        } for row in rows]
        return {"history": history, "count": len(history)}


def _fetch_scan_by_id_sync(scan_id: str) -> Dict[str, Any]:
    """Sync function to fetch a scan by ID - called via asyncio.to_thread()"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT scan_id, symbols, dte, max_price, min_premium_pct, result_count, results, created_at
            FROM premium_scan_history
            WHERE scan_id = %s
        """, (scan_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "scan_id": row[0],
            "symbols": row[1],
            "dte": row[2],
            "max_price": float(row[3]) if row[3] else 0,
            "min_premium_pct": float(row[4]) if row[4] else 0,
            "result_count": row[5],
            "results": row[6] if row[6] else [],
            "created_at": row[7].isoformat() if row[7] else None
        }

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


def save_scan_results_to_db(scan_id: str, request: ScanRequest, results: List[dict]):
    """Save scan results to database for persistence."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute("""
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

            # Insert scan results
            cursor.execute("""
                INSERT INTO premium_scan_history
                (scan_id, symbols, dte, max_price, min_premium_pct, result_count, results, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (scan_id) DO UPDATE SET
                    results = EXCLUDED.results,
                    result_count = EXCLUDED.result_count,
                    created_at = NOW()
            """, (
                scan_id,
                request.symbols,
                request.dte,
                request.max_price,
                request.min_premium_pct,
                len(results),
                json.dumps(results)
            ))

            conn.commit()
            logger.info(f"Saved scan {scan_id} with {len(results)} results to database")

    except Exception as e:
        logger.error(f"Error saving scan results: {e}")


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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan-stream")
async def scan_premiums_stream(request: ScanRequest):
    """
    Scan for premium opportunities with streaming progress updates.
    Uses Server-Sent Events to report progress.
    """
    async def generate():
        service = get_scanner_service()
        scan_id = str(uuid.uuid4())[:8]
        total_symbols = len(request.symbols)

        try:
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'scan_id': scan_id, 'total': total_symbols})}\n\n"

            all_results = []
            batch_size = 5  # Process symbols in batches

            for i in range(0, total_symbols, batch_size):
                batch = request.symbols[i:i + batch_size]
                completed = min(i + batch_size, total_symbols)
                progress = round((completed / total_symbols) * 100)

                # Send progress update
                yield f"data: {json.dumps({'type': 'progress', 'current': completed, 'total': total_symbols, 'percent': progress, 'symbols': batch})}\n\n"

                # Actually scan this batch
                try:
                    batch_results = service.scan_premiums(
                        symbols=batch,
                        max_price=request.max_price,
                        min_premium_pct=request.min_premium_pct,
                        dte=request.dte
                    )
                    all_results.extend(batch_results)

                    # Send batch results
                    yield f"data: {json.dumps({'type': 'batch', 'count': len(batch_results), 'total_so_far': len(all_results)})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'batch': batch, 'error': str(e)})}\n\n"

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)

            # Sort all results by monthly return
            all_results.sort(key=lambda x: x.get('monthly_return', 0), reverse=True)

            # Save to database
            if request.save_to_db:
                save_scan_results_to_db(scan_id, request, all_results)

            # Send complete event with results
            yield f"data: {json.dumps({'type': 'complete', 'scan_id': scan_id, 'count': len(all_results), 'results': all_results, 'saved': request.save_to_db})}\n\n"

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
    Uses asyncio.to_thread() for non-blocking database access.
    """
    try:
        return await asyncio.to_thread(_fetch_scan_history_sync, limit)
    except Exception as e:
        logger.error(f"Error fetching scan history: {e}")
        return {"history": [], "count": 0, "error": str(e)}


@router.get("/history/{scan_id}")
async def get_scan_by_id(scan_id: str):
    """
    Get a specific scan's full results by ID.
    Uses asyncio.to_thread() for non-blocking database access.
    """
    try:
        result = await asyncio.to_thread(_fetch_scan_by_id_sync, scan_id)
        if not result:
            raise HTTPException(status_code=404, detail="Scan not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching scan {scan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stored-premiums")
async def get_stored_premiums(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    option_type: Optional[str] = Query(None, description="PUT or CALL"),
    min_premium_pct: float = Query(1.0, description="Minimum premium percentage"),
    max_dte: int = Query(45, description="Maximum DTE"),
    min_dte: int = Query(0, description="Minimum DTE"),
    sort_by: str = Query("annualized_return", description="Sort field"),
    limit: int = Query(100, description="Maximum results")
):
    """
    Get stored premium opportunities from database.
    These are pre-scanned and periodically updated.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Build dynamic query
            query = """
                SELECT symbol, company_name, option_type, strike, expiration, dte,
                       stock_price, bid, ask, mid, premium, premium_pct,
                       annualized_return, monthly_return,
                       delta, gamma, theta, vega, implied_volatility,
                       volume, open_interest, break_even, pop, last_updated
                FROM premium_opportunities
                WHERE dte >= %s AND dte <= %s
                  AND premium_pct >= %s
            """
            params = [min_dte, max_dte, min_premium_pct]

            if symbol:
                query += " AND symbol = %s"
                params.append(symbol.upper())

            if option_type:
                query += " AND option_type = %s"
                params.append(option_type.upper())

            # Sort options
            sort_map = {
                'annualized_return': 'annualized_return DESC NULLS LAST',
                'monthly_return': 'monthly_return DESC NULLS LAST',
                'premium_pct': 'premium_pct DESC NULLS LAST',
                'dte': 'dte ASC',
                'symbol': 'symbol ASC',
                'delta': 'ABS(delta) ASC NULLS LAST'
            }
            order = sort_map.get(sort_by, 'annualized_return DESC NULLS LAST')
            query += f" ORDER BY {order} LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    'symbol': row[0],
                    'company_name': row[1],
                    'option_type': row[2],
                    'strike': float(row[3]) if row[3] else None,
                    'expiration': row[4].isoformat() if row[4] else None,
                    'dte': row[5],
                    'stock_price': float(row[6]) if row[6] else None,
                    'bid': float(row[7]) if row[7] else None,
                    'ask': float(row[8]) if row[8] else None,
                    'mid': float(row[9]) if row[9] else None,
                    'premium': float(row[10]) if row[10] else None,
                    'premium_pct': float(row[11]) if row[11] else None,
                    'annualized_return': float(row[12]) if row[12] else None,
                    'monthly_return': float(row[13]) if row[13] else None,
                    'delta': float(row[14]) if row[14] else None,
                    'gamma': float(row[15]) if row[15] else None,
                    'theta': float(row[16]) if row[16] else None,
                    'vega': float(row[17]) if row[17] else None,
                    'implied_volatility': float(row[18]) if row[18] else None,
                    'volume': row[19],
                    'open_interest': row[20],
                    'break_even': float(row[21]) if row[21] else None,
                    'pop': float(row[22]) if row[22] else None,
                    'last_updated': row[23].isoformat() if row[23] else None
                })

            # Get last update time
            cursor.execute("SELECT MAX(last_updated) FROM premium_opportunities")
            last_update = cursor.fetchone()[0]

            return {
                'count': len(results),
                'results': results,
                'last_updated': last_update.isoformat() if last_update else None,
                'filters': {
                    'symbol': symbol,
                    'option_type': option_type,
                    'min_premium_pct': min_premium_pct,
                    'min_dte': min_dte,
                    'max_dte': max_dte
                }
            }

    except Exception as e:
        logger.error(f"Error fetching stored premiums: {e}")
        return {'count': 0, 'results': [], 'error': str(e)}


@router.get("/premium-stats")
async def get_premium_stats():
    """Get statistics about stored premium data."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_opportunities,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    AVG(premium_pct) as avg_premium_pct,
                    MAX(annualized_return) as max_annualized,
                    MIN(last_updated) as oldest_data,
                    MAX(last_updated) as newest_data
                FROM premium_opportunities
            """)
            row = cursor.fetchone()

            return {
                'total_opportunities': row[0],
                'unique_symbols': row[1],
                'avg_premium_pct': round(float(row[2]), 2) if row[2] else 0,
                'max_annualized_return': round(float(row[3]), 2) if row[3] else 0,
                'oldest_data': row[4].isoformat() if row[4] else None,
                'newest_data': row[5].isoformat() if row[5] else None,
                'data_fresh': row[5] is not None and (datetime.now() - row[5]).total_seconds() < 3600
            }
    except Exception as e:
        logger.error(f"Error fetching premium stats: {e}")
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dte")
async def dte_scanner(
    symbols: str = Query("TSLA,NVDA,AMD,PLTR,SOFI,SNAP", description="Comma-separated symbols"),
    max_dte: int = Query(7, description="Maximum days to expiration"),
    min_premium_pct: float = Query(0.5, description="Minimum premium percentage"),
    max_strike: float = Query(100, description="Maximum strike price"),
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Scan for 0-7 DTE options opportunities using real options data.
    Returns short-term theta capture opportunities.
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]

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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dte-comparison")
async def dte_comparison(
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get premium comparison across all DTE targets using real options data.
    Returns aggregated stats for 7, 14, 30, and 45 DTE.
    """
    default_symbols = [
        'AAPL', 'AMD', 'AMZN', 'BAC', 'CSCO', 'F', 'GOOG',
        'INTC', 'META', 'MSFT', 'NVDA', 'PLTR', 'TSLA'
    ]

    try:
        results = service.scan_multiple_dte(
            symbols=default_symbols,
            max_price=200,
            min_premium_pct=0.5,
            dte_targets=[7, 14, 30, 45]
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

        return {
            "dte_targets": [7, 14, 30, 45],
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in DTE comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Build query for options flow table
            query = """
                SELECT id, timestamp, symbol, strike, expiry, option_type,
                       sentiment, premium, volume, open_interest, iv, delta,
                       size_category, spot_price, bid, ask
                FROM options_flow
                WHERE premium >= %s
            """
            params = [min_premium]

            if symbols:
                symbol_list = [s.strip().upper() for s in symbols.split(',')]
                placeholders = ','.join(['%s'] * len(symbol_list))
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbol_list)

            if sentiment == "bullish":
                query += " AND sentiment = 'Bullish'"
            elif sentiment == "bearish":
                query += " AND sentiment = 'Bearish'"

            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            flows = []
            for row in rows:
                flows.append({
                    "id": row[0],
                    "time": row[1].strftime("%H:%M:%S") if row[1] else "",
                    "symbol": row[2],
                    "strike": float(row[3]) if row[3] else 0,
                    "expiry": str(row[4]) if row[4] else "",
                    "type": row[5],
                    "sentiment": row[6],
                    "premium": float(row[7]) if row[7] else 0,
                    "volume": int(row[8]) if row[8] else 0,
                    "open_interest": int(row[9]) if row[9] else 0,
                    "iv": float(row[10]) if row[10] else 0,
                    "delta": float(row[11]) if row[11] else 0,
                    "size": row[12],
                    "spot_price": float(row[13]) if row[13] else 0,
                    "bid": float(row[14]) if row[14] else 0,
                    "ask": float(row[15]) if row[15] else 0
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
        logger.error(f"Error fetching options flow: {e}")
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
    """
    symbols_by_source = {
        "database": [],
        "tradingview": [],
        "robinhood": [],
        "xtrades": []
    }

    # Get TradingView symbols from database
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get TradingView watchlist symbols
            cursor.execute("""
                SELECT DISTINCT ts.symbol, tw.name as watchlist_name
                FROM tradingview_symbols ts
                JOIN tradingview_watchlists tw ON ts.watchlist_id = tw.id
                ORDER BY tw.name, ts.symbol
            """)
            rows = cursor.fetchall()

            watchlist_groups = {}
            for row in rows:
                symbol, watchlist = row[0], row[1]
                if watchlist not in watchlist_groups:
                    watchlist_groups[watchlist] = []
                watchlist_groups[watchlist].append(symbol)

            symbols_by_source["tradingview"] = [
                {"watchlist": name, "symbols": syms}
                for name, syms in watchlist_groups.items()
            ]

    except Exception as e:
        logger.warning(f"Error fetching TradingView symbols: {e}")

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
        logger.warning(f"Error fetching Robinhood positions: {e}")

    # Get XTrades symbols (from active profiles)
    try:
        from src.xtrades_db_manager import XtradesDBManager
        manager = XtradesDBManager()
        profiles = manager.get_active_profiles()

        xtrades_symbols = set()
        for profile in profiles:
            trades = manager.get_trades_by_profile(profile['id'], status='open', limit=50)
            for trade in trades:
                symbol = trade.get('ticker') or trade.get('symbol')
                if symbol:
                    xtrades_symbols.add(symbol.upper())

        symbols_by_source["xtrades"] = [
            {"category": "Active Trades", "symbols": list(xtrades_symbols)}
        ]

    except Exception as e:
        logger.warning(f"Error fetching XTrades symbols: {e}")

    # Get database stocks (if stocks table exists)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT symbol FROM stocks
                WHERE active = true
                ORDER BY symbol
                LIMIT 100
            """)
            rows = cursor.fetchall()
            symbols_by_source["database"] = [
                {"category": "Database Stocks", "symbols": [r[0] for r in rows]}
            ]
    except Exception as e:
        logger.warning(f"Error fetching database stocks: {e}")

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
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT watchlist_id, source, name, symbols, symbol_count, last_synced
                FROM scanner_watchlists
                WHERE is_active = true
                ORDER BY sort_order ASC, name ASC
            """)
            rows = cursor.fetchall()

            watchlists = []
            for row in rows:
                watchlists.append({
                    "source": row[1],
                    "id": row[0],
                    "name": row[2],
                    "symbols": row[3] or []
                })

            # Get last sync time
            cursor.execute("SELECT MAX(last_synced) FROM scanner_watchlists WHERE is_active = true")
            last_sync = cursor.fetchone()[0]

            return {
                "watchlists": watchlists,
                "total": len(watchlists),
                "last_synced": last_sync.isoformat() if last_sync else None,
                "generated_at": datetime.now().isoformat()
            }

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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))
