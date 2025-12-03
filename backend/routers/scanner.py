"""Scanner Router - API endpoints for premium scanning
NO MOCK DATA - All endpoints use real data sources
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import json
import asyncio
import uuid
from pydantic import BaseModel
from backend.services.scanner_service import get_scanner_service, ScannerService
from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/scanner",
    tags=["scanner"]
)

# Store active scan progress in memory
_scan_progress = {}


class ScanRequest(BaseModel):
    """Request body for scanning premiums"""
    symbols: List[str]
    max_price: float = 250  # Increased to include more stocks like NVDA, TSLA
    min_premium_pct: float = 0.5  # Lowered to show more opportunities
    dte: int = 30
    save_to_db: bool = True


class MultiDTEScanRequest(BaseModel):
    """Request body for multi-DTE scanning"""
    symbols: List[str]
    max_price: float = 250  # Increased to include more stocks
    min_premium_pct: float = 0.5  # Lowered to show more opportunities
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




@router.get("/opportunities")
async def get_scanner_opportunities(
    limit: int = Query(50, description="Maximum results"),
    service: ScannerService = Depends(get_scanner_service)
):
    """
    Get recent premium opportunities from scanner history.
    """
    try:
        # Get recent scan results from database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT scan_id, symbols, dte, result_count, results, created_at
                FROM premium_scan_history
                ORDER BY created_at DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row:
                return {
                    "scan_id": row[0],
                    "symbols": row[1],
                    "dte": row[2],
                    "result_count": row[3],
                    "opportunities": row[4] if row[4] else [],
                    "generated_at": row[5].isoformat() if row[5] else None
                }
            else:
                return {
                    "opportunities": [],
                    "result_count": 0,
                    "message": "No scan results found. Run a scan first."
                }
    except Exception as e:
        logger.error(f"Error getting opportunities: {e}")
        return {"opportunities": [], "error": str(e)}

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
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT scan_id, symbols, dte, max_price, min_premium_pct, result_count, created_at
                FROM premium_scan_history
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            rows = cursor.fetchall()

            history = []
            for row in rows:
                history.append({
                    "scan_id": row[0],
                    "symbols": row[1],
                    "symbol_count": len(row[1]) if row[1] else 0,
                    "dte": row[2],
                    "max_price": float(row[3]) if row[3] else 0,
                    "min_premium_pct": float(row[4]) if row[4] else 0,
                    "result_count": row[5],
                    "created_at": row[6].isoformat() if row[6] else None
                })

            return {
                "history": history,
                "count": len(history)
            }

    except Exception as e:
        logger.error(f"Error fetching scan history: {e}")
        return {"history": [], "count": 0, "error": str(e)}


@router.get("/history/{scan_id}")
async def get_scan_by_id(scan_id: str):
    """
    Get a specific scan's full results by ID.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT scan_id, symbols, dte, max_price, min_premium_pct, result_count, results, created_at
                FROM premium_scan_history
                WHERE scan_id = %s
            """, (scan_id,))

            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Scan not found")

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching scan {scan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    min_premium_pct: float = Query(0.3, description="Minimum premium percentage"),
    max_strike: float = Query(250, description="Maximum strike price"),
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
            max_price=250,
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
    Sources: Predefined popular lists, All Stocks (stock_data), Database watchlists (tv_watchlists_api), TradingView watchlists, Robinhood.
    NO MOCK DATA - Real database queries only.
    """
    watchlists = []

    # Predefined popular watchlists for wheel strategy
    predefined_watchlists = [
        {
            "source": "predefined",
            "id": "popular",
            "name": "Popular Wheel Stocks",
            "symbols": [
                'AAPL', 'AMD', 'AMZN', 'BAC', 'C', 'CCL', 'CSCO', 'F', 'GE', 'GOOG',
                'INTC', 'META', 'MSFT', 'NVDA', 'PLTR', 'PYPL', 'SNAP', 'SOFI', 'T',
                'TSLA', 'UAL', 'UBER', 'WFC', 'XOM'
            ]
        },
        {
            "source": "predefined",
            "id": "high_iv",
            "name": "High IV Stocks",
            "symbols": [
                'MARA', 'RIOT', 'COIN', 'GME', 'AMC', 'RIVN', 'LCID', 'NIO', 'PLUG',
                'SPCE', 'BBBY', 'HOOD', 'SOFI', 'PLTR', 'SNAP', 'PINS'
            ]
        },
        {
            "source": "predefined",
            "id": "blue_chip",
            "name": "Blue Chip Stocks",
            "symbols": [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'MA',
                'JNJ', 'UNH', 'PG', 'HD', 'DIS', 'VZ', 'KO', 'PEP', 'MRK', 'WMT'
            ]
        },
        {
            "source": "predefined",
            "id": "under_25",
            "name": "Under $25 (Low Capital)",
            "symbols": [
                'F', 'SOFI', 'PLTR', 'NIO', 'SNAP', 'T', 'CCL', 'AAL', 'PLUG', 'RIOT',
                'SIRI', 'NOK', 'HOOD', 'GRAB', 'OPEN', 'WISH', 'CLOV', 'BB'
            ]
        },
        {
            "source": "predefined",
            "id": "tech_focused",
            "name": "Tech Focused",
            "symbols": [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CRM',
                'ORCL', 'ADBE', 'NOW', 'SHOP', 'SQ', 'PYPL', 'UBER', 'ABNB', 'SNOW'
            ]
        }
    ]
    watchlists.extend(predefined_watchlists)

    # All Stocks from stock_data table (comprehensive database of stocks)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT symbol
                FROM stock_data
                WHERE symbol IS NOT NULL AND symbol != ''
                ORDER BY symbol
            """)
            all_symbols = [row[0] for row in cursor.fetchall()]
            if all_symbols:
                watchlists.append({
                    "source": "database",
                    "id": "all_stocks",
                    "name": "All Stocks",
                    "symbols": all_symbols
                })
    except Exception as e:
        logger.warning(f"Error fetching all stocks from stock_data: {e}")

    # Database watchlists from tv_watchlists_api (primary synced TradingView data)
    # This table has symbols stored directly as ARRAY column in format 'EXCHANGE:SYMBOL'
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get watchlists directly from tv_watchlists_api (symbols are stored as array)
            cursor.execute("""
                SELECT
                    watchlist_id,
                    name,
                    symbols,
                    symbol_count
                FROM tv_watchlists_api
                WHERE symbol_count > 0
                ORDER BY symbol_count DESC, name
            """)
            rows = cursor.fetchall()

            for row in rows:
                watchlist_id, name, full_symbols, count = row
                if full_symbols:
                    # Filter to only stock exchanges (NYSE, NASDAQ, AMEX, ARCA, BATS)
                    # Symbols are stored as 'EXCHANGE:SYMBOL' format
                    stock_exchanges = ('NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS')
                    stock_symbols = []
                    for fs in full_symbols:
                        if ':' in fs:
                            exchange, symbol = fs.split(':', 1)
                            if exchange.upper() in stock_exchanges:
                                stock_symbols.append(symbol)

                    if stock_symbols:  # Only add if has stock symbols
                        watchlists.append({
                            "source": "database",
                            "id": f"db_{watchlist_id}",
                            "name": name,
                            "symbols": stock_symbols
                        })

    except Exception as e:
        logger.warning(f"Error fetching database watchlists (tv_watchlists_api): {e}")

    # TradingView watchlists from tradingview_watchlists table (alternate sync)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tw.id, tw.name, array_agg(ts.symbol) as symbols
                FROM tradingview_watchlists tw
                LEFT JOIN tradingview_symbols ts ON ts.watchlist_id = tw.id
                GROUP BY tw.id, tw.name
                ORDER BY tw.name
            """)
            rows = cursor.fetchall()

            for row in rows:
                watchlists.append({
                    "source": "tradingview",
                    "id": f"tv_{row[0]}",
                    "name": f"TV: {row[1]}",
                    "symbols": row[2] if row[2] and row[2][0] else []
                })

    except Exception as e:
        logger.warning(f"Error fetching TradingView watchlists: {e}")

    # Robinhood portfolio as a watchlist
    try:
        from backend.services.portfolio_service import get_portfolio_service
        service = get_portfolio_service()
        positions = await service.get_positions()

        stock_symbols = [s.get("symbol") for s in positions.get("stocks", []) if s.get("symbol")]
        if stock_symbols:
            watchlists.append({
                "source": "robinhood",
                "id": "rh_portfolio",
                "name": "RH: My Portfolio",
                "symbols": stock_symbols
            })

    except Exception as e:
        logger.warning(f"Error fetching Robinhood portfolio: {e}")

    return {
        "watchlists": watchlists,
        "total": len(watchlists),
        "generated_at": datetime.now().isoformat()
    }


# =============================================================================
# AI-POWERED ENDPOINTS
# =============================================================================

class AIScanRequest(BaseModel):
    """Request body for AI-powered scanning"""
    symbols: List[str]
    max_price: float = 250
    min_premium_pct: float = 0.5
    dte: int = 30
    min_ai_score: int = 0  # Minimum AI score to include (0-100)
    top_n: Optional[int] = None  # Return only top N results
    save_to_db: bool = True


@router.post("/scan-ai")
async def scan_with_ai(
    request: AIScanRequest,
    background_tasks: BackgroundTasks
):
    """
    AI-Powered Premium Scanner

    Scans for premium opportunities with multi-criteria AI scoring.
    Each opportunity is scored on:
    - Fundamental factors (P/E, market cap, sector)
    - Technical factors (volume, OI, bid-ask spread)
    - Greeks analysis (delta, IV, premium ratio)
    - Risk assessment (max loss, probability, breakeven)
    - Sentiment analysis

    Returns opportunities sorted by AI score with insights.
    """
    try:
        from src.ai_premium_scanner import get_ai_scanner

        scanner = get_ai_scanner()
        results = scanner.scan_with_ai(
            symbols=request.symbols,
            max_price=request.max_price,
            min_premium_pct=request.min_premium_pct,
            dte=request.dte,
            min_ai_score=request.min_ai_score,
            top_n=request.top_n
        )

        # Save to database in background if requested
        if request.save_to_db and results.get('opportunities'):
            scan_id = f"ai_{str(uuid.uuid4())[:8]}"
            save_request = ScanRequest(
                symbols=request.symbols,
                max_price=request.max_price,
                min_premium_pct=request.min_premium_pct,
                dte=request.dte,
                save_to_db=True
            )
            background_tasks.add_task(
                save_scan_results_to_db,
                scan_id,
                save_request,
                results['opportunities']
            )
            results['scan_id'] = scan_id
            results['saved'] = True

        return results

    except Exception as e:
        logger.error(f"Error in AI scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan-ai-stream")
async def scan_with_ai_stream(request: AIScanRequest):
    """
    AI-Powered Premium Scanner with streaming progress.
    Uses Server-Sent Events to report progress and AI scoring.
    """
    async def generate():
        try:
            from src.ai_premium_scanner import get_ai_scanner

            scanner = get_ai_scanner()
            scan_id = f"ai_{str(uuid.uuid4())[:8]}"
            total_symbols = len(request.symbols)

            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'scan_id': scan_id, 'total': total_symbols, 'ai_enabled': True})}\n\n"

            all_results = []
            batch_size = 3  # Smaller batches for AI processing

            for i in range(0, total_symbols, batch_size):
                batch = request.symbols[i:i + batch_size]
                completed = min(i + batch_size, total_symbols)
                progress = round((completed / total_symbols) * 100)

                # Send progress update
                yield f"data: {json.dumps({'type': 'progress', 'current': completed, 'total': total_symbols, 'percent': progress, 'symbols': batch, 'phase': 'scanning'})}\n\n"

                try:
                    # Scan batch with AI scoring
                    batch_results = scanner.scan_with_ai(
                        symbols=batch,
                        max_price=request.max_price,
                        min_premium_pct=request.min_premium_pct,
                        dte=request.dte,
                        min_ai_score=request.min_ai_score
                    )

                    batch_opps = batch_results.get('opportunities', [])
                    all_results.extend(batch_opps)

                    # Send batch results with AI info
                    yield f"data: {json.dumps({'type': 'batch', 'count': len(batch_opps), 'total_so_far': len(all_results), 'ai_scored': True})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'batch': batch, 'error': str(e)})}\n\n"

                await asyncio.sleep(0.1)

            # Sort by AI score
            all_results.sort(key=lambda x: x.get('ai_score', 0), reverse=True)

            # Apply top_n limit
            if request.top_n and request.top_n > 0:
                all_results = all_results[:request.top_n]

            # Generate final insights
            yield f"data: {json.dumps({'type': 'progress', 'phase': 'generating_insights'})}\n\n"

            final_results = scanner.scan_with_ai(
                symbols=[],  # Empty - we'll use cached results
                dte=request.dte
            )
            final_results['opportunities'] = all_results
            final_results['total_found'] = len(all_results)

            # Recalculate insights for all results
            if all_results:
                final_results['ai_insights'] = scanner._generate_insights(all_results[:10])
                final_results['stats'] = scanner._calculate_stats(all_results)

            # Save to database
            if request.save_to_db and all_results:
                save_request = ScanRequest(
                    symbols=request.symbols,
                    max_price=request.max_price,
                    min_premium_pct=request.min_premium_pct,
                    dte=request.dte,
                    save_to_db=True
                )
                save_scan_results_to_db(scan_id, save_request, all_results)

            # Send complete event
            yield f"data: {json.dumps({'type': 'complete', 'scan_id': scan_id, 'count': len(all_results), 'results': all_results, 'stats': final_results.get('stats'), 'ai_insights': final_results.get('ai_insights'), 'saved': request.save_to_db})}\n\n"

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


@router.post("/analyze-opportunity")
async def analyze_opportunity(opportunity: dict):
    """
    Get detailed AI analysis for a specific opportunity.

    Provides:
    - Multi-criteria scoring breakdown
    - Risk/reward assessment
    - LLM-generated analysis (if available)
    - Trade recommendation
    """
    try:
        from src.ai_premium_scanner import get_ai_scanner

        scanner = get_ai_scanner()

        # Score the opportunity
        scored = scanner.score_opportunity(opportunity)

        # Try to get LLM analysis
        llm_analysis = None
        try:
            import asyncio
            llm_analysis = await scanner.generate_llm_analysis(scored)
        except Exception as e:
            logger.debug(f"LLM analysis not available: {e}")

        return {
            "opportunity": scored,
            "scores": {
                "ai_score": scored.get('ai_score'),
                "fundamental": scored.get('fundamental_score'),
                "technical": scored.get('technical_score'),
                "greeks": scored.get('greeks_score'),
                "risk": scored.get('risk_score'),
                "sentiment": scored.get('sentiment_score'),
            },
            "recommendation": scored.get('ai_recommendation'),
            "confidence": scored.get('ai_confidence'),
            "llm_analysis": llm_analysis,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error analyzing opportunity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai-insights")
async def get_ai_insights(
    symbols: str = Query("AAPL,AMD,TSLA,NVDA,SOFI,PLTR", description="Comma-separated symbols"),
    dte: int = Query(30, description="Days to expiration"),
    top_n: int = Query(20, description="Number of top opportunities to analyze")
):
    """
    Get AI-generated insights for specified symbols.

    Returns:
    - Market conditions analysis
    - Top opportunity picks
    - Sector analysis
    - Risk assessment
    - Trade recommendations
    """
    try:
        from src.ai_premium_scanner import get_ai_scanner

        scanner = get_ai_scanner()
        symbol_list = [s.strip().upper() for s in symbols.split(',')]

        results = scanner.scan_with_ai(
            symbols=symbol_list,
            max_price=250,
            min_premium_pct=0.5,
            dte=dte,
            top_n=top_n
        )

        return {
            "insights": results.get('ai_insights'),
            "stats": results.get('stats'),
            "top_opportunities": results.get('opportunities', [])[:5],
            "symbols_analyzed": len(symbol_list),
            "total_found": results.get('total_found', 0),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating AI insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendation-summary")
async def get_recommendation_summary():
    """
    Get a summary of AI recommendations from recent scans.
    Groups opportunities by recommendation level.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get most recent scan results
            cursor.execute("""
                SELECT results
                FROM premium_scan_history
                ORDER BY created_at DESC
                LIMIT 1
            """)
            row = cursor.fetchone()

            if not row or not row[0]:
                return {
                    "summary": {},
                    "message": "No scan results available. Run an AI scan first."
                }

            results = row[0]

            # Group by recommendation
            summary = {
                'STRONG_BUY': [],
                'BUY': [],
                'HOLD': [],
                'CAUTION': [],
                'AVOID': []
            }

            for opp in results:
                rec = opp.get('ai_recommendation', 'HOLD')
                if rec in summary:
                    summary[rec].append({
                        'symbol': opp.get('symbol'),
                        'strike': opp.get('strike'),
                        'expiration': opp.get('expiration'),
                        'ai_score': opp.get('ai_score'),
                        'monthly_return': opp.get('monthly_return'),
                        'premium': opp.get('premium')
                    })

            # Sort each group by AI score
            for rec in summary:
                summary[rec] = sorted(
                    summary[rec],
                    key=lambda x: x.get('ai_score', 0),
                    reverse=True
                )[:10]  # Top 10 per category

            return {
                "summary": summary,
                "counts": {rec: len(opps) for rec, opps in summary.items()},
                "generated_at": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error getting recommendation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
