"""
Stock Universe Router - API endpoints for the complete stock/ETF universe
NO MOCK DATA - All endpoints use real database from stocks_universe and etfs_universe tables

Features:
- Full company information with 60+ data points
- Advanced filtering (sector, industry, market cap, volume, price, technicals)
- Pagination and sorting
- Search across all fields
- Export-ready data
"""

from fastapi import APIRouter, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import structlog

from backend.infrastructure.database import get_database
from backend.services.universe_service import get_universe_service, UniverseFilter, AssetType

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/universe",
    tags=["universe"]
)


@router.get("/stocks")
async def get_all_stocks(
    search: Optional[str] = Query(None, description="Search symbol or company name"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    min_market_cap: Optional[float] = Query(None, description="Minimum market cap (in dollars)"),
    max_market_cap: Optional[float] = Query(None, description="Maximum market cap (in dollars)"),
    min_volume: Optional[int] = Query(None, description="Minimum average volume"),
    has_options: Optional[bool] = Query(None, description="Filter by options availability"),
    sort_by: str = Query("market_cap", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(100, le=1000, description="Maximum results"),
    offset: int = Query(0, description="Pagination offset")
):
    """
    Get all stocks from the universe with comprehensive company information.
    Includes: price, volume, market cap, sector, industry, fundamentals, technicals.
    """
    try:
        db = await get_database()

        # Build dynamic query
        conditions = ["is_active = true"]
        params = []
        param_idx = 1

        if search:
            conditions.append(f"(symbol ILIKE ${param_idx} OR company_name ILIKE ${param_idx})")
            params.append(f"%{search}%")
            param_idx += 1

        if sector:
            conditions.append(f"sector = ${param_idx}")
            params.append(sector)
            param_idx += 1

        if industry:
            conditions.append(f"industry = ${param_idx}")
            params.append(industry)
            param_idx += 1

        if exchange:
            conditions.append(f"exchange = ${param_idx}")
            params.append(exchange)
            param_idx += 1

        if min_price is not None:
            conditions.append(f"current_price >= ${param_idx}")
            params.append(min_price)
            param_idx += 1

        if max_price is not None:
            conditions.append(f"current_price <= ${param_idx}")
            params.append(max_price)
            param_idx += 1

        if min_market_cap is not None:
            conditions.append(f"market_cap >= ${param_idx}")
            params.append(min_market_cap)
            param_idx += 1

        if max_market_cap is not None:
            conditions.append(f"market_cap <= ${param_idx}")
            params.append(max_market_cap)
            param_idx += 1

        if min_volume is not None:
            conditions.append(f"avg_volume_10d >= ${param_idx}")
            params.append(min_volume)
            param_idx += 1

        if has_options is not None:
            conditions.append(f"has_options = ${param_idx}")
            params.append(has_options)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Validate sort field to prevent SQL injection
        valid_sort_fields = [
            'symbol', 'company_name', 'exchange', 'sector', 'industry',
            'current_price', 'market_cap', 'volume', 'avg_volume_10d',
            'pe_ratio', 'dividend_yield', 'beta', 'week_52_high', 'week_52_low',
            'sma_50', 'sma_200', 'rsi_14', 'revenue_growth', 'earnings_growth'
        ]
        if sort_by not in valid_sort_fields:
            sort_by = 'market_cap'

        order_direction = 'DESC' if sort_order.lower() == 'desc' else 'ASC'

        # Count total for pagination
        count_query = f"SELECT COUNT(*) FROM stocks_universe WHERE {where_clause}"
        total_count = await db.fetchval(count_query, *params)

        # Fetch stocks with all columns
        query = f"""
            SELECT
                symbol, company_name, exchange, sector, industry,
                current_price, previous_close, open_price, day_high, day_low,
                week_52_high, week_52_low,
                volume, avg_volume_10d, avg_volume_3m,
                market_cap, shares_outstanding, float_shares,
                pe_ratio, forward_pe, peg_ratio, price_to_book, price_to_sales,
                enterprise_value, ev_to_ebitda, ev_to_revenue,
                profit_margin, operating_margin, gross_margin,
                roe, roa,
                revenue_growth, earnings_growth, quarterly_revenue_growth, quarterly_earnings_growth,
                dividend_yield, dividend_rate, payout_ratio, ex_dividend_date,
                total_cash, total_debt, debt_to_equity, current_ratio, quick_ratio, free_cash_flow,
                beta, sma_50, sma_200, rsi_14,
                target_high_price, target_low_price, target_mean_price,
                recommendation_key, number_of_analysts,
                has_options, implied_volatility,
                data_source, last_updated, created_at
            FROM stocks_universe
            WHERE {where_clause}
            ORDER BY {sort_by} {order_direction} NULLS LAST
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        rows = await db.fetch(query, *params)

        stocks = []
        for row in rows:
            stocks.append({
                # Basic Info
                "symbol": row["symbol"],
                "company_name": row["company_name"],
                "exchange": row["exchange"],
                "sector": row["sector"],
                "industry": row["industry"],

                # Price Data
                "current_price": float(row["current_price"]) if row["current_price"] else None,
                "previous_close": float(row["previous_close"]) if row["previous_close"] else None,
                "open_price": float(row["open_price"]) if row["open_price"] else None,
                "day_high": float(row["day_high"]) if row["day_high"] else None,
                "day_low": float(row["day_low"]) if row["day_low"] else None,
                "week_52_high": float(row["week_52_high"]) if row["week_52_high"] else None,
                "week_52_low": float(row["week_52_low"]) if row["week_52_low"] else None,

                # Volume
                "volume": int(row["volume"]) if row["volume"] else None,
                "avg_volume_10d": int(row["avg_volume_10d"]) if row["avg_volume_10d"] else None,
                "avg_volume_3m": int(row["avg_volume_3m"]) if row["avg_volume_3m"] else None,

                # Market Cap & Shares
                "market_cap": float(row["market_cap"]) if row["market_cap"] else None,
                "shares_outstanding": float(row["shares_outstanding"]) if row["shares_outstanding"] else None,
                "float_shares": float(row["float_shares"]) if row["float_shares"] else None,

                # Valuation Ratios
                "pe_ratio": float(row["pe_ratio"]) if row["pe_ratio"] else None,
                "forward_pe": float(row["forward_pe"]) if row["forward_pe"] else None,
                "peg_ratio": float(row["peg_ratio"]) if row["peg_ratio"] else None,
                "price_to_book": float(row["price_to_book"]) if row["price_to_book"] else None,
                "price_to_sales": float(row["price_to_sales"]) if row["price_to_sales"] else None,
                "enterprise_value": float(row["enterprise_value"]) if row["enterprise_value"] else None,
                "ev_to_ebitda": float(row["ev_to_ebitda"]) if row["ev_to_ebitda"] else None,
                "ev_to_revenue": float(row["ev_to_revenue"]) if row["ev_to_revenue"] else None,

                # Profitability
                "profit_margin": float(row["profit_margin"]) if row["profit_margin"] else None,
                "operating_margin": float(row["operating_margin"]) if row["operating_margin"] else None,
                "gross_margin": float(row["gross_margin"]) if row["gross_margin"] else None,
                "roe": float(row["roe"]) if row["roe"] else None,
                "roa": float(row["roa"]) if row["roa"] else None,

                # Growth
                "revenue_growth": float(row["revenue_growth"]) if row["revenue_growth"] else None,
                "earnings_growth": float(row["earnings_growth"]) if row["earnings_growth"] else None,
                "quarterly_revenue_growth": float(row["quarterly_revenue_growth"]) if row["quarterly_revenue_growth"] else None,
                "quarterly_earnings_growth": float(row["quarterly_earnings_growth"]) if row["quarterly_earnings_growth"] else None,

                # Dividends
                "dividend_yield": float(row["dividend_yield"]) if row["dividend_yield"] else None,
                "dividend_rate": float(row["dividend_rate"]) if row["dividend_rate"] else None,
                "payout_ratio": float(row["payout_ratio"]) if row["payout_ratio"] else None,
                "ex_dividend_date": row["ex_dividend_date"].isoformat() if row["ex_dividend_date"] else None,

                # Balance Sheet
                "total_cash": float(row["total_cash"]) if row["total_cash"] else None,
                "total_debt": float(row["total_debt"]) if row["total_debt"] else None,
                "debt_to_equity": float(row["debt_to_equity"]) if row["debt_to_equity"] else None,
                "current_ratio": float(row["current_ratio"]) if row["current_ratio"] else None,
                "quick_ratio": float(row["quick_ratio"]) if row["quick_ratio"] else None,
                "free_cash_flow": float(row["free_cash_flow"]) if row["free_cash_flow"] else None,

                # Technical Indicators
                "beta": float(row["beta"]) if row["beta"] else None,
                "sma_50": float(row["sma_50"]) if row["sma_50"] else None,
                "sma_200": float(row["sma_200"]) if row["sma_200"] else None,
                "rsi_14": float(row["rsi_14"]) if row["rsi_14"] else None,

                # Analyst Data
                "target_high_price": float(row["target_high_price"]) if row["target_high_price"] else None,
                "target_low_price": float(row["target_low_price"]) if row["target_low_price"] else None,
                "target_mean_price": float(row["target_mean_price"]) if row["target_mean_price"] else None,
                "recommendation_key": row["recommendation_key"],
                "number_of_analysts": row["number_of_analysts"],

                # Options
                "has_options": bool(row["has_options"]),
                "implied_volatility": float(row["implied_volatility"]) if row["implied_volatility"] else None,

                # Metadata
                "data_source": row["data_source"],
                "last_updated": row["last_updated"].isoformat() if row["last_updated"] else None
            })

        return {
            "stocks": stocks,
            "total": total_count or 0,
            "count": len(stocks),
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(stocks)) < (total_count or 0),
            "filters": {
                "search": search,
                "sector": sector,
                "industry": industry,
                "exchange": exchange,
                "min_price": min_price,
                "max_price": max_price,
                "min_market_cap": min_market_cap,
                "max_market_cap": max_market_cap,
                "min_volume": min_volume,
                "has_options": has_options
            },
            "sort": {"by": sort_by, "order": sort_order},
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Error fetching stocks universe", error=str(e))
        return {
            "stocks": [],
            "total": 0,
            "count": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/etfs")
async def get_all_etfs(
    search: Optional[str] = Query(None, description="Search symbol or fund name"),
    category: Optional[str] = Query(None, description="Filter by category"),
    fund_family: Optional[str] = Query(None, description="Filter by fund family"),
    min_assets: Optional[float] = Query(None, description="Minimum total assets"),
    max_expense: Optional[float] = Query(None, description="Maximum expense ratio"),
    has_options: Optional[bool] = Query(None, description="Filter by options availability"),
    sort_by: str = Query("total_assets", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order"),
    limit: int = Query(100, le=500, description="Maximum results"),
    offset: int = Query(0, description="Pagination offset")
):
    """
    Get all ETFs from the universe with fund information.
    """
    try:
        db = await get_database()

        conditions = ["is_active = true"]
        params = []
        param_idx = 1

        if search:
            conditions.append(f"(symbol ILIKE ${param_idx} OR fund_name ILIKE ${param_idx})")
            params.append(f"%{search}%")
            param_idx += 1

        if category:
            conditions.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        if fund_family:
            conditions.append(f"fund_family = ${param_idx}")
            params.append(fund_family)
            param_idx += 1

        if min_assets is not None:
            conditions.append(f"total_assets >= ${param_idx}")
            params.append(min_assets)
            param_idx += 1

        if max_expense is not None:
            conditions.append(f"expense_ratio <= ${param_idx}")
            params.append(max_expense)
            param_idx += 1

        if has_options is not None:
            conditions.append(f"has_options = ${param_idx}")
            params.append(has_options)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        valid_sort_fields = ['symbol', 'fund_name', 'total_assets', 'expense_ratio', 'ytd_return', 'yield_ttm', 'avg_volume_10d']
        if sort_by not in valid_sort_fields:
            sort_by = 'total_assets'

        order_direction = 'DESC' if sort_order.lower() == 'desc' else 'ASC'

        total_count = await db.fetchval(f"SELECT COUNT(*) FROM etfs_universe WHERE {where_clause}", *params)

        query = f"""
            SELECT
                symbol, fund_name, exchange, category, fund_family,
                current_price, previous_close, open_price, day_high, day_low,
                week_52_high, week_52_low, nav_price,
                volume, avg_volume_10d, avg_volume_3m,
                total_assets, expense_ratio,
                yield_ttm, ytd_return, three_year_return, five_year_return,
                holdings_count, top_holding_symbol, top_holding_weight,
                beta, sma_50, sma_200, rsi_14,
                dividend_yield, dividend_rate, ex_dividend_date,
                has_options, implied_volatility,
                data_source, last_updated
            FROM etfs_universe
            WHERE {where_clause}
            ORDER BY {sort_by} {order_direction} NULLS LAST
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        rows = await db.fetch(query, *params)

        etfs = []
        for row in rows:
            etfs.append({
                "symbol": row["symbol"],
                "fund_name": row["fund_name"],
                "exchange": row["exchange"],
                "category": row["category"],
                "fund_family": row["fund_family"],
                "current_price": float(row["current_price"]) if row["current_price"] else None,
                "nav_price": float(row["nav_price"]) if row["nav_price"] else None,
                "week_52_high": float(row["week_52_high"]) if row["week_52_high"] else None,
                "week_52_low": float(row["week_52_low"]) if row["week_52_low"] else None,
                "volume": int(row["volume"]) if row["volume"] else None,
                "avg_volume_10d": int(row["avg_volume_10d"]) if row["avg_volume_10d"] else None,
                "total_assets": float(row["total_assets"]) if row["total_assets"] else None,
                "expense_ratio": float(row["expense_ratio"]) if row["expense_ratio"] else None,
                "yield_ttm": float(row["yield_ttm"]) if row["yield_ttm"] else None,
                "ytd_return": float(row["ytd_return"]) if row["ytd_return"] else None,
                "three_year_return": float(row["three_year_return"]) if row["three_year_return"] else None,
                "five_year_return": float(row["five_year_return"]) if row["five_year_return"] else None,
                "holdings_count": row["holdings_count"],
                "top_holding_symbol": row["top_holding_symbol"],
                "top_holding_weight": float(row["top_holding_weight"]) if row["top_holding_weight"] else None,
                "beta": float(row["beta"]) if row["beta"] else None,
                "sma_50": float(row["sma_50"]) if row["sma_50"] else None,
                "sma_200": float(row["sma_200"]) if row["sma_200"] else None,
                "rsi_14": float(row["rsi_14"]) if row["rsi_14"] else None,
                "dividend_yield": float(row["dividend_yield"]) if row["dividend_yield"] else None,
                "has_options": bool(row["has_options"]),
                "implied_volatility": float(row["implied_volatility"]) if row["implied_volatility"] else None,
                "last_updated": row["last_updated"].isoformat() if row["last_updated"] else None
            })

        return {
            "etfs": etfs,
            "total": total_count or 0,
            "count": len(etfs),
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(etfs)) < (total_count or 0),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Error fetching ETFs universe", error=str(e))
        return {
            "etfs": [],
            "total": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/stats")
async def get_universe_stats():
    """
    Get summary statistics for the stock and ETF universe.
    """
    try:
        service = get_universe_service()
        return await service.get_universe_stats()
    except Exception as e:
        logger.error("Error getting universe stats", error=str(e))
        return {"error": str(e)}


@router.get("/sectors")
async def get_sectors():
    """
    Get all unique sectors in the stock universe.
    """
    try:
        db = await get_database()

        rows = await db.fetch("""
            SELECT
                sector,
                COUNT(*) as stock_count,
                ROUND(AVG(market_cap)::numeric / 1e9, 2) as avg_market_cap_b,
                ROUND(AVG(pe_ratio)::numeric, 2) as avg_pe,
                ROUND(AVG(dividend_yield)::numeric * 100, 2) as avg_div_yield_pct
            FROM stocks_universe
            WHERE sector IS NOT NULL AND sector <> '' AND is_active = true
            GROUP BY sector
            ORDER BY stock_count DESC
        """)

        sectors = []
        for row in rows:
            sectors.append({
                "sector": row["sector"],
                "stock_count": row["stock_count"],
                "avg_market_cap_billions": float(row["avg_market_cap_b"]) if row["avg_market_cap_b"] else None,
                "avg_pe_ratio": float(row["avg_pe"]) if row["avg_pe"] else None,
                "avg_dividend_yield_pct": float(row["avg_div_yield_pct"]) if row["avg_div_yield_pct"] else None
            })

        return {
            "sectors": sectors,
            "total": len(sectors),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Error getting sectors", error=str(e))
        return {"sectors": [], "error": str(e)}


@router.get("/industries")
async def get_industries(sector: Optional[str] = Query(None, description="Filter by sector")):
    """
    Get all unique industries, optionally filtered by sector.
    """
    try:
        db = await get_database()

        if sector:
            rows = await db.fetch("""
                SELECT
                    industry,
                    sector,
                    COUNT(*) as stock_count
                FROM stocks_universe
                WHERE industry IS NOT NULL AND industry <> ''
                AND sector = $1 AND is_active = true
                GROUP BY industry, sector
                ORDER BY stock_count DESC
            """, sector)
        else:
            rows = await db.fetch("""
                SELECT
                    industry,
                    sector,
                    COUNT(*) as stock_count
                FROM stocks_universe
                WHERE industry IS NOT NULL AND industry <> '' AND is_active = true
                GROUP BY industry, sector
                ORDER BY stock_count DESC
                LIMIT 200
            """)

        industries = []
        for row in rows:
            industries.append({
                "industry": row["industry"],
                "sector": row["sector"],
                "stock_count": row["stock_count"]
            })

        return {
            "industries": industries,
            "total": len(industries),
            "sector_filter": sector,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Error getting industries", error=str(e))
        return {"industries": [], "error": str(e)}


@router.get("/exchanges")
async def get_exchanges():
    """
    Get all exchanges with stock counts.
    """
    try:
        db = await get_database()

        rows = await db.fetch("""
            SELECT
                exchange,
                COUNT(*) as stock_count
            FROM stocks_universe
            WHERE exchange IS NOT NULL AND is_active = true
            GROUP BY exchange
            ORDER BY stock_count DESC
        """)

        return {
            "exchanges": [{"exchange": r["exchange"], "count": r["stock_count"]} for r in rows],
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Error getting exchanges", error=str(e))
        return {"exchanges": [], "error": str(e)}


@router.get("/search")
async def search_universe(
    q: str = Query(..., min_length=1, description="Search query"),
    asset_type: str = Query("all", description="Filter: stocks, etfs, or all"),
    limit: int = Query(50, description="Maximum results")
):
    """
    Search stocks and ETFs by symbol or name.
    """
    try:
        db = await get_database()
        search_pattern = f"%{q.upper()}%"
        results = []

        if asset_type in ("stocks", "all"):
            stock_rows = await db.fetch("""
                SELECT symbol, company_name, exchange, sector, current_price, market_cap, 'stock' as type
                FROM stocks_universe
                WHERE (symbol ILIKE $1 OR company_name ILIKE $1) AND is_active = true
                ORDER BY market_cap DESC NULLS LAST
                LIMIT $2
            """, search_pattern, limit)

            for row in stock_rows:
                results.append({
                    "symbol": row["symbol"],
                    "name": row["company_name"],
                    "exchange": row["exchange"],
                    "sector": row["sector"],
                    "price": float(row["current_price"]) if row["current_price"] else None,
                    "market_cap": float(row["market_cap"]) if row["market_cap"] else None,
                    "type": "stock"
                })

        if asset_type in ("etfs", "all"):
            etf_rows = await db.fetch("""
                SELECT symbol, fund_name, exchange, category, current_price, total_assets, 'etf' as type
                FROM etfs_universe
                WHERE (symbol ILIKE $1 OR fund_name ILIKE $1) AND is_active = true
                ORDER BY total_assets DESC NULLS LAST
                LIMIT $2
            """, search_pattern, limit)

            for row in etf_rows:
                results.append({
                    "symbol": row["symbol"],
                    "name": row["fund_name"],
                    "exchange": row["exchange"],
                    "category": row["category"],
                    "price": float(row["current_price"]) if row["current_price"] else None,
                    "total_assets": float(row["total_assets"]) if row["total_assets"] else None,
                    "type": "etf"
                })

        return {
            "query": q,
            "results": results[:limit],
            "count": len(results[:limit]),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Error searching universe", error=str(e))
        return {"query": q, "results": [], "error": str(e)}


@router.get("/symbol/{symbol}")
async def get_symbol_details(symbol: str):
    """
    Get full details for a specific symbol (stock or ETF).
    """
    try:
        db = await get_database()
        symbol = symbol.upper()

        # Try stocks first
        stock_row = await db.fetchrow("""
            SELECT * FROM stocks_universe WHERE symbol = $1
        """, symbol)

        if stock_row:
            return {
                "symbol": symbol,
                "type": "stock",
                "data": dict(stock_row),
                "generated_at": datetime.now().isoformat()
            }

        # Try ETFs
        etf_row = await db.fetchrow("""
            SELECT * FROM etfs_universe WHERE symbol = $1
        """, symbol)

        if etf_row:
            return {
                "symbol": symbol,
                "type": "etf",
                "data": dict(etf_row),
                "generated_at": datetime.now().isoformat()
            }

        return {
            "symbol": symbol,
            "type": None,
            "data": None,
            "error": f"Symbol {symbol} not found in universe",
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Error getting symbol details", symbol=symbol, error=str(e))
        return {"symbol": symbol, "error": str(e)}
