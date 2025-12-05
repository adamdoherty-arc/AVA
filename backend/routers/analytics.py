"""
Analytics Router - Real trading performance analytics
NO MOCK DATA - All endpoints use real database queries

Performance: Uses async database connection pool for non-blocking DB calls
"""
from fastapi import APIRouter, Query
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import structlog
from backend.infrastructure.database import get_database, AsyncDatabaseManager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ============ Async Helper Functions ============

async def _fetch_performance_async(period: str) -> Dict[str, Any]:
    """Async function to fetch performance analytics"""
    days = {"1M": 30, "3M": 90, "1Y": 365}.get(period, 30)
    start_date = datetime.now() - timedelta(days=days)

    db = await get_database()

    # Get daily P&L from trade_journal
    rows = await db.fetch("""
        SELECT DATE(closed_at) as trade_date, SUM(realized_pnl) as daily_pnl
        FROM trade_journal
        WHERE closed_at >= $1 AND realized_pnl IS NOT NULL
        GROUP BY DATE(closed_at)
        ORDER BY trade_date ASC
    """, start_date)

    daily_pnl = []
    cumulative = 0
    for row in rows:
        daily = float(row["daily_pnl"]) if row["daily_pnl"] else 0
        cumulative += daily
        daily_pnl.append({
            "date": row["trade_date"].strftime("%Y-%m-%d") if row["trade_date"] else "",
            "pnl": round(daily, 2),
            "cumulative": round(cumulative, 2)
        })

    # Calculate win/loss stats
    stats = await db.fetchrow("""
        SELECT
            COUNT(*) as total_trades,
            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as wins,
            COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losses,
            AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
            AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss,
            MAX(realized_pnl) as best_trade,
            MIN(realized_pnl) as worst_trade,
            SUM(realized_pnl) as total_pnl
        FROM trade_journal
        WHERE closed_at >= $1 AND realized_pnl IS NOT NULL
    """, start_date)

    total_trades = int(stats["total_trades"]) if stats["total_trades"] else 0
    wins = int(stats["wins"]) if stats["wins"] else 0
    losses = int(stats["losses"]) if stats["losses"] else 0
    avg_win = float(stats["avg_win"]) if stats["avg_win"] else 0
    avg_loss = float(stats["avg_loss"]) if stats["avg_loss"] else 0
    total_pnl = float(stats["total_pnl"]) if stats["total_pnl"] else cumulative

    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0

    # Calculate max drawdown
    max_drawdown = 0
    peak = 0
    running = 0
    for item in daily_pnl:
        running += item["pnl"]
        if running > peak:
            peak = running
        drawdown = running - peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    return {
        "period": period,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": 0,
        "daily_pnl": daily_pnl,
        "total_trades": total_trades
    }


@router.get("/performance")
async def get_performance(period: str = "1M"):
    """
    Get trading performance analytics from real trade history.
    Uses async database connection pool for non-blocking access.
    """
    try:
        return await _fetch_performance_async(period)

    except Exception as e:
        logger.error("error_getting_performance", error=str(e), period=period)
        return {
            "period": period,
            "total_pnl": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "daily_pnl": [],
            "total_trades": 0,
            "message": f"No trade data available. Error: {str(e)}"
        }


@router.get("/strategies")
async def get_strategy_performance():
    """Get performance by strategy from real trade data"""
    try:
        db = await get_database()

        rows = await db.fetch("""
            SELECT
                COALESCE(strategy, 'Unknown') as strategy_name,
                COUNT(*) as trades,
                COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as wins,
                SUM(realized_pnl) as total_pnl,
                AVG(CASE WHEN realized_pnl != 0 THEN
                    (realized_pnl / NULLIF(ABS(entry_price * quantity), 0)) * 100
                END) as avg_return_pct,
                MAX(realized_pnl) as best_trade,
                MIN(realized_pnl) as worst_trade
            FROM trade_journal
            WHERE realized_pnl IS NOT NULL
            GROUP BY strategy
            ORDER BY total_pnl DESC
        """)

        strategies = []
        for row in rows:
            trades = int(row["trades"]) if row["trades"] else 0
            wins = int(row["wins"]) if row["wins"] else 0
            win_rate = wins / trades if trades > 0 else 0

            strategies.append({
                "name": row["strategy_name"] or "Unknown",
                "trades": trades,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(float(row["total_pnl"]), 2) if row["total_pnl"] else 0,
                "avg_return": round(float(row["avg_return_pct"]), 1) if row["avg_return_pct"] else 0,
                "best_trade": round(float(row["best_trade"]), 2) if row["best_trade"] else 0,
                "worst_trade": round(float(row["worst_trade"]), 2) if row["worst_trade"] else 0
            })

        return {"strategies": strategies}

    except Exception as e:
        logger.error("error_getting_strategy_performance", error=str(e))
        return {
            "strategies": [],
            "message": f"No strategy data available. Error: {str(e)}"
        }


@router.get("/metrics")
async def get_metrics():
    """Get key trading metrics from real data"""
    try:
        db = await get_database()

        # Total trades this month vs last month
        trade_counts = await db.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE closed_at >= DATE_TRUNC('month', CURRENT_DATE)) as this_month,
                COUNT(*) FILTER (WHERE closed_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                                 AND closed_at < DATE_TRUNC('month', CURRENT_DATE)) as last_month
            FROM trade_journal
        """)
        this_month_trades = int(trade_counts["this_month"]) if trade_counts["this_month"] else 0
        last_month_trades = int(trade_counts["last_month"]) if trade_counts["last_month"] else 0
        trade_change = this_month_trades - last_month_trades

        # Active positions
        active_row = await db.fetchrow("""
            SELECT COUNT(*) as count FROM trade_journal WHERE status = 'open'
        """)
        active_positions = int(active_row["count"]) if active_row["count"] else 0

        # Win rate this month vs last month
        wr_data = await db.fetchrow("""
            SELECT
                AVG(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) FILTER
                    (WHERE closed_at >= DATE_TRUNC('month', CURRENT_DATE)) as this_month_wr,
                AVG(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) FILTER
                    (WHERE closed_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                     AND closed_at < DATE_TRUNC('month', CURRENT_DATE)) as last_month_wr
            FROM trade_journal
            WHERE realized_pnl IS NOT NULL
        """)
        this_month_wr = float(wr_data["this_month_wr"]) * 100 if wr_data["this_month_wr"] else 0
        last_month_wr = float(wr_data["last_month_wr"]) * 100 if wr_data["last_month_wr"] else 0
        wr_change = this_month_wr - last_month_wr

        # Average hold time
        avg_hold_row = await db.fetchrow("""
            SELECT AVG(EXTRACT(EPOCH FROM (closed_at - opened_at)) / 86400) as avg_days
            FROM trade_journal
            WHERE closed_at IS NOT NULL AND opened_at IS NOT NULL
              AND closed_at >= CURRENT_DATE - INTERVAL '30 days'
        """)
        avg_hold = float(avg_hold_row["avg_days"]) if avg_hold_row["avg_days"] else 0

        # Capital deployed (sum of open position values)
        capital_row = await db.fetchrow("""
            SELECT COALESCE(SUM(ABS(entry_price * quantity)), 0) as capital
            FROM trade_journal WHERE status = 'open'
        """)
        capital_deployed = float(capital_row["capital"]) if capital_row["capital"] else 0

        # YTD ROI
        ytd_row = await db.fetchrow("""
            SELECT SUM(realized_pnl) as ytd_pnl
            FROM trade_journal
            WHERE closed_at >= DATE_TRUNC('year', CURRENT_DATE)
        """)
        ytd_pnl = float(ytd_row["ytd_pnl"]) if ytd_row["ytd_pnl"] else 0

        metrics = [
            {"name": "Total Trades", "value": this_month_trades, "change": trade_change, "period": "vs last month"},
            {"name": "Active Positions", "value": active_positions, "change": 0, "period": "current"},
            {"name": "Win Rate", "value": f"{this_month_wr:.0f}%", "change": round(wr_change, 1), "period": "vs last month"},
            {"name": "Avg Hold Time", "value": f"{avg_hold:.1f} days", "change": 0, "period": "last 30 days"},
            {"name": "Capital Deployed", "value": f"${capital_deployed:,.0f}", "change": 0, "period": "current"},
            {"name": "YTD P&L", "value": f"${ytd_pnl:,.0f}", "change": 0, "period": "year to date"}
        ]

        return {"metrics": metrics}

    except Exception as e:
        logger.error("error_getting_metrics", error=str(e))
        return {
            "metrics": [
                {"name": "Total Trades", "value": 0, "change": 0, "period": "vs last month"},
                {"name": "Active Positions", "value": 0, "change": 0, "period": "current"},
                {"name": "Win Rate", "value": "0%", "change": 0, "period": "vs last month"},
                {"name": "Avg Hold Time", "value": "0 days", "change": 0, "period": "last 30 days"},
                {"name": "Capital Deployed", "value": "$0", "change": 0, "period": "current"},
                {"name": "YTD P&L", "value": "$0", "change": 0, "period": "year to date"}
            ],
            "message": f"No data available. Error: {str(e)}"
        }


@router.get("/breakdown")
async def get_breakdown(by: str = "asset"):
    """Get P&L breakdown from real trade data"""
    try:
        db = await get_database()

        if by == "asset":
            rows = await db.fetch("""
                SELECT
                    symbol,
                    SUM(realized_pnl) as pnl,
                    COUNT(*) as trades
                FROM trade_journal
                WHERE realized_pnl IS NOT NULL
                GROUP BY symbol
                ORDER BY pnl DESC
                LIMIT 10
            """)
        elif by == "type":
            rows = await db.fetch("""
                SELECT
                    COALESCE(asset_type, 'Unknown') as type_name,
                    SUM(realized_pnl) as pnl,
                    COUNT(*) as trades
                FROM trade_journal
                WHERE realized_pnl IS NOT NULL
                GROUP BY asset_type
                ORDER BY pnl DESC
            """)
        else:  # by week
            rows = await db.fetch("""
                SELECT
                    'Week ' || EXTRACT(WEEK FROM closed_at)::text as week_name,
                    SUM(realized_pnl) as pnl,
                    COUNT(*) as trades
                FROM trade_journal
                WHERE realized_pnl IS NOT NULL
                  AND closed_at >= CURRENT_DATE - INTERVAL '4 weeks'
                GROUP BY EXTRACT(WEEK FROM closed_at)
                ORDER BY EXTRACT(WEEK FROM closed_at)
            """)

        # Calculate total for percentages
        total_pnl = sum(abs(float(row["pnl"])) for row in rows if row["pnl"]) or 1

        breakdown = []
        for row in rows:
            pnl = float(row["pnl"]) if row["pnl"] else 0
            # Handle different column names based on query
            name = row.get("symbol") or row.get("type_name") or row.get("week_name") or "Unknown"
            breakdown.append({
                "name": name,
                "pnl": round(pnl, 2),
                "trades": int(row["trades"]) if row["trades"] else 0,
                "percentage": round(abs(pnl) / total_pnl * 100, 1)
            })

        return {"breakdown": breakdown}

    except Exception as e:
        logger.error("error_getting_breakdown", error=str(e), by=by)
        return {
            "breakdown": [],
            "message": f"No breakdown data available. Error: {str(e)}"
        }


@router.get("/heatmap")
async def get_heatmap():
    """Get trading activity heatmap from real trade timestamps"""
    try:
        db = await get_database()

        # Get trade counts by day of week and hour
        rows = await db.fetch("""
            SELECT
                EXTRACT(DOW FROM opened_at) as day_of_week,
                EXTRACT(HOUR FROM opened_at) as hour,
                COUNT(*) as activity
            FROM trade_journal
            WHERE opened_at IS NOT NULL
              AND opened_at >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY EXTRACT(DOW FROM opened_at), EXTRACT(HOUR FROM opened_at)
        """)

        # Build activity map
        activity_map = {}
        for row in rows:
            key = (int(row["day_of_week"]), int(row["hour"]))
            activity_map[key] = int(row["activity"])

        # Generate full heatmap grid
        heatmap = []
        for day in range(7):
            for hour in range(24):
                heatmap.append({
                    "day": day,
                    "hour": hour,
                    "activity": activity_map.get((day, hour), 0)
                })

        return {"heatmap": heatmap}

    except Exception as e:
        logger.error("error_getting_heatmap", error=str(e))
        # Return empty heatmap grid
        heatmap = [{"day": d, "hour": h, "activity": 0} for d in range(7) for h in range(24)]
        return {"heatmap": heatmap, "message": f"No activity data available. Error: {str(e)}"}


@router.get("/calendar")
async def get_calendar(month: Optional[int] = None, year: Optional[int] = None):
    """Get trading calendar with P&L from real data"""
    try:
        today = datetime.now()
        month = month or today.month
        year = year or today.year

        db = await get_database()

        # Get daily P&L for the month
        rows = await db.fetch("""
            SELECT
                DATE(closed_at) as trade_date,
                SUM(realized_pnl) as daily_pnl,
                COUNT(*) as trade_count
            FROM trade_journal
            WHERE EXTRACT(MONTH FROM closed_at) = $1
              AND EXTRACT(YEAR FROM closed_at) = $2
              AND realized_pnl IS NOT NULL
            GROUP BY DATE(closed_at)
            ORDER BY trade_date
        """, month, year)

        # Build date map
        date_map = {}
        for row in rows:
            if row["trade_date"]:
                date_map[row["trade_date"].strftime("%Y-%m-%d")] = {
                    "pnl": round(float(row["daily_pnl"]), 2) if row["daily_pnl"] else 0,
                    "trades": int(row["trade_count"]) if row["trade_count"] else 0
                }

        # Generate calendar days
        days = []
        for day in range(1, 32):
            try:
                date = datetime(year, month, day)
                date_str = date.strftime("%Y-%m-%d")
                day_data = date_map.get(date_str, {"pnl": 0, "trades": 0})
                days.append({
                    "date": date_str,
                    "pnl": day_data["pnl"],
                    "trades": day_data["trades"]
                })
            except ValueError:
                break  # Invalid date (e.g., Feb 30)

        return {"month": month, "year": year, "days": days}

    except Exception as e:
        logger.error("error_getting_calendar", error=str(e), month=month, year=year)
        return {
            "month": month or datetime.now().month,
            "year": year or datetime.now().year,
            "days": [],
            "message": f"No calendar data available. Error: {str(e)}"
        }
