"""
Analytics Router - Real trading performance analytics
NO MOCK DATA - All endpoints use real database queries
"""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime, timedelta
import logging
from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/performance")
async def get_performance(period: str = "1M"):
    """Get trading performance analytics from real trade history"""
    try:
        days = 30 if period == "1M" else 90 if period == "3M" else 365 if period == "1Y" else 30
        start_date = datetime.now() - timedelta(days=days)

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get daily P&L from trade_journal
            cursor.execute("""
                SELECT DATE(closed_at) as trade_date,
                       SUM(realized_pnl) as daily_pnl
                FROM trade_journal
                WHERE closed_at >= %s AND realized_pnl IS NOT NULL
                GROUP BY DATE(closed_at)
                ORDER BY trade_date ASC
            """, (start_date,))

            rows = cursor.fetchall()

            daily_pnl = []
            cumulative = 0
            for row in rows:
                daily = float(row[1]) if row[1] else 0
                cumulative += daily
                daily_pnl.append({
                    "date": row[0].strftime("%Y-%m-%d") if row[0] else "",
                    "pnl": round(daily, 2),
                    "cumulative": round(cumulative, 2)
                })

            # Calculate win/loss stats
            cursor.execute("""
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
                WHERE closed_at >= %s AND realized_pnl IS NOT NULL
            """, (start_date,))

            stats = cursor.fetchone()

            total_trades = int(stats[0]) if stats[0] else 0
            wins = int(stats[1]) if stats[1] else 0
            losses = int(stats[2]) if stats[2] else 0
            avg_win = float(stats[3]) if stats[3] else 0
            avg_loss = float(stats[4]) if stats[4] else 0
            total_pnl = float(stats[7]) if stats[7] else cumulative

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
                "sharpe_ratio": 0,  # Would need daily returns std dev to calculate
                "daily_pnl": daily_pnl,
                "total_trades": total_trades
            }

    except Exception as e:
        logger.error(f"Error getting performance: {e}")
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
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
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

            rows = cursor.fetchall()

            strategies = []
            for row in rows:
                trades = int(row[1]) if row[1] else 0
                wins = int(row[2]) if row[2] else 0
                win_rate = wins / trades if trades > 0 else 0

                strategies.append({
                    "name": row[0] or "Unknown",
                    "trades": trades,
                    "win_rate": round(win_rate, 2),
                    "total_pnl": round(float(row[3]), 2) if row[3] else 0,
                    "avg_return": round(float(row[4]), 1) if row[4] else 0,
                    "best_trade": round(float(row[5]), 2) if row[5] else 0,
                    "worst_trade": round(float(row[6]), 2) if row[6] else 0
                })

            return {"strategies": strategies}

    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        return {
            "strategies": [],
            "message": f"No strategy data available. Error: {str(e)}"
        }


@router.get("/metrics")
async def get_metrics():
    """Get key trading metrics from real data"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Total trades this month vs last month
            cursor.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE closed_at >= DATE_TRUNC('month', CURRENT_DATE)) as this_month,
                    COUNT(*) FILTER (WHERE closed_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                                     AND closed_at < DATE_TRUNC('month', CURRENT_DATE)) as last_month
                FROM trade_journal
            """)
            trade_counts = cursor.fetchone()
            this_month_trades = int(trade_counts[0]) if trade_counts[0] else 0
            last_month_trades = int(trade_counts[1]) if trade_counts[1] else 0
            trade_change = this_month_trades - last_month_trades

            # Active positions
            cursor.execute("""
                SELECT COUNT(*) FROM trade_journal WHERE status = 'open'
            """)
            active_positions = cursor.fetchone()[0] or 0

            # Win rate this month vs last month
            cursor.execute("""
                SELECT
                    AVG(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) FILTER
                        (WHERE closed_at >= DATE_TRUNC('month', CURRENT_DATE)) as this_month_wr,
                    AVG(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) FILTER
                        (WHERE closed_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                         AND closed_at < DATE_TRUNC('month', CURRENT_DATE)) as last_month_wr
                FROM trade_journal
                WHERE realized_pnl IS NOT NULL
            """)
            wr_data = cursor.fetchone()
            this_month_wr = float(wr_data[0]) * 100 if wr_data[0] else 0
            last_month_wr = float(wr_data[1]) * 100 if wr_data[1] else 0
            wr_change = this_month_wr - last_month_wr

            # Average hold time
            cursor.execute("""
                SELECT AVG(EXTRACT(EPOCH FROM (closed_at - opened_at)) / 86400) as avg_days
                FROM trade_journal
                WHERE closed_at IS NOT NULL AND opened_at IS NOT NULL
                  AND closed_at >= CURRENT_DATE - INTERVAL '30 days'
            """)
            avg_hold = cursor.fetchone()[0] or 0

            # Capital deployed (sum of open position values)
            cursor.execute("""
                SELECT COALESCE(SUM(ABS(entry_price * quantity)), 0)
                FROM trade_journal WHERE status = 'open'
            """)
            capital_deployed = float(cursor.fetchone()[0] or 0)

            # YTD ROI
            cursor.execute("""
                SELECT SUM(realized_pnl)
                FROM trade_journal
                WHERE closed_at >= DATE_TRUNC('year', CURRENT_DATE)
            """)
            ytd_pnl = float(cursor.fetchone()[0] or 0)

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
        logger.error(f"Error getting metrics: {e}")
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
        with get_db_connection() as conn:
            cursor = conn.cursor()

            if by == "asset":
                cursor.execute("""
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
                cursor.execute("""
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
                cursor.execute("""
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

            rows = cursor.fetchall()

            # Calculate total for percentages
            total_pnl = sum(abs(float(row[1])) for row in rows if row[1]) or 1

            breakdown = []
            for row in rows:
                pnl = float(row[1]) if row[1] else 0
                breakdown.append({
                    "name": row[0] or "Unknown",
                    "pnl": round(pnl, 2),
                    "trades": int(row[2]) if row[2] else 0,
                    "percentage": round(abs(pnl) / total_pnl * 100, 1)
                })

            return {"breakdown": breakdown}

    except Exception as e:
        logger.error(f"Error getting breakdown: {e}")
        return {
            "breakdown": [],
            "message": f"No breakdown data available. Error: {str(e)}"
        }


@router.get("/heatmap")
async def get_heatmap():
    """Get trading activity heatmap from real trade timestamps"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get trade counts by day of week and hour
            cursor.execute("""
                SELECT
                    EXTRACT(DOW FROM opened_at) as day_of_week,
                    EXTRACT(HOUR FROM opened_at) as hour,
                    COUNT(*) as activity
                FROM trade_journal
                WHERE opened_at IS NOT NULL
                  AND opened_at >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY EXTRACT(DOW FROM opened_at), EXTRACT(HOUR FROM opened_at)
            """)

            rows = cursor.fetchall()

            # Build activity map
            activity_map = {}
            for row in rows:
                key = (int(row[0]), int(row[1]))
                activity_map[key] = int(row[2])

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
        logger.error(f"Error getting heatmap: {e}")
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

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get daily P&L for the month
            cursor.execute("""
                SELECT
                    DATE(closed_at) as trade_date,
                    SUM(realized_pnl) as daily_pnl,
                    COUNT(*) as trade_count
                FROM trade_journal
                WHERE EXTRACT(MONTH FROM closed_at) = %s
                  AND EXTRACT(YEAR FROM closed_at) = %s
                  AND realized_pnl IS NOT NULL
                GROUP BY DATE(closed_at)
                ORDER BY trade_date
            """, (month, year))

            rows = cursor.fetchall()

            # Build date map
            date_map = {}
            for row in rows:
                if row[0]:
                    date_map[row[0].strftime("%Y-%m-%d")] = {
                        "pnl": round(float(row[1]), 2) if row[1] else 0,
                        "trades": int(row[2]) if row[2] else 0
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
        logger.error(f"Error getting calendar: {e}")
        return {
            "month": month or datetime.now().month,
            "year": year or datetime.now().year,
            "days": [],
            "message": f"No calendar data available. Error: {str(e)}"
        }
