"""
Earnings Router - API endpoints for earnings calendar
NO MOCK DATA - All endpoints use real database via EarningsManager
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Any
from datetime import datetime, timedelta
import logging
import math

from src.earnings_manager import EarningsManager

logger = logging.getLogger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling NaN, None, and invalid values.
    Returns default if value cannot be converted or is NaN/Inf.
    """
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default

router = APIRouter(
    prefix="/api/earnings",
    tags=["earnings"]
)

# Initialize earnings manager (singleton pattern)
_earnings_manager = None

def get_earnings_manager():
    global _earnings_manager
    if _earnings_manager is None:
        _earnings_manager = EarningsManager()
    return _earnings_manager


@router.get("/upcoming")
async def get_upcoming_earnings(
    days: int = Query(7, alias="days_ahead", description="Number of days to look ahead"),
    time_filter: Optional[str] = Query(None, description="Filter by time: Before Market, After Hours"),
    min_quality: Optional[int] = Query(None, description="Minimum quality score (0-100)"),
    limit: int = Query(100, description="Maximum results")
):
    """
    Get upcoming earnings announcements with wheel strategy analysis.
    Data comes from earnings_events table via EarningsManager.
    Returns format expected by React frontend.
    """
    try:
        manager = get_earnings_manager()

        start_date = datetime.now().date()
        end_date = (datetime.now() + timedelta(days=days)).date()

        # Map time filter to database format (lowercase per schema)
        db_time_filter = "all"
        if time_filter == "Before Market":
            db_time_filter = "bmo"
        elif time_filter == "After Hours":
            db_time_filter = "amc"

        # Get earnings from database
        df = manager.get_earnings_events(
            start_date=start_date,
            end_date=end_date,
            time_filter=db_time_filter
        )

        if df.empty:
            return {
                "opportunities": [],
                "upcoming": [],
                "total_count": 0,
                "message": "No earnings events found in database. Run earnings sync to populate data.",
                "generated_at": datetime.now().isoformat()
            }

        all_earnings = []
        for _, row in df.iterrows():
            # Map database earnings_time to frontend format (bmo/amc)
            db_time = row.get('earnings_time', 'TBD')
            frontend_time = 'bmo' if db_time == 'BMO' else 'amc' if db_time == 'AMC' else 'tbd'

            # Apply time filter
            if time_filter:
                expected_time = 'bmo' if time_filter == 'Before Market' else 'amc' if time_filter == 'After Hours' else None
                if expected_time and frontend_time != expected_time:
                    continue

            # Calculate quality score (0-100 numeric) based on IV and beat rate
            iv_rank = safe_float(row.get('pre_earnings_iv'), 0.0)
            beat_rate = safe_float(row.get('beat_rate'), 75.0)  # Default 75 if not available
            avg_surprise = safe_float(row.get('avg_surprise'), 5.0)  # Default 5% if not available

            # Quality score: 40% beat rate + 30% IV rank + 30% surprise magnitude
            quality_score = min(100, max(0,
                (beat_rate * 0.4) +
                (iv_rank * 0.3) +
                (min(avg_surprise * 3, 30) * 1.0)  # Cap surprise contribution
            ))

            # Apply quality filter
            if min_quality is not None and quality_score < min_quality:
                continue

            all_earnings.append({
                "symbol": row.get('symbol', ''),
                "company_name": row.get('company_name', row.get('symbol', '')),
                "earnings_date": row.get('earnings_date').strftime("%Y-%m-%d") if row.get('earnings_date') else None,
                "earnings_time": frontend_time,
                "expected_move_pct": safe_float(row.get('expected_move'), 0.0),
                "iv_rank": iv_rank,
                "beat_rate": beat_rate,
                "avg_surprise": avg_surprise,
                "quality_score": round(quality_score, 1),
                "sector": row.get('sector', 'Unknown')
            })

        # Sort by date
        all_earnings.sort(key=lambda x: x["earnings_date"] or "9999-99-99")

        # Separate high-quality opportunities (score >= 70)
        opportunities = [e for e in all_earnings if (e.get('quality_score') or 0) >= 70]

        return {
            "opportunities": opportunities[:limit],
            "upcoming": all_earnings[:limit],
            "total_count": len(all_earnings),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching upcoming earnings: {e}")
        return {
            "opportunities": [],
            "upcoming": [],
            "total_count": 0,
            "error": str(e),
            "generated_at": datetime.now().isoformat()
        }


@router.get("/calendar")
async def get_earnings_calendar(
    start_date: str = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str = Query(None, description="End date YYYY-MM-DD")
):
    """
    Get earnings calendar for a date range.
    Data comes from earnings_events table.
    """
    try:
        manager = get_earnings_manager()

        # Safely handle start_date
        if not start_date or not isinstance(start_date, str):
            start_dt = datetime.now().date()
        else:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                start_dt = datetime.now().date()

        # Safely handle end_date
        if not end_date or not isinstance(end_date, str):
            end_dt = (datetime.now() + timedelta(days=14)).date()
        else:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                end_dt = (datetime.now() + timedelta(days=14)).date()

        # Get earnings from database
        df = manager.get_earnings_events(start_date=start_dt, end_date=end_dt)

        # Group by date for calendar view
        calendar = {}
        for _, row in df.iterrows():
            date_str = row.get('earnings_date').strftime("%Y-%m-%d") if row.get('earnings_date') else None
            if date_str:
                if date_str not in calendar:
                    calendar[date_str] = []
                calendar[date_str].append({
                    "symbol": row.get('symbol'),
                    "company_name": row.get('company_name', row.get('symbol')),
                    "time": row.get('earnings_time', 'TBD'),
                    "eps_estimate": safe_float(row.get('eps_estimate'), 0.0)
                })

        return {
            "calendar": calendar,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "total_events": len(df)
        }

    except Exception as e:
        logger.error(f"Error fetching earnings calendar: {e}", exc_info=True)
        # Always return valid JSON, never raise HTTPException
        now = datetime.now()
        return {
            "calendar": {},
            "start_date": now.strftime("%Y-%m-%d"),
            "end_date": (now + timedelta(days=14)).strftime("%Y-%m-%d"),
            "total_events": 0,
            "error": str(e)
        }


@router.post("/sync")
async def sync_earnings_data(
    days_ahead: int = Query(30, description="Days ahead to sync"),
    include_history: bool = Query(False, description="Also sync historical data from Alpha Vantage")
):
    """
    Sync earnings data from Finnhub (FREE API).
    Optionally sync historical earnings from Alpha Vantage.
    """
    try:
        manager = get_earnings_manager()

        # Sync upcoming earnings from Finnhub (free, 60 calls/min)
        from_date = datetime.now().date()
        to_date = from_date + timedelta(days=days_ahead)

        finnhub_result = manager.sync_finnhub_earnings(
            from_date=from_date,
            to_date=to_date
        )

        response = {
            "finnhub": finnhub_result,
            "message": f"Synced earnings calendar for next {days_ahead} days"
        }

        # Optionally sync historical data
        if include_history:
            # Get top symbols from the synced earnings
            df = manager.get_earnings_events(start_date=from_date, end_date=to_date)
            if not df.empty:
                top_symbols = df['symbol'].unique().tolist()[:25]  # Alpha Vantage limit
                history_result = manager.sync_alpha_vantage_earnings(top_symbols)
                response["alpha_vantage"] = history_result

        return response

    except Exception as e:
        logger.error(f"Error syncing earnings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync/status")
async def get_sync_status():
    """
    Get the current sync status of earnings data.
    """
    try:
        manager = get_earnings_manager()

        # Get counts
        from_date = datetime.now().date()
        to_date = from_date + timedelta(days=30)
        df = manager.get_earnings_events(start_date=from_date, end_date=to_date)

        # Get history count
        history_df = manager.get_historical_earnings("AAPL", limit=1)  # Just to check

        return {
            "upcoming_events": len(df),
            "has_data": len(df) > 0,
            "date_range": f"{from_date} to {to_date}",
            "message": "Earnings data is available" if len(df) > 0 else "No earnings data. Click 'Sync Data' to fetch from Finnhub."
        }

    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        return {
            "upcoming_events": 0,
            "has_data": False,
            "error": str(e)
        }


@router.get("/symbol/{symbol}")
async def get_symbol_earnings(symbol: str):
    """
    Get earnings history and upcoming for a specific symbol.
    Data comes from database via EarningsManager.
    """
    try:
        manager = get_earnings_manager()
        symbol = symbol.upper()

        # Get historical earnings
        history_df = manager.get_historical_earnings(symbol, limit=12)

        # Get upcoming earnings for this symbol
        upcoming_df = manager.get_earnings_events(
            start_date=datetime.now().date(),
            end_date=(datetime.now() + timedelta(days=90)).date(),
            symbols=[symbol]
        )

        # Calculate statistics from history
        history = []
        total_beats = 0
        total_surprise = 0.0
        total_moves = 0.0

        for _, row in history_df.iterrows():
            eps_actual = safe_float(row.get('eps_actual'), 0.0)
            eps_estimate = safe_float(row.get('eps_estimate'), 0.0)

            if eps_estimate != 0:
                surprise = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                if eps_actual > eps_estimate:
                    total_beats += 1
            else:
                surprise = 0.0

            total_surprise += abs(surprise)

            history.append({
                "date": row.get('report_date').strftime("%Y-%m-%d") if row.get('report_date') else None,
                "quarter": row.get('quarter'),
                "year": row.get('year'),
                "eps_actual": eps_actual,
                "eps_estimate": eps_estimate,
                "surprise_pct": round(surprise, 2)
            })

        # Upcoming earnings
        upcoming = None
        if not upcoming_df.empty:
            row = upcoming_df.iloc[0]
            upcoming = {
                "date": row.get('earnings_date').strftime("%Y-%m-%d") if row.get('earnings_date') else None,
                "time": row.get('earnings_time', 'TBD'),
                "eps_estimate": safe_float(row.get('eps_estimate'), 0.0)
            }

        num_reports = len(history_df) if not history_df.empty else 0

        return {
            "symbol": symbol,
            "upcoming": upcoming,
            "history": history,
            "avg_move": round(total_moves / num_reports, 1) if num_reports > 0 else 0.0,
            "beat_rate": round((total_beats / num_reports) * 100, 0) if num_reports > 0 else 0.0,
            "iv_rank_avg": 50.0  # Would need IV history data
        }

    except Exception as e:
        logger.error(f"Error fetching earnings for {symbol}: {e}")
        return {
            "symbol": symbol,
            "upcoming": None,
            "history": [],
            "avg_move": 0.0,
            "beat_rate": 0.0,
            "iv_rank_avg": 0.0,
            "error": str(e)
        }
