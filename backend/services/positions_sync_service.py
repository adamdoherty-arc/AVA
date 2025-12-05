"""
Positions Sync Service - Background Robinhood Position Caching
==============================================================

Syncs Robinhood positions to the database every 30 minutes so the UI
never blocks waiting for API calls.

Features:
- Background sync every 30 minutes (configurable)
- Immediate cache read for UI (never blocks)
- Graceful fallback to live API if cache is stale
- Sync history tracking for monitoring

Author: AVA Trading Platform
Created: 2025-12-05
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from backend.infrastructure.database import get_database
from backend.services.portfolio_service import get_portfolio_service

logger = structlog.get_logger(__name__)

# Sync interval in seconds (30 minutes)
SYNC_INTERVAL_SECONDS = 30 * 60
# Max cache age before forcing live fetch (35 minutes - slightly longer than sync interval)
MAX_CACHE_AGE_SECONDS = 35 * 60


class PositionsSyncService:
    """
    Background service that syncs Robinhood positions to database.

    The UI should always read from the database cache for instant loading.
    This service keeps the cache fresh in the background.
    """

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0
        self._error_count = 0

    async def start(self) -> None:
        """Start the background sync loop."""
        if self._running:
            logger.warning("positions_sync_already_running")
            return

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info("positions_sync_service_started", interval_minutes=SYNC_INTERVAL_SECONDS // 60)

    async def stop(self) -> None:
        """Stop the background sync loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("positions_sync_service_stopped")

    async def _sync_loop(self) -> None:
        """Main sync loop - runs every 30 minutes."""
        # Initial sync on startup
        await self._sync_positions("startup")

        while self._running:
            try:
                await asyncio.sleep(SYNC_INTERVAL_SECONDS)
                if self._running:
                    await self._sync_positions("scheduled")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("sync_loop_error", error=str(e))
                self._error_count += 1
                # Wait a bit before retrying
                await asyncio.sleep(60)

    async def sync_now(self) -> Dict[str, Any]:
        """Manually trigger a sync (for API endpoint)."""
        return await self._sync_positions("manual")

    async def _sync_positions(self, sync_type: str) -> Dict[str, Any]:
        """
        Fetch positions from Robinhood and save to database.

        Args:
            sync_type: Type of sync (startup, scheduled, manual)

        Returns:
            Sync result with status and counts
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc)

        logger.info("positions_sync_starting", sync_type=sync_type)

        # Start sync log
        db = await get_database()
        log_id = await db.fetchval("""
            INSERT INTO positions_sync_log (sync_type, started_at, status)
            VALUES ($1, $2, 'running')
            RETURNING id
        """, sync_type, started_at)

        try:
            # Fetch from Robinhood via existing service
            portfolio_service = get_portfolio_service()
            positions = await portfolio_service.get_positions(force_refresh=True)

            # Extract data
            summary = positions.get("summary", {})
            stocks = positions.get("stocks", [])
            options = positions.get("options", [])

            # Save to database in a transaction
            async with db.transaction() as conn:
                # Clear old data
                await conn.execute("DELETE FROM cached_stock_positions")
                await conn.execute("DELETE FROM cached_option_positions")

                # Insert stock positions
                for stock in stocks:
                    await conn.execute("""
                        INSERT INTO cached_stock_positions
                        (symbol, quantity, avg_buy_price, current_price, cost_basis,
                         current_value, pl, pl_pct, synced_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                        stock.get("symbol"),
                        stock.get("quantity", 0),
                        stock.get("avg_buy_price", 0),
                        stock.get("current_price", 0),
                        stock.get("cost_basis", 0),
                        stock.get("current_value", 0),
                        stock.get("pl", 0),
                        stock.get("pl_pct", 0),
                        started_at
                    )

                # Insert option positions
                for opt in options:
                    greeks = opt.get("greeks", {})
                    # Convert expiration string to date if needed
                    expiration = opt.get("expiration")
                    if isinstance(expiration, str):
                        try:
                            expiration = datetime.strptime(expiration, "%Y-%m-%d").date()
                        except (ValueError, TypeError):
                            expiration = None

                    await conn.execute("""
                        INSERT INTO cached_option_positions
                        (symbol, strategy, position_type, option_type, strike, expiration,
                         dte, quantity, avg_price, current_price, total_premium,
                         current_value, pl, pl_pct, breakeven,
                         delta, theta, gamma, vega, iv, synced_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                    """,
                        opt.get("symbol"),
                        opt.get("strategy"),
                        opt.get("type"),
                        opt.get("option_type"),
                        opt.get("strike", 0),
                        expiration,
                        opt.get("dte", 0),
                        opt.get("quantity", 0),
                        opt.get("avg_price", 0),
                        opt.get("current_price", 0),
                        opt.get("total_premium", 0),
                        opt.get("current_value", 0),
                        opt.get("pl", 0),
                        opt.get("pl_pct", 0),
                        opt.get("breakeven", 0),
                        greeks.get("delta", 0),
                        greeks.get("theta", 0),
                        greeks.get("gamma", 0),
                        greeks.get("vega", 0),
                        greeks.get("iv", 0),
                        started_at
                    )

                # Update portfolio summary
                await conn.execute("""
                    INSERT INTO cached_portfolio_summary
                    (id, total_equity, core_equity, buying_power, portfolio_cash,
                     uncleared_deposits, unsettled_funds, options_collateral,
                     total_stock_positions, total_option_positions, synced_at)
                    VALUES (1, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (id) DO UPDATE SET
                        total_equity = EXCLUDED.total_equity,
                        core_equity = EXCLUDED.core_equity,
                        buying_power = EXCLUDED.buying_power,
                        portfolio_cash = EXCLUDED.portfolio_cash,
                        uncleared_deposits = EXCLUDED.uncleared_deposits,
                        unsettled_funds = EXCLUDED.unsettled_funds,
                        options_collateral = EXCLUDED.options_collateral,
                        total_stock_positions = EXCLUDED.total_stock_positions,
                        total_option_positions = EXCLUDED.total_option_positions,
                        synced_at = EXCLUDED.synced_at
                """,
                    summary.get("total_equity", 0),
                    summary.get("core_equity", 0),
                    summary.get("buying_power", 0),
                    summary.get("portfolio_cash", 0),
                    summary.get("uncleared_deposits", 0),
                    summary.get("unsettled_funds", 0),
                    summary.get("options_collateral", 0),
                    len(stocks),
                    len(options),
                    started_at
                )

            # Update sync log
            duration_ms = int((time.time() - start_time) * 1000)
            await db.execute("""
                UPDATE positions_sync_log
                SET completed_at = $1, status = 'success',
                    stocks_synced = $2, options_synced = $3, duration_ms = $4
                WHERE id = $5
            """, datetime.now(timezone.utc), len(stocks), len(options), duration_ms, log_id)

            self._last_sync = started_at
            self._sync_count += 1

            logger.info(
                "positions_sync_complete",
                sync_type=sync_type,
                stocks=len(stocks),
                options=len(options),
                duration_ms=duration_ms
            )

            return {
                "status": "success",
                "sync_type": sync_type,
                "stocks_synced": len(stocks),
                "options_synced": len(options),
                "duration_ms": duration_ms,
                "synced_at": started_at.isoformat()
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            # Update sync log with error
            await db.execute("""
                UPDATE positions_sync_log
                SET completed_at = $1, status = 'failed', error_message = $2, duration_ms = $3
                WHERE id = $4
            """, datetime.now(timezone.utc), error_msg, duration_ms, log_id)

            self._error_count += 1
            logger.error("positions_sync_failed", error=error_msg, sync_type=sync_type)

            return {
                "status": "failed",
                "sync_type": sync_type,
                "error": error_msg,
                "duration_ms": duration_ms
            }

    async def get_cached_positions(self) -> Dict[str, Any]:
        """
        Get positions from the database cache.

        This is the primary method the UI should use - it's instant and never blocks.
        Falls back to live API if cache is too stale.
        """
        db = await get_database()

        # Check cache age
        cache_age = await db.fetchrow("""
            SELECT
                EXTRACT(EPOCH FROM (NOW() - synced_at))::INTEGER AS age_seconds
            FROM cached_portfolio_summary
            WHERE id = 1
        """)

        cache_age_seconds = cache_age["age_seconds"] if cache_age else None

        # If cache is too old or doesn't exist, trigger sync but still return cached data
        if cache_age_seconds is None or cache_age_seconds > MAX_CACHE_AGE_SECONDS:
            logger.warning(
                "cache_stale_triggering_sync",
                age_seconds=cache_age_seconds,
                max_age=MAX_CACHE_AGE_SECONDS
            )
            # Don't await - trigger in background
            asyncio.create_task(self._sync_positions("stale_cache"))

        # Get cached summary
        summary_row = await db.fetchrow("""
            SELECT * FROM cached_portfolio_summary WHERE id = 1
        """)

        if not summary_row:
            # No cache at all - must fetch live
            logger.warning("no_cache_fetching_live")
            portfolio_service = get_portfolio_service()
            return await portfolio_service.get_positions(force_refresh=True)

        # Get cached stocks
        stock_rows = await db.fetch("""
            SELECT symbol, quantity, avg_buy_price, current_price,
                   cost_basis, current_value, pl, pl_pct, synced_at
            FROM cached_stock_positions
            ORDER BY current_value DESC
        """)

        # Get cached options
        option_rows = await db.fetch("""
            SELECT symbol, strategy, position_type AS type, option_type,
                   strike, expiration, dte, quantity, avg_price,
                   current_price, total_premium, current_value, pl, pl_pct,
                   breakeven, delta, theta, gamma, vega, iv, synced_at
            FROM cached_option_positions
            ORDER BY expiration, symbol
        """)

        # Build response
        stocks = [
            {
                "symbol": row["symbol"],
                "quantity": float(row["quantity"]),
                "avg_buy_price": float(row["avg_buy_price"]),
                "current_price": float(row["current_price"]) if row["current_price"] else 0,
                "cost_basis": float(row["cost_basis"]) if row["cost_basis"] else 0,
                "current_value": float(row["current_value"]) if row["current_value"] else 0,
                "pl": float(row["pl"]) if row["pl"] else 0,
                "pl_pct": float(row["pl_pct"]) if row["pl_pct"] else 0,
                "type": "stock"
            }
            for row in stock_rows
        ]

        options = [
            {
                "symbol": row["symbol"],
                "strategy": row["strategy"],
                "type": row["type"],
                "option_type": row["option_type"],
                "strike": float(row["strike"]),
                "expiration": row["expiration"].isoformat() if row["expiration"] else None,
                "dte": row["dte"],
                "quantity": float(row["quantity"]),
                "avg_price": float(row["avg_price"]) if row["avg_price"] else 0,
                "current_price": float(row["current_price"]) if row["current_price"] else 0,
                "total_premium": float(row["total_premium"]) if row["total_premium"] else 0,
                "current_value": float(row["current_value"]) if row["current_value"] else 0,
                "pl": float(row["pl"]) if row["pl"] else 0,
                "pl_pct": float(row["pl_pct"]) if row["pl_pct"] else 0,
                "breakeven": float(row["breakeven"]) if row["breakeven"] else 0,
                "greeks": {
                    "delta": float(row["delta"]) if row["delta"] else 0,
                    "theta": float(row["theta"]) if row["theta"] else 0,
                    "gamma": float(row["gamma"]) if row["gamma"] else 0,
                    "vega": float(row["vega"]) if row["vega"] else 0,
                    "iv": float(row["iv"]) if row["iv"] else 0,
                }
            }
            for row in option_rows
        ]

        return {
            "summary": {
                "total_equity": float(summary_row["total_equity"]) if summary_row["total_equity"] else 0,
                "core_equity": float(summary_row["core_equity"]) if summary_row["core_equity"] else 0,
                "buying_power": float(summary_row["buying_power"]) if summary_row["buying_power"] else 0,
                "portfolio_cash": float(summary_row["portfolio_cash"]) if summary_row["portfolio_cash"] else 0,
                "uncleared_deposits": float(summary_row["uncleared_deposits"]) if summary_row["uncleared_deposits"] else 0,
                "unsettled_funds": float(summary_row["unsettled_funds"]) if summary_row["unsettled_funds"] else 0,
                "options_collateral": float(summary_row["options_collateral"]) if summary_row["options_collateral"] else 0,
                "total_positions": summary_row["total_stock_positions"] + summary_row["total_option_positions"]
            },
            "stocks": stocks,
            "options": options,
            "_cache_info": {
                "cached": True,
                "synced_at": summary_row["synced_at"].isoformat() if summary_row["synced_at"] else None,
                "age_seconds": cache_age_seconds
            }
        }

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get sync service status for monitoring."""
        db = await get_database()

        # Get last few syncs
        recent_syncs = await db.fetch("""
            SELECT sync_type, started_at, completed_at, status,
                   stocks_synced, options_synced, duration_ms, error_message
            FROM positions_sync_log
            ORDER BY started_at DESC
            LIMIT 10
        """)

        # Get cache age
        cache_age = await db.fetchrow("""
            SELECT EXTRACT(EPOCH FROM (NOW() - synced_at))::INTEGER AS age_seconds
            FROM cached_portfolio_summary WHERE id = 1
        """)

        return {
            "running": self._running,
            "sync_count": self._sync_count,
            "error_count": self._error_count,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "cache_age_seconds": cache_age["age_seconds"] if cache_age else None,
            "next_sync_in_seconds": SYNC_INTERVAL_SECONDS - (cache_age["age_seconds"] or 0) if cache_age else 0,
            "recent_syncs": [
                {
                    "sync_type": row["sync_type"],
                    "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                    "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                    "status": row["status"],
                    "stocks_synced": row["stocks_synced"],
                    "options_synced": row["options_synced"],
                    "duration_ms": row["duration_ms"],
                    "error": row["error_message"]
                }
                for row in recent_syncs
            ]
        }


# Singleton instance
_sync_service: Optional[PositionsSyncService] = None


def get_positions_sync_service() -> PositionsSyncService:
    """Get the singleton sync service instance."""
    global _sync_service
    if _sync_service is None:
        _sync_service = PositionsSyncService()
    return _sync_service


async def start_positions_sync_service() -> PositionsSyncService:
    """Start the positions sync service."""
    service = get_positions_sync_service()
    await service.start()
    return service


async def stop_positions_sync_service() -> None:
    """Stop the positions sync service."""
    global _sync_service
    if _sync_service:
        await _sync_service.stop()
